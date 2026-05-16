"""Tests for the SWE-agent trajectory adapter.

Synthetic rows mirror the EXACT real format inspected on
2026-05-15 (reasoning text + a final fenced block holding one ACI
command; roles system/user/ai; in-row `target` bool).
"""

from __future__ import annotations

from agentdiag.observable import EventType
from agentdiag.adapters.swe_agent import (
    _extract_action, row_to_events, outcome, baseline_features,
)


def _ai(text):
    return {"role": "ai", "text": text}


def _user(text):
    return {"role": "user", "text": text}


_ROW = {
    "instance_id": "Azure__x-890",
    "model_name": "swe-agent-llama-70b",
    "target": False,
    "exit_status": "submitted",
    "generated_patch": "diff --git a/x b/x\n+1\n",
    "trajectory": [
        {"role": "system", "text": ""},
        _user("ISSUE: something is broken"),
        _ai('We should search.\n```\nsearch_dir "PYTHON_THREADPOOL"\n```'),
        _user("Found 47 matches ...\n(Current directory: /repo)"),
        _ai("Open the file.\n```\nopen azure_worker/dispatcher.py\n```"),
        _user("[file contents]"),
        _ai("Fix it.\n```\nedit 12:14\nnew code\nend_of_edit\n```"),
        _user("File updated."),
        _ai("Run tests.\n```\npython -m pytest -q\n```"),
        _user("1 failed"),
        _ai("Submit.\n```\nsubmit\n```"),
        _ai("All done — no fenced block here, pure prose."),
    ],
}


def test_extract_action_takes_last_fenced_block():
    assert _extract_action("hi\n```\nopen foo.py\n```") == ("open", "foo.py")
    assert _extract_action('```\nsearch_dir "q"\n```') == ("search_dir", '"q"')
    assert _extract_action("no block here") is None
    assert _extract_action("```\n\n```") is None


def test_row_to_events_one_per_ai_action():
    ev = row_to_events(_ROW)
    # 5 ai turns have a fenced action; the 6th ai turn is pure prose
    assert len(ev) == 5
    assert [e.tool_name for e in ev] == [
        "search_dir", "open", "edit", "python", "submit"]


def test_event_type_classification():
    ev = {e.tool_name: e for e in row_to_events(_ROW)}
    assert ev["search_dir"].event_type == EventType.FILE_READ
    assert ev["open"].event_type == EventType.FILE_READ
    assert ev["edit"].event_type == EventType.FILE_WRITE
    assert ev["python"].event_type == EventType.SHELL_COMMAND
    assert ev["submit"].event_type == EventType.TOOL_CALL


def test_target_path_extracted():
    ev = {e.tool_name: e for e in row_to_events(_ROW)}
    assert ev["open"].target_path == "azure_worker/dispatcher.py"
    assert ev["search_dir"].target_path == "PYTHON_THREADPOOL"


def test_outcome_is_the_independent_label():
    assert outcome(_ROW) is False
    assert outcome({"target": True}) is True


def test_baseline_features_shape():
    bf = baseline_features(_ROW)
    assert bf["n_turns"] == len(_ROW["trajectory"])
    assert bf["n_parsed_actions"] == 5
    assert bf["patch_len"] == len(_ROW["generated_patch"])
    assert bf["model_name"] == "swe-agent-llama-70b"
    assert bf["exit_status"] == "submitted"


def test_pure_prose_ai_turn_is_skipped_not_hallucinated():
    row = {"trajectory": [{"role": "ai", "text": "I think we are done."}]}
    assert row_to_events(row) == []


def test_events_feed_the_real_pipeline():
    """End-to-end: parsed events must flow through UniversalMonitor."""
    from agentdiag.universal_monitor import UniversalMonitor
    m = UniversalMonitor(sensitivity=2.0)
    n = 0
    for e in row_to_events(_ROW):
        r = m.process(e)
        assert r.get("type") == "observation"
        n += 1
    assert n == 5
