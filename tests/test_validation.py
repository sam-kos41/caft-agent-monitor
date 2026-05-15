"""Tests for the agentdiag.validation package.

Covers digest extraction, ledger round-trip, CAFT-rater rule mapping,
and inter-rater agreement statistics. Ollama and FastAPI/uvicorn paths
are NOT tested here (they need real services); they get smoke-tested
out-of-band by the CLI entry point.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from agentdiag.validation.digest import (
    SessionDigest, build_digest, DIMENSIONS, LIKERT_DIMS, CATEGORICAL_DIMS,
)
from agentdiag.validation.ledger import Ledger, Rating
from agentdiag.validation.rate_caft import (
    CaftMetrics, rate_with_caft, _likert_from_threshold,
)
from agentdiag.validation.agreement import (
    cohens_kappa, krippendorff_alpha, spearman_rho, compute_agreement,
    interpret_kappa,
)
from agentdiag.validation.report import write_report


def _write_session(path: Path, events: list[dict]) -> None:
    with path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _user_msg(text: str) -> dict:
    return {"type": "user",
            "message": {"role": "user", "content": text},
            "timestamp": "2026-05-14T10:00:00Z"}


def _bash(cmd: str) -> dict:
    return {"type": "assistant",
            "message": {"role": "assistant", "content": [
                {"type": "tool_use", "name": "Bash",
                 "input": {"command": cmd}}]},
            "timestamp": "2026-05-14T10:00:01Z"}


def _tool_result(content: str, is_error: bool = False) -> dict:
    return {"type": "user",
            "message": {"role": "user", "content": [
                {"type": "tool_result", "content": content,
                 "is_error": is_error}]},
            "timestamp": "2026-05-14T10:00:02Z"}


# -------- digest --------


def test_digest_extracts_basic_counts(tmp_path):
    p = tmp_path / "sess.jsonl"
    _write_session(p, [
        _user_msg("hello, build me a website"),
        _bash("ls"),
        _tool_result("file1\nfile2"),
        _bash("git status"),
        _tool_result("clean", is_error=False),
        _user_msg("now deploy it"),
    ])
    d = build_digest(p)
    assert d.session_id == "sess"
    assert d.n_user_messages == 2
    assert d.n_tool_calls == 2
    assert d.n_tool_results == 2
    assert d.first_user_message.startswith("hello")
    assert d.last_user_message.startswith("now deploy")


def test_digest_detects_repeat_runs(tmp_path):
    p = tmp_path / "sess.jsonl"
    events = [_user_msg("do the thing")]
    for _ in range(7):
        events.append(_bash("git add page.tsx"))
    events.append(_bash("git status"))
    _write_session(p, events)
    d = build_digest(p)
    assert d.longest_repeat_runs, "should detect the 7x git add run"
    assert d.longest_repeat_runs[0].length == 7
    assert "git add page.tsx" in d.longest_repeat_runs[0].pattern


def test_digest_counts_errors(tmp_path):
    p = tmp_path / "sess.jsonl"
    _write_session(p, [
        _user_msg("run the script"),
        _bash("python script.py"),
        _tool_result("Traceback (most recent call last): ZeroDivisionError",
                     is_error=True),
        _bash("python script.py"),
        _tool_result("ok", is_error=False),
    ])
    d = build_digest(p)
    assert d.n_errors == 1
    assert d.n_tool_results == 2
    assert math.isclose(d.error_rate, 0.5)


def test_digest_includes_narrative(tmp_path):
    p = tmp_path / "sess.jsonl"
    _write_session(p, [
        _user_msg("build me a parser"),
        _bash("python -m pytest"),
        _tool_result("2 passed"),
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Edit",
             "input": {"file_path": "parser.py"}}]},
         "timestamp": "2026-05-14T10:00:03Z"},
    ])
    d = build_digest(p)
    assert d.narrative
    assert "USER:" in d.narrative
    assert "What the agent did" in d.to_text()


def test_narrative_is_descriptive_not_evaluative(tmp_path):
    """The narrative must never editorialize — that would anchor raters."""
    p = tmp_path / "sess.jsonl"
    events = [_user_msg("fix the bug")]
    for _ in range(8):
        events.append(_bash("python test.py"))
        events.append(_tool_result("FAILED", is_error=True))
    _write_session(p, events)
    d = build_digest(p)
    banned = ("stuck", "struggled", "confused", "lost", "floundered",
              "spinning", "wasted", "failed to make progress", "pathological")
    low = d.narrative.lower()
    for word in banned:
        assert word not in low, f"narrative editorialized with {word!r}"


def test_digest_to_text_is_bounded(tmp_path):
    p = tmp_path / "sess.jsonl"
    events = [_user_msg("x" * 5000)] + [_bash("ls")] * 50
    _write_session(p, events)
    d = build_digest(p)
    text = d.to_text(max_chars=2000)
    assert len(text) <= 2050  # +50 for truncation marker
    assert "Session:" in text


# -------- ledger --------


def test_ledger_round_trip(tmp_path):
    led = Ledger(tmp_path / "ratings.jsonl")
    r = Rating(session_id="s1", rater_type="human", rater_id="sam",
               dimension="stuck_in_loop", value=4, reasoning="lots of repeats")
    led.append(r)
    rows = led.all_rows()
    assert len(rows) == 1
    assert rows[0]["session_id"] == "s1"
    assert rows[0]["value"] == 4


def test_ledger_latest_dedupes(tmp_path):
    led = Ledger(tmp_path / "ratings.jsonl")
    led.append(Rating("s1", "human", "sam", "stuck_in_loop", 3))
    led.append(Rating("s1", "human", "sam", "stuck_in_loop", 5,
                      reasoning="changed mind"))
    latest = led.latest()
    assert latest[("s1", "human", "sam", "stuck_in_loop")]["value"] == 5


def test_ledger_rejects_bad_rater_type():
    with pytest.raises(ValueError, match="rater_type"):
        Rating("s1", "robot", "x", "stuck_in_loop", 3)


def test_rating_abstention():
    r = Rating("s1", "human", "sam", "goal_drifted", None)
    assert r.is_abstention
    r2 = Rating("s1", "human", "sam", "stuck_in_loop", 2, confidence="high")
    assert not r2.is_abstention


def test_rating_rejects_bad_confidence():
    with pytest.raises(ValueError, match="confidence"):
        Rating("s1", "human", "sam", "stuck_in_loop", 3, confidence="kinda")


def test_abstentions_excluded_from_kappa(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    # human abstains on s2; caft rates it. s1 and s3 agree perfectly.
    for sess in ("s1", "s3"):
        led.append(Rating(sess, "human", "sam", "stuck_in_loop", 4,
                          confidence="high"))
        led.append(Rating(sess, "caft", "v0", "stuck_in_loop", 4,
                          confidence="high"))
    led.append(Rating("s2", "human", "sam", "stuck_in_loop", None))
    led.append(Rating("s2", "caft", "v0", "stuck_in_loop", 1,
                      confidence="high"))
    out = compute_agreement(led)
    pair = ("caft:v0", "human:sam")
    # s2 excluded -> only s1,s3 which agree -> kappa 1.0
    assert out["per_dimension"]["stuck_in_loop"]["pair_kappa"][pair] == pytest.approx(1.0)
    assert out["abstentions"]["human:sam"]["stuck_in_loop"] == 1


def test_min_confidence_filter(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    led.append(Rating("s1", "human", "sam", "stuck_in_loop", 4, confidence="high"))
    led.append(Rating("s1", "caft", "v0", "stuck_in_loop", 4, confidence="high"))
    led.append(Rating("s2", "human", "sam", "stuck_in_loop", 1, confidence="low"))
    led.append(Rating("s2", "caft", "v0", "stuck_in_loop", 5, confidence="high"))
    full = compute_agreement(led)
    hi = compute_agreement(led, min_confidence="high")
    pair = ("caft:v0", "human:sam")
    # full includes the low-confidence disagreement on s2; high-only drops it
    assert hi["per_dimension"]["stuck_in_loop"]["pair_kappa"][pair] == pytest.approx(1.0)
    assert full["per_dimension"]["stuck_in_loop"]["pair_kappa"][pair] != pytest.approx(1.0)


# -------- caft rater --------


def test_likert_threshold_normal_and_inverted():
    assert _likert_from_threshold(0.5, 1.0, 2.0) == 1
    assert _likert_from_threshold(2.5, 1.0, 2.0) == 5
    assert _likert_from_threshold(1.5, 1.0, 2.0) == 3
    assert _likert_from_threshold(0.5, 1.0, 2.0, inverted=True) == 5
    assert _likert_from_threshold(2.5, 1.0, 2.0, inverted=True) == 1


def test_caft_rater_low_mi_means_stuck(tmp_path):
    digest = SessionDigest(
        session_id="x", source_path="/tmp/x", total_lines=10,
        n_user_messages=1, n_tool_calls=5, n_tool_results=5, n_errors=0,
        error_rate=0.0, duration_seconds=60.0, tool_distribution={"Bash": 5},
        first_user_message="", last_user_message="", sample_user_messages=[],
        longest_repeat_runs=[], top_bash_patterns=[], sample_errors=[],
    )
    metrics = CaftMetrics(action_mi=0.5, tool_entropy=1.0, kl_divergence=0.1,
                          compression_ratio=0.8, anomaly_count=80,
                          event_count=100, health="red")
    ratings = rate_with_caft(digest, metrics=metrics)
    by_dim = {r.dimension: r.value for r in ratings}
    assert by_dim["stuck_in_loop"] == 5  # low MI -> very stuck
    assert by_dim["coherent_progress"] == 1  # low MI -> not coherent
    # v0.3: overall_health retired (CONSTRUCT_REVISION) -> CAFT abstains
    assert by_dim["overall_health"] is None


def test_caft_rater_high_mi_means_healthy():
    digest = SessionDigest(
        session_id="x", source_path="/tmp/x", total_lines=10,
        n_user_messages=1, n_tool_calls=5, n_tool_results=5, n_errors=0,
        error_rate=0.0, duration_seconds=60.0, tool_distribution={"Bash": 5},
        first_user_message="", last_user_message="", sample_user_messages=[],
        longest_repeat_runs=[], top_bash_patterns=[], sample_errors=[],
    )
    metrics = CaftMetrics(action_mi=2.0, tool_entropy=3.0, kl_divergence=0.1,
                          compression_ratio=1.0, anomaly_count=2,
                          event_count=100, health="green")
    ratings = rate_with_caft(digest, metrics=metrics)
    by_dim = {r.dimension: r.value for r in ratings}
    assert by_dim["stuck_in_loop"] == 1
    assert by_dim["coherent_progress"] == 5
    assert by_dim["overall_health"] is None  # retired -> abstain


def test_scale_anchors_cover_all_likert_points():
    from agentdiag.validation.digest import (
        SCALE_ANCHORS, LIKERT_DIMS, HEALTH_ANCHORS, HEALTH_LABELS,
    )
    for d in LIKERT_DIMS:
        assert d in SCALE_ANCHORS
        assert set(SCALE_ANCHORS[d].keys()) == {1, 2, 3, 4, 5}
        for v in (1, 2, 3, 4, 5):
            assert SCALE_ANCHORS[d][v].strip()
    assert set(HEALTH_ANCHORS) == set(HEALTH_LABELS)


def test_ollama_prompt_embeds_shared_anchors():
    from agentdiag.validation.rate_ollama import SYSTEM_PROMPT
    from agentdiag.validation.digest import SCALE_ANCHORS
    # the exact anchor text must appear in the LLM rubric
    assert SCALE_ANCHORS["stuck_in_loop"][5] in SYSTEM_PROMPT
    assert SCALE_ANCHORS["coherent_progress"][1] in SYSTEM_PROMPT
    assert "null" in SYSTEM_PROMPT  # abstention instruction


def test_caft_reasoning_cites_anchor():
    digest = SessionDigest(
        session_id="x", source_path="/tmp/x", total_lines=10,
        n_user_messages=1, n_tool_calls=5, n_tool_results=5, n_errors=0,
        error_rate=0.0, duration_seconds=60.0, tool_distribution={},
        first_user_message="", last_user_message="", sample_user_messages=[],
        longest_repeat_runs=[], top_bash_patterns=[], sample_errors=[],
    )
    metrics = CaftMetrics(action_mi=0.5, tool_entropy=1.0, kl_divergence=0.1,
                          compression_ratio=0.8, anomaly_count=80,
                          event_count=100, health="red")
    ratings = {r.dimension: r for r in rate_with_caft(digest, metrics=metrics)}
    from agentdiag.validation.digest import SCALE_ANCHORS
    assert SCALE_ANCHORS["stuck_in_loop"][5] in ratings["stuck_in_loop"].reasoning


def test_ollama_normalize_allows_abstention():
    from agentdiag.validation.rate_ollama import _validate_and_normalize
    obj = {"stuck_in_loop": 2, "goal_drifted": None,
           "coherent_progress": 4, "user_satisfied": None,
           "overall_health": "healthy", "reasoning": "x"}
    out = _validate_and_normalize(obj)
    assert out["goal_drifted"] is None
    assert out["user_satisfied"] is None
    assert out["stuck_in_loop"] == 2


def test_caft_v02_abstains_on_invalid_dims():
    """CAFT must abstain on user_satisfied + goal_drifted (no valid IT basis)."""
    digest = SessionDigest(
        session_id="x", source_path="/tmp/x", total_lines=10,
        n_user_messages=1, n_tool_calls=5, n_tool_results=5, n_errors=0,
        error_rate=0.0, duration_seconds=60.0, tool_distribution={},
        first_user_message="", last_user_message="", sample_user_messages=[],
        longest_repeat_runs=[], top_bash_patterns=[], sample_errors=[],
    )
    metrics = CaftMetrics(action_mi=0.5, tool_entropy=1.0, kl_divergence=0.1,
                          compression_ratio=0.8, anomaly_count=80,
                          event_count=100, health="red")
    by = {r.dimension: r for r in rate_with_caft(digest, metrics=metrics)}
    assert by["user_satisfied"].value is None
    assert by["user_satisfied"].confidence == ""
    assert "ABSTAIN" in by["user_satisfied"].reasoning
    assert by["goal_drifted"].value is None
    # v0.3: overall_health retired (CONSTRUCT_REVISION) -> abstain too
    assert by["overall_health"].value is None
    assert "retired" in by["overall_health"].reasoning
    # still rates the two dims it has a valid IT basis for
    assert by["stuck_in_loop"].value == 5
    assert by["coherent_progress"].value == 1


def test_caft_rater_returns_one_per_dimension():
    digest = SessionDigest(
        session_id="x", source_path="/tmp/x", total_lines=10,
        n_user_messages=1, n_tool_calls=5, n_tool_results=5, n_errors=0,
        error_rate=0.0, duration_seconds=60.0, tool_distribution={},
        first_user_message="", last_user_message="", sample_user_messages=[],
        longest_repeat_runs=[], top_bash_patterns=[], sample_errors=[],
    )
    metrics = CaftMetrics(action_mi=1.3, tool_entropy=2.0, kl_divergence=0.3,
                          compression_ratio=1.0, anomaly_count=10,
                          event_count=100, health="yellow")
    ratings = rate_with_caft(digest, metrics=metrics)
    assert {r.dimension for r in ratings} == set(DIMENSIONS)
    assert all(r.rater_type == "caft" for r in ratings)


# -------- agreement --------


def test_kappa_perfect_agreement():
    a = [1, 2, 3, 4, 5, 1, 2, 3]
    assert cohens_kappa(a, a, weights="linear") == pytest.approx(1.0)
    assert cohens_kappa(a, a, weights="none") == pytest.approx(1.0)


def test_kappa_complete_disagreement():
    a = [1, 1, 1, 1, 5, 5, 5, 5]
    b = [5, 5, 5, 5, 1, 1, 1, 1]
    k = cohens_kappa(a, b, weights="none")
    assert k < 0  # worse than chance


def test_kappa_linear_weight_helps_adjacent():
    a = [1, 2, 3, 4, 5]
    b = [2, 3, 4, 5, 4]  # all 1 apart
    k_linear = cohens_kappa(a, b, weights="linear")
    k_none = cohens_kappa(a, b, weights="none")
    assert k_linear > k_none  # linear forgives adjacent disagreements


def test_krippendorff_perfect_agreement():
    ratings = {
        "r1": {"s1": 3, "s2": 4, "s3": 5},
        "r2": {"s1": 3, "s2": 4, "s3": 5},
    }
    assert krippendorff_alpha(ratings, level="ordinal") == pytest.approx(1.0)


def test_krippendorff_handles_missing_data():
    ratings = {
        "r1": {"s1": 3, "s2": 4},
        "r2": {"s1": 3, "s3": 5},
    }
    a = krippendorff_alpha(ratings, level="ordinal")
    assert not math.isnan(a)


def test_interpret_kappa_buckets():
    assert interpret_kappa(0.85) == "almost perfect"
    assert interpret_kappa(0.65) == "substantial"
    assert interpret_kappa(0.5) == "moderate"
    assert interpret_kappa(0.3) == "fair"
    assert interpret_kappa(0.1) == "slight"
    assert interpret_kappa(-0.1) == "poor"


def test_compute_agreement_end_to_end(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    for sess in ("s1", "s2", "s3"):
        led.append(Rating(sess, "human", "sam", "stuck_in_loop", 4))
        led.append(Rating(sess, "caft", "v0", "stuck_in_loop", 4))
        led.append(Rating(sess, "human", "sam", "overall_health", "degraded"))
        led.append(Rating(sess, "caft", "v0", "overall_health", "degraded"))
    out = compute_agreement(led)
    # raters are sorted alphabetically -> ("caft:v0", "human:sam")
    pair = ("caft:v0", "human:sam")
    assert pair in out["per_dimension"]["stuck_in_loop"]["pair_kappa"]
    assert out["per_dimension"]["stuck_in_loop"]["pair_kappa"][
        pair] == pytest.approx(1.0)


# -------- report --------


def test_report_writes_markdown(tmp_path):
    led = Ledger(tmp_path / "r.jsonl")
    led.append(Rating("s1", "human", "sam", "stuck_in_loop", 4))
    led.append(Rating("s1", "caft", "v0", "stuck_in_loop", 5))
    led.append(Rating("s1", "human", "sam", "overall_health", "degraded"))
    led.append(Rating("s1", "caft", "v0", "overall_health", "pathological"))
    out = tmp_path / "report.md"
    write_report(led, out)
    text = out.read_text()
    assert "# CAFT Validation Report" in text
    assert "Cohen's kappa" in text
    assert "Krippendorff" in text
    assert "human:sam" in text
