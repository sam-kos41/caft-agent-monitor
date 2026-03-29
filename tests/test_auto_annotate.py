"""Tests for the automated CAFT trace annotation pipeline.

Covers:
  1. extract_trace_summary() — all fields populated from synthetic JSONL
  2. parse_annotation_response() — valid/invalid JSON, markdown fences, validation
  3. merge_annotations() — dedup, metadata, stats
  4. validate_agreement() — precision/recall/kappa on overlapping traces
  5. prepare_batch() — manifest filtering, ground truth exclusion
  6. build_annotation_prompt() — prompt construction
  7. Integration — full pipeline round-trip
"""

import json
import textwrap
from pathlib import Path

import pytest

from agentdiag.auto_annotate import (
    extract_trace_summary,
    prepare_batch,
    parse_annotation_response,
    merge_annotations,
    validate_agreement,
)
from agentdiag.auto_annotate_prompt import (
    ANNOTATION_CRITERIA,
    OBSERVABLE_CAFT_CODES,
    OUTPUT_FORMAT,
    build_annotation_prompt,
)


# ======================================================================
# Fixtures
# ======================================================================

def _make_synthetic_session(path: Path, *, with_errors: bool = True,
                             with_plan_mode: bool = True) -> Path:
    """Write a minimal synthetic Claude Code session JSONL file.

    Contains: user message, assistant thinking, tool_use, tool_result,
    optionally errors and ExitPlanMode.
    """
    events = [
        # User message
        {
            "type": "user",
            "timestamp": "2026-03-17T10:00:00.000Z",
            "message": {
                "content": "Fix the login bug in auth.py"
            }
        },
        # Assistant thinking + tool_use
        {
            "type": "assistant",
            "timestamp": "2026-03-17T10:00:05.000Z",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "Let me read the auth file first."},
                    {"type": "tool_use", "id": "tu_001", "name": "Read",
                     "input": {"file_path": "/src/auth.py"}},
                ],
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
        },
        # Tool result (success)
        {
            "type": "user",
            "timestamp": "2026-03-17T10:00:06.000Z",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_001",
                     "is_error": False,
                     "content": "def login(user, password):\n    return True"},
                ]
            }
        },
        # Assistant text + another tool_use
        {
            "type": "assistant",
            "timestamp": "2026-03-17T10:00:10.000Z",
            "message": {
                "content": [
                    {"type": "text", "text": "I found the issue. Let me fix it."},
                    {"type": "tool_use", "id": "tu_002", "name": "Edit",
                     "input": {"file_path": "/src/auth.py",
                               "old_string": "return True",
                               "new_string": "return check_password(user, password)"}},
                ],
                "usage": {"input_tokens": 200, "output_tokens": 100},
            }
        },
        # Tool result (success)
        {
            "type": "user",
            "timestamp": "2026-03-17T10:00:11.000Z",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_002",
                     "is_error": False,
                     "content": "Edit applied successfully"},
                ]
            }
        },
    ]

    if with_errors:
        # Add an error tool result
        events.append({
            "type": "assistant",
            "timestamp": "2026-03-17T10:00:15.000Z",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "tu_003", "name": "Bash",
                     "input": {"command": "python -m pytest tests/"}},
                ],
                "usage": {"input_tokens": 50, "output_tokens": 30},
            }
        })
        events.append({
            "type": "user",
            "timestamp": "2026-03-17T10:00:20.000Z",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_003",
                     "is_error": True,
                     "content": "ModuleNotFoundError: No module named 'auth'"},
                ]
            }
        })

    if with_plan_mode:
        events.append({
            "type": "assistant",
            "timestamp": "2026-03-17T10:00:25.000Z",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "tu_004", "name": "ExitPlanMode",
                     "input": {}},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        })
        events.append({
            "type": "user",
            "timestamp": "2026-03-17T10:00:26.000Z",
            "message": {
                "content": "Looks good, proceed."
            }
        })

    # Final assistant response
    events.append({
        "type": "assistant",
        "timestamp": "2026-03-17T10:00:30.000Z",
        "message": {
            "content": [
                {"type": "text", "text": "Done! The login bug has been fixed."},
            ],
            "usage": {"input_tokens": 50, "output_tokens": 20},
        }
    })

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    return path


@pytest.fixture
def synthetic_session(tmp_path):
    """Create a synthetic session JSONL file with errors and plan mode."""
    return _make_synthetic_session(
        tmp_path / "sessions" / "abc12345-1234-1234-1234-123456789abc.jsonl",
        with_errors=True,
        with_plan_mode=True,
    )


@pytest.fixture
def clean_session(tmp_path):
    """Create a minimal clean session (no errors, no plan mode)."""
    return _make_synthetic_session(
        tmp_path / "sessions" / "def67890-1234-1234-1234-123456789abc.jsonl",
        with_errors=False,
        with_plan_mode=False,
    )


@pytest.fixture
def sample_ground_truth(tmp_path):
    """Create a sample ground truth JSON file."""
    gt = {
        "annotator": "test",
        "date": "2026-03-17",
        "method": "test",
        "criteria": ANNOTATION_CRITERIA,
        "traces": [
            {
                "trace_num": 1,
                "session_id": "existing1",
                "project": "TestProject",
                "events": 100,
                "user_goal": "Fix a bug",
                "agent_completed": True,
                "actual_failures": ["step_repetition"],
                "failure_details": [{
                    "caft_code": "2.2",
                    "caft_name": "step_repetition",
                    "onset_step": 10,
                    "severity": 3,
                    "confidence": 4,
                    "rationale": "Repeated read 5 times",
                }],
                "annotations": {},
            },
            {
                "trace_num": 2,
                "session_id": "existing2",
                "project": "TestProject",
                "events": 200,
                "user_goal": "Add feature",
                "agent_completed": True,
                "actual_failures": [],
                "failure_details": [],
                "annotations": {},
            },
        ],
    }
    path = tmp_path / "ground_truth.json"
    with open(path, "w") as f:
        json.dump(gt, f)
    return path


@pytest.fixture
def sample_manifest(tmp_path, synthetic_session, clean_session):
    """Create a manifest CSV with sessions."""
    manifest_path = tmp_path / "manifest.csv"
    sid1 = synthetic_session.stem
    sid2 = clean_session.stem
    with open(manifest_path, "w") as f:
        f.write("session_id,bucket,project,line_count,first_timestamp,"
                "last_timestamp,has_tool_use,error_count,tools_used,"
                "label_status,caft_labels,notes\n")
        f.write(f"{sid1},test,-Test-Project,100,,,True,1,"
                f"\"Read,Edit,Bash\",unlabeled,,\n")
        f.write(f"{sid2},test,-Test-Project,80,,,True,0,"
                f"\"Read,Edit\",unlabeled,,\n")
        # Add a session that's already in ground truth (should be filtered)
        f.write(f"existing1-1234-1234-1234-123456789abc,test,-Test-Project,150,"
                f",,True,0,\"Read\",unlabeled,,\n")
    return manifest_path


# ======================================================================
# 1. extract_trace_summary
# ======================================================================

class TestExtractTraceSummary:
    def test_all_fields_present(self, synthetic_session):
        summary = extract_trace_summary(synthetic_session)

        assert "total_lines" in summary
        assert "summary" in summary
        assert "heuristic_results" in summary

        s = summary["summary"]
        assert "user_messages" in s
        assert "errors" in s
        assert "first_events" in s
        assert "last_events" in s
        assert "tool_sequence" in s
        assert "plan_mode_exits" in s
        assert "error_count" in s

    def test_user_messages_extracted(self, synthetic_session):
        summary = extract_trace_summary(synthetic_session)
        user_msgs = summary["summary"]["user_messages"]

        assert len(user_msgs) >= 1
        assert user_msgs[0]["content"] == "Fix the login bug in auth.py"
        assert "line" in user_msgs[0]

    def test_errors_extracted(self, synthetic_session):
        summary = extract_trace_summary(synthetic_session)
        errors = summary["summary"]["errors"]

        assert len(errors) >= 1
        assert "ModuleNotFoundError" in errors[0]["content"]
        assert summary["summary"]["error_count"] >= 1

    def test_tool_sequence_extracted(self, synthetic_session):
        summary = extract_trace_summary(synthetic_session)
        tools = summary["summary"]["tool_sequence"]

        assert "Read" in tools
        assert "Edit" in tools
        assert "Bash" in tools
        assert "ExitPlanMode" in tools

    def test_first_last_events(self, synthetic_session):
        summary = extract_trace_summary(synthetic_session)

        assert len(summary["summary"]["first_events"]) <= 3
        assert len(summary["summary"]["last_events"]) <= 3
        assert summary["summary"]["first_events"][0]["type"] == "user"

    def test_plan_mode_exits_detected(self, synthetic_session):
        summary = extract_trace_summary(synthetic_session)
        exits = summary["summary"]["plan_mode_exits"]

        assert len(exits) >= 1
        assert "line" in exits[0]
        assert "rejected" in exits[0]
        assert exits[0]["rejected"] is False  # "Looks good, proceed" is not rejection

    def test_heuristic_results_populated(self, synthetic_session):
        summary = extract_trace_summary(synthetic_session)
        heuristic = summary["heuristic_results"]

        assert "detectors_fired" in heuristic
        assert "trust_score" in heuristic
        assert "health" in heuristic
        assert isinstance(heuristic["detectors_fired"], list)

    def test_clean_session(self, clean_session):
        summary = extract_trace_summary(clean_session)

        assert summary["summary"]["error_count"] == 0
        assert len(summary["summary"]["errors"]) == 0
        assert len(summary["summary"]["plan_mode_exits"]) == 0

    def test_user_message_truncation(self, tmp_path):
        """User messages should be truncated to 300 chars."""
        path = tmp_path / "long_msg.jsonl"
        event = {
            "type": "user",
            "timestamp": "2026-03-17T10:00:00.000Z",
            "message": {"content": "x" * 500}
        }
        with open(path, "w") as f:
            f.write(json.dumps(event) + "\n")

        summary = extract_trace_summary(path)
        assert len(summary["summary"]["user_messages"][0]["content"]) <= 300

    def test_error_message_truncation(self, tmp_path):
        """Error messages should be truncated to 200 chars."""
        path = tmp_path / "long_error.jsonl"
        events = [
            {
                "type": "assistant",
                "timestamp": "2026-03-17T10:00:00.000Z",
                "message": {
                    "content": [{"type": "tool_use", "id": "tu_x", "name": "Bash",
                                 "input": {}}],
                    "usage": {},
                }
            },
            {
                "type": "user",
                "timestamp": "2026-03-17T10:00:01.000Z",
                "message": {
                    "content": [{"type": "tool_result", "tool_use_id": "tu_x",
                                 "is_error": True, "content": "E" * 500}]
                }
            },
        ]
        with open(path, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        summary = extract_trace_summary(path)
        assert len(summary["summary"]["errors"][0]["content"]) <= 200


# ======================================================================
# 2. parse_annotation_response
# ======================================================================

class TestParseAnnotationResponse:
    def test_valid_json_array(self):
        data = json.dumps([
            {
                "trace_num": 1,
                "session_id": "abc12345",
                "project": "Test",
                "events": 100,
                "user_goal": "Fix bug",
                "agent_completed": True,
                "actual_failures": [],
                "failure_details": [],
                "annotations": {},
            }
        ])
        result = parse_annotation_response(data)
        assert len(result) == 1
        assert result[0]["session_id"] == "abc12345"
        assert result[0]["actual_failures"] == []

    def test_json_with_failures(self):
        data = json.dumps([
            {
                "trace_num": 1,
                "session_id": "abc12345",
                "project": "Test",
                "events": 100,
                "user_goal": "Fix bug",
                "agent_completed": False,
                "actual_failures": ["step_repetition"],
                "failure_details": [{
                    "caft_code": "2.2",
                    "caft_name": "step_repetition",
                    "onset_step": 15,
                    "severity": 4,
                    "confidence": 3,
                    "rationale": "Repeated read operations",
                }],
                "annotations": {},
            }
        ])
        result = parse_annotation_response(data)
        assert len(result) == 1
        assert result[0]["actual_failures"] == ["step_repetition"]
        assert result[0]["failure_details"][0]["caft_code"] == "2.2"

    def test_strips_markdown_fences(self):
        data = "```json\n" + json.dumps([{
            "trace_num": 1,
            "session_id": "abc",
            "user_goal": "test",
            "agent_completed": True,
            "actual_failures": [],
        }]) + "\n```"
        result = parse_annotation_response(data)
        assert len(result) == 1

    def test_accepts_wrapped_object(self):
        """Accepts {"traces": [...]} format."""
        data = json.dumps({
            "traces": [{
                "trace_num": 1,
                "session_id": "abc",
                "user_goal": "test",
                "agent_completed": True,
                "actual_failures": [],
            }]
        })
        result = parse_annotation_response(data)
        assert len(result) == 1

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_annotation_response("not json at all")

    def test_non_array_non_object_raises(self):
        with pytest.raises(ValueError, match="Expected JSON array"):
            parse_annotation_response('"just a string"')

    def test_defaults_for_missing_fields(self):
        data = json.dumps([{"session_id": "abc", "user_goal": "test",
                            "agent_completed": True}])
        result = parse_annotation_response(data)
        assert result[0]["actual_failures"] == []
        assert result[0]["failure_details"] == []
        assert result[0]["annotations"] == {}

    def test_warns_on_unknown_caft_name(self, capsys):
        data = json.dumps([{
            "trace_num": 1,
            "session_id": "abc",
            "user_goal": "test",
            "agent_completed": True,
            "actual_failures": ["totally_made_up"],
        }])
        result = parse_annotation_response(data)
        captured = capsys.readouterr()
        assert "unknown CAFT name" in captured.err

    def test_warns_on_invalid_severity(self, capsys):
        data = json.dumps([{
            "trace_num": 1,
            "session_id": "abc",
            "user_goal": "test",
            "agent_completed": True,
            "actual_failures": [],
            "failure_details": [{"caft_code": "2.2", "severity": 0, "confidence": 3}],
        }])
        parse_annotation_response(data)
        captured = capsys.readouterr()
        assert "severity" in captured.err

    def test_failure_details_cleaned(self):
        data = json.dumps([{
            "trace_num": 1,
            "session_id": "abc",
            "user_goal": "test",
            "agent_completed": True,
            "actual_failures": ["step_repetition"],
            "failure_details": [{
                "caft_code": 2.2,  # number, should be string
                "caft_name": "step_repetition",
                "onset_step": "10",  # string, should be int
                "severity": 3,
                "confidence": 4,
                "rationale": "test",
                "extra_field": "ignored",  # should be stripped
            }],
        }])
        result = parse_annotation_response(data)
        detail = result[0]["failure_details"][0]
        assert isinstance(detail["caft_code"], str)
        assert isinstance(detail["onset_step"], int)
        assert "extra_field" not in detail


# ======================================================================
# 3. merge_annotations
# ======================================================================

class TestMergeAnnotations:
    def test_merge_into_empty(self, tmp_path):
        output = tmp_path / "merged.json"
        new_anns = [{
            "trace_num": 1,
            "session_id": "new1",
            "project": "Test",
            "events": 100,
            "user_goal": "Fix bug",
            "agent_completed": True,
            "actual_failures": [],
            "failure_details": [],
            "annotations": {},
        }]
        result = merge_annotations(new_anns, existing_path=None,
                                    output_path=output)
        assert output.exists()
        assert len(result["traces"]) == 1
        assert result["traces"][0]["session_id"] == "new1"

    def test_merge_into_existing(self, tmp_path, sample_ground_truth):
        output = tmp_path / "merged.json"
        new_anns = [{
            "trace_num": 3,
            "session_id": "new3",
            "project": "Test",
            "events": 50,
            "user_goal": "Add tests",
            "agent_completed": True,
            "actual_failures": [],
            "failure_details": [],
            "annotations": {},
        }]
        result = merge_annotations(new_anns, existing_path=sample_ground_truth,
                                    output_path=output)
        assert len(result["traces"]) == 3  # 2 existing + 1 new

    def test_dedup_by_session_id(self, tmp_path, sample_ground_truth):
        output = tmp_path / "merged.json"
        # Update existing session
        new_anns = [{
            "trace_num": 1,
            "session_id": "existing1",
            "project": "Test",
            "events": 150,  # updated
            "user_goal": "Fix a bug (updated)",
            "agent_completed": True,
            "actual_failures": [],
            "failure_details": [],
            "annotations": {},
        }]
        result = merge_annotations(new_anns, existing_path=sample_ground_truth,
                                    output_path=output)
        assert len(result["traces"]) == 2  # still 2, one updated
        updated = [t for t in result["traces"] if t["session_id"] == "existing1"][0]
        assert updated["user_goal"] == "Fix a bug (updated)"

    def test_sorted_by_trace_num(self, tmp_path):
        output = tmp_path / "merged.json"
        new_anns = [
            {"trace_num": 3, "session_id": "c", "user_goal": "c",
             "agent_completed": True, "actual_failures": []},
            {"trace_num": 1, "session_id": "a", "user_goal": "a",
             "agent_completed": True, "actual_failures": []},
            {"trace_num": 2, "session_id": "b", "user_goal": "b",
             "agent_completed": True, "actual_failures": []},
        ]
        result = merge_annotations(new_anns, None, output)
        nums = [t["trace_num"] for t in result["traces"]]
        assert nums == [1, 2, 3]

    def test_date_updated(self, tmp_path):
        output = tmp_path / "merged.json"
        result = merge_annotations(
            [{"trace_num": 1, "session_id": "x", "user_goal": "x",
              "agent_completed": True, "actual_failures": []}],
            None, output,
        )
        assert result["date"]  # non-empty date string


# ======================================================================
# 4. validate_agreement
# ======================================================================

class TestValidateAgreement:
    def _write_gt(self, path: Path, traces: list[dict]) -> None:
        with open(path, "w") as f:
            json.dump({"traces": traces}, f)

    def test_perfect_agreement(self, tmp_path):
        traces = [
            {"session_id": "s1", "actual_failures": ["step_repetition"],
             "agent_completed": True},
            {"session_id": "s2", "actual_failures": [],
             "agent_completed": True},
        ]
        auto_path = tmp_path / "auto.json"
        manual_path = tmp_path / "manual.json"
        self._write_gt(auto_path, traces)
        self._write_gt(manual_path, traces)

        result = validate_agreement(auto_path, manual_path)
        assert result["overlap"] == 2
        assert result["cohens_kappa"] == 1.0

    def test_no_overlap(self, tmp_path):
        auto_path = tmp_path / "auto.json"
        manual_path = tmp_path / "manual.json"
        self._write_gt(auto_path, [{"session_id": "s1", "actual_failures": []}])
        self._write_gt(manual_path, [{"session_id": "s2", "actual_failures": []}])

        result = validate_agreement(auto_path, manual_path)
        assert result["overlap"] == 0

    def test_partial_agreement(self, tmp_path):
        auto_path = tmp_path / "auto.json"
        manual_path = tmp_path / "manual.json"
        self._write_gt(auto_path, [
            {"session_id": "s1", "actual_failures": ["step_repetition"],
             "agent_completed": True},
            {"session_id": "s2", "actual_failures": [],
             "agent_completed": True},
        ])
        self._write_gt(manual_path, [
            {"session_id": "s1", "actual_failures": ["step_repetition"],
             "agent_completed": True},
            {"session_id": "s2", "actual_failures": ["context_loss"],
             "agent_completed": False},
        ])

        result = validate_agreement(auto_path, manual_path)
        assert result["overlap"] == 2
        assert result["per_failure_type"]["step_repetition"]["tp"] == 1
        # context_loss: auto=no, manual=yes → fn=1
        assert result["per_failure_type"]["context_loss"]["fn"] == 1

    def test_completion_agreement(self, tmp_path):
        auto_path = tmp_path / "auto.json"
        manual_path = tmp_path / "manual.json"
        self._write_gt(auto_path, [
            {"session_id": "s1", "actual_failures": [],
             "agent_completed": True},
            {"session_id": "s2", "actual_failures": [],
             "agent_completed": False},
        ])
        self._write_gt(manual_path, [
            {"session_id": "s1", "actual_failures": [],
             "agent_completed": True},
            {"session_id": "s2", "actual_failures": [],
             "agent_completed": True},
        ])

        result = validate_agreement(auto_path, manual_path)
        assert result["completion_agreement"] == "1/2"


# ======================================================================
# 5. prepare_batch
# ======================================================================

class TestPrepareBatch:
    def test_produces_output_file(self, tmp_path, sample_manifest,
                                   synthetic_session, clean_session):
        output = tmp_path / "batch.json"
        result = prepare_batch(
            manifest_path=sample_manifest,
            ground_truth_path=None,
            traces_root=tmp_path / "sessions",
            n=10,
            output_path=output,
        )
        assert output.exists()
        assert "traces" in result
        assert "annotation_instructions" in result

    def test_excludes_already_labeled(self, tmp_path, sample_manifest,
                                       sample_ground_truth, synthetic_session,
                                       clean_session):
        output = tmp_path / "batch.json"
        result = prepare_batch(
            manifest_path=sample_manifest,
            ground_truth_path=sample_ground_truth,
            traces_root=tmp_path / "sessions",
            n=10,
            output_path=output,
        )
        # "existing1" from ground truth should not appear in batch
        session_ids = [t["session_id"] for t in result["traces"]]
        assert "existing1" not in session_ids

    def test_annotation_instructions_present(self, tmp_path, sample_manifest,
                                              synthetic_session, clean_session):
        output = tmp_path / "batch.json"
        result = prepare_batch(
            manifest_path=sample_manifest,
            ground_truth_path=None,
            traces_root=tmp_path / "sessions",
            n=10,
            output_path=output,
        )
        instructions = result["annotation_instructions"]
        assert "criteria" in instructions
        assert "caft_codes" in instructions
        assert "output_format" in instructions

    def test_trace_fields_populated(self, tmp_path, sample_manifest,
                                     synthetic_session, clean_session):
        output = tmp_path / "batch.json"
        result = prepare_batch(
            manifest_path=sample_manifest,
            ground_truth_path=None,
            traces_root=tmp_path / "sessions",
            n=10,
            output_path=output,
        )
        if result["traces"]:
            trace = result["traces"][0]
            assert "trace_num" in trace
            assert "session_id" in trace
            assert "jsonl_path" in trace
            assert "total_lines" in trace
            assert "summary" in trace
            assert "heuristic_results" in trace

    def test_respects_n_limit(self, tmp_path):
        """Verify n parameter limits the number of traces."""
        manifest_path = tmp_path / "manifest.csv"
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        lines = ["session_id,bucket,project,line_count,first_timestamp,"
                 "last_timestamp,has_tool_use,error_count,tools_used,"
                 "label_status,caft_labels,notes\n"]
        for i in range(5):
            sid = f"aaaa{i:04d}-1234-1234-1234-123456789abc"
            _make_synthetic_session(sessions_dir / f"{sid}.jsonl",
                                    with_errors=False, with_plan_mode=False)
            lines.append(f"{sid},test,-Test,80,,,True,0,\"Read\",unlabeled,,\n")

        with open(manifest_path, "w") as f:
            f.writelines(lines)

        output = tmp_path / "batch.json"
        result = prepare_batch(
            manifest_path=manifest_path,
            ground_truth_path=None,
            traces_root=sessions_dir,
            n=2,
            output_path=output,
        )
        assert len(result["traces"]) <= 2


# ======================================================================
# 6. Prompt construction
# ======================================================================

class TestPromptConstruction:
    def test_annotation_criteria_keys(self):
        expected = {
            "step_repetition", "goal_drift", "missing_verification",
            "context_loss", "premature_termination",
            "reasoning_action_mismatch", "error_cascade",
            "recovery_failure", "analysis_paralysis",
        }
        assert set(ANNOTATION_CRITERIA.keys()) == expected

    def test_observable_caft_codes_valid(self):
        from agentdiag.caft.taxonomy import CAFT_TAXONOMY
        for code in OBSERVABLE_CAFT_CODES:
            assert code in CAFT_TAXONOMY, f"CAFT code {code} not in taxonomy"

    def test_output_format_has_required_fields(self):
        required = {"trace_num", "session_id", "project", "events",
                     "user_goal", "agent_completed", "actual_failures",
                     "failure_details", "annotations"}
        assert required.issubset(set(OUTPUT_FORMAT.keys()))

    def test_build_annotation_prompt_contains_criteria(self):
        batch = {
            "traces": [{"trace_num": 1, "session_id": "abc"}],
            "annotation_instructions": {},
        }
        prompt = build_annotation_prompt(batch)
        assert "step_repetition" in prompt
        assert "goal_drift" in prompt
        assert "CRITICAL" in prompt
        assert "JSON array" in prompt

    def test_build_annotation_prompt_contains_trace_count(self):
        batch = {"traces": [{"x": 1}, {"x": 2}, {"x": 3}]}
        prompt = build_annotation_prompt(batch)
        assert "3" in prompt


# ======================================================================
# 7. Integration — round-trip
# ======================================================================

class TestIntegration:
    def test_extract_parse_merge_roundtrip(self, synthetic_session, tmp_path):
        """Full pipeline: extract summary → simulate annotation → parse → merge."""
        # Step 1: Extract
        summary = extract_trace_summary(synthetic_session)
        assert summary["total_lines"] > 0

        # Step 2: Simulate annotation response (as if from LLM)
        annotation = json.dumps([{
            "trace_num": 1,
            "session_id": synthetic_session.stem[:8],
            "project": "Test",
            "events": summary["total_lines"],
            "user_goal": "Fix the login bug in auth.py",
            "agent_completed": True,
            "actual_failures": [],
            "failure_details": [],
            "annotations": {},
        }])

        # Step 3: Parse
        parsed = parse_annotation_response(annotation)
        assert len(parsed) == 1

        # Step 4: Merge
        output = tmp_path / "output_gt.json"
        result = merge_annotations(parsed, existing_path=None,
                                    output_path=output)
        assert output.exists()
        assert len(result["traces"]) == 1
        assert result["traces"][0]["user_goal"] == "Fix the login bug in auth.py"

    def test_full_pipeline_with_failures(self, synthetic_session, tmp_path):
        """Full pipeline with a trace that has failures."""
        summary = extract_trace_summary(synthetic_session)

        annotation = json.dumps([{
            "trace_num": 1,
            "session_id": synthetic_session.stem[:8],
            "project": "Test",
            "events": summary["total_lines"],
            "user_goal": "Fix the login bug",
            "agent_completed": False,
            "actual_failures": ["recovery_failure"],
            "failure_details": [{
                "caft_code": "4.3",
                "caft_name": "recovery_failure",
                "onset_step": 5,
                "severity": 4,
                "confidence": 3,
                "rationale": "Agent failed to recover after ModuleNotFoundError",
            }],
            "annotations": {},
        }])

        parsed = parse_annotation_response(annotation)
        output = tmp_path / "output_gt.json"
        result = merge_annotations(parsed, None, output)

        assert len(result["traces"]) == 1
        assert result["traces"][0]["actual_failures"] == ["recovery_failure"]
        assert result["traces"][0]["failure_details"][0]["caft_code"] == "4.3"

    def test_validate_after_merge(self, tmp_path):
        """Merge then validate agreement between two ground truth files."""
        # Create "auto" ground truth
        auto_traces = [
            {"trace_num": 1, "session_id": "s1", "project": "T",
             "events": 100, "user_goal": "fix", "agent_completed": True,
             "actual_failures": ["step_repetition"], "failure_details": [],
             "annotations": {}},
            {"trace_num": 2, "session_id": "s2", "project": "T",
             "events": 50, "user_goal": "add", "agent_completed": True,
             "actual_failures": [], "failure_details": [],
             "annotations": {}},
        ]
        auto_output = tmp_path / "auto_gt.json"
        merge_annotations(auto_traces, None, auto_output)

        # Create "manual" ground truth
        manual_traces = [
            {"trace_num": 1, "session_id": "s1", "project": "T",
             "events": 100, "user_goal": "fix", "agent_completed": True,
             "actual_failures": ["step_repetition"], "failure_details": [],
             "annotations": {}},
            {"trace_num": 2, "session_id": "s2", "project": "T",
             "events": 50, "user_goal": "add", "agent_completed": True,
             "actual_failures": [], "failure_details": [],
             "annotations": {}},
        ]
        manual_output = tmp_path / "manual_gt.json"
        merge_annotations(manual_traces, None, manual_output)

        # Validate
        result = validate_agreement(auto_output, manual_output)
        assert result["overlap"] == 2
        assert result["cohens_kappa"] == 1.0
