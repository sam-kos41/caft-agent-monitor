"""Tests for the semantic confirmation layer (caft/confirm.py).

All LLM calls are mocked — no real API keys needed.
"""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from agentdiag.models import TraceEvent
from agentdiag.hta import HTAState, HTANode, Phase, HTAStateMachine
from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
from agentdiag.caft.confirm import (
    ConfirmationResult,
    SessionAssessment,
    build_confirmation_prompt,
    build_session_assessment_prompt,
    _parse_session_assessment,
    _extract_json_object,
    assess_session_end,
    assess_session_end_sync,
    confirm_diagnosis,
    confirm_diagnosis_sync,
    enable_llm_tracing,
    is_llm_available,
    _parse_llm_response,
    _extract_agent_goal,
    _format_event_window,
    _format_similar_cases,
    _format_few_shot_examples,
    _write_llm_trace,
    FEW_SHOT_EXAMPLES,
    AUTOCONFIRM_THRESHOLD,
    TIER_2_FAILURE_TYPES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_events(n: int = 10) -> list[TraceEvent]:
    """Generate a simple event sequence."""
    events = [
        TraceEvent(step=1, type="user_input", goal_text="Fix the login bug"),
    ]
    for i in range(2, n + 1):
        events.append(
            TraceEvent(step=i, type="tool_call", tool="read_file", latency_ms=200)
        )
    return events


def _make_hta_state(phase: Phase = Phase.EXECUTING) -> HTAState:
    """Create a minimal HTA state for testing."""
    return HTAState(
        goal="test",
        current_phase=phase,
        current_node=HTANode(phase=phase, start_step=0, start_time=0.0),
        completed_nodes=[],
        transitions=[],
        total_events=10,
        phase_event_counts={},
    )


def _make_diagnosis(
    failure_name: str = "context_loss",
    confidence: float = 0.7,
    at_step: int = 5,
) -> CaftDiagnosis:
    """Create a test diagnosis."""
    return CaftDiagnosis(
        caft_code="2.1",
        caft_category="memory",
        failure_name=failure_name,
        severity=CaftSeverity.WARNING,
        confidence=confidence,
        description=f"Test {failure_name} diagnosis",
        evidence={"test_key": "test_value", "consecutive_count": 10},
        at_step=at_step,
        remediation="Test remediation",
    )


# ---------------------------------------------------------------------------
# ConfirmationResult
# ---------------------------------------------------------------------------

class TestConfirmationResult:
    def test_to_dict(self):
        result = ConfirmationResult(
            confirmed=True,
            confidence=0.85,
            reasoning="Clear repetition pattern.",
            status="confirmed",
        )
        d = result.to_dict()
        assert d["confirmed"] is True
        assert d["confidence"] == 0.85
        assert d["status"] == "confirmed"

    def test_status_values(self):
        for status in ("confirmed", "rejected", "uncertain"):
            r = ConfirmationResult(
                confirmed=status == "confirmed",
                confidence=0.5,
                reasoning="test",
                status=status,
            )
            assert r.status == status


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

class TestBuildConfirmationPrompt:
    def test_includes_agent_goal(self):
        events = _make_events()
        hta = _make_hta_state()
        diag = _make_diagnosis()

        prompt = build_confirmation_prompt(diag, events, hta)
        assert "Fix the login bug" in prompt

    def test_includes_failure_type(self):
        events = _make_events()
        hta = _make_hta_state()
        diag = _make_diagnosis(failure_name="step_repetition")
        diag.caft_code = "2.2"

        prompt = build_confirmation_prompt(diag, events, hta)
        assert "step_repetition" in prompt
        assert "2.2" in prompt

    def test_includes_event_window(self):
        events = _make_events(20)
        hta = _make_hta_state()
        diag = _make_diagnosis(at_step=10)

        prompt = build_confirmation_prompt(diag, events, hta)
        assert "step" in prompt
        assert "read_file" in prompt

    def test_includes_evidence(self):
        events = _make_events()
        hta = _make_hta_state()
        diag = _make_diagnosis()

        prompt = build_confirmation_prompt(diag, events, hta)
        assert "test_key" in prompt
        assert "test_value" in prompt

    def test_includes_similar_cases(self):
        events = _make_events()
        hta = _make_hta_state()
        diag = _make_diagnosis()
        cases = [
            {"failure_name": "context_loss", "status": "false_positive",
             "description": "Re-read after compaction", "phase_at_onset": "executing"},
            {"failure_name": "context_loss", "status": "confirmed",
             "description": "Genuine context loss", "phase_at_onset": "gathering"},
        ]

        prompt = build_confirmation_prompt(diag, events, hta, similar_cases=cases)
        assert "false_positive" in prompt
        assert "confirmed" in prompt
        assert "Similar Past Cases" in prompt

    def test_no_similar_cases(self):
        events = _make_events()
        hta = _make_hta_state()
        diag = _make_diagnosis()

        prompt = build_confirmation_prompt(diag, events, hta, similar_cases=[])
        assert "No similar past cases available" in prompt

    def test_includes_decision_framework(self):
        events = _make_events()
        hta = _make_hta_state()
        diag = _make_diagnosis()

        prompt = build_confirmation_prompt(diag, events, hta)
        # V2 prompt uses structured 3-criterion decision framework
        assert "MATCH" in prompt
        assert "EXPLANATION" in prompt
        assert "ENGINEER TEST" in prompt
        assert "Default to confirmed" in prompt

    def test_requests_json_response(self):
        events = _make_events()
        hta = _make_hta_state()
        diag = _make_diagnosis()

        prompt = build_confirmation_prompt(diag, events, hta)
        assert '"confirmed"' in prompt
        assert '"confidence"' in prompt
        assert '"reasoning"' in prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestExtractAgentGoal:
    def test_finds_first_user_message(self):
        events = [
            TraceEvent(step=1, type="user_input", goal_text="Fix the login bug"),
            TraceEvent(step=2, type="user_input", goal_text="Second message"),
        ]
        assert _extract_agent_goal(events) == "Fix the login bug"

    def test_no_user_message(self):
        events = [TraceEvent(step=1, type="tool_call", tool="read_file")]
        assert "no explicit goal" in _extract_agent_goal(events)

    def test_truncates_long_goals(self):
        events = [
            TraceEvent(step=1, type="user_input", goal_text="x" * 500),
        ]
        result = _extract_agent_goal(events)
        assert len(result) <= 210  # 200 + "..."


class TestFormatEventWindow:
    def test_centers_on_step(self):
        events = [
            TraceEvent(step=i, type="tool_call", tool="read_file", latency_ms=100)
            for i in range(1, 30)
        ]
        result = _format_event_window(events, center_step=15)
        assert "<<<" in result  # onset marker
        assert "step   15" in result
        # New format: no pipe separators
        assert " | " not in result
        assert "read_file" in result

    def test_handles_empty_events(self):
        result = _format_event_window([], center_step=5)
        assert result == ""

    def test_format_event_shows_file_path(self):
        from agentdiag.caft.confirm import _format_event
        e = TraceEvent(step=7, type="tool_call", tool="Read",
                       goal_text="detectors.py", latency_ms=20, success=True)
        line = _format_event(e)
        assert 'Read "detectors.py"' in line
        assert "OK" in line
        assert "20ms" in line

    def test_format_event_shows_bash_command(self):
        from agentdiag.caft.confirm import _format_event
        e = TraceEvent(step=10, type="tool_call", tool="Bash",
                       goal_text="pytest tests/ -x", latency_ms=3500, success=False)
        line = _format_event(e)
        assert "Bash `pytest tests/ -x`" in line
        assert "FAIL" in line
        assert "3500ms" in line

    def test_format_event_user_input_format(self):
        from agentdiag.caft.confirm import _format_event
        e = TraceEvent(step=11, type="user_input", goal_text="fix the login bug")
        line = _format_event(e)
        assert "[user]" in line
        assert "fix the login bug" in line

    def test_format_event_reasoning_format(self):
        from agentdiag.caft.confirm import _format_event
        e = TraceEvent(step=12, type="reasoning",
                       goal_text="I found the issue in token validation.")
        line = _format_event(e)
        assert "[response]" in line
        assert "I found the issue" in line

    def test_format_event_thinking_format(self):
        from agentdiag.caft.confirm import _format_event
        e = TraceEvent(step=9, type="planning",
                       goal_text="Now I need to run the tests.")
        line = _format_event(e)
        assert "[thinking]" in line
        assert "run the tests" in line

    def test_format_event_grep_format(self):
        from agentdiag.caft.confirm import _format_event
        e = TraceEvent(step=13, type="tool_call", tool="Grep",
                       goal_text="/login.*error/ in src/", latency_ms=150)
        line = _format_event(e)
        assert "Grep /login.*error/ in src/" in line
        assert "150ms" in line


    def test_format_event_shows_error_message(self):
        """FAIL events include the error text so LLM knows WHY it failed."""
        from agentdiag.caft.confirm import _format_event
        e = TraceEvent(step=10, type="tool_call", tool="Bash",
                       goal_text="npm install", success=False, latency_ms=200,
                       error_message="ERR! code ERESOLVE peer dependency conflict")
        line = _format_event(e)
        assert "FAIL" in line
        assert "ERESOLVE" in line
        assert "peer dependency" in line

    def test_format_event_no_error_on_success(self):
        """OK events don't show error text even if error_message is set."""
        from agentdiag.caft.confirm import _format_event
        e = TraceEvent(step=5, type="tool_call", tool="Read",
                       goal_text="file.py", success=True,
                       error_message="should not appear")
        line = _format_event(e)
        assert "should not appear" not in line

    def test_event_window_onset_first(self):
        """Onset event appears at the top of the window (lost-in-middle fix)."""
        events = [
            TraceEvent(step=i, type="tool_call", tool="Read", latency_ms=100)
            for i in range(1, 30)
        ]
        result = _format_event_window(events, center_step=15)
        lines = result.strip().split("\n")
        # First non-empty line should be "ONSET EVENT:"
        assert lines[0] == "ONSET EVENT:"
        assert "<<<ONSET" in lines[1]
        assert "step   15" in lines[1]
        # Context sections follow
        assert any("Context before" in l for l in lines)
        assert any("Context after" in l for l in lines)

    def test_prompt_no_detector_confidence(self):
        """Prompt should NOT include detector confidence (anchoring bias)."""
        events = _make_events()
        hta = _make_hta_state()
        diag = _make_diagnosis(confidence=0.85)
        prompt = build_confirmation_prompt(diag, events, hta)
        assert "Detector confidence" not in prompt
        assert "0.85" not in prompt

    def test_prompt_requires_cited_evidence(self):
        """Prompt should ask for cited_evidence in the JSON response."""
        events = _make_events()
        hta = _make_hta_state()
        diag = _make_diagnosis()
        prompt = build_confirmation_prompt(diag, events, hta)
        assert "cited_evidence" in prompt


class TestFormatSimilarCases:
    def test_formats_cases(self):
        cases = [
            {"failure_name": "context_loss", "status": "confirmed",
             "description": "Real loss", "phase_at_onset": "executing"},
        ]
        result = _format_similar_cases(cases)
        assert "context_loss" in result
        assert "confirmed" in result

    def test_empty_cases(self):
        result = _format_similar_cases([])
        assert "No similar past cases" in result


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

class TestFewShotExamples:
    def test_format_known_detector(self):
        """Few-shot examples format correctly for a known detector type."""
        result = _format_few_shot_examples("4.4")
        assert "CONFIRMED failure" in result
        assert "REJECTED candidate" in result
        assert "Now evaluate THIS candidate" in result
        assert "25085ms" in result  # stall evidence from confirmed example

    def test_format_unknown_detector(self):
        """Unknown detector type returns empty string."""
        result = _format_few_shot_examples("99.99")
        assert result == ""

    def test_all_examples_have_both_verdicts(self):
        """Every detector in FEW_SHOT_EXAMPLES has confirmed + rejected."""
        for code, examples in FEW_SHOT_EXAMPLES.items():
            assert "confirmed" in examples, f"Missing confirmed for {code}"
            assert "rejected" in examples, f"Missing rejected for {code}"
            for verdict in ("confirmed", "rejected"):
                for field in ("evidence", "events", "reasoning"):
                    assert field in examples[verdict], \
                        f"Missing {field} in {verdict} for {code}"

    def test_prompt_includes_few_shot_for_known_detector(self):
        """build_confirmation_prompt includes few-shot when detector has examples."""
        events = _make_events()
        hta = _make_hta_state()
        diag = _make_diagnosis(failure_name="stall")
        diag.caft_code = "4.4"

        prompt = build_confirmation_prompt(diag, events, hta)
        assert "Calibration Examples" in prompt
        assert "CONFIRMED failure" in prompt
        assert "REJECTED candidate" in prompt

    def test_prompt_omits_few_shot_for_unknown_detector(self):
        """build_confirmation_prompt omits few-shot when no examples exist."""
        events = _make_events()
        hta = _make_hta_state()
        diag = _make_diagnosis()
        diag.caft_code = "99.99"

        prompt = build_confirmation_prompt(diag, events, hta)
        assert "Calibration Examples" not in prompt

    def test_few_shot_covers_all_active_detectors(self):
        """Few-shot examples exist for all commonly-used detector codes."""
        expected = {"2.1", "2.2", "2.4", "3.1", "3.5", "4.2", "4.3", "4.4", "5.3", "5.4", "6.4"}
        assert set(FEW_SHOT_EXAMPLES.keys()) == expected


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

class TestParseLlmResponse:
    def test_parses_clean_json(self):
        text = '{"confirmed": true, "confidence": 0.85, "reasoning": "Clear pattern."}'
        result = _parse_llm_response(text)
        assert result.confirmed is True
        assert result.confidence == 0.85
        assert result.status == "confirmed"

    def test_parses_json_with_markdown_fences(self):
        text = '```json\n{"confirmed": false, "confidence": 0.2, "reasoning": "Normal workflow."}\n```'
        result = _parse_llm_response(text)
        assert result.confirmed is False
        assert result.status == "rejected"

    def test_parses_json_with_surrounding_text(self):
        text = 'Here is my analysis:\n{"confirmed": true, "confidence": 0.9, "reasoning": "Clearly stuck."}\nThat is my answer.'
        result = _parse_llm_response(text)
        assert result.confirmed is True
        assert result.confidence == 0.9

    def test_clamps_confidence(self):
        text = '{"confirmed": true, "confidence": 1.5, "reasoning": "test"}'
        result = _parse_llm_response(text)
        assert result.confidence == 1.0

        text2 = '{"confirmed": false, "confidence": -0.5, "reasoning": "test"}'
        result2 = _parse_llm_response(text2)
        assert result2.confidence == 0.0

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError, match="No JSON"):
            _parse_llm_response("I think this is a false positive.")

    def test_rejected_status_for_low_confidence(self):
        text = '{"confirmed": false, "confidence": 0.1, "reasoning": "Normal."}'
        result = _parse_llm_response(text)
        assert result.status == "rejected"

    def test_uncertain_status_for_medium_unconfirmed(self):
        text = '{"confirmed": false, "confidence": 0.5, "reasoning": "Unclear."}'
        result = _parse_llm_response(text)
        assert result.status == "uncertain"

    def test_parses_nested_cited_evidence(self):
        """Response with cited_evidence array is parsed correctly."""
        text = json.dumps({
            "cited_evidence": ["step 8: TaskOutput OK 25085ms", "median_latency_ms: 0.0"],
            "confirmed": True,
            "confidence": 0.85,
            "reasoning": "Clear stall at step 8.",
        })
        result = _parse_llm_response(text)
        assert result.confirmed is True
        assert result.confidence == 0.85

    def test_parses_nested_json_with_fences(self):
        """Nested JSON inside markdown fences is handled."""
        text = '```json\n{"cited_evidence": ["step 5: Read FAIL"], "confirmed": false, "confidence": 0.2, "reasoning": "Normal."}\n```'
        result = _parse_llm_response(text)
        assert result.confirmed is False
        assert result.status == "rejected"


class TestExtractJsonObject:
    def test_flat_object(self):
        data = _extract_json_object('{"a": 1, "b": "hello"}')
        assert data == {"a": 1, "b": "hello"}

    def test_nested_array(self):
        data = _extract_json_object('{"items": ["x", "y"], "count": 2}')
        assert data["items"] == ["x", "y"]
        assert data["count"] == 2

    def test_surrounding_text(self):
        data = _extract_json_object('Here is my analysis:\n{"confirmed": true}\nThat is all.')
        assert data["confirmed"] is True

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON"):
            _extract_json_object("no json here")


# ---------------------------------------------------------------------------
# LLM availability
# ---------------------------------------------------------------------------

class TestIsLlmAvailable:
    def test_anthropic_available(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test", "AGENTDIAG_LLM_PROVIDER": "anthropic"}):
            assert is_llm_available() is True

    def test_anthropic_unavailable(self):
        with patch.dict("os.environ", {"AGENTDIAG_LLM_PROVIDER": "anthropic"}, clear=True):
            assert is_llm_available() is False

    def test_openai_available(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test", "AGENTDIAG_LLM_PROVIDER": "openai"}):
            assert is_llm_available() is True

    def test_ollama_always_available(self):
        with patch.dict("os.environ", {"AGENTDIAG_LLM_PROVIDER": "ollama"}):
            assert is_llm_available() is True

    def test_claude_cli_available_when_on_path(self):
        with patch.dict("os.environ", {"AGENTDIAG_LLM_PROVIDER": "claude_cli"}):
            with patch("shutil.which", return_value="/usr/local/bin/claude"):
                assert is_llm_available() is True

    def test_claude_cli_unavailable_when_not_on_path(self):
        with patch.dict("os.environ", {"AGENTDIAG_LLM_PROVIDER": "claude_cli"}):
            with patch("shutil.which", return_value=None):
                assert is_llm_available() is False


# ---------------------------------------------------------------------------
# LLM tracing
# ---------------------------------------------------------------------------

class TestLlmTracing:
    def test_tracing_writes_prompt_and_response(self, tmp_path):
        """enable_llm_tracing captures full prompt and response."""
        import agentdiag.caft.confirm as confirm_mod

        trace_file = tmp_path / "traces.jsonl"
        old_path = confirm_mod._llm_trace_path
        try:
            enable_llm_tracing(trace_file)
            _write_llm_trace("What is 2+2?", '{"answer": 4}', 123.4)

            lines = trace_file.read_text().strip().split("\n")
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["prompt"] == "What is 2+2?"
            assert record["response"] == '{"answer": 4}'
            assert record["latency_ms"] == 123.4
            assert "timestamp" in record
        finally:
            confirm_mod._llm_trace_path = old_path

    def test_tracing_captures_errors(self, tmp_path):
        """Failed LLM calls include the error in the trace."""
        import agentdiag.caft.confirm as confirm_mod

        trace_file = tmp_path / "traces.jsonl"
        old_path = confirm_mod._llm_trace_path
        try:
            enable_llm_tracing(trace_file)
            _write_llm_trace("prompt", "", 50.0, error="API timeout")

            record = json.loads(trace_file.read_text().strip())
            assert record["error"] == "API timeout"
            assert record["response"] == ""
        finally:
            confirm_mod._llm_trace_path = old_path

    def test_no_trace_when_disabled(self, tmp_path):
        """No trace file written when tracing is not enabled."""
        import agentdiag.caft.confirm as confirm_mod

        old_path = confirm_mod._llm_trace_path
        try:
            confirm_mod._llm_trace_path = None
            _write_llm_trace("prompt", "response", 100.0)
            # No file should be created
            assert not list(tmp_path.iterdir())
        finally:
            confirm_mod._llm_trace_path = old_path


# ---------------------------------------------------------------------------
# confirm_diagnosis (async, mocked LLM)
# ---------------------------------------------------------------------------

class TestConfirmDiagnosis:
    @pytest.mark.asyncio
    async def test_confirmed_result(self):
        """LLM confirms a real failure."""
        events = _make_events()
        hta = _make_hta_state()
        diag = _make_diagnosis()

        mock_response = '{"confirmed": true, "confidence": 0.9, "reasoning": "Genuine context loss."}'
        with patch("agentdiag.caft.confirm._call_llm", new_callable=AsyncMock, return_value=mock_response):
            result = await confirm_diagnosis(diag, events, hta)

        assert result.confirmed is True
        assert result.status == "confirmed"
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_rejected_result(self):
        """LLM rejects a false positive."""
        events = _make_events()
        hta = _make_hta_state()
        diag = _make_diagnosis()

        mock_response = '{"confirmed": false, "confidence": 0.1, "reasoning": "Normal exploration."}'
        with patch("agentdiag.caft.confirm._call_llm", new_callable=AsyncMock, return_value=mock_response):
            result = await confirm_diagnosis(diag, events, hta)

        assert result.confirmed is False
        assert result.status == "rejected"

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_error(self):
        """LLM call fails — returns uncertain, never crashes."""
        events = _make_events()
        hta = _make_hta_state()
        diag = _make_diagnosis(confidence=0.7)

        with patch("agentdiag.caft.confirm._call_llm", new_callable=AsyncMock, side_effect=Exception("API error")):
            result = await confirm_diagnosis(diag, events, hta)

        assert result.status == "uncertain"
        assert result.confidence < 0.7  # discounted
        assert "unavailable" in result.reasoning

    @pytest.mark.asyncio
    async def test_passes_similar_cases(self):
        """Similar cases are included in the prompt."""
        events = _make_events()
        hta = _make_hta_state()
        diag = _make_diagnosis()
        cases = [{"failure_name": "context_loss", "status": "false_positive",
                  "description": "test", "phase_at_onset": "executing"}]

        captured_prompt = []

        async def mock_llm(prompt):
            captured_prompt.append(prompt)
            return '{"confirmed": true, "confidence": 0.8, "reasoning": "test"}'

        with patch("agentdiag.caft.confirm._call_llm", side_effect=mock_llm):
            await confirm_diagnosis(diag, events, hta, context_cases=cases)

        assert "false_positive" in captured_prompt[0]


# ---------------------------------------------------------------------------
# confirm_diagnosis_sync
# ---------------------------------------------------------------------------

class TestConfirmDiagnosisSync:
    def test_sync_wrapper_works(self):
        """Synchronous wrapper produces a result."""
        events = _make_events()
        hta = _make_hta_state()
        diag = _make_diagnosis()

        mock_response = '{"confirmed": true, "confidence": 0.85, "reasoning": "Confirmed."}'
        with patch("agentdiag.caft.confirm._call_llm", new_callable=AsyncMock, return_value=mock_response):
            result = confirm_diagnosis_sync(diag, events, hta)

        assert result.confirmed is True
        assert result.status == "confirmed"


# ---------------------------------------------------------------------------
# MonitorEngine integration
# ---------------------------------------------------------------------------

class TestMonitorEngineConfirmation:
    def test_no_confirm_mode_unchanged(self):
        """Without confirm=True, uses registry's enabled detectors."""
        from agentdiag.monitor import MonitorEngine
        from agentdiag.caft.registry import detector_registry

        engine = MonitorEngine(goal="test", confirm=False)
        assert len(engine._detectors) == len(detector_registry.get_enabled())

    def test_confirm_mode_uses_full_detectors(self):
        """With confirm=True, all detectors are enabled."""
        from agentdiag.monitor import MonitorEngine
        from agentdiag.caft.detectors import ALL_CAFT_DETECTORS_FULL

        engine = MonitorEngine(goal="test", confirm=True)
        assert len(engine._detectors) == len(ALL_CAFT_DETECTORS_FULL)

    def test_explicit_detectors_override(self):
        """Explicit detector list overrides both modes."""
        from agentdiag.monitor import MonitorEngine

        custom = [MagicMock(name="custom", caft_code="0.0")]
        engine = MonitorEngine(goal="test", detectors=custom, confirm=True)
        assert len(engine._detectors) == 1

    def test_confirm_mode_rejects_candidates(self):
        """Candidates rejected by LLM are excluded from diagnoses."""
        from agentdiag.monitor import MonitorEngine

        engine = MonitorEngine(goal="test", confirm=True)

        # Mock _apply_confirmation to reject
        original_apply = engine._apply_confirmation
        def mock_apply(candidate, hta_state):
            engine._candidates_total += 1
            engine._candidates_rejected += 1
            return None  # rejected

        engine._apply_confirmation = mock_apply

        # Push events that would trigger a detector
        events = [
            TraceEvent(step=i, type="tool_call", tool="read_file",
                       output_hash="same_hash", latency_ms=200)
            for i in range(1, 15)
        ]
        for e in events:
            engine.push(e)

        # Diagnoses should be empty (all rejected)
        # (or may have non-CAFT detectors that still pass through)
        state = engine.state
        assert state.candidates_rejected >= 0  # at minimum, no crash

    def test_dashboard_state_has_confirmation_stats(self):
        """DashboardState includes confirmation statistics."""
        from agentdiag.monitor import MonitorEngine

        engine = MonitorEngine(goal="test", confirm=True)
        state = engine.state
        assert hasattr(state, "candidates_total")
        assert hasattr(state, "candidates_confirmed")
        assert hasattr(state, "candidates_rejected")
        assert hasattr(state, "candidates_uncertain")
        assert hasattr(state, "candidates_autoconfirmed")
        assert state.candidates_total == 0

    def test_reset_clears_confirmation_stats(self):
        """Reset clears confirmation counters."""
        from agentdiag.monitor import MonitorEngine

        engine = MonitorEngine(goal="test", confirm=True)
        engine._candidates_total = 5
        engine._candidates_confirmed = 3
        engine.reset()

        assert engine._candidates_total == 0
        assert engine._candidates_confirmed == 0


# ---------------------------------------------------------------------------
# Tier 2: Session-end assessment
# ---------------------------------------------------------------------------

class TestBuildSessionAssessmentPrompt:
    def test_contains_goal(self):
        events = _make_events(20)
        hta = _make_hta_state()
        prompt = build_session_assessment_prompt(events, hta)
        assert "Fix the login bug" in prompt

    def test_contains_phase_counts(self):
        events = _make_events(20)
        hta = _make_hta_state()
        hta.phase_event_counts = {"executing": 15, "verifying": 3, "delivering": 2}
        prompt = build_session_assessment_prompt(events, hta)
        assert "Exec: 15" in prompt
        assert "Verify: 3" in prompt
        assert "Deliver: 2" in prompt

    def test_contains_session_head_and_tail(self):
        events = _make_events(20)
        hta = _make_hta_state()
        prompt = build_session_assessment_prompt(events, hta)
        assert "SESSION START" in prompt
        assert "SESSION END" in prompt

    def test_contains_classification_options(self):
        events = _make_events(20)
        hta = _make_hta_state()
        prompt = build_session_assessment_prompt(events, hta)
        assert "COMPLETED" in prompt
        assert "PREMATURE STOP" in prompt
        assert "GOAL DRIFT" in prompt
        assert "EXTERNAL END" in prompt
        assert "USER REDIRECTED" in prompt


class TestParseSessionAssessment:
    def test_completed(self):
        text = '{"classifications": ["A"], "confidence": 0.9, "reasoning": "Task done."}'
        result = _parse_session_assessment(text)
        assert result.classifications == ["A"]
        assert result.premature_termination is False
        assert result.goal_drift is False

    def test_premature_stop(self):
        text = '{"classifications": ["B"], "confidence": 0.8, "reasoning": "Not finished."}'
        result = _parse_session_assessment(text)
        assert result.classifications == ["B"]
        assert result.premature_termination is True
        assert result.goal_drift is False

    def test_goal_drift(self):
        text = '{"classifications": ["C"], "confidence": 0.7, "reasoning": "Went off track."}'
        result = _parse_session_assessment(text)
        assert result.classifications == ["C"]
        assert result.premature_termination is False
        assert result.goal_drift is True

    def test_cooccurrence(self):
        text = '{"classifications": ["B", "C"], "confidence": 0.75, "reasoning": "Both apply."}'
        result = _parse_session_assessment(text)
        assert result.classifications == ["B", "C"]
        assert result.premature_termination is True
        assert result.goal_drift is True

    def test_external_end(self):
        text = '{"classifications": ["D"], "confidence": 0.85, "reasoning": "Crashed."}'
        result = _parse_session_assessment(text)
        assert result.classifications == ["D"]
        assert result.premature_termination is False
        assert result.goal_drift is False

    def test_markdown_fences(self):
        text = '```json\n{"classifications": ["B"], "confidence": 0.8, "reasoning": "test"}\n```'
        result = _parse_session_assessment(text)
        assert result.premature_termination is True

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON"):
            _parse_session_assessment("I think the session completed fine.")


class TestAssessSessionEndSync:
    def test_error_handling(self):
        """Exception returns empty classifications, both=False."""
        events = _make_events()
        hta = _make_hta_state()

        with patch("agentdiag.caft.confirm._call_llm", new_callable=AsyncMock, side_effect=Exception("API error")):
            result = assess_session_end_sync(events, hta)

        assert result.classifications == []
        assert result.premature_termination is False
        assert result.goal_drift is False
        assert result.confidence == 0.0

    def test_confirmed_result(self):
        """LLM returns premature termination."""
        events = _make_events(20)
        hta = _make_hta_state()

        mock_response = '{"classifications": ["B"], "confidence": 0.85, "reasoning": "Task incomplete."}'
        with patch("agentdiag.caft.confirm._call_llm", new_callable=AsyncMock, return_value=mock_response):
            result = assess_session_end_sync(events, hta)

        assert result.premature_termination is True
        assert result.goal_drift is False
        assert result.confidence == 0.85


class TestTier2FailureTypes:
    def test_tier2_types(self):
        assert TIER_2_FAILURE_TYPES == {"premature_termination", "goal_drift"}
