"""Tests for the explanation engine."""

import pytest

from agentdiag.explain import explain, ExplanationResult
from agentdiag.explain.llm import LLMExplainer
from agentdiag.models import Diagnosis


# Sample evidence dicts for each failure type
_SAMPLE_EVIDENCE = {
    "LOOP": {
        "repeated_pattern": ["search_docs", "search_docs"],
        "pattern_count": 5,
        "pattern_length": 2,
        "total_tool_calls": 15,
        "state_changes_during_loop": 0,
    },
    "TOOL_THRASH": {
        "worst_window_tools": ["search", "read", "write", "search", "read"],
        "unique_tools_in_window": ["search", "read", "write"],
        "switch_rate": 0.8,
        "window_start_step": 5,
    },
    "STALL": {
        "stall_steps": [7, 13],
        "max_latency_ms": 15000.0,
        "median_latency_ms": 600.0,
        "threshold_ms": 1500.0,
        "stall_count": 2,
        "worst_step": 7,
    },
    "DRIFT": {
        "tool_distribution_shift": 0.35,
        "error_rate_first_half": 0.05,
        "error_rate_second_half": 0.4,
        "latency_ratio_2nd_vs_1st": 2.5,
        "new_tools_in_second_half": ["debug"],
        "dropped_tools_in_second_half": ["search"],
    },
    "CASCADE": {
        "longest_error_chain": 5,
        "chain_start_step": 6,
        "chain_end_step": 10,
        "total_error_chains": 1,
        "total_errors": 5,
        "tools_in_chain": ["run_code"],
    },
    "TOKEN_EXPLOSION": {
        "growth_ratio_last_vs_first_quarter": 5.0,
        "tokens_per_step_slope": 200.0,
        "acceleration": 15.0,
        "first_quarter_avg_tokens": 500.0,
        "last_quarter_avg_tokens": 2500.0,
        "total_tokens": 30000,
    },
    "DEAD_END": {
        "max_consecutive_reasoning": 8,
        "dead_end_start_step": 6,
        "dead_end_end_step": 13,
        "total_reasoning_events": 8,
        "total_events": 20,
    },
    "RECOVERY_FAILURE": {
        "total_errors": 6,
        "failed_recoveries": 4,
        "recovery_failure_rate": 0.67,
        "same_tool_retries": 3,
        "worst_error_step": 5,
        "worst_consecutive_failures_after": 2,
    },
}


def _make_diag(failure_type: str) -> Diagnosis:
    return Diagnosis(
        failure_type=failure_type,
        confidence=0.8,
        evidence=_SAMPLE_EVIDENCE[failure_type],
        explanation=f"Test {failure_type} diagnosis.",
    )


class TestTemplateExplanations:
    def test_loop_explanation(self):
        diag = _make_diag("LOOP")
        result = explain(diag)
        assert isinstance(result, ExplanationResult)
        assert result.failure_type == "LOOP"
        assert len(result.remediation) > 0
        assert "pattern" in result.description.lower() or "repeated" in result.description.lower()

    def test_all_failure_types_have_templates(self):
        for failure_type in _SAMPLE_EVIDENCE:
            diag = _make_diag(failure_type)
            result = explain(diag)
            assert result.failure_type == failure_type
            assert result.likely_cause != "Unknown failure type — no template available."
            assert len(result.remediation) >= 2

    def test_unknown_failure_type_fallback(self):
        diag = Diagnosis(
            failure_type="UNKNOWN_TYPE",
            confidence=0.5,
            evidence={"foo": "bar"},
            explanation="Something unknown happened.",
        )
        result = explain(diag)
        assert result.failure_type == "UNKNOWN_TYPE"
        assert "Unknown" in result.likely_cause

    def test_explanation_to_dict(self):
        diag = _make_diag("STALL")
        result = explain(diag)
        d = result.to_dict()
        assert "failure_type" in d
        assert "remediation" in d
        assert isinstance(d["remediation"], list)


class TestLLMExplainer:
    def test_llm_explainer_raises(self):
        explainer = LLMExplainer()
        diag = Diagnosis(
            failure_type="LOOP",
            confidence=0.9,
            evidence={},
            explanation="test",
        )
        with pytest.raises(NotImplementedError, match="agentdiag-pro"):
            explainer.explain(diag, [])
