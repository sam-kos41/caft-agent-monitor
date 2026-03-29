"""Tests for typed evidence schema."""

import time
import pytest

from agentdiag.evidence import (
    LoopEvidence,
    ThrashEvidence,
    StallEvidence,
    DriftEvidence,
    CascadeEvidence,
    TokenExplosionEvidence,
    DeadEndEvidence,
    RecoveryFailureEvidence,
    parse_evidence,
    EVIDENCE_TYPES,
)
from agentdiag.hta import HTAState, HTANode, Phase
from agentdiag.caft.detectors import (
    StallDetector,
    ErrorCascadeDetector,
    TokenExplosionDetector,
    AnalysisParalysisDetector,
    RecoveryFailureDetector,
)
from agentdiag.synthetic import (
    generate_stall_trace,
    generate_cascade_trace,
    generate_token_explosion_trace,
    generate_dead_end_trace,
    generate_recovery_failure_trace,
)


def _default_hta() -> HTAState:
    return HTAState(
        goal="test",
        current_phase=Phase.EXECUTING,
        current_node=HTANode(
            phase=Phase.EXECUTING,
            start_step=0,
            start_time=time.monotonic(),
        ),
        completed_nodes=[],
        transitions=[],
        total_events=0,
        phase_event_counts={"executing": 10},
    )


class TestEvidenceConstruction:
    def test_loop_evidence(self):
        ev = LoopEvidence(
            repeated_pattern=["search", "search"],
            pattern_count=5,
            pattern_length=2,
            total_tool_calls=15,
            state_changes_during_loop=0,
        )
        d = ev.to_dict()
        assert d["pattern_count"] == 5
        assert isinstance(d["repeated_pattern"], list)

    def test_stall_evidence(self):
        ev = StallEvidence(
            stall_steps=[7, 13],
            max_latency_ms=15000.0,
            median_latency_ms=600.0,
            threshold_ms=1500.0,
            stall_count=2,
            worst_step=7,
        )
        d = ev.to_dict()
        assert d["worst_step"] == 7
        assert d["stall_count"] == 2

    def test_all_evidence_types_constructable(self):
        """Ensure every evidence type can be constructed with sample data."""
        samples = {
            "LOOP": {"repeated_pattern": ["a"], "pattern_count": 3,
                     "pattern_length": 1, "total_tool_calls": 10,
                     "state_changes_during_loop": 0},
            "TOOL_THRASH": {"worst_window_tools": ["a", "b"], "unique_tools_in_window": ["a", "b"],
                            "switch_rate": 0.8, "window_start_step": 5},
            "STALL": {"stall_steps": [1], "max_latency_ms": 5000.0,
                      "median_latency_ms": 500.0, "threshold_ms": 1500.0,
                      "stall_count": 1, "worst_step": 1},
            "DRIFT": {"tool_distribution_shift": 0.3, "error_rate_first_half": 0.0,
                      "error_rate_second_half": 0.4, "latency_ratio_2nd_vs_1st": 2.5,
                      "new_tools_in_second_half": ["x"], "dropped_tools_in_second_half": ["y"]},
            "CASCADE": {"longest_error_chain": 5, "chain_start_step": 6,
                        "chain_end_step": 10, "total_error_chains": 1,
                        "total_errors": 5, "tools_in_chain": ["run_code"]},
            "TOKEN_EXPLOSION": {"growth_ratio_last_vs_first_quarter": 5.0,
                                "tokens_per_step_slope": 200.0, "acceleration": 15.0,
                                "first_quarter_avg_tokens": 500.0,
                                "last_quarter_avg_tokens": 2500.0, "total_tokens": 30000},
            "DEAD_END": {"max_consecutive_reasoning": 8, "dead_end_start_step": 6,
                         "dead_end_end_step": 13, "total_reasoning_events": 8,
                         "total_events": 20},
            "RECOVERY_FAILURE": {"total_errors": 6, "failed_recoveries": 4,
                                 "recovery_failure_rate": 0.67, "same_tool_retries": 3,
                                 "worst_error_step": 5, "worst_consecutive_failures_after": 2},
        }
        for failure_type, data in samples.items():
            ev = parse_evidence(failure_type, data)
            roundtrip = ev.to_dict()
            for key in data:
                assert roundtrip[key] == data[key]


class TestEvidenceRoundTrip:
    def test_parse_evidence_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown failure type"):
            parse_evidence("NONEXISTENT", {})

    def test_caft_detector_evidence_compatible(self):
        """CAFT detector evidence dicts are compatible with typed evidence."""
        hta = _default_hta()

        # STALL → same evidence keys
        events = generate_stall_trace()
        diag = StallDetector().check(events, hta)
        assert diag is not None
        ev = parse_evidence("STALL", diag.evidence)
        assert ev.to_dict() == diag.evidence

        # CASCADE → same evidence keys
        events = generate_cascade_trace()
        diag = ErrorCascadeDetector().check(events, hta)
        assert diag is not None
        ev = parse_evidence("CASCADE", diag.evidence)
        assert ev.to_dict() == diag.evidence

        # TOKEN_EXPLOSION → same evidence keys
        events = generate_token_explosion_trace()
        diag = TokenExplosionDetector().check(events, hta)
        assert diag is not None
        ev = parse_evidence("TOKEN_EXPLOSION", diag.evidence)
        assert ev.to_dict() == diag.evidence

        # DEAD_END → same evidence keys
        events = generate_dead_end_trace()
        diag = AnalysisParalysisDetector().check(events, hta)
        assert diag is not None
        ev = parse_evidence("DEAD_END", diag.evidence)
        assert ev.to_dict() == diag.evidence

        # RECOVERY_FAILURE → same evidence keys
        events = generate_recovery_failure_trace()
        diag = RecoveryFailureDetector().check(events, hta)
        assert diag is not None
        ev = parse_evidence("RECOVERY_FAILURE", diag.evidence)
        assert ev.to_dict() == diag.evidence
