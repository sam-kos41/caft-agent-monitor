"""Tests for IP stage mapping on CAFT taxonomy entries."""

import pytest

from agentdiag.caft.taxonomy import (
    CAFT_TAXONOMY,
    get_ip_stage_for_detector,
    get_detectors_by_ip_stage,
    get_ip_stage_map,
)


VALID_IP_STAGES = {
    "perception", "attention", "working_memory",
    "decision_making", "action", "feedback",
}


class TestIPStageMapping:
    def test_all_entries_have_ip_stage(self):
        """Every taxonomy entry must have a non-empty ip_stage."""
        for code, t in CAFT_TAXONOMY.items():
            assert t.ip_stage, f"CAFT {code} ({t.name}) has no ip_stage"

    def test_all_ip_stages_valid(self):
        """Every ip_stage must be one of the 6 valid stages."""
        for code, t in CAFT_TAXONOMY.items():
            assert t.ip_stage in VALID_IP_STAGES, (
                f"CAFT {code} ({t.name}) has invalid ip_stage '{t.ip_stage}'"
            )

    def test_all_6_stages_covered(self):
        """At least one CAFT type maps to each IP stage."""
        covered = {t.ip_stage for t in CAFT_TAXONOMY.values()}
        assert covered == VALID_IP_STAGES

    def test_perception_failures_map_to_perception(self):
        """Category 1 failures should map to 'perception'."""
        for code in ["1.1", "1.2", "1.3", "1.4"]:
            assert CAFT_TAXONOMY[code].ip_stage == "perception"

    def test_context_loss_maps_to_attention(self):
        """2.1 context_loss is an attention failure."""
        assert CAFT_TAXONOMY["2.1"].ip_stage == "attention"

    def test_step_repetition_maps_to_working_memory(self):
        """2.2 step_repetition is a working memory failure."""
        assert CAFT_TAXONOMY["2.2"].ip_stage == "working_memory"

    def test_goal_drift_maps_to_decision_making(self):
        """2.4 goal_drift is a decision-making failure."""
        assert CAFT_TAXONOMY["2.4"].ip_stage == "decision_making"

    def test_error_cascade_maps_to_feedback(self):
        """4.2 error_cascade is a feedback failure."""
        assert CAFT_TAXONOMY["4.2"].ip_stage == "feedback"

    def test_tool_misuse_maps_to_action(self):
        """4.1 tool_misuse is an action failure."""
        assert CAFT_TAXONOMY["4.1"].ip_stage == "action"

    def test_resource_exhaustion_maps_to_working_memory(self):
        """4.4 resource_exhaustion is a working memory failure."""
        assert CAFT_TAXONOMY["4.4"].ip_stage == "working_memory"

    def test_strategic_myopia_maps_to_decision_making(self):
        """3.5 strategic_myopia is a decision-making failure."""
        assert CAFT_TAXONOMY["3.5"].ip_stage == "decision_making"

    def test_communication_failures_map_to_feedback(self):
        """Category 7 failures should map to 'feedback'."""
        for code in ["7.1", "7.2", "7.3", "7.4"]:
            assert CAFT_TAXONOMY[code].ip_stage == "feedback"

    def test_metacognition_failures_map_to_decision_making(self):
        """Category 8 failures should map to 'decision_making'."""
        for code in ["8.1", "8.2", "8.3", "8.4"]:
            assert CAFT_TAXONOMY[code].ip_stage == "decision_making"


class TestIPStageHelpers:
    def test_get_ip_stage_for_known_detector(self):
        assert get_ip_stage_for_detector("context_loss") == "attention"
        assert get_ip_stage_for_detector("stall") is not None

    def test_get_ip_stage_for_unknown_detector(self):
        assert get_ip_stage_for_detector("nonexistent") == "unknown"

    def test_get_detectors_by_ip_stage(self):
        perception = get_detectors_by_ip_stage("perception")
        assert len(perception) >= 4  # at least the 4 perception failures
        assert all(t.ip_stage == "perception" for t in perception)

    def test_get_detectors_by_ip_stage_empty(self):
        assert get_detectors_by_ip_stage("nonexistent") == []

    def test_get_ip_stage_map_complete(self):
        stage_map = get_ip_stage_map()
        assert len(stage_map) == len(CAFT_TAXONOMY)
        for code, stage in stage_map.items():
            assert stage in VALID_IP_STAGES
