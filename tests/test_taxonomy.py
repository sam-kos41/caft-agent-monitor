"""Tests for the CAFT taxonomy and unified detector mapping."""

import pytest

from agentdiag.caft.taxonomy import (
    CAFT_TAXONOMY,
    CaftType,
    Detectability,
    BATCH_DETECTOR_TO_CAFT,
    FAILURE_TYPE_TO_CAFT,
    get_type,
    get_type_by_name,
    get_observable_types,
    get_latent_types,
    get_categories,
    get_category_types,
    map_batch_diagnosis,
)


class TestTaxonomyCompleteness:
    def test_has_33_types(self):
        assert len(CAFT_TAXONOMY) == 33

    def test_has_8_categories(self):
        cats = get_categories()
        assert len(cats) == 8

    def test_all_codes_unique(self):
        codes = list(CAFT_TAXONOMY.keys())
        assert len(codes) == len(set(codes))

    def test_all_names_unique(self):
        names = [t.name for t in CAFT_TAXONOMY.values()]
        assert len(names) == len(set(names))

    def test_observable_count(self):
        obs = get_observable_types()
        assert len(obs) == 13

    def test_latent_count(self):
        lat = get_latent_types()
        assert len(lat) == 20

    def test_observable_plus_latent_is_33(self):
        obs = get_observable_types()
        lat = get_latent_types()
        assert len(obs) + len(lat) == 33


class TestTaxonomyQueries:
    def test_get_type_by_code(self):
        t = get_type("2.2")
        assert t.name == "step_repetition"
        assert t.category == "memory"

    def test_get_type_unknown_raises(self):
        with pytest.raises(KeyError):
            get_type("99.99")

    def test_get_type_by_name(self):
        t = get_type_by_name("goal_drift")
        assert t is not None
        assert t.code == "2.4"

    def test_get_type_by_name_unknown(self):
        assert get_type_by_name("nonexistent") is None

    def test_get_category_types(self):
        memory = get_category_types("memory")
        assert len(memory) == 4
        assert all(t.category == "memory" for t in memory)

    def test_categories_ordered(self):
        cats = get_categories()
        assert cats[0] == "perception"
        assert cats[-1] == "metacognition"


class TestDetectorMapping:
    def test_all_8_batch_detectors_mapped(self):
        assert len(BATCH_DETECTOR_TO_CAFT) == 8
        expected = {"loop", "thrash", "stall", "drift", "cascade",
                    "token_explosion", "dead_end", "recovery_failure"}
        assert set(BATCH_DETECTOR_TO_CAFT.keys()) == expected

    def test_all_failure_types_mapped(self):
        assert len(FAILURE_TYPE_TO_CAFT) == 8
        expected = {"LOOP", "TOOL_THRASH", "STALL", "DRIFT", "CASCADE",
                    "TOKEN_EXPLOSION", "DEAD_END", "RECOVERY_FAILURE"}
        assert set(FAILURE_TYPE_TO_CAFT.keys()) == expected

    def test_mapped_codes_exist_in_taxonomy(self):
        for code in BATCH_DETECTOR_TO_CAFT.values():
            assert code in CAFT_TAXONOMY, f"Code {code} not in taxonomy"

    def test_map_batch_diagnosis(self):
        assert map_batch_diagnosis("LOOP") == "2.2"
        assert map_batch_diagnosis("CASCADE") == "4.2"
        assert map_batch_diagnosis("UNKNOWN") is None

    def test_observable_types_have_detectors(self):
        """Every observable type should be covered by at least one detector."""
        for t in get_observable_types():
            assert t.detector_names, (
                f"Observable type {t.code} ({t.name}) has no detector_names"
            )

    def test_latent_types_have_no_detectors(self):
        """Latent types should NOT claim detector coverage."""
        for t in get_latent_types():
            assert not t.detector_names, (
                f"Latent type {t.code} ({t.name}) should not have detectors: "
                f"{t.detector_names}"
            )


class TestCaftTypeFields:
    def test_type_is_frozen(self):
        t = get_type("2.2")
        with pytest.raises(AttributeError):
            t.code = "99.9"

    def test_all_types_have_description(self):
        for t in CAFT_TAXONOMY.values():
            assert t.description, f"{t.code} missing description"
            assert len(t.description) > 10, f"{t.code} description too short"
