"""Tests for the data split manager."""

import json
import warnings

import pytest

from agentdiag.splits import SplitManager, SplitSummary, VALID_SPLITS


class TestSplitManager:
    def test_assign_and_get(self, tmp_path):
        sm = SplitManager(tmp_path / "splits.json")
        sm.assign("t1", "development", source="synthetic")
        assert sm.get_split("t1") == "development"

    def test_get_unassigned_returns_none(self, tmp_path):
        sm = SplitManager(tmp_path / "splits.json")
        assert sm.get_split("nonexistent") is None

    def test_invalid_split_raises(self, tmp_path):
        sm = SplitManager(tmp_path / "splits.json")
        with pytest.raises(ValueError, match="Invalid split"):
            sm.assign("t1", "invalid_split")

    def test_get_traces(self, tmp_path):
        sm = SplitManager(tmp_path / "splits.json")
        sm.assign("t1", "development", source="synthetic")
        sm.assign("t2", "validation", source="claude-code")
        sm.assign("t3", "test", source="claude-code")
        sm.assign("t4", "development", source="synthetic")

        assert set(sm.get_traces("development")) == {"t1", "t4"}
        assert sm.get_traces("validation") == ["t2"]
        assert sm.get_traces("test") == ["t3"]

    def test_persistence(self, tmp_path):
        path = tmp_path / "splits.json"
        sm1 = SplitManager(path)
        sm1.assign("t1", "development", source="synthetic")
        sm1.assign("t2", "test", source="claude-code")

        sm2 = SplitManager(path)
        assert sm2.get_split("t1") == "development"
        assert sm2.get_split("t2") == "test"

    def test_reassign_warns(self, tmp_path):
        sm = SplitManager(tmp_path / "splits.json")
        sm.assign("t1", "development")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sm.assign("t1", "test")
            assert len(w) == 1
            assert "leakage" in str(w[0].message).lower()

    def test_lock_prevents_reassign(self, tmp_path):
        sm = SplitManager(tmp_path / "splits.json")
        sm.assign("t1", "test")
        sm.lock("t1")
        with pytest.raises(ValueError, match="locked"):
            sm.assign("t1", "development")

    def test_lock_unassigned_raises(self, tmp_path):
        sm = SplitManager(tmp_path / "splits.json")
        with pytest.raises(ValueError, match="not assigned"):
            sm.lock("nonexistent")


class TestLeakageDetection:
    def test_no_leak_for_correct_use(self, tmp_path):
        sm = SplitManager(tmp_path / "splits.json")
        sm.assign("t1", "test")
        assert sm.check_leakage("t1", "test") is False

    def test_leak_when_dev_used_for_test(self, tmp_path):
        sm = SplitManager(tmp_path / "splits.json")
        sm.assign("t1", "development")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            is_leak = sm.check_leakage("t1", "test")
            assert is_leak is True
            assert len(w) == 1

    def test_leak_when_test_used_for_tuning(self, tmp_path):
        sm = SplitManager(tmp_path / "splits.json")
        sm.assign("t1", "test")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            is_leak = sm.check_leakage("t1", "threshold_tuning")
            assert is_leak is True

    def test_no_leak_for_unassigned(self, tmp_path):
        sm = SplitManager(tmp_path / "splits.json")
        assert sm.check_leakage("unknown", "test") is False


class TestSplitSummary:
    def test_summary_counts(self, tmp_path):
        sm = SplitManager(tmp_path / "splits.json")
        sm.assign("t1", "development", source="synthetic")
        sm.assign("t2", "development", source="synthetic")
        sm.assign("t3", "validation", source="claude-code")
        sm.assign("t4", "test", source="claude-code")
        sm.assign("t5", "test", source="claude-code")
        sm.assign("t6", "test", source="claude-code")

        s = sm.summary()
        assert s.development == 2
        assert s.validation == 1
        assert s.test == 3

    def test_summary_by_source(self, tmp_path):
        sm = SplitManager(tmp_path / "splits.json")
        sm.assign("t1", "development", source="synthetic")
        sm.assign("t2", "validation", source="claude-code")

        s = sm.summary()
        assert "synthetic" in s.by_source
        assert s.by_source["synthetic"]["development"] == 1
        assert "claude-code" in s.by_source

    def test_summary_str(self, tmp_path):
        sm = SplitManager(tmp_path / "splits.json")
        sm.assign("t1", "development", source="synthetic")
        s = sm.summary()
        text = str(s)
        assert "Development" in text
        assert "synthetic" in text
