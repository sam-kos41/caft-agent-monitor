"""Tests for QueueStream and generate_demo_jsonl."""

import json
import threading
import time

import pytest

from agentdiag.visualize import QueueStream
from agentdiag.caft.demo import (
    generate_demo_jsonl,
    _event_to_jsonl,
    SHOWCASE_SEQUENCE,
    _scenario_description,
)
from agentdiag.caft.synthetic import CAFT_GENERATORS


class TestQueueStream:
    def test_write_and_iterate(self):
        qs = QueueStream()
        qs.write("line1\n")
        qs.write("line2\n")
        qs.close()

        lines = list(qs)
        assert lines == ["line1\n", "line2\n"]

    def test_empty_stream(self):
        qs = QueueStream()
        qs.close()
        assert list(qs) == []

    def test_threaded_producer_consumer(self):
        qs = QueueStream()
        results = []

        def consumer():
            for line in qs:
                results.append(line.strip())

        t = threading.Thread(target=consumer)
        t.start()

        qs.write("a\n")
        qs.write("b\n")
        qs.write("c\n")
        qs.close()
        t.join(timeout=2)

        assert results == ["a", "b", "c"]

    def test_next_raises_stop_iteration(self):
        qs = QueueStream()
        qs.close()
        with pytest.raises(StopIteration):
            next(qs)

    def test_iter_returns_self(self):
        qs = QueueStream()
        assert iter(qs) is qs


class TestGenerateDemoJsonl:
    def test_single_scenario_clean(self):
        qs = QueueStream()
        t = threading.Thread(
            target=generate_demo_jsonl,
            args=("clean", 0.0, qs),
        )
        t.start()

        lines = []
        for line in qs:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
        t.join(timeout=5)

        assert len(lines) > 0
        assert all("step" in ev for ev in lines)

    def test_single_scenario_step_repetition(self):
        qs = QueueStream()
        t = threading.Thread(
            target=generate_demo_jsonl,
            args=("step_repetition", 0.0, qs),
        )
        t.start()

        lines = []
        for line in qs:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
        t.join(timeout=5)

        assert len(lines) > 0
        # Steps should be renumbered globally
        steps = [ev["step"] for ev in lines]
        assert steps == list(range(1, len(lines) + 1))

    def test_all_scenarios(self):
        qs = QueueStream()
        t = threading.Thread(
            target=generate_demo_jsonl,
            args=("all", 0.0, qs),
        )
        t.start()

        count = 0
        for line in qs:
            if line.strip():
                count += 1
        t.join(timeout=10)

        # Should have events from all scenarios
        assert count > 20

    def test_showcase_emits_boundaries(self):
        qs = QueueStream()
        t = threading.Thread(
            target=generate_demo_jsonl,
            args=("showcase", 0.0, qs),
        )
        t.start()

        boundaries = []
        events = []
        for line in qs:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("type") == "scenario_boundary":
                boundaries.append(data)
            else:
                events.append(data)
        t.join(timeout=15)

        # Should have one boundary per showcase scenario
        assert len(boundaries) == len(SHOWCASE_SEQUENCE)
        # Each boundary has scenario and description
        for b in boundaries:
            assert "scenario" in b
            assert "description" in b
        # Should also have actual events
        assert len(events) > 0


class TestShowcaseSequence:
    def test_showcase_has_entries(self):
        assert len(SHOWCASE_SEQUENCE) >= 3

    def test_showcase_entries_are_valid(self):
        valid = set(CAFT_GENERATORS.keys()) | {"e2e"}
        for name, pause in SHOWCASE_SEQUENCE:
            assert name in valid, f"Unknown scenario: {name}"
            assert isinstance(pause, (int, float))
            assert pause >= 0


class TestScenarioDescription:
    def test_known_scenarios_have_descriptions(self):
        for name, _ in SHOWCASE_SEQUENCE:
            desc = _scenario_description(name)
            assert len(desc) > 0
            assert isinstance(desc, str)


class TestEventToJsonl:
    def test_includes_agent_id(self):
        from agentdiag.models import TraceEvent
        event = TraceEvent(step=1, type="tool_call", tool="read_file", agent_id="S1")
        line = _event_to_jsonl(event)
        data = json.loads(line)
        assert data["agent_id"] == "S1"

    def test_omits_none_agent_id(self):
        from agentdiag.models import TraceEvent
        event = TraceEvent(step=1, type="tool_call", tool="read_file")
        line = _event_to_jsonl(event)
        data = json.loads(line)
        assert "agent_id" not in data
