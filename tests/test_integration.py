"""End-to-end integration tests — verify full pipeline works as a unit.

These tests exercise the connections between subsystems built by four
different agents, ensuring they compose into a working whole.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from agentdiag.observable import (
    ObservableEvent,
    EventType,
    HarnessPhase,
    AgentRole,
    MemoryTier,
    file_read_event,
    file_write_event,
    tool_call_event,
    phase_boundary_event,
    evaluation_event,
    memory_load_event,
    memory_store_event,
    tier_escalation_event,
)


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Demo mode end-to-end
# ═══════════════════════════════════════════════════════════════════════════

class TestDemoEndToEnd:
    """Generate synthetic events via demo, feed through UniversalMonitor,
    verify the full state payload is populated."""

    def test_demo_generates_events(self):
        """Demo event generator produces a non-trivial stream."""
        from agentdiag.demo import _generate_events
        events = _generate_events()
        assert len(events) >= 200, f"Expected ≥200 events, got {len(events)}"
        # Should contain multiple event types
        types = {e.event_type for e in events}
        assert EventType.FILE_READ in types
        assert EventType.FILE_WRITE in types
        assert EventType.PHASE_BOUNDARY in types

    def test_demo_through_monitor_produces_full_state(self):
        """Events fed through UniversalMonitor produce a complete state."""
        from agentdiag.demo import _generate_events
        from agentdiag.universal_monitor import UniversalMonitor

        monitor = UniversalMonitor(calibration_window=80, sensitivity=2.5)
        events = _generate_events()

        for event in events:
            monitor.process(event)

        state = monitor.get_state()
        assert state["total_events"] >= 200
        assert isinstance(state["anomalies"], list)
        assert isinstance(state["baseline"], dict)
        assert isinstance(state["compositor"], dict)

    def test_demo_detects_at_least_one_anomaly(self):
        """The demo stream has planted anomalies that the pipeline should catch."""
        from agentdiag.demo import _generate_events
        from agentdiag.universal_monitor import UniversalMonitor

        monitor = UniversalMonitor(calibration_window=80, sensitivity=2.5)
        for event in _generate_events():
            monitor.process(event)

        state = monitor.get_state()
        assert len(state["anomalies"]) > 0, "Demo has planted anomalies but none were detected"

    def test_demo_state_has_info_theoretic_metrics(self):
        """get_state() includes IT metrics with non-empty histories."""
        from agentdiag.demo import _generate_events
        from agentdiag.universal_monitor import UniversalMonitor

        monitor = UniversalMonitor(calibration_window=80, sensitivity=2.5)
        for event in _generate_events():
            monitor.process(event)

        state = monitor.get_state()
        it = state["info_theoretic"]
        assert isinstance(it, dict)
        # Should have MI and KL divergence history from SymbolStream
        assert "action_mi" in it, f"Missing action_mi in IT keys: {sorted(it.keys())}"
        assert "kl_divergence" in it
        assert "action_mi_history" in it
        assert len(it["action_mi_history"]) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Synthetic eval end-to-end
# ═══════════════════════════════════════════════════════════════════════════

class TestSyntheticEvalEndToEnd:
    """Generate trace via trace_generator, run through runner, verify output."""

    def test_generate_and_run_one_task(self):
        """Full trace_generator → runner pipeline for a single task."""
        from agentdiag.eval.trace_generator import generate_trace, EXPECTED_SIGNATURES
        from agentdiag.eval.tasks import TASK_BANK
        from agentdiag.eval.runner import run_trace

        task = TASK_BANK[0]
        entries, meta = generate_trace(task, variant="loop", seed=42)

        # Write to temp file (runner expects a file path)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for entry in entries:
                f.write(json.dumps(entry, default=str) + "\n")
            trace_path = f.name

        try:
            result = run_trace(trace_path)
        finally:
            Path(trace_path).unlink()

        # Verify result structure
        assert result["total_events"] > 0
        assert isinstance(result["anomalies"], list)
        assert isinstance(result["metrics_timeline"], list)
        assert len(result["metrics_timeline"]) > 0

    def test_anomalous_trace_has_detections(self):
        """A loop-injected trace should produce anomaly detections."""
        from agentdiag.eval.trace_generator import generate_trace
        from agentdiag.eval.tasks import TASK_BANK
        from agentdiag.eval.runner import run_trace

        task = TASK_BANK[0]
        entries, meta = generate_trace(task, variant="loop", seed=42)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for entry in entries:
                f.write(json.dumps(entry, default=str) + "\n")
            trace_path = f.name

        try:
            result = run_trace(trace_path)
        finally:
            Path(trace_path).unlink()

        assert result["total_anomalies"] > 0, "Loop injection should produce anomalies"
        sigs = {a.get("signature") for a in result["anomalies"]}
        assert "mechanical_repetition" in sigs, (
            f"Expected mechanical_repetition in {sigs}"
        )

    def test_clean_trace_structure(self):
        """A clean trace should still produce valid output structure."""
        from agentdiag.eval.trace_generator import generate_trace
        from agentdiag.eval.tasks import TASK_BANK
        from agentdiag.eval.runner import run_trace

        task = TASK_BANK[0]
        entries, meta = generate_trace(task, variant="clean", seed=42)
        assert meta is None, "Clean variant should have no injection metadata"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for entry in entries:
                f.write(json.dumps(entry, default=str) + "\n")
            trace_path = f.name

        try:
            result = run_trace(trace_path)
        finally:
            Path(trace_path).unlink()

        assert result["total_events"] > 0
        assert isinstance(result["anomalies"], list)
        assert isinstance(result["metrics_timeline"], list)


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Harness end-to-end
# ═══════════════════════════════════════════════════════════════════════════

class TestHarnessEndToEnd:
    """Run HarnessOrchestrator with mock agents, verify lifecycle."""

    @staticmethod
    def _mock_planner(goal: str, context: dict) -> list[dict]:
        return [
            {"goal": "Build feature A", "deliverables": ["file_a.py"],
             "success_criteria": ["tests pass"]},
        ]

    @staticmethod
    def _mock_generator(contract, context, feedback=None):
        return {"file_a.py": "def feature_a(): return True"}

    @staticmethod
    def _mock_evaluator(contract, artifacts, context):
        from agentdiag.harness import EvaluationGrade
        return EvaluationGrade(
            sprint_number=contract.sprint_number,
            overall_score=0.85,
            criteria_scores={"tests pass": 0.85},
            passed=True,
            critique="Looks good",
        )

    @staticmethod
    def _mock_evaluator_fail_then_pass():
        """Returns a factory that fails first, passes second."""
        call_count = [0]

        def evaluator(contract, artifacts, context):
            from agentdiag.harness import EvaluationGrade
            call_count[0] += 1
            if call_count[0] == 1:
                return EvaluationGrade(
                    sprint_number=contract.sprint_number,
                    overall_score=0.3,
                    criteria_scores={"tests pass": 0.3},
                    passed=False,
                    critique="Tests fail — missing edge case",
                )
            return EvaluationGrade(
                sprint_number=contract.sprint_number,
                overall_score=0.9,
                criteria_scores={"tests pass": 0.9},
                passed=True,
                critique="Fixed",
            )
        return evaluator

    def test_harness_emits_phase_boundaries_in_order(self):
        """Phase boundaries follow the expected lifecycle order."""
        from agentdiag.harness import HarnessOrchestrator
        from agentdiag.context.instrumented import InstrumentedContextStore

        events: list[ObservableEvent] = []
        store = InstrumentedContextStore(db_path="/tmp/agentdiag_test_ctx")
        store.on_event = lambda e: events.append(e)

        harness = HarnessOrchestrator(
            context_store=store,
            planner=self._mock_planner,
            generator=self._mock_generator,
            evaluator=self._mock_evaluator,
            on_event=lambda e: events.append(e),
        )

        result = harness.run("Test goal", max_sprints=1)

        # Extract phase boundary events
        phases = [
            e.phase.value
            for e in events
            if e.event_type == EventType.PHASE_BOUNDARY and e.phase is not None
        ]

        # Must see planning, contract negotiation, executing, verifying, retrospective
        assert "planning" in phases, f"Missing PLANNING in {phases}"
        assert "contract_negotiation" in phases, f"Missing CONTRACT_NEGOTIATION in {phases}"
        assert "executing" in phases, f"Missing EXECUTING in {phases}"
        assert "verifying" in phases, f"Missing VERIFYING in {phases}"
        assert "retrospective" in phases, f"Missing RETROSPECTIVE in {phases}"

        # Phases should be in lifecycle order
        pi = phases.index("planning")
        ci = phases.index("contract_negotiation")
        ei = phases.index("executing")
        vi = phases.index("verifying")
        ri = phases.index("retrospective")
        assert pi < ci < ei < vi < ri, f"Phase order wrong: {phases}"

    def test_harness_result_structure(self):
        """HarnessResult has expected fields when run completes."""
        from agentdiag.harness import HarnessOrchestrator
        from agentdiag.context.instrumented import InstrumentedContextStore

        store = InstrumentedContextStore(db_path="/tmp/agentdiag_test_ctx2")
        harness = HarnessOrchestrator(
            context_store=store,
            planner=self._mock_planner,
            generator=self._mock_generator,
            evaluator=self._mock_evaluator,
        )

        result = harness.run("Build a widget", max_sprints=1)
        assert result.goal == "Build a widget"
        assert len(result.sprints) == 1
        assert result.sprints[0].sprint_number == 1
        assert result.sprints[0].final_passed is True
        assert result.overall_passed is True
        assert result.duration_sec > 0

    def test_harness_retrospective_writes_memory(self):
        """Retrospective emits memory_store events to skill/memory paths."""
        from agentdiag.harness import HarnessOrchestrator
        from agentdiag.context.instrumented import InstrumentedContextStore

        events: list[ObservableEvent] = []

        def collect(e):
            events.append(e)

        store = InstrumentedContextStore(db_path="/tmp/agentdiag_test_ctx3")
        store.on_event = collect

        harness = HarnessOrchestrator(
            context_store=store,
            planner=self._mock_planner,
            generator=self._mock_generator,
            evaluator=self._mock_evaluator,
            on_event=collect,
        )

        result = harness.run("Test retrospective", max_sprints=1)

        # Filter memory store events
        memory_stores = [
            e for e in events
            if e.event_type == EventType.MEMORY_STORE
        ]

        # Since the sprint passes, the retrospective should write to design_patterns
        uris = [e.viking_uri or "" for e in memory_stores]
        design_patterns = [u for u in uris if "design_patterns" in u]
        assert len(design_patterns) > 0, (
            f"Expected design_patterns write in retrospective. URIs: {uris}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Adapter round-trip
# ═══════════════════════════════════════════════════════════════════════════

class TestAdapterRoundTrip:
    """Serialize ObservableEvent → JSONL → ClaudeCodeAdapter → verify."""

    def test_observable_event_dict_round_trip(self):
        """ObservableEvent survives to_dict() → JSON → reconstruction."""
        original_events = [
            file_read_event(step=1, timestamp=100.0, path="/src/main.py",
                            output_tokens=500),
            file_write_event(step=2, timestamp=101.0, path="/src/main.py",
                             input_tokens=200),
            tool_call_event(step=3, timestamp=102.0, tool_name="Bash",
                            duration_ms=1500, input_tokens=50, output_tokens=300),
            phase_boundary_event(step=4, timestamp=103.0,
                                 phase=HarnessPhase.EXECUTING),
            memory_load_event(step=5, timestamp=104.0,
                              uri="viking://test", tier=MemoryTier.L0,
                              token_count=100),
        ]

        for orig in original_events:
            serialized = orig.to_dict()
            json_str = json.dumps(serialized, default=str)
            d = json.loads(json_str)

            # Verify key fields survive
            assert d["step"] == orig.step
            assert d["timestamp"] == orig.timestamp
            assert d["event_type"] == orig.event_type.value
            assert d["symbol"] == orig.to_symbol()

    def test_claude_adapter_parses_raw_cc_format(self):
        """ClaudeCodeAdapter handles raw Claude Code JSONL dict format."""
        from agentdiag.adapters.claude_adapter import ClaudeCodeAdapter

        adapter = ClaudeCodeAdapter()

        # Raw CC-style dicts (tool name + type fields)
        raw_entries = [
            {"step": 1, "type": "tool_call", "tool": "read",
             "timestamp": 100.0, "tokens_out": 500},
            {"step": 2, "type": "tool_call", "tool": "edit",
             "timestamp": 101.0, "tokens_in": 200},
            {"step": 3, "type": "tool_call", "tool": "bash",
             "timestamp": 102.0, "tokens_out": 300},
        ]

        expected_types = [EventType.FILE_READ, EventType.FILE_WRITE,
                          EventType.SHELL_COMMAND]

        for raw, expected_type in zip(raw_entries, expected_types):
            events = adapter.parse(raw)
            assert len(events) == 1
            assert events[0].event_type == expected_type
            assert events[0].step == raw["step"]

    def test_symbols_match_after_round_trip(self):
        """to_symbol() produces consistent output regardless of path."""
        events = [
            file_read_event(step=1, timestamp=100.0, path="/src/config.ts",
                            output_tokens=200),
            file_write_event(step=2, timestamp=101.0, path="/src/api.py",
                             input_tokens=100),
            tool_call_event(step=3, timestamp=102.0, tool_name="grep",
                            output_tokens=50),
        ]

        for event in events:
            sym1 = event.to_symbol()
            # Reconstruct from dict
            d = event.to_dict()
            rebuilt = ObservableEvent(
                step=d["step"],
                timestamp=d["timestamp"],
                event_type=EventType(d["event_type"]),
                tool_name=d.get("tool_name"),
                target_path=d.get("target_path"),
            )
            sym2 = rebuilt.to_symbol()
            assert sym1 == sym2, f"Symbol mismatch: {sym1!r} vs {sym2!r}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Graceful degradation
# ═══════════════════════════════════════════════════════════════════════════

class TestGracefulDegradation:
    """Verify UniversalMonitor works with partial event streams."""

    def test_tool_calls_only(self):
        """Monitor produces IT metrics from tool-call-only events."""
        from agentdiag.universal_monitor import UniversalMonitor

        monitor = UniversalMonitor(calibration_window=30, sensitivity=2.0)
        t = 1000.0

        for i in range(1, 60):
            t += 0.5
            event = tool_call_event(
                step=i, timestamp=t,
                tool_name=["Read", "Edit", "Bash", "Grep"][i % 4],
                output_tokens=100,
            )
            result = monitor.process(event)
            assert result is not None
            if result.get("metrics"):
                assert "action_entropy" in result["metrics"]

        state = monitor.get_state()
        assert state["total_events"] == 59
        assert isinstance(state["info_theoretic"], dict)

    def test_adding_phase_boundaries(self):
        """Phase-conditional baselines activate when phase events arrive."""
        from agentdiag.universal_monitor import UniversalMonitor

        monitor = UniversalMonitor(calibration_window=30, sensitivity=2.0)
        t = 1000.0

        # Tool calls only (calibration)
        for i in range(1, 31):
            t += 0.5
            monitor.process(tool_call_event(
                step=i, timestamp=t,
                tool_name="Read", output_tokens=100,
            ))

        # Add a phase boundary
        t += 0.5
        result = monitor.process(phase_boundary_event(
            step=31, timestamp=t,
            phase=HarnessPhase.EXECUTING,
            agent_role=AgentRole.GENERATOR,
        ))
        assert result is not None
        assert result.get("type") == "phase_marker"

        state = monitor.get_state()
        assert state.get("current_phase") is not None

    def test_adding_memory_ops(self):
        """Memory track populates when memory events arrive."""
        from agentdiag.universal_monitor import UniversalMonitor

        monitor = UniversalMonitor(calibration_window=30, sensitivity=2.0)
        t = 1000.0

        # Build some baseline
        for i in range(1, 31):
            t += 0.5
            monitor.process(tool_call_event(
                step=i, timestamp=t,
                tool_name="Read", output_tokens=100,
            ))

        # Add memory events
        for i in range(31, 41):
            t += 0.5
            monitor.process(memory_load_event(
                step=i, timestamp=t,
                uri=f"viking://agent/planner/skills/item_{i}",
                tier=MemoryTier.L0,
                token_count=500,
            ))

        state = monitor.get_state()
        wm = state.get("working_memory", {})
        assert wm.get("total_items", 0) > 0, "Memory track should show items after memory events"


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: Compositor signature discrimination
# ═══════════════════════════════════════════════════════════════════════════

class TestCompositorSignatureDiscrimination:
    """Each named signature fires on its exact trigger and nothing else."""

    @pytest.fixture
    def compositor(self):
        from agentdiag.compositor import CompositionalAnomalyDetector
        return CompositionalAnomalyDetector()

    def _anomaly(self, **kwargs) -> dict:
        """Build a minimal anomaly dict. kwargs are metric_name=direction pairs."""
        return {
            name: {"value": 1.0, "z_score": 5.0, "direction": direction}
            for name, direction in kwargs.items()
        }

    def test_empty_anomaly_dict_returns_none(self, compositor):
        """No signature fires on empty input."""
        assert compositor.analyze({}) is None

    def test_single_metric_returns_none(self, compositor):
        """Single-metric anomaly is noise, not a signature."""
        result = compositor.analyze(
            self._anomaly(action_entropy="low"),
            step=1,
        )
        assert result is None

    def test_mechanical_repetition(self, compositor):
        """Low entropy + low MI = mechanical_repetition."""
        result = compositor.analyze(
            self._anomaly(action_entropy="low", action_mi="low"),
            step=1,
        )
        assert result is not None
        assert result.name == "mechanical_repetition"
        assert result.wickens_stage == "response_selection"

    def test_tight_iteration(self, compositor):
        """Low entropy + high MI = tight_iteration."""
        result = compositor.analyze(
            self._anomaly(action_entropy="low", action_mi="high"),
            step=1,
        )
        assert result is not None
        assert result.name == "tight_iteration"
        assert result.wickens_stage == "response_execution"

    def test_distributional_shift(self, compositor):
        """KL high + action_entropy high = distributional_shift."""
        result = compositor.analyze(
            self._anomaly(kl_divergence="high", action_entropy="high"),
            step=1,
        )
        assert result is not None
        assert result.name == "distributional_shift"
        assert result.wickens_stage == "perceptual"

    def test_context_thrashing(self, compositor):
        """KL high + MI high (without entropy high) = context_thrashing."""
        result = compositor.analyze(
            self._anomaly(kl_divergence="high", action_mi="high"),
            step=1,
        )
        assert result is not None
        assert result.name == "context_thrashing"
        assert result.wickens_stage == "working_memory"

    def test_execution_regression(self, compositor):
        """Compression low + entropy low = execution_regression."""
        result = compositor.analyze(
            self._anomaly(compression_ratio="low", action_entropy="low"),
            step=1,
        )
        assert result is not None
        assert result.name == "execution_regression"
        assert result.wickens_stage == "response_execution"

    def test_stagnation(self, compositor):
        """Compression low + KL low = stagnation."""
        result = compositor.analyze(
            self._anomaly(compression_ratio="low", kl_divergence="low"),
            step=1,
        )
        assert result is not None
        assert result.name == "stagnation"
        assert result.wickens_stage == "response_selection"

    def test_distributional_anomaly(self, compositor):
        """KL high + entropy low (and no higher-priority match) = distributional_anomaly."""
        result = compositor.analyze(
            self._anomaly(kl_divergence="high", action_entropy="low"),
            step=1,
        )
        assert result is not None
        assert result.name == "distributional_anomaly"
        assert result.wickens_stage == "response_selection"

    def test_unclassified_anomaly(self, compositor):
        """Multi-metric anomaly with no matching pattern = unclassified."""
        result = compositor.analyze(
            self._anomaly(
                tool_entropy="high",
                read_entropy="low",
            ),
            step=1,
        )
        assert result is not None
        assert result.name == "unclassified_anomaly"

    def test_no_cross_fire_between_drift_and_thrash(self, compositor):
        """Drift trigger doesn't fire thrash, and vice versa."""
        # Drift: KL high + entropy high → distributional_shift (NOT context_thrashing)
        drift = self._anomaly(kl_divergence="high", action_entropy="high")
        r = compositor.analyze(drift, step=1)
        assert r.name == "distributional_shift"

        # Thrash: KL high + MI high (no entropy high) → context_thrashing (NOT distributional_shift)
        thrash = self._anomaly(kl_divergence="high", action_mi="high")
        r = compositor.analyze(thrash, step=2)
        assert r.name == "context_thrashing"


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: Cross-agent file consistency
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossAgentConsistency:
    """Modules from different agents agree on shared interfaces."""

    def test_observable_event_symbol_consistency(self):
        """to_symbol() returns the same value everywhere the event is used."""
        event = file_read_event(step=1, timestamp=100.0, path="/src/main.py",
                                output_tokens=500)
        sym = event.to_symbol()

        # Verify symbol format is consistent
        assert sym.startswith("read:"), f"Expected read: prefix, got {sym!r}"

        # Rebuild from dict (as adapters do)
        d = event.to_dict()
        rebuilt = ObservableEvent(
            step=d["step"],
            timestamp=d["timestamp"],
            event_type=EventType(d["event_type"]),
            tool_name=d.get("tool_name"),
            target_path=d.get("target_path"),
        )
        assert rebuilt.to_symbol() == sym

    def test_all_event_types_have_symbols(self):
        """Every EventType variant produces a non-empty symbol."""
        events = [
            tool_call_event(step=1, timestamp=100.0, tool_name="Bash"),
            file_read_event(step=2, timestamp=101.0, path="/a.py"),
            file_write_event(step=3, timestamp=102.0, path="/b.py"),
            ObservableEvent(step=4, timestamp=103.0,
                            event_type=EventType.SHELL_COMMAND, tool_name="npm"),
            phase_boundary_event(step=5, timestamp=104.0,
                                 phase=HarnessPhase.PLANNING),
            evaluation_event(step=6, timestamp=105.0, score=0.8,
                             criterion="correctness"),
            memory_load_event(step=7, timestamp=106.0,
                              uri="viking://agent/planner/skills/x",
                              tier=MemoryTier.L0, token_count=100),
            memory_store_event(step=8, timestamp=107.0,
                               uri="viking://agent/gen/skills/y",
                               token_count=200),
            tier_escalation_event(step=9, timestamp=108.0,
                                  uri="viking://test/esc",
                                  from_tier=MemoryTier.L0,
                                  to_tier=MemoryTier.L1,
                                  token_count=300),
        ]

        for event in events:
            sym = event.to_symbol()
            assert isinstance(sym, str) and len(sym) > 0, (
                f"Empty symbol for {event.event_type}"
            )

    def test_monitor_and_adapter_agree_on_event_types(self):
        """Events from ClaudeCodeAdapter are accepted by UniversalMonitor."""
        from agentdiag.adapters.claude_adapter import ClaudeCodeAdapter
        from agentdiag.universal_monitor import UniversalMonitor

        adapter = ClaudeCodeAdapter()
        monitor = UniversalMonitor(calibration_window=10, sensitivity=2.0)

        raw_entries = [
            {"step": 1, "timestamp": 100.0, "tool": "Read", "type": "tool_call",
             "tokens_out": 500},
            {"step": 2, "timestamp": 101.0, "tool": "Edit", "type": "tool_call",
             "tokens_in": 200},
            {"step": 3, "timestamp": 102.0, "tool": "Bash", "type": "tool_call",
             "tokens_out": 100},
        ]

        for raw in raw_entries:
            events = adapter.parse(raw)
            for event in events:
                result = monitor.process(event)
                assert result is not None

    def test_trace_generator_events_run_through_full_pipeline(self):
        """Events from trace_generator work with baseline + compositor."""
        from agentdiag.eval.trace_generator import generate_trace
        from agentdiag.eval.tasks import TASK_BANK
        from agentdiag.baseline import SelfCalibratingBaseline
        from agentdiag.compositor import CompositionalAnomalyDetector
        from agentdiag.cognitive import SymbolStream

        task = TASK_BANK[0]
        entries, meta = generate_trace(task, variant="clean", seed=42)

        stream = SymbolStream(window=50)
        baseline = SelfCalibratingBaseline(calibration_window=80, sensitivity=3.0)
        compositor = CompositionalAnomalyDetector()

        for entry in entries:
            tool = entry.get("tool", "unknown")
            target = entry.get("target_path", "")
            symbol = f"{tool}:{target}" if target else tool
            stream.push(symbol)

            metrics = {
                "action_entropy": stream.entropy(),
                "action_mi": stream.mi(),
                "compression_ratio": stream.compression(),
                "last_surprisal": stream.surprisal(),
                "kl_divergence": stream.kl_divergence(),
            }
            anomalies = baseline.observe(metrics)
            if anomalies:
                compositor.analyze(anomalies, step=entry.get("step"))

        # Verify the full pipeline ran without errors
        assert len(entries) > 0

    def test_harness_and_monitor_share_event_contract(self):
        """Events emitted by HarnessOrchestrator are valid for UniversalMonitor."""
        from agentdiag.harness import HarnessOrchestrator, EvaluationGrade
        from agentdiag.context.instrumented import InstrumentedContextStore
        from agentdiag.universal_monitor import UniversalMonitor

        monitor = UniversalMonitor(calibration_window=20, sensitivity=2.0)
        results = []

        def on_event(e: ObservableEvent):
            r = monitor.process(e)
            results.append(r)

        store = InstrumentedContextStore(db_path="/tmp/agentdiag_test_cross")
        store.on_event = on_event

        harness = HarnessOrchestrator(
            context_store=store,
            planner=lambda goal, ctx: [{"goal": "test", "deliverables": ["x"]}],
            generator=lambda contract, ctx, fb=None: {"x": "done"},
            evaluator=lambda contract, art, ctx: EvaluationGrade(
                sprint_number=contract.sprint_number,
                overall_score=0.9, passed=True,
            ),
            on_event=on_event,
        )

        result = harness.run("Cross-agent test", max_sprints=1)

        # Verify monitor processed the events without error
        assert len(results) > 0
        state = monitor.get_state()
        assert state["total_events"] > 0
