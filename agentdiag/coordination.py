"""Cross-agent coordination monitoring — mutual information between agent pairs.

Measures whether agents working on related code have correlated action
sequences, detects coordination signals and failures, and infers dependency
graphs from the cross-agent MI matrix.

Theory: if Agent A writes a file and Agent B reads that file within N steps,
the write→read pair is a coordination signal (shared mental model). If Agent A
changes a file that Agent B previously read but B doesn't re-read it, that's
a coordination failure (stale mental model).

Cross-agent MI quantifies how much one agent's actions predict another's.
High MI = tightly coupled agents. Low MI = decoupled agents. Sudden MI
drop after sustained coupling = coordination breakdown.

Usage::

    from agentdiag.coordination import CoordinationTracker
    from agentdiag.universal_monitor import UniversalMonitor

    tracker = CoordinationTracker()
    tracker.register_agent("agent_a", monitor_a)
    tracker.register_agent("agent_b", monitor_b)

    # Feed events with agent_id
    tracker.observe("agent_a", event)
    tracker.observe("agent_b", event)

    state = tracker.get_state()
    # {"agents": [...], "edges": [...], "failures": [...], "signals": [...]}
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from agentdiag.observable import ObservableEvent, EventType
from agentdiag.universal_monitor import UniversalMonitor


# ── Data types ────────────────────────────────────────────────────────────

@dataclass
class CoordinationSignal:
    """A detected coordination event between two agents."""
    signal_type: str          # "write_read", "handoff", "shared_resource"
    source_agent: str
    target_agent: str
    resource: str             # file path or resource identifier
    source_step: int
    target_step: int
    latency_steps: int        # steps between source and target event
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "signal_type": self.signal_type,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "resource": self.resource,
            "source_step": self.source_step,
            "target_step": self.target_step,
            "latency_steps": self.latency_steps,
        }


@dataclass
class CoordinationFailure:
    """A detected coordination failure between two agents."""
    failure_type: str         # "stale_read", "race_condition", "contract_violation"
    agents: tuple[str, str]
    resource: str
    description: str
    severity: str             # "info", "warning", "critical"
    step: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "agents": list(self.agents),
            "resource": self.resource,
            "description": self.description,
            "severity": self.severity,
            "step": self.step,
        }


# ── Per-agent file access tracker ────────────────────────────────────────

class _AgentFileTracker:
    """Tracks which files an agent has read/written and when."""

    def __init__(self, max_history: int = 500) -> None:
        self._writes: dict[str, int] = {}       # resource → last write step
        self._reads: dict[str, int] = {}         # resource → last read step
        self._write_history: deque[tuple[str, int]] = deque(maxlen=max_history)
        self._read_history: deque[tuple[str, int]] = deque(maxlen=max_history)
        self._step = 0
        self._event_count = 0

    def record_write(self, resource: str, step: int) -> None:
        self._writes[resource] = step
        self._write_history.append((resource, step))
        self._step = step
        self._event_count += 1

    def record_read(self, resource: str, step: int) -> None:
        self._reads[resource] = step
        self._read_history.append((resource, step))
        self._step = step
        self._event_count += 1

    def last_write(self, resource: str) -> Optional[int]:
        return self._writes.get(resource)

    def last_read(self, resource: str) -> Optional[int]:
        return self._reads.get(resource)

    def has_read(self, resource: str) -> bool:
        return resource in self._reads

    def has_written(self, resource: str) -> bool:
        return resource in self._writes

    @property
    def written_resources(self) -> set[str]:
        return set(self._writes.keys())

    @property
    def read_resources(self) -> set[str]:
        return set(self._reads.keys())

    @property
    def step(self) -> int:
        return self._step

    @property
    def event_count(self) -> int:
        return self._event_count


# ── Cross-agent MI computer ──────────────────────────────────────────────

class CrossAgentMI:
    """Computes mutual information between two agents' action sequences.

    Tracks (agent_a_action, agent_b_action) bigrams within a temporal window.
    High MI means knowing what Agent A did tells you something about what
    Agent B will do — they are coordinating (or conflicting).
    """

    def __init__(self, window: int = 100, temporal_gap: int = 10) -> None:
        self._window = window
        self._temporal_gap = temporal_gap  # max steps between paired events
        self._pairs: deque[tuple[str, str]] = deque(maxlen=window)
        self._joint: dict[tuple[str, str], int] = {}
        self._margin_a: dict[str, int] = {}
        self._margin_b: dict[str, int] = {}
        self._total: int = 0
        self._history: deque[float] = deque(maxlen=200)

        # Pending events from each agent waiting to be paired
        self._pending_a: deque[tuple[str, int]] = deque(maxlen=50)
        self._pending_b: deque[tuple[str, int]] = deque(maxlen=50)

    def record_a(self, symbol: str, step: int) -> None:
        """Record an action from agent A."""
        self._pending_a.append((symbol, step))
        self._try_pair()

    def record_b(self, symbol: str, step: int) -> None:
        """Record an action from agent B."""
        self._pending_b.append((symbol, step))
        self._try_pair()

    def _try_pair(self) -> None:
        """Pair pending events from A and B that are within temporal_gap."""
        while self._pending_a and self._pending_b:
            sym_a, step_a = self._pending_a[0]
            sym_b, step_b = self._pending_b[0]

            if abs(step_a - step_b) <= self._temporal_gap:
                self._pending_a.popleft()
                self._pending_b.popleft()
                self._add_pair(sym_a, sym_b)
            elif step_a < step_b - self._temporal_gap:
                # A is too old, discard it
                self._pending_a.popleft()
            else:
                # B is too old, discard it
                self._pending_b.popleft()

    def _add_pair(self, sym_a: str, sym_b: str) -> None:
        pair = (sym_a, sym_b)

        # Evict oldest if at capacity
        if len(self._pairs) == self._pairs.maxlen:
            old = self._pairs[0]
            self._joint[old] -= 1
            if self._joint[old] == 0:
                del self._joint[old]
            self._margin_a[old[0]] -= 1
            if self._margin_a[old[0]] == 0:
                del self._margin_a[old[0]]
            self._margin_b[old[1]] -= 1
            if self._margin_b[old[1]] == 0:
                del self._margin_b[old[1]]
            self._total -= 1

        self._pairs.append(pair)
        self._joint[pair] = self._joint.get(pair, 0) + 1
        self._margin_a[pair[0]] = self._margin_a.get(pair[0], 0) + 1
        self._margin_b[pair[1]] = self._margin_b.get(pair[1], 0) + 1
        self._total += 1
        self._history.append(round(self._compute_mi(), 4))

    def _compute_mi(self) -> float:
        if self._total < 2:
            return 0.0
        mi = 0.0
        for (a, b), jc in self._joint.items():
            p_xy = jc / self._total
            p_x = self._margin_a.get(a, 0) / self._total
            p_y = self._margin_b.get(b, 0) / self._total
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * math.log2(p_xy / (p_x * p_y))
        return max(mi, 0.0)

    def mi(self) -> float:
        return self._history[-1] if self._history else 0.0

    @property
    def history(self) -> list[float]:
        return list(self._history)

    @property
    def pair_count(self) -> int:
        return self._total


# ── Coordination Tracker (main class) ────────────────────────────────────

class CoordinationTracker:
    """Monitors coordination between multiple agents.

    Given N agents (each with a UniversalMonitor), tracks:
    1. Cross-agent MI for each pair
    2. Coordination signals (A writes → B reads same file)
    3. Coordination failures (A changes file B already read, B doesn't re-read)
    4. Dependency graph (inferred from MI matrix)

    Degrades gracefully: with 0 or 1 agents, everything returns empty/defaults.
    """

    def __init__(
        self,
        coordination_window: int = 20,
        mi_window: int = 100,
        temporal_gap: int = 10,
    ) -> None:
        """
        Args:
            coordination_window: steps within which a read-after-write counts
                as a coordination signal.
            mi_window: sliding window for cross-agent MI computation.
            temporal_gap: max step difference for pairing cross-agent events.
        """
        self._coordination_window = coordination_window
        self._mi_window = mi_window
        self._temporal_gap = temporal_gap

        # Registered agents
        self._agents: dict[str, UniversalMonitor] = {}
        self._file_trackers: dict[str, _AgentFileTracker] = {}

        # Pairwise MI trackers: key = (agent_a, agent_b) sorted
        self._mi_trackers: dict[tuple[str, str], CrossAgentMI] = {}

        # Detected signals and failures
        self._signals: deque[CoordinationSignal] = deque(maxlen=500)
        self._failures: deque[CoordinationFailure] = deque(maxlen=500)

        # Global step counter (monotonic across all agents)
        self._global_step = 0

    def register_agent(
        self, agent_id: str, monitor: Optional[UniversalMonitor] = None
    ) -> None:
        """Register an agent for coordination tracking.

        Args:
            agent_id: Unique identifier for this agent.
            monitor: Optional UniversalMonitor instance. If None, a new one
                     is created. The monitor is used for individual agent
                     health; coordination metrics are computed separately.
        """
        if monitor is None:
            monitor = UniversalMonitor()
        self._agents[agent_id] = monitor
        self._file_trackers[agent_id] = _AgentFileTracker()

        # Create MI trackers for all pairs
        for other_id in self._agents:
            if other_id != agent_id:
                pair_key = self._pair_key(agent_id, other_id)
                if pair_key not in self._mi_trackers:
                    self._mi_trackers[pair_key] = CrossAgentMI(
                        window=self._mi_window,
                        temporal_gap=self._temporal_gap,
                    )

    def observe(self, agent_id: str, event: ObservableEvent) -> dict:
        """Process an event from a specific agent.

        Routes the event through:
        1. The agent's own UniversalMonitor
        2. The file tracker (for coordination signal/failure detection)
        3. Cross-agent MI computation (for all pairs involving this agent)

        Returns:
            dict with agent result + any coordination signals/failures detected.
        """
        if agent_id not in self._agents:
            raise ValueError(f"Unknown agent: {agent_id}. Call register_agent first.")

        self._global_step += 1
        monitor = self._agents[agent_id]
        tracker = self._file_trackers[agent_id]

        # 1. Process through the agent's own monitor
        result = monitor.process(event)

        # 2. Track file operations
        resource = self._extract_resource(event)
        new_signals = []
        new_failures = []

        if event.event_type == EventType.FILE_WRITE and resource:
            tracker.record_write(resource, self._global_step)
            # Check: did this agent overwrite a file another agent is reading?
            new_failures.extend(
                self._check_write_conflicts(agent_id, resource, self._global_step)
            )
            # Check for stale reads
            new_failures.extend(
                self._check_stale_reads(agent_id, resource, self._global_step)
            )

        elif event.event_type == EventType.FILE_READ and resource:
            tracker.record_read(resource, self._global_step)
            # Check: was this file recently written by another agent?
            new_signals.extend(
                self._check_read_after_write(agent_id, resource, self._global_step)
            )

        # 3. Update cross-agent MI for all pairs involving this agent
        symbol = event.to_symbol()
        for other_id in self._agents:
            if other_id == agent_id:
                continue
            pair_key = self._pair_key(agent_id, other_id)
            mi_tracker = self._mi_trackers[pair_key]
            if pair_key[0] == agent_id:
                mi_tracker.record_a(symbol, self._global_step)
            else:
                mi_tracker.record_b(symbol, self._global_step)

        for sig in new_signals:
            self._signals.append(sig)
        for fail in new_failures:
            self._failures.append(fail)

        return {
            "agent_result": result,
            "coordination_signals": [s.to_dict() for s in new_signals],
            "coordination_failures": [f.to_dict() for f in new_failures],
        }

    # ── Coordination detection ────────────────────────────────────────────

    def _check_read_after_write(
        self, reader_id: str, resource: str, step: int
    ) -> list[CoordinationSignal]:
        """Check if this read follows a recent write by another agent."""
        signals = []
        for other_id, other_tracker in self._file_trackers.items():
            if other_id == reader_id:
                continue
            write_step = other_tracker.last_write(resource)
            if write_step is not None:
                latency = step - write_step
                if latency <= self._coordination_window:
                    signals.append(CoordinationSignal(
                        signal_type="write_read",
                        source_agent=other_id,
                        target_agent=reader_id,
                        resource=resource,
                        source_step=write_step,
                        target_step=step,
                        latency_steps=latency,
                        timestamp=time.time(),
                    ))
        return signals

    def _check_write_conflicts(
        self, writer_id: str, resource: str, step: int
    ) -> list[CoordinationFailure]:
        """Check if this write conflicts with another agent's recent activity."""
        failures = []
        for other_id, other_tracker in self._file_trackers.items():
            if other_id == writer_id:
                continue

            # Race condition: two agents editing the same file within N steps
            other_write = other_tracker.last_write(resource)
            if other_write is not None:
                gap = step - other_write
                if gap <= self._coordination_window:
                    failures.append(CoordinationFailure(
                        failure_type="race_condition",
                        agents=(writer_id, other_id),
                        resource=resource,
                        description=(
                            f"Both {writer_id} and {other_id} wrote to "
                            f"{resource} within {gap} steps. "
                            f"Risk of merge conflict or overwritten work."
                        ),
                        severity="warning",
                        step=step,
                        timestamp=time.time(),
                    ))

            # Overwrite while reading: agent B is actively reading a file
            # that agent A just changed
            other_read = other_tracker.last_read(resource)
            if other_read is not None:
                gap = step - other_read
                if gap <= self._coordination_window:
                    failures.append(CoordinationFailure(
                        failure_type="concurrent_modification",
                        agents=(writer_id, other_id),
                        resource=resource,
                        description=(
                            f"{writer_id} modified {resource} which "
                            f"{other_id} read {gap} steps ago. "
                            f"{other_id} may be working with stale data."
                        ),
                        severity="info",
                        step=step,
                        timestamp=time.time(),
                    ))
        return failures

    def _check_stale_reads(
        self, writer_id: str, resource: str, step: int
    ) -> list[CoordinationFailure]:
        """Check if any other agent previously read this file but hasn't re-read it.

        This is the "stale mental model" failure: Agent B read file X at step 50,
        Agent A rewrites file X at step 200, and Agent B never re-reads it.
        """
        failures = []
        for other_id, other_tracker in self._file_trackers.items():
            if other_id == writer_id:
                continue

            read_step = other_tracker.last_read(resource)
            if read_step is None:
                continue

            # The other agent read this file BEFORE this write, and the gap
            # is large enough that they've moved on
            if read_step < step:
                gap = step - read_step
                if gap > self._coordination_window:
                    failures.append(CoordinationFailure(
                        failure_type="stale_read",
                        agents=(other_id, writer_id),
                        resource=resource,
                        description=(
                            f"{other_id} last read {resource} at step "
                            f"{read_step}, but {writer_id} modified it at "
                            f"step {step} ({gap} steps later). "
                            f"{other_id} may be working with outdated content."
                        ),
                        severity="warning",
                        step=step,
                        timestamp=time.time(),
                    ))
        return failures

    # ── State / serialization ─────────────────────────────────────────────

    def get_state(self) -> dict:
        """Full coordination state for dashboard rendering.

        Returns a graph structure:
        - nodes: one per agent with health color
        - edges: one per agent pair with MI weight
        - signals: recent coordination signals
        - failures: recent coordination failures
        """
        nodes = []
        for agent_id, monitor in self._agents.items():
            state = monitor.get_state()
            tracker = self._file_trackers[agent_id]
            nodes.append({
                "id": agent_id,
                "events": state["total_events"],
                "anomalies": len(state.get("anomalies", [])),
                "health": self._agent_health(state),
                "files_read": len(tracker.read_resources),
                "files_written": len(tracker.written_resources),
            })

        edges = []
        for (a, b), mi_tracker in self._mi_trackers.items():
            mi_val = mi_tracker.mi()
            edges.append({
                "source": a,
                "target": b,
                "mi": round(mi_val, 4),
                "pairs": mi_tracker.pair_count,
                "weight": self._edge_weight(mi_val),
                "status": self._edge_status(a, b),
                "mi_history": mi_tracker.history[-50:],
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "signals": [s.to_dict() for s in list(self._signals)[-20:]],
            "failures": [f.to_dict() for f in list(self._failures)[-20:]],
            "global_step": self._global_step,
            "dependency_graph": self._infer_dependencies(),
            "summary": self._summary(),
        }

    def get_mi_matrix(self) -> dict[tuple[str, str], float]:
        """Raw MI values for all agent pairs."""
        return {
            pair: tracker.mi()
            for pair, tracker in self._mi_trackers.items()
        }

    def _infer_dependencies(self) -> list[dict]:
        """Infer agent dependencies from write→read signal patterns.

        If Agent A frequently writes files that Agent B reads, B depends on A.
        """
        dep_counts: dict[tuple[str, str], int] = {}
        for sig in self._signals:
            if sig.signal_type == "write_read":
                key = (sig.source_agent, sig.target_agent)
                dep_counts[key] = dep_counts.get(key, 0) + 1

        deps = []
        for (source, target), count in sorted(
            dep_counts.items(), key=lambda x: -x[1]
        ):
            deps.append({
                "producer": source,
                "consumer": target,
                "signal_count": count,
                "strength": "strong" if count >= 5 else "moderate" if count >= 2 else "weak",
            })
        return deps

    def _summary(self) -> dict:
        """High-level coordination summary."""
        agent_count = len(self._agents)
        pair_count = len(self._mi_trackers)
        avg_mi = 0.0
        if pair_count > 0:
            mi_values = [t.mi() for t in self._mi_trackers.values()]
            avg_mi = sum(mi_values) / len(mi_values)

        failure_counts: dict[str, int] = {}
        for f in self._failures:
            failure_counts[f.failure_type] = failure_counts.get(f.failure_type, 0) + 1

        return {
            "agent_count": agent_count,
            "pair_count": pair_count,
            "average_mi": round(avg_mi, 4),
            "total_signals": len(self._signals),
            "total_failures": len(self._failures),
            "failure_breakdown": failure_counts,
            "coordination_health": self._coordination_health(avg_mi),
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _pair_key(a: str, b: str) -> tuple[str, str]:
        """Canonical pair key (sorted) so (a,b) == (b,a)."""
        return (min(a, b), max(a, b))

    @staticmethod
    def _extract_resource(event: ObservableEvent) -> Optional[str]:
        """Extract the file/resource identifier from an event."""
        if event.target_path:
            return event.target_path
        return None

    @staticmethod
    def _agent_health(state: dict) -> str:
        anomalies = state.get("anomalies", [])
        total = state.get("total_events", 0)
        if total == 0:
            return "unknown"
        rate = len(anomalies) / total if total > 0 else 0
        if rate > 0.15:
            return "red"
        elif rate > 0.05:
            return "yellow"
        return "green"

    @staticmethod
    def _edge_weight(mi: float) -> str:
        """Classify MI into weight for visualization."""
        if mi >= 1.5:
            return "thick"
        elif mi >= 0.5:
            return "medium"
        elif mi >= 0.1:
            return "thin"
        return "none"

    def _edge_status(self, agent_a: str, agent_b: str) -> str:
        """Determine edge status (normal, warning, breakdown)."""
        recent_failures = [
            f for f in list(self._failures)[-50:]
            if set(f.agents) == {agent_a, agent_b}
        ]
        if any(f.severity == "critical" for f in recent_failures):
            return "breakdown"
        if any(f.severity == "warning" for f in recent_failures):
            return "warning"
        return "normal"

    @staticmethod
    def _coordination_health(avg_mi: float) -> str:
        """Overall coordination health from average MI."""
        if avg_mi >= 0.5:
            return "coordinated"
        elif avg_mi >= 0.1:
            return "loosely_coupled"
        return "decoupled"
