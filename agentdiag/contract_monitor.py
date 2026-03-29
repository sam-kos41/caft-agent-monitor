"""
Contract-Aware Monitor — enriches IT anomaly detection with harness semantics.

Sits between the harness orchestrator and UniversalMonitor, translating
contract deliverables, evaluator findings, and planner expectations into
monitoring context that makes anomaly detection semantically meaningful.

Data flow:
    Agent -> ObservableEvent -> ContractAwareMonitor -> Visualization
                                       |        |
                              UniversalMonitor   |
                                       |        |
                              Harness --+--------+
                              (phases + contracts + evaluations + plans)

The IT foundation (UniversalMonitor, baseline, compositor, cognitive) is
unchanged. This module reads from it and adds to it, never replaces it.

Usage:
    from agentdiag.universal_monitor import UniversalMonitor
    from agentdiag.contract_monitor import ContractAwareMonitor

    monitor = ContractAwareMonitor(UniversalMonitor(sensitivity=2.0))
    monitor.set_contract(sprint_contract)

    for event in events:
        result = monitor.process(event)
        # result has standard IT fields + contract enrichment
"""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from agentdiag.observable import (
    EventType,
    HarnessPhase,
    ObservableEvent,
)
from agentdiag.universal_monitor import UniversalMonitor


# ---------------------------------------------------------------------------
# File pattern inference
# ---------------------------------------------------------------------------

# Map deliverable keywords to expected file patterns
_PATTERN_RULES = [
    # Frontend
    (r"frontend|react|ui|component|lobby|game.?room|scoreboard",
     [r"\.tsx?$", r"\.jsx?$", r"\.css$", r"\.html$", r"package\.json"]),
    # Backend
    (r"backend|fastapi|server|api|route|websocket|endpoint",
     [r"\.py$", r"main\.py", r"server\.py", r"routes/", r"requirements\.txt"]),
    # Database
    (r"database|sqlite|schema|migration|model",
     [r"\.py$", r"\.sql$", r"database", r"models?\."]),
    # Tests
    (r"test|pytest|vitest|playwright|e2e|integration",
     [r"test", r"spec\.", r"\.test\.", r"conftest"]),
    # Config / infra
    (r"docker|compose|readme|ci|cd|deploy",
     [r"docker", r"compose", r"README", r"\.yml$", r"\.yaml$", r"Dockerfile"]),
]


def _extract_file_patterns(contract) -> list[re.Pattern]:
    """Derive expected file path patterns from a sprint contract.

    Prefers explicit deliverable paths. Only falls back to keyword inference
    from the goal text if no deliverables look like file paths.
    """
    patterns = set()
    has_explicit_paths = False

    # From explicit deliverable paths (high priority)
    for d in contract.deliverables:
        if "/" in d or "." in d:
            has_explicit_paths = True
            # Match the file itself and its parent directory
            escaped = re.escape(d).replace(r"\*", ".*")
            patterns.add(escaped)
            # Also match the top-level directory (e.g., "backend/" from "backend/main.py")
            parts = d.split("/")
            if len(parts) > 1:
                patterns.add(re.escape(parts[0]) + r"/")

    # Only infer from keywords if no explicit paths were given
    if not has_explicit_paths:
        for d in contract.deliverables:
            for keyword_re, file_patterns in _PATTERN_RULES:
                if re.search(keyword_re, d, re.IGNORECASE):
                    patterns.update(file_patterns)
        for keyword_re, file_patterns in _PATTERN_RULES:
            if re.search(keyword_re, contract.goal, re.IGNORECASE):
                patterns.update(file_patterns)

    return [re.compile(p, re.IGNORECASE) for p in patterns] if patterns else []


def _path_matches(path: str, patterns: list[re.Pattern]) -> bool:
    """Check if a file path matches any of the expected patterns."""
    if not path or not patterns:
        return False
    return any(p.search(path) for p in patterns)


# ---------------------------------------------------------------------------
# Learned pattern storage
# ---------------------------------------------------------------------------

@dataclass
class LearnedPattern:
    """An IT signature that preceded (or missed) a bug."""
    signature: str
    preceded_bug: bool
    bug_criteria: list[str] = field(default_factory=list)
    metric_snapshot: dict = field(default_factory=dict)
    sprint_number: int = 0
    timestamp: float = 0.0


@dataclass
class DetectionGap:
    """A bug that had no preceding IT anomaly."""
    bug_criteria: list[str] = field(default_factory=list)
    metrics_at_failure: dict = field(default_factory=dict)
    sprint_number: int = 0
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# ContractAwareMonitor
# ---------------------------------------------------------------------------

class ContractAwareMonitor:
    """Enriches IT anomaly detection with harness semantic context.

    Wraps UniversalMonitor, adding:
    - Contract adherence tracking (is the agent working on the right files?)
    - Scope drift measurement (KL divergence from expected file distribution)
    - Evaluator-informed learning (which IT patterns predict real bugs?)
    - Planner-informed baselines (expected entropy per phase)
    - Cross-sprint learning (accumulated knowledge from past sprints)
    """

    def __init__(
        self,
        universal_monitor: UniversalMonitor,
        adherence_window: int = 30,
    ):
        self.monitor = universal_monitor
        self._adherence_window = adherence_window

        # Contract state
        self._current_contract: Optional[object] = None
        self._current_phase: Optional[str] = None
        self._expected_patterns: list[re.Pattern] = []
        self._steps_since_contract = 0

        # Adherence tracking (rolling window)
        self._recent_on_contract: deque[bool] = deque(maxlen=adherence_window)
        self._file_distribution: defaultdict[str, int] = defaultdict(int)
        self._expected_distribution: defaultdict[str, float] = defaultdict(float)

        # Sprint-level tracking
        self._sprint_step_ranges: dict[int, tuple[int, int]] = {}
        self._current_sprint_start: int = 0
        self._total_steps: int = 0

        # Metrics history (for evaluator correlation)
        self._metrics_history: deque[dict] = deque(maxlen=500)
        self._anomaly_history: deque[dict] = deque(maxlen=200)

        # Learning state
        self._learned_patterns: list[LearnedPattern] = []
        self._detection_gaps: list[DetectionGap] = []
        self._z_threshold_adjustments: dict[str, float] = {}

        # Planner expectations
        self._expected_entropy_profile: dict[str, tuple[float, float]] = {
            "planning": (3.0, 0.5),
            "contract_negotiation": (2.5, 0.5),
            "executing": (2.0, 0.4),
            "verifying": (2.5, 0.5),
            "iterating": (2.2, 0.4),
            "retrospective": (1.5, 0.5),
        }
        self._expected_phases: list[str] = []
        self._feature_list: list[str] = []

        # Contract anomaly timeline
        self._contract_anomalies: list[dict] = []

    # ------------------------------------------------------------------
    # Contract lifecycle
    # ------------------------------------------------------------------

    def set_contract(self, contract) -> None:
        """Called when a new sprint contract is accepted."""
        self._current_contract = contract
        self._expected_patterns = _extract_file_patterns(contract)
        self._steps_since_contract = 0
        self._recent_on_contract.clear()
        self._file_distribution.clear()
        self._current_sprint_start = self._total_steps

        # Build expected file distribution from deliverables
        self._expected_distribution.clear()
        for d in contract.deliverables:
            # Normalize to directory category
            cat = self._categorize_path(d)
            self._expected_distribution[cat] += 1.0
        # Normalize
        total = sum(self._expected_distribution.values()) or 1.0
        for k in self._expected_distribution:
            self._expected_distribution[k] /= total

    def set_phase(self, phase: str) -> None:
        """Called when the harness enters a new phase."""
        self._current_phase = phase

    def set_plan(self, plan: dict) -> None:
        """Set expectations from the planner's spec."""
        self._expected_phases = plan.get("phases", [])
        self._feature_list = plan.get("features", [])

        # Override entropy profile if plan provides one
        if "entropy_profile" in plan:
            self._expected_entropy_profile.update(plan["entropy_profile"])

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def process(self, event: ObservableEvent) -> dict:
        """Process event through both IT and contract-aware layers."""
        self._total_steps += 1

        # Handle phase boundaries
        if event.is_phase_marker():
            phase = event.phase
            if phase is not None:
                phase_str = phase.value if hasattr(phase, "value") else str(phase)
                self.set_phase(phase_str)

        # Handle contract events
        if event.event_type == EventType.CONTRACT_EVENT:
            if event.contract_status == "accepted" and event.metadata:
                # Contract accepted — extract from metadata if available
                pass

        # Standard IT analysis
        it_result = self.monitor.process(event)

        # Record metrics for later correlation
        if it_result.get("metrics"):
            self._metrics_history.append({
                "step": self._total_steps,
                "metrics": it_result["metrics"].copy(),
                "phase": self._current_phase,
                "sprint": self._current_contract.sprint_number if self._current_contract else None,
            })

        # Record anomalies for later correlation
        if it_result.get("anomalies"):
            self._anomaly_history.append({
                "step": self._total_steps,
                "anomalies": it_result["anomalies"],
                "phase": self._current_phase,
                "sprint": self._current_contract.sprint_number if self._current_contract else None,
            })

        # Contract-aware enrichment
        if self._current_contract and not event.is_phase_marker():
            self._steps_since_contract += 1
            contract_result = self._enrich_with_contract(event, it_result)
            it_result["contract"] = contract_result

            # Check for contract-specific anomalies
            contract_anomaly = self._check_contract_anomaly(contract_result)
            if contract_anomaly:
                it_result["contract_anomaly"] = contract_anomaly
                self._contract_anomalies.append({
                    **contract_anomaly,
                    "step": self._total_steps,
                    "phase": self._current_phase,
                })

        # Planner-informed entropy check
        if self._current_phase and it_result.get("metrics"):
            plan_check = self._check_against_plan(it_result["metrics"])
            if plan_check:
                it_result["plan_deviation"] = plan_check

        return it_result

    # ------------------------------------------------------------------
    # Contract adherence
    # ------------------------------------------------------------------

    def _enrich_with_contract(self, event: ObservableEvent, it_result: dict) -> dict:
        """Compute contract-relative metrics for this event."""
        path = event.target_path or ""
        on_contract = _path_matches(path, self._expected_patterns)
        self._recent_on_contract.append(on_contract)

        # Update file distribution
        cat = self._categorize_path(path)
        if cat:
            self._file_distribution[cat] += 1

        # Adherence: fraction of recent actions that match contract files
        if self._recent_on_contract:
            adherence = sum(self._recent_on_contract) / len(self._recent_on_contract)
        else:
            adherence = 1.0  # No data yet, assume on track

        # Scope drift: KL divergence between actual and expected file distributions
        scope_drift = self._compute_scope_drift()

        return {
            "adherence": round(adherence, 3),
            "scope_drift": round(scope_drift, 4),
            "on_task": on_contract,
            "current_file": path,
            "expected_patterns": len(self._expected_patterns),
            "steps_since_contract": self._steps_since_contract,
            "sprint_number": self._current_contract.sprint_number if self._current_contract else None,
        }

    def _compute_scope_drift(self) -> float:
        """KL divergence between actual and expected file distribution."""
        if not self._expected_distribution or not self._file_distribution:
            return 0.0

        total_actual = sum(self._file_distribution.values()) or 1.0
        all_cats = set(self._expected_distribution.keys()) | set(self._file_distribution.keys())

        kl = 0.0
        epsilon = 1e-6
        for cat in all_cats:
            p = self._file_distribution.get(cat, 0) / total_actual  # actual
            q = self._expected_distribution.get(cat, epsilon)       # expected
            p = max(p, epsilon)
            q = max(q, epsilon)
            kl += p * math.log2(p / q)

        return max(0.0, kl)

    def _categorize_path(self, path: str) -> str:
        """Categorize a file path into a broad category."""
        if not path:
            return ""
        path_lower = path.lower()
        if any(x in path_lower for x in ["test", "spec", "conftest"]):
            return "test"
        if any(x in path_lower for x in [".tsx", ".jsx", ".ts", ".js", ".css", ".html", "frontend", "src/component"]):
            return "frontend"
        if any(x in path_lower for x in [".py", "backend", "routes/", "main.py", "server.py"]):
            return "backend"
        if any(x in path_lower for x in ["docker", "compose", ".yml", ".yaml", "readme"]):
            return "config"
        if any(x in path_lower for x in [".sql", "database", "migration"]):
            return "database"
        return "other"

    # ------------------------------------------------------------------
    # Contract anomaly detection
    # ------------------------------------------------------------------

    def _check_contract_anomaly(self, contract_result: dict) -> Optional[dict]:
        """Check for contract-specific anomalies."""
        adherence = contract_result["adherence"]
        scope_drift = contract_result["scope_drift"]
        steps = contract_result["steps_since_contract"]

        # Don't flag in the first 20 steps — agent may be reading context
        if steps < 20:
            return None

        anomalies = []

        # Low adherence: agent working on wrong files
        if adherence < 0.3:
            anomalies.append({
                "type": "scope_deviation",
                "severity": "warning" if adherence > 0.15 else "critical",
                "message": (
                    f"Agent is off-contract: {adherence:.0%} of last "
                    f"{len(self._recent_on_contract)} actions match deliverables"
                ),
                "wickens_stage": "response_selection",
                "adherence": adherence,
            })

        # High scope drift: file distribution doesn't match expectations
        if scope_drift > 2.0 and steps > 30:
            anomalies.append({
                "type": "scope_drift",
                "severity": "warning" if scope_drift < 3.0 else "critical",
                "message": (
                    f"Scope drift KL={scope_drift:.2f}: agent's file access "
                    f"pattern diverges from contract expectations"
                ),
                "wickens_stage": "attention",
                "scope_drift": scope_drift,
            })

        # High adherence but no progress (reading contract files but not writing)
        if adherence > 0.8 and steps > 50:
            # Check consolidation from IT metrics
            recent_metrics = list(self._metrics_history)[-10:]
            if recent_metrics:
                # If mostly reads and no writes, flag stall
                pass  # This is already caught by IT consolidation metric

        return anomalies[0] if anomalies else None

    # ------------------------------------------------------------------
    # Planner-informed checks
    # ------------------------------------------------------------------

    def _check_against_plan(self, metrics: dict) -> Optional[dict]:
        """Check if current metrics deviate from planner's phase expectations."""
        phase = self._current_phase
        if not phase or phase not in self._expected_entropy_profile:
            return None

        expected_mean, expected_std = self._expected_entropy_profile[phase]
        actual_entropy = metrics.get("action_entropy", metrics.get("tool_entropy", 0))

        if actual_entropy == 0:
            return None

        z = abs(actual_entropy - expected_mean) / max(expected_std, 0.01)
        if z > 2.5:
            direction = "higher" if actual_entropy > expected_mean else "lower"
            return {
                "phase": phase,
                "expected_entropy": expected_mean,
                "actual_entropy": round(actual_entropy, 3),
                "z_score": round(z, 2),
                "message": (
                    f"During {phase} phase, entropy is {direction} than expected "
                    f"({actual_entropy:.2f} vs {expected_mean:.1f} +/- {expected_std:.1f})"
                ),
            }

        return None

    # ------------------------------------------------------------------
    # Evaluator-informed learning
    # ------------------------------------------------------------------

    def process_evaluation(
        self,
        grade,
        step_range: Optional[tuple[int, int]] = None,
    ) -> dict:
        """Learn from evaluator findings to improve future detection.

        Called after each sprint evaluation. Correlates evaluator grades
        with preceding IT anomalies to learn which signals predict bugs.

        Returns a summary of what was learned.
        """
        if step_range is None:
            sprint_num = grade.sprint_number
            start = self._sprint_step_ranges.get(sprint_num, (0, self._total_steps))[0]
            step_range = (start, self._total_steps)

        # Record sprint step range
        self._sprint_step_ranges[grade.sprint_number] = step_range

        # Find IT anomalies in the window preceding the evaluation
        preceding_window = (step_range[0], step_range[1])
        preceding_anomalies = [
            a for a in self._anomaly_history
            if preceding_window[0] <= a["step"] <= preceding_window[1]
        ]

        # Find contract anomalies in the same window
        preceding_contract_anomalies = [
            a for a in self._contract_anomalies
            if preceding_window[0] <= a["step"] <= preceding_window[1]
        ]

        # Identify failed criteria
        failed_criteria = [
            k for k, v in grade.criteria_scores.items()
            if v < 0.7
        ]

        learned = {
            "sprint_number": grade.sprint_number,
            "overall_score": grade.overall_score,
            "passed": grade.passed,
            "failed_criteria": failed_criteria,
            "it_anomalies_preceding": len(preceding_anomalies),
            "contract_anomalies_preceding": len(preceding_contract_anomalies),
            "patterns_learned": [],
            "gaps_found": [],
        }

        if grade.overall_score < 0.7:
            # Sprint had issues
            if preceding_anomalies or preceding_contract_anomalies:
                # IT/contract measures predicted the bug
                for a in preceding_anomalies:
                    pattern = LearnedPattern(
                        signature=str(a.get("anomalies", {}).get("signature", "unknown")),
                        preceded_bug=True,
                        bug_criteria=failed_criteria,
                        metric_snapshot=dict(list(self._metrics_history)[-1].get("metrics", {}))
                        if self._metrics_history else {},
                        sprint_number=grade.sprint_number,
                        timestamp=time.time(),
                    )
                    self._learned_patterns.append(pattern)
                    learned["patterns_learned"].append({
                        "signature": pattern.signature,
                        "bug_criteria": failed_criteria,
                    })

                    # Lower z-threshold for this signature in future sprints
                    sig = pattern.signature
                    current = self._z_threshold_adjustments.get(sig, 0.0)
                    self._z_threshold_adjustments[sig] = current - 0.5
                    learned["threshold_adjustment"] = {
                        sig: self._z_threshold_adjustments[sig]
                    }
            else:
                # IT measures missed the bug
                metrics_at_bug = dict(list(self._metrics_history)[-1].get("metrics", {})) \
                    if self._metrics_history else {}
                gap = DetectionGap(
                    bug_criteria=failed_criteria,
                    metrics_at_failure=metrics_at_bug,
                    sprint_number=grade.sprint_number,
                    timestamp=time.time(),
                )
                self._detection_gaps.append(gap)
                learned["gaps_found"].append({
                    "bug_criteria": failed_criteria,
                    "note": "No IT anomaly preceded this bug",
                    "metrics_at_failure": {
                        k: round(v, 3) if isinstance(v, float) else v
                        for k, v in metrics_at_bug.items()
                    },
                })

        return learned

    # ------------------------------------------------------------------
    # Cross-sprint learning
    # ------------------------------------------------------------------

    def get_learned_state(self) -> dict:
        """Export learned patterns for storage in OpenViking.

        Returns a dict suitable for storing at:
            viking://agent/monitor/learned_patterns/
            viking://agent/monitor/detection_gaps/
            viking://agent/monitor/baseline_adjustments/
        """
        return {
            "learned_patterns": [
                {
                    "signature": p.signature,
                    "preceded_bug": p.preceded_bug,
                    "bug_criteria": p.bug_criteria,
                    "sprint_number": p.sprint_number,
                }
                for p in self._learned_patterns
            ],
            "detection_gaps": [
                {
                    "bug_criteria": g.bug_criteria,
                    "sprint_number": g.sprint_number,
                    "metrics_at_failure": {
                        k: round(v, 3) if isinstance(v, float) else v
                        for k, v in g.metrics_at_failure.items()
                    },
                }
                for g in self._detection_gaps
            ],
            "baseline_adjustments": dict(self._z_threshold_adjustments),
            "total_patterns": len(self._learned_patterns),
            "total_gaps": len(self._detection_gaps),
        }

    def load_learned_state(self, state: dict) -> None:
        """Load previously learned patterns (from OpenViking).

        Called at the start of a new run to bootstrap from past experience.
        """
        for p in state.get("learned_patterns", []):
            self._learned_patterns.append(LearnedPattern(
                signature=p["signature"],
                preceded_bug=p["preceded_bug"],
                bug_criteria=p.get("bug_criteria", []),
                sprint_number=p.get("sprint_number", 0),
            ))

        for g in state.get("detection_gaps", []):
            self._detection_gaps.append(DetectionGap(
                bug_criteria=g.get("bug_criteria", []),
                metrics_at_failure=g.get("metrics_at_failure", {}),
                sprint_number=g.get("sprint_number", 0),
            ))

        self._z_threshold_adjustments.update(
            state.get("baseline_adjustments", {})
        )

    # ------------------------------------------------------------------
    # State export
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Full state for visualization / WebSocket serialization."""
        base_state = self.monitor.get_state()

        # Add contract layer
        base_state["contract"] = {
            "active": self._current_contract is not None,
            "sprint_number": self._current_contract.sprint_number
            if self._current_contract else None,
            "phase": self._current_phase,
            "adherence_window": len(self._recent_on_contract),
            "current_adherence": (
                sum(self._recent_on_contract) / len(self._recent_on_contract)
                if self._recent_on_contract else None
            ),
            "scope_drift": self._compute_scope_drift(),
            "expected_patterns": len(self._expected_patterns),
            "file_distribution": dict(self._file_distribution),
            "steps_since_contract": self._steps_since_contract,
        }

        # Add learning layer
        base_state["learning"] = self.get_learned_state()

        # Add contract anomaly timeline
        base_state["contract_anomalies"] = self._contract_anomalies[-50:]

        # Add plan expectations
        if self._current_phase:
            expected = self._expected_entropy_profile.get(self._current_phase)
            if expected:
                base_state["plan_expectations"] = {
                    "phase": self._current_phase,
                    "expected_entropy_mean": expected[0],
                    "expected_entropy_std": expected[1],
                }

        return base_state
