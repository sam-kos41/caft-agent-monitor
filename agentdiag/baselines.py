"""Normative baselines for agent behavior.

Learns what "normal" agent behavior looks like from real validation traces,
then provides statistical baselines for detecting genuine anomalies.

Core principle (from GazeVLM): NEVER use arbitrary thresholds.
All thresholds are data-derived percentiles, scale-invariant and fair.

Three models:
  NormativePhaseModel  — Per-phase distributions (step count, tool diversity, etc.)
  TransitionModel      — Markov chain P(next_phase | current_phase)
  ActionBaselineModel  — Within-phase action norms and n-gram frequencies

Usage:
    from agentdiag.baselines import CalibrationPipeline

    pipeline = CalibrationPipeline()
    pipeline.fit(traces_path="~/.claude/projects", splits_file="splits.json")
    pipeline.save("baselines.json")

    # Later:
    pipeline = CalibrationPipeline.load("baselines.json")
    is_anomalous = pipeline.phase_model.is_anomalous("gathering", "step_count", 50)
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from agentdiag.models import TraceEvent
from agentdiag.hta import Phase, HTAStateMachine, classify_event


# =========================================================================
# Phase-level feature extraction (from a list of TraceEvents)
# =========================================================================

@dataclass
class PhaseStats:
    """Statistics for one contiguous phase segment."""
    phase: str
    step_count: int
    tool_calls: int
    unique_tools: int
    tool_diversity: float       # unique_tools / tool_calls
    repetition_rate: float      # consecutive identical tools / tool_calls
    planning_events: int
    error_count: int
    error_rate: float
    total_tokens: int
    avg_latency_ms: float
    duration_sec: float
    action_entropy: float       # Shannon entropy of tool distribution

    def to_dict(self) -> dict:
        return asdict(self)


def _shannon_entropy(counts: dict[str, int]) -> float:
    """Compute Shannon entropy (bits) from a frequency dict."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def extract_phase_segments(events: list[TraceEvent]) -> list[tuple[str, list[TraceEvent]]]:
    """Run HTA on events and return contiguous phase segments.

    Returns list of (phase_name, events_in_that_phase).
    """
    if not events:
        return []

    sm = HTAStateMachine()
    segments: list[tuple[str, list[TraceEvent]]] = []
    current_events: list[TraceEvent] = []
    current_phase = None

    for event in events:
        state = sm.push(event)
        phase_name = state.current_phase.label

        if current_phase is None:
            current_phase = phase_name

        if phase_name != current_phase:
            if current_events:
                segments.append((current_phase, current_events))
            current_events = [event]
            current_phase = phase_name
        else:
            current_events.append(event)

    if current_events:
        segments.append((current_phase, current_events))

    return segments


def compute_phase_stats(phase_name: str, events: list[TraceEvent]) -> PhaseStats:
    """Compute aggregate stats for events in one phase segment."""
    n = len(events)
    tool_events = [e for e in events if e.type == "tool_call"]
    tools_used = [e.tool or "unknown" for e in tool_events]
    tool_counts = Counter(tools_used)

    # Tool diversity
    n_tools = len(tool_events)
    unique_tools = len(set(tools_used)) if tools_used else 0
    diversity = unique_tools / max(n_tools, 1)

    # Repetition rate
    consecutive = sum(
        1 for i in range(1, len(tools_used))
        if tools_used[i] == tools_used[i - 1]
    )
    rep_rate = consecutive / max(len(tools_used), 1)

    # Planning events
    planning = sum(1 for e in events if e.type in ("planning", "reasoning"))

    # Errors
    errors = sum(1 for e in events if not e.success)

    # Tokens
    total_tokens = sum(e.tokens_in + e.tokens_out for e in events)

    # Latency
    latencies = [e.latency_ms for e in events if e.latency_ms > 0]
    avg_latency = sum(latencies) / max(len(latencies), 1)

    # Duration
    timestamps = [e.timestamp for e in events if e.timestamp is not None]
    if len(timestamps) >= 2:
        duration = max(timestamps) - min(timestamps)
    else:
        duration = 0.0

    # Entropy
    entropy = _shannon_entropy(tool_counts)

    return PhaseStats(
        phase=phase_name,
        step_count=n,
        tool_calls=n_tools,
        unique_tools=unique_tools,
        tool_diversity=round(diversity, 4),
        repetition_rate=round(rep_rate, 4),
        planning_events=planning,
        error_count=errors,
        error_rate=round(errors / max(n, 1), 4),
        total_tokens=total_tokens,
        avg_latency_ms=round(avg_latency, 1),
        duration_sec=round(duration, 2),
        action_entropy=round(entropy, 4),
    )


# =========================================================================
# NormativePhaseModel — per-phase distributions with percentile thresholds
# =========================================================================

# Metrics tracked per phase
PHASE_METRICS = [
    "step_count", "tool_diversity", "repetition_rate",
    "action_entropy", "error_rate", "avg_latency_ms",
]


@dataclass
class PhaseDistribution:
    """Distribution of a metric within a phase, fitted from validation data."""
    phase: str
    metric: str
    values: list[float]
    mean: float = 0.0
    std: float = 0.0
    p5: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    n: int = 0

    def fit(self) -> None:
        """Compute distribution stats from values."""
        if not self.values:
            return
        self.n = len(self.values)
        self.mean = sum(self.values) / self.n
        variance = sum((v - self.mean) ** 2 for v in self.values) / max(self.n - 1, 1)
        self.std = variance ** 0.5
        sorted_vals = sorted(self.values)
        self.p5 = self._percentile(sorted_vals, 5)
        self.p25 = self._percentile(sorted_vals, 25)
        self.p50 = self._percentile(sorted_vals, 50)
        self.p75 = self._percentile(sorted_vals, 75)
        self.p95 = self._percentile(sorted_vals, 95)

    @staticmethod
    def _percentile(sorted_vals: list[float], pct: int) -> float:
        if not sorted_vals:
            return 0.0
        k = (len(sorted_vals) - 1) * pct / 100.0
        f = int(k)
        c = f + 1
        if c >= len(sorted_vals):
            return sorted_vals[-1]
        return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

    def is_anomalous(self, value: float, threshold_pct: int = 95) -> bool:
        """Check if value exceeds the threshold percentile.

        For metrics where high = bad (repetition_rate, error_rate),
        we flag values above the threshold.
        For metrics where low = bad (tool_diversity, action_entropy),
        the caller should negate or use is_below_threshold.

        Uses pre-computed percentile stats (survives serialization).
        """
        if self.n < 3:
            return False  # not enough data

        # Use pre-computed percentile (works after save/load)
        if threshold_pct == 95:
            return value > self.p95
        elif threshold_pct == 75:
            return value > self.p75
        elif threshold_pct == 50:
            return value > self.p50
        elif threshold_pct == 25:
            return value > self.p25
        elif threshold_pct == 5:
            return value > self.p5
        else:
            # Fall back to computing from values if available
            if self.values:
                threshold = self._percentile(sorted(self.values), threshold_pct)
                return value > threshold
            # Use P95 as default
            return value > self.p95

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "metric": self.metric,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "p5": round(self.p5, 4),
            "p25": round(self.p25, 4),
            "p50": round(self.p50, 4),
            "p75": round(self.p75, 4),
            "p95": round(self.p95, 4),
            "n": self.n,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PhaseDistribution":
        dist = cls(
            phase=d["phase"],
            metric=d["metric"],
            values=[],  # values not persisted (only stats)
        )
        dist.mean = d.get("mean", 0.0)
        dist.std = d.get("std", 0.0)
        dist.p5 = d.get("p5", 0.0)
        dist.p25 = d.get("p25", 0.0)
        dist.p50 = d.get("p50", 0.0)
        dist.p75 = d.get("p75", 0.0)
        dist.p95 = d.get("p95", 0.0)
        dist.n = d.get("n", 0)
        return dist


class NormativePhaseModel:
    """Per-phase statistical distributions fitted from validation traces.

    For each phase (gathering, planning, executing, verifying, delivering),
    tracks the distribution of key metrics (step_count, tool_diversity,
    repetition_rate, action_entropy, error_rate, avg_latency_ms).

    Anomaly detection uses the 95th percentile: a value is anomalous if
    it exceeds the 95th percentile of the validation distribution for
    that phase and metric.
    """

    def __init__(self):
        # phase -> metric -> PhaseDistribution
        self._distributions: dict[str, dict[str, PhaseDistribution]] = {}
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, all_phase_stats: list[PhaseStats]) -> None:
        """Fit distributions from a collection of PhaseStats.

        Args:
            all_phase_stats: PhaseStats from many sessions. There will be
                multiple PhaseStats for the same phase (one per session
                segment).
        """
        if not all_phase_stats:
            return

        # Group by phase
        by_phase: dict[str, list[PhaseStats]] = {}
        for ps in all_phase_stats:
            if ps.phase not in by_phase:
                by_phase[ps.phase] = []
            by_phase[ps.phase].append(ps)

        self._distributions = {}
        for phase, stats_list in by_phase.items():
            self._distributions[phase] = {}
            for metric in PHASE_METRICS:
                values = [getattr(s, metric) for s in stats_list]
                dist = PhaseDistribution(phase=phase, metric=metric, values=values)
                dist.fit()
                self._distributions[phase][metric] = dist

        self._fitted = True

    def get_distribution(self, phase: str, metric: str) -> Optional[PhaseDistribution]:
        """Get the distribution for a phase/metric."""
        return self._distributions.get(phase, {}).get(metric)

    def is_anomalous(
        self,
        phase: str,
        metric: str,
        value: float,
        threshold_pct: int = 95,
    ) -> bool:
        """Check if a value is anomalous for a given phase and metric."""
        dist = self.get_distribution(phase, metric)
        if dist is None or dist.n < 3:
            return False  # not enough data to judge
        return dist.is_anomalous(value, threshold_pct)

    def score_phase(self, phase_stats: PhaseStats) -> dict[str, bool]:
        """Score all metrics for a phase segment.

        Returns dict of {metric: is_anomalous}.
        """
        results = {}
        for metric in PHASE_METRICS:
            value = getattr(phase_stats, metric, None)
            if value is not None:
                results[metric] = self.is_anomalous(
                    phase_stats.phase, metric, value
                )
            else:
                results[metric] = False
        return results

    def get_phases(self) -> list[str]:
        """Get all phases with fitted distributions."""
        return list(self._distributions.keys())

    def to_dict(self) -> dict:
        result = {}
        for phase, metrics in self._distributions.items():
            result[phase] = {
                metric: dist.to_dict()
                for metric, dist in metrics.items()
            }
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "NormativePhaseModel":
        model = cls()
        for phase, metrics in d.items():
            model._distributions[phase] = {}
            for metric, dist_d in metrics.items():
                model._distributions[phase][metric] = PhaseDistribution.from_dict(dist_d)
        model._fitted = True
        return model


# =========================================================================
# TransitionModel — Markov chain P(next_phase | current_phase)
# =========================================================================

class TransitionModel:
    """Markov chain model of phase transitions.

    Learns P(next_phase | current_phase) from validation traces.
    Flags transitions with P < threshold as anomalous (unusual transitions
    that normal agents rarely make).
    """

    def __init__(self, min_probability: float = 0.05):
        # transition_counts[from_phase][to_phase] = count
        self._counts: dict[str, dict[str, int]] = {}
        # transition_probs[from_phase][to_phase] = probability
        self._probs: dict[str, dict[str, float]] = {}
        self._min_probability = min_probability
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, all_transitions: list[tuple[str, str]]) -> None:
        """Fit from observed phase transitions.

        Args:
            all_transitions: List of (from_phase, to_phase) pairs
                from all validation sessions.
        """
        if not all_transitions:
            return

        self._counts = {}
        for from_p, to_p in all_transitions:
            if from_p not in self._counts:
                self._counts[from_p] = {}
            self._counts[from_p][to_p] = self._counts[from_p].get(to_p, 0) + 1

        # Normalize to probabilities
        self._probs = {}
        for from_p, to_counts in self._counts.items():
            total = sum(to_counts.values())
            self._probs[from_p] = {}
            for to_p, count in to_counts.items():
                self._probs[from_p][to_p] = count / total

        self._fitted = True

    def probability(self, from_phase: str, to_phase: str) -> float:
        """Get P(to_phase | from_phase)."""
        return self._probs.get(from_phase, {}).get(to_phase, 0.0)

    def is_anomalous(self, from_phase: str, to_phase: str) -> bool:
        """Check if a transition is anomalous (P < min_probability)."""
        if not self._fitted:
            return False
        # If we've never seen transitions from this phase, can't judge
        if from_phase not in self._probs:
            return False
        p = self.probability(from_phase, to_phase)
        return p < self._min_probability

    def get_expected_next(self, from_phase: str) -> list[tuple[str, float]]:
        """Get likely next phases sorted by probability."""
        probs = self._probs.get(from_phase, {})
        return sorted(probs.items(), key=lambda x: -x[1])

    def to_dict(self) -> dict:
        return {
            "counts": self._counts,
            "probs": self._probs,
            "min_probability": self._min_probability,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TransitionModel":
        model = cls(min_probability=d.get("min_probability", 0.05))
        model._counts = d.get("counts", {})
        model._probs = d.get("probs", {})
        model._fitted = True
        return model


# =========================================================================
# ActionBaselineModel — within-phase action norms and n-gram frequencies
# =========================================================================

class ActionBaselineModel:
    """Per-phase action frequency norms.

    Learns the typical tool distribution within each phase and common
    tool bigrams. Flags unusual tool usage patterns.
    """

    def __init__(self):
        # phase -> Counter of tool names
        self._tool_freqs: dict[str, dict[str, float]] = {}
        # phase -> Counter of tool bigrams
        self._bigram_freqs: dict[str, dict[str, float]] = {}
        # phase -> set of known tools
        self._known_tools: dict[str, set[str]] = {}
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, phase_events: dict[str, list[list[str]]]) -> None:
        """Fit from per-phase tool sequences.

        Args:
            phase_events: {phase_name: [list_of_tool_sequences]},
                where each tool_sequence is from one session's phase segment.
        """
        if not phase_events:
            return

        self._tool_freqs = {}
        self._bigram_freqs = {}
        self._known_tools = {}

        for phase, sequences in phase_events.items():
            # Aggregate tool counts across all sessions
            total_tools = Counter()
            total_bigrams = Counter()
            all_tools = set()

            for seq in sequences:
                for tool in seq:
                    total_tools[tool] += 1
                    all_tools.add(tool)
                for i in range(len(seq) - 1):
                    bigram = f"{seq[i]}→{seq[i+1]}"
                    total_bigrams[bigram] += 1

            # Normalize to proportions
            total = sum(total_tools.values()) or 1
            self._tool_freqs[phase] = {
                tool: count / total for tool, count in total_tools.items()
            }

            total_bi = sum(total_bigrams.values()) or 1
            self._bigram_freqs[phase] = {
                bg: count / total_bi for bg, count in total_bigrams.items()
            }

            self._known_tools[phase] = all_tools

        self._fitted = True

    def get_tool_frequency(self, phase: str, tool: str) -> float:
        """Get the normalized frequency of a tool in a phase."""
        return self._tool_freqs.get(phase, {}).get(tool, 0.0)

    def is_novel_tool(self, phase: str, tool: str) -> bool:
        """Check if a tool was never seen in this phase during training."""
        known = self._known_tools.get(phase, set())
        if not known:
            return False  # no data, can't judge
        return tool not in known

    def get_novel_tools(self, phase: str, tools: list[str]) -> list[str]:
        """Return tools that were never seen in this phase during fitting."""
        known = self._known_tools.get(phase, set())
        if not known:
            return []
        return [t for t in tools if t not in known]

    def compute_tool_divergence(
        self,
        phase: str,
        observed_tools: list[str],
    ) -> float:
        """Compute KL-divergence-like score between observed and baseline.

        Returns a score >= 0. Higher = more divergent from baseline.
        Score of 0 means identical distribution.
        """
        expected = self._tool_freqs.get(phase, {})
        if not expected or not observed_tools:
            return 0.0

        observed_counts = Counter(observed_tools)
        total = sum(observed_counts.values())
        observed_freq = {t: c / total for t, c in observed_counts.items()}

        # Symmetric JS-like divergence (avoids log(0))
        all_tools = set(expected) | set(observed_freq)
        divergence = 0.0
        for tool in all_tools:
            p = observed_freq.get(tool, 0.0)
            q = expected.get(tool, 0.0)
            m = (p + q) / 2
            if m > 0:
                if p > 0:
                    divergence += p * math.log2(p / m)
                if q > 0:
                    divergence += q * math.log2(q / m)
        return divergence / 2.0  # normalize to [0, 1] for identical distributions

    def to_dict(self) -> dict:
        return {
            "tool_freqs": self._tool_freqs,
            "bigram_freqs": self._bigram_freqs,
            "known_tools": {p: list(tools) for p, tools in self._known_tools.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActionBaselineModel":
        model = cls()
        model._tool_freqs = d.get("tool_freqs", {})
        model._bigram_freqs = d.get("bigram_freqs", {})
        model._known_tools = {
            p: set(tools) for p, tools in d.get("known_tools", {}).items()
        }
        model._fitted = True
        return model


# =========================================================================
# CalibrationProfile — the complete fitted baseline
# =========================================================================

@dataclass
class CalibrationProfile:
    """Complete fitted baseline from validation traces."""
    phase_model: NormativePhaseModel
    transition_model: TransitionModel
    action_model: ActionBaselineModel
    n_sessions: int = 0
    n_phase_segments: int = 0
    n_transitions: int = 0
    metadata: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save calibration profile to JSON."""
        data = {
            "phase_model": self.phase_model.to_dict(),
            "transition_model": self.transition_model.to_dict(),
            "action_model": self.action_model.to_dict(),
            "n_sessions": self.n_sessions,
            "n_phase_segments": self.n_phase_segments,
            "n_transitions": self.n_transitions,
            "metadata": self.metadata,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        """Load calibration profile from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            phase_model=NormativePhaseModel.from_dict(data["phase_model"]),
            transition_model=TransitionModel.from_dict(data["transition_model"]),
            action_model=ActionBaselineModel.from_dict(data["action_model"]),
            n_sessions=data.get("n_sessions", 0),
            n_phase_segments=data.get("n_phase_segments", 0),
            n_transitions=data.get("n_transitions", 0),
            metadata=data.get("metadata", {}),
        )


# =========================================================================
# CalibrationPipeline — orchestrates fitting from validation traces
# =========================================================================

class CalibrationPipeline:
    """Fits normative baselines from validation traces.

    Usage:
        pipeline = CalibrationPipeline()
        profile = pipeline.fit_from_sessions(trace_events_list)
        profile.save("baselines.json")
    """

    def fit_from_sessions(
        self,
        session_events: list[list[TraceEvent]],
    ) -> CalibrationProfile:
        """Fit all three models from a list of session event sequences.

        Args:
            session_events: List of event lists, one per session.

        Returns:
            CalibrationProfile with fitted models.
        """
        all_phase_stats: list[PhaseStats] = []
        all_transitions: list[tuple[str, str]] = []
        phase_tool_sequences: dict[str, list[list[str]]] = {}

        for events in session_events:
            segments = extract_phase_segments(events)

            # Phase stats
            for phase_name, phase_events in segments:
                stats = compute_phase_stats(phase_name, phase_events)
                all_phase_stats.append(stats)

                # Tool sequences for action model
                tools = [
                    e.tool or "unknown"
                    for e in phase_events
                    if e.type == "tool_call" and e.tool
                ]
                if tools:
                    if phase_name not in phase_tool_sequences:
                        phase_tool_sequences[phase_name] = []
                    phase_tool_sequences[phase_name].append(tools)

            # Transitions
            for i in range(len(segments) - 1):
                from_p = segments[i][0]
                to_p = segments[i + 1][0]
                all_transitions.append((from_p, to_p))

        # Fit models
        phase_model = NormativePhaseModel()
        phase_model.fit(all_phase_stats)

        transition_model = TransitionModel()
        transition_model.fit(all_transitions)

        action_model = ActionBaselineModel()
        action_model.fit(phase_tool_sequences)

        return CalibrationProfile(
            phase_model=phase_model,
            transition_model=transition_model,
            action_model=action_model,
            n_sessions=len(session_events),
            n_phase_segments=len(all_phase_stats),
            n_transitions=len(all_transitions),
        )

    def fit_from_traces_path(
        self,
        traces_path: str | Path,
        splits_file: Optional[str | Path] = None,
        split: str = "validation",
        min_lines: int = 10,
    ) -> CalibrationProfile:
        """Fit from Claude Code sessions using the split manager.

        Args:
            traces_path: Root path to Claude Code sessions.
            splits_file: Path to splits.json (optional, uses all if None).
            split: Which split to fit on (default: validation).
            min_lines: Skip sessions with fewer lines.
        """
        from agentdiag.adapters.claude_code import ClaudeCodeExtractor
        from agentdiag.splits import SplitManager

        extractor = ClaudeCodeExtractor()
        all_sessions = extractor.discover(traces_path, min_lines=min_lines)

        # Filter to specified split if splits file exists
        if splits_file and Path(splits_file).exists():
            sm = SplitManager(splits_file)
            valid_ids = set(sm.get_traces(split))
            sessions = [s for s in all_sessions if s.session_id in valid_ids]
        else:
            sessions = all_sessions

        # Parse all sessions
        session_events = []
        for session in sessions:
            try:
                events = extractor.parse_session(session)
                if events:
                    session_events.append(events)
            except Exception:
                continue  # skip unparseable sessions

        if not session_events:
            # Return empty profile
            return CalibrationProfile(
                phase_model=NormativePhaseModel(),
                transition_model=TransitionModel(),
                action_model=ActionBaselineModel(),
            )

        profile = self.fit_from_sessions(session_events)
        profile.metadata = {
            "traces_path": str(traces_path),
            "split": split,
            "n_sessions_discovered": len(sessions),
            "n_sessions_parsed": len(session_events),
        }
        return profile

    def fit_from_annotated(
        self,
        traces_path: str | Path,
        annotation_ledger_path: str | Path,
        label_filter: str = "trainable",
        min_lines: int = 10,
    ) -> CalibrationProfile:
        """Fit baselines from annotation-filtered sessions only.

        Only sessions with trusted labels (per label_filter) are used
        for threshold calibration. This prevents fitting on unverified
        data.

        Args:
            traces_path: Root path to Claude Code sessions.
            annotation_ledger_path: Path to annotation_ledger.jsonl.
            label_filter: "trainable" (adjudicated + human_reviewed) or
                         "gold" (adjudicated only).
            min_lines: Skip sessions with fewer lines.

        Returns:
            CalibrationProfile fitted on trusted sessions only.
        """
        from agentdiag.adapters.claude_code import ClaudeCodeExtractor
        from agentdiag.annotation_store import AnnotationLedger

        # Load annotation ledger and get allowed session IDs
        ledger = AnnotationLedger(annotation_ledger_path)
        if label_filter == "gold":
            records = ledger.get_gold_annotations()
        else:
            records = ledger.get_trainable_annotations()

        allowed_sessions = {r.effective_session_id for r in records}
        # Also include 8-char prefixes for matching
        allowed_prefixes = {sid[:8] for sid in allowed_sessions}

        extractor = ClaudeCodeExtractor()
        all_sessions = extractor.discover(traces_path, min_lines=min_lines)

        # Filter to annotated sessions
        sessions = [
            s for s in all_sessions
            if s.session_id in allowed_sessions
            or s.session_id[:8] in allowed_prefixes
        ]

        # Parse
        session_events = []
        for session in sessions:
            try:
                events = extractor.parse_session(session)
                if events:
                    session_events.append(events)
            except Exception:
                continue

        if not session_events:
            return CalibrationProfile(
                phase_model=NormativePhaseModel(),
                transition_model=TransitionModel(),
                action_model=ActionBaselineModel(),
            )

        profile = self.fit_from_sessions(session_events)
        profile.metadata = {
            "traces_path": str(traces_path),
            "label_filter": label_filter,
            "n_annotated_sessions": len(allowed_sessions),
            "n_sessions_parsed": len(session_events),
        }
        return profile
