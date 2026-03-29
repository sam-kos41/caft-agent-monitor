"""Cognitive Load Monitor (CLM) — real-time information processing model.

Models the LLM agent's cognitive state across the 6 stages of the human
Information Processing (IP) model:

  Perception → Attention → Working Memory → Decision Making → Action → Feedback

The core insight: an LLM coding agent IS an information processing system.
Unlike humans (where you need EEG/fMRI), every stage of the LLM's processing
is observable in the JSONL trace — the trace IS the brain scan.

Components:
  - CognitiveState: 18+ normalized metrics across 6 IP stages
  - CognitiveStateTracker: O(1) incremental updater (no re-scanning)
  - WorkingMemoryModel: Context window utilization estimator
  - DecisionPointLog: reasoning → action → outcome triplets
  - MetacognitiveState: Models the detection system as metacognition

Usage::

    engine = MonitorEngine(goal="Fix bug", cognitive=True)
    for event in events:
        engine.push(event)
    cog = engine.cognitive_state  # CognitiveState snapshot
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from agentdiag.hta import Phase
from agentdiag.models import TraceEvent


# ── Gathering tools (from hta.py) — used for perception/attention tracking ──

_GATHERING_TOOLS = {
    "read_file", "search_docs", "web_search", "search_codebase",
    "list_files", "glob", "grep", "find", "cat", "head", "tail",
    "read", "fetch", "get", "list", "ls", "describe",
}

_PLANNING_TYPES = {"reasoning", "planning", "thinking"}


# ── Phase → IP stage mapping ────────────────────────────────────────────────

_PHASE_TO_IP_STAGE = {
    Phase.IDLE: "perception",
    Phase.GATHERING: "perception",
    Phase.PLANNING: "decision_making",
    Phase.EXECUTING: "action",
    Phase.VERIFYING: "feedback",
    Phase.DELIVERING: "feedback",
}


# ── SymbolStream — unified IT computation over a symbol distribution ───────
#
# A single class that accepts arbitrary string symbols via push() and provides
# multiple information-theoretic views: entropy, mutual information, LZ
# compression ratio, surprisal, and KL divergence.  One deque, one count dict,
# multiple views.  Never inspects what symbols represent — the
# ObservableEvent.to_symbol() contract is the boundary.


class SymbolStream:
    """Unified information-theoretic computation over a symbol distribution.

    Accepts string symbols via push() and maintains sliding-window counts
    that drive all IT measures.  Each instance tracks ONE signal type.
    The EventRouter (below) creates separate instances per signal type
    (tool, read, write, memory, action) and dispatches events.

    All windows default to the same size so sparkline x-axes align.
    LZ compression uses a larger window (150) for better phrase detection.
    """

    def __init__(self, window: int = 50, lz_window: int = 150) -> None:
        # Primary sliding window + frequency counts (shared by entropy, surprisal)
        self._window: deque[str] = deque(maxlen=window)
        self._counts: dict[str, int] = {}
        self._total: int = 0

        # Bigram tracking for MI
        self._bigrams: deque[tuple[str, str]] = deque(maxlen=window)
        self._joint: dict[tuple[str, str], int] = {}
        self._margin_x: dict[str, int] = {}
        self._margin_y: dict[str, int] = {}
        self._prev_symbol: Optional[str] = None
        self._bigram_total: int = 0

        # LZ compression uses a larger window
        self._lz_window: deque[str] = deque(maxlen=lz_window)

        # KL divergence: baseline locks after BASELINE_WINDOW pushes
        self._baseline_counts: dict[str, int] = {}
        self._baseline_total: int = 0
        self._baseline_locked: bool = False
        self._kl_current: deque[str] = deque(maxlen=window)
        self._kl_current_counts: dict[str, int] = {}
        self._all_symbols_seen: set[str] = set()
        self._baseline_window = 100

        # Histories (last 200 points for sparklines)
        self._entropy_history: deque[float] = deque(maxlen=200)
        self._mi_history: deque[float] = deque(maxlen=200)
        self._compression_history: deque[float] = deque(maxlen=200)
        self._surprisal_history: deque[tuple[int, float]] = deque(maxlen=200)
        self._kl_history: deque[float] = deque(maxlen=200)

        self._step: int = 0

    def push(self, symbol: str, step: Optional[int] = None) -> None:
        """Push a symbol and update all IT measures."""
        self._step = step if step is not None else self._step + 1
        self._all_symbols_seen.add(symbol)

        # ── Update primary window + counts ──
        if len(self._window) == self._window.maxlen:
            evicted = self._window[0]
            self._counts[evicted] -= 1
            if self._counts[evicted] == 0:
                del self._counts[evicted]
            self._total -= 1

        self._window.append(symbol)
        self._counts[symbol] = self._counts.get(symbol, 0) + 1
        self._total += 1

        # ── Entropy ──
        self._entropy_history.append(round(self._compute_entropy(), 4))

        # ── Surprisal (Laplace smoothing) ──
        k = len(self._all_symbols_seen)
        p = (self._counts.get(symbol, 0)) / (self._total + k)
        bits = -math.log2(max(p, 1e-12))
        self._surprisal_history.append((self._step, round(bits, 4)))

        # ── MI (bigrams) ──
        if self._prev_symbol is not None:
            bigram = (self._prev_symbol, symbol)
            if len(self._bigrams) == self._bigrams.maxlen:
                old = self._bigrams[0]
                self._joint[old] -= 1
                if self._joint[old] == 0:
                    del self._joint[old]
                self._margin_x[old[0]] -= 1
                if self._margin_x[old[0]] == 0:
                    del self._margin_x[old[0]]
                self._margin_y[old[1]] -= 1
                if self._margin_y[old[1]] == 0:
                    del self._margin_y[old[1]]
                self._bigram_total -= 1

            self._bigrams.append(bigram)
            self._joint[bigram] = self._joint.get(bigram, 0) + 1
            self._margin_x[bigram[0]] = self._margin_x.get(bigram[0], 0) + 1
            self._margin_y[bigram[1]] = self._margin_y.get(bigram[1], 0) + 1
            self._bigram_total += 1
        self._prev_symbol = symbol
        self._mi_history.append(round(self._compute_mi(), 4))

        # ── LZ compression (larger window) ──
        self._lz_window.append(symbol)
        self._compression_history.append(round(self._compute_lz(), 4))

        # ── KL divergence ──
        if not self._baseline_locked:
            self._baseline_counts[symbol] = self._baseline_counts.get(symbol, 0) + 1
            self._baseline_total += 1
            if self._baseline_total >= self._baseline_window:
                self._baseline_locked = True
            self._kl_history.append(0.0)
        else:
            if len(self._kl_current) == self._kl_current.maxlen:
                ev = self._kl_current[0]
                self._kl_current_counts[ev] -= 1
                if self._kl_current_counts[ev] == 0:
                    del self._kl_current_counts[ev]
            self._kl_current.append(symbol)
            self._kl_current_counts[symbol] = self._kl_current_counts.get(symbol, 0) + 1
            self._kl_history.append(round(self._compute_kl(), 4))

    # ── Public views ──

    def entropy(self) -> float:
        return self._entropy_history[-1] if self._entropy_history else 0.0

    def mi(self) -> float:
        return self._mi_history[-1] if self._mi_history else 0.0

    def compression(self) -> float:
        return self._compression_history[-1] if self._compression_history else 0.0

    def surprisal(self, symbol: Optional[str] = None) -> float:
        """Last surprisal, or compute for a specific symbol."""
        if symbol is not None and self._total > 0:
            k = len(self._all_symbols_seen)
            p = (self._counts.get(symbol, 0) + 1) / (self._total + k)
            return round(-math.log2(p), 4)
        return self._surprisal_history[-1][1] if self._surprisal_history else 0.0

    def kl_divergence(self) -> float:
        return self._kl_history[-1] if self._kl_history else 0.0

    @property
    def entropy_history(self) -> list[float]:
        return list(self._entropy_history)

    @property
    def mi_history(self) -> list[float]:
        return list(self._mi_history)

    @property
    def compression_history(self) -> list[float]:
        return list(self._compression_history)

    @property
    def surprisal_history(self) -> list[tuple[int, float]]:
        return list(self._surprisal_history)

    @property
    def kl_history(self) -> list[float]:
        return list(self._kl_history)

    @property
    def event_count(self) -> int:
        return len(self._entropy_history)

    # ── Private computation methods ──

    def _compute_entropy(self) -> float:
        if self._total == 0:
            return 0.0
        h = 0.0
        for c in self._counts.values():
            if c > 0:
                p = c / self._total
                h -= p * math.log2(p)
        return h

    def _compute_mi(self) -> float:
        if self._bigram_total == 0:
            return 0.0
        mi = 0.0
        for (x, y), jc in self._joint.items():
            p_xy = jc / self._bigram_total
            p_x = self._margin_x.get(x, 0) / self._bigram_total
            p_y = self._margin_y.get(y, 0) / self._bigram_total
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * math.log2(p_xy / (p_x * p_y))
        return max(mi, 0.0)

    _LZ_MIN_SYMBOLS = 20  # minimum symbols before LZ is meaningful

    def _compute_lz(self) -> float:
        seq = list(self._lz_window)
        n = len(seq)
        if n < self._LZ_MIN_SYMBOLS:
            return 1.0  # not enough data for meaningful compression
        # LZ76 factorisation on available symbols (adaptive — doesn't
        # require full window to fill)
        phrases = 0
        i = 0
        while i < n:
            length = 1
            while i + length <= n:
                substr = seq[i:i + length]
                found = False
                for j in range(i):
                    if seq[j:j + length] == substr and j + length <= i:
                        found = True
                        break
                if found:
                    length += 1
                else:
                    break
            phrases += 1
            i += max(length - 1, 1)
        k = len(set(seq))
        if k <= 1:
            return 0.0
        # Normalise by ACTUAL buffer length, not max window size
        expected = n / math.log2(k)
        return min(phrases / max(expected, 1.0), 1.0)

    def _compute_kl(self) -> float:
        vocab = self._all_symbols_seen
        k = len(vocab)
        if k == 0:
            return 0.0
        current_total = len(self._kl_current)
        kl = 0.0
        for s in vocab:
            p = (self._kl_current_counts.get(s, 0) + 1) / (current_total + k)
            q = (self._baseline_counts.get(s, 0) + 1) / (self._baseline_total + k)
            if p > 0 and q > 0:
                kl += p * math.log2(p / q)
        return max(kl, 0.0)

    def to_dict(self) -> dict:
        """Serialise current state for a single stream."""
        return {
            "entropy": self.entropy(),
            "mi": self.mi(),
            "compression": self.compression(),
            "last_surprisal": self.surprisal(),
            "kl_divergence": self.kl_divergence(),
            "entropy_history": self.entropy_history,
            "mi_history": self.mi_history,
            "compression_history": self.compression_history,
            "surprisal_history": self.surprisal_history,
            "kl_history": self.kl_history,
            "event_count": self.event_count,
        }


# ── EventRouter — dispatches ObservableEvents to per-signal SymbolStreams ──


class FeedbackActionMI:
    """Cross-stream MI between feedback events and subsequent actions.

    Measures whether the agent adapts its behavior after receiving feedback
    (test results, error messages, evaluator grades).  High MI = agent
    responds to feedback.  Low MI = agent ignores it.

    Wickens: feedback → sensory processing → response selection loop.
    """

    def __init__(self, response_window: int = 5) -> None:
        self._response_window = response_window
        self._feedback_action_pairs: deque[tuple[str, str]] = deque(maxlen=100)
        self._joint: dict[tuple[str, str], int] = {}
        self._margin_fb: dict[str, int] = {}
        self._margin_act: dict[str, int] = {}
        self._total: int = 0
        self._pending_feedback: Optional[str] = None
        self._pending_countdown: int = 0
        self._history: deque[float] = deque(maxlen=200)

    def record_feedback(self, symbol: str) -> None:
        """Record a feedback event. Next N actions will be paired with it."""
        self._pending_feedback = symbol
        self._pending_countdown = self._response_window

    def record_action(self, symbol: str, step: int) -> None:
        """Record an action. If within response window of feedback, create pair."""
        if self._pending_feedback is not None and self._pending_countdown > 0:
            pair = (self._pending_feedback, symbol)

            if len(self._feedback_action_pairs) == self._feedback_action_pairs.maxlen:
                old = self._feedback_action_pairs[0]
                self._joint[old] = self._joint.get(old, 1) - 1
                if self._joint[old] <= 0:
                    del self._joint[old]
                self._margin_fb[old[0]] = self._margin_fb.get(old[0], 1) - 1
                if self._margin_fb[old[0]] <= 0:
                    del self._margin_fb[old[0]]
                self._margin_act[old[1]] = self._margin_act.get(old[1], 1) - 1
                if self._margin_act[old[1]] <= 0:
                    del self._margin_act[old[1]]
                self._total -= 1

            self._feedback_action_pairs.append(pair)
            self._joint[pair] = self._joint.get(pair, 0) + 1
            self._margin_fb[pair[0]] = self._margin_fb.get(pair[0], 0) + 1
            self._margin_act[pair[1]] = self._margin_act.get(pair[1], 0) + 1
            self._total += 1

            self._pending_countdown -= 1
            if self._pending_countdown <= 0:
                self._pending_feedback = None

            self._history.append(round(self._compute(), 4))

    def mi(self) -> float:
        return self._history[-1] if self._history else 0.0

    @property
    def history(self) -> list[float]:
        return list(self._history)

    def _compute(self) -> float:
        if self._total < 2:
            return 0.0
        mi = 0.0
        for (fb, act), jc in self._joint.items():
            p_xy = jc / self._total
            p_x = self._margin_fb.get(fb, 0) / self._total
            p_y = self._margin_act.get(act, 0) / self._total
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * math.log2(p_xy / (p_x * p_y))
        return max(mi, 0.0)


class MemoryFlowTracker:
    """Tracks consolidation (WM→LTM) and retrieval (LTM→WM) rates.

    Wickens model: WM interacts bidirectionally with LTM.
    - Consolidation = writes to persistent storage (file writes, memory stores,
      git commits) that are NOT immediately read back.
    - Retrieval = reads of previously-written files or OpenViking loads.
    """

    def __init__(self, window: int = 30) -> None:
        self._written_resources: set[str] = set()
        self._consolidation_events: deque[int] = deque(maxlen=window)
        self._retrieval_events: deque[int] = deque(maxlen=window)
        self._consolidation_history: deque[float] = deque(maxlen=200)
        self._retrieval_history: deque[float] = deque(maxlen=200)
        self._total_events: int = 0

    def record_write(self, resource: str, step: int) -> None:
        """Agent wrote to persistent storage."""
        self._written_resources.add(resource)
        self._consolidation_events.append(step)
        self._total_events += 1
        self._update_rates()

    def record_read(self, resource: str, step: int) -> None:
        """Agent read from storage. If it reads own prior write, that's retrieval."""
        self._total_events += 1
        if resource in self._written_resources:
            self._retrieval_events.append(step)
        self._update_rates()

    def record_memory_load(self, step: int) -> None:
        """OpenViking memory load = retrieval from LTM."""
        self._retrieval_events.append(step)
        self._total_events += 1
        self._update_rates()

    def record_memory_store(self, step: int) -> None:
        """OpenViking memory store = consolidation to LTM."""
        self._consolidation_events.append(step)
        self._total_events += 1
        self._update_rates()

    def _update_rates(self) -> None:
        n = max(self._total_events, 1)
        self._consolidation_history.append(
            round(len(self._consolidation_events) / n, 4)
        )
        self._retrieval_history.append(
            round(len(self._retrieval_events) / n, 4)
        )

    def consolidation_rate(self) -> float:
        return self._consolidation_history[-1] if self._consolidation_history else 0.0

    def retrieval_rate(self) -> float:
        return self._retrieval_history[-1] if self._retrieval_history else 0.0

    @property
    def consolidation_history(self) -> list[float]:
        return list(self._consolidation_history)

    @property
    def retrieval_history(self) -> list[float]:
        return list(self._retrieval_history)


class EventRouter:
    """Routes ObservableEvents to per-signal SymbolStreams aligned to Wickens IP model.

    Streams map to Wickens stages:
      input_stream   → Sensory processing (input entropy)
      read_stream    → Perceptual processing (read entropy, selectivity)
      action_stream  → Response selection (action entropy, MI, compression)
      tool_stream    → Response execution (tool-specific patterns)
      write_stream   → Response execution (output patterns)
      memory_stream  → Working memory / LTM operations
      feedback_stream → System environment feedback

    Cross-stream measures:
      feedback_action_mi → Feedback→action MI (does agent adapt to feedback?)
      memory_flow       → Consolidation rate (WM→LTM) and retrieval rate (LTM→WM)

    Attention resources = action_stream.entropy() normalized to [0,1]
    """

    _FEEDBACK_TOOLS = {"pytest", "test", "lint", "check", "verify"}
    _FEEDBACK_PATTERNS = [
        "pytest", "python -m pytest", "python -m unittest", "python -m doctest",
        "npm test", "npm run test", "jest", "mocha", "vitest",
        "cargo test", "go test", "make test", "make check",
        "rspec", "bundle exec rspec", "phpunit",
        "lint", "eslint", "pylint", "flake8", "mypy", "tsc --noemit",
        "ruff", "black --check", "isort --check",
        "grep -r", "git diff", "git status",
        "curl ", "wget ",
    ]
    _INPUT_EVENT_TYPES = {"file_read", "tool_call", "shell_command", "memory_load"}

    def __init__(self, window: int = 50) -> None:
        # Per-signal streams (Wickens-aligned)
        self.input_stream = SymbolStream(window=window, lz_window=150)   # sensory
        self.read_stream = SymbolStream(window=window, lz_window=150)    # perceptual
        self.action_stream = SymbolStream(window=window, lz_window=150)  # response selection
        self.tool_stream = SymbolStream(window=window, lz_window=150)    # response execution
        self.write_stream = SymbolStream(window=window, lz_window=150)   # response execution
        self.memory_stream = SymbolStream(window=window, lz_window=150)  # WM/LTM
        self.feedback_stream = SymbolStream(window=window, lz_window=150)  # feedback

        # Cross-stream Wickens measures
        self.feedback_action_mi = FeedbackActionMI(response_window=5)
        self.memory_flow = MemoryFlowTracker(window=30)

        # Input token tracking for sensory stage
        self._input_tokens_window: deque[int] = deque(maxlen=window)
        self._total_input_tokens: int = 0

        # Execution tracking
        self._success_window: deque[bool] = deque(maxlen=window)

        self.phase_markers: deque[tuple[int, str]] = deque(maxlen=200)
        self.memory_events: deque[dict] = deque(maxlen=200)
        self.evaluation_events: deque[dict] = deque(maxlen=200)
        self.feedback_events: deque[dict] = deque(maxlen=200)

    def _is_feedback_event(self, event: Any, tool: str, etype_val: str) -> bool:
        """Check if event represents environmental feedback (Wickens feedback stage).

        Checks tool name, event type, command content, and evaluation signals.
        Catches Bash commands that run test suites, linters, and check commands.
        """
        if tool in self._FEEDBACK_TOOLS:
            return True
        if etype_val == "error":
            return True
        if hasattr(event, "is_evaluation_signal") and event.is_evaluation_signal():
            return True
        # Check full command content for test/lint patterns
        cmd = (getattr(event, "target_path", None) or "")
        meta = getattr(event, "metadata", None) or {}
        if isinstance(meta, dict):
            cmd = cmd + " " + (meta.get("command", ""))
        cmd_lower = cmd.lower()
        return any(pattern in cmd_lower for pattern in self._FEEDBACK_PATTERNS)

    def process_event(self, event: Any) -> None:
        """Route an ObservableEvent to Wickens-aligned streams."""
        if event.is_phase_marker():
            self.phase_markers.append((event.step, event.to_symbol()))
            return

        symbol = event.to_symbol()
        step = event.step
        etype = getattr(event, "event_type", None)
        etype_val = etype.value if etype is not None else ""
        tool = (getattr(event, "tool_name", None) or "").lower()

        # ── Aggregate action stream (response selection) ──
        self.action_stream.push(symbol, step)

        # ── Sensory: anything that produces input to the agent ──
        if etype_val in self._INPUT_EVENT_TYPES:
            self.input_stream.push(symbol, step)
            tokens = getattr(event, "output_tokens", None) or 0
            self._input_tokens_window.append(tokens)
            self._total_input_tokens += tokens

        # ── Perceptual: reads (what the agent selectively attends to) ──
        if etype_val == "file_read":
            self.read_stream.push(symbol, step)
            path = getattr(event, "target_path", None) or symbol
            self.memory_flow.record_read(path, step)

        # ── Response execution: tool calls and writes ──
        if etype_val in ("tool_call", "shell_command"):
            self.tool_stream.push(symbol, step)
            # Track success for execution efficiency
            success = getattr(event, "success", True)
            if success is not None:
                self._success_window.append(bool(success))

        if etype_val == "file_write":
            self.write_stream.push(symbol, step)
            path = getattr(event, "target_path", None) or symbol
            self.memory_flow.record_write(path, step)

        # ── Working memory / LTM operations ──
        if etype_val in ("memory_load", "memory_store", "memory_evict",
                         "memory_tier_escalation"):
            self.memory_stream.push(symbol, step)
            if etype_val == "memory_load":
                self.memory_flow.record_memory_load(step)
            elif etype_val == "memory_store":
                self.memory_flow.record_memory_store(step)

        if event.is_memory_operation():
            self.memory_events.append(event.to_dict())

        # ── Feedback: tests, errors, evaluations ──
        is_feedback = self._is_feedback_event(event, tool, etype_val)
        if is_feedback:
            self.feedback_stream.push(symbol, step)
            # Use richer symbol for feedback-action MI (include command + success)
            success = getattr(event, "success", True)
            cmd_hint = (getattr(event, "target_path", None) or "")[:40]
            fb_symbol = f"fb:{tool}:{cmd_hint}:{'ok' if success else 'err'}"
            self.feedback_action_mi.record_feedback(fb_symbol)
            self.feedback_events.append(event.to_dict())

        if hasattr(event, "is_evaluation_signal") and event.is_evaluation_signal():
            self.evaluation_events.append(event.to_dict())

        # ── Feedback-action MI: every non-feedback action is a potential response ──
        if not is_feedback:
            self.feedback_action_mi.record_action(symbol, step)

    # ── Wickens-aligned state accessors ──

    def attention_resource(self) -> float:
        """Attention resources = action entropy normalized to [0,1].

        Max possible entropy for a window of N with K symbols is log2(K).
        Normalize by log2(min(window_size, observed_symbols)).
        """
        h = self.action_stream.entropy()
        k = len(self.action_stream._counts) or 1
        max_h = math.log2(k) if k > 1 else 1.0
        return min(h / max_h, 1.0) if max_h > 0 else 0.0

    def execution_efficiency(self) -> float:
        """Success rate of recent tool calls."""
        if not self._success_window:
            return 1.0
        return sum(self._success_window) / len(self._success_window)

    def input_rate(self) -> float:
        """Average input tokens per event in the current window."""
        if not self._input_tokens_window:
            return 0.0
        return sum(self._input_tokens_window) / len(self._input_tokens_window)

    def to_dict(self) -> dict:
        """Serialise all streams for WebSocket transport."""
        act = self.action_stream
        return {
            # Legacy top-level keys (backward compat with existing frontend)
            "tool_entropy": act.entropy(),
            "read_entropy": self.read_stream.entropy(),
            "action_mi": act.mi(),
            "compression_ratio": act.compression(),
            "last_surprisal": act.surprisal(),
            "kl_divergence": act.kl_divergence(),
            "tool_entropy_history": act.entropy_history,
            "read_entropy_history": self.read_stream.entropy_history,
            "action_mi_history": act.mi_history,
            "compression_history": act.compression_history,
            "surprisal_history": act.surprisal_history,
            "kl_divergence_history": act.kl_history,

            # Wickens-aligned per-stage metrics
            "wickens": {
                "sensory": {
                    "input_entropy": self.input_stream.entropy(),
                    "input_entropy_history": self.input_stream.entropy_history,
                    "input_rate": round(self.input_rate(), 1),
                    "input_tokens_total": self._total_input_tokens,
                },
                "perceptual": {
                    "read_entropy": self.read_stream.entropy(),
                    "read_entropy_history": self.read_stream.entropy_history,
                    "read_events": self.read_stream.event_count,
                    "focus": round(1.0 - min(self.read_stream.entropy() / 3.0, 1.0), 3),
                },
                "attention": {
                    "resource": round(self.attention_resource(), 3),
                    "action_entropy": act.entropy(),
                },
                "working_memory": {
                    "memory_entropy": self.memory_stream.entropy(),
                    "memory_entropy_history": self.memory_stream.entropy_history,
                    "consolidation_rate": self.memory_flow.consolidation_rate(),
                    "retrieval_rate": self.memory_flow.retrieval_rate(),
                    "consolidation_history": self.memory_flow.consolidation_history,
                    "retrieval_history": self.memory_flow.retrieval_history,
                },
                "ltm": {
                    "stored_items": len(self.memory_flow._written_resources),
                    "retrieval_rate": self.memory_flow.retrieval_rate(),
                    "retrieval_action_mi": self.feedback_action_mi.mi(),
                },
                "response_selection": {
                    "action_mi": act.mi(),
                    "action_mi_history": act.mi_history,
                    "coherence": round(min(act.mi() / 2.0, 1.0), 3),
                    "surprisal": act.surprisal(),
                    "surprisal_history": act.surprisal_history,
                },
                "response_execution": {
                    "compression": act.compression(),
                    "compression_history": act.compression_history,
                    "efficiency": round(self.execution_efficiency(), 3),
                    "tool_entropy": self.tool_stream.entropy(),
                    "tool_entropy_history": self.tool_stream.entropy_history,
                },
                "feedback": {
                    "feedback_action_mi": self.feedback_action_mi.mi(),
                    "feedback_action_mi_history": self.feedback_action_mi.history,
                    "feedback_events": self.feedback_stream.event_count,
                    "adaptation": round(min(self.feedback_action_mi.mi() / 1.5, 1.0), 3),
                },
            },

            # Per-stream summaries (for advanced/debug views)
            "streams": {
                "action": act.to_dict(),
                "input": self.input_stream.to_dict(),
                "read": self.read_stream.to_dict(),
                "tool": self.tool_stream.to_dict(),
                "write": self.write_stream.to_dict(),
                "memory": self.memory_stream.to_dict(),
                "feedback": self.feedback_stream.to_dict(),
            },

            # Markers and events
            "phase_markers": list(self.phase_markers),
            "memory_events": list(self.memory_events),
            "evaluation_events": list(self.evaluation_events),
            "feedback_events": list(self.feedback_events),
        }


# ── TraceEvent → ObservableEvent bridge ────────────────────────────────────


def trace_event_to_observable(event: TraceEvent) -> Any:
    """Bridge old TraceEvent to ObservableEvent contract.

    Used when consuming existing Claude Code traces that haven't been
    updated to emit ObservableEvent directly.  Returns an ObservableEvent
    with tool-level fields populated.
    """
    from agentdiag.observable import ObservableEvent, EventType

    tool = (event.tool or event.type or "unknown").lower()

    if event.type in ("reasoning", "planning", "thinking"):
        return ObservableEvent(
            step=event.step,
            timestamp=event.timestamp or 0.0,
            event_type=EventType.TOOL_CALL,
            tool_name=event.type,
            duration_ms=event.latency_ms,
            input_tokens=event.tokens_in,
            output_tokens=event.tokens_out,
        )

    if tool in _GATHERING_TOOLS:
        return ObservableEvent(
            step=event.step,
            timestamp=event.timestamp or 0.0,
            event_type=EventType.FILE_READ,
            tool_name=event.tool,
            target_path=event.target_path,
            output_tokens=event.tokens_out,
            duration_ms=event.latency_ms,
        )

    if tool in ("write_file", "edit", "write", "edit_file"):
        return ObservableEvent(
            step=event.step,
            timestamp=event.timestamp or 0.0,
            event_type=EventType.FILE_WRITE,
            tool_name=event.tool,
            target_path=event.target_path,
            input_tokens=event.tokens_in,
            duration_ms=event.latency_ms,
        )

    if tool in ("bash", "shell", "run_code", "terminal"):
        return ObservableEvent(
            step=event.step,
            timestamp=event.timestamp or 0.0,
            event_type=EventType.SHELL_COMMAND,
            tool_name=event.tool,
            target_path=event.target_path,
            duration_ms=event.latency_ms,
            input_tokens=event.tokens_in,
            output_tokens=event.tokens_out,
        )

    return ObservableEvent(
        step=event.step,
        timestamp=event.timestamp or 0.0,
        event_type=EventType.TOOL_CALL,
        tool_name=event.tool or event.type,
        duration_ms=event.latency_ms,
        input_tokens=event.tokens_in,
        output_tokens=event.tokens_out,
    )


# ── CognitiveState ──────────────────────────────────────────────────────────


@dataclass
class CognitiveState:
    """Real-time model of agent's information processing state.

    Each field represents a computable metric for one IP stage.
    All values are 0.0-1.0 normalized (suitable for bar chart rendering).
    """

    # Stage 1: Perception — how much information has been gathered
    perception_breadth: float = 0.0
    perception_depth: float = 0.0
    perception_recency: float = 0.0

    # Stage 2: Attention — focus vs. breadth
    attention_diversity: float = 0.0
    attention_focus: float = 0.0
    attention_tunnel_risk: float = 0.0

    # Stage 3: Working Memory — estimated context utilization
    memory_utilization: float = 0.0
    memory_recency_bias: float = 0.0
    memory_at_risk_items: int = 0

    # Stage 4: Decision Making — deliberation patterns
    decision_deliberation: float = 0.0
    decision_options_explored: float = 0.0
    decision_latency_trend: float = 0.0

    # Stage 5: Action — execution patterns
    action_diversity: float = 0.0
    action_success_rate: float = 1.0
    action_repetition_risk: float = 0.0

    # Stage 6: Feedback — verification patterns
    feedback_verify_ratio: float = 0.0
    feedback_error_response: float = 1.0
    feedback_loop_closed: float = 0.0

    # Aggregate
    overall_cognitive_load: float = 0.0

    # Metadata
    last_updated_step: int = 0
    active_ip_stage: str = "perception"
    bottleneck_stage: str = ""  # stage with highest load

    def _stage_load(self, stage: str) -> float:
        """Compute single load value for an IP stage."""
        if stage == "perception":
            return round(1.0 - (0.4 * self.perception_recency + 0.3 * self.perception_breadth + 0.3 * self.perception_depth), 3)
        elif stage == "attention":
            return round(0.6 * self.attention_tunnel_risk + 0.4 * (1.0 - self.attention_diversity), 3)
        elif stage == "working_memory":
            return round(self.memory_utilization, 3)
        elif stage == "decision_making":
            return round(0.6 * self.decision_deliberation + 0.4 * max(self.decision_latency_trend, 0.0), 3)
        elif stage == "action":
            return round(0.5 * (1.0 - self.action_success_rate) + 0.5 * self.action_repetition_risk, 3)
        elif stage == "feedback":
            return round(0.5 * (1.0 - self.feedback_verify_ratio) + 0.5 * (1.0 - self.feedback_loop_closed), 3)
        return 0.0

    @staticmethod
    def _load_to_status(load: float, is_active: bool) -> str:
        """Convert a load value to a status string."""
        if load > 0.8:
            return "overloaded"
        if load > 0.6:
            return "at_risk"
        if is_active:
            return "active"
        if load > 0.1:
            return "normal"
        return "idle"

    def to_dict(self) -> dict:
        """JSON-serializable snapshot with per-stage load/status for UI."""
        stages = ["perception", "attention", "working_memory", "decision_making", "action", "feedback"]

        stage_dicts: dict[str, dict] = {}
        for stage in stages:
            load = self._stage_load(stage)
            is_active = (stage == self.active_ip_stage)
            status = self._load_to_status(load, is_active)

            if stage == "perception":
                details = {"breadth": round(self.perception_breadth, 3), "depth": round(self.perception_depth, 3), "recency": round(self.perception_recency, 3)}
            elif stage == "attention":
                details = {"diversity": round(self.attention_diversity, 3), "focus": round(self.attention_focus, 3), "tunnel_risk": round(self.attention_tunnel_risk, 3)}
            elif stage == "working_memory":
                details = {"utilization": round(self.memory_utilization, 3), "recency_bias": round(self.memory_recency_bias, 3), "at_risk_items": self.memory_at_risk_items}
            elif stage == "decision_making":
                details = {"deliberation": round(self.decision_deliberation, 3), "options_explored": round(self.decision_options_explored, 3), "latency_trend": round(self.decision_latency_trend, 3)}
            elif stage == "action":
                details = {"diversity": round(self.action_diversity, 3), "success_rate": round(self.action_success_rate, 3), "repetition_risk": round(self.action_repetition_risk, 3)}
            elif stage == "feedback":
                details = {"verify_ratio": round(self.feedback_verify_ratio, 3), "error_response": round(self.feedback_error_response, 3), "loop_closed": round(self.feedback_loop_closed, 3)}
            else:
                details = {}

            stage_dicts[stage] = {"load": load, "status": status, **details}

        return {
            **stage_dicts,
            "overall_cognitive_load": round(self.overall_cognitive_load, 3),
            "bottleneck_stage": self.bottleneck_stage,
            "active_ip_stage": self.active_ip_stage,
            "last_updated_step": self.last_updated_step,
        }


# ── WorkingMemoryModel ──────────────────────────────────────────────────────


@dataclass
class MemoryItem:
    """A single item in the working memory model."""
    resource_hash: str
    resource_name: str
    acquired_step: int
    last_accessed_step: int
    token_estimate: int
    access_count: int

    def retention_probability(self, current_step: int) -> float:
        """Estimated probability this item is still in active context.

        Based on recency and access consolidation.
        """
        steps_since = current_step - self.last_accessed_step
        # Recency component: decays over 50 steps
        recency = max(0.0, 1.0 - steps_since / 50.0)
        # Consolidation: more accesses = more robust (diminishing returns)
        consolidation = min(self.access_count / 5.0, 1.0)
        # Weighted combination
        return min(1.0, 0.6 * recency + 0.4 * consolidation)


class WorkingMemoryModel:
    """Estimates what information is in the LLM's active context.

    The LLM's context window has properties analogous to human working memory:
    - Limited capacity (token limit)
    - Recency bias (recent content is uncompressed)
    - Consolidation (repeatedly accessed info is more robust)
    - Decay (old, unreferenced info gets compressed/lost)
    """

    MAX_ITEMS = 200
    DECAY_RATE = 50
    MAX_CONTEXT_TOKENS = 200_000

    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}
        self._current_step: int = 0
        self._cumulative_tokens: int = 0

    def record_access(self, event: TraceEvent) -> None:
        """Record that a resource was accessed."""
        self._current_step = event.step
        self._cumulative_tokens += event.tokens_in + event.tokens_out

        resource_hash = event.output_hash or event.input_hash
        if not resource_hash:
            return

        resource_name = event.tool or event.type
        tokens = event.tokens_in + event.tokens_out

        if resource_hash in self._items:
            item = self._items[resource_hash]
            item.last_accessed_step = event.step
            item.access_count += 1
            item.token_estimate = max(item.token_estimate, tokens)
        else:
            # Evict oldest if at capacity
            if len(self._items) >= self.MAX_ITEMS:
                oldest_hash = min(
                    self._items,
                    key=lambda h: self._items[h].last_accessed_step,
                )
                del self._items[oldest_hash]

            self._items[resource_hash] = MemoryItem(
                resource_hash=resource_hash,
                resource_name=resource_name,
                acquired_step=event.step,
                last_accessed_step=event.step,
                token_estimate=tokens,
                access_count=1,
            )

    def get_active_items(self) -> list[MemoryItem]:
        """Items likely still in active context (accessed recently)."""
        return sorted(
            [i for i in self._items.values()
             if (self._current_step - i.last_accessed_step) <= self.DECAY_RATE],
            key=lambda i: i.last_accessed_step,
            reverse=True,
        )

    def get_at_risk_items(self) -> list[MemoryItem]:
        """Items probably lost from context (not accessed in DECAY_RATE steps)."""
        return sorted(
            [i for i in self._items.values()
             if (self._current_step - i.last_accessed_step) > self.DECAY_RATE],
            key=lambda i: i.last_accessed_step,
        )

    def estimate_utilization(self) -> float:
        """Estimated context window utilization (0.0-1.0)."""
        return min(self._cumulative_tokens / self.MAX_CONTEXT_TOKENS, 1.0)

    def to_dict(self) -> dict:
        """Serializable snapshot for API/WebSocket."""
        active = self.get_active_items()
        at_risk = self.get_at_risk_items()
        return {
            "total_items": len(self._items),
            "active_count": len(active),
            "at_risk_count": len(at_risk),
            "utilization": round(self.estimate_utilization(), 3),
            "cumulative_tokens": self._cumulative_tokens,
            "active_items": [
                {"hash": i.resource_hash[:8], "name": i.resource_name,
                 "step": i.last_accessed_step, "accesses": i.access_count}
                for i in active[:20]
            ],
            "at_risk_items": [
                {"hash": i.resource_hash[:8], "name": i.resource_name,
                 "step": i.last_accessed_step, "accesses": i.access_count}
                for i in at_risk[:10]
            ],
        }


# ── DecisionPointLog ────────────────────────────────────────────────────────


@dataclass
class DecisionPoint:
    """A single decision point: reasoning -> action -> outcome."""
    # The deliberation phase
    reasoning_steps: list[int] = field(default_factory=list)
    reasoning_duration_ms: float = 0.0

    # The decision
    action_step: int = 0
    action_tool: str = ""
    action_phase: str = ""

    # The outcome
    action_success: bool = True
    outcome_steps: list[int] = field(default_factory=list)
    outcome_changed_state: bool = False
    followed_by_verification: bool = False

    # Quality signals
    strategy_changed: bool = False

    def to_dict(self) -> dict:
        return {
            "reasoning_steps": self.reasoning_steps,
            "reasoning_duration_ms": round(self.reasoning_duration_ms, 1),
            "action_step": self.action_step,
            "action_tool": self.action_tool,
            "action_phase": self.action_phase,
            "action_success": self.action_success,
            "outcome_changed_state": self.outcome_changed_state,
            "followed_by_verification": self.followed_by_verification,
            "strategy_changed": self.strategy_changed,
        }


class DecisionPointLog:
    """Accumulates decision points from the event stream.

    A decision point begins when a reasoning/planning event occurs,
    and ends when the subsequent tool_call completes and we can
    observe its outcome.
    """

    def __init__(self) -> None:
        self._points: list[DecisionPoint] = []
        self._pending_reasoning: list[tuple[int, float]] = []  # (step, latency_ms)
        self._pending_action: Optional[DecisionPoint] = None
        self._last_output_hash: Optional[str] = None
        self._last_tool: str = ""
        self._steps_since_action: int = 0

    def push(self, event: TraceEvent, phase: Phase) -> Optional[DecisionPoint]:
        """Process an event. Returns a completed DecisionPoint if one closes."""
        completed = None

        if event.type in _PLANNING_TYPES:
            # Accumulate reasoning events
            self._pending_reasoning.append((event.step, event.latency_ms))

            # If we had a pending action, close it (next reasoning = new decision cycle)
            if self._pending_action is not None:
                completed = self._finalize_pending()

        elif event.type == "tool_call":
            # Close previous action if pending
            if self._pending_action is not None:
                self._pending_action.outcome_steps.append(event.step)
                # Check for verification
                if phase == Phase.VERIFYING:
                    self._pending_action.followed_by_verification = True
                # Check state change
                if event.output_hash and event.output_hash != self._last_output_hash:
                    self._pending_action.outcome_changed_state = True
                self._steps_since_action += 1

                # Close after 5 steps or if new reasoning starts
                if self._steps_since_action >= 5:
                    completed = self._finalize_pending()

            # No pending action — this tool_call IS the decision
            if self._pending_action is None:
                strategy_changed = (event.tool or "") != self._last_tool
                dp = DecisionPoint(
                    reasoning_steps=[s for s, _ in self._pending_reasoning],
                    reasoning_duration_ms=sum(l for _, l in self._pending_reasoning),
                    action_step=event.step,
                    action_tool=event.tool or event.type,
                    action_phase=phase.label,
                    action_success=event.success,
                    strategy_changed=strategy_changed,
                )
                self._pending_action = dp
                self._pending_reasoning.clear()
                self._steps_since_action = 0
                self._last_tool = event.tool or ""

        # Track output hash for state change detection
        if event.output_hash:
            self._last_output_hash = event.output_hash

        return completed

    def _finalize_pending(self) -> Optional[DecisionPoint]:
        """Close and return the pending decision point."""
        if self._pending_action is None:
            return None
        dp = self._pending_action
        self._pending_action = None
        self._points.append(dp)
        return dp

    def flush(self) -> Optional[DecisionPoint]:
        """Flush any pending decision point (e.g., at end of stream)."""
        return self._finalize_pending()

    def get_recent(self, n: int = 10) -> list[DecisionPoint]:
        """Get the last N decision points."""
        return self._points[-n:]

    def get_decision_quality_metrics(self) -> dict:
        """Aggregate quality metrics across all decision points."""
        if not self._points:
            return {"total_decisions": 0}

        n = len(self._points)
        return {
            "total_decisions": n,
            "success_rate": round(
                sum(1 for d in self._points if d.action_success) / n, 3),
            "avg_deliberation_ms": round(
                sum(d.reasoning_duration_ms for d in self._points) / n, 1),
            "verification_rate": round(
                sum(1 for d in self._points if d.followed_by_verification) / n, 3),
            "strategy_change_rate": round(
                sum(1 for d in self._points if d.strategy_changed) / n, 3),
            "state_change_rate": round(
                sum(1 for d in self._points if d.outcome_changed_state) / n, 3),
        }

    def to_dict(self) -> dict:
        return {
            "decisions": [d.to_dict() for d in self._points[-50:]],
            "metrics": self.get_decision_quality_metrics(),
        }


# ── MetacognitiveState ──────────────────────────────────────────────────────


@dataclass
class MetacognitiveState:
    """Models the system's metacognitive awareness.

    In human IP theory, metacognition = "thinking about thinking."
    In agentdiag, this maps to:
    - Detectors = metacognitive monitors (detect when processing fails)
    - LLM confirmation = metacognitive evaluation (is the monitor right?)
    - Remediation = metacognitive control (what to do about the failure)
    - OpenViking memory = metacognitive learning (improve from experience)
    """
    monitors_active: int = 0
    detections_total: int = 0
    detections_confirmed: int = 0
    detections_rejected: int = 0
    learning_available: bool = False
    past_cases_count: int = 0
    fp_rates: dict[str, float] = field(default_factory=dict)

    most_stressed_stage: str = ""
    stage_failure_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "monitors_active": self.monitors_active,
            "detections_total": self.detections_total,
            "detections_confirmed": self.detections_confirmed,
            "detections_rejected": self.detections_rejected,
            "learning_available": self.learning_available,
            "past_cases_count": self.past_cases_count,
            "fp_rates": dict(self.fp_rates),
            "most_stressed_stage": self.most_stressed_stage,
            "stage_failure_counts": dict(self.stage_failure_counts),
        }


# ── CognitiveStateTracker ──────────────────────────────────────────────────


class CognitiveStateTracker:
    """Incrementally maintains CognitiveState from TraceEvent stream.

    All updates are O(1) — uses running counters and fixed-size windows.
    Integrated into MonitorEngine.push() alongside HTA and CAFT.
    """

    ATTENTION_WINDOW = 20
    ACTION_WINDOW = 10
    FEEDBACK_WINDOW = 20
    MEMORY_DECAY_STEPS = 50
    MAX_CONTEXT_TOKENS = 200_000
    DELIBERATION_THRESHOLD = 4
    REPETITION_THRESHOLD = 9

    def __init__(self) -> None:
        # Perception tracking
        self._unique_read_targets: set[str] = set()
        self._total_reads: int = 0
        self._last_read_step: int = 0

        # Attention tracking (fixed-size deque)
        self._recent_reads: deque[tuple[int, str]] = deque(maxlen=self.ATTENTION_WINDOW)
        self._all_known_hashes: set[str] = set()

        # Working memory
        self._memory = WorkingMemoryModel()

        # Decision tracking
        self._current_reasoning_streak: int = 0
        self._recent_tools: deque[str] = deque(maxlen=self.ACTION_WINDOW)
        self._all_tools_seen: set[str] = set()
        self._recent_latencies: deque[float] = deque(maxlen=self.ACTION_WINDOW)

        # Action tracking
        self._recent_actions: deque[tuple[str, str, bool]] = deque(maxlen=self.ACTION_WINDOW)
        self._consecutive_same: int = 0
        self._last_tool_input: tuple[str, str] = ("", "")

        # Feedback tracking
        self._recent_phases: deque[str] = deque(maxlen=self.FEEDBACK_WINDOW)
        self._strategy_changes_after_error: int = 0
        self._total_errors_tracked: int = 0
        self._last_was_error: bool = False
        self._last_error_tool: str = ""
        self._actions_followed_by_verify: int = 0
        self._total_execute_actions: int = 0
        self._last_was_execute: bool = False

        # Decision point log
        self._decision_log = DecisionPointLog()

        # Metacognitive state
        self._metacognitive = MetacognitiveState()

        # Information-theoretic symbol stream
        self._symbol_stream = EventRouter()

        self._step: int = 0
        self._state = CognitiveState()

    @property
    def state(self) -> CognitiveState:
        """Current cognitive state snapshot."""
        return self._state

    @property
    def working_memory(self) -> WorkingMemoryModel:
        return self._memory

    @property
    def decision_log(self) -> DecisionPointLog:
        return self._decision_log

    @property
    def metacognitive(self) -> MetacognitiveState:
        return self._metacognitive

    @property
    def symbol_stream(self) -> EventRouter:
        return self._symbol_stream

    def update(
        self,
        event: TraceEvent,
        phase: Phase,
        num_detectors: int = 0,
        num_diagnoses: int = 0,
        num_confirmed: int = 0,
        num_rejected: int = 0,
        has_context_store: bool = False,
    ) -> CognitiveState:
        """O(1) incremental update from a single event.

        Args:
            event: The current trace event.
            phase: Current HTA phase after this event.
            num_detectors: Number of active CAFT detectors.
            num_diagnoses: Cumulative diagnosis count.
            num_confirmed: Cumulative confirmed count.
            num_rejected: Cumulative rejected count.
            has_context_store: Whether OpenViking is connected.
        """
        self._step = event.step
        s = self._state
        tool_lower = (event.tool or "").lower().replace("-", "_").replace(" ", "_")
        is_read = any(t in tool_lower for t in _GATHERING_TOOLS) if tool_lower else False
        is_reasoning = event.type in _PLANNING_TYPES

        # ── Stage 1: Perception ──────────────────────────────────────
        if is_read:
            self._total_reads += 1
            self._last_read_step = event.step
            if event.output_hash:
                self._unique_read_targets.add(event.output_hash)

        n_unique = len(self._unique_read_targets)
        s.perception_breadth = min(n_unique / 20.0, 1.0)
        s.perception_depth = min(self._total_reads / 40.0, 1.0)
        s.perception_recency = max(0.0, 1.0 - (self._step - self._last_read_step) / 20.0) if self._step > 0 else 0.0

        # ── Stage 2: Attention ───────────────────────────────────────
        if is_read and event.output_hash:
            was_known = event.output_hash in self._all_known_hashes
            self._recent_reads.append((event.step, event.output_hash))
            self._all_known_hashes.add(event.output_hash)

        if self._recent_reads:
            recent_hashes = {h for _, h in self._recent_reads}
            s.attention_diversity = len(recent_hashes) / len(self._recent_reads)
            s.attention_focus = 1.0 - s.attention_diversity

            # Tunnel risk: fraction of recent reads targeting already-known hashes
            if len(self._all_known_hashes) > 1:
                known_reads = sum(
                    1 for _, h in self._recent_reads
                    if h in self._all_known_hashes
                )
                # Only count as tunnel risk if ALL recent reads are known
                # (offset by 1 because the first read is always "new then known")
                total_in_window = len(self._recent_reads)
                new_in_window = sum(
                    1 for _, h in self._recent_reads
                    if h not in self._all_known_hashes or self._recent_reads.count((_, h)) == 1
                )
                # Simpler: count unique hashes that were first seen inside this window
                hashes_in_window = [h for _, h in self._recent_reads]
                # Re-reads of same resource in window = tunnel vision
                if total_in_window > 1:
                    unique_in_window = len(set(hashes_in_window))
                    s.attention_tunnel_risk = 1.0 - (unique_in_window / total_in_window)
                else:
                    s.attention_tunnel_risk = 0.0
            else:
                s.attention_tunnel_risk = 0.0
        else:
            s.attention_diversity = 0.0
            s.attention_focus = 0.0
            s.attention_tunnel_risk = 0.0

        # ── Stage 3: Working Memory ──────────────────────────────────
        self._memory.record_access(event)
        s.memory_utilization = self._memory.estimate_utilization()
        at_risk = self._memory.get_at_risk_items()
        s.memory_at_risk_items = len(at_risk)

        # Recency bias: proportion of active items from recent steps
        active = self._memory.get_active_items()
        if active:
            recent_active = sum(
                1 for i in active
                if (self._step - i.last_accessed_step) <= 20
            )
            s.memory_recency_bias = recent_active / len(active)
        else:
            s.memory_recency_bias = 0.0

        # ── Stage 4: Decision Making ─────────────────────────────────
        if is_reasoning:
            self._current_reasoning_streak += 1
        else:
            self._current_reasoning_streak = 0

        s.decision_deliberation = min(
            self._current_reasoning_streak / self.DELIBERATION_THRESHOLD, 1.0
        )

        if event.tool:
            self._recent_tools.append(event.tool)
            self._all_tools_seen.add(event.tool)

        if self._all_tools_seen:
            recent_unique = len(set(self._recent_tools))
            s.decision_options_explored = min(
                recent_unique / max(len(self._all_tools_seen), 1), 1.0
            )
        else:
            s.decision_options_explored = 0.0

        # Latency trend (simple slope estimate from deque)
        if event.latency_ms > 0:
            self._recent_latencies.append(event.latency_ms)

        if len(self._recent_latencies) >= 3:
            first_half = list(self._recent_latencies)[:len(self._recent_latencies) // 2]
            second_half = list(self._recent_latencies)[len(self._recent_latencies) // 2:]
            avg_first = sum(first_half) / len(first_half) if first_half else 0
            avg_second = sum(second_half) / len(second_half) if second_half else 0
            # Normalize: positive = latency increasing (harder decisions)
            if avg_first > 0:
                s.decision_latency_trend = min(max(
                    (avg_second - avg_first) / avg_first, -1.0
                ), 1.0)
            else:
                s.decision_latency_trend = 0.0
        else:
            s.decision_latency_trend = 0.0

        # ── Stage 5: Action ──────────────────────────────────────────
        if event.type == "tool_call":
            tool_input = (event.tool or "", event.input_hash or "")
            success = event.success

            self._recent_actions.append((tool_input[0], tool_input[1], success))

            # Consecutive same tracking
            if tool_input == self._last_tool_input and tool_input[0]:
                self._consecutive_same += 1
            else:
                self._consecutive_same = 1
            self._last_tool_input = tool_input

        if self._recent_actions:
            unique_tools = len({t for t, _, _ in self._recent_actions})
            s.action_diversity = min(unique_tools / max(len(self._all_tools_seen), 1), 1.0)

            successes = sum(1 for _, _, ok in self._recent_actions if ok)
            s.action_success_rate = successes / len(self._recent_actions)
        else:
            s.action_diversity = 0.0
            s.action_success_rate = 1.0

        s.action_repetition_risk = min(
            self._consecutive_same / self.REPETITION_THRESHOLD, 1.0
        )

        # ── Stage 6: Feedback ────────────────────────────────────────
        self._recent_phases.append(phase.label)

        # Verify ratio: VERIFYING / EXECUTING events in window
        verify_count = sum(1 for p in self._recent_phases if p == "verifying")
        execute_count = sum(1 for p in self._recent_phases if p == "executing")
        if execute_count > 0:
            s.feedback_verify_ratio = min(verify_count / execute_count, 1.0)
        else:
            s.feedback_verify_ratio = 0.0

        # Error response: strategy change after error
        if event.type == "tool_call":
            if self._last_was_error:
                self._total_errors_tracked += 1
                current_tool = event.tool or ""
                if current_tool != self._last_error_tool:
                    self._strategy_changes_after_error += 1

            if not event.success:
                self._last_was_error = True
                self._last_error_tool = event.tool or ""
            else:
                self._last_was_error = False

        if self._total_errors_tracked > 0:
            s.feedback_error_response = min(
                self._strategy_changes_after_error / self._total_errors_tracked, 1.0
            )
        else:
            s.feedback_error_response = 1.0

        # Loop closed: action followed by verification
        if phase == Phase.VERIFYING and self._last_was_execute:
            self._actions_followed_by_verify += 1
        if phase == Phase.EXECUTING:
            self._last_was_execute = True
            self._total_execute_actions += 1
        elif phase != Phase.EXECUTING:
            self._last_was_execute = False

        if self._total_execute_actions > 0:
            s.feedback_loop_closed = min(
                self._actions_followed_by_verify / self._total_execute_actions, 1.0
            )
        else:
            s.feedback_loop_closed = 0.0

        # ── Decision point tracking ──────────────────────────────────
        self._decision_log.push(event, phase)

        # ── Metacognitive state ──────────────────────────────────────
        self._metacognitive.monitors_active = num_detectors
        self._metacognitive.detections_total = num_diagnoses
        self._metacognitive.detections_confirmed = num_confirmed
        self._metacognitive.detections_rejected = num_rejected
        self._metacognitive.learning_available = has_context_store

        # ── Active IP stage ──────────────────────────────────────────
        s.active_ip_stage = _PHASE_TO_IP_STAGE.get(phase, "perception")

        # ── Overall cognitive load ───────────────────────────────────
        s.overall_cognitive_load = round(
            0.15 * (1.0 - s.perception_recency)
            + 0.15 * s.attention_tunnel_risk
            + 0.25 * s.memory_utilization
            + 0.15 * s.decision_deliberation
            + 0.15 * (1.0 - s.action_success_rate)
            + 0.15 * (1.0 - s.feedback_verify_ratio),
            3,
        )

        # ── Bottleneck detection ──────────────────────────────────
        _stages = ["perception", "attention", "working_memory", "decision_making", "action", "feedback"]
        stage_loads = {st: s._stage_load(st) for st in _stages}
        s.bottleneck_stage = max(stage_loads, key=stage_loads.get) if any(v > 0.3 for v in stage_loads.values()) else ""

        # ── Information-theoretic symbol stream ──────────────────────
        obs_event = trace_event_to_observable(event)
        self._symbol_stream.process_event(obs_event)

        s.last_updated_step = event.step
        return s

    def to_dict(self) -> dict:
        """Full cognitive state snapshot including sub-models."""
        return {
            **self._state.to_dict(),
            "working_memory": self._memory.to_dict(),
            "decision_points": self._decision_log.to_dict(),
            "metacognitive": self._metacognitive.to_dict(),
            "info_theoretic": self._symbol_stream.to_dict(),
        }
