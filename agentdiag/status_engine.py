"""Human-readable status engine.

Consumes raw IT metrics and compositor output, produces plain-English
status that a human can glance at and immediately understand.

The IT metrics power it underneath but the surface is:
  - A status color (green / yellow / red)
  - A one-sentence explanation in plain language
  - A current activity description
  - A translated list of recent actions
  - A timeline of alerts with drill-down data

Usage::

    engine = StatusEngine()
    engine.update(event, compositor_result, monitor_state)
    summary = engine.get_summary()
    # summary = {
    #   "color": "red",
    #   "message": "Stuck — re-reading config.py for 35 steps",
    #   "activity": "Reading: config.py",
    #   "recent_actions": ["Read config.py", "Read config.py", ...],
    #   "alerts": [{"step": 142, "message": "Started looping", ...}],
    #   ...
    # }
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional


# ── Compositor signature → human sentence ──────────────────────────────────

_SIGNATURE_MESSAGES = {
    "mechanical_repetition": "Stuck in a loop — repeating the same actions",
    "distributional_shift": "Goal may have drifted — working on unexpected files",
    "context_thrashing": "Confused — rapidly scanning files without progress",
    "stagnation": "Stalled — producing output but not making progress",
    "tight_iteration": "Iterating tightly — may be debugging",
    "incoherent_exploration": "Lost — selecting random actions with no pattern",
    "action_cycling": "Oscillating — cycling through the same tools repeatedly",
    "goal_discontinuity": "Abrupt shift — switched to unrelated work",
    "unclassified_anomaly": "Unusual behavior pattern detected",
}

_SIGNATURE_SEVERITY = {
    "mechanical_repetition": "red",
    "stagnation": "red",
    "context_thrashing": "red",
    "incoherent_exploration": "red",
    "distributional_shift": "yellow",
    "goal_discontinuity": "yellow",
    "action_cycling": "yellow",
    "tight_iteration": "yellow",
    "unclassified_anomaly": "yellow",
}


_READ_TOOLS = {"read", "read_file", "glob", "grep"}
_WRITE_TOOLS = {"edit", "write", "write_file"}
_SHELL_TOOLS = {"bash", "shell"}
_THINK_TYPES = {"reasoning", "thinking", "planning", "output"}
_TASK_TOOLS = {"taskcreate", "taskupdate", "taskget", "tasklist"}

def _translate_action(event_dict: dict) -> str:
    """Translate a raw event dict into a human-readable action string."""
    etype = event_dict.get("event_type", "")
    tool = event_dict.get("tool_name", "") or ""
    path = event_dict.get("target_path", "") or ""
    symbol = event_dict.get("symbol", "") or ""
    tool_lower = tool.lower()

    # Extract filename from path
    if path and "/" in path:
        filename = path.rstrip("/").split("/")[-1]
    elif path:
        filename = path
    else:
        filename = ""
    if len(filename) > 35:
        filename = filename[:32] + "..."

    # Skip thinking/output — don't show as actions
    if etype in _THINK_TYPES or tool_lower in _THINK_TYPES:
        return ""

    # Skip task management noise
    if tool_lower in _TASK_TOOLS:
        return ""

    # Reads
    if etype == "file_read" or tool_lower in _READ_TOOLS:
        return f"Read {filename}" if filename else "Read file"

    # Writes
    if etype == "file_write" or tool_lower in _WRITE_TOOLS:
        return f"Wrote {filename}" if filename else "Wrote code"

    # Shell commands — translate by content
    if etype == "shell_command" or tool_lower in _SHELL_TOOLS:
        cmd = path.strip()
        cmd_lower = cmd.lower()
        if "pytest" in cmd_lower or "python -m unittest" in cmd_lower:
            return "Ran tests"
        if "npm test" in cmd_lower or "jest" in cmd_lower:
            return "Ran tests"
        if "git commit" in cmd_lower:
            return "Committed changes"
        if "git push" in cmd_lower:
            return "Pushed to remote"
        if "git " in cmd_lower:
            git_verb = cmd.split("git ", 1)[-1].split()[0] if "git " in cmd else "git"
            return f"Ran git {git_verb}"
        if cmd:
            short = cmd.split("|")[0].split("&&")[0].strip()[:35]
            return f"Ran: {short}"
        return "Ran command"

    # Search tools
    if tool_lower in ("grep", "glob"):
        return f"Searched {filename}" if filename else "Searched codebase"

    # Memory operations
    if etype == "memory_load":
        ns = event_dict.get("namespace", "")
        return f"Loaded: {ns}" if ns else "Loaded context"
    if etype == "memory_store":
        ns = event_dict.get("namespace", "")
        return f"Saved: {ns}" if ns else "Saved to memory"

    # Evaluation
    if etype == "evaluation_result":
        score = event_dict.get("evaluation_score")
        criterion = event_dict.get("evaluation_criterion", "")
        if score is not None:
            return f"Evaluated: {criterion} {int(score * 100)}%"
        return f"Evaluated: {criterion}"

    # Fallback
    if tool and tool_lower not in ("unknown", "none", ""):
        return f"Used {tool}" + (f" on {filename}" if filename else "")
    return ""


def _describe_activity(recent_events: list[dict]) -> str:
    """Describe what the agent is currently doing based on recent events."""
    if not recent_events:
        return "idle"

    # Find the most recent non-thinking action
    for last in reversed(recent_events):
        tool = (last.get("tool_name") or "").lower()
        etype = last.get("event_type", "")
        path = last.get("target_path") or ""

        if etype in _THINK_TYPES or tool in _THINK_TYPES:
            continue
        if tool in _TASK_TOOLS:
            continue

        filename = path.rstrip("/").split("/")[-1] if "/" in path else path

        if etype == "file_read" or tool in _READ_TOOLS:
            return f"reading {filename}" if filename else "reading files"
        if etype == "file_write" or tool in _WRITE_TOOLS:
            return f"writing {filename}" if filename else "writing code"
        if etype == "shell_command" or tool in _SHELL_TOOLS:
            cmd_lower = path.lower()
            if "pytest" in cmd_lower or "test" in cmd_lower:
                return "running tests"
            if "git" in cmd_lower:
                return "using git"
            return "running commands"
        if tool:
            return f"using {last.get('tool_name', tool)}"

    return "thinking"


class StatusEngine:
    """Produces human-readable status from IT metrics and compositor output.

    Maintains a rolling window of recent events and anomalies to determine
    the current status color, message, and activity summary.
    """

    ALERT_LOOKBACK = 30  # steps to look back for color determination

    def __init__(self) -> None:
        self._recent_events: deque[dict] = deque(maxlen=20)
        self._alerts: deque[dict] = deque(maxlen=100)
        self._last_anomaly_step: int = -100
        self._last_anomaly_name: str = ""
        self._current_step: int = 0
        self._normal_since: int = 0  # step when last returned to normal
        self._session_label: str = ""
        self._repetition_target: str = ""  # file being repeated on
        self._repetition_count: int = 0

    def set_session_label(self, label: str) -> None:
        self._session_label = label

    def update(
        self,
        event_dict: Optional[dict] = None,
        anomaly: Optional[dict] = None,
        gradual_declines: Optional[dict] = None,
    ) -> None:
        """Update with a new event and/or anomaly result."""
        if event_dict:
            self._current_step = event_dict.get("step", self._current_step + 1)
            self._recent_events.append(event_dict)
            self._track_repetition(event_dict)

        if anomaly:
            sig = anomaly.get("signature", "unknown")
            step = anomaly.get("step", self._current_step)
            self._last_anomaly_step = step
            self._last_anomaly_name = sig

            # Create alert
            human_msg = _SIGNATURE_MESSAGES.get(sig, f"Anomaly: {sig}")

            # Enrich with context
            if sig == "mechanical_repetition" and self._repetition_target:
                human_msg = f"Stuck — repeating on {self._repetition_target} for {self._repetition_count} steps"
            elif sig == "context_thrashing":
                n_reads = sum(1 for e in self._recent_events if (e.get("event_type") or "").startswith("file_read"))
                human_msg = f"Confused — rapidly scanning {n_reads} files without writing"

            self._alerts.append({
                "step": step,
                "signature": sig,
                "message": human_msg,
                "severity": _SIGNATURE_SEVERITY.get(sig, "yellow"),
                "metrics": anomaly.get("metrics", {}),
                "wickens_stage": anomaly.get("wickens_stage", ""),
            })

        # Track return to normal
        if gradual_declines:
            for metric, info in gradual_declines.items():
                # Only alert once per metric decline
                existing = [a for a in self._alerts if a.get("signature") == "gradual_decline" and a.get("metric") == metric]
                if not existing:
                    self._alerts.append({
                        "step": self._current_step,
                        "signature": "gradual_decline",
                        "metric": metric,
                        "message": f"Gradually losing focus — {metric.replace('_', ' ')} declining",
                        "severity": "yellow",
                        "metrics": info,
                        "wickens_stage": "",
                    })

        steps_since_anomaly = self._current_step - self._last_anomaly_step
        if steps_since_anomaly == self.ALERT_LOOKBACK and self._last_anomaly_name:
            self._alerts.append({
                "step": self._current_step,
                "signature": "recovery",
                "message": "Returned to normal",
                "severity": "green",
                "metrics": {},
                "wickens_stage": "",
            })
            self._normal_since = self._current_step

    def _track_repetition(self, event_dict: dict) -> None:
        """Track if the agent is repeating the same action."""
        path = event_dict.get("target_path", "")
        symbol = event_dict.get("symbol", "")
        target = path or symbol

        if not target:
            return

        if len(self._recent_events) >= 2:
            prev = self._recent_events[-2]
            prev_target = prev.get("target_path", "") or prev.get("symbol", "")
            if target == prev_target:
                self._repetition_count += 1
                filename = target.rstrip("/").split("/")[-1] if "/" in target else target
                self._repetition_target = filename[:30]
            else:
                self._repetition_count = 0
                self._repetition_target = ""

    def get_summary(self) -> dict:
        """Produce the human-readable status summary."""
        color = self._compute_color()
        message = self._compute_message(color)
        activity = _describe_activity(list(self._recent_events))
        # Translate and filter out empty/thinking actions
        recent = [_translate_action(e) for e in list(self._recent_events)[-10:]]
        recent = [a for a in recent if a]  # filter empties
        # Deduplicate consecutive identical actions
        deduped = []
        for a in recent:
            if not deduped or deduped[-1] != a:
                deduped.append(a)
        deduped = deduped[-5:]  # last 5 unique actions

        return {
            "color": color,
            "message": message,
            "activity": activity,
            "recent_actions": deduped,
            "alerts": list(self._alerts)[-20:],
            "session_label": self._session_label,
            "current_step": self._current_step,
            "total_alerts": len(self._alerts),
        }

    def _compute_color(self) -> str:
        """Green / yellow / red based on recent anomaly history."""
        steps_since = self._current_step - self._last_anomaly_step

        if steps_since > self.ALERT_LOOKBACK:
            return "green"

        severity = _SIGNATURE_SEVERITY.get(self._last_anomaly_name, "yellow")
        return severity

    def _compute_message(self, color: str) -> str:
        """One-sentence status in plain English."""
        if color == "green":
            activity = _describe_activity(list(self._recent_events))
            return f"Working normally — {activity.lower()}"

        # Find the most recent non-recovery alert
        for alert in reversed(list(self._alerts)):
            if alert["severity"] != "green":
                return alert["message"]

        return "Working normally"
