"""Compact, deterministic summary of a Claude Code session.

A 7B local model can't read a 40MB session JSONL. The digest extracts
the signals a rater (human or LLM) actually needs to judge whether the
agent was stuck, drifting, or coherent: tool distribution, top user
messages, repeated-action runs, error patterns.

Output is structured (SessionDigest) and renders to either:
  - JSON dict (for ledger storage and Ollama rating)
  - Plain-text report (for human rating UI)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


DIMENSIONS = ("stuck_in_loop", "goal_drifted", "coherent_progress",
              "user_satisfied", "overall_health")
LIKERT_DIMS = ("stuck_in_loop", "goal_drifted", "coherent_progress",
               "user_satisfied")
CATEGORICAL_DIMS = ("overall_health",)
HEALTH_LABELS = ("healthy", "degraded", "pathological")

# Single source of truth for the rating scale. The human UI, the Ollama
# prompt, and the CAFT rule-mapping ALL render from this dict so every
# rater applies the identical, behaviorally-anchored definition. Polarity
# differs by dimension (noted in DIM_POLARITY); the anchors encode it.
SCALE_ANCHORS: dict[str, dict[int, str]] = {
    "stuck_in_loop": {
        1: "No repetition — each action was distinct and advanced the task",
        2: "Minor repetition — a few repeats but promptly moved on",
        3: "Mixed — some repeated cycles alongside real progress",
        4: "Substantial — long stretches repeating with little new progress",
        5: "Severe — looped on the same action(s), effectively no progress",
    },
    "goal_drifted": {
        1: "Fully on-task — all work served the user's request",
        2: "Slight tangent — brief unrelated detour, returned quickly",
        3: "Partial — meaningful time spent on work not asked for",
        4: "Substantial — often working on unrelated things",
        5: "Lost the goal — mostly doing something other than asked",
    },
    "coherent_progress": {
        1: "Incoherent — actions did not build on each other at all",
        2: "Mostly disjoint — little logical connection between steps",
        3: "Mixed — some logical sequences, some disconnected jumps",
        4: "Mostly coherent — actions generally build on prior ones",
        5: "Highly coherent — each step clearly follows from the last",
    },
    "user_satisfied": {
        1: "Clearly dissatisfied — explicit frustration or abandonment",
        2: "Mostly negative — complaints or repeated corrections",
        3: "Mixed / unclear — weak or conflicting signal",
        4: "Mostly positive — generally approving",
        5: "Clearly satisfied — explicit approval, thanks, or success",
    },
}

# When evidence shows NO resolution signal, abstain — do not default to 3.
SCALE_NOTES: dict[str, str] = {
    "user_satisfied": ("If the session has no resolution signal at all "
                       "(ends mid-task, on a log paste, etc.), choose "
                       "\"Can't tell\" — do NOT default to 3."),
}

HEALTH_ANCHORS: dict[str, str] = {
    "healthy": ("Sustained, organized, productive work; any issues are "
                "minor or environmental"),
    "degraded": ("Real friction or inefficiency, but the session still "
                 "functioned and produced work"),
    "pathological": ("The session broke down — looping, lost goal, or no "
                     "recovery from failure"),
}

# Higher value = worse for these; higher = better for the rest.
DIM_POLARITY: dict[str, str] = {
    "stuck_in_loop": "higher_is_worse",
    "goal_drifted": "higher_is_worse",
    "coherent_progress": "higher_is_better",
    "user_satisfied": "higher_is_better",
}


@dataclass
class RepeatRun:
    """A consecutive run of identical or near-identical commands."""
    pattern: str
    length: int


@dataclass
class SessionDigest:
    """Structured summary of one session, suitable for human or LLM rating."""
    session_id: str
    source_path: str
    total_lines: int
    n_user_messages: int
    n_tool_calls: int
    n_tool_results: int
    n_errors: int
    error_rate: float
    duration_seconds: Optional[float]
    tool_distribution: dict[str, int]
    first_user_message: str
    last_user_message: str
    sample_user_messages: list[str]
    longest_repeat_runs: list[RepeatRun]
    top_bash_patterns: list[tuple[str, int]]
    sample_errors: list[str]
    narrative: str = ""
    user_reactions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["longest_repeat_runs"] = [asdict(r) for r in self.longest_repeat_runs]
        return d

    def to_text(self, max_chars: int = 9000) -> str:
        """Plain-text rendering for human reading (UI panels, terminal)."""
        lines: list[str] = []
        lines.append(f"Session: {self.session_id}")
        lines.append(f"Source:  {self.source_path}")
        lines.append("")
        lines.append("--- Volume ---")
        lines.append(f"  Lines:        {self.total_lines}")
        lines.append(f"  User msgs:    {self.n_user_messages}")
        lines.append(f"  Tool calls:   {self.n_tool_calls}")
        lines.append(f"  Tool results: {self.n_tool_results}")
        lines.append(f"  Errors:       {self.n_errors} ({self.error_rate:.1%})")
        if self.duration_seconds:
            hours = self.duration_seconds / 3600
            if hours > 12:
                days = hours / 24
                lines.append(
                    f"  Span:         {days:.1f} days wall-clock "
                    f"(multi-day RESUMED session — not active work time)"
                )
            else:
                lines.append(f"  Duration:     {hours * 60:.1f} min")
        lines.append("")
        lines.append("--- Tool distribution (top 10) ---")
        for tool, n in sorted(self.tool_distribution.items(),
                              key=lambda kv: -kv[1])[:10]:
            lines.append(f"  {n:>5}  {tool}")
        lines.append("")
        lines.append("--- First user message ---")
        lines.append(f"  {self.first_user_message[:500]}")
        lines.append("")
        if self.narrative:
            lines.append("--- What the agent did (factual timeline) ---")
            lines.append(self.narrative)
            lines.append("")
        if self.user_reactions:
            lines.append("--- User reactions, verbatim (you judge the tone) ---")
            for q in self.user_reactions[:12]:
                lines.append(f'  "{q}"')
            lines.append("")
        if self.sample_user_messages:
            lines.append("--- Sample user messages (mid-session) ---")
            for m in self.sample_user_messages[:5]:
                lines.append(f"  - {m[:200]}")
            lines.append("")
        lines.append("--- Last user message ---")
        lines.append(f"  {self.last_user_message[:500]}")
        lines.append("")
        if self.longest_repeat_runs:
            lines.append("--- Longest consecutive identical-command runs ---")
            for r in self.longest_repeat_runs[:5]:
                lines.append(f"  {r.length:>3}x  {r.pattern[:80]}")
            lines.append("")
        if self.top_bash_patterns:
            lines.append("--- Top bash command patterns ---")
            for pat, n in self.top_bash_patterns[:8]:
                lines.append(f"  {n:>4}  {pat[:80]}")
            lines.append("")
        if self.sample_errors:
            lines.append("--- Sample errors (first 3) ---")
            for e in self.sample_errors[:3]:
                lines.append(f"  - {e[:200]}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (digest truncated)"
        return text


def _normalize_bash(cmd: str) -> str:
    """Reduce a bash command to its first sub-command for grouping."""
    first = cmd.split("\n")[0]
    first = first.split(" && ")[0]
    return re.sub(r"\s+", " ", first).strip()[:80]


def _is_error_result(blk: dict) -> bool:
    if blk.get("is_error"):
        return True
    cc = blk.get("content")
    s = cc if isinstance(cc, str) else json.dumps(cc) if cc else ""
    s_low = s[:300].lower()
    return any(t in s_low for t in ("error", "failed", "exception", "traceback"))


def build_digest(jsonl_path: str | Path,
                 sample_msgs_n: int = 5) -> SessionDigest:
    """Read a Claude Code session JSONL and produce a SessionDigest.

    Reads the file once, line-by-line, so works on multi-GB sessions.
    """
    path = Path(jsonl_path)
    user_msgs: list[str] = []
    tool_uses: Counter[str] = Counter()
    bash_cmds: list[str] = []
    errors: list[str] = []
    n_tool_results = 0
    n_errors = 0
    total_lines = 0
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = d.get("timestamp")
            if ts:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
            msg = d.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                if isinstance(content, str) and content.strip() and not content.startswith("<"):
                    user_msgs.append(content.strip())
                elif isinstance(content, list):
                    for blk in content:
                        if isinstance(blk, dict) and blk.get("type") == "tool_result":
                            n_tool_results += 1
                            if _is_error_result(blk):
                                n_errors += 1
                                cc = blk.get("content")
                                s = cc if isinstance(cc, str) else json.dumps(cc)[:300]
                                errors.append(s.replace("\n", " | ")[:250])
            elif role == "assistant":
                if isinstance(content, list):
                    for blk in content:
                        if isinstance(blk, dict) and blk.get("type") == "tool_use":
                            name = blk.get("name", "?")
                            tool_uses[name] += 1
                            if name == "Bash":
                                cmd = (blk.get("input") or {}).get("command", "")
                                bash_cmds.append(_normalize_bash(cmd))

    runs: list[RepeatRun] = []
    cur, cur_n = None, 0
    for c in bash_cmds:
        if c == cur:
            cur_n += 1
        else:
            if cur and cur_n >= 3:
                runs.append(RepeatRun(pattern=cur, length=cur_n))
            cur, cur_n = c, 1
    if cur and cur_n >= 3:
        runs.append(RepeatRun(pattern=cur, length=cur_n))
    runs.sort(key=lambda r: -r.length)

    bash_top = Counter(bash_cmds).most_common(10)

    sample = []
    if len(user_msgs) > 2:
        step = max(1, len(user_msgs) // (sample_msgs_n + 1))
        for i in range(step, len(user_msgs) - 1, step):
            sample.append(user_msgs[i])
            if len(sample) >= sample_msgs_n:
                break

    duration = None
    if first_ts and last_ts:
        try:
            from datetime import datetime
            t0 = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            duration = (t1 - t0).total_seconds()
        except Exception:
            pass

    n_tool_calls = sum(tool_uses.values())
    error_rate = n_errors / max(n_tool_results, 1)

    try:
        from agentdiag.validation.narrative import build_narrative
        narrative, user_reactions = build_narrative(path)
    except Exception:
        narrative, user_reactions = "", []

    return SessionDigest(
        session_id=path.stem,
        source_path=str(path),
        total_lines=total_lines,
        n_user_messages=len(user_msgs),
        n_tool_calls=n_tool_calls,
        n_tool_results=n_tool_results,
        n_errors=n_errors,
        error_rate=error_rate,
        duration_seconds=duration,
        tool_distribution=dict(tool_uses),
        first_user_message=(user_msgs[0] if user_msgs else "")[:1000],
        last_user_message=(user_msgs[-1] if user_msgs else "")[:1000],
        sample_user_messages=[m[:300] for m in sample],
        longest_repeat_runs=runs[:5],
        top_bash_patterns=bash_top,
        sample_errors=errors[:5],
        narrative=narrative,
        user_reactions=user_reactions,
    )
