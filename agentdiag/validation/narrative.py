"""Strictly-descriptive 'what the agent did' narrative.

Turns a raw Claude Code session into a phased, factual timeline a human
(or LLM) can actually read and rate from. Reuses status_engine's
_translate_action for per-action phrasing so wording matches the rest
of the product.

DESIGN CONSTRAINT — descriptive, never evaluative.
This module reports WHAT happened ("edited page.tsx 9x"), never HOW
WELL it went ("got stuck", "struggled"). Judgment is exactly what we
ask the rater to supply; if the narrative editorializes it anchors the
rater and contaminates the agreement statistics. Every phrase here is a
count, a filename, an error string, a timestamp, or a verbatim user
quote.

Provenance of errors is distinguished (environmental vs command) but
NOT scored — a cancelled parallel call or unmounted filesystem is not
an agent-quality signal and is labelled as such so the rater isn't
nudged. Sub-agent (Agent tool) invocations show their verbatim
description, since their internal work is not visible in this trace.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from agentdiag.status_engine import _translate_action


_CATEGORY = {
    "explore": ("Read", "Searched"),
    "implement": ("Wrote",),
    "verify": ("Ran tests", "Committed", "Pushed", "Ran git"),
}

# Substrings that mark an error as ENVIRONMENTAL (not an agent-quality
# signal): cancelled fan-out calls, unmounted/again filesystems,
# connectivity. Conservative on purpose — anything not matched is
# labelled the neutral "command-error", never "agent failure".
_ENV_ERROR_MARKERS = (
    "cancelled: parallel tool call",
    "<tool_use_error>",
    "transport endpoint is not connected",
    "could not connect",
    "connection refused",
    "input/output error",
    "resource temporarily unavailable",
    "device not configured",
)

# A user message is a "reaction" (worth quoting verbatim for the
# user_satisfied judgement) if it is short and prose-like, not a pasted
# terminal log / command dump.
_LOG_NOISE_MARKERS = ("[skoscel@", "Traceback", "INFO -", "WARNING",
                       "pytorch_env)", "$ ", ">>>", "node0", "tensorflow/")


def _categorize(phrase: str) -> str:
    for cat, prefixes in _CATEGORY.items():
        if any(phrase.startswith(p) for p in prefixes):
            return cat
    if phrase.startswith("Ran"):
        return "command"
    if phrase.startswith("Spawned sub-agent"):
        return "delegate"
    return "other"


def _classify_error(text: str) -> str:
    low = text.lower()
    if any(m in low for m in _ENV_ERROR_MARKERS):
        return "environmental"
    return "command"


def _looks_like_reaction(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 160:
        return False
    if "\n" in t.strip():
        return False
    if any(m in t for m in _LOG_NOISE_MARKERS):
        return False
    letters = sum(c.isalpha() or c.isspace() for c in t)
    return letters / max(len(t), 1) > 0.7


def _cc_tool_to_event_dict(tool_name: str, tinput: dict) -> dict:
    tn = (tool_name or "").lower()
    if tn in ("read", "glob"):
        return {"event_type": "file_read", "tool_name": tool_name,
                "target_path": tinput.get("file_path") or tinput.get("pattern", "")}
    if tn == "grep":
        return {"event_type": "", "tool_name": "grep",
                "target_path": tinput.get("path", "")}
    if tn in ("edit", "write", "notebookedit", "multiedit"):
        return {"event_type": "file_write", "tool_name": tool_name,
                "target_path": tinput.get("file_path", "")}
    if tn == "bash":
        return {"event_type": "shell_command", "tool_name": "Bash",
                "target_path": tinput.get("command", "")}
    return {"event_type": "", "tool_name": tool_name or "",
            "target_path": tinput.get("file_path", "")}


def _short(text: str, n: int = 130) -> str:
    text = " ".join(text.split())
    return text[:n] + ("…" if len(text) > n else "")


def _parse_ts(ts: str):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def build_narrative(jsonl_path: str | Path, max_phases: int = 14) -> tuple[str, list[str]]:
    """Return (narrative_text, user_reactions_verbatim).

    user_reactions is a list of short prose user messages quoted exactly
    (no classification) so a rater can judge in-session satisfaction
    without the digest editorializing.
    """
    path = Path(jsonl_path)
    stream: list[tuple] = []  # (kind, payload, ts)
    reactions: list[str] = []
    t0 = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_ts(d.get("timestamp", "")) if d.get("timestamp") else None
            if ts and t0 is None:
                t0 = ts
            msg = d.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                if isinstance(content, str) and content.strip() and not content.startswith("<"):
                    stream.append(("user", content.strip(), ts))
                    if _looks_like_reaction(content):
                        reactions.append(_short(content, 120))
                elif isinstance(content, list):
                    for blk in content:
                        if (isinstance(blk, dict)
                                and blk.get("type") == "tool_result"
                                and blk.get("is_error")):
                            cc = blk.get("content")
                            s = cc if isinstance(cc, str) else json.dumps(cc)
                            stream.append(("error", _classify_error(s), ts))
            elif role == "assistant" and isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "tool_use":
                        name = blk.get("name", "")
                        tinput = blk.get("input") or {}
                        if name == "Agent":
                            desc = (tinput.get("description")
                                    or tinput.get("subagent_type") or "task")
                            stream.append(
                                ("act", f'Spawned sub-agent: "{_short(desc, 70)}"', ts))
                            continue
                        ed = _cc_tool_to_event_dict(name, tinput)
                        phrase = _translate_action(ed)
                        if phrase:
                            stream.append(("act", phrase, ts))

    if not stream:
        return "(no readable actions in this session)", reactions

    phases: list[dict] = []
    cur = None

    def _new_phase(trigger_user, ts):
        nonlocal cur
        cur = {"user": trigger_user, "acts": [], "files": Counter(),
               "err_cmd": 0, "err_env": 0, "subagents": [], "ts": ts}
        phases.append(cur)

    for kind, payload, ts in stream:
        if kind == "user":
            _new_phase(_short(payload), ts)
            continue
        if cur is None:
            _new_phase(None, ts)
        if cur["ts"] is None and ts is not None:
            cur["ts"] = ts
        if kind == "error":
            if payload == "environmental":
                cur["err_env"] += 1
            else:
                cur["err_cmd"] += 1
            continue
        phrase = payload
        if phrase.startswith("Spawned sub-agent:"):
            cur["subagents"].append(phrase)
        cur["acts"].append(phrase)
        for verb in ("Read ", "Wrote ", "Searched "):
            if phrase.startswith(verb):
                cur["files"][phrase[len(verb):]] += 1

    phases = [p for p in phases if p["acts"] or p["user"]]
    if len(phases) > max_phases:
        scored = sorted(
            enumerate(phases),
            key=lambda ip: (ip[1]["user"] is not None, len(ip[1]["acts"])),
            reverse=True,
        )
        keep = sorted(i for i, _ in scored[:max_phases])
        phases = [phases[i] for i in keep]

    def _day_marker(ts) -> str:
        if ts is None or t0 is None:
            return ""
        day = (ts.date() - t0.date()).days + 1
        return f"day {day} {ts.strftime('%H:%M')}"

    out: list[str] = []
    step = 0
    for i, p in enumerate(phases, 1):
        n = len(p["acts"])
        rng = f"~{step+1}-{step+n}" if n else f"~{step+1}"
        step += n
        when = _day_marker(p["ts"])
        head = f"[{i}] steps {rng}" + (f" · {when}" if when else "")
        if p["user"] is not None:
            out.append(f'{head} — USER: "{p["user"]}"')
        else:
            out.append(head)
        if not p["acts"]:
            continue
        non_sub = [a for a in p["acts"] if not a.startswith("Spawned sub-agent")]
        top_acts = Counter(non_sub).most_common(4)
        if top_acts:
            acts_str = "; ".join(
                f"{a} ×{c}" if c > 1 else a for a, c in top_acts)
            out.append(f"      did: {acts_str}")
        for sa in p["subagents"][:4]:
            out.append(f"      {sa}  (sub-agent internals not in this trace)")
        files = [f for f, _ in p["files"].most_common(5) if f]
        if files:
            out.append(f"      files: {', '.join(files)}")
        if p["err_cmd"] or p["err_env"]:
            bits = []
            if p["err_cmd"]:
                bits.append(f"{p['err_cmd']} command-error(s)")
            if p["err_env"]:
                bits.append(f"{p['err_env']} environmental "
                            f"(cancelled/unmounted — not an agent signal)")
            out.append(f"      tool results: {', '.join(bits)}")

    return "\n".join(out), reactions
