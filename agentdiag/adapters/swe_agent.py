"""Adapter: nebius/SWE-agent-trajectories chat rows -> ObservableEvent.

Verified format (2026-05-15, real rows inspected):
  row = {instance_id, model_name, target(bool), exit_status,
         generated_patch, eval_logs,
         trajectory: [{role, text, ...}]}
  roles: 'system' (1), 'user' (observations incl. the issue),
         'ai' (the agent: free-text reasoning + ONE fenced block
          containing exactly one SWE-agent ACI command).

Example 'ai' text:
    ... reasoning ...
    ```
    search_dir "PYTHON_THREADPOOL_THREAD_COUNT"
    ```

So each `ai` turn yields exactly one action event. `user` turns are
environment observations (used only for error tagging, not emitted as
agent actions). This is the same shape of parsing as the Claude Code
Bash adapter — and, per docs/PILOT_PREREGISTRATION.md §5, its action
vocabulary is NEW, so the corpus-specific symbolization audit is
mandatory before any H1/H2 interpretation.

Pure parsing, no network. Sampling/loading lives in the pilot harness
(gated by the committed pre-registration), not here.
"""

from __future__ import annotations

import re
from typing import Iterator, Optional

from agentdiag.observable import ObservableEvent, EventType

_FENCE = re.compile(r"```[a-zA-Z0-9_]*\n?(.*?)```", re.DOTALL)

# SWE-agent ACI verbs -> coarse event class. Anything not listed is
# treated as a shell command (SWE-agent allows raw bash).
_READ_VERBS = {
    "open", "goto", "scroll_down", "scroll_up", "search_dir",
    "search_file", "find_file", "ls", "cat", "grep", "head", "tail",
}
_WRITE_VERBS = {"edit", "create", "insert", "append"}
_TERMINAL_VERBS = {"submit"}


def _extract_action(ai_text: str) -> Optional[tuple[str, str]]:
    """Return (command, argstr) from the LAST fenced block, or None.

    SWE-agent convention: the action is the final fenced block; the
    command is its first non-empty line.
    """
    blocks = _FENCE.findall(ai_text or "")
    if not blocks:
        return None
    body = blocks[-1].strip()
    if not body:
        return None
    first = body.splitlines()[0].strip()
    if not first:
        return None
    parts = first.split(None, 1)
    cmd = parts[0].strip()
    args = parts[1].strip() if len(parts) > 1 else ""
    return cmd, args


def _first_pathish(args: str) -> Optional[str]:
    """Best-effort target: a quoted string or the first token."""
    m = re.search(r'["\']([^"\']+)["\']', args)
    if m:
        return m.group(1)[:200]
    tok = args.split(None, 1)[0] if args.strip() else ""
    return tok[:200] or None


def _event_for(step: int, cmd: str, args: str) -> ObservableEvent:
    """Always carry the ACI command in tool_name.

    The convenience constructors (file_read_event/file_write_event)
    intentionally drop tool_name, which would erase the command
    identity from the symbol stream and make the pre-registered
    symbolization audit meaningless. So construct directly: event_type
    by verb class, tool_name = the SWE-agent command, always.
    """
    target = _first_pathish(args)
    low = cmd.lower()
    if low in _READ_VERBS:
        etype = EventType.FILE_READ
    elif low in _WRITE_VERBS:
        etype = EventType.FILE_WRITE
    elif low in _TERMINAL_VERBS:
        etype = EventType.TOOL_CALL
    else:                                   # unknown ACI verb / raw bash
        etype = EventType.SHELL_COMMAND
    return ObservableEvent(step=step, timestamp=float(step),
                           event_type=etype, tool_name=cmd,
                           target_path=target)


def iter_events(row: dict) -> Iterator[ObservableEvent]:
    """Yield one ObservableEvent per parsed `ai` action turn."""
    step = 0
    for turn in row.get("trajectory") or []:
        if turn.get("role") not in ("ai", "assistant"):
            continue
        act = _extract_action(turn.get("text", ""))
        if act is None:
            continue
        cmd, args = act
        yield _event_for(step, cmd, args)
        step += 1


def row_to_events(row: dict) -> list[ObservableEvent]:
    return list(iter_events(row))


def outcome(row: dict) -> bool:
    """The independent, test-based label. Definitionally not behavioral."""
    return bool(row.get("target"))


def baseline_features(row: dict) -> dict:
    """Trivial 'could a dumb heuristic do this' controls (pre-reg §4)."""
    traj = row.get("trajectory") or []
    return {
        "n_turns": len(traj),
        "n_parsed_actions": sum(1 for _ in iter_events(row)),
        "patch_len": len(row.get("generated_patch") or ""),
        "exit_status": row.get("exit_status") or "unknown",
        "model_name": row.get("model_name") or "unknown",
    }
