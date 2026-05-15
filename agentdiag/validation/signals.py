"""Programmatic ground-truth signal extraction from raw Claude Code JSONL.

Every signal here is DETERMINISTIC and OBSERVABLE — a count, a literal
match, a structural pattern. No judgment, no model, no proxy. Two runs
on the same file produce identical output. This is the ground truth the
validation harness measures CAFT against (convergent validity), because
human Likert ratings proved too abstract, slow, and polarity-error-prone.

The objects extracted are facts about the trace:

  - literal_loop_max:  longest run of identical (tool, normalized-args)
  - error_retry_cycles: error -> near-identical retry -> error chains
  - user_reprompts:    user message ~= an earlier user message
  - frustration_hits:  profanity / wtf / ugh in USER messages only
  - correction_hits:   "no,", "that's not", "i said", "undo", "revert"...
  - git_undo:          git revert / reset --hard / checkout -- in Bash
  - tasks_created / tasks_completed: TaskCreate vs TaskUpdate(completed)
  - resolution:        positive | negative | none  (from final user msg)
  - abandoned:         session ends on an error with no user follow-up

These do not depend on CAFT in any way — they are computed straight off
the transcript so CAFT can be scored against them without circularity.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path


_FRUSTRATION = re.compile(
    r'\b(fuck\w*|shit\w*|damn|damnit|goddamn|wtf|bullshit|crap|ugh+)\b', re.I)
# Deliberately tight — bare "no," / "stop" were greedy and false-fired
# on "now", "no problem", "non-stop". Require explicit corrective intent.
_CORRECTION = re.compile(
    r"(that'?s not (?:what|right|correct)|that'?s wrong|not what i (?:asked|wanted|said)|"
    r"\bi (?:said|asked you to)\b|you (?:didn'?t|did not) (?:do|follow)|"
    r"\bundo (?:that|this|it)\b|\brevert (?:that|this|it)\b|"
    r"why did you|you broke|stop doing that)", re.I)
_POSITIVE_CLOSE = re.compile(
    r"\b(thanks|thank you|perfect|great|awesome|works?|working|nice|"
    r"looks good|lgtm|exactly|that'?s it|solved|fixed it|beautiful)\b", re.I)
# Used ONLY to classify the FINAL user message as a negative close.
# Broader than _CORRECTION (which counts mid-session corrections) because
# at session end a bare "still broken" is a legitimate dissatisfaction.
_NEGATIVE_CLOSE = re.compile(
    r"\b(still (?:wrong|broken|not working|failing|doesn'?t work|won'?t)|"
    r"that'?s (?:wrong|not (?:it|right|what))|not what i|nope\b|"
    r"doesn'?t work|didn'?t work|broke(?:n)?\b|undo\b|revert\b|"
    r"frustrat|give up|forget it)", re.I)
_GIT_UNDO = re.compile(
    r"git\s+(revert|reset\s+--hard|checkout\s+--|restore\b|clean\s+-[a-z]*f)", re.I)
_LOG_NOISE = ("[skoscel@", "Traceback", "INFO -", "WARNING", "pytorch_env)",
              "$ ", ">>>", "node0", "tensorflow/", "====")


@dataclass
class SessionSignals:
    session_id: str
    n_user_messages: int = 0
    n_tool_calls: int = 0
    n_tool_results: int = 0
    n_errors: int = 0
    literal_loop_max: int = 0
    literal_loop_tool: str = ""
    error_retry_cycles: int = 0
    user_reprompts: int = 0
    frustration_hits: int = 0
    correction_hits: int = 0
    git_undo: int = 0
    tasks_created: int = 0
    tasks_completed: int = 0
    resolution: str = "none"          # positive | negative | none
    abandoned: bool = False
    frustration_quotes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _norm_args(tool: str, tinput: dict) -> str:
    """Normalized signature of a tool call for literal-loop detection."""
    tl = (tool or "").lower()
    if tl == "bash":
        cmd = (tinput.get("command", "") or "").split("\n")[0]
        norm = re.sub(r"\s+", " ", cmd).strip()[:120]
        return f"bash:{norm}"
    if tl in ("read", "edit", "write", "notebookedit"):
        return f"{tl}:{tinput.get('file_path', '')}"
    if tl in ("grep", "glob"):
        return f"{tl}:{tinput.get('pattern', '')}:{tinput.get('path', '')}"
    return f"{tl}:{json.dumps(tinput, sort_keys=True)[:120]}"


def _similar(a: str, b: str) -> float:
    """Cheap token Jaccard — no deps. 1.0 == identical token sets."""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _is_prose(text: str) -> bool:
    t = text.strip()
    if not t or t.startswith("<"):
        return False
    if any(m in t for m in _LOG_NOISE):
        return False
    letters = sum(c.isalpha() or c.isspace() for c in t)
    return letters / max(len(t), 1) > 0.65


RATER_ID = "signal-v2"  # v1 over-reached (proxy coherence, hand health
                         # rule); v2 = strict facts only, abstain on the rest


def signals_to_ratings(sig: SessionSignals):
    """Map ONLY strictly-observable facts onto the shared scale.

    Per docs/CONSTRUCT_REVISION.md the extractor asserts only what it
    can literally prove and ABSTAINS on everything inferential — the
    same discipline applied to CAFT. It earlier reproduced CAFT's sin
    (a hand-rule that called a rage session "healthy", a task-ratio
    proxy for "coherence"); those are now removed, not patched.

      - stuck_in_loop  : the longest literal identical-call run IS the
                          construct -> high confidence
      - user_satisfied : ONLY from the FINAL user message's resolution
                          (positive -> 5, negative -> 1). Mid-session
                          frustration does NOT flip it (a session can
                          recover). No closing signal -> abstain.
      - goal_drifted     : needs intent -> abstain
      - coherent_progress: not literally observable -> abstain
      - overall_health   : retired quality verdict -> abstain

    The raw counts (frustration_hits + verbatim quotes, error_retry,
    correction_hits, git_undo, task ledger) remain available on
    SessionSignals as facts for analysis, but are NOT collapsed into a
    judgment here.
    """
    from agentdiag.validation.ledger import Rating

    L = sig.literal_loop_max
    stuck = 1 if L < 3 else 2 if L < 5 else 3 if L < 8 else 4 if L < 15 else 5
    stuck_r = (f"longest identical-call run = {L}x "
               f"({sig.literal_loop_tool[:50]})")

    # user_satisfied strictly from the FINAL message's resolution only
    if sig.resolution == "positive":
        sat, sat_c = 5, "high"
    elif sig.resolution == "negative":
        sat, sat_c = 1, "high"
    else:
        sat, sat_c = None, ""
    sat_r = (f"final-message resolution={sig.resolution} "
             f"(mid-session frustration={sig.frustration_hits} recorded as "
             f"a fact but NOT used to flip this label)")

    mk = lambda dim, v, c, r: Rating(
        session_id=sig.session_id, rater_type="signal", rater_id=RATER_ID,
        dimension=dim, value=v, confidence=c, reasoning=r)

    return [
        mk("stuck_in_loop", stuck, "high", stuck_r),
        mk("user_satisfied", sat, sat_c, sat_r),
        mk("goal_drifted", None, "", "needs intent — not trace-observable"),
        mk("coherent_progress", None, "",
           "not literally observable — abstain (was an invalid proxy)"),
        mk("overall_health", None, "",
           "retired quality verdict (see docs/CONSTRUCT_REVISION.md)"),
    ]


def rate_with_signals(jsonl_path: str | Path):
    """Convenience: extract + map in one call. Returns list[Rating]."""
    return signals_to_ratings(extract_signals(jsonl_path))


def extract_signals(jsonl_path: str | Path) -> SessionSignals:
    path = Path(jsonl_path)
    sig = SessionSignals(session_id=path.stem)

    tool_sig_stream: list[str] = []          # normalized (tool,args) in order
    user_prose: list[str] = []
    last_final_user = ""
    # rolling structure for error->retry->error
    pending_error_after: str | None = None    # tool sig that just errored

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = d.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")

            if role == "user":
                if isinstance(content, str) and content.strip() and not content.startswith("<"):
                    sig.n_user_messages += 1
                    txt = content.strip()
                    if _is_prose(txt):
                        last_final_user = txt
                        # re-prompt: similar to an earlier prose user msg
                        for prev in user_prose:
                            if _similar(prev, txt) >= 0.6:
                                sig.user_reprompts += 1
                                break
                        user_prose.append(txt)
                    if _FRUSTRATION.search(txt):
                        sig.frustration_hits += 1
                        sig.frustration_quotes.append(
                            " ".join(txt.split())[:160])
                    if _CORRECTION.search(txt):
                        sig.correction_hits += 1
                elif isinstance(content, list):
                    for blk in content:
                        if isinstance(blk, dict) and blk.get("type") == "tool_result":
                            sig.n_tool_results += 1
                            if blk.get("is_error"):
                                sig.n_errors += 1
                                pending_error_after = (
                                    tool_sig_stream[-1] if tool_sig_stream else None)

            elif role == "assistant" and isinstance(content, list):
                for blk in content:
                    if not (isinstance(blk, dict) and blk.get("type") == "tool_use"):
                        continue
                    name = blk.get("name", "")
                    tinput = blk.get("input") or {}
                    sig.n_tool_calls += 1
                    if name == "TaskCreate":
                        sig.tasks_created += 1
                    if name == "TaskUpdate" and (
                            str(tinput.get("status", "")).lower() == "completed"):
                        sig.tasks_completed += 1
                    if name == "Bash" and _GIT_UNDO.search(
                            tinput.get("command", "") or ""):
                        sig.git_undo += 1
                    s = _norm_args(name, tinput)
                    # error -> near-identical retry (same normalized sig)
                    if pending_error_after is not None and s == pending_error_after:
                        sig.error_retry_cycles += 1
                    pending_error_after = None
                    tool_sig_stream.append(s)

    # longest consecutive identical (tool,args) run
    cur, run = None, 0
    for s in tool_sig_stream:
        if s == cur:
            run += 1
        else:
            if run > sig.literal_loop_max:
                sig.literal_loop_max = run
                sig.literal_loop_tool = cur or ""
            cur, run = s, 1
    if run > sig.literal_loop_max:
        sig.literal_loop_max = run
        sig.literal_loop_tool = cur or ""

    # resolution strictly from the FINAL prose user message
    if last_final_user:
        if (_FRUSTRATION.search(last_final_user)
                or _NEGATIVE_CLOSE.search(last_final_user)):
            sig.resolution = "negative"
        elif _POSITIVE_CLOSE.search(last_final_user):
            sig.resolution = "positive"
        else:
            sig.resolution = "none"

    # abandoned: last meaningful event was an errored tool result
    sig.abandoned = (sig.n_errors > 0 and not last_final_user
                     and sig.resolution == "none")
    return sig
