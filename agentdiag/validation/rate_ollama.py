"""Local LLM rater via Ollama HTTP API.

Sends a session digest to a local Ollama instance with a JSON-mode
prompt and parses out 4 Likert scores + 1 categorical label + reasoning.

Defaults to http://localhost:11434 with llama3.2:3b. Override either
via the function args.

The rater produces one Rating per dimension (5 total) so they can be
analyzed independently.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Optional

from agentdiag.validation.digest import (
    SessionDigest, DIMENSIONS, LIKERT_DIMS, CATEGORICAL_DIMS, HEALTH_LABELS,
    SCALE_ANCHORS, SCALE_NOTES, HEALTH_ANCHORS,
)
from agentdiag.validation.ledger import Rating


DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"
TIMEOUT_SECONDS = 120


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns malformed output."""


def _build_rubric() -> str:
    """Render the shared SCALE_ANCHORS into the LLM rubric verbatim.

    The human UI renders the same dict, so all raters apply the
    identical anchored definition.
    """
    parts: list[str] = []
    for dim in LIKERT_DIMS:
        parts.append(f'  "{dim}" (1-5):')
        for v in (1, 2, 3, 4, 5):
            parts.append(f"      {v} = {SCALE_ANCHORS[dim][v]}")
        if dim in SCALE_NOTES:
            parts.append(f"      NOTE: {SCALE_NOTES[dim]} "
                         f"(use the JSON value null to abstain)")
    parts.append('  "overall_health" (categorical):')
    for label in HEALTH_LABELS:
        parts.append(f"      {label} = {HEALTH_ANCHORS[label]}")
    return "\n".join(parts)


SYSTEM_PROMPT = """You are an expert reviewer of AI coding agent sessions.
You will read a compact, factual summary of a Claude Code session and
rate it on five dimensions using the EXACT anchored scale below. The
human reviewer and the automated system use this identical scale, so
apply each anchor literally — do not invent your own interpretation.

The summary includes a phased "what the agent did" timeline (factual,
no judgments), verbatim user reactions, tool counts, and error
provenance. Base your ratings only on this evidence.

RATING SCALE — apply these anchors exactly:
{rubric}

If a dimension genuinely cannot be judged from the evidence, use the
JSON value null for it (an honest abstention is better than a guess).

Respond with ONLY a JSON object, these keys exactly:
  "stuck_in_loop", "goal_drifted", "coherent_progress",
  "user_satisfied"  -> integer 1-5 or null
  "overall_health"  -> "healthy" | "degraded" | "pathological"
  "reasoning"       -> short string (<200 chars) citing the evidence

Begin your response with `{{` and end with `}}`. No prose, no fences.""".format(
    rubric=_build_rubric())


def _build_user_prompt(digest: SessionDigest) -> str:
    return f"Session summary to rate:\n\n{digest.to_text(max_chars=6000)}"


def _post_ollama(host: str, model: str, prompt: str,
                 timeout: int = TIMEOUT_SECONDS) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2, "num_predict": 400},
    }
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise OllamaError(f"Cannot reach Ollama at {host}: {e}") from e
    try:
        envelope = json.loads(body)
    except json.JSONDecodeError as e:
        raise OllamaError(f"Ollama returned non-JSON envelope: {body[:200]}") from e
    return envelope.get("response", "")


def _parse_rating(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise OllamaError(f"LLM output is not valid JSON: {text[:300]}") from e
    if not isinstance(obj, dict):
        raise OllamaError(f"LLM output is not a JSON object: {type(obj)}")
    return obj


def _validate_and_normalize(obj: dict) -> dict:
    out: dict[str, int | str | None] = {}
    for d in LIKERT_DIMS:
        if d not in obj:
            raise OllamaError(f"LLM missing dimension {d!r}")
        v = obj.get(d)
        if v is None:                       # explicit abstention
            out[d] = None
            continue
        try:
            iv = int(v)
        except (ValueError, TypeError) as e:
            raise OllamaError(f"LLM gave non-integer for {d}: {v!r}") from e
        if iv < 1 or iv > 5:
            iv = max(1, min(5, iv))
        out[d] = iv
    cat = obj.get("overall_health", "")
    if cat is None:
        out["overall_health"] = None
        out["reasoning"] = str(obj.get("reasoning", ""))[:500]
        return out
    if isinstance(cat, str):
        cat = cat.strip().lower()
    if cat not in HEALTH_LABELS:
        for label in HEALTH_LABELS:
            if isinstance(cat, str) and label in cat:
                cat = label
                break
        else:
            raise OllamaError(
                f"overall_health must be one of {HEALTH_LABELS}, got {cat!r}"
            )
    out["overall_health"] = cat
    out["reasoning"] = str(obj.get("reasoning", ""))[:500]
    return out


def rate_with_ollama(digest: SessionDigest,
                     model: str = DEFAULT_MODEL,
                     host: str = DEFAULT_HOST,
                     max_retries: int = 2) -> list[Rating]:
    """Rate one session with a local Ollama model. Returns 5 Rating rows."""
    prompt = _build_user_prompt(digest)
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = _post_ollama(host, model, prompt)
            obj = _parse_rating(response)
            normalized = _validate_and_normalize(obj)
            break
        except OllamaError as e:
            last_err = e
            if attempt == max_retries:
                raise
    reasoning = str(normalized.pop("reasoning", ""))
    out: list[Rating] = []
    for dim in DIMENSIONS:
        v = normalized.get(dim)
        out.append(Rating(
            session_id=digest.session_id,
            rater_type="ollama",
            rater_id=model,
            dimension=dim,
            value=v,
            # small local model — never authoritative; "" when abstaining
            confidence="" if v is None else "med",
            reasoning=reasoning,
        ))
    return out


def is_ollama_available(host: str = DEFAULT_HOST) -> bool:
    """Quick liveness check — used by CLI to fail fast with a clear message."""
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False
