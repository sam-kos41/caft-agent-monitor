"""TAC Robustness 2 — LLM-graded, NON-GATING, descriptive only.

PREREG_AGENT_NATIVE_TAC.md §3 Robustness 2. Frozen sub-sample = first
150 trajectories by sel_key_int (manifest order). The local Ollama
model rates overall thought-action correspondence 1-5 from a compact
digest of (thought, action) pairs. Reported as correlation with the
PRIMARY deterministic `tac.mean`. By construction this CANNOT change
the locked §8 decision (already recorded, commit 259a470) — it is a
convergent-evidence descriptor: does an independent grader's view of
coherence track the deterministic measure?

Robustness 1 (semantic embedding) is documented-EXCLUDED:
sentence-transformers unavailable offline (honest-scoping, no faked
proxy).
"""

from __future__ import annotations

import csv
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

from agentdiag.adapters.swe_agent import _extract_action, _FENCE
from agentdiag.validation.rate_ollama import (
    is_ollama_available, DEFAULT_MODEL, DEFAULT_HOST,
)

SUBSAMPLE_N = 150


def _thought(ai_text: str) -> str:
    return _FENCE.sub("", ai_text or "").strip()


def _pairs(trajectory: list, k: int = 6) -> list[tuple[str, str]]:
    out = []
    for t in trajectory or []:
        if t.get("role") not in ("ai", "assistant"):
            continue
        a = _extract_action(t.get("text", ""))
        if a is None:
            continue
        th = " ".join(_thought(t["text"]).split())[:240]
        out.append((th, f"{a[0]} {a[1]}".strip()[:120]))
    if len(out) <= k:
        return out
    idx = np.linspace(0, len(out) - 1, k).astype(int)
    return [out[i] for i in idx]


_SYS = ("You rate how consistently an AI agent's stated intent matches "
        "the action it then takes. Output ONLY JSON: {\"tac\": N} where "
        "N is 1 (intent and action rarely match) to 5 (intent and "
        "action consistently match). No prose.")


def _llm_tac(pairs, model, host) -> int | None:
    """Own minimal Ollama call with the TAC-specific system prompt
    (deliberately NOT reusing rate_ollama._post_ollama, which injects
    the unrelated validation-rater system prompt)."""
    body = "\n".join(f"- THOUGHT: {th}\n  ACTION: {ac}" for th, ac in pairs)
    prompt = (f"Turns (thought then action taken):\n{body}\n\n"
              "Rate overall thought-action consistency 1-5 as JSON.")
    payload = {
        "model": model, "prompt": prompt, "system": _SYS,
        "stream": False, "format": "json",
        "options": {"temperature": 0.0, "num_predict": 40},
    }
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            env = json.loads(resp.read().decode("utf-8"))
        raw = env.get("response", "")
    except Exception:
        return None
    m = re.search(r'"tac"\s*:\s*([1-5])', raw)
    if m:
        return int(m.group(1))
    m2 = re.search(r"\b([1-5])\b", raw)
    return int(m2.group(1)) if m2 else None


def run(out_dir: str, model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST) -> dict:
    base = Path("/tmp/caft_pilot")
    # frozen subsample: first SUBSAMPLE_N instance_ids by sel_key_int
    rows = []
    with open("/Users/samkoscelny/GazeVLM-local/agentdiag/docs/pilot/"
              "sample_manifest.csv") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["sel_key_int"]), r["instance_id"]))
    rows.sort()
    keep = {iid for _, iid in rows[:SUBSAMPLE_N]}

    # join trajectories + primary tac.mean by instance_id (sample.jsonl
    # and tac.json are in the same row order)
    traj_by_id, tacmean_by_id = {}, {}
    tac = json.loads((base / "tac.json").read_text())
    with open(base / "sample.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            iid = d.get("instance_id")
            if iid in keep:
                traj_by_id[iid] = d.get("trajectory") or []
                tacmean_by_id[iid] = tac[i]["tac"]["tac.mean"]

    if not is_ollama_available(host):
        res = {"status": "ollama_unavailable",
                "note": "Robustness 2 skipped — Ollama down. Non-gating; "
                        "primary decision (259a470) unaffected."}
        (base / "tac_robustness.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return res

    llm, det, graded, failed = [], [], 0, 0
    for iid in sorted(keep):
        if iid not in traj_by_id:
            continue
        r = _llm_tac(_pairs(traj_by_id[iid]), model, host)
        if r is None:
            failed += 1
            continue
        llm.append(r)
        det.append(tacmean_by_id[iid])
        graded += 1
        if graded % 25 == 0:
            print(f"  ...graded {graded}", flush=True)

    llm = np.array(llm, float)
    det = np.array(det, float)
    pear = float(np.corrcoef(llm, det)[0, 1]) if len(llm) > 2 else float("nan")
    lr = np.argsort(np.argsort(llm))
    dr = np.argsort(np.argsort(det))
    spear = (float(np.corrcoef(lr, dr)[0, 1])
             if len(llm) > 2 and llm.std() > 0 and det.std() > 0
             else float("nan"))
    res = {
        "status": "ok", "model": model,
        "subsample_n": SUBSAMPLE_N, "graded": graded, "failed": failed,
        "pearson_llm_vs_deterministic_tac": pear,
        "spearman_llm_vs_deterministic_tac": spear,
        "llm_mean": float(llm.mean()) if len(llm) else None,
        "deterministic_tac_mean": float(det.mean()) if len(det) else None,
        "note": ("DESCRIPTIVE, NON-GATING. Convergent check only: does an "
                 "independent LLM grader's coherence rating track the "
                 "deterministic primary measure? Cannot and does not "
                 "alter the locked §8 decision (TAC reduces to IT)."),
    }
    (base / "tac_robustness.json").write_text(json.dumps(res, indent=2))
    md = "\n".join([
        "# TAC Robustness 2 — LLM-graded (descriptive, NON-gating)", "",
        f"- model {model}, frozen subsample first {SUBSAMPLE_N} by "
        f"sel_key_int; graded {graded}, failed {failed}",
        f"- Pearson(LLM rating, deterministic tac.mean) = "
        f"**{pear:.3f}**",
        f"- Spearman = **{spear:.3f}**",
        f"- means: LLM {res['llm_mean']:.2f}/5 vs deterministic "
        f"tac.mean {res['deterministic_tac_mean']:.3f}", "",
        "Robustness 1 (semantic embedding): documented-EXCLUDED "
        "(sentence-transformers unavailable offline; honest-scoping).",
        "", "_Convergent evidence only. The locked §8 decision (TAC-H3 "
        "fail; TAC reduces to IT) was recorded before this ran and is "
        "not affected._"])
    (base / "tac_robustness.md").write_text(md)
    print(md)
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/caft_pilot")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    run(a.out, model=a.model)


if __name__ == "__main__":
    main()
