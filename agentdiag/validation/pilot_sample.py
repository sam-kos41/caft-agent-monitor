"""Pilot Phase A — draw the frozen N=2000 sample.

Implements docs/PILOT_PREREGISTRATION.md §2 as amended by A1, exactly:

  sel_key_int = int(blake2b(f"{SEED}|{instance_id}|{model_name}")
                     .hexdigest()[:16], 16)
  within each target class, keep the 1000 rows with the SMALLEST
  sel_key_int.

Deterministic, stream-order-independent, O(2000 rows) memory (a
size-1000 max-heap per class evicts the current worst). Requires a
full single pass over the 80,036-row stream — a partial pass would NOT
reproduce the spec'd sample and is therefore not permitted as the real
draw (only `--smoke` does a bounded, clearly-labelled non-spec pass).

Artifacts:
  - <out>/sample.jsonl   full selected rows (LOCAL cache, gitignored)
  - <out>/manifest.csv   instance_id,model_name,target,sel_key_int
                         (committed — reproducibility record)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import itertools
import json
from pathlib import Path

SEED = 20260515
DATASET = "nebius/SWE-agent-trajectories"
PER_CLASS = 1000


def sel_key_int(instance_id: str, model_name: str) -> int:
    h = hashlib.blake2b(
        f"{SEED}|{instance_id}|{model_name}".encode()).hexdigest()
    return int(h[:16], 16)


class _Reservoir:
    """Keep the PER_CLASS rows with the smallest key via a size-capped
    max-heap (root = current worst kept, evicted first)."""

    def __init__(self, cap: int = PER_CLASS):
        self.cap = cap
        self._h: list = []          # (-key, tiebreak, row)
        self._c = itertools.count()

    def offer(self, key: int, row: dict) -> None:
        item = (-key, next(self._c), row)
        if len(self._h) < self.cap:
            heapq.heappush(self._h, item)
        elif -key > self._h[0][0]:        # key < current worst kept
            heapq.heapreplace(self._h, item)

    def rows(self) -> list[tuple[int, dict]]:
        return sorted(((-nk, r) for nk, _, r in self._h),
                      key=lambda x: x[0])


def draw(out_dir: str | Path, smoke: int | None = None) -> dict:
    from datasets import load_dataset

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {True: _Reservoir(), False: _Reservoir()}

    ds = load_dataset(DATASET, split="train", streaming=True)
    n = 0
    for row in ds:
        tgt = bool(row.get("target"))
        k = sel_key_int(str(row.get("instance_id")),
                        str(row.get("model_name")))
        res[tgt].offer(k, row)
        n += 1
        if n % 5000 == 0:
            print(f"  ...streamed {n} rows "
                  f"(kept T={len(res[True]._h)} F={len(res[False]._h)})",
                  flush=True)
        if smoke and n >= smoke:
            print(f"  [SMOKE] stopped at {n} rows — NOT the spec sample")
            break

    sample_path = out / "sample.jsonl"
    manifest_path = out / "manifest.csv"
    counts = {True: 0, False: 0}
    models: dict = {}
    with sample_path.open("w", encoding="utf-8") as sf, \
         manifest_path.open("w", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        w.writerow(["instance_id", "model_name", "target", "sel_key_int"])
        for tgt in (True, False):
            for key, row in res[tgt].rows():
                counts[tgt] += 1
                m = row.get("model_name")
                models[m] = models.get(m, 0) + 1
                sf.write(json.dumps(row, ensure_ascii=False) + "\n")
                w.writerow([row.get("instance_id"), m, tgt, key])

    summary = {
        "dataset": DATASET, "seed": SEED, "streamed_rows": n,
        "selected_true": counts[True], "selected_false": counts[False],
        "model_distribution": models,
        "spec_sample": smoke is None,
        "sample_path": str(sample_path),
        "manifest_path": str(manifest_path),
    }
    (out / "sample_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if smoke is None and (counts[True] < PER_CLASS
                          or counts[False] < PER_CLASS):
        print("WARNING: a class had fewer than "
              f"{PER_CLASS} rows — sample is short, investigate.")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/caft_pilot")
    ap.add_argument("--smoke", type=int, default=None,
                    help="bounded NON-spec pass for plumbing checks only")
    args = ap.parse_args()
    draw(args.out, smoke=args.smoke)


if __name__ == "__main__":
    main()
