"""Inter-rater agreement statistics for the validation harness.

All implementations use the standard library plus numpy. No scipy
dependency — keeps the install footprint small and matches the rest
of CAFT's stats infrastructure.

Metrics provided:
  - Cohen's kappa (pairwise, ordinal-aware via linear weights)
  - Krippendorff's alpha (handles >2 raters and missing data)
  - Spearman rank correlation (for Likert dimensions)
  - Bootstrap confidence intervals (uses agentdiag/eval/stats.py)

Interpretation thresholds (Landis & Koch 1977):
  < 0.00  poor
  0.00–0.20  slight
  0.21–0.40  fair
  0.41–0.60  moderate
  0.61–0.80  substantial
  0.81–1.00  almost perfect
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence
import numpy as np

from agentdiag.validation.digest import LIKERT_DIMS, CATEGORICAL_DIMS, DIMENSIONS
from agentdiag.validation.ledger import Ledger


def cohens_kappa(rater_a: Sequence[int | str],
                 rater_b: Sequence[int | str],
                 weights: str = "linear") -> float:
    """Cohen's kappa between two raters on the same N items.

    For Likert (1-5) data, weights='linear' gives ordinal-aware kappa
    that penalizes adjacent disagreements less than far ones.
    For categorical data, pass weights='none'.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("rater sequences must be the same length")
    if not rater_a:
        return float("nan")

    categories = sorted(set(list(rater_a) + list(rater_b)),
                        key=lambda x: (isinstance(x, str), x))
    k = len(categories)
    if k < 2:
        return 1.0
    cat_idx = {c: i for i, c in enumerate(categories)}

    obs = np.zeros((k, k), dtype=float)
    for a, b in zip(rater_a, rater_b):
        obs[cat_idx[a]][cat_idx[b]] += 1.0
    n = obs.sum()
    obs /= n

    row = obs.sum(axis=1)
    col = obs.sum(axis=0)
    exp = np.outer(row, col)

    if weights == "linear":
        w = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                w[i][j] = abs(i - j) / (k - 1)
    elif weights == "none":
        w = 1.0 - np.eye(k)
    else:
        raise ValueError(f"unknown weights: {weights!r}")

    num = (w * obs).sum()
    den = (w * exp).sum()
    if den == 0:
        return 1.0
    return 1.0 - (num / den)


def krippendorff_alpha(ratings: dict[str, dict[str, int | str]],
                       level: str = "ordinal") -> float:
    """Krippendorff's alpha across any number of raters with missing data.

    Args:
      ratings: {rater_id: {item_id: value}}
      level: "ordinal" (Likert) or "nominal" (categorical)
    """
    items: set[str] = set()
    for r in ratings.values():
        items.update(r.keys())
    items_list = sorted(items)
    if not items_list:
        return float("nan")

    matrix: list[list[int | str | None]] = []
    for it in items_list:
        row = [ratings[r].get(it) for r in ratings]
        matrix.append(row)

    pairs: list[tuple] = []
    for row in matrix:
        present = [v for v in row if v is not None]
        if len(present) < 2:
            continue
        for i in range(len(present)):
            for j in range(len(present)):
                if i != j:
                    pairs.append((present[i], present[j]))

    if not pairs:
        return float("nan")

    all_vals = [v for row in matrix for v in row if v is not None]
    cats = sorted(set(all_vals), key=lambda x: (isinstance(x, str), x))

    def delta(a, b) -> float:
        if level == "nominal":
            return 0.0 if a == b else 1.0
        ia, ib = cats.index(a), cats.index(b)
        return ((ia - ib) / max(len(cats) - 1, 1)) ** 2

    obs_disagree = sum(delta(a, b) for a, b in pairs) / len(pairs)

    n = len(all_vals)
    exp_pairs = 0.0
    cnt = 0
    for i, a in enumerate(all_vals):
        for j, b in enumerate(all_vals):
            if i == j:
                continue
            exp_pairs += delta(a, b)
            cnt += 1
    exp_disagree = exp_pairs / max(cnt, 1)

    if exp_disagree == 0:
        return 1.0
    return 1.0 - (obs_disagree / exp_disagree)


def spearman_rho(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation. Returns NaN if either side is constant."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    if np.std(ar) == 0 or np.std(br) == 0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def interpret_kappa(k: float) -> str:
    if np.isnan(k): return "n/a"
    if k < 0: return "poor"
    if k < 0.21: return "slight"
    if k < 0.41: return "fair"
    if k < 0.61: return "moderate"
    if k < 0.81: return "substantial"
    return "almost perfect"


_CONF_RANK = {"": 0, "low": 1, "med": 2, "high": 3}


def compute_agreement(ledger: Ledger, min_confidence: str = "") -> dict:
    """End-to-end agreement computation across all rater pairs / dimensions.

    Abstentions (value is None) are excluded — never guessed. If
    min_confidence is set ("low"/"med"/"high"), ratings below that
    confidence are also excluded, so you can compute agreement on the
    high-confidence subset separately.

    Returns nested dict:
      {
        "rater_pairs": [("human:sam", "ollama:llama3.2:3b"), ...],
        "per_dimension": {
          dim: {
            "pair_kappa": {(rA, rB): kappa},
            "alpha": float,
          }
        },
        "n_sessions_per_pair": {(rA, rB): int},
        "abstentions": {rater: {dim: count}},
        "min_confidence": str,
      }
    """
    latest = ledger.latest()
    conf_floor = _CONF_RANK.get(min_confidence, 0)

    abstentions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_rater_dim: dict[tuple[str, str], dict[str, int | str]] = defaultdict(dict)
    for (sess, rt, rid, dim), row in latest.items():
        rater = f"{rt}:{rid}"
        val = row.get("value")
        if val is None:
            abstentions[rater][dim] += 1
            continue
        if _CONF_RANK.get(row.get("confidence", ""), 0) < conf_floor:
            continue
        by_rater_dim[(rater, dim)][sess] = val

    raters: list[str] = sorted({r for (r, _d) in by_rater_dim.keys()})

    per_dim: dict[str, dict] = {}
    for dim in DIMENSIONS:
        pair_kappa: dict[tuple[str, str], float] = {}
        per_rater = {r: by_rater_dim.get((r, dim), {}) for r in raters}
        for i, ra in enumerate(raters):
            for rb in raters[i + 1:]:
                shared = sorted(set(per_rater[ra]) & set(per_rater[rb]))
                if not shared:
                    pair_kappa[(ra, rb)] = float("nan")
                    continue
                a_vals = [per_rater[ra][s] for s in shared]
                b_vals = [per_rater[rb][s] for s in shared]
                w = "none" if dim in CATEGORICAL_DIMS else "linear"
                pair_kappa[(ra, rb)] = cohens_kappa(a_vals, b_vals, weights=w)

        level = "nominal" if dim in CATEGORICAL_DIMS else "ordinal"
        alpha = krippendorff_alpha(per_rater, level=level)
        per_dim[dim] = {"pair_kappa": pair_kappa, "alpha": alpha}

    def _rater_sessions(rater: str) -> set[str]:
        out: set[str] = set()
        for dim in DIMENSIONS:
            out |= set(by_rater_dim.get((rater, dim), {}).keys())
        return out

    n_sessions_per_pair: dict[tuple[str, str], int] = {}
    for i, ra in enumerate(raters):
        ra_sess = _rater_sessions(ra)
        for rb in raters[i + 1:]:
            n_sessions_per_pair[(ra, rb)] = len(ra_sess & _rater_sessions(rb))

    return {
        "raters": raters,
        "rater_pairs": [(raters[i], raters[j])
                        for i in range(len(raters))
                        for j in range(i + 1, len(raters))],
        "per_dimension": per_dim,
        "n_sessions_per_pair": n_sessions_per_pair,
        "abstentions": {r: dict(d) for r, d in abstentions.items()},
        "min_confidence": min_confidence,
    }
