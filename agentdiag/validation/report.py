"""Markdown validation report generator.

Reads a Ledger of ratings, computes agreement, and writes a
presentation-ready markdown report:

  - Summary table: kappa per rater pair × dimension, with interpretation
  - Krippendorff alpha per dimension
  - Top disagreements (sessions where raters most diverged) — these
    are the cases worth manually inspecting to refine constructs
  - Per-rater rating distributions (sanity check for floor/ceiling effects)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from agentdiag.validation.agreement import (
    compute_agreement, interpret_kappa,
)
from agentdiag.validation.digest import DIMENSIONS, LIKERT_DIMS
from agentdiag.validation.ledger import Ledger


def _fmt_kappa(k: float) -> str:
    if k != k:
        return "n/a"
    return f"{k:+.2f}"


def _disagreement_score(values: list[int | str]) -> float:
    """Numeric proxy for how much raters disagreed on this session/dim.

    For Likert: range (max - min). For categorical: 0/1 (any disagreement).
    """
    if not values:
        return 0.0
    nums = [v for v in values if isinstance(v, int)]
    if nums:
        return float(max(nums) - min(nums))
    return 0.0 if all(v == values[0] for v in values) else 1.0


def write_report(ledger: Ledger, output_path: str | Path) -> str:
    """Compute agreement and write a markdown report. Returns the path."""
    output_path = Path(output_path)
    agreement = compute_agreement(ledger)
    hi_agreement = compute_agreement(ledger, min_confidence="high")
    raters = agreement["raters"]
    pairs = agreement["rater_pairs"]
    per_dim = agreement["per_dimension"]

    latest = ledger.latest()
    by_session_dim: dict[tuple[str, str], list[tuple[str, int | str]]] = defaultdict(list)
    for (sess, rt, rid, dim), row in latest.items():
        rater = f"{rt}:{rid}"
        by_session_dim[(sess, dim)].append((rater, row["value"]))

    disagreements = []
    for (sess, dim), entries in by_session_dim.items():
        if len(entries) < 2:
            continue
        score = _disagreement_score([v for _, v in entries])
        if score > 0:
            disagreements.append((score, sess, dim, entries))
    disagreements.sort(reverse=True)

    rater_dim_dist: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for (sess, rt, rid, dim), row in latest.items():
        rater_dim_dist[f"{rt}:{rid}"][dim].append(row["value"])

    lines: list[str] = []
    lines.append("# CAFT Validation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Ledger: `{ledger.path}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append("| Rater pair | Sessions in common |")
    lines.append("|---|---:|")
    for (ra, rb), n in agreement["n_sessions_per_pair"].items():
        lines.append(f"| {ra} ↔ {rb} | {n} |")
    lines.append("")

    lines.append("## Cohen's kappa (pairwise, per dimension)")
    lines.append("")
    if not pairs:
        lines.append("_Need at least two raters with overlapping sessions._")
    else:
        header = "| Dimension | " + " | ".join(f"{a} ↔ {b}" for a, b in pairs) + " |"
        sep = "|---|" + "|".join(["---:"] * len(pairs)) + "|"
        lines.append(header)
        lines.append(sep)
        for dim in DIMENSIONS:
            row = [dim]
            for p in pairs:
                k = per_dim[dim]["pair_kappa"].get(p, float("nan"))
                row.append(f"{_fmt_kappa(k)} ({interpret_kappa(k)})")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Krippendorff's α (all raters, per dimension)")
    lines.append("")
    lines.append("| Dimension | α | Interpretation |")
    lines.append("|---|---:|---|")
    for dim in DIMENSIONS:
        a = per_dim[dim]["alpha"]
        lines.append(f"| {dim} | {_fmt_kappa(a)} | {interpret_kappa(a)} |")
    lines.append("")

    lines.append("## Cohen's kappa — HIGH-CONFIDENCE ratings only")
    lines.append("")
    lines.append("Same as above but excluding abstentions AND any rating "
                 "not marked high-confidence. This is the scientifically "
                 "load-bearing number: agreement when the rater was sure.")
    lines.append("")
    hi_pairs = hi_agreement["rater_pairs"]
    hi_per_dim = hi_agreement["per_dimension"]
    if not hi_pairs:
        lines.append("_Not enough high-confidence overlapping ratings yet._")
    else:
        header = "| Dimension | " + " | ".join(f"{a} ↔ {b}" for a, b in hi_pairs) + " |"
        lines.append(header)
        lines.append("|---|" + "|".join(["---:"] * len(hi_pairs)) + "|")
        for dim in DIMENSIONS:
            row = [dim]
            for p in hi_pairs:
                k = hi_per_dim[dim]["pair_kappa"].get(p, float("nan"))
                row.append(f"{_fmt_kappa(k)} ({interpret_kappa(k)})")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Abstentions (\"can't tell\" — a construct-clarity signal)")
    lines.append("")
    lines.append("High abstention on a dimension means it is not reliably "
                 "assessable from the evidence shown — that dimension or the "
                 "digest needs work, independent of any kappa.")
    lines.append("")
    abst = agreement.get("abstentions", {})
    if not abst:
        lines.append("_No abstentions recorded._")
    else:
        lines.append("| Rater | " + " | ".join(DIMENSIONS) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(DIMENSIONS)) + "|")
        for rater in sorted(abst):
            row = [rater] + [str(abst[rater].get(d, 0)) for d in DIMENSIONS]
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Per-rater rating distributions")
    lines.append("")
    lines.append("Quick sanity check for floor/ceiling effects "
                 "(a rater always saying 5, etc.).")
    lines.append("")
    for rater in raters:
        lines.append(f"### {rater}")
        lines.append("")
        lines.append("| Dimension | n | distribution |")
        lines.append("|---|---:|---|")
        for dim in DIMENSIONS:
            vals = rater_dim_dist[rater].get(dim, [])
            if not vals:
                continue
            if dim in LIKERT_DIMS:
                cnt = {i: vals.count(i) for i in (1, 2, 3, 4, 5)}
                bar = " ".join(f"{i}:{cnt[i]}" for i in (1, 2, 3, 4, 5))
            else:
                from collections import Counter
                cnt = Counter(vals)
                bar = " ".join(f"{k}:{v}" for k, v in cnt.most_common())
            lines.append(f"| {dim} | {len(vals)} | {bar} |")
        lines.append("")

    lines.append("## Top disagreements (sessions worth inspecting manually)")
    lines.append("")
    lines.append("Sessions × dimensions where raters diverged most. "
                 "Reviewing these is how you refine the constructs themselves.")
    lines.append("")
    if not disagreements:
        lines.append("_No disagreements yet — need more rated sessions._")
    else:
        lines.append("| Session | Dimension | Spread | Ratings |")
        lines.append("|---|---|---:|---|")
        for score, sess, dim, entries in disagreements[:15]:
            r_str = ", ".join(f"{r}={v}" for r, v in entries)
            lines.append(f"| `{sess}` | {dim} | {score:.0f} | {r_str} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Generated by `agentdiag.validation.report`. "
                 "Re-run after each new human rating to track agreement evolution.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return str(output_path)
