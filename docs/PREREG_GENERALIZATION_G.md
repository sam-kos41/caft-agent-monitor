# Pre-Registration — Generalization Discriminant G: Outcome Granularity

**Date:** 2026-05-16
**Status:** DRAFT FOR HUMAN REVIEW. No G-analysis code is written or
run until reviewed and committed. On commit:
sample/parse-rule/features/statistics/thresholds/decision-rule FROZEN;
changes require a dated §9 amendment. Inherits all standing discipline
in `PROGRAM.md`.

## 0. The question this answers

The program established (binary test-pass outcome, nebius SWE-agent /
Llama): IT behavioral structure is load-bearing; 5/5 other constructs
collapse into it. Two rival explanations for the collapse:
(H_flat) agents are genuinely "flat" — externalized behavior does not
decompose into separable functional dimensions; or
(H_artifact) the *binary* outcome is too lossy to resolve separable
dimensions. These have opposite implications for everything built next.
G tests **outcome-granularity generalization only** (one axis): does
the collapse survive a *graded* outcome on the *same* frozen sample?
It does NOT test agent-architecture or corpus generalization — that is
a separate, explicitly-future axis.

## 1. Precondition — VERIFIED MET (data on the record)

Honest-scoping check completed 2026-05-16 before drafting this:
- Graded outcome `g = passed / (passed + failed + error)` is
  deterministically parseable from `eval_logs` via the FROZEN regexes
  `(\d+)\s+passed`, `(\d+)\s+failed`, `(\d+)\s+error` (case-insensitive;
  first match of each; row excluded if passed+failed+error == 0 or no
  match).
- Parseable for **1,696 / 2,000** frozen-sample rows.
- Validity: `corr(g, binary target) = 0.539`; mean g|target=1 = 0.974,
  mean g|target=0 = 0.689 → a genuine graded refinement of the same
  resolution outcome, not a non-authoritative test run.

**Documented limitations (carried into every claim G produces):**
1. **Selection bias.** Parse coverage is asymmetric: 971/1000
   resolved vs 725/1000 unresolved (failed-early runs have no test
   summary). The G analysis set over-represents successes and is
   conditional on "the run reached test execution." G claims are
   explicitly conditional on this subpopulation.
2. **Granularity concentrated in the unresolved region.** Resolved is
   saturated (g≈1, 89% exactly 1.0); the only place graded outcome
   adds resolving power over binary is among unresolved tasks. G's
   real sensitivity is "does a construct re-separate via partial
   credit on not-fully-resolved runs."

## 2. Analysis set (frozen)

The deterministically-parseable subset of the existing frozen 2,000
sample (the rows where the §1 rule yields a defined `g`). Exact n
reported on execution (expected ≈1,696). No new data, no new draw.
Features are the EXACT cached, already-symbolization-gated feature
sets from Legs 1-4 + TAC (`/tmp/caft_pilot/{features,workload,sa,
error,tac}.json`), aligned positionally by row (the alignment the
primary pipeline uses; 0 mismatches previously verified). **No new
symbolization gate**: feature sets are unchanged and were each gated
in their own leg; only the dependent variable changes.

## 3. Statistic family (adapted for a continuous outcome, pre-stated)

Binary AUC does not apply to continuous `g`. Frozen analogs:
- Estimator: `Ridge(alpha=1.0)` (same family as the audits).
- Metric: out-of-fold **Spearman** correlation between predicted and
  actual `g`, under `KFold(5, shuffle=True, random_state=20260515)`.
- Nested comparison statistic: ΔSpearman (OOF), 2000-bootstrap
  instance CI.
- Null (for the standalone test): 1000 permutations of `g`.
These thresholds are deliberate analogs of the binary AUC family,
chosen identical-in-spirit for cross-result comparability (a stated
choice, not a re-derivation; no per-test tuning).

## 4. Hypotheses (locked)

- **G-H1 (IT predicts graded outcome).** IT-only Ridge; OOF Spearman
  vs `g`. PASS iff observed > null 95th pct **AND** |Spearman| ≥
  **0.10**.
- **G-H2 (IT beyond trivial baseline).** baseline (n_turns,
  n_parsed_actions, patch_len, exit_status, model_name) vs
  baseline+IT. ΔSpearman, 2000-boot CI. PASS iff ΔSpearman ≥ **0.03**
  AND 95% CI excludes 0.
- **G-H3 (the decisive test — does ANY construct re-separate under
  graded outcome).** For EACH of the four collapsed constructs
  {workload, SA, error_recovery, TAC}: baseline+IT vs
  baseline+IT+construct. ΔSpearman, 2000-boot CI. A construct
  "re-separates" iff ΔSpearman ≥ **0.02** AND 95% CI excludes 0.
  Multiplicity: the per-construct locked rule is the headline; the
  Bonferroni-adjusted (α/4) CI is ALSO reported. A construct counts
  as re-separated for the §5 decision only if it clears the locked
  rule; Bonferroni survival is reported as strength-of-evidence, not
  a second gate.

## 5. Decision rule (LOCKED — executes itself)

| Outcome | Conclusion | Action |
|---|---|---|
| **G-H1 fails** | IT does not even predict the graded outcome → the binary IT result may be binarization-specific | Major caveat on the whole program; document; reassess before any build |
| G-H1 & G-H2 pass, **no construct re-separates (G-H3)** | Parsimony generalizes across outcome granularity. H_flat supported over H_artifact (on this corpus/axis). | Strong corroboration; document; the IT-load-bearing finding stands |
| G-H1 & G-H2 pass, **≥1 construct re-separates** | The collapse was partly a binarization artifact: {named construct(s)} carry distinct signal under partial-credit grading | Paper-defining reframe; the "agents are flat" interpretation is **rejected**; name the construct(s); reassess directions |

No interpretation outside this table without a §9 amendment. All
conclusions are explicitly conditional on the §1 selection bias and
single corpus/agent family — G is an outcome-granularity test, not a
generalization claim across agents.

## 6. What G does NOT establish

Not agent-architecture generalization, not corpus generalization, not
a per-instance/early-warning claim (that regime is contraindicated by
the program's own negatives). G answers exactly one rival-explanation
question and nothing more.

## 7-8. (reserved)

## 9. Frozen constants & amendment log

Frozen on commit: parse regexes + g definition + exclusion rule (§1);
analysis set = parseable subset of the frozen 2000; feature caches as
listed (§2); Ridge(alpha=1.0); KFold(5, random_state=20260515); OOF
Spearman metric; G-H1 floor |ρ|≥0.10 + null 95th pct (1000 perms);
G-H2 ΔSpearman≥0.03 CI excl 0; G-H3 per-construct ΔSpearman≥0.02 CI
excl 0 over {workload,SA,error,TAC}; 2000-boot CI; Bonferroni α/4
reported not gating.

Amendments (dated, with rationale) — none.

## 10. Review sign-off

[ ] Human review complete — precondition limits acceptable, graded
outcome valid enough, statistic family + thresholds + decision rule
acceptable. → then, and only then, build `pilot_generalization.py`
+ tests, run G, let the locked rule fire.
