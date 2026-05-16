# Pre-Registration — Leg 2: Cognitive Workload Analogs

**Date:** 2026-05-15
**Status:** PRE-REGISTERED. Committed BEFORE any Leg-2 data-touching
code. Sample, features, statistics, thresholds, and the decision rule
are FROZEN by this commit. Changes require a dated §9 amendment.
Inherits all standing discipline in `PROGRAM.md`.

## 1. Question

Do **cognitive-workload-analog** features, aggregated per trajectory,
(W-H1) separate task outcome above a label-shuffle null, (W-H2) add
predictive power beyond a trivial baseline, and — the leg-defining
test — (W-H3) add predictive power **beyond Leg-1 IT features**
(discriminant validity: workload must not be IT relabeled)?

## 2. Theoretical roots (cited, not invented)

Wickens' Multiple Resource Theory; Cognitive Load Theory (Sweller);
the NASA-TLX measurement tradition (Hart & Staveland). Workload =
resource demand inferred from observable processing signals.

## 3. Sample (frozen — REUSED from Leg 1)

The **exact** Leg-1 frozen sample: the N=2000 instances in
`docs/pilot/sample_manifest.csv` (seed 20260515, 1000/1000). No new
draw. Reuse is mandatory: W-H3 compares workload vs IT on the
*identical* trajectories.

## 4. Observable workload features (frozen)

Computed from the nebius trajectory (`[{role,text}]`). Per-trajectory:

- `reasoning_len.{mean,max,total,slope}` — chars of the `ai` turn's
  thought text *before* its fenced action (sustained-processing proxy,
  Sweller CLT). slope = OLS over turn index.
- `context_cum.{final,slope}` — cumulative sum of all turn text
  lengths up to each step (working-memory-load PROXY, Wickens MRT).
- `reasoning_density.mean` — thought chars per emitted action
  (effort-per-output proxy, NASA-TLX tradition).
- `error_recovery.n_episodes`, `error_recovery.mean_latency_turns` —
  episodes where an observation turn matches an error signature
  (`error|traceback|no such file|command not found|not found|
  exception`), and mean #turns until the next non-error observation
  (resource-reallocation-cost proxy).

**9 workload features.** No others added post-commit.

### 4a. EXCLUDED signals (corpus cannot support — honest scoping)

- **Wall-clock time-per-action / latency / action-density-per-time:**
  the nebius trajectory has **no timestamps**. Excluded. Would require
  a timestamped corpus (the captured Claude Code traces).
- **Reasoning branching factor / considered-options:** the agent emits
  one action per turn; alternatives weighed are not in the data.
  Excluded — not observably computable.
- True token-level context-window utilization is **proxied** by
  cumulative character length (no tokenizer / no raw `query` field in
  nebius); flagged as a proxy, not silently equated.

## 5. Symbolization-audit GATE (runs first; standing rule)

Same procedure as `PILOT_PREREGISTRATION.md` §5/A2: Ridge(alpha=1.0),
per-trajectory tool_name count design matrix, KFold(5, shuffle,
random_state=20260515) CV R². **Gate features:** `reasoning_len.mean`
and `context_cum.final`. Pilot INVALID if CV R² ≥ **0.80** for either.
Mandatory human checkpoint at this gate before W-H1/H2/H3.

## 6. W-H1 — workload vs label-shuffle null

L2 logistic regression, **workload features only**; mean test ROC AUC
under StratifiedKFold(5, random_state=20260515); null = 1000 label
permutations. **PASS iff** observed AUC > null 95th pct **AND** ≥
**0.55**.

## 7. W-H2 — incremental over trivial baseline

`model_B` = baseline (n_turns, n_parsed_actions, patch_len,
exit_status, model_name); `model_BW` = baseline + 9 workload features.
Out-of-fold ΔAUC, 2000-bootstrap instance CI. **PASS iff** mean ΔAUC ≥
**0.03** **AND** 95% CI excludes 0.

## 8. W-H3 — discriminant validity vs Leg 1 (the leg-defining test)

`model_BI` = baseline + 20 Leg-1 IT features; `model_BIW` = baseline +
IT + 9 workload features. Out-of-fold ΔAUC, 2000-boot CI. **PASS iff**
mean ΔAUC ≥ **0.02** **AND** 95% CI excludes 0. Threshold 0.02 (vs
0.03 in W-H2) is pre-stated and frozen: an increment *over a richer
model* is expected smaller; this is not post-hoc. Also report the
workload↔IT feature correlation matrix as descriptive discriminant
evidence (no threshold attached — descriptive only).

## 9. Decision rule (LOCKED — executes itself; audit runs first)

| Outcome | Conclusion | Action |
|---|---|---|
| Audit FAILS | Workload features are tool-API artifacts | Leg-2 INVALID; methods finding; W-H* not interpreted |
| Audit passes, **W-H1 fails** | Workload carries no outcome signal | Document negative leg; proceed to Leg 3 |
| Audit passes, W-H1 passes, **W-H3 fails** | Workload is redundant with Leg-1 IT (not a distinct construct) | Documented honest finding; leg folds into Leg 1; proceed to Leg 3 |
| Audit passes, **W-H1 + W-H3 pass** | Workload is a distinct, validated measurement leg | Leg 2 validated; proceed to Leg 3 |

(W-H2 is reported and informative but the leg-defining gate is W-H3:
distinctness from IT. A leg that only restates Leg 1 is not a leg.)
No interpretation outside this table without a §10 amendment.

## 10. Frozen constants & amendment log

Frozen: sample = Leg-1 manifest (seed 20260515); the 9 workload
features (§4); excluded signals (§4a); audit estimator + gate features
+ R²<0.80; W-H1 floor 0.55 + null 95th pct (1000 perms); W-H2 ΔAUC
≥0.03; W-H3 ΔAUC ≥0.02; both CIs exclude 0; StratifiedKFold(5,
random_state=20260515); 2000-boot CI.

Amendments (dated, with rationale) — none.

## 11. What a pass does NOT claim

A pass means workload-analog features are a distinct, outcome-relevant
measurement on this one agent family — *not* that the agent has
"cognitive load," and not construct validity of the Wickens/Sweller
mapping itself. Same honest ceiling as Leg 1.
