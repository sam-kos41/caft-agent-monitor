# Pre-Registration — Leg 4: Error Recovery & Adaptive Behavior

**Date:** 2026-05-15
**Status:** DRAFT FOR HUMAN REVIEW. No Leg-4 data-touching code exists
or will be written until this is reviewed and committed. Once
committed, sample/features/statistics/thresholds/decision-rule are
FROZEN; changes require a dated §10 amendment. Inherits all standing
discipline in `PROGRAM.md`.

## 0. Framing — confirmatory test of the parsimony thesis (stated prior, not pre-committed)

After Legs 2 and 3 both reduced to Leg-1 IT (n=2 convergent
discriminant negatives, incl. the strongest candidate, SA), the
program's working thesis is: *IT behavioral structure is the
load-bearing construct; HF analogs reduce to it.* Leg 4 is run **as a
confirmatory test of that thesis**, with an explicit **stated prior
that error recovery also reduces to IT**.

BUT — held genuinely open: error recovery is conceptually the **most
distinct axis from IT** of all four legs. IT measures the *steady-state
shape* of an action sequence; error recovery is about *discrete
adverse events and the agent's adaptive response to them* — a
different object entirely. If any HF analog is going to survive as
distinct, this is the candidate. The pre-registered test fires; the
locked rule decides; no outcome is pre-committed. A leg-defining pass
re-opens the parsimony thesis and is paper-defining the other way; a
fail completes it and makes it airtight.

**This is the final empirical leg. Regardless of outcome, the
empirical phase ends here and methodology-paper drafting begins.**

## 1. Question

Do **error-recovery / adaptive-behavior** features, aggregated per
trajectory, (E-H1) separate outcome above a label-shuffle null, (E-H2)
add predictive power beyond a trivial baseline, and — leg-defining —
(E-H3) add predictive power **beyond Leg-1 IT** (discriminant
validity)?

## 2. Theoretical roots (cited)

Reason's human-error taxonomy (slips/lapses/mistakes; error detection
& recovery); Hollnagel's resilience engineering / FRAM (graceful vs
brittle response to disturbance); adaptive-automation literature
(Parasuraman, Sheridan, Wickens). Construct: how the agent *responds
to adverse events* — detect, change strategy, recover, or fail
catastrophically.

## 3. Sample (frozen — REUSED)

The exact Leg-1/2/3 frozen sample (`docs/pilot/sample_manifest.csv`,
seed 20260515, 1000/1000). No new draw. E-H3 compares ER vs IT on
identical trajectories (caches align by row; verified for IT/WL/SA,
same for ER).

## 4. Observable error-recovery features (frozen)

Deterministic from the parsed action stream + observation turns. An
**error episode** = an observation (`user`) turn, after the initial
issue, whose text matches the FROZEN signature:

    error | traceback | no such file | command not found |
    not found | exception | failed | cannot | invalid syntax |
    permission denied                                  (case-insensitive)

Consecutive error observations with no intervening action collapse to
one episode. For each episode: the **failing action** = the last
parsed action (`tool_name`,`target`) before the error obs; the
**response action** = the next parsed action after it.

**6 features** (Reason/Hollnagel-grounded; deliberately NOT
IT-flavored — no entropy/MI/compression, which would be Leg 1):

- `error.n_episodes` — count of error episodes.
- `error.recurrence_rate` — repeated identical failing
  (tool,target) signatures / n_episodes (Reason: same mistake
  recurring = poor error detection). 0 if no episodes.
- `error.strategy_change_rate` — fraction of episodes whose response
  action differs (tool OR target) from the failing action (adaptive
  shift vs blind identical retry). 0 if no episodes.
- `error.recovery_success_rate` — fraction of episodes followed
  within **K=3** turns by a non-error observation (frozen K).
- `error.mean_latency_turns` — mean turns from error obs to the next
  non-error obs (turn-based; see §4a). 0 if no episodes.
- `error.terminal_unresolved` — 1 iff the trajectory's last
  observation is an error and no `submit` occurs after it
  (catastrophic vs graceful close). Distinct from baseline
  `exit_status`, which stays in the trivial baseline only.

### 4a. EXCLUDED / scoped (honest-scoping rule)

- **Wall-clock recovery latency** — nebius has no timestamps.
  Excluded. `error.mean_latency_turns` is a *turn-count* surrogate,
  explicitly not a time measure; included as such, not as latency.
- **Semantic error-acknowledgement** (does the agent's *thought*
  reference the error) — requires NLP over thought text = unvalidated
  instrument. Excluded (same rule that excluded SA-L2).
- Overlap note: `signals.py` and Leg-2 contained crude error
  primitives (`error_retry_cycles`, `error_recovery.*`). Leg 4
  supersedes them with the Reason/Hollnagel-grounded set above; the
  discriminant test is vs **Leg-1 IT**, not vs the collapsed Leg 2.

## 5. Symbolization-audit GATE (runs first; standing rule)

Same procedure (`PILOT_PREREGISTRATION.md` §5/A2): Ridge(alpha=1.0),
per-trajectory tool_name count design matrix, KFold(5, shuffle,
random_state=20260515) CV R². **Gate features (the adaptive/relational
pair, per the Leg-3 lesson):** `error.strategy_change_rate` and
`error.recovery_success_rate`. Leg-4 INVALID if CV R² ≥ **0.80** for
either. (`n_episodes`/`mean_latency_turns` are more volume-like by
construction and are intentionally NOT the gate features.) **Mandatory
human checkpoint at this gate before E-H1/H2/H3.**

## 6. Hypotheses & thresholds

Thresholds are **identical to Legs 2 and 3 by deliberate choice**:
cross-leg comparability is the whole point of a parsimony argument;
a leg-specific threshold would introduce an analytic degree of freedom
and undermine the comparison. This is the defensible choice for error
recovery *specifically* — not laziness, but the requirement that the
four legs be judged on one ruler.

- **E-H1** — ER-only L2 logistic; mean test ROC AUC, StratifiedKFold
  (5, random_state=20260515); null = 1000 label permutations. **PASS
  iff** observed > null 95th pct **AND** ≥ **0.55**.
- **E-H2** — `model_B` = baseline (n_turns, n_parsed_actions,
  patch_len, exit_status, model_name); `model_BE` = baseline + 6 ER
  features. OOF ΔAUC, 2000-bootstrap instance CI. **PASS iff** mean
  ΔAUC ≥ **0.03** **AND** 95% CI excludes 0.
- **E-H3 (LEG-DEFINING)** — `model_BI` = baseline + 20 Leg-1 IT;
  `model_BIE` = baseline + IT + 6 ER. OOF ΔAUC, 2000-boot CI. **PASS
  iff** mean ΔAUC ≥ **0.02** **AND** 95% CI excludes 0. Also report
  max |corr| ER-vs-IT (descriptive, no threshold).

## 7. Decision rule (LOCKED — executes itself; audit first; FINAL leg)

| Outcome | Conclusion | Action |
|---|---|---|
| Audit FAILS | ER features are tool-API artifacts | Leg-4 INVALID; methods finding; E-H* not interpreted; **end empirical phase → paper** |
| Audit passes, **E-H1 fails** | ER carries no outcome signal | Document negative leg; **end empirical phase → paper** |
| Audit passes, E-H1 passes, **E-H3 fails** | ER redundant with Leg-1 IT — not distinct. **Completes the parsimony result: all 3 tested HF analogs reduce to IT (airtight).** | Documented; **end empirical phase → paper** |
| Audit passes, **E-H1 + E-H3 pass** | ER is a distinct, validated leg. **RE-OPENS the parsimony thesis** — a genuine second construct exists. Paper-defining the other way. | Document; **end empirical phase → paper** (thesis reframed) |

Every branch ends the empirical phase and proceeds to methodology
paper drafting (per program instruction). No interpretation outside
this table without a §10 amendment.

## 8. Frozen constants & amendment log

Frozen on commit: sample = Leg-1 manifest (seed 20260515); the error
signature regex (§4); K=3; the 6 ER features (§4); exclusions (§4a);
gate features strategy_change_rate + recovery_success_rate, R²<0.80;
E-H1 floor 0.55 + null 95th pct (1000 perms); E-H2 ΔAUC ≥0.03; E-H3
ΔAUC ≥0.02; both CIs exclude 0; StratifiedKFold(5,
random_state=20260515); 2000-boot CI.

Amendments (dated, with rationale) — none.

## 9. What a pass/fail does NOT claim

A pass: ER-analog features are a distinct outcome-relevant measurement
on this one agent family — not that the agent "recovers from errors"
in a cognitive sense, nor construct validity of the Reason/Hollnagel
mapping. A collapse-into-IT completes the parsimony result *on this
corpus and these operationalizations* — not a universal proof.

## 10. Review sign-off

[ ] Human review complete — features defensible, exclusions honest,
thresholds justified, decision rule acceptable. → then, and only then,
build `pilot_error.py` + tests, run the gate, stop for the gate
checkpoint.
