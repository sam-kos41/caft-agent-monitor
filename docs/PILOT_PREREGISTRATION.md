# Pilot Pre-Registration: Population-Level IT Signal vs. Clean Outcomes

**Date:** 2026-05-15
**Status:** PRE-REGISTERED. Written and committed BEFORE any
data-touching analysis code exists. Everything below — sample, feature
lists, statistics, thresholds, decision rule — is FROZEN by this
commit. Changes require a dated amendment with rationale appended to
§9; silent edits are a protocol violation.

This document exists because the entire CAFT arc's credibility came
from refusing to move goalposts after seeing results
(`CONSTRUCT_REVISION.md`, the artifact-guard incidents). The pilot only
means something if its decision rule is fixed in advance.

## 1. Question

Does any information-theoretic metric, **aggregated to one value per
trajectory**, carry information about task success that **(H1)**
exceeds a label-shuffle null and **(H2)** adds predictive power
**beyond a trivial baseline**?

This tests the *population/aggregate* framing only. It does NOT test
per-step single-session detection — that already failed and is not
re-litigated here (`CONSTRUCT_REVISION.md §5b/§5c`).

## 2. Corpus & sample (frozen)

- Source: `nebius/SWE-agent-trajectories` (HuggingFace), 80,036 rows,
  verified 2026-05-15: `target` bool (13,389 True / 66,647 False),
  `model_name` ∈ {swe-agent-llama-8b, -70b, -405b}.
- Outcome label: the in-row `target` (bool) = SWE-bench test-based
  resolved. Definitionally independent of any behavioral metric.
- Sample: **N = 2,000 trajectories**, drawn with **`numpy` seed =
  20260515**, stratified to **1,000 `target=True` / 1,000
  `target=False`** (balanced so the classifier is not dominated by the
  83% failure base rate). `model_name` is NOT balanced (405b too rare)
  — it is recorded and used as a covariate.
- The realized `model_name` distribution and any rows dropped for
  parse failure are reported, not hidden.

## 3. Unit of analysis (frozen)

One trajectory = one data point. All features are aggregate summaries
over the trajectory. No per-step detection, no per-step verdict.

## 4. Features (frozen — no post-hoc additions)

**IT feature set** — from the existing pipeline's per-step
`result["metrics"]`, summarized per trajectory. For each base metric in
`{action_mi, action_entropy, tool_entropy, compression_ratio,
kl_divergence}`: `{mean, final, max, slope}` (slope = OLS slope over
step index). 5 × 4 = **20 IT features**.

**Trivial baseline feature set** — the "could a dumb heuristic do
this" control: `n_turns`, `n_parsed_actions`, `patch_len`
(`len(generated_patch)`), `exit_status` (one-hot), `model_name`
(one-hot).

No other features may be added after this commit.

## 5. Symbolization-audit GATE (runs before H1/H2)

Corpus-specific (the SWE-agent action vocabulary is new). Operational
pass criterion: an L2 logistic/linear model predicting each IT feature
from **tool-name one-hot counts only** must leave substantial residual
variance — **R² < 0.80** for `action_mi.mean` and
`compression_ratio.mean` specifically. If the audit FAILS (IT features
are essentially reconstructible from raw tool-API surface counts), the
pilot is **INVALID**: the result is "instrument confounded by API
surface," NOT a verdict about information theory. Reported as a methods
finding; H1/H2 are not interpreted.

## 6. H1 — separation above a label-shuffle null

- Model: L2 logistic regression, IT features only.
- Statistic: mean ROC AUC over **stratified 5-fold CV** (seed 20260515).
- Null: **≥1,000 label permutations**, full CV pipeline re-run each.
- **Pre-stated pass:** observed mean AUC > 95th percentile of the null
  distribution **AND** observed mean AUC ≥ **0.55** (minimal-effect
  floor — a significant-but-trivial separation does not count).

## 7. H2 — incremental value over the trivial baseline

- `model_B`: LR on baseline features only.
- `model_BI`: LR on baseline + 20 IT features.
- Same CV folds and seed.
- Statistic: ΔAUC = AUC(model_BI) − AUC(model_B), paired across folds;
  95% CI by bootstrap over folds×instances.
- **Pre-stated pass:** mean ΔAUC ≥ **0.03** **AND** the 95% CI of
  ΔAUC excludes 0.

## 8. Decision rule (LOCKED — 4 outcomes)

| Outcome | Conclusion | Action |
|---|---|---|
| Symbolization audit FAILS | Instrument confounded by tool-API surface | Pilot invalid; methods finding only; do NOT infer about IT or the program |
| Audit passes, **H1 fails** | IT carries no population signal even on a clean outcome | **Bank the methodology paper. Do not commit the year.** Clean stop. |
| Audit passes, H1 passes, **H2 fails** | IT real but redundant with trivial features (length etc.) | Documented honest finding; default to banking; forward case is weak |
| Audit passes, **H1 + H2 pass** | IT adds genuine incremental signal on a clean independent outcome | **Commit to the fuller four-leg program**; operationalize the other legs |

No other interpretation is permitted without a §9 amendment.

## 9. Frozen constants & amendment log

Frozen: N=2000; seed=20260515; balance 1000/1000; the 20 IT features
and the baseline feature list (§4); audit R² threshold 0.80 on
action_mi.mean & compression_ratio.mean; H1 floor AUC 0.55 and null
95th-pct rule with ≥1000 permutations; H2 ΔAUC ≥0.03 with 95% CI
excluding 0; stratified 5-fold CV.

Amendments (dated, with rationale):

**A1 — 2026-05-15 — sampling mechanism made exact.**
§2 said the N=2000 sample is "drawn with numpy seed = 20260515,
stratified". This is under-specified for an 80,036-row *streaming*
corpus: `np.random` over a stream whose length is not known a priori
does not define a unique balanced-N sample, and the result would
depend on shard/iteration order. Replaced with an exact,
stream-order-independent, O(2000)-memory mechanism, reproducible from
the seed alone:

  for each row:  sel_key = blake2b(
      f"{20260515}|{instance_id}|{model_name}".encode()).hexdigest()
  within each target class (True, False): select the 1000 rows with
  the lexicographically smallest sel_key.

Seed value (20260515) and the 1000/1000 balance are unchanged. Outcome
balance, feature lists, audit gate, H1/H2, and the decision rule are
unchanged. Rationale: determinism and reproducibility independent of
the HF streaming order; no degrees of freedom added.

**A2 — 2026-05-15 — symbolization-audit estimator made exact.**
§5 pins the model class ("an L2 logistic/linear model"), the gate
threshold (R² < 0.80), and the two gated features (`action_mi.mean`,
`compression_ratio.mean`), but does NOT specify in-sample vs
cross-validated R². That choice materially moves the gate: in-sample
R² over a wide tool-name one-hot basis overfits and would spuriously
*fail* the reconstructability gate. Pinned: estimator =
`sklearn.linear_model.Ridge(alpha=1.0)`; design matrix = per-trajectory
counts of each distinct `tool_name` (one column per tool name observed
in the sample); R² = mean of per-fold test R² under `KFold(n_splits=5,
shuffle=True, random_state=20260515)`. The gate uses this CV R².
Threshold (0.80) and gated features unchanged. Rationale: CV R²
measures genuine reconstructability of an IT feature from tool-API
surface; no threshold or decision-rule change.

## 10. Effort

~2 weeks, one person: adapter (1–2d) → symbolization audit (1d) →
feature extraction (2–3d) → H1 (1d) → H2 (2d) → write-up + decision
(2d). Reuses `adapters/` pattern, `signals.py` command parsing,
`eval/stats.py` permutation/bootstrap, and the artifact-guard
discipline throughout.

## 11. What this pilot does NOT claim

A pass is not "CAFT works." It is "the aggregate IT framing has signal
worth a fuller, equally pre-registered study." A fail is a clean,
publishable negative that strengthens the methodology contribution.
Either way the methodology paper is the guaranteed asset.
