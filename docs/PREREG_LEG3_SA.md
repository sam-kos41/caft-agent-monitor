# Pre-Registration — Leg 3: Situation Awareness Analogs

**Date:** 2026-05-15
**Status:** PRE-REGISTERED. Committed BEFORE any Leg-3 data-touching
code. Sample, features, statistics, thresholds, decision rule FROZEN.
Inherits all standing discipline in `PROGRAM.md`. Changes require a
dated §10 amendment.

Framing note (per program guidance): the Leg-2 negative is the
framework working — a clean discriminant-validity finding, not a
setback. The 4-leg-vs-IT-core meta-question is held OPEN; Leg 3 is the
strongest remaining distinctness candidate and either outcome
(distinct, or collapses into IT) is a paper-defining result. No
framing is pre-committed.

## 1. Question

Do **situation-awareness-analog** features, aggregated per trajectory,
(SA-H1) separate outcome above a label-shuffle null, (SA-H2) add
predictive power beyond a trivial baseline, and — leg-defining —
(SA-H3) add predictive power **beyond Leg-1 IT** (discriminant
validity)? SA is conceptually the strongest distinctness candidate: it
concerns the agent's *relationship to its environment* (did it look at
what it changed; did it check consequences), not the *shape of its
action sequence* (IT).

## 2. Theoretical roots (cited)

Endsley's three-level SA model: L1 perception, L2 comprehension, L3
projection. SAGAT / SART measurement tradition.

## 3. Sample (frozen — REUSED)

The exact Leg-1/Leg-2 frozen sample (`docs/pilot/sample_manifest.csv`,
seed 20260515, 1000/1000). No new draw. Reuse mandatory: SA-H3
compares SA vs IT on identical trajectories (caches align by row,
verified for IT/workload; same for SA).

## 4. Observable SA features (frozen) — L1 + L3 only; L2 EXCLUDED

All features are deterministic from the parsed action stream
(`tool_name`, `target_path` from `adapters/swe_agent.py`). **No NLP,
no lexical-style heuristics** — those would be unvalidated instruments
(honest-scoping rule).

**Level 1 — Perception (4 features):** did the agent observe the
things it acted on?
- `perception.coverage` — fraction of distinct edit/create targets
  that were observed earlier (open/search_dir/search_file/find_file/
  grep/cat/scroll/goto on that target) before being mutated. [0,1].
- `perception.explore_ratio` — observation actions / total actions.
- `perception.read_before_first_edit` — observation actions before the
  first mutation / steps-until-first-mutation (orienting before
  acting); 0 if a mutation is the first action.
- `perception.blind_edit_rate` — edit/create actions whose target was
  never previously observed, / total actions.

**Level 3 — Projection (2 features):** did the agent check consequences
before committing?
- `projection.verify_before_submit` — 1 iff ≥1 verification action
  (python/pytest/test/grep on code) strictly after the last
  edit/create and before `submit`; else 0.
- `projection.verify_rate` — verification-type actions / total actions.

**6 SA features.** No others added post-commit.

### 4a. EXCLUDED — Level 2 Comprehension (documented)

Endsley L2 (integrating perception into a situation *model*: does the
agent's understanding match observed state, vs. acting on a
hallucinated model) is **excluded as a scored construct**. It cannot
be operationalized cleanly from nebius trace text without inferring
the agent's internal situation model — which requires semantic NLP
over the `thought` text. Any such proxy would itself be an
unvalidated measurement instrument, violating the honest-scoping
rule. Per program guidance, two well-grounded levels (L1, L3) with L2
documented-excluded is more defensible than three weak proxies. L2 is
future work requiring a corpus with ground-truth state annotations.

## 5. Symbolization-audit GATE (runs first; standing rule)

Same procedure (`PILOT_PREREGISTRATION.md` §5/A2): Ridge(alpha=1.0),
per-trajectory tool_name count design matrix, KFold(5, shuffle,
random_state=20260515) CV R². **Gate features chosen deliberately:**
`perception.coverage` and `projection.verify_before_submit` — the two
*relational/ordering* SA features (do they encode something beyond
tool composition?). `explore_ratio`/`verify_rate` are more
composition-like by construction and are intentionally NOT the gate
features; if the relational features are reconstructible from tool
counts (CV R² ≥ **0.80** for either), Leg-3 is INVALID. Mandatory
human checkpoint at this gate before SA-H1/H2/H3.

## 6. SA-H1 — SA vs label-shuffle null

L2 logistic, SA features only; mean test ROC AUC under
StratifiedKFold(5, random_state=20260515); null = 1000 label
permutations. **PASS iff** observed > null 95th pct **AND** ≥ **0.55**.

## 7. SA-H2 — incremental over trivial baseline

`model_B` = baseline (n_turns, n_parsed_actions, patch_len,
exit_status, model_name); `model_BS` = baseline + 6 SA features.
Out-of-fold ΔAUC, 2000-bootstrap instance CI. **PASS iff** mean ΔAUC ≥
**0.03** **AND** 95% CI excludes 0.

## 8. SA-H3 — discriminant vs Leg 1 (LEG-DEFINING)

`model_BI` = baseline + 20 Leg-1 IT features; `model_BIS` = baseline +
IT + 6 SA features. Out-of-fold ΔAUC, 2000-boot CI. **PASS iff** mean
ΔAUC ≥ **0.02** **AND** 95% CI excludes 0. (0.02, as in Leg 2,
pre-stated: increment over a richer model is expected smaller.) Also
report max |corr| SA-vs-IT (descriptive, no threshold).

## 9. Decision rule (LOCKED — executes itself; audit first)

| Outcome | Conclusion | Action |
|---|---|---|
| Audit FAILS | SA features are tool-API artifacts | Leg-3 INVALID; methods finding; SA-H* not interpreted |
| Audit passes, **SA-H1 fails** | SA carries no outcome signal | Document negative leg; proceed to Leg 4 |
| Audit passes, SA-H1 passes, **SA-H3 fails** | SA redundant with Leg-1 IT — not distinct. **Strong evidence for the "IT is the load-bearing construct" thesis** (SA was the strongest distinctness candidate). | Documented major finding; proceed to Leg 4 |
| Audit passes, **SA-H1 + SA-H3 pass** | SA is a distinct, validated leg. **Evidence for the multi-leg framework thesis.** | Leg 3 validated; proceed to Leg 4 |

The meta-thesis is updated by this result but not pre-committed; both
non-invalid outcomes are paper-defining. No interpretation outside
this table without a §10 amendment.

## 10. Frozen constants & amendment log

Frozen: sample = Leg-1 manifest (seed 20260515); the 6 SA features
(§4); L2 excluded (§4a); gate features perception.coverage +
projection.verify_before_submit, R²<0.80; SA-H1 floor 0.55 + null
95th pct (1000 perms); SA-H2 ΔAUC ≥0.03; SA-H3 ΔAUC ≥0.02; both CIs
exclude 0; StratifiedKFold(5, random_state=20260515); 2000-boot CI.

Amendments (dated, with rationale) — none.

## 11. What a pass/fail does NOT claim

A pass: SA-analog features are a distinct outcome-relevant measurement
on this one agent family — not that the agent "has situation
awareness", and not construct validity of the Endsley mapping itself.
A collapse-into-IT is strong evidence for the IT-core thesis on this
corpus/operationalization, not proof for all agents/corpora.
