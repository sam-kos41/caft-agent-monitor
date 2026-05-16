# Program: Human-Factors Measurement of AI Agent Behavior

**Status:** active. Governing document. Each leg has its own
pre-registration; this file is the index + the standing discipline
every leg inherits.

The through-line: AI agents are observable systems whose behavior can
be measured, characterized, and validated against external outcomes —
at the **population** level, with construct-validity discipline
throughout. Per-instance pathology *detection* was tested and failed
(`CONSTRUCT_REVISION.md`); the program is the disciplined recovery.

## Standing discipline (inherited by every leg — non-negotiable)

These are the rules that survived the CAFT arc. They are not optional
and not re-litigated per leg.

1. **Pre-registration before any data-touching code.** Hypotheses,
   sample, features, statistics, thresholds, and a locked decision
   rule are committed first. (`PILOT_PREREGISTRATION.md` is the
   template.)
2. **Symbolization-audit gate before any hypothesis test.** Every new
   corpus/leg: can the feature be reconstructed from raw tool-API
   surface counts? Gate is objective and locked. (Method established
   in the Leg-1 pilot, §5.)
3. **Mandatory human checkpoint at the symbolization-audit gate**,
   before hypothesis tests run. Standing rule, every leg.
4. **Null model for every comparison.** A result a label-shuffle
   reproduces is not a result. (`eval/stats.py`, pilot H1.)
5. **Convergent AND discriminant validity.** A leg must add signal
   *and* be separable from the other legs — redundancy with Leg 1 is a
   failure mode, not a success.
6. **No goalpost-moving.** Thresholds frozen pre-hoc; the decision
   rule executes itself from the booleans; any change is a dated
   amendment with rationale (see the A1/A2 precedent).
7. **Honest scoping.** Only operationalize what is *observably*
   computable from the corpus. Signals the data cannot support are
   excluded with a documented reason, not proxied silently.
8. **Test-first.** Pure machinery is unit-tested before it runs on
   real data (this caught real bugs at ≥3 points in Leg 1).

A leg can fail without the program failing. A documented negative
leg strengthens the framework paper; it does not weaken it.

## The four legs

| Leg | Construct family | Status |
|---|---|---|
| **1 — Behavioral structure (information theory)** | predictability (MI), redundancy (compression), distributional shift (KL), diversity (entropy), cross-agent MI | **Validated at population level** on SWE-agent/Llama-70b (pilot, 2026-05). Modest robust effect ΔAUC +0.044 (CI excludes 0), non-artifactual. Convergent validity w/ test outcomes shown; construct validity of specific feature→construct mappings still open. |
| **2 — Cognitive workload analogs** | resource demand (Wickens MRT, Sweller CLT, NASA-TLX) | **Documented NEGATIVE (2026-05): not a distinct leg.** Locked W-H3 failed (ΔAUC +0.006 over baseline+IT, CI spans 0; max workload↔IT corr 0.74). Workload predicts outcome (W-H1 pass) but the signal is already captured by Leg-1 IT — folds into Leg 1. Corpus-limited (timing/branching excluded by honest-scoping; no timestamps in nebius). Strengthens the framework paper as a discriminant-validity finding. |
| **3 — Situation awareness analogs** | perception / comprehension / projection (Endsley, SAGAT/SART) | **Documented NEGATIVE (2026-05): not a distinct leg.** Locked SA-H3 failed (ΔAUC +0.003 over baseline+IT, CI spans 0; max SA↔IT corr 0.64). SA-H1 weakly passes (AUC 0.65 — lowest of any leg) but folds into Leg 1. L2 comprehension excluded by honest-scoping. The strongest distinctness candidate collapsed into IT — see meta-thesis note below. |
| **4 — Error recovery & adaptive behavior** | resilience (Reason; Hollnagel/FRAM; adaptive automation) | **Pre-registration drafted, IN HUMAN REVIEW** (`PREREG_LEG4_ERROR_RECOVERY.md`). Run as a **confirmatory** test of the parsimony thesis with a stated prior it also reduces to IT — but error recovery is the most distinct axis from IT (discrete adverse events vs steady-state sequence shape), so a leg-defining pass is held genuinely open. **Final empirical leg: regardless of outcome, the empirical phase ends here → methodology-paper drafting begins.** |

Per-leg theoretical roots, candidate constructs, and validation plans
are specified in each leg's pre-registration, grounded in cited HF
literature — not invented.

## Meta-thesis status (data-driven; updated 2026-05, not pre-committed)

The four-leg-vs-IT-core question was held open. After two leg-pair
discriminant tests it has shifted on the evidence: **Leg 2 (workload)
and Leg 3 (situation awareness) both reduced to Leg 1 IT** — and SA
was the conceptually strongest distinctness candidate (different axis:
agent↔environment, not action-sequence shape). n=2 convergent
discriminant negatives, including the strongest case.

Working thesis (this corpus / these operationalizations, not yet
universal): **information-theoretic behavioral structure is the
load-bearing construct for population-level agent measurement; HF
analogs carry outcome signal but reduce to it.** This is a parsimony
finding — arguably a stronger contribution than four independent
legs. Leg 4 (error recovery) is now the test of whether *any* HF
analog survives distinct from IT, with a strong prior it will not.
Still data-driven: a distinct Leg 4 would re-open the question.

## Shared infrastructure (build once, use everywhere)

- **Event stream:** `ObservableEvent` + adapters (`claude_code`,
  `swe_agent` done; `cursor`/`aider`/`openhands` as cross-agent
  breadth requires).
- **Symbolization-audit framework:** `pilot_audit.py` pattern,
  re-pointed per leg/corpus.
- **Stats:** permutation null + bootstrap CI + nested-model
  comparison (`pilot_hypotheses.py`, `eval/stats.py`).
- **Pre-registration template:** `PILOT_PREREGISTRATION.md`.
- **Feature spine:** `pilot_features.py` (per-trajectory extraction;
  each leg adds a feature module, same cache discipline).

## Sequencing

Legs run **in sequence, not parallel** (the ambition failure mode is
four half-validated legs). Order: 1 (done) → 2 → 3 → 4. Each leg added
strengthens the framework only if it gets the same rigor Leg 1 got.

## Honest framing (fixed, for any write-up)

"Per-step single-session detection fails. Aggregate population-level
behavioral measurement carries small, robust, non-artifactual,
outcome-relevant signal — demonstrated for the IT leg on one agent
family. Generalization across agents and construct validity of the HF
mappings are open and tested leg by leg." Not "CAFT works."
