# What CAFT Measures: A Construct Revision

**Date:** 2026-05-15
**Status:** Adopted. Supersedes the "anomaly detection / health classification"
framing in README.md and the audit collateral.

This document exists because, during validation, we discovered that CAFT
measures something different from what it was presented as measuring. It
is written by the people who found the gap, on purpose, so the gap is on
the record and drives the project forward instead of being papered over.

---

## 1. What was originally claimed

- "Real-time **anomaly detection** for AI coding agents."
- A per-session **health verdict**: red / yellow / green
  (`_assess_health`), surfaced in the CLI, the audit PDF, the MCP
  server, and the dashboard.
- An audit report with a **dollar-cost-of-waste** estimate
  ("~$35,135/month") derived from the anomaly rate.
- Validation numbers (TPR 94.2%, signature accuracy 98%) on **synthetic**
  traces with planted failures.

## 2. What the math actually computes

Tracing the real code path (`plugin._analyze_session` →
`UniversalMonitor` → `cognitive.SymbolStream` → `SelfCalibratingBaseline`
→ `CompositionalAnomalyDetector` → `plugin._assess_health`):

1. **IT metrics** (action MI, tool entropy, KL divergence, LZ
   compression) over a sliding window of tool-type tokens. **Sound.**
2. **`SelfCalibratingBaseline` calibrates on the first ~100 events of
   the *same session*** and z-scores everything after against that.
   "Anomalous" therefore means **"statistically unlike how *this
   session* started"** — i.e. *within-session phase change* — not
   "unlike a healthy session." A session that legitimately moves
   explore → implement → verify, or that the user redirects, flags
   itself **by construction**.
3. **`_assess_health`** then trips on `rate > 0.15 OR mi < 0.3 OR
   kl > 0.5` (red) / looser ORs (yellow). The thresholds were tuned on
   synthetic traces; real Claude Code sessions sit at `mi ≈ 0.9–1.6`,
   `kl ≈ 0.25–0.54`, so `mi < 0.8 → yellow` and `kl > 0.3 → yellow`
   fire on **normal** sessions. The OR-logic means any one loose
   condition is sufficient.

Net: the "health" verdict is largely a **within-session
phase-variation + structural-repetition** measurement, relabeled as a
quality judgment. Those constructs coincide for genuinely stuck/looping
sessions (low MI + high compression is real) but diverge for normal
varied work — which is most real sessions.

## 3. The evidence that surfaced the gap

- **κ = −0.04 (worse than chance)** for human ↔ CAFT on
  `overall_health` over a 7-session real corpus. CAFT called 5/7
  pathological; the domain expert called 6/7 healthy.
- **Discriminant probe failed**: the session where the user wrote
  *"are you fucking stupid"* was rated `user_satisfied = 5 (high
  confidence)` by CAFT v0.1, because that dimension was mapped from
  *inverse anomaly rate*. (Fixed in v0.2 by abstaining.)
- **Website-session false positive** (recorded earlier): a 40 MB
  session the user considered a success was flagged RED with 1931
  anomaly windows; manual inspection showed reactive micro-tasking, not
  failure.
- **Self-baseline analysis** (this document, §2): the over-flagging is
  structural, not a tuning accident.
- **Our own programmatic extractor reproduced the same sin**: when it
  mapped objective signals onto "health"/"coherence" it produced
  `health = healthy` for the rage session. Determinism does not rescue
  an invalid construct — a reproducibly wrong rule is still wrong. The
  defensible extractor asserts only literally-observable facts and
  abstains otherwise.

## 4. The revised framing (adopted)

We considered three options for the Step-2 baseline problem:

- **(A)** Keep self-calibration, rename to within-session phase-shift
  detection. Cheap, honest, loses the "anomaly detection" story.
- **(B)** Build a cross-session baseline from a corpus labeled normal
  via objective signals. Most defensible, most expensive; a research
  track.
- **(C)** Drop the single anomaly/health verdict; report each metric as
  its own descriptive behavioral dimension. No good/bad collapse.

**Decision: (C) for the product surface, (B) as the research track.**

Reasoning: the IT metrics are real and individually interpretable; the
damage is entirely in collapsing them into a single evaluative verdict
the math doesn't support. (C) removes the indefensible claim without
discarding the sound machinery, and it can ship now. (B) is the only
path that would re-earn a true cross-session "anomaly" claim, but it
depends on a corpus the objective-signal labeler has yet to produce, so
it is explicitly future work, not a current claim.

Concrete consequences (executed alongside this note):

- `_assess_health` → `_assess_behavioral_state`; output relabeled
  `phase_shifting / steady / looping` — **descriptive of what the math
  computes, carrying no quality claim.**
- The audit dollar-waste estimate is **removed** until there is a
  validated basis for it.
- CAFT **abstains** on the evaluative validation dimensions it has no
  validated basis for (`overall_health`, `user_satisfied`,
  `goal_drifted`); it reports only `stuck_in_loop` and
  `coherent_progress`, the dimensions its IT metrics bear on.
- MCP/CLI/dashboard wording updated to the descriptive framing.

## 5. Validation plan going forward

Ground truth is **programmatic and behavioral**, not human Likert
(which gave us mixed-polarity keying errors) and not a small LLM (which
floor/ceilinged). The objective-signal labeler asserts only literally
observable facts (longest identical-call run, error→retry chains,
verbatim frustration, final-message resolution) and abstains on
everything inferential.

The convergent-validity test that *means something* is **not** "does
CAFT agree with the literal loop detector" — two repetition detectors
agreeing proves nothing. It is:

> Does CAFT's signal rise **before** the trivial threshold trips
> (e.g. compression_ratio climbs N events before the 8×-identical-tool
> rule fires), or detect thrash that is **not** literal error→retry but
> shares the IT signature?

If yes, CAFT carries information beyond a regex and the project has a
defensible core. If it only co-fires with the trivial detector, CAFT is
an expensive regex and we will say so.

## 5a. Post-revision measurement (2026-05-15)

After (C) was applied (CAFT v0.3 abstains on overall_health /
user_satisfied / goal_drifted) and the extractor was narrowed to
strict facts (signal-v2 abstains on everything except literal loop and
final-message resolution), CAFT and the objective extractor overlap on
exactly **one** dimension, `stuck_in_loop`. Measured on the 7-session
real corpus:

| pair | kappa |
|---|---|
| signal-v2 ↔ caft-v0.3 | +0.15 (slight) |
| signal-v2 ↔ human | +0.07 (slight) |
| caft-v0.3 ↔ human | +0.19 (slight) |

Two findings:

- **CAFT's "stuck" does not track literal repetition.** Session
  `ddc2fba4`: literal_loop_max ⇒ signal=1 (no repetition), human=1
  (not stuck), **CAFT=5 (max)**. CAFT's stuck score is driven by low
  `action_mi`, which is not repetition. CAFT is therefore not even a
  more-expensive loop detector — on this evidence it does not add
  *valid* information beyond the trivial detector for this construct.
  This is the convergent-validity test from the validation plan, and
  CAFT does not currently pass it.
- **A mapped fact is not ground truth.** Sessions `6badcc8f` /
  `71ec0464` have genuine long identical-call runs (signal=4, a true
  fact) but the user rated them 1 — the runs were legitimate iterative
  work. `literal_loop_max = 4` is ground truth; `stuck_in_loop = 4` is
  an interpretation and is wrong here. **Only the raw count is ground
  truth; any mapping to a 1–5 construct re-injects judgment.**

Consequence for the validation plan: stop computing κ on mapped 1–5
constructs. Report raw fact distributions per session and ask narrow
falsifiable questions (e.g. "of sessions with literal_loop_max ≥ 8,
what fraction did the domain expert call problematic?"). The
inter-rater scale machinery stays for provenance, but the *claims*
must attach to raw facts, not mapped scores.

## 5b. The decisive test: lead/lag (resolved 2026-05-15)

The §5 validation plan named the one test that could still rescue
CAFT-as-detector: does its signal fire **before** the trivial
literal-loop detector (early warning) or only co-fire (expensive
regex)? Measured on the same event axis CAFT consumes, with CAFT's
fire = the real pipeline emitting a named signature
(`agentdiag/validation/leadlag.py`):

```
session     n_evt  t_literal  t_caft  verdict
137f1898     2129       1828     105   "led" +1723   <-- artifact
6badcc8f     2103        236     134   "led"  +102   <-- artifact
47b5baa3      265          –     183   caft_only
71ec0464      831          –     180   caft_only
73eef0e5      253          –     100   caft_only
9971516a      386          –     106   caft_only
ddc2fba4      114          –     101   caft_only
```

**CAFT fires in 7/7 sessions, t_caft clustered at events 100–183,
regardless of whether a literal loop exists.** That index is the
~100-event `SelfCalibratingBaseline` calibration window: CAFT emits
its first signature as soon as calibration closes and subsequent
events read as "deviation from this session's own start." The
apparent +1723 / +102 "lead" is **constant firing, not prescience** —
a detector that triggers in 100 % of sessions at a fixed point has
≈ zero discriminative power about the thing it claims to detect.

(Note: the analysis tool's first formatter naively concluded "CAFT is
an early-warning signal" from `led > lagged`. That was the same class
of error this whole document is about — mistaking a constant for a
signal. An artifact guard was added; the tool now reports the honest
result. Recorded here because the failure of one's own instrument is
itself data.)

**Decision (corrected 2026-05-15, narrowed):** the earlier draft of
this section claimed "CAFT-as-detector does not survive validation."
That over-claimed. The precise, defensible statement is:

> **The self-calibrating-on-first-~100-events baseline variant
> failed.** Its verdict fires on a clock (calibration-window close),
> not on session content. This indicts that *specific calibration
> mechanism*, not the information-theoretic approach, which has never
> been run with a baseline meaning "unlike normal sessions."

The negative finding is partly a consequence of a fixable design
decision, so the honest next step is an experiment, not an obituary.
Two steelman variants are under test (`baseline_variants.py`):

  1. **Corpus baseline (leave-one-out)** — z-score each session's
     per-step metrics against a reference distribution pooled across
     the *other* sessions, so "anomaly" = "unlike normal sessions."
  2. **Change-point detection** — abandon z-vs-baseline entirely;
     detect whether the IT trajectory *shifts* at the event a literal
     loop/thrash actually begins.

Pre-stated caveats (so a fix can't silently move the goalposts):
small corpus (n≈7, LOO reference is thin → directional only); and the
session-level κ evidence suggests the IT metrics may not track
human/objective judgment *even with a correct baseline* — fixing
calibration is necessary but possibly not sufficient. If the steelman
variants still fire indiscriminately or still fail convergent
validity, the negative result stands but is then *earned* (the strong
version was tested), not an artifact.

Durable regardless of the steelman outcome: (1) the construct-
validation methodology and harness; (2) the honestly reframed
descriptive `behavioral_state` surface (no detection claim). Whether
(3) "a documented negative result" or "a recovered detection claim"
is the third asset is what the steelman experiment decides.

## 5c. Steelman result (resolved 2026-05-15, `baseline_variants.py`)

Both alternatives were implemented and run on the 7-session corpus,
same event axis, pipeline's own per-step metrics, leave-one-out for
the corpus reference:

```
method     fire-rate  cluster        verdict
self        100%       events 100-183  ARTIFACT (calibration window)
corpus      100%       event 0         ARTIFACT (cold-start degeneracy)
changept    100%       events 20-53    ARTIFACT (window-fill transient)
```

All three fire in **7/7 sessions at an early clustered index,
regardless of whether a literal loop exists**. They fail for a
*common, deeper* reason than the calibration window: the per-step IT
metrics have a large content-independent transient while the sliding
window fills, and every threshold/shift method trips on that transient
in every session. Changing the baseline (corpus) moved the artifact to
event 0; changing the method entirely (change-point) moved it to
events 20-53. The artifact moves; it does not disappear.

(Honesty note: the change-point detector's first implementation could
not detect a clean synthetic step — a unit test caught it. "changept
never fires" was briefly reported; that was a broken instrument, not a
CAFT result. Fixed, re-run; the finding above is post-fix.)

**This answers the steelman challenge.** The negative result is not an
artifact of one design choice: the two strongest alternatives were
tested and both fail, for a reason intrinsic to using these
window-based IT metrics as per-step detectors on real sessions of this
length. One lever remains untried — excluding a fixed warmup before
any detection — but the pattern (each method fires precisely at *its
own* transient) predicts that will merely relocate the artifact again
unless there is genuine post-warmup, content-discriminative signal;
the session-level κ work already indicated there is not. The
responsible position: the evidence has converged. CAFT-as-per-step-
detector is not supported; the third asset is the documented,
steelmanned negative result.

## 6. What is explicitly NOT being done

- No more inter-rater work until the revision is in.
- No new detectors/signatures — the issue is meaning, not coverage.
- The consulting pitch is paused until the surface hangs off constructs
  that hold up. The product machinery (MCP, audit, dashboard) stays;
  it gets re-hooked to the descriptive framing.
- No from-scratch rewrite. The IT metrics, the phase-shift detector
  (once correctly labeled), and the Wickens grounding are sound. The
  surgery is narrow: the construct mapping at the top of the stack.
