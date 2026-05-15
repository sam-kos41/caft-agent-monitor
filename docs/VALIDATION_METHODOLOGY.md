# CAFT Validation Methodology

How we establish that CAFT's outputs mean something — and the running
log of what we have learned doing it. This is a methods document, not a
results document; numbers live in the generated validation reports.

## 1. The question

CAFT emits anomaly signatures and per-session metrics. The open
question is **construct validity**: do those outputs correspond to
properties a competent human would recognize and agree on? Internal
consistency on synthetic traces (TPR/FPR on planted failures) does not
answer this — it only shows the detectors do what they were designed to
do. The website-session episode (a session CAFT flagged RED that the
user considered a success) made the gap concrete: "anomaly" did not yet
have a stable, agreed meaning.

## 2. Design: three raters, same sessions, same constructs

| Rater | Role | Confidence semantics |
|---|---|---|
| Human (domain expert) | Gold standard | self-reported low/med/high; may abstain |
| Local LLM (Ollama) | Cheap scalable second rater | fixed "med" — never authoritative |
| CAFT | The system under test | "high" (deterministic rule output) |

Constructs (1–5 Likert unless noted):

- `stuck_in_loop` — repeating actions without progress
- `goal_drifted` — work unrelated to the user's request
- `coherent_progress` — each action follows logically from the last
- `user_satisfied` — final user message suggests satisfaction
- `overall_health` — categorical: healthy / degraded / pathological

Agreement: pairwise Cohen's κ (linear-weighted for ordinal dims),
Krippendorff's α across all raters. The **primary** number is
high-confidence κ (see §3).

## 3. The abstention protocol (and why it exists)

A gold-standard rater forced to guess injects noise that destroys the
very κ being measured. So the instrument allows, per dimension:

- an explicit **"can't tell"** abstention — *excluded* from κ, never
  imputed; and
- a **confidence** flag (low/med/high).

Two κ values are always reported: over all non-abstained ratings, and
over the **high-confidence subset only**. The latter is the
load-bearing number. Abstention *rate per dimension* is itself a result:
a dimension nobody can rate is a dimension that is not operationalized,
independent of any κ.

## 4. Finding 1 — abstention is a function of evidence quality, not only construct ambiguity

The first version of the session digest presented counts only (tool
histogram, top bash patterns, error totals). A domain-expert rater
(the project owner) rating session `137f1898` (a 4.6-day, 679-tool-call
VLM research session) produced:

| Digest version | Confident ratings | Abstentions |
|---|---:|---:|
| v1 — counts only | 2 / 5 | 3 / 5 |
| v2 — + descriptive phased timeline | 4 / 5 | 1 / 5 |

Same rater, same session, same constructs, same scale. The only change
was a strictly-descriptive "what the agent did" timeline (phased,
factual, no evaluative language). Three dimensions
(`goal_drifted`, `coherent_progress`, `overall_health`) moved from
"can't tell" to confidently rated.

**Methodological consequence.** Abstention rate conflates two distinct
causes: (a) the construct is ill-defined, and (b) the evidence shown is
insufficient to apply a well-defined construct. These require opposite
fixes (redefine the construct vs. enrich the evidence). Therefore:

- evidence presentation must be **held constant** when comparing raters
  or tracking agreement over time; a κ change across digest versions is
  not a change in CAFT;
- the digest is itself an experimental variable and is versioned;
- if a 3B model floor/ceilings while a human discriminates on the same
  digest, that is a model-capability finding; if *both* abstain, it is
  an evidence-sufficiency finding. Do not collapse the two.

## 5. Descriptive-not-evaluative constraint

The narrative timeline reports facts (counts, filenames, timestamps,
verbatim user quotes, error provenance) and never judgments
("stuck", "struggled", "eventually"). Rationale: judgment is the
dependent variable; a narrative that editorializes anchors the rater
and contaminates κ. This is enforced by a test
(`test_narrative_is_descriptive_not_evaluative`) that fails the build
if banned evaluative tokens appear in generated narratives.

Error provenance is **distinguished but not scored**: environmental
results (cancelled fan-out calls, unmounted filesystem, connectivity)
are labelled as such and explicitly marked "not an agent signal", so a
filesystem-not-mounted error does not nudge the rater the way a raw
"10 errors" annotation did. Sub-agent (`Agent` tool) invocations show
their verbatim description and are flagged as black boxes, since their
internal work is not present in the parent trace.

## Finding 2 — CAFT's `user_satisfied` is an invalid proxy (confidently wrong)

A session with an unambiguous negative user-sentiment signal
(`fig-generation-protocol/47b5baa3`; user messages include *"are you
fucking stupid"*, *"do u not understand at all"*) was added to the
corpus as a discriminant-validity probe. Result:

| Rater | user_satisfied | confidence |
|---|---|---|
| CAFT | **5 ("clearly satisfied")** | high |
| Ollama 3B | abstained (null) | — |
| Human (ground truth) | 1 ("clearly dissatisfied") | high |

CAFT rated the angriest session in the corpus as *maximally satisfied,
with high confidence*. Root cause: `rate_caft.py` maps `user_satisfied`
from **inverse anomaly rate** — there is no user-sentiment model. Low
anomaly rate (the agent mechanically produced figure revisions) →
"satisfied". The proxy is construct-invalid.

Consequences:

1. The failure is **systematic, not noise**: every CAFT
   `user_satisfied` value is an anomaly-rate proxy mislabelled as
   satisfaction. The same critique applies to `goal_drifted`
   (mapped from KL divergence).
2. **A confidently-wrong deterministic rule is worse than an
   abstaining weak model.** The 3B model's abstention was the more
   honest output. This validates the abstention protocol (§3): a rater
   without a construct-valid basis for a dimension should abstain, not
   emit a high-confidence proxy.
3. **Remedy direction** (pending decision): CAFT should abstain on
   dimensions for which it has no construct-valid signal
   (`user_satisfied`, likely `goal_drifted`) rather than report a
   proxy. Reporting only `stuck_in_loop` / `coherent_progress` /
   `overall_health` — the dimensions its IT metrics actually bear on —
   is more defensible than a full vector of which 40% is invalid.

This is the central argument for the whole harness: it surfaced an
invalid construct automatically, on the second corpus expansion,
before any external party did.

## 6. Open construct issues (not yet resolved)

1. **`goal_drifted` is confounded with user-directed topic shifts.** In
   long collaborative sessions the goal is a moving target *by design*;
   the agent following a user redirect is not drift. The construct
   needs to separate *unprompted* wandering from *user-directed*
   re-scoping. Until then, `goal_drifted` ratings on multi-turn
   sessions carry reduced confidence by default.

2. **`user_satisfied` operationalization is unsettled.** Final-state
   (last message), peak (best in-flight reaction), and trajectory
   (direction over the session) can disagree — e.g. a session with
   "perfect get started!!" mid-way that ends on an inconclusive log
   paste. The verbatim user-reactions panel was added so raters can see
   in-flight signal, but the construct definition itself still needs to
   commit to final / peak / trajectory.

3. **Multi-day rhythm.** Phase timestamps now show day-of-session;
   whether `coherent_progress` should be conditioned on temporal
   compression (14 phases in one day vs. dribbled across a week) is an
   open question for the construct definition.

## 7. Status

Harness, three raters, abstention/confidence protocol, descriptive
narrative, and report generator are implemented and tested. Human
ratings are being collected on a 6-session bootstrap corpus. No
human↔CAFT validity claim is made yet — the corpus is too small and
single-rater. Next: complete the 6-session human pass, then expand to
a multi-rater corpus large enough for a defensible κ with CIs.
