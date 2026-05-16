# Demo — CAFT → Parsimony walk-through

A static, self-contained interactive walk-through of the
construct-validation arc. **No analysis runs in the page** — every
number is transcribed from the frozen, seed-locked pipeline on branch
`construct-validation-pivot` and is reproducible there.

## Run it

Just open `index.html` in a browser. No build, no server, no
dependencies. Suitable for embedding on a personal site or hosting via
GitHub Pages.

## What it shows

Step-by-step, the locked pre-registered arc:

1. **The autopsy** — the original per-session detector fails
   validation (κ = −0.04; steelman + lead/lag confirm it fires on an
   artifact). Documented, not buried.
2. **IT validated** — at the population level, on an external
   test-based outcome, information-theoretic behavioral structure is a
   modest but robust, non-artifactual predictor (ΔAUC +0.044, CI
   excludes 0).
3–6. **Four constructs collapse into it** — cognitive workload,
   situation awareness, error recovery (human-factors ports), and
   thought-action coherence (agent-native). Each predicts the outcome
   on its own; none adds predictive value beyond IT. The agent-native
   one is the sharpest case (least IT-correlated, still zero gain).
7. **Generalization** — the collapse survives a graded outcome, so it
   is not an artifact of a blunt pass/fail label.

**Parsimony thesis:** for predicting coding-task outcomes in agent
trajectories, the information-theoretic structure of the action
sequence is doing the work.

## Scope (read this — it is part of the result)

One corpus (nebius SWE-agent trajectories), one agent family (Llama,
reactive ReAct), one outcome family (SWE-bench resolution; binary and
graded). Generalization is conditional on a documented selection bias.
**It is not claimed that "agents are universally flat."**
Cross-architecture generalization is the single open axis and was
deliberately deferred.

The durable asset is the methodology: pre-registration before data, a
symbolization-audit gate, null models, locked decision rules, honest
scoping, and no goalpost-moving. Full narrative:
[`../docs/PROJECT_SNAPSHOT.html`](../docs/PROJECT_SNAPSHOT.html);
governing document and per-leg pre-registrations under `../docs/`.
