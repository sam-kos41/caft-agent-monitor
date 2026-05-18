# Measuring AI Coding Agents: a construct-validation study

**One-sentence finding:** to predict whether an AI coding agent
finishes its task, the information-theoretic structure of its action
sequence is the measurement that does the work. Four other constructs
(three from human factors, one agent-native) each predict the outcome
on their own, but none of them add anything once you already have it.
This still holds when the outcome is graded instead of pass/fail.

![Four ways to measure an AI agent, and why each reduces to the action sequence](demo/preview.png)

*(The image above is a snapshot of the figure in
[`demo/index.html`](demo/index.html). The full detail and evidence are
in [`docs/PROJECT_SNAPSHOT.html`](docs/PROJECT_SNAPSHOT.html).)*

---

## What this is

A construct-validation study of AI coding-agent behavior. It tests
whether a set of behavioral constructs predict whether an agent
finishes its task, using an external corpus of agent runs with real
outcome labels (SWE-agent trajectories, scored by whether the
repository's tests pass, so the label does not depend on any of the
measurements).

The constructs tested:

- the information-theoretic (IT) structure of the action sequence
  (mutual information, compression, KL, entropy);
- cognitive workload (Wickens, Sweller, NASA-TLX);
- situation awareness (Endsley);
- error recovery (Reason, Hollnagel);
- thought-action coherence (agent-native, no human-factors source).

## Findings

1. **Per-session, single-session failure detection does not work.** A
   detector that watches one session and flags trouble from
   information-theoretic metrics over a sliding window agreed with a
   domain expert worse than chance (kappa = -0.04). Three independent
   baseline designs all fired on a startup transient while the window
   filled, not on anything in the session.
2. **At the population level, IT predicts task outcome.**
   Pre-registered pilot, frozen sample of N = 2,000. H1 AUC 0.75 (the
   permutation null is 0.53 at the 95th percentile), and IT adds over
   a trivial baseline (delta AUC +0.044, 95% CI [+0.026, +0.062]).
   The effect is modest but robust and not an artifact.
3. **The other four constructs fold into IT.** Each was
   pre-registered, gated, and tested with a decision rule fixed in
   advance. Each predicts the outcome on its own (AUC 0.65 to 0.71).
   None adds anything once IT is in the model (every key delta AUC is
   at most 0.006, with a 95% CI that includes 0). The agent-native
   construct is the clearest case: it is the least correlated with IT
   (0.43), so it is genuinely a different measurement, yet it still
   adds essentially nothing.
4. **It is not a pass/fail artifact.** Re-run on a graded outcome (the
   fraction of tests that pass), IT still predicts it (Spearman 0.277
   vs a null of 0.046; delta Spearman +0.101, CI above 0), and no
   construct comes back.

## How it was validated

Pre-registration was committed before any code touched the data.
Every leg passed a symbolization-audit gate and a required human
checkpoint. Every comparison had a label-shuffle null model. The study
tests for both convergent and discriminant validity. Decision rules
were fixed in advance and run straight from the data, with no
threshold changed after the fact. Only constructs the corpus could
actually support were measured; the rest were left out with documented
reasons. Everything is test-first (about 1,000 tests, several of which
caught real bugs before they could corrupt a result).

## Scope of the claim

One corpus (nebius SWE-agent trajectories), one agent family (Llama,
reactive ReAct), one outcome family (SWE-bench resolution). The graded
outcome result is conditional on a selection bias, because its
parseable subset over-represents successes. Whether this holds for a
different agent architecture (a planning agent instead of a reactive
one) is the one open question, and it is left for later. This study
does not claim that all agents are flat.

## Repository layout

| Path | What it is |
|------|------------|
| `demo/` | The figure: `index.html` (self-contained, no deps), `preview.png`, and how it is built (`build_trajectory_demo.py`) |
| `docs/PROJECT_SNAPSHOT.html` | The full study as a one-page visual |
| `docs/PROGRAM.md` | Governing document and the standing discipline every leg inherited |
| `docs/PREREG_*.md`, `docs/PILOT_PREREGISTRATION.md` | Per-leg pre-registrations (committed before data) |
| `docs/CONSTRUCT_REVISION.md` | Why per-session detection was rejected |
| `docs/VALIDATION_METHODOLOGY.md`, `docs/CORPUS_SCOPING.md` | Method and scoping decisions |
| `docs/pilot/` | Result artifacts (AUCs, delta AUCs, p-values) |
| `agentdiag/validation/` | The validation pipeline (sampling, features, audit, hypotheses, per-leg modules) |
| `agentdiag/adapters/` | The `ObservableEvent` contract and agent adapters (SWE-agent, Claude Code) |
| `agentdiag/eval/` | The per-session detector and its eval harness |
| `tests/` | About 1,000 tests covering the pure machinery |

## Reproduce

```bash
pip install -e .
python -m pytest tests/ -q                 # the test suite
python demo/build_trajectory_demo.py        # regenerate demo/index.html
```

Every number in the figure is reproducible from the frozen,
seed-locked pipeline on branch `construct-validation-pivot` (it is
also on `main`).
