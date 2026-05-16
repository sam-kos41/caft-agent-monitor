# External Corpus Scoping

**Date:** 2026-05-15
**Purpose:** the population-level HF-measurement program needs a corpus
that internal Claude Code sessions cannot provide. This document scopes
the candidates against hard requirements and states what must be
verified before committing. It is a decision aid, not a decision.

**Honesty flag:** authored from knowledge with a Jan-2026 cutoff. Every
availability/format claim below is marked **[VERIFY]** and must be
confirmed against the live source before any commitment. Do not treat
this as confirmed fact.

## Why internal data is insufficient (established, see CONSTRUCT_REVISION.md)

- Raw Claude Code sessions rotate on a ~3-week window. Of the original
  ~618-trace CAFT corpus, 617 raw traces are already gone; only derived
  labels/summaries survived (uncomputable for new metrics).
- The capture hook (`agentdiag/validation/capture.py`) now preserves
  *future* sessions, but that is a single-agent, single-user, slow
  accretion with no independent outcome labels.
- The program needs: many sessions, multiple agents/configs on
  comparable tasks, and a per-instance outcome that is **not** derived
  from any behavioral metric. Internal data fails the last three.

## Hard requirements

| ID | Requirement | Why it is non-negotiable |
|----|-------------|--------------------------|
| R1 | Raw action/observation traces (event-level, reconstructable into the ObservableEvent stream) | Every leg (IT, workload, SA, error-recovery) is computed from the event stream. Final-answer-only datasets are useless here. |
| R2 | Per-instance ground-truth outcome **independent of any behavioral metric** (e.g. test-suite pass/fail) | This is the external anchor that the entire construct-validation discipline requires (H2). Without it we repeat CAFT's circularity. |
| R3 | Multiple agents/configs on the **same** tasks | The population claim is comparative ("config A's signature differs from B's"). Needs paired tasks across configs. |
| R4 | Scale (hundreds+ instances) | Population statistics + label-shuffle null need n. Tens is a pilot, not a study. |
| R5 | Accessible + licensed + tractable format | Adaptation cost is real engineering; a closed or undocumented format can sink the timeline. |

## Candidate decision matrix

Verdict legend: ✓ likely meets · ~ partial / conditional · ✗ likely fails · ? unknown until verified.

| Candidate | R1 traces | R2 outcome | R3 multi-agent | R4 scale | R5 access/format | Overall |
|-----------|-----------|-----------|----------------|----------|------------------|---------|
| **SWE-bench (Verified/Lite) + released agent trajectories** | ~ depends on trajectory format **[VERIFY]** | ✓ test-based pass/fail (strong, independent) | ✓ many systems on identical instances | ✓ 500/2294 | ~ trajectories scattered across submissions **[VERIFY]** | **Primary candidate** |
| **SWE-agent `.traj` dumps (Princeton)** | ✓ thought/action/observation logs **[VERIFY format]** | ✓ inherits SWE-bench tests | ~ mostly SWE-agent + LLM-backend variants | ✓ | ✓ structured JSON, adapter-friendly | **Strong primary** |
| **OpenHands eval outputs** | ✓ per-instance event history **[VERIFY schema]** | ✓ `resolved` bool on SWE-bench | ~ one framework, many LLM backends | ✓ | ~ schema churn across versions **[VERIFY]** | Strong fallback |
| **Aider polyglot benchmark logs** | ~ chat + edit history, not full tool stream | ✓ exercise pass/fail | ✓ many models | ✓ | ~ log format **[VERIFY]** | Fallback (weaker R1) |
| **τ-bench (tool-agent)** | ✓ trajectories + reward **[VERIFY]** | ✓ task reward | ~ | ~ | ? | Niche; different domain |
| **AgentBench / GAIA / WebArena** | ~ varies | ✓/✗ varies | ~ | ✓ | ✗ heterogeneous, web/assistant domain mismatch | Not recommended (domain drift) |
| **Internal captured Claude Code (capture.py)** | ✓ full fidelity | ✗ no independent outcome | ✗ single agent | ✗ slow accretion | ✓ native | Pilot/architecture-shakedown only |

## Recommendation

1. **Primary: SWE-agent `.traj` trajectories over SWE-bench Verified.**
   It is the only candidate that plausibly satisfies R1+R2+R3+R4
   simultaneously: action/observation logs (R1), test-based outcomes
   that are *definitionally independent* of any behavioral metric (R2),
   many model/config variants run on the *same* 500 verified instances
   (R3), at scale (R4).
2. **Fallback: OpenHands eval histories** if SWE-agent trajectory
   availability/format does not check out.
3. **Internal captured corpus**: use ONLY to shake down the pipeline
   architecture (symbolization audit, plugin extractors) while the
   external corpus is secured — never as the validation substrate.

## MUST-VERIFY checklist (before any commitment)

These are the gating unknowns. Each is a quick web/dataset check; I can
run them via WebFetch on request.

- [ ] SWE-agent trajectories: are full `.traj` (thought/action/
      observation) dumps publicly downloadable for SWE-bench Verified,
      and for **how many distinct model/config runs**? (R3 hinges on
      this.)
- [ ] Trajectory schema: can an action be mapped to our
      `ObservableEvent` (tool name + target + outcome)? Estimate the
      adapter cost (target: < ~1 day, like the existing adapters).
- [ ] Outcome join: is each trajectory keyed to its SWE-bench
      instance_id so the independent pass/fail label can be joined 1:1?
- [ ] Licensing: redistribution / derivative-analysis terms compatible
      with a methods paper.
- [ ] Scale realized: after filtering to trajectories that actually
      have both traces AND outcomes, is n still in the hundreds?

## Adaptation cost note

Existing adapters (`adapters/claude_code.py`, `claude.py`, `openai.py`,
`langchain.py`) already normalize disparate logs into the event stream.
A new `adapters/swe_agent.py` mapping `.traj` steps →
`ObservableEvent` is the same shape of work and the realistic unit of
adaptation effort. The symbolization audit (Phase 1) must be run on the
SWE-agent stream specifically — its action vocabulary differs from
Claude Code's and the tool-API-artifact risk is corpus-specific.

## What this does NOT decide

Whether the program is worth running at all (the career/time question)
remains open and upstream. This document only establishes that *if* the
program runs, SWE-agent-over-SWE-bench-Verified is the substrate to
verify first, and exactly what to verify.
