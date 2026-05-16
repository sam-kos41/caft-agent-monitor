# Paper Outline — Construct Validation of Agent Behavioral Measurement

**Status:** drafting spine (empirical phase closed 2026-05-15). This
is the skeleton + evidence ledger; section prose is written next.

## Working title (candidates)

- *"What an Information-Theoretic Agent Monitor Actually Measures: A
  Construct-Validation Study and a Parsimony Result"*
- *"Human-Factors Measures of LLM Agent Behavior Reduce to a Single
  Information-Theoretic Construct"*

## One-paragraph thesis

We set out to build a multi-construct, human-factors-grounded
measurement framework for LLM coding agents. Rigorous
construct-validation instead produced two findings: (1) per-step
single-session anomaly/health detection is not supported (it measures
within-session phase variation, not pathology — multiply confirmed);
(2) at the population level on a clean test-based outcome, an
information-theoretic behavioral-structure construct is real, modest,
and non-artifactual — and three independent human-factors analogs
(cognitive workload, situation awareness, error recovery), each a
plausible distinct construct, all reduce to it. The contribution is
the methodology and the parsimony result, both honest negatives
included.

## Section structure

1. **Introduction** — agent observability; the appeal and the trap of
   naming a method by the task ("anomaly detection") rather than what
   it computes.
2. **Background** — Wickens IP / information-theoretic operator
   measurement; Endsley SA; Reason/Hollnagel resilience; SWE-bench as
   an outcome-labeled substrate; construct validity (convergent /
   discriminant) as the lens.
3. **The CAFT autopsy (negative result #1)** — what was claimed;
   tracing the real pipeline; κ = −0.04 vs domain expert;
   self-calibrating baseline analysis; the steelman (corpus baseline +
   change-point both fail the same way); lead/lag artifact. Per-step
   single-session detection does not survive.
4. **Methodology** — the construct-validation harness as the reusable
   contribution: pre-registration before data, symbolization-audit
   gate, label-shuffle null, convergent + discriminant tests,
   bootstrap CIs, locked decision rules, honest scoping, dated
   amendments, the anti-rotation capture problem. The discipline that
   made the negatives trustworthy.
5. **Population-level validation (positive result)** — pre-registered
   pilot; symbolization gate passed; H1/H2 pass; IT carries a small
   (ΔAUC +0.044, CI excludes 0) robust non-artifactual outcome-relevant
   signal. Honest ceiling stated.
6. **The parsimony result (negative result #2, the headline)** — three
   pre-registered HF-analog legs; the consistent pattern; all reduce
   to IT. Evidence ledger below.
7. **Threats to validity / boundaries** — one agent family (Llama
   SWE-agent), one corpus, deterministic operationalizations, excluded
   constructs (SA-L2, wall-clock latency, semantic ack), modest effect
   sizes, no causal claim, generalization unproven.
8. **Discussion** — why HF analogs collapse to IT (action-sequence
   structure may be the common substrate they all proxy); what this
   implies for agent-evaluation research (parsimony over construct
   proliferation); what would re-open it (timestamped/multi-agent
   corpora; SA-L2 with state ground truth).
9. **Conclusion** — method + parsimony; the value of killing one's own
   construct on the record.

## Evidence ledger (frozen, reproducible from `construct-validation-pivot`)

| Claim | Evidence | Artifact |
|---|---|---|
| Per-step detection fails | κ=−0.04 vs human; 3 baselines co-fail; lead/lag artifact | `CONSTRUCT_REVISION.md`, `leadlag.py`, `baseline_variants.py` |
| Sample is principled | N=2000, seed-frozen, 1000/1000, full 80k pass | `PILOT_PREREGISTRATION.md` A1, `docs/pilot/sample_manifest.csv` |
| IT not a tool-API artifact | symbolization gate, CV R² mostly negative | `docs/pilot/symbolization_audit.md` |
| IT carries population signal | H1 AUC 0.75 > null p95 0.53; H2 ΔAUC +0.044 CI [.026,.062] | `docs/pilot/hypotheses_result.md` |
| Workload reduces to IT | W-H3 +0.006 CI∋0; corr 0.74 | `docs/pilot/leg2_hypotheses.md` |
| SA reduces to IT (strongest distinct axis) | SA-H3 +0.003 CI∋0; corr 0.64 | `docs/pilot/leg3_hypotheses.md` |
| Error recovery reduces to IT | E-H3 +0.004 CI∋0; corr 0.75 | `docs/pilot/leg4_hypotheses.md` |
| Discipline upheld | pre-regs before code; A1/A2 amendments; locked rules; ~1000 tests | git history of the branch |

## Honest framing constraints (fixed)

- Negatives are the contribution, not failures. State plainly.
- No claim beyond corpus/operationalization. "On nebius SWE-agent /
  Llama, with deterministic trace features."
- Effect sizes are modest; report them, do not inflate.
- The methodology is reusable independent of these specific results.

## Open drafting decisions (for the author)

- Venue/audience (HF/HCI vs AI-eval vs ML methods) → affects depth of
  the HF-theory exposition vs the statistics.
- Whether the CAFT autopsy is §3 of one paper or a separate short
  "lessons" companion.
- How much of the capture/anti-rotation infrastructure story belongs
  in the methods section vs an appendix.
