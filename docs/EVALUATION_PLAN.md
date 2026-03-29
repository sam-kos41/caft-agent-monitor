# Evaluation Plan: agentdiag Ablation Study

## Overview

Compares 4 detection modes on annotated traces to measure the value of the semantic confirmation layer. Implementation in `agentdiag/metrics.py`, orchestration in `scripts/run_ablation.py`.

## Modes

| # | Mode | Detectors | LLM | Description |
|---|------|-----------|-----|-------------|
| 1 | `strict` | ALL_CAFT_DETECTORS (9 active, production thresholds) | No | Current production behavior |
| 2 | `loose` | ALL_CAFT_DETECTORS_FULL (13, lowered thresholds) | No | Candidate generator mode |
| 3 | `loose+llm` | ALL_CAFT_DETECTORS_FULL (13) | Yes | Candidate generator + LLM confirmation |
| 4 | `oracle` | ALL_CAFT_DETECTORS_FULL (13) | Ground truth | Theoretical ceiling (perfect LLM) |

## Primary Metrics (per-detector AND aggregate)

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1**: Harmonic mean of precision and recall
- **Macro-average**: Each detector weighted equally (mean of per-detector scores)
- **Micro-average**: Each detection weighted equally (sum all TP/FP/FN then compute)

## Secondary Metrics

- **Candidates per trace**: Total loose candidates / N traces (measures detector sensitivity)
- **LLM confirmation rate**: confirmed / (confirmed + rejected + uncertain)
- **LLM agreement**: % of LLM decisions matching ground truth
- **FP rate per detector per HTA phase**: Phase-specific accuracy
- **Latency**: Rule detection time + LLM confirmation time per candidate (ms)
- **Cost**: Tokens per LLM call, total cost per trace

## Matching Rules (Critical)

A detection matches a ground truth annotation if:
1. **Same failure_name** (e.g., both `premature_termination`)
2. **onset_step within +/-5 steps** of annotated onset (configurable via `--match-window`)

Edge cases:
- Correct CAFT code, within step window: **full match** (1.0 TP)
- Correct CAFT code, outside step window: **partial match** (0.5 TP)
- Wrong CAFT code at right step: **FP** (wrong diagnosis)
- Ground truth failure with no detector candidate: **FN**
- Ground truth failure, detector fires, LLM rejects: **FN** (LLM error)
- Latent CAFT types (20 types requiring semantic understanding): tracked separately, don't penalize detectors

Matching algorithm: greedy best-match-first sorted by detection confidence (prevents lower-confidence detections from stealing matches from higher-confidence ones).

## Statistical Rigor

- **Bootstrap 95% CI**: 1000 iterations (configurable), resample traces with replacement
- **McNemar's test**: Pairwise comparison between modes (strict vs loose+llm, loose vs loose+llm, loose+llm vs oracle)
- **Report**: N (traces), n (total annotations), k (total candidates) for power analysis
- **Split discipline**: ALL tuning on train split, final numbers on test split ONLY
- **Normal CDF approximation** via Abramowitz & Stegun (numpy only, no scipy)

## Split Usage

| Split | Size | Purpose | Rules |
|-------|------|---------|-------|
| train | 15 traces | Threshold tuning, LLM prompt optimization | Iterate freely |
| val | 8 traces | Hyperparameter selection, early stopping | Check periodically |
| test | 7 traces | Final reported numbers | **NEVER** used for tuning |

## Implementation

### `agentdiag/metrics.py`

Core types:
- `Detection` — single detector firing for evaluation
- `Annotation` — single ground-truth label
- `MatchResult` — detection-annotation pair with match type
- `DetectorResult` — per-detector P/R/F1 with LLM stats
- `EvalReport` — full report for one mode
- `ComparisonTable` — pairwise tests + per-detector winners

Core functions:
- `match_detections(detections, annotations, window)` — matching engine
- `compute_evaluation(annotations, detections, mode)` — full evaluation
- `bootstrap_ci(annotations, detections)` — confidence intervals
- `mcnemar_test(annotations, dets_a, dets_b)` — pairwise test
- `compare_modes(reports, annotations, dets_by_mode)` — ablation comparison
- `format_comparison_table(reports, comparison)` — human-readable output

### `scripts/run_ablation.py`

Orchestrates all 4 modes. Outputs:
- `comparison_table.json` / `.txt` — machine/human readable
- `per_detector_breakdown.json` — per-detector per-mode
- `llm_decisions.jsonl` — every LLM call logged
- `bootstrap_distributions.json` — CI data for plotting

### CLI

```bash
agentdiag evaluate --ablation --annotations labels.jsonl --split test
```

## Win Condition

`loose+llm` improves precision materially over `loose` while preserving recall. Target per premature_termination POC: P >= 60%, R >= 80%.

## Prompt Evaluation Rubric

| Score | Label | Description |
|-------|-------|-------------|
| +1 | Correct confirm | Real failure correctly confirmed |
| +1 | Correct reject | Normal workflow correctly rejected |
| 0 | Uncertain (acceptable) | LLM unsure, falls back to rule confidence |
| -1 | False confirm | Normal workflow incorrectly confirmed as failure |
| -2 | False reject | Real failure incorrectly rejected (worse than false confirm) |

Error weighting: False rejects are 2x worse than false confirms (missing a real failure is worse than a false alarm).
