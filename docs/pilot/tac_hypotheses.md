# Agent-Native TAC — H1/H2/H3 (locked PREREG §5-§8)

n=2000  (PRIMARY deterministic operationalization; robustness 1/2 reported separately, non-gating)

## TAC-H1 — TAC-only vs label-perm null
- obs CV AUC **0.6671** vs null p95 0.5274 (null mean 0.5002, 1000 perms), floor 0.55
- **TAC-H1 PASS**

## TAC-H2 — beyond trivial baseline
- AUC base 0.7239 -> base+TAC 0.7303; ΔAUC **+0.0064** CI [-0.0095,+0.0229] (2000 boot)
- **TAC-H2 FAIL** (needs ≥0.03 & CI>0)

## TAC-H3 — LEG-DEFINING: beyond baseline+Leg-1 IT
- AUC base+IT 0.7681 -> +TAC 0.7679; ΔAUC **-0.0002** CI [-0.0042,+0.0040] (2000 boot)
- **TAC-H3 FAIL** (needs ≥0.02 & CI>0)

- descriptive: max |corr| any TAC vs any IT feature = 0.434

## DECISION (locked §8): TAC_H1_PASS_TAC_H3_FAIL

**TAC reduces to IT — even an agent-native construct (on an axis that ignores the action-sequence shape and uses the thought channel) collapses. DEEPENS the parsimony / IT-load-bearing thesis.**

Action: Document; stop.

_Objective; rule executed from booleans; thresholds not re-weighed. Robustness checks cannot alter this decision._