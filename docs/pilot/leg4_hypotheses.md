# Leg 4 — E-H1/H2/H3 (locked PREREG_LEG4 §6-§7) — FINAL LEG

n=2000

## E-H1 — ER-only vs label-perm null
- obs CV AUC **0.7127** vs null p95 0.5273 (null mean 0.5002, 1000 perms), floor 0.55
- **E-H1 PASS**

## E-H2 — beyond trivial baseline
- AUC base 0.7239 -> base+ER 0.7522; ΔAUC **+0.0283** CI [+0.0112,+0.0467] (2000 boot)
- **E-H2 FAIL** (needs ≥0.03 & CI>0)

## E-H3 — LEG-DEFINING / DECISIVE: beyond baseline+Leg-1 IT
- AUC base+IT 0.7681 -> +ER 0.7719; ΔAUC **+0.0038** CI [-0.0046,+0.0116] (2000 boot)
- **E-H3 FAIL** (needs ≥0.02 & CI>0)

- descriptive: max |corr| any ER vs any IT feature = 0.747

## DECISION (locked §7): E_H1_PASS_E_H3_FAIL

**Error recovery redundant with Leg-1 IT — not distinct. COMPLETES the parsimony result: all 3 tested HF analogs reduce to IT (airtight; strongest-distinct axis too).**

Action: Document; END empirical phase -> paper.

_Objective; rule executed from booleans; thresholds not re-weighed. Empirical phase ends here regardless._