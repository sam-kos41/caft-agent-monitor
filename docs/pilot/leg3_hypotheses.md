# Leg 3 — SA-H1/H2/H3 (locked PREREG_LEG3_SA §6-§9)

n=2000

## SA-H1 — SA-only vs label-perm null
- obs CV AUC **0.6500** vs null p95 0.5265 (null mean 0.5003, 1000 perms), floor 0.55
- **SA-H1 PASS**

## SA-H2 — beyond trivial baseline
- AUC base 0.7239 -> base+SA 0.7294; ΔAUC **+0.0056** CI [-0.0130,+0.0245] (2000 boot)
- **SA-H2 FAIL** (needs ≥0.03 & CI>0)

## SA-H3 — LEG-DEFINING: beyond baseline+Leg-1 IT
- AUC base+IT 0.7681 -> +SA 0.7714; ΔAUC **+0.0033** CI [-0.0036,+0.0100] (2000 boot)
- **SA-H3 FAIL** (needs ≥0.02 & CI>0)

- descriptive: max |corr| any SA vs any IT feature = 0.638

## DECISION (locked §9): SA_H1_PASS_SA_H3_FAIL

**SA redundant with Leg-1 IT — not distinct. Strong evidence for the 'IT is the load-bearing construct' thesis (SA was the strongest distinctness candidate).**

Action: Documented major finding; proceed to Leg 4.

_Objective; rule executed from booleans; thresholds not re-weighed after seeing numbers._