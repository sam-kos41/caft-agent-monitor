# Leg 2 — W-H1/H2/H3 (locked PREREG_LEG2_WORKLOAD §6-§9)

n=2000

## W-H1 — workload-only vs label-perm null
- obs CV AUC **0.7014** vs null p95 0.5272 (null mean 0.5013, 1000 perms), floor 0.55
- **W-H1 PASS**

## W-H2 — beyond trivial baseline (controls volume)
- AUC base 0.7239 -> base+workload 0.7480; ΔAUC **+0.0242** CI [+0.0080,+0.0413] (2000 boot)
- **W-H2 FAIL** (needs ≥0.03 & CI>0)

## W-H3 — LEG-DEFINING: beyond baseline+Leg-1 IT
- AUC base+IT 0.7681 -> +workload 0.7739; ΔAUC **+0.0058** CI [-0.0033,+0.0151] (2000 boot)
- **W-H3 FAIL** (needs ≥0.02 & CI>0)

- descriptive: max |corr| any workload vs any IT feature = 0.736

## DECISION (locked §9): W_H1_PASS_W_H3_FAIL

**Workload is redundant with Leg-1 IT — not a distinct construct.**  Action: Fold into Leg 1; proceed to Leg 3.

_Objective; rule executed from booleans; thresholds not re-weighed after seeing numbers._