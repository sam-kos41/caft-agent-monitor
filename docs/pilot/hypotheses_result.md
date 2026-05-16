# Pilot H1/H2 Result (locked pre-registration §6-§8)

n=2000 (1000 resolved / 1000 unresolved)

## H1 — IT-only vs label-permutation null

- observed CV AUC: **0.7504**
- null 95th pct: 0.5281 (null mean 0.4992, 1000 perms)
- floor: 0.55
- **H1 PASS** (needs AUC > null p95 AND >= 0.55)

## H2 — incremental value over trivial baseline

- AUC baseline: 0.7239
- AUC baseline + IT: 0.7681
- ΔAUC: **+0.0442**  (95% CI [+0.0260, +0.0622], 2000 boot)
- **H2 PASS** (needs ΔAUC >= 0.03 AND CI excludes 0)

## DECISION (locked §8, audit already passed): H1_PASS_H2_PASS

**IT adds genuine incremental signal on a clean independent outcome.**

Action: Commit to the fuller four-leg program; operationalize the other legs.

_Objective. The rule executed from the H1/H2 booleans; thresholds were not re-weighed after seeing the numbers._