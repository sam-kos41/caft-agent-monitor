# Leg 3 (Situation Awareness) Symbolization Audit

- trajectories: 2000  tool columns: 165
- Ridge(alpha=1.0), KFold(5, random_state=20260515)
- gate: Leg-3 INVALID if CV R² >= 0.8 for perception.coverage or projection.verify_before_submit (the relational features)

| gate feature | CV R² | fails? |
|---|---:|---|
| perception.coverage | -2.3330 | False |
| projection.verify_before_submit | -1.7821 | False |

## GATE: passes — relational SA not reconstructible from tool counts

| SA feature | CV R² |
|---|---:|
| perception.coverage | -2.3330 |
| perception.explore_ratio | 0.1211 |
| perception.read_before_first_edit | -1.3819 |
| perception.blind_edit_rate | 0.3708 |
| projection.verify_before_submit | -1.7821 |
| projection.verify_rate | 0.0791 |

_Locked, objective. STOP for the mandatory human checkpoint before SA-H1/H2/H3 (PROGRAM.md rule 3)._