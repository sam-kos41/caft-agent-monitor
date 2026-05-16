# Agent-Native TAC Symbolization Audit

- trajectories: 2000  tool columns: 165
- Ridge(alpha=1.0), KFold(5, random_state=20260515)
- gate: INVALID if CV R² >= 0.8 for tac.mean or tac.target_match_rate (construct-bearing pair)

| gate feature | CV R² | fails? |
|---|---:|---|
| tac.mean | 0.1094 | False |
| tac.target_match_rate | -0.1855 | False |

## GATE: passes — TAC not reconstructible from tool counts

| TAC feature | CV R² |
|---|---:|
| tac.mean | 0.1094 |
| tac.min | -0.6755 |
| tac.final | -0.6719 |
| tac.slope | -0.3137 |
| tac.verb_align_rate | -0.4999 |
| tac.target_match_rate | -0.1855 |

_Locked, objective. STOP for the mandatory human checkpoint before TAC-H1/H2/H3 (PROGRAM.md rule 3)._