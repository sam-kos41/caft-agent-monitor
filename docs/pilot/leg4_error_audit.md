# Leg 4 (Error Recovery) Symbolization Audit

- trajectories: 2000  tool columns: 165
- Ridge(alpha=1.0), KFold(5, random_state=20260515)
- gate: Leg-4 INVALID if CV R² >= 0.8 for error.strategy_change_rate or error.recovery_success_rate (the adaptive/relational pair)

| gate feature | CV R² | fails? |
|---|---:|---|
| error.strategy_change_rate | -0.1002 | False |
| error.recovery_success_rate | -0.4816 | False |

## GATE: passes — adaptive ER not reconstructible from tool counts

| ER feature | CV R² |
|---|---:|
| error.n_episodes | 0.6307 |
| error.recurrence_rate | 0.2750 |
| error.strategy_change_rate | -0.1002 |
| error.recovery_success_rate | -0.4816 |
| error.mean_latency_turns | -0.2249 |
| error.terminal_unresolved | 0.2116 |

_Locked, objective. STOP for the mandatory human checkpoint before E-H1/H2/H3 (PROGRAM.md rule 3)._