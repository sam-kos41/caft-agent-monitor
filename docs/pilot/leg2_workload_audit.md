# Leg 2 (Workload) Symbolization Audit

- trajectories: 2000  tool columns: 165
- Ridge(alpha=1.0), KFold(5, random_state=20260515) [standing rule]
- gate: Leg-2 INVALID if CV R² >= 0.8 for reasoning_len.mean or context_cum.final

| gate feature | CV R² | fails? |
|---|---:|---|
| reasoning_len.mean | -0.2563 | False |
| context_cum.final | 0.6606 | False |

## GATE: passes — workload not reconstructible from tool counts

| workload feature | CV R² |
|---|---:|
| reasoning_len.mean | -0.2563 |
| reasoning_len.max | -0.3339 |
| reasoning_len.total | 0.4405 |
| reasoning_len.slope | -0.0984 |
| context_cum.final | 0.6606 |
| context_cum.slope | -0.6348 |
| reasoning_density.mean | -0.2563 |
| error_recovery.n_episodes | 0.6296 |
| error_recovery.mean_latency_turns | -0.1585 |

_Locked, objective. STOP here for the mandatory human checkpoint before W-H1/H2/H3 (PROGRAM.md rule 3)._