# Pilot Symbolization Audit

- trajectories: 2000
- tool-name columns (design matrix): 165
- estimator: Ridge(alpha=1.0), KFold(5, shuffle, random_state=20260515)  [pre-reg A2]
- gate: pilot INVALID if CV R² >= 0.8 for action_mi.mean or compression_ratio.mean

## Gated features

| feature | CV R² | fails gate? |
|---|---:|---|
| action_mi.mean | -0.8815 | False |
| compression_ratio.mean | 0.1711 | False |

## GATE OUTCOME: audit passes — IT not reconstructible from tool-API counts

## All 20 IT features — CV R² (tool-counts -> feature)

| feature | CV R² |
|---|---:|
| action_mi.mean | -0.8815 |
| action_mi.final | -0.6511 |
| action_mi.max | -0.4274 |
| action_mi.slope | -0.1118 |
| action_entropy.mean | -1.2204 |
| action_entropy.final | -1.1841 |
| action_entropy.max | -1.1339 |
| action_entropy.slope | -0.1519 |
| tool_entropy.mean | -4.9173 |
| tool_entropy.final | -2.6457 |
| tool_entropy.max | -3.0303 |
| tool_entropy.slope | -0.5857 |
| compression_ratio.mean | 0.1711 |
| compression_ratio.final | 0.1777 |
| compression_ratio.max | n/a (near-constant) |
| compression_ratio.slope | -0.0636 |
| kl_divergence.mean | 0.2035 |
| kl_divergence.final | -0.3668 |
| kl_divergence.max | 0.3255 |
| kl_divergence.slope | -0.1542 |

_Locked, objective. Not re-litigated. Per pre-reg, if the gate fails the pilot is INVALID and H1/H2 are not interpreted._