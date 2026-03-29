# Sensitivity Analysis

## Overall Optimal Operating Point

| Metric | z=3.0 (default) | z=2.75 (optimal) |
|--------|-----------------|--------------------|
| TPR | 0.942 | 0.981 |
| FPR | 0.0868 | 0.1212 |
| Youden's J | 0.856 | 0.860 |
| Sig. Accuracy | 0.980 | 0.980 |
| Latency (median) | 26.0 steps | 24.0 steps |

## Optimal Threshold by Failure Type

| Failure Type | Optimal z | J | TPR | FPR | TPR @z=3.0 |
|-------------|-----------|-----|-----|-----|-----------|
| loop | 3.50 | 0.924 | 1.000 | 0.0757 | 1.000 |
| drift | 4.75 | 1.000 | 1.000 | 0.0000 | 1.000 |
| thrash | 3.75 | 0.929 | 1.000 | 0.0707 | 1.000 |
| stall | 2.50 | 0.850 | 1.000 | 0.1496 | 0.769 |

## Threshold Divergence

Optimal thresholds diverge by 2.25 across failure types (range: 2.50 to 4.75). Consider per-failure-type thresholds for production use.
