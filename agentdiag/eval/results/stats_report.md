# Statistical Analysis Report

## 1. Binomial Test: Detection Above Chance

- H0: TPR = 0.087 (chance = FPR)
- H1: TPR > chance
- Observed TPR: 0.942
- p-value: 0.0000
- **Significant** at alpha=0.05
- 95% CI: [0.858, 1.000]

### Per-Variant Binomial Tests

| Variant | TPR | p-value | Significant |
|---------|-----|---------|-------------|
| loop | 1.000 | 0.0000 | Yes |
| drift | 1.000 | 0.0000 | Yes |
| thrash | 1.000 | 0.0000 | Yes |
| stall | 0.769 | 0.0000 | Yes |

## 2. Bootstrap 95% Confidence Intervals

- **TPR**: 0.942 [0.865, 1.000]
- **Latency (median)**: 26.0 steps [18.0, 37.0]
- Resamples: 10000

### Per-Variant Bootstrap CIs

| Variant | TPR [95% CI] | Latency [95% CI] |
|---------|-------------|------------------|
| loop | 1.000 [1.000, 1.000] | 38.0 [37.0, 40.0] |
| drift | 1.000 [1.000, 1.000] | 14.0 [10.0, 17.0] |
| thrash | 1.000 [1.000, 1.000] | 19.0 [15.0, 24.0] |
| stall | 0.769 [0.538, 1.000] | 46.0 [44.0, 48.0] |

## 3. Kruskal-Wallis: Cross-Domain Comparison

- H statistic: 5.898
- p-value: 0.4347
- **Not significant** difference across 7 domains
