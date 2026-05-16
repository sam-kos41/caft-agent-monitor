# Generalization G — outcome granularity (locked §5)

analysis n=1696/2000 (parseable subset; §1 selection bias applies). g mean=0.852, frac(g==1)=52%

## G-H1 — IT predicts graded outcome (vs permuted-g null)
- OOF Spearman **0.2770** vs null p95 0.0462 (null mean -0.0080), floor |ρ|≥0.1
- **G-H1 PASS**

## G-H2 — IT beyond trivial baseline
- Spearman base 0.1819 -> base+IT 0.2829; ΔSpearman **+0.1011** CI [+0.0649,+0.1659]
- **G-H2 PASS** (needs ≥0.03 & CI>0)

## G-H3 — does ANY collapsed construct re-separate (beyond baseline+IT)?

| construct | ΔSpearman | 95% CI | re-separates? | Bonf α/4 |
|---|---:|---|---|---|
| workload | -0.0107 | [-0.0171,+0.0220] | False | no |
| situation_awareness | -0.0091 | [-0.0177,+0.0339] | False | no |
| error_recovery | +0.0030 | [-0.0085,+0.0318] | False | no |
| thought_action_coherence | -0.0121 | [-0.0153,+0.0225] | False | no |

## DECISION (locked §5): PARSIMONY_GENERALIZES

**No construct re-separates under a graded outcome. Parsimony generalizes across outcome granularity (H_flat supported over H_artifact, on this corpus/axis).**

Action: Strong corroboration; IT-load-bearing finding stands (conditional on §1 limits).

_Conclusions conditional on §1 selection bias + single corpus/agent family. Outcome-granularity test only._