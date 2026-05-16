# TAC Robustness 2 — LLM-graded (descriptive, NON-gating)

- model llama3.2:3b, frozen subsample first 150 by sel_key_int; graded 149, failed 1
- Pearson(LLM rating, deterministic tac.mean) = **-0.042**
- Spearman = **-0.214**
- means: LLM 3.77/5 vs deterministic tac.mean 0.685

Robustness 1 (semantic embedding): documented-EXCLUDED (sentence-transformers unavailable offline; honest-scoping).

_Convergent evidence only. The locked §8 decision (TAC-H3 fail; TAC reduces to IT) was recorded before this ran and is not affected._