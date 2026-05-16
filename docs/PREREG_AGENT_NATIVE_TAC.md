# Pre-Registration — Agent-Native Pilot: Thought-Action Coherence (TAC)

**Date:** 2026-05-15
**Status:** PRE-REGISTERED. Locked before any TAC data-touching code.
Sample/features/statistics/thresholds/decision-rule FROZEN on commit;
changes require a dated §10 amendment. Inherits all standing
discipline in `PROGRAM.md`.

## 0. Framing

The empirical phase closed with a parsimony result: three
human-factors *ports* (workload, situation awareness, error recovery)
each predicted the outcome but all reduced to Leg-1 IT structure. This
pilot tests a different class: an **agent-native** construct with no
clean human analog. Humans do not externalize an explicit reasoning
channel before acting; LLM agents do (`thought` precedes `action`
every turn). TAC = the consistency between stated intent and the
action taken. It is conceptually orthogonal to IT (which measures the
*shape of the action sequence over time*, ignoring `thought`
entirely). This is a genuine test of whether agent-native constructs
can survive the discriminant gate the HF ports failed. Confirmatory
of neither outcome; the locked rule fires. **Not** a return to the
paper phase.

## 1. Question

Do thought-action-coherence features, aggregated per trajectory,
(TAC-H1) separate the test-based outcome above a label-shuffle null,
(TAC-H2) add predictive power beyond a trivial baseline, and —
leg-defining — (TAC-H3) add predictive power **beyond Leg-1 IT**
(discriminant validity, the gate the HF ports failed 3/3)?

## 2. Sample (frozen — REUSED)

The exact frozen sample used by Legs 1-4 (`docs/pilot/
sample_manifest.csv`, seed 20260515, 1000/1000). No new draw. TAC-H3
compares TAC vs IT on identical trajectories (caches align by row;
verified for IT/WL/SA/ER, same for TAC).

## 3. Construct & operationalizations

**Construct:** thought-action coherence = the degree to which the
agent's stated intent in a turn's `thought` corresponds to the
`action` it then emits that turn.

**PRIMARY (deterministic, decision-bearing).** Honest-scoping note:
the loose phrase "literal command match" is not faithfully
computable — `thought` is prose ("let me open the dispatcher file"),
`action` is a structured command (`open .../dispatcher.py`). The
deterministic operationalization is **lexical intent-action
alignment**, defined exactly:
- Per turn with a non-empty thought (ai text minus fenced blocks) and
  a parsed action `(verb, target)`:
  - `verb_align` = 1 iff the thought contains ≥1 token from the
    verb-class lexicon matching the action's category, else 0.
    Lexicon (FROZEN): observe={look,read,open,view,inspect,check,
    see,examine}; search={search,find,locate,grep,where}; mutate=
    {edit,change,modify,fix,add,create,write,update,replace,implement};
    verify={run,execute,test,verify,confirm,check}; submit={submit,
    done,finish,complete}. Action category from the swe_agent
    verb→category map (observe/search/mutate/verify/submit/other).
  - `target_present` = 1 iff the action has a target token and a
    case-insensitive normalized form of it (basename, or the search
    query's first alpha token) appears as a substring in the thought;
    if the action has no target (e.g. `submit`), `target_present` :=
    `verb_align` (do not penalize target-less actions).
  - per-turn `tac = 0.5*verb_align + 0.5*target_present`.
- Per-trajectory features (6, FROZEN): `tac.mean`, `tac.min`,
  `tac.final`, `tac.slope` (OLS over turns), `tac.verb_align_rate`
  (mean verb_align), `tac.target_match_rate` (mean target_present
  over turns whose action had a target; 0 if none).

**ROBUSTNESS 1 — semantic (descriptive, NON-gating).** Cosine
similarity between an embedding of the thought and an embedding of a
templated action description, per turn, aggregated to `tac_sem.mean`.
Run **only if** a local embedding model is available offline; if not,
documented-excluded (honest-scoping, no faked proxy). Reported as:
correlation with primary `tac.mean`, and whether TAC-H1/H3 directions
replicate. Does NOT change the locked decision.

**ROBUSTNESS 2 — LLM-graded (descriptive, NON-gating).** On a frozen
sub-sample of **150** trajectories (first 150 by `sel_key_int` order
from the manifest), the local Ollama model rates overall
thought-action correspondence 1-5 from the trajectory digest.
Reported as: correlation with primary `tac.mean`. Does NOT change the
locked decision. (Subsample + non-gating by design — avoids a
multiple-operationalization goalpost.)

## 4. Symbolization-audit GATE (runs first; standing rule)

Same procedure (`PILOT_PREREGISTRATION.md` §5/A2): Ridge(alpha=1.0),
per-trajectory tool_name count design matrix, KFold(5, shuffle,
random_state=20260515) CV R². **Gate features (construct-bearing,
least composition-like):** `tac.mean` and `tac.target_match_rate`.
Pilot INVALID if CV R² ≥ **0.80** for either. Mandatory human
checkpoint at this gate before TAC-H1/H2/H3.

## 5-7. Hypotheses & thresholds (identical family to Legs 1-4)

Identical thresholds by deliberate choice (cross-leg comparability;
no per-pilot analytic degree of freedom).
- **TAC-H1** — TAC-only L2 logistic; mean test ROC AUC,
  StratifiedKFold(5, random_state=20260515); null = 1000 label
  permutations. PASS iff observed > null 95th pct AND ≥ **0.55**.
- **TAC-H2** — baseline (n_turns, n_parsed_actions, patch_len,
  exit_status, model_name) vs baseline+6 TAC. OOF ΔAUC, 2000-boot
  CI. PASS iff mean ΔAUC ≥ **0.03** AND CI excludes 0.
- **TAC-H3 (LEG-DEFINING)** — baseline+20 IT vs baseline+IT+6 TAC.
  OOF ΔAUC, 2000-boot CI. PASS iff mean ΔAUC ≥ **0.02** AND CI
  excludes 0. Report max |corr| TAC-vs-IT (descriptive).

## 8. Decision rule (LOCKED — executes itself; audit first)

| Outcome | Conclusion | Action |
|---|---|---|
| Audit FAILS | TAC = tool-API artifact | INVALID; methods finding; TAC-H* not interpreted |
| Audit passes, **TAC-H1 fails** | TAC carries no outcome signal | Document negative; stop |
| Audit passes, TAC-H1 passes, **TAC-H3 fails** | TAC reduces to IT — even an agent-native construct collapses. **Deepens the parsimony / IT-load-bearing thesis** (now also resists an agent-native construct). | Document; stop |
| Audit passes, **TAC-H1 + TAC-H3 pass** | TAC is a DISTINCT, validated agent-native construct — the first to survive the gate the 3 HF ports failed. Evidence that agent-native measurement is a viable research program. | Document; this becomes the lead for any future program |

Robustness 1/2 are reported alongside but do NOT alter this rule.
No interpretation outside the table without a §10 amendment.

## 9. Frozen constants & amendment log

Frozen: sample = Legs-1-4 manifest (seed 20260515); verb-class
lexicon (§3); the 6 TAC features; per-turn tac = 0.5*verb_align +
0.5*target_present; target-less := verb_align; gate features tac.mean
+ tac.target_match_rate, R²<0.80; H1 floor 0.55 + null 95th pct (1000
perms); H2 ΔAUC ≥0.03; H3 ΔAUC ≥0.02; both CIs exclude 0;
StratifiedKFold(5, random_state=20260515); 2000-boot CI; Robustness-2
subsample = first 150 by sel_key_int, non-gating.

Amendments (dated, with rationale) — none.

## 10. What a pass/fail does NOT claim

A pass: TAC is a distinct outcome-relevant agent-native measurement on
this one agent family / corpus — not that the agent "means what it
says", nor a causal/faithfulness claim about reasoning. A
collapse-into-IT deepens the parsimony result on this
corpus/operationalization; not a universal proof.
