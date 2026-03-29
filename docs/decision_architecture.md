# agentdiag Decision Architecture Map

> Generated 2026-03-18 from source analysis of the full detection pipeline.
> All thresholds, line numbers, and code references are from the actual codebase.

---

## 1. Event Flow Diagram (ASCII)

```
                    Raw JSONL (Claude Code session log)
                                  |
                                  v
                    +----------------------------+
                    | ClaudeCodeExtractor         |
                    | adapters/claude_code.py     |
                    | .discover() -> SessionInfo  |
                    | .parse_session() -> events  |
                    +----------------------------+
                                  |
                          list[TraceEvent]
                    (step, type, tool, latency_ms,
                     success, tokens_in/out,
                     output_hash, input_hash,
                     goal_text, error_message)
                                  |
                                  v
               +--------------------------------------+
               | MonitorEngine.push(event)            |
               | monitor.py:375                       |
               |                                      |
               |  1. Append to self._events           |
               |  2. Count errors                     |
               |  3. HTA classification ----------+   |
               |  4. Track highest phase           |   |
               |  5. Build ActionEntry             |   |
               |  6. Run CAFT detectors --------+  |   |
               |  7. Apply confirmation ------+  |  |   |
               |  8. Update trust score       |  |  |   |
               |  9. Record to context store  |  |  |   |
               +--------------------------------------+
                                  |  |  |
          +-----------------------+  |  +------------------------+
          |                          |                           |
          v                          v                           v
  +----------------+    +----------------------+    +------------------+
  | HTAStateMachine|    | run_caft_detectors() |    | _apply_          |
  | hta.py         |    | detectors.py:1459    |    | confirmation()   |
  |                |    |                      |    | monitor.py:284   |
  | classify_event |    | For each detector:   |    |                  |
  |  -> (Phase,    |    |  if name in seen:    |    | conf >= 0.9?     |
  |     is_strong) |    |    SKIP              |    |  -> auto-confirm |
  |                |    |  d.check(events,hta) |    |                  |
  | Apply          |    |  if diagnosis:       |    | LLM available?   |
  | hysteresis     |    |    seen.add(name)    |    |  -> call LLM     |
  | (2 events or   |    |    yield diagnosis   |    |                  |
  |  strong signal)|    |                      |    | else:            |
  +----------------+    +----------------------+    |  -> uncertain    |
          |                          |              +------------------+
          v                          v                       |
     HTAState               list[CaftDiagnosis]              v
     (phase,                  (candidates)           ConfirmationResult
      transitions,                                   (confirmed/rejected/
      regression_count)                               uncertain)
                                                             |
                                                             v
                                                    +------------------+
                                                    | DashboardState   |
                                                    | monitor.py:52    |
                                                    |                  |
                                                    | .diagnoses[]     |
                                                    | .health          |
                                                    | .trust_score     |
                                                    | .completion_rate |
                                                    | .failure_density |
                                                    +------------------+

  POST-PROCESSING (ablation / batch mode only):
  =============================================

  After all events pushed:

    +-------------------+     +---------------------+     +------------------+
    | _RETRACTABLE      |     | _RECONFIRMABLE      |     | Session Dedup    |
    | {"recovery_       |     | {"stall"}           |     |                  |
    |   failure"}       |     |                     |     | Group by 8-char  |
    |                   |     | Re-run detector on  |     | prefix, keep     |
    | Re-run detector   |     | full trace. If new  |     | longest ID.      |
    | on full trace.    |     | confidence > old,   |     | Remap            |
    | If returns None,  |     | upgrade. Also       |     | annotations.     |
    | retract diagnosis.|     | update force_llm_   |     |                  |
    |                   |     | review flag.        |     |                  |
    +-------------------+     +---------------------+     +------------------+
```


## 2. HTA Phase State Machine

### 2.1 Phases (IntEnum ordering)

| Phase | Value | Description | Color |
|-------|-------|-------------|-------|
| IDLE | 0 | No activity yet | dim |
| GATHERING | 1 | Reading files, searching, fetching context | cyan |
| PLANNING | 2 | Reasoning, outlining, designing approach | yellow |
| EXECUTING | 3 | Writing code, editing files, running commands | green |
| VERIFYING | 4 | Running tests, reviewing output, checking results | magenta |
| DELIVERING | 5 | Committing, summarizing, presenting final output | blue |

Source: `agentdiag/hta.py:25-32`

### 2.2 Classification Rules

Events are classified by tool name substring matching, checked in order of specificity (most specific phase first):

| Priority | Phase | Tool Set (substring match) | Source Line |
|----------|-------|---------------------------|-------------|
| 1 | PLANNING | Event type in `{"reasoning", "planning", "thinking"}` | hta.py:90 |
| 2 | DELIVERING | `{"commit", "git_commit", "push", "deploy", "publish", "submit", "send", "deliver", "output", "summarize"}` | hta.py:68-71 |
| 3 | VERIFYING | `{"run_tests", "test", "pytest", "jest", "check", "lint", "validate", "verify", "review", "diff", "compare"}` | hta.py:64-67 |
| 4 | EXECUTING | `{"write_file", "edit_file", "edit", "write", "create_file", "run_code", "execute", "bash", "shell", "npm", "pip", "install", "build", "compile", "generate"}` | hta.py:59-63 |
| 5 | GATHERING | `{"read_file", "search_docs", "web_search", "search_codebase", "list_files", "glob", "grep", "find", "cat", "head", "tail", "read", "fetch", "get", "list", "ls", "describe"}` | hta.py:53-57 |
| 6 | DELIVERING | Event type == `"output"` | hta.py:110-111 |
| 7 | EXECUTING | Event type == `"tool_call"` (unclassified fallback) | hta.py:114-115 |
| 8 | PLANNING | Default for everything else | hta.py:117 |

### 2.3 Hysteresis Mechanism

- **HYSTERESIS = 2**: A transition to a new phase requires 2 consecutive events classified to that phase before the transition commits. (`hta.py:209`)
- **Pending state**: `_pending_phase` and `_pending_count` track the candidate transition.
- **Reset**: If an event matches the current phase while a transition is pending, the pending state resets. (`hta.py:277-279`)
- **First event**: IDLE always transitions on the first event regardless of hysteresis. (`hta.py:282-283`)

### 2.4 Strong Signals (Bypass Hysteresis)

Strong signals are unambiguous phase indicators that cause immediate transition, bypassing the 2-event hysteresis. (`hta.py:73-79, 262-266`)

| Phase | Strong Signal Tools |
|-------|-------------------|
| EXECUTING | `{"write", "edit", "write_file", "edit_file", "create_file", "bash"}` |
| VERIFYING | `{"pytest", "jest", "run_tests"}` |
| DELIVERING | `{"commit", "git_commit", "push", "deploy"}` |
| GATHERING | None (no strong signals) |
| PLANNING | None (no strong signals) |

### 2.5 Regression Detection

A regression is recorded whenever `new_phase < current_phase` (the IntEnum comparison). Regressions are normal (e.g., EXECUTING -> GATHERING to read another file) but are tracked for anomaly detection. (`hta.py:292`)

Phase regression count is used by:
- `DashboardState.completion_rate`: penalized by `regression_count * 5` (`monitor.py:247-251`)
- `GoalDriftDetector`: counts unprompted regressions (`detectors.py:791-793`)


## 3. Per-Detector Decision Table

### 3.1 Summary Table

| # | Detector | CAFT | Trigger Pattern | Confidence Formula | Auto-Confirm Possible? | force_llm_review? | Retraction? | Reconfirm? | Status |
|---|----------|------|-----------------|-------------------|----------------------|-------------------|-------------|------------|--------|
| 1 | step_repetition | 2.2 | 9+ consecutive identical (tool, input_hash) | `min(max_run / 18, 1.0)` | Yes (if conf >= 0.9) | No | No | No | **Active** (strict+loose) |
| 2 | context_loss | 2.1 | Re-read resource with gap >= max(5, 8% of events) | `min(intervening / 10.0, 1.0)` | Yes (if conf >= 0.9) | No | No | No | **Active** (strict+loose) |
| 3 | premature_termination | 5.4 | 3 modes (see below) | 0.85 / 0.5+0.1*n / 0.4 | Mode 1 only | No | No | No | **Active** (strict+loose) |
| 4 | tool_misuse | 4.1 | Sliding window: high switch rate * stagnation >= 0.4 | `min(score, 0.85)` | No (max 0.85) | No | No | No | **Active** (strict+loose) |
| 5 | stall | 4.4 | IQR outlier on latencies, 5%+ fraction | `min(0.5 + excess*0.3, 1.0)` | Yes, but sparse stalls forced to LLM | **Yes** (if stall_count <= 2) | No | **Yes** (`_RECONFIRMABLE`) | **Active** (strict+loose) |
| 6 | error_cascade | 4.2 | 3+ consecutive failures | `min(chain_len / (events*0.3), 1.0)` | Yes (if conf >= 0.9) | **Yes** (if chain < 5) | No | No | **Active** (strict+loose) |
| 7 | token_explosion | 4.4 | Growth >3x first-to-last quarter + acceleration | `min(score, 0.9)` | Yes (if conf >= 0.9) | No | No | No | **Active** (strict+loose) |
| 8 | analysis_paralysis | 3.4 | 4+ consecutive reasoning/planning without tool_call | `min(max_run / 12, 0.85)` | No (max 0.85) | No | No | No | **Active** (strict+loose) |
| 9 | recovery_failure | 4.3 | 40%+ recovery failure rate with progress filters | `min(failure_rate, 1.0)` | Yes (if conf >= 0.9) | No | **Yes** (`_RETRACTABLE`) | No | **Active** (strict+loose) |
| 10 | missing_verification | 5.3 | 15+ execution events with writes, no verification | `min(exec / 45, 0.9)` | No (max 0.9) | No | No | No | **Disabled** (loose only, 14 FP) |
| 11 | goal_drift | 2.4 | 3+ drifted blocks + 3+ unprompted regressions | `min(drift_blocks / 5.0, 0.85)` | No (max 0.85) | No | No | No | **Disabled** (loose only, 11 FP) |
| 12 | tool_thrashing | 3.1 | 15+ consecutive read-only ops in EXECUTING (25+ any phase) | `min(max_run / (thresh*2), 0.85)` | No (max 0.85) | No | No | No | **Disabled** (loose only, 9+ FP) |
| 13 | reasoning_action_mismatch | 6.4 | Reasoning intent contradicts next tool call | 0.65-0.70 (fixed) | No | No | No | No | **Disabled** (loose only, keyword-brittle) |

### 3.2 Detailed Detector Specifications

#### Detector 1: StepRepetitionDetector (CAFT 2.2)
- **File**: `detectors.py:151-256`
- **Trigger**: `THRESHOLD = 9` consecutive identical `(tool, input_hash)` pairs in the last `THRESHOLD * 3 = 27` events
- **Output diversity check**: If >80% unique `output_hash` in the run, suppressed (agent making progress). (`detectors.py:218-221`)
- **Phase-aware**: In GATHERING phase, threshold raised to `max(9, 10) = 10`. (`detectors.py:225-228`)
- **Meta-tool exclusion**: Tools in `{"task", "exitplanmode", "enterplanmode", "askuserquestion", "todowrite", ...}` are excluded. (`detectors.py:231-236`)
- **Confidence**: `min(max_run / (THRESHOLD * 2), 1.0)` = `min(max_run / 18, 1.0)`. Reaches 1.0 at 18 repetitions.
- **Severity**: WARNING if `max_run < 10`, else CRITICAL. (`detectors.py:242`)

#### Detector 2: ContextLossDetector (CAFT 2.1)
- **File**: `detectors.py:259-365`
- **Trigger**: Re-read of same `output_hash` with gap >= `max(MIN_INTERVENING=5, int(len(events) * 0.08))` intervening non-read tool calls
- **Exclusions**: Pairs with user_input or continuation event between are skipped. (`detectors.py:313-320`)
- **Selection**: All valid pairs collected; fires on the strongest (largest gap). (`detectors.py:341`)
- **Confidence**: `min(interv_count / 10.0, 1.0)`. Reaches 1.0 at 10 intervening operations.
- **Severity**: Always WARNING. (`detectors.py:348`)

#### Detector 3: PrematureTerminationDetector (CAFT 5.4)
- **File**: `detectors.py:368-537`
- **Mode 1 (skip_verification)**: Current phase == DELIVERING, delivering_count >= 3, has executed >= 3 steps, never visited VERIFYING, no delegated verification. Confidence = 0.85 (fixed). Severity = CRITICAL. (`detectors.py:409-440`)
- **Mode 2 (plan_not_executed)**: Current phase in {GATHERING, PLANNING}, exec_count < 3, execution intent detected in reasoning text, >= 10 events after last intent (or >= 5 if session > 50 events). Confidence = `min(0.5 + len(intent_steps) * 0.1, 0.85)`. Severity = WARNING. (`detectors.py:442-484`)
- **Mode 3 (no_delivery)**: exec_count >= 7, delivering_count == 0, verifying_count == 0, phase in {EXECUTING, PLANNING, GATHERING}, events >= 25, no user cancellation in last 10 events. Confidence = 0.4 (fixed, designed for LLM confirmation). Severity = WARNING. (`detectors.py:492-535`)
- **User cancellation check**: Pattern `(?:never ?mind|stop|cancel|forget it|don't|do not)` in last 10 events. (`detectors.py:395-398, 504-509`)
- **Delegated verification**: Checked via `_has_delegated_verification()` using `_VERIFICATION_KEYWORDS` regex. (`detectors.py:67-78, 118-128`)

#### Detector 4: ToolMisuseDetector (CAFT 4.1)
- **File**: `detectors.py:902-968`
- **Trigger**: Sliding window of size 5 over tool_call events. `score = switch_rate * stagnation`. Fires if `max_thrash_score >= 0.4`. (`detectors.py:942`)
- **switch_rate**: Fraction of consecutive pairs with different tools in window.
- **stagnation**: `1.0 - (state_changes / 4)` where state_changes counts consecutive pairs with different output_hash.
- **Confidence**: `min(max_thrash_score, 0.85)`. Maximum possible = 0.85 (never auto-confirms at 0.9).
- **Severity**: WARNING if confidence < 0.7, else CRITICAL.

#### Detector 5: StallDetector (CAFT 4.4)
- **File**: `detectors.py:971-1070`
- **IQR_MULTIPLIER** = 1.5, **MIN_EVENTS** = 10, **MIN_STALL_MS** = 3000.0ms, **MAX_INTER_TURN_MS** = 120,000ms (2min), **MIN_STALL_FRACTION** = 0.05 (5%). (`detectors.py:985-989`)
- **Threshold**: `max(Q75 + 1.5 * IQR, 3000.0)` on active (intra-turn) latencies only.
- **Inter-turn filtering**: Latencies > 120s excluded as user idle time. (`detectors.py:1003-1004`)
- **Minimum fraction**: At least 5% of active steps must be stalls. (`detectors.py:1020-1021`)
- **Confidence**: `min(0.5 + excess * 0.3, 1.0)` where `excess = (max_latency - threshold) / threshold`. (`detectors.py:1030-1031`)
- **force_llm_review**: Set to True when `stall_count <= 2` (sparse stalls may be single slow Bash ops). (`detectors.py:1039, 1069`)
- **Minimum confidence gate**: Returns None if confidence < 0.3. (`detectors.py:1041-1042`)
- **Severity**: WARNING if confidence < 0.7, else CRITICAL.
- **Post-processing**: In `_RECONFIRMABLE` set. Full-trace re-evaluation may upgrade confidence and clear force_llm_review. (`run_ablation.py:297, 310-318`)

#### Detector 6: ErrorCascadeDetector (CAFT 4.2)
- **File**: `detectors.py:1073-1139`
- **MIN_CHAIN** = 3 consecutive failures. (`detectors.py:1082`)
- **Confidence**: `min(len(longest_chain) / max(len(events) * 0.3, 1), 1.0)`. For a 100-event trace, chain of 30 = conf 1.0. (`detectors.py:1110`)
- **Minimum confidence gate**: Returns None if confidence < 0.3. (`detectors.py:1111-1112`)
- **force_llm_review**: Set to True when `len(longest_chain) < 5` (short chains may be normal search refinement). (`detectors.py:1116, 1138`)
- **Severity**: WARNING if confidence < 0.7, else CRITICAL.

#### Detector 7: TokenExplosionDetector (CAFT 4.4)
- **File**: `detectors.py:1142-1220`
- **GROWTH_THRESHOLD** = 3.0x, **MIN_SCORE** = 0.5, **MIN_EVENTS** = 50, **MIN_FIRST_Q_TOKENS** = 50. (`detectors.py:1154-1157`)
- **Growth ratio**: `last_quarter_mean / first_quarter_mean`. Only fires if > 3.0x.
- **Score**: `growth_component + acceleration_component`, each capped at 0.5. Must exceed 0.5 total.
- **Confidence**: `min(score, 0.9)`. (`detectors.py:1199`)
- **Severity**: WARNING if confidence < 0.7, else CRITICAL.

#### Detector 8: AnalysisParalysisDetector (CAFT 3.4)
- **File**: `detectors.py:1223-1281`
- **THRESHOLD** = 4 consecutive reasoning/planning events with no tool_call. (`detectors.py:1232`)
- **Confidence**: `min(max_run / (THRESHOLD * 3), 0.85)` = `min(max_run / 12, 0.85)`. Reaches 0.85 at 12+ events.
- **Severity**: WARNING if confidence < 0.7, else CRITICAL.

#### Detector 9: RecoveryFailureDetector (CAFT 4.3)
- **File**: `detectors.py:1284-1414`
- **RECOVERY_WINDOW** = 3 events after each error. Counts failures in that window.
- **failure_rate threshold**: >= 0.4 (40%). (`detectors.py:1343`)
- **Progress filter 1**: If 5+ unique successful operations after the last failed recovery, suppressed. (`detectors.py:1349-1356`)
- **Stuck retry detection**: Counts same tool + same input_hash retries after errors. (`detectors.py:1363-1376`)
- **Progress filter 2 (productive session)**: 50+ tool_call events with 80%+ success rate and 0 stuck retries = suppressed. (`detectors.py:1382-1387`)
- **Confidence**: `min(failure_rate, 1.0)`. (`detectors.py:1389`)
- **Post-processing**: In `_RETRACTABLE` set. Full-trace re-evaluation may retract if progress filters now pass. (`run_ablation.py:296, 306-309`)

#### Detector 10: MissingVerificationDetector (CAFT 5.3) -- DISABLED
- **File**: `detectors.py:540-657`
- **EXECUTE_THRESHOLD** = 15. Only fires after 15+ execution events with writes and no verification across 5 patterns (HTA verification phase, bash test commands, Task subagent, delegated verification, agent acknowledging results). (`detectors.py:558, 578-607`)
- **Confidence**: `min(exec_count / 45, 0.9)`. (`detectors.py:644`)
- **Status**: Disabled in registry (14 FP on 20 real traces). Available in `ALL_CAFT_DETECTORS_FULL` (loose mode). (`registry.py:155`)

#### Detector 11: GoalDriftDetector (CAFT 2.4) -- DISABLED
- **File**: `detectors.py:743-828`
- **MIN_EVENTS** = 20. Requires 3+ agent blocks. (`detectors.py:757, 769`)
- **Composite trigger**: (drift_blocks >= 3 AND unprompted_regressions >= 3) OR (drift_blocks >= 2 AND unprompted_regressions >= 5). (`detectors.py:799-804`)
- **Block drift**: >50% novel tools in block with >= 3 events. (`detectors.py:787`)
- **Confidence**: `min(drift_blocks / 5.0, 0.85)` or `min(drift_blocks / 4.0, 0.85)`. (`detectors.py:801, 804`)
- **Status**: Disabled in registry (11 FP). Available in `ALL_CAFT_DETECTORS_FULL`. (`registry.py:156`)

#### Detector 12: ToolThrashingDetector (CAFT 3.1) -- DISABLED
- **File**: `detectors.py:831-895`
- **EXECUTING_THRESHOLD** = 15 (consecutive read-only ops in EXECUTING). **ANY_PHASE_THRESHOLD** = 25 (in other phases). (`detectors.py:841-842`)
- **Read-only tools**: `{"read_file", "read", "cat", "head", "tail", "grep", "glob", "find", "list_files", "ls", "search", "fetch", "get", "describe", "search_docs", "web_search", "search_codebase"}`. (`detectors.py:844-848`)
- **Confidence**: `min(max_run / (threshold * 2), 0.85)`. (`detectors.py:883`)
- **Status**: Disabled in registry (9+ FP). Available in `ALL_CAFT_DETECTORS_FULL`. (`registry.py:157`)

#### Detector 13: ReasoningActionMismatchDetector (CAFT 6.4) -- DISABLED
- **File**: `detectors.py:660-740`
- **Trigger**: Reasoning event contains read/test keywords, next tool_call is a write operation.
- **Confidence**: 0.70 (read mismatch) or 0.65 (test mismatch). Fixed values. (`detectors.py:704, 726`)
- **Status**: Disabled (0 TP, 0 FP in ground truth; keyword matching too brittle). (`registry.py:158`)


## 4. Confirmation Routing Logic

### 4.1 Decision Flow

```
Candidate arrives from detector
         |
         v
  conf >= AUTOCONFIRM_THRESHOLD (0.9)?  ----YES----> Auto-confirmed
         |                                            (no LLM call)
         NO
         |
         v
  force_llm_review == True?  ----YES----> Send to LLM regardless
         |                                 of confidence
         NO
         |
         v
  conf < 0.9 (normal case)
         |
         v
  is_llm_available()?  ----NO----> Pass as "uncertain"
         |                          (original confidence * 0.7)
         YES
         |
         v
  Call LLM (confirm_diagnosis_sync)
         |
         v
  Parse JSON response
         |
         +-- confirmed=true  --> status="confirmed", conf=max(original, LLM)
         +-- confirmed=false, conf < 0.3 --> status="rejected", dropped
         +-- confirmed=false, conf >= 0.3 --> status="uncertain", conf=LLM value
```

Source: `monitor.py:284-373`, `confirm.py:927-977`

### 4.2 Auto-Confirm Threshold

- **Default**: 0.9. Configurable via `MonitorEngine.__init__(autoconfirm_threshold=)` (`monitor.py:153-164`) or env var `AGENTDIAG_AUTOCONFIRM_THRESHOLD` (`confirm.py:912`).
- **In ablation mode**: The `apply_llm_confirmation()` in `run_ablation.py:452` checks both `det.confidence >= AUTOCONFIRM_THRESHOLD` AND `not det.force_llm_review`. The `force_llm_review` flag overrides auto-confirm.

### 4.3 force_llm_review Override

Two detectors set `force_llm_review = True`:
1. **StallDetector**: When `stall_count <= 2` (sparse stalls). A single slow Bash op can produce high confidence but be FP. (`detectors.py:1039, 1069`)
2. **ErrorCascadeDetector**: When `len(longest_chain) < 5` (short chains may be search refinement). (`detectors.py:1116, 1138`)

### 4.4 LLM Providers

| Provider | Env Var | Default Model | Config |
|----------|---------|---------------|--------|
| anthropic (default) | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5-20250929` | `confirm.py:789` |
| openai | `OPENAI_API_KEY` | `gpt-4o-mini` | `confirm.py:790` |
| ollama | (always available) | `llama3.2` | `confirm.py:791` |

Override model: `AGENTDIAG_LLM_MODEL` env var. (`confirm.py:793`)

### 4.5 Prompt Structure (V2 Design)

The prompt (`confirm.py:657-771`) is structured as:

1. **Role**: "You are a QA analyst reviewing an AI agent's work session."
2. **Context**: Agent goal (from first user_input), current HTA phase, user message count near onset.
3. **Failure definition**: From CAFT taxonomy + detector-specific criteria from `DETECTOR_CRITERIA` dict (`confirm.py:62-165`).
4. **Few-shot examples**: Detector-specific confirmed + rejected example pairs from train split ground truth (`confirm.py:182-493`). Available for all 13 detector types.
5. **Detector evidence**: JSON dump of `candidate.evidence`.
6. **Activity log**: 15 events before and 10 after the onset step (`confirm.py:525-547`).
7. **Session context** (for end-sensitive detectors: `premature_termination`, `goal_drift`, `error_cascade`, `recovery_failure`, `missing_verification`): Session head (first 5 events) + tail (last 12 events) (`confirm.py:693-710`).
8. **Similar past cases**: From OpenViking context store if available.
9. **Decision framework**: 3-criterion evaluation (MATCH, EXPLANATION, ENGINEER TEST) with explicit decision rules (`confirm.py:749-771`).

**Important design principle**: The prompt is calibrated toward confirmation. The detector already found a structural anomaly; the LLM's job is to find a *specific, concrete* counter-explanation, not to decide if the behavior is "normal." (`confirm.py:750-768`)

### 4.6 Graceful Degradation

If the LLM call fails for any reason, the system returns `status="uncertain"` with `confidence = original * 0.7` and never crashes. (`confirm.py:965-977`)

### 4.7 Response Parsing

LLM response is expected as JSON: `{"confirmed": bool, "confidence": float, "reasoning": str}`. Parser handles markdown code fences, extra text, and missing fields. (`confirm.py:865-901`)

Status derivation: `confirmed=true -> "confirmed"`, `confirmed=false AND confidence < 0.3 -> "rejected"`, else `"uncertain"`. (`confirm.py:889-894`)


## 5. Post-Processing Pipeline

### 5.1 Retraction (`_RETRACTABLE`)

- **Set**: `{"recovery_failure"}` (`run_ablation.py:296`)
- **Mechanism**: After all events are pushed, the detector is re-run on the full trace. If it returns `None` (progress filters now pass with complete data), the diagnosis is retracted by removing it from `engine._diagnoses`. (`run_ablation.py:306-309, 319-323`)
- **Rationale**: RecoveryFailureDetector's progress filters (post-recovery progress, productive session filter) need the full trace to accurately assess whether recovery succeeded.

### 5.2 Reconfirmation (`_RECONFIRMABLE`)

- **Set**: `{"stall"}` (`run_ablation.py:297`)
- **Mechanism**: After all events are pushed, the detector is re-run on the full trace. If the new diagnosis has higher confidence, the original diagnosis's confidence is upgraded and `force_llm_review` is updated. (`run_ablation.py:310-318`)
- **Rationale**: Early incremental firing computes IQR on few events, producing lower confidence. Full-trace IQR is more stable, often yielding confidence >= 0.9 (auto-confirm threshold). Also, early sparse stalls set `force_llm_review=True`, but full-trace may show enough stall events to clear it.
- **Key**: Never retracts. A stall detected at step 10 is real even if the agent recovered by step 200.

### 5.3 Session Deduplication

- **Problem**: Short-prefix session IDs (e.g., `"e2eff792"`) and their full UUID counterparts (`"e2eff792-fb22-4f3c-..."`) refer to the same underlying session. Annotations use full UUIDs; short prefixes from trace discovery count as separate sessions, inflating FP counts.
- **Mechanism**: Group all session IDs by their 8-character prefix, keep only the longest (canonical) ID from each group. Remap all annotations to use canonical IDs. Deduplicate remapped annotations by `(canonical_id, failure_name)`. (`run_ablation.py:613-642`)
- **Impact**: In V6 ablation, eliminated 11 duplicate entries (26 -> 15 unique sessions).


## 6. Decision Bottleneck Analysis / Vulnerabilities

### 6.1 False Positive Entry Points

| Source | Detectors | Mechanism | Severity |
|--------|-----------|-----------|----------|
| **Normal exploration classified as repetition** | step_repetition | Agent reading multiple similar files triggers identical (tool, input_hash). Output diversity check mitigates but requires output_hash. | Low (threshold=9 is high) |
| **Post-continuation re-reads** | context_loss | Agent legitimately re-reads after context continuation. Continuation keyword detection (`_CONTINUATION_KEYWORDS`) may miss novel phrasing. | Medium |
| **Productive sessions with late errors** | recovery_failure | Long session with 80%+ success hits a few errors at the end. Productive session filter (50 events, 80% success) mitigates but requires 0 stuck retries to suppress. | Medium |
| **Bash commands with high latency** | stall | Legitimate long-running commands (compilation, test suites) produce latency outliers. MIN_STALL_MS=3000 and IQR help, but a single 25s compilation in a session of 100ms events scores high. force_llm_review for sparse stalls mitigates. | **High** (4 FP in V6 ablation) |
| **Search refinement as error cascade** | error_cascade | 3+ grep/search failures before finding the right pattern. force_llm_review for chains <5 mitigates. | Medium |
| **Lifecycle tool evolution as goal drift** | goal_drift | Agent transitions from Read/Grep (GATHERING) to Edit/Bash (EXECUTING). Block-novelty metric picks this up as >50% novel tools. Disabled for this reason. | High (why disabled) |
| **Bash-as-test not recognized** | missing_verification, premature_termination | Claude Code runs tests via Bash (classified as EXECUTING, not VERIFYING by HTA). Multi-pattern check in MissingVerification helps but HTA still misclassifies. | Medium |

### 6.2 True Positive Drop Points

| Drop Point | File:Line | Mechanism | Impact |
|------------|-----------|-----------|--------|
| **`seen` set deduplication** | `detectors.py:1483-1487` | Once a detector fires for a session, it can NEVER fire again, even if a new instance of the same failure occurs later. | **Critical**: If stall fires early with low confidence and is rejected by LLM, a genuine stall later in the session is invisible. |
| **LLM rejection of real failures** | `monitor.py:360-368` | LLM returns `confirmed=false` for genuine failures. Prompt calibration toward confirmation mitigates, but the LLM can still reject. | Medium (2 stall TPs dropped in V6 due to LLM API failure) |
| **LLM unavailable -> uncertain -> dropped** | `monitor.py:316-323`, `confirm.py:965-977` | In MonitorEngine, uncertain candidates ARE kept (returned with reduced confidence). But in ablation `apply_llm_confirmation`, candidates where LLM is unavailable are simply skipped (`run_ablation.py:477-478`). | **High when API key expires** |
| **Retraction over-aggressiveness** | `run_ablation.py:306-309` | RecoveryFailureDetector's progress filters may retract a genuine recovery failure if the agent coincidentally makes 5 unique successful ops afterward (even if unrelated to the error). | Low |
| **Auto-confirm threshold gate** | `monitor.py:305-313` | Detectors with max confidence < 0.9 (tool_misuse: 0.85, analysis_paralysis: 0.85, goal_drift: 0.85, tool_thrashing: 0.85) can NEVER auto-confirm. They always require LLM. If LLM is unavailable, they pass as uncertain (kept in MonitorEngine but dropped in ablation). | Medium |
| **Confidence floor gates** | Various | StallDetector (conf < 0.3 -> None at line 1041), ErrorCascade (conf < 0.3 -> None at line 1111) silently drop marginal signals. | Low |

### 6.3 Most Sensitive Thresholds

| Threshold | Current Value | Location | Sensitivity |
|-----------|---------------|----------|-------------|
| **AUTOCONFIRM_THRESHOLD** | 0.9 | `confirm.py:912` | **Very High**. Determines which candidates bypass LLM. If raised to 0.95, most stall TPs (which reach 0.9-1.0 via reconfirmation) would go to LLM. If lowered to 0.8, tool_misuse and analysis_paralysis could auto-confirm. |
| **StallDetector.IQR_MULTIPLIER** | 1.5 | `detectors.py:985` | **High**. Standard statistical outlier threshold. Lowering to 1.0 would detect more stalls but increase FP. |
| **StallDetector.MIN_STALL_FRACTION** | 0.05 | `detectors.py:989` | **High**. At 5%, a session needs 1 stall per 20 active events. Raising to 0.10 would eliminate sparse-stall FPs but miss some TPs. |
| **ErrorCascade.MIN_CHAIN** | 3 | `detectors.py:1082` | **Medium**. 2 would catch more cascades but also catch normal search refinement (2 failures then success). |
| **RecoveryFailure.failure_rate threshold** | 0.4 | `detectors.py:1343` | **Medium**. Sessions with 2 errors, 1 recovery failure = 50% rate. Lowering captures more; raising requires more evidence. |
| **StepRepetition.THRESHOLD** | 9 | `detectors.py:162` | **Low** (well-tuned). 6 produced FP on trace 4; 9 is stable. |

### 6.4 Edge Cases and System Breakpoints

1. **The `seen` set is permanent and per-detector-name, not per-instance**: If a session has two distinct stall episodes (e.g., steps 10-20 and steps 80-90), only the first is ever detected. The second is invisible because `"stall"` is already in the `seen` set. This is the single most significant architectural limitation. (`detectors.py:1483-1487`)

2. **Timestamp synthesis from cumulative latency**: When events lack timestamps, `_synthesize_timestamps()` reconstructs them from cumulative `latency_ms`. This means inter-event gaps are entirely determined by `latency_ms`, which may not reflect actual wall-clock time. (`detectors.py:40-50`)

3. **output_hash as proxy**: When `output_hash` is missing, `_compute_output_hashes()` synthesizes it from `(tool, tokens_out, success)`. Two different read operations returning different content but with the same token count will get the same hash, masking context_loss detection and biasing tool_misuse's stagnation metric. (`detectors.py:53-59`)

4. **HTA classification by tool name substring**: `any(t in tool_lower for t in _GATHERING_TOOLS)` does substring matching. A tool named `"readability_check"` would match both "read" (GATHERING) and "check" (VERIFYING). Priority ordering (DELIVERING checked first) prevents some conflicts, but `"bash"` matching "bash" in EXECUTING means all bash commands are always EXECUTING, even `bash pytest tests/`. (`hta.py:97-107`)

5. **MonitorEngine vs ablation confirmation paths diverge**: In `MonitorEngine._apply_confirmation()`, uncertain results ARE returned as diagnoses (with reduced confidence). In `run_ablation.py:apply_llm_confirmation()`, candidates where LLM is unavailable are simply `continue`d (dropped). This means the same trace produces different results depending on which path runs it. (`monitor.py:316-323` vs `run_ablation.py:477-478`)

6. **Hysteresis makes GATHERING a "sticky" phase**: Because GATHERING tools have no strong signals, transitioning OUT of GATHERING requires 2 consecutive non-GATHERING events. An interleaved Read-Write-Read-Write pattern would never leave GATHERING (each Write resets the pending count). The strong signal set for EXECUTING (`{"write", "edit", "bash", ...}`) was added to fix this, but any new tool not in the strong set would be subject to the same problem. (`hta.py:73-79, 261-279`)

7. **Race condition in incremental detection**: Detectors receive the full `events` list on every push. The stall detector computes IQR on all events so far. With 10 events, IQR is unstable. With 100 events, it stabilizes. Early firing on unstable statistics is the fundamental reason `_RECONFIRMABLE` exists. Any new detector using statistical methods (percentiles, regression) will face the same problem. (`detectors.py:996-1010`)

8. **ContextLoss O(n^3) worst case**: The detector iterates all pairs of re-read steps (`O(k^2)` where k = number of reads of the same resource) and for each pair scans all events for user_input and intervening operations. For a session that reads the same file 20 times, this is `O(20^2 * n)`. Not a problem in practice (typical k < 5) but could become one for sessions with heavy file polling. (`detectors.py:308-335`)


## 7. Improvement Recommendations

### 7.1 Replace `seen` Set with Per-Detector Cooldown Window

**Impact**: High (fixes the single biggest architectural limitation)
**Effort**: Medium
**Location**: `detectors.py:1482-1488`

The current `seen` set permanently blocks a detector after first fire. Replace with a cooldown mechanism:

```python
# Current (detectors.py:1483-1487):
if det.name in seen:
    continue
# ...
seen.add(det.name)

# Proposed: cooldown window (e.g., 50 events)
if det.name in seen and (len(events) - seen[det.name]) < COOLDOWN:
    continue
# ...
seen[det.name] = len(events)  # record step, not just presence
```

This would allow a second stall to be detected if it occurs 50+ events after the first, while still preventing the same stall from firing on every subsequent push. Change `seen` from `set[str]` to `dict[str, int]` (mapping detector name to the event index when it last fired).

**Files to modify**: `detectors.py:1459-1490` (run_caft_detectors), `monitor.py:169` (_seen_failures type), `monitor.py:504` (reset).

### 7.2 Fix HTA Bash-as-Test Misclassification

**Impact**: Medium (directly reduces FP for missing_verification and premature_termination)
**Effort**: Low
**Location**: `hta.py:59-63, 82-117`

Currently, `"bash"` is in `_STRONG_EXECUTING`, so `Bash pytest tests/` is classified as EXECUTING with an immediate strong-signal transition. Add a content-aware check: if the bash command's `goal_text` matches test patterns, classify as VERIFYING.

```python
# In classify_event (hta.py:82), before the EXECUTING check:
if event.tool and "bash" in event.tool.lower() and event.goal_text:
    if _TEST_CMD_PATTERN.search(event.goal_text):  # reuse from MissingVerificationDetector
        return Phase.VERIFYING, True  # strong signal
```

This would fix a class of FP where the agent runs tests via Bash but HTA never enters VERIFYING. The `MissingVerificationDetector` already has the `_TEST_CMD_PATTERN` regex; factor it into a shared location.

**Files to modify**: `hta.py:82-117` (classify_event).

### 7.3 Unify MonitorEngine and Ablation Confirmation Paths

**Impact**: Medium (eliminates a source of evaluation/production divergence)
**Effort**: Low
**Location**: `run_ablation.py:477-478`, `monitor.py:316-323`

The ablation path drops candidates when LLM is unavailable (`continue`), while MonitorEngine keeps them as uncertain. This means ablation results undercount detections when the API key is expired/missing. Align the behavior:

```python
# run_ablation.py:477 - currently:
if not is_llm_available():
    continue

# Proposed: keep as uncertain (matching MonitorEngine behavior)
if not is_llm_available():
    if det.confidence >= 0.5:  # only keep reasonable candidates
        confirmed_dets.append(Detection(
            trace_id=det.trace_id,
            failure_name=det.failure_name,
            ...
            confirmed=True,  # treat uncertain as confirmed with discounted confidence
            confidence=det.confidence * 0.7,
        ))
    continue
```

**Files to modify**: `run_ablation.py:476-478`.

### 7.4 Add Stall Detector Context-Aware Filtering

**Impact**: High (addresses 4 FP in V6 -- the largest remaining FP source)
**Effort**: Medium
**Location**: `detectors.py:971-1070`

The stall detector's 4 FP in V6 come from legitimate long-running operations (compilation, large test suites). Add a whitelist of "expected slow" contexts:

```python
# After computing stall_indices (detectors.py:1014), filter out expected slow ops:
_EXPECTED_SLOW_PATTERNS = {"pytest", "test", "build", "compile", "install", "npm", "pip"}
stall_indices = [
    i for i in stall_indices
    if not (events[i].tool and
            any(p in events[i].tool.lower() for p in _EXPECTED_SLOW_PATTERNS))
    and not (events[i].goal_text and
             any(p in events[i].goal_text.lower() for p in _EXPECTED_SLOW_PATTERNS))
]
```

This is less principled than the IQR approach (it's a tool-name heuristic), so it should be optional (e.g., `--stall-filter-known-slow`).

**Files to modify**: `detectors.py:1012-1016`.

### 7.5 Add Incremental Confidence Tracking for Early-Warning Dashboard

**Impact**: Medium (improves real-time monitoring UX, no ablation impact)
**Effort**: Medium
**Location**: `monitor.py:375-452`

Currently, detectors fire once (see `seen` set) and produce a single confidence value. For the real-time dashboard, track *rising* confidence across pushes for detectors that haven't yet fired. This provides early warning:

```python
# In MonitorEngine, add:
self._rising_signals: dict[str, float] = {}  # detector_name -> current sub-threshold confidence

# In push(), before the seen-set check:
for det in self._detectors:
    if det.name in self._seen_failures:
        continue
    # Peek at confidence without committing
    diag = det.check(events, hta_state)
    if diag and diag.confidence < FIRE_THRESHOLD:
        self._rising_signals[det.name] = diag.confidence
```

This lets the dashboard show "stall confidence rising: 0.4 -> 0.6 -> 0.8" before it fires, giving operators time to intervene. The cost is running detectors that don't fire, but most are O(n) in event count.

**Files to modify**: `monitor.py:166-175` (new state), `monitor.py:414-420` (detection loop), `monitor.py:52-87` (DashboardState).

---

## Appendix: File Reference Index

| File | Absolute Path | Key Lines |
|------|--------------|-----------|
| MonitorEngine | `/Users/samkoscelny/GazeVLM-local/agentdiag/agentdiag/monitor.py` | push():375, _apply_confirmation():284, _resolve_detectors():187, DashboardState:52 |
| HTA State Machine | `/Users/samkoscelny/GazeVLM-local/agentdiag/agentdiag/hta.py` | classify_event():82, HTAStateMachine.push():240, HYSTERESIS:209, strong signals:77-79 |
| CAFT Detectors | `/Users/samkoscelny/GazeVLM-local/agentdiag/agentdiag/caft/detectors.py` | run_caft_detectors():1459, ALL_CAFT_DETECTORS:1423, ALL_CAFT_DETECTORS_FULL:1440, seen dedup:1483-1487 |
| Confirmation Layer | `/Users/samkoscelny/GazeVLM-local/agentdiag/agentdiag/caft/confirm.py` | build_confirmation_prompt():657, confirm_diagnosis():927, AUTOCONFIRM_THRESHOLD:912, DETECTOR_CRITERIA:62, FEW_SHOT_EXAMPLES:182 |
| Base Classes | `/Users/samkoscelny/GazeVLM-local/agentdiag/agentdiag/caft/base.py` | CaftDiagnosis:27, CaftDetector protocol:46, force_llm_review:38 |
| Data Models | `/Users/samkoscelny/GazeVLM-local/agentdiag/agentdiag/models.py` | TraceEvent:11, TraceFeatures:34, Diagnosis:50 |
| Detector Registry | `/Users/samkoscelny/GazeVLM-local/agentdiag/agentdiag/caft/registry.py` | _build_default_registry():141, disabled set:155-158, detector_registry singleton:170 |
| Ablation Runner | `/Users/samkoscelny/GazeVLM-local/agentdiag/scripts/run_ablation.py` | _RETRACTABLE:296, _RECONFIRMABLE:297, session dedup:613-642, apply_llm_confirmation():414 |
