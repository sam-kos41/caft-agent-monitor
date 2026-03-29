# CAFT: Cognitive Agent Failure Taxonomy

## Mapping Human Error Science onto AI Agent Behavior

**Version 1.0 — March 2026**

---

## Abstract

AI agents fail in ways that are structurally identical to how humans fail. The human factors field has developed rigorous, empirically-validated taxonomies for classifying human error over five decades of research in aviation, nuclear power, medicine, and military operations. This paper applies five established human error frameworks — Reason's GEMS, Rasmussen's SRK, Hierarchical Task Analysis, the Swiss Cheese Model, and HFACS — to the domain of AI agent behavior. We produce complete mapping tables from each framework, then synthesize them into a unified **Cognitive Agent Failure Taxonomy (CAFT)** that is hierarchical, evidence-based, actionable, and exhaustive. We validate coverage against the MAST taxonomy's 14 empirically-derived failure categories (Cemri et al., 2025) and the Microsoft Agentic AI Failure Modes whitepaper (2025). Each failure type in CAFT includes observable trace indicators and detection methods, making the taxonomy directly implementable in diagnostic tooling.

---

## 1. Introduction

### 1.1 The Structural Isomorphism Thesis

Human factors research classifies errors not by their *domain* but by their *cognitive mechanism*. A pilot who retracks a heading they already corrected and a surgeon who re-incises a site they already closed commit structurally identical errors — both are *capture errors* at the skill-based level of processing (Reason, 1990). The claim of this paper is that the same structural isomorphism holds between human operators and AI agents:

| Human Operator Property | AI Agent Equivalent |
|---|---|
| Working memory (limited, decays) | Context window (limited, truncated) |
| Long-term memory (schemas, scripts) | Training data, in-context examples, system prompts |
| Attention (selective, limited) | Token attention mechanism (finite, position-biased) |
| Skill automaticity | Learned tool-use patterns, boilerplate generation |
| Rule application | If-then heuristics from training (e.g., "if test fails, check imports") |
| Mental model construction | Reasoning about unfamiliar architecture, API inference |
| Fatigue / cognitive load | Context window saturation, degraded late-context attention |
| Supervisory control | System prompts, guardrails, tool permission systems |

Because the cognitive architecture maps, the *failure taxonomy* maps. An agent that repeats a tool call it already executed is committing a lapse analogous to a pharmacist who fills the same prescription twice. An agent that applies Python idioms in a JavaScript file is committing a rule-based mistake analogous to a driver who turns on windshield wipers when intending to signal (a *mode error*). These are not metaphors — they are instances of the same error-production mechanisms operating on different substrates.

### 1.2 Why This Matters

Existing AI agent failure taxonomies (MAST, Microsoft's Agentic AI taxonomy) describe *what* goes wrong but not *why* at the cognitive-mechanism level. Human factors taxonomies explain *why* and predict *when*. Combining them yields a taxonomy that is simultaneously:

1. **Descriptive** — covers the full space of observed failures
2. **Explanatory** — identifies the generating mechanism
3. **Predictive** — indicates preconditions that make failures likely
4. **Prescriptive** — points to specific interventions

### 1.3 Scope

This taxonomy addresses failures in *agentic* AI systems — systems that take sequences of actions (tool calls, code execution, API requests) to accomplish goals, with multi-step reasoning and state management. It covers single-agent and multi-agent architectures. It does not address failures in pure generation (hallucination in single-turn completions) except where those failures occur within an agentic execution trace.

---

## 2. Framework 1 — Reason's Generic Error Modeling System (GEMS)

GEMS (Reason, 1990) classifies errors by the cognitive level at which they occur and the relationship between intention and outcome. It distinguishes errors (unintended outcomes) from violations (deliberate deviations), and within errors, separates execution failures (slips/lapses) from planning failures (mistakes).

### 2.1 Complete Mapping Table

| GEMS Category | Human Error | Agent Equivalent | Trace Example |
|---|---|---|---|
| **SLIP: Action-not-as-planned** | | | |
| Capture error | Habitual action overrides intended action | Agent calls frequently-used tool instead of correct tool | `tool_call: search_docs` when task requires `write_file` after a long sequence of search calls |
| Description error | Right action on wrong object | Agent edits correct function but wrong file | `write_file(path="/src/old_main.py")` instead of `/src/main.py` |
| Mode error | Action appropriate to one mode but not current mode | Agent uses Python syntax while editing JavaScript | Tool call with wrong language conventions for current file type |
| Perceptual confusion | Misidentifying a stimulus | Agent confuses similarly-named variables, functions, or files | References `getUserId` when the actual function is `getUserID` |
| Associative activation | Thought triggered by co-occurring associations | Agent suggests import of package commonly co-occurring with current one but irrelevant | Adds `import pandas` to a file that already uses a different data library |
| **LAPSE: Memory failure** | | | |
| Omission following interruption | Forget to resume task after interruption | Agent loses track of multi-step plan after a tool error | Was modifying files A, B, C; after error in B, only continues to C, forgetting to retry B |
| Repetition | Perform already-completed action again | Agent re-executes a tool call it already completed | `read_file(main.py)` called twice in sequence with identical output |
| Loss of place | Forget position in task sequence | Agent restarts a procedure from the beginning | Begins reinstalling dependencies already installed 10 steps earlier |
| Loss of intention | Forget original goal mid-execution | Agent drifts from debugging task to refactoring unrelated code | Goal shifts from "fix login bug" to "clean up CSS" mid-trace |
| Prospective memory failure | Forget to do something in the future | Agent plans to run tests after changes but never does | Explicit "I'll run tests after this change" followed by no test execution |
| **RULE-BASED MISTAKE: Wrong rule applied** | | | |
| Misapplication of good rule | Correct rule in wrong context | Agent applies Python pattern in JavaScript context | Uses `list.append()` syntax when working in JS (should be `array.push()`) |
| Application of bad rule | Flawed rule applied correctly | Agent follows outdated API documentation | Calls deprecated endpoint consistently, following an incorrect but internally consistent pattern |
| Encoding deficiency | Relevant feature not attended to | Agent ignores error message content, retries blindly | Error says "auth token expired" but agent retries with same token |
| Action deficiency | Right diagnosis, wrong action | Agent correctly identifies bug but applies wrong fix | Identifies null pointer but adds try/catch instead of null check |
| **KNOWLEDGE-BASED MISTAKE: Incomplete mental model** | | | |
| Selectivity | Incomplete situation assessment | Agent makes architectural decision based on partial codebase scan | Recommends microservice split without discovering the monolith's internal event bus |
| Workspace limitation | Too many factors for working memory | Agent loses coherence in long traces as context fills | Quality of reasoning degrades monotonically with trace length |
| Confirmation bias | Seeks evidence for current hypothesis only | Agent fixates on one error cause, ignores contradicting evidence | Keeps checking imports after first failed import, ignoring runtime error in output |
| Overconfidence | Unjustified certainty in incorrect model | Agent states incorrect API behavior with high confidence | "This endpoint returns JSON" when it actually returns XML |
| Incomplete mental model | Fundamental misunderstanding of system | Agent misunderstands async/sync architecture | Treats async function as synchronous, doesn't await promises |

### 2.2 Observable Trace Indicators

| GEMS Category | Observable in Trace |
|---|---|
| Slip (capture) | High-frequency tool appears where low-frequency tool is contextually correct; tool selection inversely correlated with recency-frequency |
| Slip (mode) | Tool parameters inconsistent with current file type / language / API version |
| Lapse (repetition) | Identical `(tool, parameters)` tuple appearing within N steps with identical output hash |
| Lapse (loss of place) | Task subsequence repeated; same sub-goal addressed twice |
| Rule-based mistake | Consistent application of wrong pattern across multiple steps (not random — structured incorrectly) |
| Knowledge-based mistake | Increasing error rate over time; multiple failed hypotheses tested in sequence; no convergence |

### 2.3 Detection Methods

| GEMS Category | Detection | Method |
|---|---|---|
| Capture error | Rule-based | Tool-call frequency analysis; detect when most-frequent tool appears in context where it has low relevance |
| Mode error | Rule-based | Cross-reference tool parameters against current file type / language metadata |
| Repetition | Rule-based | Exact or near-exact match on `(tool, params, output_hash)` within sliding window |
| Loss of intention | ML-based | Embed goal_text and recent tool descriptions; measure cosine drift |
| Rule misapplication | ML-based | Pattern matching against known anti-patterns per language/framework |
| Knowledge-based | ML-based | Reasoning chain analysis; detect non-converging hypothesis testing |

### 2.4 Remediation

| GEMS Category | Remediation |
|---|---|
| Slips | Add tool-call validation: "Is this tool appropriate for the current file type / context?" |
| Lapses | Implement external memory: checklist of completed sub-goals, deduplication of tool calls |
| Rule-based mistakes | Provide context-specific rules: inject language/framework documentation into context |
| Knowledge-based mistakes | Escalate to human; add architecture documentation to context; use RAG for codebase understanding |

---

## 3. Framework 2 — Rasmussen's SRK (Skill-Rule-Knowledge)

Rasmussen's (1983) three-level model describes performance on a continuum from automatic (skill-based) to conscious analytical (knowledge-based). Each level has characteristic error types and different detection/remediation profiles.

### 3.1 Complete Mapping Table

| SRK Level | Human Behavior | Agent Behavior | Error Types | Agent Error Example |
|---|---|---|---|---|
| **Skill-Based** (automatic, routine) | Driving a familiar route; touch-typing | Routine tool calls; boilerplate generation; file I/O | Slips: capture, mode, description | Agent auto-completes `git commit` when user asked for `git stash`; agent generates Python boilerplate in a TypeScript file |
| | | | Lapses: omission, repetition | Agent skips `git add` before `git commit`; re-reads file already in context |
| **Rule-Based** (if-then patterns) | "If oil light is on, check oil level" | "If tests fail, check imports"; "If 403, refresh auth" | Wrong rule: similar cue, wrong rule fires | Agent sees `ModuleNotFoundError`, installs package (correct for missing package) but the actual issue is a circular import |
| | | | Good rule, misapplied | Agent applies `try/except` pattern (generally good) around code that should fail loudly for debugging |
| | | | Encoding deficiency: missed cue | Agent acts on first error message, ignoring that it's a warning not an error |
| **Knowledge-Based** (novel problem-solving) | Diagnosing unfamiliar engine noise; debugging novel system | Reasoning about unfamiliar architecture; debugging novel errors; designing new features | Incomplete model | Agent assumes REST API when the system uses GraphQL; misunderstands data flow |
| | | | Confirmation bias | Agent hypothesizes "database connection issue," only queries DB-related logs, misses the actual network timeout |
| | | | Fixation | Agent spends 20 steps debugging function A when the bug is in function B that calls A |

### 3.2 Observable Trace Indicators

| SRK Level | Trace Signature |
|---|---|
| Skill-based errors | Low latency, high confidence, wrong tool; error occurs in midst of otherwise fluent execution; typically single-step errors |
| Rule-based errors | Medium latency; agent explicitly references a pattern ("Let me check the imports since..."); consistent but wrong approach applied across multiple steps |
| Knowledge-based errors | High latency; multiple reasoning steps; hypothesis-test-revise cycle visible; increasing token usage as agent explores; non-converging error rate |

### 3.3 Detection Methods

| SRK Level | Detection | Method |
|---|---|---|
| Skill-based | Rule-based | Tool-call validation against current context (file type, operation type); sequence deduplication |
| Rule-based | Rule-based + ML | Pattern matching for known anti-patterns; detect consistent-but-wrong tool sequences; compare against known-good sequences for similar tasks |
| Knowledge-based | ML-based | Reasoning chain analysis; measure hypothesis diversity (low = fixation); track convergence rate of error-fix cycle |

### 3.4 Remediation by SRK Level

| SRK Level | Remediation Strategy |
|---|---|
| Skill-based | **Forcing functions**: tool-call validators, pre-execution checks, mode indicators. Low overhead since these are routine operations. |
| Rule-based | **Better rules**: inject domain-specific heuristics, framework documentation, codebase conventions. The agent has a rule — give it a better one. |
| Knowledge-based | **Expand the model**: provide architecture docs, use RAG for codebase search, escalate to human for novel situations. Accept higher latency and cost for genuine novelty. |

---

## 4. Framework 3 — Hierarchical Task Analysis (HTA)

HTA (Annett & Duncan, 1967; Stanton, 2006) decomposes tasks into a hierarchy: Goal → Sub-goals → Operations, connected by Plans that specify sequencing. Failures can occur at each level of the hierarchy.

### 4.1 Agent Task Decomposition Model

```
Level 0: Overall Goal
  "Fix the authentication bug in the login flow"

Level 1: Sub-goals (Plan: do 1, then 2, then 3, then 4)
  1. Understand the bug
  2. Locate the faulty code
  3. Implement the fix
  4. Verify the fix

Level 2: Sub-sub-goals
  1.1 Read the error report
  1.2 Reproduce the error
  2.1 Search codebase for auth-related files
  2.2 Read candidate files
  2.3 Identify the faulty function
  3.1 Edit the faulty function
  3.2 Update related tests
  4.1 Run tests
  4.2 Verify no regressions

Level 3: Operations (atomic tool calls)
  1.1.1 read_file("error_log.txt")
  2.1.1 search_codebase("auth login")
  3.1.1 edit_file("auth.py", line=42, ...)
  4.1.1 run_tests("test_auth.py")
```

### 4.2 Failure Mapping by HTA Level

| HTA Level | Failure Type | Description | Agent Example | Trace Indicator |
|---|---|---|---|---|
| **Level 0: Goal** | Goal misinterpretation | Agent pursues wrong overall objective | User asks "fix the auth bug" → agent refactors entire auth module | Tool calls diverge from stated goal; high tool diversity unrelated to goal keywords |
| | Goal substitution | Agent replaces hard goal with easier one | Asked to fix a race condition → agent adds error handling to mask it | Bug symptoms disappear but root cause untouched |
| **Level 1: Sub-goals** | Wrong decomposition | Agent omits a required sub-goal | Skips "verify the fix" sub-goal entirely | No test execution after code changes |
| | Wrong ordering | Agent executes sub-goals out of sequence | Attempts to fix code before understanding the bug | `edit_file` appears before any `read_file` or `search` |
| | Unnecessary sub-goals | Agent adds sub-goals not required by the task | Installs linting tools, reformats code, upgrades packages while fixing a single bug | High tool-call count with many tools unrelated to primary goal |
| **Level 2: Plans** | Missing precondition | Agent attempts operation without required prior step | Edits a file without first reading it to understand current state | `edit_file` with no preceding `read_file` for that path |
| | Wrong sequencing | Operations within a plan executed out of order | Runs tests before saving changes | `run_tests` precedes `write_file` in the same sub-goal |
| | Missing exit condition | Agent continues executing a plan past its termination point | Keeps searching after finding the target file | Repeated `search` calls after a search already returned the target |
| **Level 3: Operations** | Wrong operation | Incorrect atomic action | Uses `delete_file` instead of `edit_file` | Destructive tool call where non-destructive was appropriate |
| | Missing operation | Required atomic action omitted | Edits function but doesn't save | Operation sequence missing a required step |
| | Repeated operation | Same atomic action executed twice | Reads the same file twice consecutively | Identical `(tool, params)` in adjacent steps |

### 4.3 Detection Methods

| HTA Level | Detection | Method |
|---|---|---|
| Goal-level | ML-based | Embed user goal and trace summary; measure semantic alignment score over time |
| Sub-goal level | Rule-based | Check for known required sub-goals (e.g., verify after modify); detect missing sub-goal patterns |
| Plan level | Rule-based | Precondition checking (read before edit, save before test); sequence validation |
| Operation level | Rule-based | Tool-call deduplication; parameter validation; destructive-action detection |

### 4.4 Remediation

| HTA Level | Remediation |
|---|---|
| Goal | Re-inject user's original goal periodically; add goal-alignment checkpoints |
| Sub-goal | Enforce sub-goal templates for common tasks (e.g., "debug" always includes "verify") |
| Plan | Add precondition checks to tool execution; enforce plan ordering constraints |
| Operation | Tool-call validators; parameter validation; confirmation prompts for destructive operations |

---

## 5. Framework 4 — Swiss Cheese Model

Reason's (1990, 1997) Swiss Cheese Model describes accidents as the result of failures at multiple defensive layers. Each layer has "holes" (weaknesses), and an accident occurs when holes across all layers momentarily align. The model distinguishes *active failures* (immediate unsafe acts) from *latent conditions* (dormant system weaknesses that create the conditions for active failures).

### 5.1 Defense Layers for AI Agents

```
Layer 1: Organizational Defenses
  Model selection, training data, API design,
  tool ecosystem, deployment architecture

Layer 2: Supervisory Defenses
  System prompt, guardrails, tool permissions,
  rate limits, output filters

Layer 3: Preconditions
  Context window state, memory/history,
  available tools, current task complexity

Layer 4: Agent Acts
  Individual tool calls, reasoning steps,
  parameter selections
```

### 5.2 Complete Mapping Table

| Layer | Human Domain | Agent Domain | Hole (Weakness) | Example |
|---|---|---|---|---|
| **L4: Active Failures (Unsafe Acts)** | Pilot makes wrong control input | Agent calls wrong tool or wrong parameters | Immediate execution error | `delete_file` instead of `edit_file`; wrong API endpoint |
| | Pilot violates procedure | Agent bypasses safety check | Intentional deviation from instructions | Agent uses `--force` flag despite instructions to be cautious |
| **L3: Preconditions** | Pilot is fatigued | Context window is saturated | Degraded processing capacity | Agent makes more errors in steps 80-100 than steps 1-20 |
| | Pilot lacks situation awareness | Agent has incomplete context | Missing information | Agent doesn't know about a recent file rename |
| | Crew communication breakdown | Multi-agent information loss | Inter-agent misalignment | Agent A modifies file that Agent B is also modifying |
| **L2: Supervisory Defenses** | Inadequate training oversight | Poor system prompt | Insufficient guidance | System prompt doesn't specify error handling behavior |
| | Failure to correct known problem | Known tool limitation not addressed | Unmitigated known issue | Tool has known failure mode on large files; no workaround provided |
| | Inadequate supervision | No guardrails on destructive actions | Missing safety controls | Agent can delete production database without confirmation |
| **L1: Organizational Influences** | Safety culture | Model selection and training | Systemic capability limitations | Model chosen lacks tool-use training; small context window |
| | Resource management | Token budget / API rate limits | Resource constraints | Token limit forces context truncation mid-task |
| | Organizational process | Tool API design | Systemic design flaws | Tool returns ambiguous error codes; no structured error format |

### 5.3 Accident Trajectory: Holes Aligning

An example of holes aligning across all four layers:

```
L1 (Org): Model has 8K context window (organizational constraint)
  ↓ hole: insufficient capacity for complex tasks
L2 (Supervision): System prompt doesn't instruct agent to summarize context
  ↓ hole: no mitigation for context limitations
L3 (Precondition): Task requires analyzing 15 files (exceeds capacity)
  ↓ hole: context saturated, early information lost
L4 (Active): Agent edits file based on stale context, introduces bug
  ↓ ACCIDENT: incorrect code change deployed
```

**Removing any single hole prevents the accident.** This is the power of the Swiss Cheese Model — it shows that agent failures are rarely caused by a single factor, and defense-in-depth is the correct strategy.

### 5.4 Detection: Identifying Holes

| Layer | Detection Method | Trace Indicators |
|---|---|---|
| L4: Active failures | Rule-based | Wrong tool, failed execution, parameter errors (same as GEMS slip detection) |
| L3: Preconditions | Rule-based | Context utilization >80%, token_in growing faster than token_out, information-retrieval steps increasing |
| L2: Supervisory | Audit-based | Static analysis of system prompt for missing guardrails; check tool permissions configuration |
| L1: Organizational | Audit-based | Model capability benchmarks; tool API quality metrics; token budget adequacy analysis |

### 5.5 Remediation: Defense-in-Depth

| Layer | Remediation |
|---|---|
| L4 | Tool-call validation, pre-execution checks, confirmation for destructive actions |
| L3 | Context management (summarization, RAG), memory systems, state checkpointing |
| L2 | System prompt engineering, guardrail frameworks, tool permission systems |
| L1 | Model selection for task complexity, adequate token budgets, well-designed tool APIs |

---

## 6. Framework 5 — HFACS (Human Factors Analysis and Classification System)

HFACS (Shappell & Wiegmann, 2000) operationalizes the Swiss Cheese Model into a systematic classification system with specific, codified categories at each layer. It has been validated across aviation (>14,000 accidents), healthcare, mining, and military operations. We apply the full HFACS hierarchy to AI agents.

### 6.1 Complete Mapping Table

#### Level 1: Unsafe Acts

| HFACS Category | Human | Agent | Trace Indicator | Detection |
|---|---|---|---|---|
| **Skill-based error** | Inadvertent stick input | Wrong tool called during routine sequence | High-frequency tool appears where contextually wrong tool is correct | Rule: tool-context mismatch |
| **Decision error** | Chose wrong approach | Applied incorrect fix strategy | Multiple fix attempts with increasing error rate | Rule: non-converging fix cycle |
| **Perceptual error** | Misread instrument | Misread error message | Agent action contradicts information in preceding tool output | ML: output-action consistency |
| **Routine violation** | Habitual shortcut | Skips recommended steps | Missing expected sub-goal (e.g., no tests after code change) | Rule: sub-goal completeness check |
| **Exceptional violation** | One-time deliberate risk | Uses dangerous command | `--force`, `--no-verify`, `rm -rf`, `DROP TABLE` | Rule: dangerous command detection |

#### Level 2: Preconditions for Unsafe Acts

| HFACS Category | Human | Agent | Trace Indicator | Detection |
|---|---|---|---|---|
| **Adverse mental state: task saturation** | Information overload | Context window saturation | tokens_in growing; error rate increasing with step count | Rule: token growth + error correlation |
| **Adverse mental state: loss of awareness** | Loss of situation awareness | Loss of context from truncation | Agent contradicts its earlier statements; re-asks answered questions | ML: self-consistency analysis |
| **Adverse mental state: channelized attention** | Fixation | Agent fixates on one hypothesis | Same diagnostic tool called repeatedly despite negative results | Rule: tool repetition without state change |
| **Physical/technological environment** | Equipment malfunction | Tool failure / API timeout | Tool returns error or timeout | Rule: tool success rate monitoring |
| **Crew resource management** | Communication breakdown | Multi-agent misalignment | Agents produce contradictory outputs; duplicated work | Rule: output contradiction detection |

#### Level 3: Unsafe Supervision

| HFACS Category | Human | Agent | Design Indicator | Mitigation |
|---|---|---|---|---|
| **Inadequate supervision** | Insufficient oversight | System prompt lacks guidance for edge cases | Agent improvises in situations where explicit instructions should exist | Add edge-case handling to system prompt |
| **Planned inappropriate operation** | Assigning unqualified pilot | Assigning task beyond model capability | Task requires capabilities the model lacks (e.g., vision for text-only model) | Model-task capability matching |
| **Failure to correct known problem** | Not addressing repeated violations | Known tool failure not documented | Same tool failure recurs across traces | Tool documentation and workaround injection |
| **Supervisory violation** | Allowing unauthorized action | Guardrails too permissive | Agent performs destructive actions without confirmation | Tighten tool permission policies |

#### Level 4: Organizational Influences

| HFACS Category | Human | Agent | System Indicator | Mitigation |
|---|---|---|---|---|
| **Resource management** | Insufficient funding/equipment | Insufficient token budget / weak model | Tasks fail due to context limits; cheap model used for hard tasks | Right-size model and token budget to task complexity |
| **Organizational climate** | Safety culture | Evaluation and testing culture | No systematic testing of agent behavior; no failure analysis | Implement agent trace analysis and failure taxonomization |
| **Organizational process** | Poor procedures | Poor tool API design | Tools return ambiguous outputs; inconsistent error formats | Standardize tool response schemas; add structured errors |

### 6.2 HFACS Agent Application Process

For any observed agent failure:

1. **Classify the unsafe act** (Level 1) — What did the agent do wrong?
2. **Identify preconditions** (Level 2) — What conditions enabled the error?
3. **Assess supervision** (Level 3) — Was the system prompt/guardrails adequate?
4. **Examine organizational factors** (Level 4) — Is the model/tooling/architecture appropriate?

This forces analysis beyond the immediate error to systemic causes — the same principle that revolutionized aviation safety analysis.

---

## 7. Unified Taxonomy: CAFT (Cognitive Agent Failure Taxonomy)

### 7.1 Design Principles

The synthesis combines:
- **GEMS** → the error-mechanism dimension (slip/lapse/mistake)
- **SRK** → the processing-level dimension (automatic/rule/novel)
- **HTA** → the task-hierarchy dimension (goal/sub-goal/operation)
- **Swiss Cheese** → the defense-layer dimension (active/precondition/supervision/organizational)
- **HFACS** → the operational classification categories

The result is a two-axis taxonomy:

- **Primary axis: Error mechanism** (adapted from GEMS × SRK) — answers "*What kind of cognitive failure is this?*"
- **Secondary axis: System layer** (adapted from Swiss Cheese × HFACS) — answers "*Where in the system did the defense fail?*"

### 7.2 The CAFT Hierarchy

```
CAFT
├── 1. EXECUTION ERRORS (Slips — correct intention, wrong execution)
│   ├── 1.1 Capture Error
│   ├── 1.2 Mode Error
│   ├── 1.3 Description Error
│   └── 1.4 Perceptual Confusion
│
├── 2. MEMORY FAILURES (Lapses — information lost or degraded)
│   ├── 2.1 Context Loss
│   ├── 2.2 Repetition
│   ├── 2.3 Omission
│   └── 2.4 Goal Drift
│
├── 3. RULE APPLICATION FAILURES (Mistakes — wrong rule or wrong application)
│   ├── 3.1 Wrong Rule Selected
│   ├── 3.2 Good Rule Misapplied
│   ├── 3.3 Encoding Deficiency
│   └── 3.4 Action Deficiency
│
├── 4. KNOWLEDGE FAILURES (Mistakes — incomplete or incorrect mental model)
│   ├── 4.1 Incomplete Model
│   ├── 4.2 Confirmation Bias
│   ├── 4.3 Fixation
│   └── 4.4 Overconfidence
│
├── 5. PLAN STRUCTURE FAILURES (HTA decomposition and sequencing)
│   ├── 5.1 Goal Misinterpretation
│   ├── 5.2 Wrong Decomposition
│   ├── 5.3 Sequencing Error
│   └── 5.4 Missing Precondition
│
├── 6. COORDINATION FAILURES (Multi-agent and system interaction)
│   ├── 6.1 Information Withholding
│   ├── 6.2 Conversation Reset
│   ├── 6.3 Input Disregard
│   └── 6.4 Reasoning-Action Mismatch
│
├── 7. RESOURCE EXHAUSTION (Preconditions degrading performance)
│   ├── 7.1 Context Saturation
│   ├── 7.2 Token Explosion
│   ├── 7.3 Latency Cascade
│   └── 7.4 Attention Degradation
│
└── 8. SYSTEMIC DEFICIENCIES (Latent conditions — supervision and organizational)
    ├── 8.1 Inadequate Guidance
    ├── 8.2 Missing Guardrails
    ├── 8.3 Tool Design Flaws
    └── 8.4 Capability Mismatch
```

### 7.3 Complete CAFT Specification

---

#### Category 1: EXECUTION ERRORS

**Cognitive mechanism:** Skill-based level. Intention is correct, execution is faulty. Analogous to GEMS slips. Occurs during routine, automated behavior.

##### 1.1 Capture Error

| Attribute | Value |
|---|---|
| **Definition** | A frequently-executed action sequence "captures" control from the intended action. The most habitual response fires instead of the correct one. |
| **Human analogy** | Driving to work on a day off; typing a habitual password on a new system |
| **Agent manifestation** | Agent calls its most frequently-used tool instead of the contextually-appropriate one |
| **Trace indicators** | Tool T appears where context requires tool T'; T has highest call frequency in trace history; T' has low frequency but high contextual relevance |
| **Detection method** | Rule-based: compute tool frequency × recency score; flag when highest-frequency tool appears in context where semantic relevance score (measured against goal text or recent output) is below threshold |
| **MAST coverage** | FM-1.2 (Disobey role specification), partial FM-2.6 (Reasoning-action mismatch) |
| **Remediation** | Pre-execution tool validation: "Is this the right tool for the current sub-goal?"; reduce tool palette to context-relevant subset |

##### 1.2 Mode Error

| Attribute | Value |
|---|---|
| **Definition** | Action appropriate for one mode/context applied in the wrong mode/context |
| **Human analogy** | Turning windshield wipers instead of turn signal in unfamiliar car |
| **Agent manifestation** | Agent applies conventions from language A while editing language B; uses API v1 patterns on API v2; applies local file operations on remote resources |
| **Trace indicators** | Tool parameters contain syntax/conventions inconsistent with current file extension, API version, or runtime environment |
| **Detection method** | Rule-based: cross-reference tool call parameters against file metadata (extension, framework, API version) |
| **MAST coverage** | FM-1.1 (Disobey task specification) |
| **Remediation** | Mode indicators: inject current context metadata (language, framework, API version) into each tool call prompt |

##### 1.3 Description Error

| Attribute | Value |
|---|---|
| **Definition** | Correct action performed on wrong object due to physical/nominal similarity |
| **Human analogy** | Pouring orange juice into coffee mug instead of glass (both containers) |
| **Agent manifestation** | Agent edits the correct function but in the wrong file; modifies `auth_v2.py` instead of `auth.py`; references similarly-named variable |
| **Trace indicators** | Tool target (file path, variable name) has high string similarity to the correct target; correct target exists in context |
| **Detection method** | Rule-based: fuzzy string matching on tool parameters against recently-mentioned targets; flag edit distance < threshold |
| **MAST coverage** | FM-1.1 (Disobey task specification) |
| **Remediation** | Target confirmation: display disambiguated target before execution; prefer fully-qualified paths |

##### 1.4 Perceptual Confusion

| Attribute | Value |
|---|---|
| **Definition** | Misidentifying a stimulus due to similarity or ambiguity |
| **Human analogy** | Misreading a medication label due to similar packaging |
| **Agent manifestation** | Agent confuses error types (warning vs. error); misreads tool output format; confuses similar function signatures |
| **Trace indicators** | Agent action contradicts information in immediately preceding tool output; agent references incorrect value that is similar to correct value in output |
| **Detection method** | ML-based: compare extracted entities from tool output against agent's subsequent references to those entities |
| **MAST coverage** | FM-2.6 (Reasoning-action mismatch) |
| **Remediation** | Structured tool outputs with typed fields; explicit error severity levels; disambiguation prompts |

---

#### Category 2: MEMORY FAILURES

**Cognitive mechanism:** Skill/rule-based level. Information required for task execution is lost, degraded, or inaccessible. Analogous to GEMS lapses. In agents, "memory" is the context window, conversation history, and any external state.

##### 2.1 Context Loss

| Attribute | Value |
|---|---|
| **Definition** | Agent loses access to information that was previously in context, either through truncation, window overflow, or attention degradation |
| **Human analogy** | Forgetting what you came to the store to buy; losing your place in a long document |
| **Agent manifestation** | Agent contradicts its earlier statements; re-asks a question already answered; loses track of which files were modified |
| **Trace indicators** | Agent references information from early context incorrectly or not at all; statements at step N contradict statements at step M (M << N); re-execution of information-gathering steps |
| **Detection method** | Rule-based: detect re-execution of identical queries; ML-based: self-consistency analysis between early and late trace segments |
| **MAST coverage** | FM-1.4 (Loss of conversation history), FM-2.1 (Conversation reset) |
| **Remediation** | Context summarization; external memory/scratchpad; periodic state checkpointing; RAG for conversation history |

##### 2.2 Repetition

| Attribute | Value |
|---|---|
| **Definition** | Agent performs an action that was already successfully completed |
| **Human analogy** | Pharmacist filling same prescription twice; checking a lock you already checked |
| **Agent manifestation** | Identical tool call with identical parameters and identical output; re-reading a file already read and unchanged |
| **Trace indicators** | `(tool, params) → output_hash` tuple appears twice within N steps; no state change between occurrences |
| **Detection method** | Rule-based: exact match on (tool, normalized_params) within sliding window, with output hash comparison |
| **MAST coverage** | FM-1.3 (Step repetition) |
| **Remediation** | Tool-call deduplication layer; "already completed" cache; execution history visible to agent |

##### 2.3 Omission

| Attribute | Value |
|---|---|
| **Definition** | Agent skips a required step in a task sequence, typically after an interruption (error, context switch) |
| **Human analogy** | Forgetting to turn off the stove after answering the phone |
| **Agent manifestation** | Agent was modifying files A, B, C; after an error in B, continues to C without retrying B; planned test execution never occurs |
| **Trace indicators** | Explicitly planned step (mentioned in reasoning) never executed; required sub-goal (e.g., verification after modification) absent from trace |
| **Detection method** | Rule-based: match planned steps (extracted from reasoning events) against executed steps; check for required sub-goal patterns |
| **MAST coverage** | FM-3.1 (Premature termination), FM-3.2 (No or incomplete verification) |
| **Remediation** | Task checklists; post-error resumption prompts; required verification steps enforced by tooling |

##### 2.4 Goal Drift

| Attribute | Value |
|---|---|
| **Definition** | Agent's effective goal gradually shifts away from the stated objective, often without a single identifiable point of deviation |
| **Human analogy** | Starting to clean one room and ending up reorganizing the entire house |
| **Agent manifestation** | Agent begins debugging a login bug, drifts into refactoring the auth module, then starts "improving" the CSS |
| **Trace indicators** | Tool distribution shift between first and second half of trace (Jensen-Shannon divergence); new tool categories appearing; decreasing semantic similarity between recent actions and original goal |
| **Detection method** | Rule-based: JSD on tool distributions (current agentdiag DriftDetector); ML-based: embed goal text and tool descriptions, track cosine distance over sliding window |
| **MAST coverage** | FM-2.3 (Task derailment) |
| **Remediation** | Periodic goal re-injection; goal-alignment scoring at checkpoints; explicit "am I still on track?" self-checks |

---

#### Category 3: RULE APPLICATION FAILURES

**Cognitive mechanism:** Rule-based level. Agent applies an if-then pattern that is wrong for the current situation. The application is systematic (not random) — the agent is following a rule, just the wrong one.

##### 3.1 Wrong Rule Selected

| Attribute | Value |
|---|---|
| **Definition** | A valid rule for a different situation is applied to the current situation due to superficial similarity of cues |
| **Human analogy** | Treating a gas leak with the fire evacuation procedure (both emergencies, wrong procedure) |
| **Agent manifestation** | Agent sees `ModuleNotFoundError` and installs a package (correct rule for missing packages), but the actual cause is a circular import. Agent sees a test failure and immediately modifies source code, when the test itself was wrong. |
| **Trace indicators** | Agent applies a known-good pattern (identifiable from common sequences) that addresses a superficially similar but actually different situation; error persists or changes character after "fix" |
| **Detection method** | ML-based: classify error type from error message, compare against action taken; detect mismatch between error category and fix category |
| **MAST coverage** | FM-2.6 (Reasoning-action mismatch) |
| **Remediation** | Error classification before action; force agent to state diagnosis before applying fix; provide decision trees for common error types |

##### 3.2 Good Rule Misapplied

| Attribute | Value |
|---|---|
| **Definition** | A generally correct rule is applied outside its valid scope or boundary conditions |
| **Human analogy** | Applying CPR to someone who is breathing (CPR is good, but wrong in this context) |
| **Agent manifestation** | Agent wraps code in try/except (generally good defensive practice) but this masks a critical error that should fail loudly. Agent adds type checking (generally good) to a hot loop, causing performance regression. |
| **Trace indicators** | Pattern is recognizable as a known-good practice; applied in context where preconditions for that practice are not met |
| **Detection method** | ML-based: identify applied pattern; check whether its preconditions hold in current context |
| **MAST coverage** | FM-1.1 (Disobey task specification) |
| **Remediation** | Context-conditional rule application; specify preconditions for common patterns in tool documentation |

##### 3.3 Encoding Deficiency

| Attribute | Value |
|---|---|
| **Definition** | Agent fails to attend to a critical feature of the situation that would indicate the current rule is wrong |
| **Human analogy** | Treating a patient's symptoms without checking medication interactions |
| **Agent manifestation** | Error message says "auth token expired" but agent retries with the same token. Error log contains both a warning and an error, but agent only addresses the warning. |
| **Trace indicators** | Critical information present in tool output that is not reflected in subsequent agent action; agent acts on partial information from a multi-part response |
| **Detection method** | ML-based: extract key entities/signals from tool output, verify they appear in agent's reasoning or are addressed by subsequent actions |
| **MAST coverage** | FM-2.2 (Fail to ask for clarification), FM-2.5 (Ignored other agent's input) |
| **Remediation** | Structured error parsing; force agent to enumerate all signals before acting; highlight critical fields in tool output |

##### 3.4 Action Deficiency

| Attribute | Value |
|---|---|
| **Definition** | Agent correctly diagnoses the problem but selects the wrong corrective action |
| **Human analogy** | Doctor correctly diagnoses infection but prescribes wrong antibiotic |
| **Agent manifestation** | Agent correctly identifies a null pointer error but adds a try/catch (masking) instead of a null check (fixing). Agent identifies a race condition but adds a sleep() instead of proper synchronization. |
| **Trace indicators** | Reasoning text contains correct diagnosis; subsequent tool call implements a different fix category than what the diagnosis implies |
| **Detection method** | ML-based: classify diagnosis from reasoning, classify fix from action, detect diagnosis-fix category mismatch |
| **MAST coverage** | FM-2.6 (Reasoning-action mismatch), FM-3.3 (Incorrect verification) |
| **Remediation** | Fix-type validation: "Does this fix address the diagnosed root cause?"; provide fix templates per diagnosis type |

---

#### Category 4: KNOWLEDGE FAILURES

**Cognitive mechanism:** Knowledge-based level. Agent's internal model of the system is fundamentally incomplete or incorrect. These are the hardest errors to detect because the agent's reasoning appears internally consistent — it's just built on wrong assumptions.

##### 4.1 Incomplete Model

| Attribute | Value |
|---|---|
| **Definition** | Agent operates on a mental model that is missing critical components of the actual system |
| **Human analogy** | Mechanic who doesn't know about the turbocharger, diagnosing engine problems incorrectly |
| **Agent manifestation** | Agent assumes REST API when system uses GraphQL. Agent doesn't know about the event bus connecting services. Agent treats async functions as synchronous. |
| **Trace indicators** | Agent makes assumptions (visible in reasoning) that contradict system reality; increasing error rate as agent builds on incorrect foundation; hypothesis-test cycles that never converge |
| **Detection method** | ML-based: detect non-converging error-fix cycles (>N attempts with no success rate improvement); compare agent's stated assumptions against codebase facts via RAG |
| **MAST coverage** | FM-1.1 (Disobey task specification), FM-1.2 (Disobey role specification) |
| **Remediation** | Architecture documentation injection; codebase RAG; escalate to human after N non-converging attempts |

##### 4.2 Confirmation Bias

| Attribute | Value |
|---|---|
| **Definition** | Agent seeks evidence supporting its current hypothesis while ignoring or discounting contradictory evidence |
| **Human analogy** | Investigator who builds a case around the first suspect, ignoring exculpatory evidence |
| **Agent manifestation** | Agent hypothesizes "database issue," queries only DB-related logs, ignores network timeout in application logs. Agent focuses only on the function it initially suspected. |
| **Trace indicators** | Low diagnostic diversity: agent queries narrow scope despite broad error indicators; information available in context that contradicts hypothesis is not referenced in reasoning |
| **Detection method** | ML-based: measure diversity of diagnostic actions relative to breadth of error signals; detect when contradictory evidence in tool output is not addressed |
| **MAST coverage** | FM-2.5 (Ignored other agent's input) |
| **Remediation** | Force hypothesis enumeration: "List 3 possible causes before investigating"; devil's advocate prompting; require agent to explain why alternative hypotheses are unlikely |

##### 4.3 Fixation

| Attribute | Value |
|---|---|
| **Definition** | Agent persists with a failing approach long past the point where a competent operator would switch strategies |
| **Human analogy** | Pilot who keeps trying to restart a failed engine instead of preparing for emergency landing |
| **Agent manifestation** | Agent spends 20 steps debugging function A when the bug is in function B that calls A. Agent keeps trying to install a package that doesn't exist instead of checking for alternative packages. |
| **Trace indicators** | Same tool or tool sequence repeated N+ times with persistent failure; no strategy change despite repeated errors; high step count in single sub-goal |
| **Detection method** | Rule-based: detect N consecutive failed attempts with same tool/approach (current agentdiag LoopDetector + RecoveryFailureDetector) |
| **MAST coverage** | FM-1.3 (Step repetition), FM-1.5 (Unaware of termination conditions) |
| **Remediation** | Strategy rotation: force approach change after N failures; timeout per sub-goal; explicit "if this doesn't work in 3 tries, try a different approach" instructions |

##### 4.4 Overconfidence

| Attribute | Value |
|---|---|
| **Definition** | Agent states conclusions with certainty that exceeds the evidence available |
| **Human analogy** | Expert who gives definitive diagnosis based on one data point |
| **Agent manifestation** | Agent states "this endpoint returns JSON" (when it returns XML). Agent claims "I've fixed the bug" without running tests. Agent asserts API behavior from training data that contradicts current documentation. |
| **Trace indicators** | High-confidence assertions followed by failures; no verification steps between assertion and action; hallucinated API behavior (agent actions don't match actual API responses) |
| **Detection method** | ML-based: detect assertions in reasoning that are not supported by preceding tool outputs; detect missing verification between diagnosis and action |
| **MAST coverage** | FM-3.2 (No or incomplete verification), FM-3.3 (Incorrect verification) |
| **Remediation** | Require evidence citation: agent must reference tool output supporting assertions; enforce verification steps; calibration training |

---

#### Category 5: PLAN STRUCTURE FAILURES

**Cognitive mechanism:** Meta-cognitive level. Failures in the task decomposition and sequencing process itself, independent of whether individual operations are correctly executed. Derived from HTA analysis.

##### 5.1 Goal Misinterpretation

| Attribute | Value |
|---|---|
| **Definition** | Agent pursues a different objective than what was intended by the user |
| **Human analogy** | Builder who constructs a deck when the owner asked for a dock |
| **Agent manifestation** | User asks "fix the auth bug" → agent refactors entire auth module. User asks "make it faster" → agent adds caching instead of fixing the O(n^2) algorithm. |
| **Trace indicators** | Tool calls diverge from stated goal keywords; high tool diversity unrelated to goal; user clarification requests |
| **Detection method** | ML-based: semantic similarity between goal text and trace actions; detect divergence onset |
| **MAST coverage** | FM-1.1 (Disobey task specification), FM-2.3 (Task derailment) |
| **Remediation** | Goal paraphrasing: agent restates task before beginning; periodic goal-alignment checkpoints |

##### 5.2 Wrong Decomposition

| Attribute | Value |
|---|---|
| **Definition** | Agent decomposes the goal into sub-goals that are incorrect, incomplete, or unnecessary |
| **Human analogy** | Plumber who tears out the wall to fix a leak that's accessible from the access panel |
| **Agent manifestation** | Agent skips verification sub-goal entirely. Agent adds unnecessary sub-goals (linting, formatting, package upgrades) while fixing a single bug. |
| **Trace indicators** | Missing expected sub-goal (e.g., no test run after code change); excess sub-goals unrelated to primary task; tool-call count disproportionate to task complexity |
| **Detection method** | Rule-based: check for required sub-goal patterns per task type; flag excessive tool diversity for simple tasks |
| **MAST coverage** | FM-3.1 (Premature termination), FM-3.2 (No or incomplete verification) |
| **Remediation** | Sub-goal templates for common tasks; required verification checklist; scope guardrails |

##### 5.3 Sequencing Error

| Attribute | Value |
|---|---|
| **Definition** | Sub-goals or operations executed in wrong order, violating logical dependencies |
| **Human analogy** | Painting the wall before patching the holes |
| **Agent manifestation** | Agent edits code before reading the file. Runs tests before saving changes. Deploys before testing. |
| **Trace indicators** | `edit_file` without preceding `read_file` for that path; `run_tests` before `write_file`; logical dependency violations |
| **Detection method** | Rule-based: define tool-pair ordering constraints (read→edit, edit→save, change→test); detect violations |
| **MAST coverage** | FM-1.1 (Disobey task specification) |
| **Remediation** | Tool-chain validators: enforce precondition checks; planning step before execution |

##### 5.4 Missing Precondition

| Attribute | Value |
|---|---|
| **Definition** | Agent attempts an operation without first establishing its required preconditions |
| **Human analogy** | Trying to start a car without putting it in park first |
| **Agent manifestation** | Agent tries to run tests without installing dependencies. Edits a file without checking if it exists. Calls API without authentication. |
| **Trace indicators** | Tool call fails with precondition-related error (file not found, permission denied, dependency missing); no preceding step that establishes the precondition |
| **Detection method** | Rule-based: map tool calls to known preconditions; check if precondition-establishing steps occurred before tool call |
| **MAST coverage** | FM-1.1 (Disobey task specification) |
| **Remediation** | Precondition checking: validate requirements before tool execution; provide setup scaffolding |

---

#### Category 6: COORDINATION FAILURES

**Cognitive mechanism:** Inter-agent and system-interaction level. Failures in how multiple agents (or agent + human + tools) exchange information and align on goals. Derived from HFACS crew resource management and MAST inter-agent misalignment categories.

##### 6.1 Information Withholding

| Attribute | Value |
|---|---|
| **Definition** | Agent fails to share critical information that other agents or the user need for decision-making |
| **Human analogy** | Surgical nurse not reporting unusual vital signs to surgeon |
| **Agent manifestation** | Agent discovers a critical issue but doesn't surface it to the user or other agents. Agent finds that a dependency is deprecated but only fixes the immediate issue. |
| **Trace indicators** | Critical information in tool output not surfaced in agent's response; inter-agent messages missing key findings |
| **Detection method** | ML-based: extract critical entities from tool outputs; verify they appear in agent responses |
| **MAST coverage** | FM-2.4 (Information withholding) |
| **Remediation** | Critical information extraction and mandatory surfacing; structured handoff protocols |

##### 6.2 Conversation Reset

| Attribute | Value |
|---|---|
| **Definition** | Agent unexpectedly restarts the dialogue context, losing prior progress |
| **Human analogy** | Surgeon re-starting a procedure from scratch mid-operation |
| **Agent manifestation** | Agent re-introduces itself mid-conversation; re-asks questions already answered; restarts task from scratch |
| **Trace indicators** | Sudden decrease in contextual reference density; re-execution of early-stage steps; reasoning text shows no awareness of prior steps |
| **Detection method** | Rule-based: detect task-restart patterns (information-gathering steps appearing after execution steps); ML-based: context-continuity scoring |
| **MAST coverage** | FM-2.1 (Conversation reset), FM-1.4 (Loss of conversation history) |
| **Remediation** | Context persistence mechanisms; state checkpointing; conversation history summarization |

##### 6.3 Input Disregard

| Attribute | Value |
|---|---|
| **Definition** | Agent ignores or fails to incorporate input from other agents, the user, or tool outputs |
| **Human analogy** | Pilot who ignores co-pilot's warnings |
| **Agent manifestation** | User specifies a constraint that agent doesn't follow. Agent B ignores Agent A's findings. Agent doesn't use information from tool output in subsequent reasoning. |
| **Trace indicators** | User instruction appears in context but is not reflected in agent actions; tool output contains information not referenced in subsequent steps |
| **Detection method** | ML-based: instruction-following score; measure information utilization from tool outputs |
| **MAST coverage** | FM-2.5 (Ignored other agent's input), FM-2.2 (Fail to ask for clarification) |
| **Remediation** | Instruction echo: agent must paraphrase constraints before executing; input acknowledgment protocols |

##### 6.4 Reasoning-Action Mismatch

| Attribute | Value |
|---|---|
| **Definition** | Agent's stated reasoning is logically inconsistent with the action it takes |
| **Human analogy** | Doctor who says "we should run more tests" and then prescribes medication without testing |
| **Agent manifestation** | Agent reasons "I should check the logs first" then immediately edits a file. Agent says "this looks correct" then makes changes. |
| **Trace indicators** | Semantic contradiction between reasoning event text and the immediately-following tool call type/parameters |
| **Detection method** | ML-based: embed reasoning and action separately; detect low cosine similarity or logical contradiction |
| **MAST coverage** | FM-2.6 (Reasoning-action mismatch) |
| **Remediation** | Reasoning-action consistency checks; plan-then-execute workflow enforcement |

---

#### Category 7: RESOURCE EXHAUSTION

**Cognitive mechanism:** Precondition degradation. The agent's processing capacity becomes insufficient for the task, analogous to human fatigue, cognitive overload, or environmental stressors. Derived from Swiss Cheese Layer 3 and HFACS preconditions.

##### 7.1 Context Saturation

| Attribute | Value |
|---|---|
| **Definition** | Context window fills to capacity, forcing truncation of earlier information |
| **Human analogy** | Information overload — too many documents to review, starts forgetting early ones |
| **Agent manifestation** | Agent's performance degrades with trace length; early context information is lost; errors increase monotonically with step count |
| **Trace indicators** | tokens_in approaching model context limit; error rate positively correlated with step number; references to early context become incorrect or absent |
| **Detection method** | Rule-based: track cumulative token count; monitor error rate vs. step count correlation |
| **MAST coverage** | FM-1.4 (Loss of conversation history) |
| **Remediation** | Context window management: summarization, pruning, RAG; break long tasks into subtasks with clean context |

##### 7.2 Token Explosion

| Attribute | Value |
|---|---|
| **Definition** | Per-step token consumption grows super-linearly, rapidly consuming the available budget |
| **Human analogy** | Increasingly verbose explanations that crowd out productive work |
| **Agent manifestation** | Tool outputs grow exponentially (e.g., reading increasingly large files); agent's reasoning sections grow longer with each step; cumulative tokens show accelerating growth |
| **Trace indicators** | Quadratic or exponential fit on per-step token count; last-quarter avg tokens >> first-quarter avg tokens; positive second derivative of cumulative token function |
| **Detection method** | Rule-based: polynomial regression on token time series; growth ratio thresholding (current agentdiag TokenExplosionDetector) |
| **MAST coverage** | Not directly covered by MAST (novel contribution) |
| **Remediation** | Token budgets per step and per task; output truncation; streaming detection and abort |

##### 7.3 Latency Cascade

| Attribute | Value |
|---|---|
| **Definition** | Progressive increase in step latency indicating systemic slowdown |
| **Human analogy** | Fatigue causing progressively slower reaction times |
| **Agent manifestation** | Tool calls take longer and longer; reasoning steps become more verbose; overall task velocity decreases |
| **Trace indicators** | Positive slope on latency time series; IQR-based outlier detection identifies stall events; latency ratio between halves > 1.5 |
| **Detection method** | Rule-based: latency trend analysis; IQR outlier detection (current agentdiag StallDetector) |
| **MAST coverage** | Not directly covered by MAST (novel contribution) |
| **Remediation** | Timeout limits; latency monitoring with adaptive thresholds; escalation after N stalls |

##### 7.4 Attention Degradation

| Attribute | Value |
|---|---|
| **Definition** | Agent's effective attention to relevant information decreases, typically due to position in context window or competing signals |
| **Human analogy** | Nurse who misses critical vital sign change at end of a 12-hour shift |
| **Agent manifestation** | Agent's responses become less specific, more generic; references to task details decrease; quality of reasoning text degrades in later steps |
| **Trace indicators** | Decreasing specificity of reasoning text; increasing use of generic phrases; decreasing overlap between tool output content and subsequent reasoning |
| **Detection method** | ML-based: reasoning quality scoring over time; information extraction rate from tool outputs |
| **MAST coverage** | FM-1.4 (Loss of conversation history) |
| **Remediation** | Attention refreshing: periodic re-injection of key task details; recency-boosted context management |

---

#### Category 8: SYSTEMIC DEFICIENCIES

**Cognitive mechanism:** Latent conditions (Swiss Cheese Layers 1-2, HFACS Levels 3-4). These are not agent errors per se but system-level weaknesses that create the conditions for agent errors. They are "holes in the cheese" that are present before any active failure occurs.

##### 8.1 Inadequate Guidance

| Attribute | Value |
|---|---|
| **Definition** | System prompt, instructions, or role specification fails to provide adequate guidance for the task or edge cases |
| **Human analogy** | Airline with inadequate standard operating procedures |
| **Agent manifestation** | Agent improvises in situations where explicit instructions should exist; inconsistent behavior across similar situations; edge cases handled ad hoc |
| **Trace indicators** | High variance in agent behavior for similar tasks; agent reasoning shows uncertainty about how to proceed; multiple approach changes within a single task |
| **Detection method** | Audit-based: systematic review of system prompt coverage; trace-based: detect high behavioral variance across similar tasks |
| **MAST coverage** | FM-1.2 (Disobey role specification) |
| **Remediation** | System prompt engineering: comprehensive role specification, edge case handling, escalation procedures |

##### 8.2 Missing Guardrails

| Attribute | Value |
|---|---|
| **Definition** | Safety controls are absent or insufficient, allowing dangerous actions without verification |
| **Human analogy** | Nuclear plant without safety interlocks |
| **Agent manifestation** | Agent can execute destructive operations (delete, drop, force-push) without confirmation; no rate limits on expensive operations; no scope limits on file modifications |
| **Trace indicators** | Destructive tool calls without preceding confirmation step; irreversible operations executed routinely; no safety-check events in trace |
| **Detection method** | Audit-based: review tool permission configuration; rule-based: detect destructive operations without preceding verification |
| **MAST coverage** | FM-1.5 (Unaware of termination conditions) |
| **Remediation** | Guardrail framework: require confirmation for destructive ops; rate limiting; scope restrictions; human-in-the-loop for high-risk operations |

##### 8.3 Tool Design Flaws

| Attribute | Value |
|---|---|
| **Definition** | Tool APIs return ambiguous, inconsistent, or misleading outputs that predispose the agent to errors |
| **Human analogy** | Medical device with confusing interface leading to medication errors |
| **Agent manifestation** | Tool returns success status but actually failed; error messages are ambiguous; output format changes between calls; tool has undocumented side effects |
| **Trace indicators** | Agent acts on tool output that later proves incorrect; high error rate with specific tool; agent's interpretation of tool output contradicts actual outcome |
| **Detection method** | Audit-based: tool API quality review; trace-based: tool-specific error rate analysis |
| **MAST coverage** | Not directly covered by MAST (novel contribution — infrastructure-level) |
| **Remediation** | Tool API standardization: structured outputs, consistent error formats, explicit success/failure semantics |

##### 8.4 Capability Mismatch

| Attribute | Value |
|---|---|
| **Definition** | The model, context window, or token budget is insufficient for the assigned task |
| **Human analogy** | Assigning an intern to perform brain surgery |
| **Agent manifestation** | Task requires capabilities the model lacks (e.g., vision, long context, specific domain knowledge); task complexity exceeds model's reasoning capacity; token budget is insufficient for the required number of steps |
| **Trace indicators** | Systematic failure pattern across all attempts (not improving with retries); context truncation events; budget exhaustion before task completion |
| **Detection method** | Audit-based: model capability benchmarks vs. task requirements; trace-based: detect systematic non-improvement across retries |
| **MAST coverage** | Not directly covered by MAST (organizational-level factor) |
| **Remediation** | Model-task matching: assess task complexity before assignment; right-size model, context, and budget; fallback escalation to more capable model |

---

## 8. MAST Coverage Validation

The following table demonstrates that CAFT covers all 14 MAST failure modes and maps them to their cognitive generating mechanisms.

| MAST Code | MAST Failure Mode | CAFT Category | CAFT Code(s) | Cognitive Mechanism |
|---|---|---|---|---|
| FM-1.1 | Disobey task specification | Multiple | 1.2, 3.2, 4.1, 5.1, 5.3, 5.4 | Mode error, rule misapplication, incomplete model, goal/sequence/precondition failures |
| FM-1.2 | Disobey role specification | Execution / Systemic | 1.1, 4.1, 8.1 | Capture error, incomplete model, inadequate guidance |
| FM-1.3 | Step repetition | Memory / Knowledge | 2.2, 4.3 | Repetition lapse, fixation |
| FM-1.4 | Loss of conversation history | Memory / Resource | 2.1, 6.2, 7.1, 7.4 | Context loss, conversation reset, saturation, attention degradation |
| FM-1.5 | Unaware of termination conditions | Knowledge / Systemic | 4.3, 8.2 | Fixation, missing guardrails |
| FM-2.1 | Conversation reset | Coordination | 6.2 | Conversation reset |
| FM-2.2 | Fail to ask for clarification | Rule / Coordination | 3.3, 6.3 | Encoding deficiency, input disregard |
| FM-2.3 | Task derailment | Memory / Plan | 2.4, 5.1 | Goal drift, goal misinterpretation |
| FM-2.4 | Information withholding | Coordination | 6.1 | Information withholding |
| FM-2.5 | Ignored other agent's input | Rule / Coordination / Knowledge | 3.3, 4.2, 6.3 | Encoding deficiency, confirmation bias, input disregard |
| FM-2.6 | Reasoning-action mismatch | Execution / Rule / Coordination | 1.1, 1.4, 3.1, 3.4, 6.4 | Capture error, perceptual confusion, wrong rule, action deficiency, reasoning-action mismatch |
| FM-3.1 | Premature termination | Memory / Plan | 2.3, 5.2 | Omission, wrong decomposition |
| FM-3.2 | No or incomplete verification | Memory / Plan / Knowledge | 2.3, 4.4, 5.2 | Omission, overconfidence, wrong decomposition |
| FM-3.3 | Incorrect verification | Rule / Knowledge | 3.4, 4.4 | Action deficiency, overconfidence |

**Coverage:** 14/14 MAST failure modes mapped. CAFT additionally covers 4 failure types not in MAST: Token Explosion (7.2), Latency Cascade (7.3), Tool Design Flaws (8.3), and Capability Mismatch (8.4). These represent infrastructure and resource-level failures that MAST's trace-annotation methodology would not capture, as they are latent conditions rather than observable agent acts.

---

## 9. Cross-Framework Concordance Matrix

This matrix shows how each CAFT category maps back to all five source frameworks, demonstrating that CAFT is a genuine synthesis rather than a superset.

| CAFT Category | GEMS | SRK Level | HTA Level | Swiss Cheese Layer | HFACS Level |
|---|---|---|---|---|---|
| 1. Execution Errors | Slips | Skill-based | Operation | L4: Active | L1: Skill-based error |
| 2. Memory Failures | Lapses | Skill/Rule | Sub-goal/Operation | L3: Precondition + L4: Active | L1: Skill-based error + L2: Adverse mental state |
| 3. Rule Application | Rule-based mistakes | Rule-based | Sub-goal | L4: Active | L1: Decision error |
| 4. Knowledge Failures | Knowledge-based mistakes | Knowledge-based | Goal/Sub-goal | L4: Active | L1: Decision error + L2: Channelized attention |
| 5. Plan Structure | (Not in GEMS) | Rule/Knowledge | Goal/Sub-goal/Plan | L4: Active | L1: Decision error |
| 6. Coordination | (Not in GEMS) | Rule/Knowledge | Inter-task | L3: Precondition | L2: Crew resource management |
| 7. Resource Exhaustion | (Not in GEMS) | (Affects all levels) | (Affects all levels) | L3: Precondition | L2: Adverse mental state |
| 8. Systemic Deficiencies | (Not in GEMS) | (Enables all levels) | (Enables all levels) | L1-L2: Latent | L3: Supervision + L4: Organizational |

---

## 10. Implementation Architecture

CAFT is designed for direct implementation in the `agentdiag` diagnostic framework. Each CAFT failure type maps to a detector that conforms to the `Detector` protocol:

```python
@dataclass
class DetectorMeta:
    name: str               # "capture_error", "context_loss", etc.
    version: str
    failure_type: str       # CAFT code: "EXEC_CAPTURE", "MEM_CONTEXT_LOSS", etc.
    description: str
    required_fields: list[str]
    is_ml: bool = False     # True = commercial tier (knowledge-based detectors)
    caft_category: str = "" # "execution", "memory", "rule", "knowledge", etc.
    gems_type: str = ""     # "slip", "lapse", "rule_mistake", "knowledge_mistake"
    srk_level: str = ""     # "skill", "rule", "knowledge"
```

### Detector Implementation Tiers

| CAFT Category | Detection Tier | Rationale |
|---|---|---|
| 1. Execution Errors | Rule-based (free) | Pattern matching on tool metadata; no ML needed |
| 2. Memory Failures | Rule-based (free) | Deduplication, sequence analysis; existing agentdiag detectors cover most |
| 3. Rule Application | ML-based (commercial) | Requires understanding error semantics and fix appropriateness |
| 4. Knowledge Failures | ML-based (commercial) | Requires reasoning chain analysis and model consistency checking |
| 5. Plan Structure | Rule-based (free) + ML (commercial) | Sequence validation is rule-based; goal alignment requires ML |
| 6. Coordination | ML-based (commercial) | Requires cross-agent reasoning analysis |
| 7. Resource Exhaustion | Rule-based (free) | Statistical analysis on token/latency time series; existing detectors cover this |
| 8. Systemic Deficiencies | Audit-based (free) | Static analysis of configuration, not runtime detection |

### Mapping to Existing agentdiag Detectors

| Current Detector | CAFT Code(s) | Notes |
|---|---|---|
| LoopDetector | 2.2, 4.3 | Captures repetition (lapse) and fixation (knowledge failure) |
| ThrashDetector | 3.1 | Captures wrong-rule-selected via rapid tool switching |
| StallDetector | 7.3 | Captures latency cascade |
| DriftDetector | 2.4, 5.1 | Captures goal drift and goal misinterpretation |
| CascadeDetector | 7.3, 3.3 | Captures latency cascade and encoding deficiency |
| TokenExplosionDetector | 7.2 | Captures token explosion |
| DeadEndDetector | 4.3, 5.2 | Captures fixation and wrong decomposition |
| RecoveryFailureDetector | 3.1, 4.3 | Captures wrong-rule-selected and fixation in error recovery |

The existing 8 detectors cover 11 of the 32 CAFT failure types (Categories 2, 3, 4, 5, 7). The remaining 21 types represent the expansion roadmap for future detectors, with Categories 1 and 6 being the highest-priority gaps.

---

## 11. Limitations and Future Work

**Limitations:**
1. The structural isomorphism between human cognition and LLM processing is an analogy, not an identity. LLMs do not have working memory or attention in the same way humans do. The mapping is useful because it predicts observable behavior, not because the mechanisms are identical.
2. CAFT Categories 3 and 4 (rule and knowledge failures) are difficult to detect without ML-based semantic analysis. Rule-based detectors can catch surface symptoms but not root cognitive mechanisms.
3. The taxonomy is validated against MAST (multi-agent coding/math tasks) and may require extension for other agentic domains (robotics, web browsing, scientific research).

**Future work:**
1. **Empirical validation**: Annotate 1000+ agent traces with CAFT codes; measure inter-annotator agreement; compare against MAST annotations.
2. **ML detectors**: Build classifiers for Categories 3, 4, and 6 using trace embeddings.
3. **Predictive model**: Use CAFT precondition categories (7, 8) to predict active failures (1-6) before they occur.
4. **Cross-domain extension**: Apply CAFT to web-browsing agents, robotic agents, and scientific research agents.

---

## References

Annett, J., & Duncan, K. D. (1967). Task analysis and training design. *Occupational Psychology*, 41, 211-221.

Cemri, M., Pan, M. Z., Yang, S., et al. (2025). Why Do Multi-Agent LLM Systems Fail? *arXiv:2503.13657*. [https://arxiv.org/abs/2503.13657](https://arxiv.org/abs/2503.13657)

Microsoft. (2025). Taxonomy of Failure Mode in Agentic AI Systems. [https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf)

Rasmussen, J. (1983). Skills, rules, and knowledge; signals, signs, and symbols, and other distinctions in human performance models. *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-13(3), 257-266.

Reason, J. (1990). *Human Error*. Cambridge University Press.

Reason, J. (1997). *Managing the Risks of Organizational Accidents*. Ashgate.

Shappell, S. A., & Wiegmann, D. A. (2000). The Human Factors Analysis and Classification System — HFACS. *Federal Aviation Administration Report*, DOT/FAA/AM-00/7.

Stanton, N. A. (2006). Hierarchical task analysis: Developments, applications, and extensions. *Applied Ergonomics*, 37(1), 55-79.

---

## Appendix A: CAFT Quick Reference Card

| Code | Name | Mechanism | Detection | Tier |
|---|---|---|---|---|
| 1.1 | Capture Error | Slip | Tool frequency × context relevance | Rule |
| 1.2 | Mode Error | Slip | Parameter × context metadata mismatch | Rule |
| 1.3 | Description Error | Slip | Fuzzy target matching | Rule |
| 1.4 | Perceptual Confusion | Slip | Output entity × action consistency | ML |
| 2.1 | Context Loss | Lapse | Re-execution detection; self-consistency | Rule+ML |
| 2.2 | Repetition | Lapse | (tool, params, hash) deduplication | Rule |
| 2.3 | Omission | Lapse | Planned-vs-executed step matching | Rule |
| 2.4 | Goal Drift | Lapse | Tool distribution shift (JSD) | Rule |
| 3.1 | Wrong Rule Selected | Rule mistake | Error-type × fix-type mismatch | ML |
| 3.2 | Good Rule Misapplied | Rule mistake | Pattern precondition validation | ML |
| 3.3 | Encoding Deficiency | Rule mistake | Output entity utilization rate | ML |
| 3.4 | Action Deficiency | Rule mistake | Diagnosis × action category match | ML |
| 4.1 | Incomplete Model | Knowledge mistake | Non-converging error-fix cycles | ML |
| 4.2 | Confirmation Bias | Knowledge mistake | Diagnostic diversity scoring | ML |
| 4.3 | Fixation | Knowledge mistake | Repeated failure without strategy change | Rule |
| 4.4 | Overconfidence | Knowledge mistake | Assertion × evidence gap | ML |
| 5.1 | Goal Misinterpretation | Plan failure | Goal-action semantic similarity | ML |
| 5.2 | Wrong Decomposition | Plan failure | Sub-goal template matching | Rule |
| 5.3 | Sequencing Error | Plan failure | Tool-pair ordering constraints | Rule |
| 5.4 | Missing Precondition | Plan failure | Precondition dependency checking | Rule |
| 6.1 | Information Withholding | Coordination | Output entity surfacing rate | ML |
| 6.2 | Conversation Reset | Coordination | Restart pattern detection | Rule |
| 6.3 | Input Disregard | Coordination | Instruction-following score | ML |
| 6.4 | Reasoning-Action Mismatch | Coordination | Reasoning-action embedding distance | ML |
| 7.1 | Context Saturation | Resource | Cumulative token tracking | Rule |
| 7.2 | Token Explosion | Resource | Token growth regression | Rule |
| 7.3 | Latency Cascade | Resource | Latency trend + IQR outliers | Rule |
| 7.4 | Attention Degradation | Resource | Reasoning quality over time | ML |
| 8.1 | Inadequate Guidance | Systemic | Behavioral variance audit | Audit |
| 8.2 | Missing Guardrails | Systemic | Permission configuration audit | Audit |
| 8.3 | Tool Design Flaws | Systemic | Tool-specific error rate analysis | Audit |
| 8.4 | Capability Mismatch | Systemic | Model benchmark vs. task requirements | Audit |
