# CAFT: Cognitive Agent Failure Taxonomy

Real-time anomaly detection for AI coding agents using information theory. CAFT monitors agent execution traces, computes Shannon entropy, mutual information, LZ compression, and KL divergence over a sliding window of agent actions, then flags multi-metric co-occurrences as named failure signatures — all without training data, labeled examples, or ground truth. The system models the agent as an information processing system using Wickens' six-stage framework (perception, attention, working memory, decision making, action, feedback) and detects when that processing breaks down.

## Quick Start

```bash
# Install
pip install -e .

# Watch a live Claude Code session
python -m agentdiag live

# Run the synthetic demo (no live session needed)
python -m agentdiag monitor-demo

# Run the three-agent harness with mock agents
python -m agentdiag harness "Build a REST API with authentication"
```

<!-- Screenshot: the Wickens IP dashboard showing sparklines, pipeline strip,
     working memory panel, and anomaly annotations. Place a 1200px-wide PNG
     at docs/images/dashboard.png and uncomment the line below. -->
<!-- ![CAFT Dashboard](docs/images/dashboard.png) -->

## Architecture

```
Layer 5: Feedback / Learning
  OpenViking context DB accumulates skills across sessions

Layer 4: Visualization
  Web dashboard with sparklines, pipeline strip, working memory panel

Layer 3: Analysis
  SymbolStream (IT metrics) -> SelfCalibratingBaseline (z-scores) -> Compositor (signatures)

Layer 2: Instrumentation
  ObservableEvent contract — adapters convert any agent framework to a universal event stream

Layer 1: Agent Execution
  Claude Code / LangChain / CrewAI / custom agents producing tool calls, file reads, shell commands
```

**Layer 1 — Agent Execution.** Any agent framework that produces tool calls: Claude Code, LangChain, CrewAI, or the built-in three-agent harness.

**Layer 2 — Instrumentation.** Framework-specific adapters convert raw agent output into `ObservableEvent` objects using the shared contract in `observable.py`. The harness orchestrator and OpenViking client emit events directly.

**Layer 3 — Analysis.** The `UniversalMonitor` routes events through `EventRouter` (per-signal-type `SymbolStream` instances computing entropy/MI/compression/surprisal/KL), `SelfCalibratingBaseline` (z-scores with phase-conditional thresholds), and `CompositionalAnomalyDetector` (multi-metric signature matching).

**Layer 4 — Visualization.** FastAPI server with WebSocket streaming to a browser dashboard. Three-column layout: working memory panel (left), sparklines + pipeline strip (center), anomaly annotations (right).

**Layer 5 — Feedback / Learning.** OpenViking-backed persistent context accumulates confirmed/false-positive case patterns across sessions, adjusting detector confidence over time. The harness orchestrator's retrospective phase crystallizes sprint outcomes into reusable skills.

## Evaluation Results

Evaluated on 65 synthetic traces (52 with injected failures, 13 clean) across 7 task domains (web apps, CLI tools, data pipelines, games, documentation, testing, DevOps) at the default z=3.0 threshold:

| Metric | z=3.0 (default) | z=2.75 (optimal) |
|--------|-----------------|-------------------|
| True Positive Rate | 94.2% [86.5%, 100%] | 98.1% |
| False Positive Rate | 8.7% | 12.1% |
| Signature Accuracy | 98.0% | 98.0% |
| Detection Latency (median) | 26 steps [18, 37] | 24 steps |

Failure detection by type:

| Failure Type | TPR | Signature Accuracy | Expected Signature |
|-------------|-----|--------------------|--------------------|
| loop (stuck rereading) | 100% | 92.3% | `mechanical_repetition` |
| drift (goal shift) | 100% | 100% | `distributional_shift` |
| thrash (context overload) | 100% | 100% | `context_thrashing` |
| stall (no progress) | 76.9% | 100% | `distributional_anomaly` |

Cross-task generalization: 100% TPR on 5 of 7 domains; 87.5% on web apps, 75% on games (stall detection is harder in complex domains). All results are zero-shot — no per-domain tuning.

Confidence intervals from 10,000 bootstrap resamples. Full report: `agentdiag/eval/results/report.md`.

## The Wickens IP Model Mapping

The system maps agent behavior onto Wickens' six-stage Information Processing model:

| IP Stage | Agent Behavior | Stream | Key Metrics | Anomaly When... |
|----------|---------------|--------|-------------|-----------------|
| Perception | File reads, searches, web fetches | `read_stream` | read entropy, perception breadth | High entropy = unfocused scanning |
| Attention | Tool selection patterns | `tool_stream` | tool entropy, attention focus | Low diversity = tunnel vision |
| Working Memory | Context loads, file access recency | `memory_stream` | memory entropy, escalation rate, tier distribution | High escalation = thrashing |
| Decision Making | Planning steps, reasoning events | `action_stream` | action MI, compression ratio | Low MI = incoherent decisions |
| Action | Code edits, shell commands | `write_stream` | action entropy, success rate | Low entropy + low MI = stuck loop |
| Feedback | Test runs, verification steps | `action_stream` | KL divergence, verify ratio | High KL = distribution shift from baseline |

## Anomaly Signatures

Named multi-metric co-occurrence patterns, derived from information theory:

| Signature | Condition | Severity | IP Stage |
|-----------|-----------|----------|----------|
| `mechanical_repetition` | low entropy + low MI | warning | response selection |
| `tight_iteration` | low entropy + high MI | info | response execution |
| `distributional_shift` | high KL + high surprisal | warning | perceptual |
| `context_thrashing` | high KL + high MI or read entropy | warning | working memory |
| `execution_regression` | low compression + low entropy | warning | response execution |
| `stagnation` | low compression + low KL | critical | response selection |
| `distributional_anomaly` | high KL + abnormal entropy | info | response selection |
| `memory_thrashing` | high escalation rate + high namespace entropy | warning | working memory |

Single-metric blips are ignored (noise). Only when 2+ metrics are simultaneously anomalous does the system flag a signature.

## Entry Points

### `live` — Watch Real Agents

```bash
# Auto-detect the most recent Claude Code session
python -m agentdiag live

# Watch a specific session file
python -m agentdiag live --session ~/.claude/projects/-Users-me-myproject/session.jsonl

# Replay a past session at 10x speed
python -m agentdiag live --replay past_session.jsonl --speed 10

# Replay a saved harness result
python -m agentdiag live --replay-harness harness_result.json --speed 5
```

### `monitor-demo` — Synthetic Demo

```bash
# Run all demo scenarios (step_repetition, context_loss, premature_termination, clean)
python -m agentdiag monitor-demo

# Run a specific scenario
python -m agentdiag monitor-demo step_repetition
```

### `harness` — Orchestrated Agents

```bash
# Run the three-agent harness with mock agents
python -m agentdiag harness "Build a login page with OAuth" --max-sprints 3

# Save the result for replay
python -m agentdiag harness "Fix the authentication bug" --output result.json

# Replay later
python -m agentdiag live --replay-harness result.json
```

The harness runs a planner (decomposes the goal into sprints), generator (produces code), and evaluator (grades output) in a GAN-inspired loop. Phase boundaries, contract negotiations, evaluation scores, and retrospective skill crystallization are all visible in the dashboard.

## Evaluation

Run the full evaluation suite:

```bash
# Generate synthetic traces
python -m agentdiag.eval.trace_generator --output agentdiag/eval/traces/

# Run all traces through the pipeline
python -m agentdiag.eval.runner --all --z-threshold 3.0

# Analyze results and generate report
python -m agentdiag.eval.analyze --results agentdiag/eval/results/ --output report.md

# Parameter sweep across z-thresholds
python -m agentdiag.eval.runner --all --sweep --z-range 1.5,5.0,0.5

# Statistical analysis (bootstrap CI, binomial tests)
python -m agentdiag.eval.stats --results agentdiag/eval/results/
```

## Prospective Validation

For human-vs-system agreement testing on real, unlabeled agent sessions:

```bash
# Watch a live session with validation logging enabled
python -m agentdiag live --log-validation validation.jsonl

# During the session, press:
#   [s] = "agent is struggling here"
#   [f] = "agent is fine here"
#   [q] = stop logging

# After the session, compute agreement
python scripts/compare_validation.py validation.jsonl
python scripts/compare_validation.py validation.jsonl --window 20 --json
```

The comparison uses a temporal window (default +/-15 steps): a human "struggling" mark matches if the system detected a named anomaly nearby, "fine" matches if no anomaly nearby. Output includes confusion matrix, precision/recall/F1, per-signature hit rates, and a step-by-step timeline.

## Framework Adapters

CAFT works with any agent framework through the `AgentAdapter` interface:

| Adapter | Status | Source |
|---------|--------|--------|
| Claude Code | Built-in | `adapters/claude_adapter.py` |
| Claude API | Built-in | `adapters/claude.py` |
| OpenAI | Built-in | `adapters/openai.py` |
| LangChain | Built-in | `adapters/langchain.py` |
| Harness replay | Built-in | `adapters/harness_adapter.py` |
| OpenViking replay | Built-in | `adapters/viking_adapter.py` |
| Custom | See guide | `docs/ADDING_AN_ADAPTER.md` |

The `ObservableEvent` contract (`observable.py`) is the bridge: adapters produce events, the analysis layer consumes them via `event.to_symbol()`. Neither side knows about the other's internals.

```python
from agentdiag.adapters import get_adapter

adapter = get_adapter("claude_code")  # or "viking", "harness", "mixed"
```

See `docs/ADDING_AN_ADAPTER.md` for a step-by-step guide to adding a new framework.

## Contributing

Architecture decisions and the multi-agent coordination plan are documented in `docs/INTEGRATION_PLAN.md`. The codebase was built by two parallel agents coordinating through the `ObservableEvent` shared contract — see `docs/ARCHITECTURE.md` for the file map and data flow.

```bash
# Run the test suite (502+ tests)
python -m pytest tests/ -x -q

# Run only the analysis layer tests
python -m pytest tests/test_analysis_layer.py -v

# Run only the validation log tests
python -m pytest tests/test_validation_log.py -v
```
