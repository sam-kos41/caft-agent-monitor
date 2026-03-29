# Architecture

## File Map

Every `.py` file in `agentdiag/` with its purpose. Files marked **(A1)** were built by Agent 1 (harness/OpenViking), **(A2)** by Agent 2 (visualization/IT measures), **(shared)** by both, or **(original)** if they predate the two-agent build.

### Core Pipeline

| File | Description |
|------|-------------|
| `__init__.py` | Package entry, exports `TraceEvent` and `load_trace` |
| `__main__.py` | CLI dispatcher: monitor, live, harness, evaluate, annotate, demo, visualize **(shared)** |
| `models.py` | `TraceEvent`, `TraceFeatures`, `Diagnosis`, `DiagnosticReport` **(original)** |
| `loading.py` | `load_trace()` — auto-detects JSON/JSONL format **(original)** |
| `hta.py` | `HTAStateMachine` — infers GATHERING/PLANNING/EXECUTING/VERIFYING/DELIVERING phases from tool usage **(original)** |
| `monitor.py` | `MonitorEngine` — legacy real-time pipeline (adapters -> HTA -> CAFT detectors -> dashboard) **(original)** |
| `decision_trace.py` | `DecisionTrace` — per-step record of what every detector computed **(original)** |
| `observable.py` | `ObservableEvent` shared contract: event types, memory tiers, harness phases, convenience constructors **(shared)** |

### Information-Theoretic Analysis

| File | Description |
|------|-------------|
| `cognitive.py` | `SymbolStream` (sliding-window IT measures), `EventRouter` (per-signal dispatch), `CognitiveStateTracker` (6-stage IP model), `WorkingMemoryModel` **(A2)** |
| `baseline.py` | `SelfCalibratingBaseline` — calibrates on first 100 events, z-score anomaly detection with phase-conditional thresholds **(A2)** |
| `compositor.py` | `CompositionalAnomalyDetector` — matches multi-metric co-occurrences to named anomaly signatures **(A2)** |
| `universal_monitor.py` | `UniversalMonitor` — routes `ObservableEvent` through EventRouter/baseline/compositor, exposes `get_state()` for visualization. Also: `CrossStageMI`, `InferredWorkingMemory`, `_MemoryOpsTracker` **(shared)** |

### Harness Orchestration

| File | Description |
|------|-------------|
| `harness.py` | `HarnessOrchestrator` — three-agent (planner/generator/evaluator) orchestration with sprint contracts, GAN-style iteration loop, retrospective skill crystallization **(A1)** |

### Persistent Context (OpenViking)

| File | Description |
|------|-------------|
| `context/__init__.py` | `get_context_store()`, `get_instrumented_store()` — factory functions with graceful degradation **(shared)** |
| `context/openviking.py` | `ContextStore` — two-tier persistence (in-memory buffer + promoted cases), case ledger, feedback loop, FP rate computation **(original)** |
| `context/instrumented.py` | `InstrumentedContextStore` — wraps ContextStore, emits `ObservableEvent` for every viking:// operation **(A1)** |

### CAFT Detectors

| File | Description |
|------|-------------|
| `caft/__init__.py` | Exports `CaftDetector`, `CaftDiagnosis`, `run_caft_detectors()` |
| `caft/base.py` | `CaftDetector` protocol, `CaftDiagnosis` dataclass, `CaftSeverity` enum **(original)** |
| `caft/detectors.py` | 12 detectors: StepRepetition, ContextLoss, ErrorCascade, Stall, TokenExplosion, AnalysisParalysis, StrategicMyopia, ToolMisuse, RecoveryFailure, PrematureTermination, MissingVerification, GoalDrift **(original)** |
| `caft/taxonomy.py` | `CAFT_TAXONOMY` — 8 categories, 33 failure types with observable/latent classification and IP stage mapping **(original)** |
| `caft/registry.py` | `DetectorRegistry` — loader with optional calibration from baselines.json **(original)** |
| `caft/confirm.py` | LLM semantic confirmation layer (Claude CLI or Ollama) **(original)** |
| `caft/calibrated.py` | Phase-specific calibrated detector wrappers **(original)** |
| `caft/benchmark.py` | Validation benchmark runner **(original)** |
| `caft/synthetic.py` | Synthetic trace generators per failure mode **(original)** |
| `caft/demo.py` | Live demo mode with synthetic events **(original)** |

### Adapters

| File | Description |
|------|-------------|
| `adapters/__init__.py` | `auto_parse()` (TraceEvent), `get_adapter()` factory (ObservableEvent), `AgentAdapter` base, `MixedAdapter` **(shared)** |
| `adapters/base.py` | `TraceAdapter` protocol, `AdapterMeta` **(original)** |
| `adapters/generic.py` | Fallback adapter for pre-formatted TraceEvent dicts **(original)** |
| `adapters/claude.py` | Claude Messages API adapter **(original)** |
| `adapters/claude_code.py` | Claude Code session log extractor (`ClaudeCodeExtractor`) **(original)** |
| `adapters/claude_adapter.py` | Claude Code JSONL -> `ObservableEvent` bridge **(A2)** |
| `adapters/openai.py` | OpenAI Chat Completions adapter **(original)** |
| `adapters/langchain.py` | LangChain/LangSmith callback adapter **(original)** |
| `adapters/harness_adapter.py` | Replays serialized `HarnessResult` as `ObservableEvent` stream **(A1)** |
| `adapters/viking_adapter.py` | Replays OpenViking session logs as `ObservableEvent` stream **(A1)** |

### Visualization & Live Mode

| File | Description |
|------|-------------|
| `visualize.py` | FastAPI server with WebSocket, serves `visualize.html` dashboard **(A2)** |
| `live.py` | Live observation: tails Claude Code JSONL, bridges to viz server. Also: harness replay, validation logging **(shared)** |
| `tui.py` | Terminal dashboard (Rich-based, 3-panel layout) **(original)** |

### Evaluation Framework

| File | Description |
|------|-------------|
| `eval/__init__.py` | Eval package |
| `eval/runner.py` | `run_trace()` — feeds traces through ClaudeCodeAdapter -> UniversalMonitor, records metrics **(A2)** |
| `eval/analyze.py` | Computes TPR/FPR/latency/signature accuracy from runner output **(A2)** |
| `eval/trace_generator.py` | Generates synthetic traces with planted failures across 7 task domains **(A2)** |
| `eval/tasks.py` | Task bank: web_app, cli_tool, data_pipeline, game, docs, testing, devops **(A2)** |
| `eval/stats.py` | Bootstrap CI, binomial tests, statistical analysis **(A2)** |
| `eval/plot.py` | Publication-quality figure generation **(A2)** |

### Annotation System

| File | Description |
|------|-------------|
| `annotation.py` | `Annotation`, `AnnotationStore`, `compute_kappa()` **(original)** |
| `annotation_models.py` | `AnnotationRecord` with 4-layer label lifecycle **(original)** |
| `annotation_store.py` | JSONL-backed annotation ledger **(original)** |
| `auto_annotate.py` | 3-step automated annotation pipeline **(original)** |
| `auto_annotate_prompt.py` | LLM annotation prompt templates **(original)** |
| `disagreement.py` | Inter-annotator disagreement analysis **(original)** |

### Other

| File | Description |
|------|-------------|
| `validation_log.py` | `ValidationLog` — human marks + system detections JSONL for prospective validation **(A1)** |
| `evidence.py` | Typed evidence dataclasses for detectors **(original)** |
| `baselines.py` | `CalibrationProfile`, `NormativePhaseModel` — phase-specific thresholds from training traces **(original)** |
| `evaluate.py` | Legacy unified evaluation pipeline **(original)** |
| `metrics.py` | Detection metrics with bootstrap CI and McNemar's test **(original)** |
| `splits.py` | Dev/validation/test split enforcement **(original)** |
| `pilot.py` | Backward-compatible evaluation wrapper **(original)** |
| `explain/templates.py` | Rule-based explanation templates **(original)** |
| `explain/llm.py` | LLM explanation stub (not implemented) **(original)** |
| `testing/__init__.py` | Testing utilities package **(A1)** |
| `testing/synthetic_events.py` | `generate_healthy_run()`, `generate_anomalous_run()` — realistic mixed-source event streams **(A1)** |

---

## Data Flow

```
                        Agent Framework
                             |
                    framework-specific adapter
                             |
                      ObservableEvent
                             |
                    UniversalMonitor.process()
                        /    |    \
                       /     |     \
              EventRouter  Cross   InferredWM
              (per-signal  StageMI  (file/memory
              SymbolStream)         tracking)
                   |
            IT metrics dict
        {action_entropy, action_mi,
         compression_ratio, last_surprisal,
         tool_entropy, kl_divergence, ...}
                   |
          SelfCalibratingBaseline
              .observe(metrics, phase)
                   |
            anomaly dict (z-scores)
        {metric: {value, z_score, direction}}
                   |
        CompositionalAnomalyDetector
              .analyze(anomalies)
                   |
           AnomalySignature | None
                   |
            UniversalMonitor
              .get_state()
                   |
            WebSocket JSON payload
                   |
              Browser Dashboard
```

### Event Routing Rules

| Event Type | SymbolStream? | Baseline? | Working Memory? | Phase Switch? |
|------------|:---:|:---:|:---:|:---:|
| `PHASE_BOUNDARY` | No (stored as marker) | No (switches phase context) | No | Yes |
| `SESSION_START/END` | No | No | No | No |
| `EVALUATION_RESULT` | No (stored for retrospective) | No | No | No |
| `MEMORY_*` | Yes (via memory_stream) | Yes (+ memory metrics) | Yes (explicit items) | No |
| `TOOL_CALL` | Yes (action + tool streams) | Yes | No | No |
| `FILE_READ` | Yes (action + read streams) | Yes | Yes (inferred items) | No |
| `FILE_WRITE` | Yes (action + write streams) | Yes | Yes (inferred items) | No |
| Everything else | Yes (action stream) | Yes | No | No |

---

## The ObservableEvent Contract

`observable.py` defines the universal event type that both the harness/instrumentation layer (Agent 1) and the visualization/analysis layer (Agent 2) build against. Neither agent modifies the other's code — the contract is the bridge.

### Structure

```python
@dataclass
class ObservableEvent:
    # Required
    step: int
    timestamp: float
    event_type: EventType

    # Tool-level (adapters populate these)
    tool_name: Optional[str]
    target_path: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    duration_ms: Optional[float]

    # Memory-level (OpenViking instrumentation populates these)
    viking_uri: Optional[str]
    memory_tier: Optional[MemoryTier]  # L0, L1, L2
    token_count: Optional[int]
    namespace: Optional[str]

    # Harness-level (orchestrator populates these)
    phase: Optional[HarnessPhase]
    agent_role: Optional[AgentRole]
    evaluation_score: Optional[float]
    sprint_number: Optional[int]
    contract_status: Optional[str]
```

### Key Methods

- `to_symbol()` — produces a string for IT computation. The `SymbolStream` never inspects event internals; it receives strings and computes entropy over their distribution.
- `is_phase_marker()` — returns True for PHASE_BOUNDARY, SESSION_START/END. These are rendered as vertical lines on sparklines, not data points.
- `is_memory_operation()` — returns True for MEMORY_LOAD/STORE/EVICT/TIER_ESCALATION. These go to both the SymbolStream and a dedicated memory metrics track.
- `to_dict()` — JSON-serializable dict for WebSocket transport.

### Convenience Constructors

Adapters use these instead of constructing `ObservableEvent` directly:

- `tool_call_event()`, `file_read_event()`, `file_write_event()` — for tool-level events
- `memory_load_event()`, `memory_store_event()`, `tier_escalation_event()` — for OpenViking operations
- `phase_boundary_event()`, `evaluation_event()` — for harness orchestration

---

## The Wickens IP Mapping (Detailed)

| IP Stage | Mapped From | SymbolStream | Metrics Tracked | Anomaly Signatures |
|----------|-------------|--------------|-----------------|-------------------|
| Perception | FILE_READ, MEMORY_LOAD, search/fetch tools | `read_stream` | read_entropy, perception_breadth, perception_depth | High read entropy without convergence |
| Attention | Tool selection diversity within a window | `tool_stream` | tool_entropy, attention_focus, attention_tunnel_risk | Extreme focus (tunnel) or extreme scatter |
| Working Memory | MEMORY_LOAD/EVICT with tier info, file access recency | `memory_stream` | memory_entropy, namespace_entropy, escalation_rate, tier_distribution | `memory_thrashing`, `context_thrashing` |
| Decision Making | Planning/reasoning events, agent_role=planner | `action_stream` | action_mi, compression_ratio, deliberation | Low MI = incoherent decisions, high compression = cycling |
| Action | Code edits, shell commands, agent_role=generator | `action_stream` + `write_stream` | action_entropy, success_rate, repetition_risk | `mechanical_repetition`, `execution_regression`, `stagnation` |
| Feedback | Test runs, evaluation results, agent_role=evaluator | `action_stream` | KL divergence, verify_ratio, evaluation_score | `distributional_shift`, `distributional_anomaly` |

The `CrossStageMI` tracker in `universal_monitor.py` computes mutual information between adjacent stages for the pipeline strip arrow thickness. High MI between decision_making and action means coherent planning-to-execution flow; low MI suggests disconnection.

---

## Graceful Degradation

The system works at every level of integration:

| Components Present | What Works | What Degrades |
|---|---|---|
| **Analysis layer only** (no viz, no harness, no OpenViking) | IT metrics computation, baseline calibration, anomaly detection via `UniversalMonitor.process()` | No visual output, no phase-conditional baselines, no memory tracking |
| **+ Visualization** (standard `live` mode) | Dashboard with sparklines, working memory (inferred from file events), anomaly annotations | Memory panel uses file-access inference, no tier visibility |
| **+ Harness** (no OpenViking) | Phase-conditional baselines, harness phase markers on sparklines, evaluation result overlays, sprint structure | Memory still inferred, no accumulated skills across runs |
| **+ OpenViking** (no harness) | Explicit working memory from viking:// loads, memory operations track, namespace_entropy/escalation_rate metrics | No phase-conditional baselines, no evaluation calibration |
| **Full stack** (all three) | Phase-conditional baselines, explicit memory with tier visibility, evaluation calibration, cross-run learning, `memory_thrashing` detection | Nothing degrades |
| **Non-Claude agent** (e.g., via LangChain adapter) | IT metrics, self-calibrating baselines, anomaly detection — same analysis, different input | No OpenViking memory (unless separately integrated), no harness phases |

---

## Known Limitations

1. **Compression at 1.0 for short sessions.** The LZ76 phrase parser needs ~50+ events to produce meaningful compression ratios. For the first 20-30 events, compression reads as ~1.0 (incompressible), which can suppress `execution_regression` and `stagnation` signatures early in a session.

2. **Stall detection is the weakest failure type.** TPR is 76.9% at z=3.0 (vs 100% for loop/drift/thrash). Stalls produce subtle IT signatures because the agent is active (diverse tools, normal entropy) but not making progress — progress is a semantic property that IT measures can't directly observe. Raising the z-threshold to 2.5 catches all stalls but increases FPR to 15%.

3. **FPR on complex tasks.** Clean traces for complex tasks (games, full-stack apps) can trigger `distributional_anomaly` when the agent legitimately shifts strategies mid-task. The z=3.0 default balances this, but operators monitoring complex builds may want z=3.5.

4. **Phase-conditional baselines need calibration data from each phase.** If the calibration period (first 100 events) falls entirely within one phase (e.g., GATHERING), the baseline has no data for other phases and falls back to global. This is handled gracefully but reduces sensitivity.

5. **Evaluation uses synthetic traces.** The 65-trace evaluation set is generated by `eval/trace_generator.py`, not from real Claude Code sessions. Synthetic traces model realistic tool call patterns but may not capture all real-world failure modes. Prospective validation on real sessions (via `--log-validation`) is the path to real-world evidence.

6. **Window size tradeoff.** The SymbolStream uses a 50-event window (with 150 for LZ compression). Larger windows detect more subtle anomalies but are slower to respond. Smaller windows respond faster but are noisier. The 50/150 split is a tuned compromise.

7. **Memory metrics require 10+ events.** The `_MemoryOpsTracker` only contributes `namespace_entropy` and `memory_escalation_rate` to the baseline after 10 memory operations. This prevents false positives from zero-variance calibration when early events have no memory operations, but means `memory_thrashing` can't be detected in the first ~10 memory ops.
