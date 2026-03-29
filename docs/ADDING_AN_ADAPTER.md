# Adding an Adapter

This guide walks through adding support for a new agent framework to CAFT. By the end, your framework's execution traces will flow through the full information-theoretic analysis pipeline and appear in the visualization dashboard.

## How Adapters Work

CAFT has two adapter layers:

1. **`TraceAdapter`** (legacy) — converts framework-specific data to `TraceEvent` objects. Used by the older `MonitorEngine` and evaluation pipeline.
2. **`AgentAdapter`** (current) — converts data to `ObservableEvent` objects. Used by `UniversalMonitor` and the visualization layer.

New adapters should implement the `AgentAdapter` interface. The `ObservableEvent` is the shared contract that the analysis layer consumes — your adapter's job is to translate your framework's output into these events.

## Step-by-Step

### 1. Create your adapter file

Create `agentdiag/adapters/myframework_adapter.py`:

```python
"""MyFramework adapter — converts MyFramework traces to ObservableEvent."""

from __future__ import annotations

from typing import Iterator

from agentdiag.observable import (
    ObservableEvent,
    EventType,
    tool_call_event,
    file_read_event,
    file_write_event,
)


class MyFrameworkAdapter:
    """Converts MyFramework execution logs to ObservableEvent streams."""

    name: str = "myframework"

    def parse(self, raw: dict) -> list[ObservableEvent]:
        """Convert a single raw log entry to ObservableEvent(s).

        Args:
            raw: One entry from your framework's log format.

        Returns:
            List of ObservableEvent (may be empty if entry is not actionable).
        """
        events: list[ObservableEvent] = []

        action_type = raw.get("action", "")
        step = raw.get("step", 0)
        timestamp = raw.get("ts", 0.0)

        if action_type == "tool_use":
            tool = raw.get("tool_name", "unknown")
            target = raw.get("file_path")

            # Classify: is this a read, write, or generic tool call?
            if tool in ("read_file", "search"):
                events.append(file_read_event(
                    step=step,
                    timestamp=timestamp,
                    path=target or "",
                    output_tokens=raw.get("output_length", 0) // 4,
                ))
            elif tool in ("write_file", "edit_file"):
                events.append(file_write_event(
                    step=step,
                    timestamp=timestamp,
                    path=target or "",
                    input_tokens=raw.get("input_length", 0) // 4,
                ))
            else:
                events.append(tool_call_event(
                    step=step,
                    timestamp=timestamp,
                    tool_name=tool,
                    target_path=target,
                    duration_ms=raw.get("duration_ms", 0.0),
                ))

        elif action_type == "reasoning":
            events.append(tool_call_event(
                step=step,
                timestamp=timestamp,
                tool_name="reasoning",
            ))

        return events

    def replay(self, source: str, **kwargs) -> list[ObservableEvent]:
        """Replay a full log file as an event stream."""
        import json
        events: list[ObservableEvent] = []
        with open(source) as f:
            for line in f:
                raw = json.loads(line.strip())
                events.extend(self.parse(raw))
        return events
```

### 2. Register in the adapter factory

Edit `agentdiag/adapters/__init__.py` and add your adapter to `get_adapter()`:

```python
def get_adapter(source: str) -> AgentAdapter:
    # ... existing adapters ...
    elif source == "myframework":
        from agentdiag.adapters.myframework_adapter import MyFrameworkAdapter
        return MyFrameworkAdapter()
    # ...
```

### 3. Use it

```python
from agentdiag.adapters import get_adapter
from agentdiag.universal_monitor import UniversalMonitor

adapter = get_adapter("myframework")
monitor = UniversalMonitor()

events = adapter.replay("myframework_session.jsonl")
for event in events:
    result = monitor.process(event)
    if result.get("anomalies"):
        print(f"Anomaly at step {event.step}: {result['anomalies']['signature']}")

# Or with the visualization:
# python -m agentdiag live --session myframework_session.jsonl
```

## What Fields Matter

The `ObservableEvent` has many optional fields. Here's what actually drives the analysis:

### Critical (the analysis breaks without these)

| Field | Why It Matters |
|-------|---------------|
| `step` | Orders events chronologically. Must be monotonically increasing. |
| `timestamp` | Used for replay pacing and latency computation. Epoch seconds. |
| `event_type` | Determines routing: which `SymbolStream` receives the event, whether it's a phase marker, whether it goes to memory tracking. |

### Important (significantly improves detection quality)

| Field | Why It Matters |
|-------|---------------|
| `tool_name` | Becomes part of the symbol via `to_symbol()` → `"tool:Read"`. Tool diversity directly drives entropy and MI. |
| `target_path` | File reads/writes use this for `"read:main.py"` symbols. Without it, all reads look identical to the SymbolStream. |

### Optional (enhances specific features)

| Field | Feature It Enables |
|-------|-------------------|
| `input_tokens` / `output_tokens` | Token explosion detection, working memory utilization estimate |
| `duration_ms` | Latency trend tracking in the cognitive state model |
| `viking_uri` / `memory_tier` / `namespace` | Memory operations track, tier distribution, `memory_thrashing` detection (only relevant if your framework uses OpenViking) |
| `phase` / `agent_role` / `sprint_number` | Harness phase boundaries, pipeline strip role mapping (only relevant for multi-agent orchestration) |

### How `to_symbol()` Works

The analysis layer never inspects event internals directly. It calls `event.to_symbol()` which produces a string:

| Event Type | Symbol Format | Example |
|------------|--------------|---------|
| `TOOL_CALL` | `tool:{tool_name}` | `tool:Read` |
| `FILE_READ` | `read:{last_2_path_components}` | `read:src/main.py` |
| `FILE_WRITE` | `write:{last_2_path_components}` | `write:src/main.py` |
| `SHELL_COMMAND` | `shell:{tool_name}` | `shell:bash` |
| `MEMORY_LOAD` | `mem_load:{tier}:{namespace}` | `mem_load:l2:agent/generator/skills` |
| `MEMORY_STORE` | `mem_store:{namespace}` | `mem_store:resources/current_project` |
| `PHASE_BOUNDARY` | `phase:{phase}` | `phase:executing` |

The `SymbolStream` computes entropy over the distribution of these strings. More distinct symbols = higher entropy. Repeated identical symbols = lower entropy. The symbol encoding determines what the IT measures capture: if your adapter produces detailed symbols (`tool:Read:src/auth.py`), entropy will be sensitive to file-level patterns; if it produces coarse symbols (`tool:Read`), entropy only captures tool-level diversity.

## Example: Minimal 30-Line Adapter

For a framework that logs `{"action": "read", "file": "main.py", "step": 1}`:

```python
from agentdiag.observable import (
    ObservableEvent, EventType, tool_call_event, file_read_event, file_write_event,
)

_READ_ACTIONS = {"read", "search", "grep", "find", "list"}
_WRITE_ACTIONS = {"write", "edit", "create", "patch"}

class MinimalAdapter:
    name = "minimal"

    def parse(self, raw: dict) -> list[ObservableEvent]:
        action = raw.get("action", "")
        step = raw.get("step", 0)
        ts = raw.get("timestamp", 0.0)
        path = raw.get("file", raw.get("target", ""))

        if action in _READ_ACTIONS:
            return [file_read_event(step=step, timestamp=ts, path=path)]
        if action in _WRITE_ACTIONS:
            return [file_write_event(step=step, timestamp=ts, path=path)]
        if action:
            return [tool_call_event(step=step, timestamp=ts, tool_name=action, target_path=path)]
        return []

    def replay(self, source: str, **kwargs) -> list[ObservableEvent]:
        import json
        events = []
        with open(source) as f:
            for line in f:
                events.extend(self.parse(json.loads(line.strip())))
        return events
```

## Testing Your Adapter

### Unit test: adapter produces valid events

```python
def test_my_adapter_produces_events():
    adapter = MyFrameworkAdapter()
    raw = {"action": "tool_use", "tool_name": "read_file", "file_path": "main.py", "step": 1, "ts": 1.0}
    events = adapter.parse(raw)
    assert len(events) == 1
    assert events[0].event_type == EventType.FILE_READ
    assert events[0].to_symbol() == "read:main.py"
```

### Integration test: events flow through the full pipeline

```python
def test_my_adapter_through_monitor():
    from agentdiag.universal_monitor import UniversalMonitor

    adapter = MyFrameworkAdapter()
    monitor = UniversalMonitor()

    events = adapter.replay("test_trace.jsonl")
    assert len(events) > 50  # need enough for calibration

    for event in events:
        monitor.process(event)

    state = monitor.get_state()
    assert state["info_theoretic"]["tool_entropy"] > 0.0
    assert not monitor.is_calibrating
```

### Evaluation test: run through the eval framework

```bash
# Generate a trace in your framework's format, then convert:
python -c "
from agentdiag.adapters.myframework_adapter import MyFrameworkAdapter
from agentdiag.eval.runner import run_trace
import json

adapter = MyFrameworkAdapter()
events = adapter.replay('my_trace.jsonl')

# Write as CAFT-format JSONL for the eval runner
with open('converted_trace.jsonl', 'w') as f:
    for e in events:
        f.write(json.dumps(e.to_dict(), default=str) + '\n')
"

# Run through evaluation
python -m agentdiag.eval.runner --trace converted_trace.jsonl
```

## Common Pitfalls

1. **Don't hard-code `event.tool` or `event.type` in the analysis layer.** The analysis layer uses `event.to_symbol()` exclusively. If your adapter sets the right `event_type` and `tool_name`, the symbols are generated correctly.

2. **Classify reads and writes correctly.** The `EventRouter` dispatches `FILE_READ` events to the `read_stream` and `FILE_WRITE` to the `write_stream`. If all your events are `TOOL_CALL`, the read/write streams stay empty and read-heavy anomalies won't be detected.

3. **Keep steps monotonically increasing.** The baseline and compositor use step numbers for temporal windowing. Non-monotonic steps will confuse anomaly localization.

4. **Provide `target_path` when you have it.** Without file paths, all reads produce the symbol `read:unknown`, collapsing diversity to zero. Even partial paths help: `target_path="auth.py"` is better than nothing.

5. **Don't emit phase markers unless your framework has real phase transitions.** Only emit `PHASE_BOUNDARY` events if your framework genuinely transitions between planning/executing/verifying stages. Spurious phase boundaries will fragment the baseline calibration.
