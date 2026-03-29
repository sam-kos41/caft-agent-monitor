"""Terminal dashboard UI for real-time agent monitoring.

Uses rich.live to render a 3-panel layout that updates on every event:
  - Left:   Live action stream (tool calls with phase colors)
  - Right:  HTA phase tree with progress bar
  - Bottom: CAFT diagnostic warnings

Usage:
    agentdiag monitor --input stdin
    cat trace.jsonl | agentdiag monitor --input stdin
"""

from __future__ import annotations

import json
import sys
import time
from typing import IO, Optional

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress_bar import ProgressBar
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from agentdiag.hta import Phase
from agentdiag.caft.base import CaftDiagnosis, CaftSeverity
from agentdiag.monitor import MonitorEngine, DashboardState, ActionEntry


# Phase colors for rich markup
_PHASE_COLORS = {
    Phase.IDLE: "dim",
    Phase.GATHERING: "cyan",
    Phase.PLANNING: "yellow",
    Phase.EXECUTING: "green",
    Phase.VERIFYING: "magenta",
    Phase.DELIVERING: "blue",
}

_SEVERITY_COLORS = {
    CaftSeverity.INFO: "dim",
    CaftSeverity.WARNING: "yellow",
    CaftSeverity.CRITICAL: "red bold",
}

_HEALTH_COLORS = {
    "healthy": "green",
    "degraded": "yellow",
    "failing": "red bold",
}


def _build_action_table(actions: list[ActionEntry], max_rows: int = 20) -> Table:
    """Build the live action stream table."""
    table = Table(
        title="Live Actions",
        expand=True,
        show_header=True,
        header_style="bold",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Phase", width=10)
    table.add_column("Tool/Type", ratio=1)
    table.add_column("ms", width=6, justify="right")
    table.add_column("", width=1)  # status indicator

    recent = actions[-max_rows:]
    for a in recent:
        color = _PHASE_COLORS.get(a.phase, "white")
        status = "[green].[/]" if a.success else "[red]X[/]"
        table.add_row(
            str(a.step),
            f"[{color}]{a.phase.label}[/]",
            a.tool,
            f"{a.latency_ms:.0f}" if a.latency_ms else "-",
            status,
        )

    return table


def _build_hta_panel(state: DashboardState) -> Panel:
    """Build the HTA phase tree panel."""
    text = Text()

    if state.hta_state is None:
        text.append("Waiting for events...", style="dim")
        return Panel(text, title="HTA Phase", border_style="cyan")

    hta = state.hta_state

    # Phase progression
    for phase in Phase:
        if phase == Phase.IDLE:
            continue
        color = _PHASE_COLORS.get(phase, "white")
        count = hta.phase_event_counts.get(phase.label, 0)
        is_current = phase == hta.current_phase

        marker = ">" if is_current else " "
        style = f"{color} bold" if is_current else color

        text.append(f" {marker} ", style=style)
        text.append(f"{phase.label.upper():<12}", style=style)
        text.append(f" ({count} events)\n", style="dim" if not is_current else style)

    # Progress
    text.append("\n")
    pct = state.progress_pct
    bar_width = 20
    filled = int(pct * bar_width)
    bar = "=" * filled + "-" * (bar_width - filled)
    text.append(f" Progress: [{bar}] {pct:.0%}\n", style="green")

    # Stats
    text.append(f"\n Events: {state.total_events}", style="dim")
    text.append(f"   Errors: {state.total_errors}", style="red" if state.total_errors else "dim")
    text.append(f"\n Rate: {state.events_per_minute:.1f}/min", style="dim")

    # Regressions
    regressions = hta.regression_count
    if regressions > 0:
        text.append(f"\n Regressions: {regressions}", style="yellow")

    # Trust
    health = state.health
    health_color = _HEALTH_COLORS.get(health, "white")
    text.append(f"\n\n Trust: {state.trust_score:.0%}", style=health_color)
    text.append(f"  ({health})", style=health_color)

    # Goal
    if hta.goal:
        text.append(f"\n\n Goal: {hta.goal[:50]}", style="dim italic")

    return Panel(text, title="HTA State", border_style="cyan")


def _build_caft_panel(diagnoses: list[CaftDiagnosis]) -> Panel:
    """Build the CAFT diagnostics panel."""
    if not diagnoses:
        return Panel(
            Text(" No failures detected", style="green"),
            title="CAFT Diagnostics",
            border_style="green",
        )

    table = Table(
        expand=True,
        show_header=True,
        header_style="bold",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("CAFT", width=5)
    table.add_column("Severity", width=9)
    table.add_column("Failure", width=20)
    table.add_column("Description", ratio=1)
    table.add_column("Step", width=5, justify="right")

    for d in diagnoses:
        sev_color = _SEVERITY_COLORS.get(d.severity, "white")
        table.add_row(
            d.caft_code,
            f"[{sev_color}]{d.severity.value.upper()}[/]",
            d.failure_name,
            d.description[:60] + ("..." if len(d.description) > 60 else ""),
            str(d.at_step),
        )

    worst = max(diagnoses, key=lambda d: _SEVERITY_RANK_NUM.get(d.severity, 0))
    border_color = "red" if worst.severity == CaftSeverity.CRITICAL else "yellow"
    return Panel(table, title=f"CAFT Diagnostics ({len(diagnoses)})", border_style=border_color)


_SEVERITY_RANK_NUM = {
    CaftSeverity.INFO: 0,
    CaftSeverity.WARNING: 1,
    CaftSeverity.CRITICAL: 2,
}


def _build_layout(state: DashboardState) -> Layout:
    """Build the full dashboard layout from current state."""
    layout = Layout()

    layout.split_column(
        Layout(name="top", ratio=3),
        Layout(name="bottom", size=8),
    )

    layout["top"].split_row(
        Layout(name="actions", ratio=3),
        Layout(name="hta", ratio=2),
    )

    # Populate panels
    layout["actions"].update(
        Panel(_build_action_table(state.actions), title="", border_style="dim")
    )
    layout["hta"].update(_build_hta_panel(state))
    layout["bottom"].update(_build_caft_panel(state.diagnoses))

    return layout


def _build_header(state: DashboardState) -> Text:
    """Build the header bar."""
    text = Text()
    text.append(" agentdiag monitor ", style="bold white on blue")
    text.append("  ")
    health_color = _HEALTH_COLORS.get(state.health, "white")
    text.append(f" {state.health.upper()} ", style=f"bold white on {health_color.split()[0]}")
    text.append(f"  events={state.total_events}", style="dim")
    text.append(f"  trust={state.trust_score:.0%}", style="dim")
    if state.diagnoses:
        text.append(f"  warnings={len(state.diagnoses)}", style="yellow")
    return text


def run_dashboard(
    goal: str = "",
    stream: IO[str] | None = None,
    refresh_rate: float = 4.0,
    context_store: "ContextStore | None" = None,
) -> DashboardState:
    """Run the live terminal dashboard.

    Reads JSONL from stream (default: stdin), renders a live rich dashboard.
    Returns the final DashboardState when input is exhausted.
    """
    if not HAS_RICH:
        print(
            "Error: rich library required for terminal dashboard.\n"
            "Install with: pip install rich",
            file=sys.stderr,
        )
        sys.exit(1)

    console = Console()
    engine = MonitorEngine(goal=goal, context_store=context_store)
    source = stream or sys.stdin

    if context_store is not None:
        engine.start_context_session(goal=goal, source="dashboard")

    with Live(
        _build_layout(engine.state),
        console=console,
        refresh_per_second=refresh_rate,
        screen=True,
    ) as live:
        for line in source:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            engine.push_raw(data)

            # Update display
            state = engine.state
            layout = _build_layout(state)
            # Add header above
            header = _build_header(state)
            full = Layout()
            full.split_column(
                Layout(header, size=1),
                layout,
            )
            live.update(full)

    if context_store is not None:
        engine.end_context_session()

    return engine.state


def run_dashboard_demo(
    scenario: str | None = None,
    delay: float = 0.3,
    refresh_rate: float = 8.0,
) -> DashboardState:
    """Run the rich dashboard with synthetic CAFT events — single process.

    This is the primary way to SEE the dashboard in action. No piping needed.
    Events are fed directly into the engine with timing delays.

    Args:
        scenario: Scenario name or None for all scenarios.
        delay: Seconds between events.
        refresh_rate: Rich refresh rate.

    Returns:
        Final DashboardState.
    """
    if not HAS_RICH:
        print(
            "Error: rich library required for terminal dashboard.\n"
            "Install with: pip install rich",
            file=sys.stderr,
        )
        sys.exit(1)

    from agentdiag.caft.synthetic import CAFT_GENERATORS

    if scenario and scenario != "all":
        scenarios = {scenario: CAFT_GENERATORS[scenario]}
    else:
        scenarios = CAFT_GENERATORS

    console = Console()
    engine = MonitorEngine(goal="CAFT Live Demo")
    global_step = 0

    with Live(
        _build_layout(engine.state),
        console=console,
        refresh_per_second=refresh_rate,
        screen=True,
    ) as live:
        for scenario_name, gen_fn in scenarios.items():
            events, expected = gen_fn()

            for event in events:
                global_step += 1
                event.step = global_step
                engine.push(event)

                # Update display
                state = engine.state
                layout = _build_layout(state)
                header = _build_header(state)
                # Add scenario indicator
                scenario_text = Text()
                scenario_text.append(f" Scenario: {scenario_name} ", style="bold")
                exp_str = ", ".join(sorted(expected)) if expected else "none (clean)"
                scenario_text.append(f" Expected: {exp_str}", style="dim")

                full = Layout()
                full.split_column(
                    Layout(header, size=1),
                    Layout(scenario_text, size=1),
                    layout,
                )
                live.update(full)
                time.sleep(delay)

            # Pause between scenarios
            time.sleep(delay * 3)

            # Reset engine for next scenario
            engine.reset()

    return engine.state


def run_plain_monitor(
    goal: str = "",
    stream: IO[str] | None = None,
    context_store: "ContextStore | None" = None,
) -> DashboardState:
    """Fallback plain-text monitor when rich is not available.

    Prints phase transitions and CAFT diagnoses to stdout.
    """
    engine = MonitorEngine(goal=goal, context_store=context_store)
    source = stream or sys.stdin

    if context_store is not None:
        engine.start_context_session(goal=goal, source="plain_monitor")
    last_phase = None

    for line in source:
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        new_diagnoses = engine.push_raw(data)
        state = engine.state

        # Print phase transitions
        if state.hta_state and state.hta_state.current_phase != last_phase:
            last_phase = state.hta_state.current_phase
            print(f"[PHASE] {last_phase.label.upper()} "
                  f"(progress={state.progress_pct:.0%}, "
                  f"trust={state.trust_score:.0%})")

        # Print new diagnoses
        for d in new_diagnoses:
            print(f"[CAFT {d.caft_code}] {d.severity.value.upper()}: "
                  f"{d.failure_name} — {d.description}")

    final = engine.state
    print(f"\n--- Monitor Summary ---")
    print(f"Events: {final.total_events}, Errors: {final.total_errors}")
    print(f"Health: {final.health}, Trust: {final.trust_score:.0%}")
    if final.diagnoses:
        print(f"Diagnoses: {len(final.diagnoses)}")
        for d in final.diagnoses:
            print(f"  [{d.caft_code}] {d.failure_name}: {d.description}")
    else:
        print("No failures detected.")

    if context_store is not None:
        engine.end_context_session()

    return final
