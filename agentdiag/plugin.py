"""
CAFT Plugin Interface — zero-config monitoring for any coding agent.

Quick start:

    # CLI (auto-detects running agents)
    caft monitor
    caft monitor --agent claude-code
    caft monitor --traces /path/to/traces/
    caft analyze /path/to/session.jsonl
    caft audit /path/to/traces/ --output report.txt

    # Python API
    from agentdiag.plugin import CAFT

    caft = CAFT()
    caft.monitor()                          # Auto-detect and watch
    caft.analyze("session.jsonl")           # Analyze a single session
    caft.audit("/path/to/traces/")          # Batch audit, produce report

    # Streaming API (for integration into your own agent framework)
    from agentdiag.plugin import CAFT, Event

    caft = CAFT()
    caft.start()
    caft.feed(Event.tool_call("Read", target="server.py"))
    caft.feed(Event.tool_call("Write", target="models.py"))
    caft.feed(Event.shell("pytest tests/"))
    state = caft.status()       # {"health": "green", "anomalies": 0, ...}
    report = caft.stop()        # Final report

    # Webhook/callback API
    caft = CAFT(on_anomaly=lambda a: slack.post(a["message"]))
    caft.monitor()
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from agentdiag.observable import EventType, ObservableEvent
from agentdiag.universal_monitor import UniversalMonitor


# ---------------------------------------------------------------------------
# Event builder — simple API for feeding events
# ---------------------------------------------------------------------------

class Event:
    """Convenience constructors for ObservableEvent.

    Usage:
        Event.tool_call("Read", target="server.py")
        Event.file_read("models.py", tokens=500)
        Event.file_write("output.py", tokens=200)
        Event.shell("pytest tests/", success=True)
        Event.shell("npm run build", success=False)
    """

    _step = 0

    @classmethod
    def _next_step(cls) -> int:
        cls._step += 1
        return cls._step

    @classmethod
    def reset(cls):
        cls._step = 0

    @classmethod
    def tool_call(
        cls, tool: str, target: str = "", tokens_in: int = 0,
        tokens_out: int = 0, duration_ms: float = 0.0,
    ) -> ObservableEvent:
        tool_lower = tool.lower()
        if tool_lower in ("read", "grep", "glob", "search", "find", "cat"):
            etype = EventType.FILE_READ
        elif tool_lower in ("write", "edit", "multiedit", "patch"):
            etype = EventType.FILE_WRITE
        elif tool_lower in ("bash", "shell", "terminal", "exec", "run"):
            etype = EventType.SHELL_COMMAND
        else:
            etype = EventType.TOOL_CALL

        return ObservableEvent(
            step=cls._next_step(),
            timestamp=time.time(),
            event_type=etype,
            tool_name=tool,
            target_path=target or None,
            input_tokens=tokens_in or None,
            output_tokens=tokens_out or None,
            duration_ms=duration_ms or None,
        )

    @classmethod
    def file_read(cls, path: str, tokens: int = 0) -> ObservableEvent:
        return cls.tool_call("Read", target=path, tokens_out=tokens)

    @classmethod
    def file_write(cls, path: str, tokens: int = 0) -> ObservableEvent:
        return cls.tool_call("Write", target=path, tokens_in=tokens)

    @classmethod
    def shell(
        cls, command: str, success: bool = True, duration_ms: float = 0.0,
    ) -> ObservableEvent:
        e = cls.tool_call("Bash", target=command[:200], duration_ms=duration_ms)
        if not success:
            e.event_type = EventType.ERROR
        return e


# ---------------------------------------------------------------------------
# Agent detection
# ---------------------------------------------------------------------------

@dataclass
class AgentSource:
    """A discovered agent trace source."""
    name: str                   # "claude-code", "codex", "cursor", etc.
    path: Path                  # Path to trace file or directory
    format: str                 # "jsonl", "json", "sqlite"
    session_count: int = 0
    total_size_kb: int = 0


def detect_agents() -> list[AgentSource]:
    """Auto-detect running or recent agent trace sources.

    Checks common locations for:
    - Claude Code: ~/.claude/projects/*/*.jsonl
    - Codex/OpenAI: ~/.codex/ or OpenAI traces
    - Cursor: ~/.cursor/ or cursor logs
    - Custom: CAFT_TRACES env var
    """
    sources = []
    home = Path.home()

    # Claude Code
    claude_projects = home / ".claude" / "projects"
    if claude_projects.exists():
        for project_dir in claude_projects.iterdir():
            if project_dir.is_dir():
                jsonl_files = list(project_dir.glob("*.jsonl"))
                if jsonl_files:
                    total_kb = sum(f.stat().st_size for f in jsonl_files) // 1024
                    sources.append(AgentSource(
                        name="claude-code",
                        path=project_dir,
                        format="jsonl",
                        session_count=len(jsonl_files),
                        total_size_kb=total_kb,
                    ))

    # Codex CLI (OpenAI)
    codex_dir = home / ".codex"
    if codex_dir.exists():
        json_files = list(codex_dir.rglob("*.json")) + list(codex_dir.rglob("*.jsonl"))
        if json_files:
            sources.append(AgentSource(
                name="codex",
                path=codex_dir,
                format="json",
                session_count=len(json_files),
                total_size_kb=sum(f.stat().st_size for f in json_files) // 1024,
            ))

    # Cursor
    for cursor_path in [
        home / ".cursor",
        home / "Library" / "Application Support" / "Cursor",
        home / ".config" / "Cursor",
    ]:
        if cursor_path.exists():
            log_files = list(cursor_path.rglob("*.jsonl")) + list(cursor_path.rglob("*.log"))
            if log_files:
                sources.append(AgentSource(
                    name="cursor",
                    path=cursor_path,
                    format="jsonl",
                    session_count=len(log_files),
                    total_size_kb=sum(f.stat().st_size for f in log_files) // 1024,
                ))
                break

    # Aider
    aider_logs = list(Path.cwd().glob(".aider.logs/*.jsonl"))
    if aider_logs:
        sources.append(AgentSource(
            name="aider",
            path=Path.cwd() / ".aider.logs",
            format="jsonl",
            session_count=len(aider_logs),
            total_size_kb=sum(f.stat().st_size for f in aider_logs) // 1024,
        ))

    # Custom: CAFT_TRACES env var
    custom = os.environ.get("CAFT_TRACES")
    if custom:
        custom_path = Path(custom)
        if custom_path.exists():
            traces = list(custom_path.rglob("*.jsonl")) + list(custom_path.rglob("*.json"))
            if traces:
                sources.append(AgentSource(
                    name="custom",
                    path=custom_path,
                    format="jsonl",
                    session_count=len(traces),
                    total_size_kb=sum(f.stat().st_size for f in traces) // 1024,
                ))

    return sources


# ---------------------------------------------------------------------------
# CAFT — main plug-in class
# ---------------------------------------------------------------------------

class CAFT:
    """Zero-config monitoring for AI coding agents.

    Usage:
        caft = CAFT()
        caft.monitor()              # Watch running agents
        caft.analyze("session.jsonl")  # Analyze one session
        caft.audit("/traces/")      # Batch audit
    """

    def __init__(
        self,
        sensitivity: float = 2.0,
        on_anomaly: Optional[Callable[[dict], None]] = None,
        on_status_change: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Args:
            sensitivity: Z-score threshold for anomaly detection (lower = more sensitive).
            on_anomaly: Callback when anomaly detected. Receives anomaly dict.
            on_status_change: Callback when health status changes. Receives (old, new).
        """
        self.sensitivity = sensitivity
        self.on_anomaly = on_anomaly
        self.on_status_change = on_status_change
        self._monitor: Optional[UniversalMonitor] = None
        self._health = "unknown"
        self._anomaly_count = 0
        self._event_count = 0
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Start a monitoring session for the streaming API."""
        self._monitor = UniversalMonitor(sensitivity=self.sensitivity)
        self._health = "green"
        self._anomaly_count = 0
        self._event_count = 0
        self._start_time = time.time()
        Event.reset()

    def feed(self, event: ObservableEvent) -> dict:
        """Feed a single event and get the result.

        Returns:
            dict with keys: health, anomalies, metrics, event_count
        """
        if self._monitor is None:
            self.start()

        result = self._monitor.process(event)
        self._event_count += 1

        # Check for anomalies
        if result and result.get("anomalies"):
            self._anomaly_count += 1
            old_health = self._health
            self._health = self._compute_health()
            if self.on_anomaly:
                self.on_anomaly(result["anomalies"])
            if old_health != self._health and self.on_status_change:
                self.on_status_change(old_health, self._health)

        return {
            "health": self._health,
            "anomaly_count": self._anomaly_count,
            "event_count": self._event_count,
            "metrics": result.get("metrics") if result else None,
            "anomalies": result.get("anomalies") if result else None,
        }

    def status(self) -> dict:
        """Get current monitoring status."""
        if self._monitor is None:
            return {"health": "not_started", "event_count": 0}

        state = self._monitor.get_state()
        return {
            "health": self._health,
            "event_count": self._event_count,
            "anomaly_count": self._anomaly_count,
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "info_theoretic": {
                k: round(v, 3) if isinstance(v, float) else v
                for k, v in state.get("info_theoretic", {}).items()
                if k in ("action_mi", "tool_entropy", "kl_divergence",
                         "compression_ratio", "last_surprisal")
            },
            "current_phase": state.get("current_phase"),
        }

    def stop(self) -> dict:
        """Stop monitoring and return final report."""
        if self._monitor is None:
            return {"error": "Not started"}

        state = self._monitor.get_state()
        report = {
            "health": self._health,
            "event_count": self._event_count,
            "anomaly_count": self._anomaly_count,
            "duration_seconds": time.time() - self._start_time if self._start_time else 0,
            "state": state,
        }
        self._monitor = None
        return report

    # ------------------------------------------------------------------
    # High-level commands
    # ------------------------------------------------------------------

    def analyze(self, path: str, verbose: bool = False) -> dict:
        """Analyze a single session file and return its IP profile.

        Args:
            path: Path to a JSONL/JSON trace file.
            verbose: Print progress to stdout.

        Returns:
            dict with session profile, anomalies, and health assessment.
        """
        try:
            from agentdiag.plugin_analyze import replay_and_profile
            return replay_and_profile(path, self.sensitivity, verbose)
        except ImportError:
            pass

        # Inline implementation (the standard path)
        return self._analyze_session(path, verbose)

    def _analyze_session(self, path: str, verbose: bool = False) -> dict:
        """Analyze a session using the core pipeline."""
        monitor = UniversalMonitor(sensitivity=self.sensitivity)
        events_processed = 0
        anomalies = []

        # Try to parse as Claude Code JSONL
        try:
            from agentdiag.live import _extract_trace_events_from_cc
            from agentdiag.cognitive import trace_event_to_observable
            from agentdiag.models import TraceEvent

            step_counter = [0]
            with open(path) as f:
                for line in f:
                    try:
                        raw = json.loads(line.strip())
                        extracted = _extract_trace_events_from_cc(raw, step_counter)
                        for te in extracted:
                            tev = TraceEvent(**{
                                k: te.get(k)
                                for k in TraceEvent.__dataclass_fields__
                                if k in te
                            })
                            obs = trace_event_to_observable(tev)
                            if obs:
                                result = monitor.process(obs)
                                events_processed += 1
                                if result and result.get("anomalies"):
                                    anomalies.append(result["anomalies"])
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
        except ImportError:
            pass

        # Fallback: direct JSONL parsing
        if events_processed == 0:
            with open(path) as f:
                for line in f:
                    try:
                        raw = json.loads(line.strip())
                        if "step" in raw and "tool" in raw:
                            tool = raw.get("tool", "unknown")
                            target = raw.get("target_path", "")
                            tl = tool.lower()
                            if tl in ("read", "grep", "glob"):
                                etype = EventType.FILE_READ
                            elif tl in ("write", "edit"):
                                etype = EventType.FILE_WRITE
                            elif tl in ("bash",):
                                etype = EventType.SHELL_COMMAND
                            else:
                                etype = EventType.TOOL_CALL

                            obs = ObservableEvent(
                                step=raw["step"],
                                timestamp=raw.get("timestamp", 0.0),
                                event_type=etype,
                                tool_name=tool,
                                target_path=str(target)[:200] if target else None,
                            )
                            result = monitor.process(obs)
                            events_processed += 1
                            if result and result.get("anomalies"):
                                anomalies.append(result["anomalies"])
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue

        state = monitor.get_state()
        it = state.get("info_theoretic", {})

        return {
            "path": path,
            "events": events_processed,
            "anomaly_count": len(anomalies),
            "behavioral_state": self._assess_behavioral_state(
                events_processed, len(anomalies), it),
            "metrics": {
                "action_mi": it.get("action_mi", 0),
                "tool_entropy": it.get("tool_entropy", 0),
                "kl_divergence": it.get("kl_divergence", 0),
                "compression_ratio": it.get("compression_ratio", 0),
            },
            "anomalies": anomalies,
        }

    def audit(
        self,
        path: str,
        output: Optional[str] = None,
        verbose: bool = True,
    ) -> dict:
        """Batch audit all sessions in a directory.

        Args:
            path: Directory containing trace files, or glob pattern.
            output: Optional path to save the report.
            verbose: Print progress.

        Returns:
            dict with per-session profiles and overall assessment.
        """
        import glob as glob_mod

        trace_path = Path(path)
        if trace_path.is_dir():
            files = sorted(trace_path.glob("*.jsonl"))
        elif "*" in path:
            files = sorted(Path(p) for p in glob_mod.glob(path))
        else:
            files = [trace_path]

        results = []
        for f in files:
            if verbose:
                print(f"  Analyzing {f.name}...", end=" ", flush=True)
            try:
                r = self._analyze_session(str(f))
                results.append(r)
                if verbose:
                    print(f"{r['events']} events, {r['anomaly_count']} anomalies, "
                          f"state={r['behavioral_state']}")
            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")

        # Summary
        if not results:
            return {"error": "No sessions analyzed"}

        def _count(state):
            return sum(1 for r in results
                       if r.get("behavioral_state") == state)

        summary = {
            "sessions": len(results),
            "steady": _count("steady"),
            "phase_shifting": _count("phase_shifting"),
            "looping": _count("looping"),
            "total_events": sum(r["events"] for r in results),
            "total_anomalies": sum(r["anomaly_count"] for r in results),
            "results": results,
        }

        if output:
            Path(output).write_text(json.dumps(summary, indent=2, default=str))
            if verbose:
                print(f"\nReport saved to {output}")

        return summary

    def monitor(
        self,
        agent: Optional[str] = None,
        traces: Optional[str] = None,
        port: int = 8080,
        dashboard: bool = True,
    ) -> None:
        """Start live monitoring.

        Args:
            agent: Agent type to monitor ("claude-code", "codex", "cursor").
                   If None, auto-detects.
            traces: Path to trace directory. If None, auto-detects.
            port: Dashboard port.
            dashboard: Whether to open the web dashboard.
        """
        # Find traces
        if traces:
            trace_path = traces
        elif agent:
            sources = detect_agents()
            matching = [s for s in sources if s.name == agent]
            if not matching:
                print(f"No {agent} sessions found. Detected agents:")
                for s in sources:
                    print(f"  {s.name}: {s.session_count} sessions at {s.path}")
                return
            trace_path = str(matching[0].path)
        else:
            sources = detect_agents()
            if not sources:
                print("No agent traces detected. Specify --traces or set CAFT_TRACES.")
                return
            # Pick the most recent source
            source = max(sources, key=lambda s: s.path.stat().st_mtime
                        if s.path.exists() else 0)
            trace_path = str(source.path)
            print(f"Auto-detected: {source.name} ({source.session_count} sessions)")

        # Delegate to the live module
        from agentdiag.live import main as live_main
        import sys

        args = ["live", "--project", trace_path, "--all-sessions",
                "--port", str(port)]
        if not dashboard:
            args.append("--no-browser")

        # Patch sys.argv and run
        old_argv = sys.argv
        sys.argv = ["agentdiag"] + args
        try:
            live_main()
        finally:
            sys.argv = old_argv

    # ------------------------------------------------------------------
    # Health assessment
    # ------------------------------------------------------------------

    def _compute_health(self) -> str:
        """Compute health from anomaly rate."""
        if self._event_count < 10:
            return "green"
        rate = self._anomaly_count / self._event_count
        if rate > 0.15:
            return "red"
        elif rate > 0.05:
            return "yellow"
        return "green"

    @staticmethod
    def _assess_behavioral_state(events: int, anomalies: int,
                                 metrics: dict) -> str:
        """Describe the session's behavioral state from IT metrics.

        DESCRIPTIVE, NOT EVALUATIVE. These labels say what the math
        computed, not whether the session was good or bad. See
        docs/CONSTRUCT_REVISION.md — the prior red/yellow/green "health"
        verdict was not supported by the math (kappa = -0.04 vs a
        domain expert) and has been retired.

          looping        — repetition dominates (high LZ compressibility
                            or sustained low action-MI)
          phase_shifting — the action distribution keeps changing
                            (high KL / high within-session deviation)
          steady         — neither; stable, varied flow

        No state implies a quality judgment. A long healthy research
        session is legitimately phase_shifting; a focused grep sweep is
        legitimately looping. Interpretation is the consumer's job.
        """
        if events == 0:
            return "unknown"
        rate = anomalies / events if events > 0 else 0
        mi = metrics.get("action_mi", 0)
        kl = metrics.get("kl_divergence", 0)
        comp = metrics.get("compression_ratio", 1.0)

        if comp < 0.7 or mi < 0.4:
            return "looping"
        if kl > 0.45 or rate > 0.15:
            return "phase_shifting"
        return "steady"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli():
    """CLI entry point: caft monitor|analyze|audit|detect"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="caft",
        description="CAFT: Zero-config anomaly detection for AI coding agents",
    )
    sub = parser.add_subparsers(dest="command")

    # caft detect
    p_detect = sub.add_parser("detect", help="Auto-detect agent trace sources")

    # caft monitor
    p_monitor = sub.add_parser("monitor", help="Live-monitor running agents")
    p_monitor.add_argument("--agent", help="Agent type (claude-code, codex, cursor)")
    p_monitor.add_argument("--traces", help="Path to trace directory")
    p_monitor.add_argument("--port", type=int, default=8080, help="Dashboard port")
    p_monitor.add_argument("--no-dashboard", action="store_true")

    # caft analyze
    p_analyze = sub.add_parser("analyze", help="Analyze a single session")
    p_analyze.add_argument("path", help="Path to session JSONL file")
    p_analyze.add_argument("--sensitivity", type=float, default=2.0)

    # caft audit
    p_audit = sub.add_parser("audit", help="Batch audit all sessions in a directory")
    p_audit.add_argument("path", help="Directory with trace files")
    p_audit.add_argument("--output", "-o", help="Save report (.txt or .pdf)")
    p_audit.add_argument("--sensitivity", type=float, default=2.0)
    p_audit.add_argument("--team-size", type=int, default=1, help="Number of developers")
    p_audit.add_argument("--hourly-rate", type=float, default=75.0, help="Loaded engineering cost/hr")
    p_audit.add_argument("--sessions-per-week", type=int, default=0, help="Estimated sessions/week")
    p_audit.add_argument("--company", default="Agent Audit", help="Company name for report header")

    # caft dashboard
    p_dash = sub.add_parser("dashboard", help="Open the supervisor dashboard")
    p_dash.add_argument("--port", type=int, default=8080, help="Dashboard port")
    p_dash.add_argument("--traces", help="Path to trace directory")
    p_dash.add_argument("--agent", help="Agent type (claude-code, codex, cursor)")

    p_val = sub.add_parser(
        "validate",
        help="Inter-rater validation harness (human + Ollama + CAFT)")
    p_val.add_argument("corpus", help="Directory of session JSONL files")
    p_val.add_argument("--ledger", default="validation_ledger.jsonl",
                       help="Path to ratings ledger (created if missing)")
    p_val.add_argument("--human-id", default="default",
                       help="Identifier for the human rater")
    p_val.add_argument("--ollama-model", default="llama3.2:3b",
                       help="Ollama model name")
    p_val.add_argument("--port", type=int, default=8090,
                       help="Rating UI port")
    p_val.add_argument("--report-only", action="store_true",
                       help="Just write the markdown report and exit")
    p_val.add_argument("--auto-rate", action="store_true",
                       help="Run CAFT+Ollama on every session, then exit "
                            "(no UI) — pre-populates the ledger")

    args = parser.parse_args()

    if args.command == "detect":
        sources = detect_agents()
        if not sources:
            print("No agent traces detected.")
            print("Set CAFT_TRACES=/path/to/traces or specify --traces")
            return
        print(f"Detected {len(sources)} agent source(s):\n")
        for s in sources:
            print(f"  {s.name}")
            print(f"    Path:     {s.path}")
            print(f"    Sessions: {s.session_count}")
            print(f"    Size:     {s.total_size_kb}KB")
            print()

    elif args.command == "monitor":
        caft = CAFT()
        caft.monitor(
            agent=args.agent,
            traces=args.traces,
            port=args.port,
            dashboard=not args.no_dashboard,
        )

    elif args.command == "analyze":
        caft = CAFT(sensitivity=args.sensitivity)
        result = caft._analyze_session(args.path, verbose=True)
        print(f"\nBehavioral state: {result['behavioral_state']} "
              f"(descriptive, not a quality verdict)")
        print(f"Events: {result['events']}")
        print(f"Anomalies: {result['anomaly_count']}")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    elif args.command == "audit":
        caft = CAFT(sensitivity=args.sensitivity)
        summary = caft.audit(args.path, verbose=True)

        output = args.output
        if output and output.endswith(".pdf"):
            try:
                from agentdiag.audit_pdf import export_pdf
                export_pdf(
                    summary, output,
                    team_size=args.team_size,
                    sessions_per_week=args.sessions_per_week,
                    hourly_rate=args.hourly_rate,
                    company_name=args.company,
                )
                print(f"\nPDF report saved to {output}")
            except ImportError:
                print("\nPDF export requires: pip install reportlab")
                print("Falling back to text report...")
                output = output.replace(".pdf", ".txt")
        elif output and output.endswith(".txt"):
            from agentdiag.audit_report import generate_audit_report
            report = generate_audit_report(
                summary,
                team_size=args.team_size,
                sessions_per_week=args.sessions_per_week,
                hourly_rate=args.hourly_rate,
                company_name=args.company,
            )
            Path(output).write_text(report)
            print(f"\nText report saved to {output}")
        elif output:
            Path(output).write_text(json.dumps(summary, indent=2, default=str))
            print(f"\nJSON saved to {output}")

        # Always print summary to stdout
        from agentdiag.audit_report import generate_audit_report
        print()
        print(generate_audit_report(
            summary,
            team_size=args.team_size,
            sessions_per_week=args.sessions_per_week,
            hourly_rate=args.hourly_rate,
            company_name=args.company,
        ))

    elif args.command == "dashboard":
        # Serve the React supervisor dashboard
        caft = CAFT()
        caft.monitor(
            agent=getattr(args, "agent", None),
            traces=getattr(args, "traces", None),
            port=args.port,
            dashboard=True,
        )

    elif args.command == "validate":
        from pathlib import Path
        from agentdiag.validation.digest import build_digest
        from agentdiag.validation.ledger import Ledger
        from agentdiag.validation.rate_caft import rate_with_caft
        from agentdiag.validation.rate_ollama import (
            rate_with_ollama, is_ollama_available,
        )
        from agentdiag.validation.signals import rate_with_signals
        from agentdiag.validation.report import write_report

        corpus = Path(args.corpus)
        if not corpus.exists():
            print(f"Corpus not found: {corpus}")
            return
        ledger = Ledger(args.ledger)
        report_path = Path(args.ledger).with_suffix(".report.md")

        if args.report_only:
            write_report(ledger, report_path)
            print(f"Report written to {report_path}")
            return

        if args.auto_rate:
            sessions = sorted(corpus.glob("*.jsonl"))
            ollama_up = is_ollama_available()
            print(f"Auto-rating {len(sessions)} sessions "
                  f"(Ollama: {'UP' if ollama_up else 'DOWN — skipping'})")
            for p in sessions:
                d = build_digest(p)
                rows = rate_with_caft(d)
                # deterministic programmatic ground truth (no judgment)
                rows += rate_with_signals(p)
                tag = "caft+signal"
                if ollama_up:
                    try:
                        rows += rate_with_ollama(d, model=args.ollama_model)
                        tag = "caft+ollama"
                    except Exception as e:
                        print(f"  {d.session_id}: ollama failed ({e})")
                ledger.append_many(rows)
                print(f"  {d.session_id}: {len(rows)} ratings ({tag})")
            write_report(ledger, report_path)
            print(f"\nReport written to {report_path}")
            print(f"Now rate them yourself:  caft validate {args.corpus} "
                  f"--ledger {args.ledger} --human-id {args.human_id}")
            return

        # Launch the web UI
        from agentdiag.validation import server as val_server
        val_server._corpus_root = corpus.resolve()
        val_server._ledger = ledger
        val_server._human_id = args.human_id
        val_server._ollama_model = args.ollama_model
        import uvicorn
        print(f"Corpus: {corpus.resolve()}  ({len(list(corpus.glob('*.jsonl')))} sessions)")
        print(f"Ledger: {ledger.path}")
        print(f"Ollama: {args.ollama_model} "
              f"({'UP' if is_ollama_available() else 'DOWN'})")
        print(f"Open:   http://127.0.0.1:{args.port}/")
        app = val_server._make_app()
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")

    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
