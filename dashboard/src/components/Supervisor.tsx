import type { SupervisorState, SessionState } from "../types";
import { AgentCard } from "./AgentCard";
import { TimelineBar } from "./TimelineBar";
import { AlertFeed } from "./AlertFeed";
import { CoordinationGraph } from "./CoordinationGraph";

function formatUptime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  if (m < 1) return `${Math.floor(seconds)}s`;
  const h = Math.floor(m / 60);
  if (h < 1) return `${m}m`;
  return `${h}h ${m % 60}m`;
}

export function Supervisor({
  state,
  onSelectSession,
}: {
  state: SupervisorState;
  onSelectSession?: (session: SessionState) => void;
}) {
  const { sessions, alerts, uptimeSeconds, connected } = state;

  const totalAgents = sessions.length;
  const problemCount = sessions.filter(
    (s) => s.health === "red" || s.health === "yellow"
  ).length;

  return (
    <div className="supervisor">
      {/* Header */}
      <header className="supervisor__header">
        <div className="supervisor__title">
          <h1>CAFT Supervisor</h1>
          <span className={`connection-dot ${connected ? "connected" : "disconnected"}`} />
        </div>
        <div className="supervisor__meta">
          <span>{totalAgents} agent{totalAgents !== 1 ? "s" : ""}</span>
          <span className="separator">|</span>
          <span>{formatUptime(uptimeSeconds)}</span>
          {problemCount > 0 && (
            <>
              <span className="separator">|</span>
              <span className="problem-count">{problemCount} need attention</span>
            </>
          )}
        </div>
      </header>

      {/* Agent Cards */}
      <section className="supervisor__cards">
        {sessions.length === 0 ? (
          <div className="supervisor__empty">
            {connected
              ? "Waiting for agent events..."
              : "Connecting to CAFT server..."}
          </div>
        ) : (
          sessions.map((session) => (
            <AgentCard
              key={session.sessionId}
              session={session}
              onClick={() => onSelectSession?.(session)}
            />
          ))
        )}
      </section>

      {/* Timeline Bars */}
      {sessions.length > 0 && (
        <section className="supervisor__timelines">
          <h2>Timeline</h2>
          <div className="timeline-legend">
            <span className="legend-item"><span className="legend-swatch" style={{ background: "#22c55e" }} /> writing</span>
            <span className="legend-item"><span className="legend-swatch" style={{ background: "#86efac" }} /> reading</span>
            <span className="legend-item"><span className="legend-swatch" style={{ background: "#3b82f6" }} /> running</span>
            <span className="legend-item"><span className="legend-swatch" style={{ background: "#ef4444" }} /> anomaly</span>
            <span className="legend-item"><span className="legend-swatch" style={{ background: "#eab308" }} /> warning</span>
          </div>
          {sessions.map((session) => (
            <TimelineBar key={session.sessionId} session={session} />
          ))}
        </section>
      )}

      {/* Cross-Agent Coordination */}
      {state.coordination && state.coordination.nodes.length > 0 && (
        <section className="supervisor__coordination">
          <CoordinationGraph state={state.coordination} />
        </section>
      )}

      {/* Alert Feed */}
      <section className="supervisor__alerts">
        <h2>Alerts</h2>
        <AlertFeed alerts={alerts} />
      </section>
    </div>
  );
}
