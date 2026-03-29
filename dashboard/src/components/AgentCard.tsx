import { useEffect, useRef } from "react";
import type { SessionState } from "../types";

const HEARTBEAT_TIMEOUT = 30000;

function HealthDot({ lastEventTime }: { lastEventTime: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const alive = Date.now() - lastEventTime < HEARTBEAT_TIMEOUT;

  useEffect(() => {
    if (!ref.current || !alive) return;
    ref.current.classList.remove("pulse");
    void ref.current.offsetWidth; // force reflow
    ref.current.classList.add("pulse");
  }, [lastEventTime, alive]);

  return (
    <div
      ref={ref}
      className={`heartbeat-dot ${alive ? "alive" : "stalled"}`}
    />
  );
}

export function AgentCard({
  session,
  onClick,
}: {
  session: SessionState;
  onClick?: () => void;
}) {
  const { health, name, currentAction, currentFile, eventCount, anomalyCount } =
    session;

  const statusLine =
    health === "red"
      ? `${currentAction} — ${currentFile}`
      : health === "yellow"
        ? currentFile || currentAction
        : `${currentAction} ${currentFile}`.trim();

  return (
    <div
      className={`agent-card agent-card--${health}`}
      onClick={onClick}
      role="button"
      tabIndex={0}
    >
      <div className="agent-card__name">{name}</div>

      <div className="agent-card__status">
        {health === "red" ? (
          <span className="agent-card__stuck">{statusLine}</span>
        ) : (
          <span className="agent-card__action">{statusLine}</span>
        )}
      </div>

      <div className="agent-card__footer">
        <span className="agent-card__events">{eventCount} steps</span>
        {anomalyCount > 0 && (
          <span className="agent-card__anomalies">
            {anomalyCount} anomal{anomalyCount === 1 ? "y" : "ies"}
          </span>
        )}
        <HealthDot lastEventTime={session.lastEventTime} />
      </div>
    </div>
  );
}
