import type { AnomalyAlert } from "../types";

function formatTime(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

export function AlertFeed({
  alerts,
  onClickAlert,
}: {
  alerts: AnomalyAlert[];
  onClickAlert?: (alert: AnomalyAlert) => void;
}) {
  if (alerts.length === 0) {
    return (
      <div className="alert-feed">
        <div className="alert-feed__empty">No anomalies detected</div>
      </div>
    );
  }

  return (
    <div className="alert-feed">
      {alerts.slice(0, 5).map((alert) => (
        <div
          key={alert.id}
          className={`alert-item alert-item--${alert.severity}`}
          onClick={() => onClickAlert?.(alert)}
          role="button"
          tabIndex={0}
        >
          <span className="alert-item__dot" />
          <span className="alert-item__time">{formatTime(alert.timestamp)}</span>
          <span className="alert-item__session">{alert.sessionName}</span>
          <span className="alert-item__message">{alert.message}</span>
        </div>
      ))}
    </div>
  );
}
