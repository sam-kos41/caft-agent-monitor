import type { SessionState, EventCategory } from "../types";

const CATEGORY_COLORS: Record<EventCategory, string> = {
  write: "#22c55e",        // green — actively producing
  read: "#86efac",         // light green — exploring
  shell: "#3b82f6",        // blue — running/testing
  anomaly_named: "#ef4444", // red — named anomaly
  anomaly_unclassified: "#eab308", // yellow — unclassified anomaly
  idle: "#374151",         // dark gray
  phase: "#8b5cf6",        // purple — phase boundary
};

export function TimelineBar({ session }: { session: SessionState }) {
  const segments = session.timeline;
  if (segments.length === 0) {
    return (
      <div className="timeline-row">
        <span className="timeline-label">{session.name}</span>
        <div className="timeline-bar timeline-bar--empty" />
      </div>
    );
  }

  // Normalize to fill the bar width
  const total = segments.length;

  return (
    <div className="timeline-row">
      <span className="timeline-label">{session.name}</span>
      <div className="timeline-bar">
        {segments.map((seg, i) => (
          <div
            key={i}
            className="timeline-segment"
            style={{
              width: `${(1 / total) * 100}%`,
              backgroundColor: CATEGORY_COLORS[seg.category],
            }}
          />
        ))}
      </div>
    </div>
  );
}
