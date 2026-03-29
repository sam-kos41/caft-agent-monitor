export type HealthStatus = "green" | "yellow" | "red" | "unknown";

export type EventCategory = "write" | "read" | "shell" | "anomaly_named" | "anomaly_unclassified" | "idle" | "phase";

export interface TimelineSegment {
  step: number;
  category: EventCategory;
  timestamp: number;
}

export interface AnomalyAlert {
  id: string;
  timestamp: number;
  sessionId: string;
  sessionName: string;
  signature: string;
  message: string;
  severity: "warning" | "critical";
  step: number;
}

export interface SessionState {
  sessionId: string;
  name: string;
  health: HealthStatus;
  eventCount: number;
  anomalyCount: number;
  lastEventTime: number;
  currentAction: string;
  currentFile: string;
  actionMI: number;
  toolEntropy: number;
  klDivergence: number;
  coherence: number;
  timeline: TimelineSegment[];
  anomalies: AnomalyAlert[];
  phase: string;
}

export interface SupervisorState {
  sessions: SessionState[];
  alerts: AnomalyAlert[];
  uptimeSeconds: number;
  connected: boolean;
}
