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
  coordination?: CoordinationState;
}

// ── Cross-agent coordination types ──────────────────────────────────────

export type EdgeWeight = "thick" | "medium" | "thin" | "none";
export type EdgeStatus = "normal" | "warning" | "breakdown";
export type CoordHealth = "coordinated" | "loosely_coupled" | "decoupled";

export interface CoordinationNode {
  id: string;
  events: number;
  anomalies: number;
  health: HealthStatus;
  filesRead: number;
  filesWritten: number;
}

export interface CoordinationEdge {
  source: string;
  target: string;
  mi: number;
  pairs: number;
  weight: EdgeWeight;
  status: EdgeStatus;
  miHistory: number[];
}

export interface CoordinationFailureEvent {
  failureType: string;
  agents: string[];
  resource: string;
  description: string;
  severity: string;
  step: number;
}

export interface CoordinationSignalEvent {
  signalType: string;
  sourceAgent: string;
  targetAgent: string;
  resource: string;
  latencySteps: number;
}

export interface DependencyEdge {
  producer: string;
  consumer: string;
  signalCount: number;
  strength: "strong" | "moderate" | "weak";
}

export interface CoordinationState {
  nodes: CoordinationNode[];
  edges: CoordinationEdge[];
  signals: CoordinationSignalEvent[];
  failures: CoordinationFailureEvent[];
  globalStep: number;
  dependencyGraph: DependencyEdge[];
  summary: {
    agentCount: number;
    pairCount: number;
    averageMi: number;
    totalSignals: number;
    totalFailures: number;
    coordinationHealth: CoordHealth;
  };
}
