import { useEffect, useRef, useState, useCallback } from "react";
import type {
  SupervisorState,
  SessionState,
  AnomalyAlert,
  HealthStatus,
  TimelineSegment,
  EventCategory,
} from "../types";

const MAX_ALERTS = 50;
const MAX_TIMELINE = 300;
function classifyEvent(action: {
  tool?: string;
  success?: boolean;
}): EventCategory {
  const tool = (action.tool || "").toLowerCase();
  if (["write", "edit", "multiedit"].includes(tool)) return "write";
  if (["read", "grep", "glob"].includes(tool)) return "read";
  if (["bash", "shell"].includes(tool)) return "shell";
  return "read";
}

function computeHealth(
  anomalyCount: number,
  eventCount: number,
  recentDiagnoses: Array<{ severity?: string; caft_code?: string }>,
): HealthStatus {
  // Any named critical anomaly in recent diagnoses = red
  const hasCritical = recentDiagnoses.some(
    (d) => d.severity === "critical"
  );
  if (hasCritical) return "red";

  // Any warning-level diagnosis = yellow
  const hasWarning = recentDiagnoses.some(
    (d) => d.severity === "warning"
  );

  // High anomaly rate = red
  if (eventCount > 20 && anomalyCount / eventCount > 0.15) return "red";
  if (eventCount > 20 && anomalyCount / eventCount > 0.05) return "yellow";

  if (hasWarning) return "yellow";
  if (eventCount === 0) return "unknown";
  return "green";
}

function extractCurrentAction(state: Record<string, unknown>): {
  action: string;
  file: string;
} {
  const actions = (state.actions || []) as Array<{
    tool?: string;
    step?: number;
  }>;
  if (actions.length === 0) return { action: "idle", file: "" };

  const last = actions[actions.length - 1];
  const tool = last.tool || "unknown";
  return {
    action: tool.toLowerCase(),
    file: "", // Will be filled from target_path when available
  };
}

export function useCAFT(
  wsUrl: string = `ws://${window.location.host}/ws`,
): SupervisorState {
  const [state, setState] = useState<SupervisorState>({
    sessions: [],
    alerts: [],
    uptimeSeconds: 0,
    connected: false,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const startTimeRef = useRef(Date.now());
  const sessionsRef = useRef<Map<string, SessionState>>(new Map());
  const alertsRef = useRef<AnomalyAlert[]>([]);

  const processMessage = useCallback((data: Record<string, unknown>) => {
    const type = data.type as string;
    if (type === "waiting") return;

    // Extract session ID (may come from session_context or default to "main")
    const ctx = (data.session_context || {}) as Record<string, string>;
    const sessionId = ctx.session_id || "main";
    const projectDir = ctx.project_dir || "";
    const sessionName =
      projectDir || sessionId.slice(0, 8);

    // Get or create session state
    const totalEvents = (data.total_events as number) || 0;
    const diagnoses = (data.diagnoses || []) as Array<{
      severity?: string;
      caft_code?: string;
      failure_name?: string;
      description?: string;
      at_step?: number;
      confidence?: number;
    }>;

    // Count anomalies from diagnoses
    const anomalyCount = diagnoses.filter(
      (d) => d.severity === "warning" || d.severity === "critical"
    ).length;

    const health = computeHealth(anomalyCount, totalEvents, diagnoses);
    const { action, file } = extractCurrentAction(data);

    // Build timeline from actions
    const actions = (data.actions || []) as Array<{
      tool?: string;
      step?: number;
      success?: boolean;
    }>;
    const timeline: TimelineSegment[] = actions.map((a, i) => ({
      step: a.step || i,
      category: classifyEvent(a),
      timestamp: Date.now(),
    }));

    // Check for new anomaly alerts
    const newAlerts: AnomalyAlert[] = [];
    for (const d of diagnoses) {
      if (d.severity === "warning" || d.severity === "critical") {
        const alertId = `${sessionId}-${d.at_step}-${d.caft_code}`;
        const alreadyExists = alertsRef.current.some(
          (a) => a.id === alertId
        );
        if (!alreadyExists) {
          newAlerts.push({
            id: alertId,
            timestamp: Date.now(),
            sessionId,
            sessionName,
            signature: d.failure_name || d.caft_code || "unknown",
            message:
              d.description || `${d.failure_name} detected`,
            severity: d.severity as "warning" | "critical",
            step: d.at_step || totalEvents,
          });
        }
      }
    }

    if (newAlerts.length > 0) {
      alertsRef.current = [
        ...newAlerts,
        ...alertsRef.current,
      ].slice(0, MAX_ALERTS);
    }

    // IT metrics
    const it = (data.info_theoretic || {}) as Record<string, number>;
    // Build status text for red state
    let currentAction = action;
    let currentFile = file;
    if (health === "red" && diagnoses.length > 0) {
      const worst = diagnoses.find((d) => d.severity === "critical") || diagnoses[0];
      currentAction = `STUCK`;
      currentFile = worst.failure_name || worst.description || "anomaly detected";
    } else {
      // Normal action display
      const lastAction = actions[actions.length - 1];
      if (lastAction) {
        const tool = (lastAction.tool || "").toLowerCase();
        if (["write", "edit"].includes(tool)) currentAction = "writing";
        else if (["read", "grep", "glob"].includes(tool)) currentAction = "reading";
        else if (["bash", "shell"].includes(tool)) currentAction = "running";
        else currentAction = tool;
      }
    }

    const session: SessionState = {
      sessionId,
      name: sessionName,
      health,
      eventCount: totalEvents,
      anomalyCount,
      lastEventTime: Date.now(),
      currentAction,
      currentFile,
      actionMI: (it.action_mi as number) || 0,
      toolEntropy: (it.tool_entropy as number) || 0,
      klDivergence: (it.kl_divergence as number) || 0,
      coherence: 0,
      timeline: timeline.slice(-MAX_TIMELINE),
      anomalies: newAlerts,
      phase: (data.phase as string) || "unknown",
    };

    sessionsRef.current.set(sessionId, session);

    setState({
      sessions: Array.from(sessionsRef.current.values()),
      alerts: alertsRef.current,
      uptimeSeconds: (Date.now() - startTimeRef.current) / 1000,
      connected: true,
    });
  }, []);

  useEffect(() => {
    let reconnectTimer: ReturnType<typeof setTimeout>;

    function connect() {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setState((prev) => ({ ...prev, connected: true }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          processMessage(data);
        } catch {
          // ignore parse errors
        }
      };

      ws.onclose = () => {
        setState((prev) => ({ ...prev, connected: false }));
        reconnectTimer = setTimeout(connect, 2000);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, [wsUrl, processMessage]);

  return state;
}
