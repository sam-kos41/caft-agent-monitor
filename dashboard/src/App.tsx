import { useState } from "react";
import { useCAFT } from "./hooks/useCAFT";
import { Supervisor } from "./components/Supervisor";
import type { SessionState } from "./types";
import "./App.css";

function App() {
  const [selectedSession, setSelectedSession] = useState<SessionState | null>(null);

  const wsHost = window.location.host || "localhost:8080";
  const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const state = useCAFT(`${wsProtocol}//${wsHost}/ws`);

  if (selectedSession) {
    const session = state.sessions.find(
      (s) => s.sessionId === selectedSession.sessionId
    ) || selectedSession;

    return (
      <div className="app app--detail">
        <button className="back-button" onClick={() => setSelectedSession(null)}>
          Back to Supervisor
        </button>
        <div className="detail-view">
          <h1>{session.name}</h1>
          <div className="detail-grid">
            <div className="detail-card">
              <h3>Health</h3>
              <div className={`detail-health detail-health--${session.health}`}>
                {session.health.toUpperCase()}
              </div>
            </div>
            <div className="detail-card">
              <h3>Events</h3>
              <div className="detail-value">{session.eventCount}</div>
            </div>
            <div className="detail-card">
              <h3>Anomalies</h3>
              <div className="detail-value">{session.anomalyCount}</div>
            </div>
            <div className="detail-card">
              <h3>Action MI</h3>
              <div className="detail-value">{session.actionMI.toFixed(2)}b</div>
            </div>
            <div className="detail-card">
              <h3>Tool Entropy</h3>
              <div className="detail-value">{session.toolEntropy.toFixed(2)}b</div>
            </div>
            <div className="detail-card">
              <h3>KL Divergence</h3>
              <div className="detail-value">{session.klDivergence.toFixed(3)}</div>
            </div>
          </div>
          {session.anomalies.length > 0 && (
            <div className="detail-anomalies">
              <h3>Anomalies</h3>
              {session.anomalies.map((a) => (
                <div key={a.id} className={`detail-anomaly detail-anomaly--${a.severity}`}>
                  <span className="detail-anomaly__step">Step {a.step}</span>
                  <span className="detail-anomaly__sig">{a.signature}</span>
                  <span className="detail-anomaly__msg">{a.message}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <Supervisor state={state} onSelectSession={setSelectedSession} />
    </div>
  );
}

export default App;
