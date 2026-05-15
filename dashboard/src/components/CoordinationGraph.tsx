import type {
  CoordinationState,
  CoordinationNode,
  CoordinationEdge,
  EdgeStatus,
  CoordHealth,
} from "../types";

const HEALTH_COLORS: Record<string, string> = {
  green: "#22c55e",
  yellow: "#eab308",
  red: "#ef4444",
  unknown: "#6b7280",
};

const EDGE_STROKE: Record<string, number> = {
  thick: 4,
  medium: 2.5,
  thin: 1.5,
  none: 0.5,
};

const EDGE_COLORS: Record<EdgeStatus, string> = {
  normal: "#94a3b8",
  warning: "#eab308",
  breakdown: "#ef4444",
};

const COORD_HEALTH_LABELS: Record<CoordHealth, string> = {
  coordinated: "Coordinated",
  loosely_coupled: "Loosely Coupled",
  decoupled: "Decoupled",
};

const COORD_HEALTH_COLORS: Record<CoordHealth, string> = {
  coordinated: "#22c55e",
  loosely_coupled: "#eab308",
  decoupled: "#6b7280",
};

function nodePositions(
  count: number,
  cx: number,
  cy: number,
  radius: number
): { x: number; y: number }[] {
  if (count === 0) return [];
  if (count === 1) return [{ x: cx, y: cy }];
  return Array.from({ length: count }, (_, i) => {
    const angle = (2 * Math.PI * i) / count - Math.PI / 2;
    return { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
  });
}

function MiniSparkline({ data, width = 60, height = 16 }: { data: number[]; width?: number; height?: number }) {
  if (data.length < 2) return null;
  const max = Math.max(...data, 0.001);
  const points = data
    .map((v, i) => `${(i / (data.length - 1)) * width},${height - (v / max) * height}`)
    .join(" ");
  return (
    <svg width={width} height={height} style={{ display: "inline-block", verticalAlign: "middle" }}>
      <polyline points={points} fill="none" stroke="#94a3b8" strokeWidth={1} />
    </svg>
  );
}

export function CoordinationGraph({ state }: { state: CoordinationState }) {
  const { nodes, edges, failures, summary, dependencyGraph } = state;

  if (nodes.length === 0) {
    return (
      <div className="coordination-graph coordination-graph--empty">
        <p>No agents registered for coordination monitoring.</p>
      </div>
    );
  }

  const svgW = 400;
  const svgH = 300;
  const cx = svgW / 2;
  const cy = svgH / 2;
  const radius = Math.min(cx, cy) - 60;
  const positions = nodePositions(nodes.length, cx, cy, radius);

  const nodeIndex: Record<string, number> = {};
  nodes.forEach((n, i) => {
    nodeIndex[n.id] = i;
  });

  const healthColor = COORD_HEALTH_COLORS[summary.coordinationHealth] ?? "#6b7280";

  return (
    <div className="coordination-graph">
      <div className="coordination-graph__header">
        <h2>Cross-Agent Coordination</h2>
        <span
          className="coordination-graph__health"
          style={{ color: healthColor }}
        >
          {COORD_HEALTH_LABELS[summary.coordinationHealth] ?? "Unknown"}
        </span>
        <span className="coordination-graph__meta">
          {summary.agentCount} agents &middot; MI avg {summary.averageMi.toFixed(3)}b
          &middot; {summary.totalSignals} signals &middot; {summary.totalFailures} failures
        </span>
      </div>

      {/* Graph SVG */}
      <svg
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="coordination-graph__svg"
        style={{ width: "100%", maxWidth: svgW, height: "auto" }}
      >
        {/* Edges */}
        {edges.map((edge) => {
          const si = nodeIndex[edge.source];
          const ti = nodeIndex[edge.target];
          if (si === undefined || ti === undefined) return null;
          const p1 = positions[si];
          const p2 = positions[ti];
          return (
            <g key={`${edge.source}-${edge.target}`}>
              <line
                x1={p1.x}
                y1={p1.y}
                x2={p2.x}
                y2={p2.y}
                stroke={EDGE_COLORS[edge.status]}
                strokeWidth={EDGE_STROKE[edge.weight]}
                strokeDasharray={edge.status === "breakdown" ? "6,3" : undefined}
                opacity={edge.weight === "none" ? 0.3 : 0.8}
              />
              {/* MI label on edge midpoint */}
              <text
                x={(p1.x + p2.x) / 2}
                y={(p1.y + p2.y) / 2 - 6}
                textAnchor="middle"
                fontSize={10}
                fill="#94a3b8"
              >
                {edge.mi.toFixed(2)}b
              </text>
            </g>
          );
        })}

        {/* Nodes */}
        {nodes.map((node, i) => {
          const pos = positions[i];
          const color = HEALTH_COLORS[node.health] ?? HEALTH_COLORS.unknown;
          return (
            <g key={node.id}>
              <circle
                cx={pos.x}
                cy={pos.y}
                r={24}
                fill={color}
                opacity={0.2}
                stroke={color}
                strokeWidth={2}
              />
              <circle cx={pos.x} cy={pos.y} r={6} fill={color} />
              <text
                x={pos.x}
                y={pos.y + 36}
                textAnchor="middle"
                fontSize={11}
                fill="#e2e8f0"
                fontWeight={600}
              >
                {node.id}
              </text>
              <text
                x={pos.x}
                y={pos.y + 48}
                textAnchor="middle"
                fontSize={9}
                fill="#94a3b8"
              >
                {node.events}ev {node.filesRead}R/{node.filesWritten}W
              </text>
            </g>
          );
        })}

        {/* Dependency arrows */}
        {dependencyGraph.map((dep) => {
          const si = nodeIndex[dep.producer];
          const ti = nodeIndex[dep.consumer];
          if (si === undefined || ti === undefined) return null;
          const p1 = positions[si];
          const p2 = positions[ti];
          const dx = p2.x - p1.x;
          const dy = p2.y - p1.y;
          const len = Math.sqrt(dx * dx + dy * dy);
          if (len === 0) return null;
          const nx = dx / len;
          const ny = dy / len;
          // Arrow from producer outer circle to consumer outer circle
          const x1 = p1.x + nx * 28;
          const y1 = p1.y + ny * 28;
          const x2 = p2.x - nx * 28;
          const y2 = p2.y - ny * 28;
          const strokeW = dep.strength === "strong" ? 2 : dep.strength === "moderate" ? 1.5 : 1;
          return (
            <g key={`dep-${dep.producer}-${dep.consumer}`}>
              <line
                x1={x1} y1={y1} x2={x2} y2={y2}
                stroke="#60a5fa"
                strokeWidth={strokeW}
                markerEnd="url(#arrowhead)"
                opacity={0.6}
              />
            </g>
          );
        })}

        {/* Arrow marker definition */}
        <defs>
          <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#60a5fa" opacity={0.6} />
          </marker>
        </defs>
      </svg>

      {/* Edge details table */}
      {edges.length > 0 && (
        <div className="coordination-graph__edges">
          <h3>Pairwise MI</h3>
          <table>
            <thead>
              <tr>
                <th>Pair</th>
                <th>MI</th>
                <th>Trend</th>
                <th>Status</th>
                <th>Pairs</th>
              </tr>
            </thead>
            <tbody>
              {edges.map((edge) => (
                <tr key={`${edge.source}-${edge.target}`}>
                  <td>{edge.source} &harr; {edge.target}</td>
                  <td>{edge.mi.toFixed(3)}b</td>
                  <td><MiniSparkline data={edge.miHistory} /></td>
                  <td style={{ color: EDGE_COLORS[edge.status] }}>{edge.status}</td>
                  <td>{edge.pairs}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Recent failures */}
      {failures.length > 0 && (
        <div className="coordination-graph__failures">
          <h3>Coordination Failures</h3>
          <ul>
            {failures.slice(-5).map((f, i) => (
              <li key={i} className={`failure-item failure-item--${f.severity}`}>
                <span className="failure-type">{f.failureType}</span>
                <span className="failure-desc">{f.description}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
