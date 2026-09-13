"use client";

interface AgentStatusBadgeProps {
  agents: string[];
}

const AGENT_CONFIG: Record<string, { label: string; icon: string; color: string }> = {
  supervisor_agent: { label: "Supervisor", icon: "🧠", color: "#818cf8" },
  sql_agent: { label: "SQL Agent", icon: "⚡", color: "#22d3ee" },
  data_analyst_agent: { label: "Data Analyst", icon: "📈", color: "#34d399" },
  ml_agent: { label: "ML Agent", icon: "🤖", color: "#f472b6" },
  visualization_agent: { label: "Visualization", icon: "📊", color: "#fbbf24" },
  report_agent: { label: "Report Synthesizer", icon: "📋", color: "#a78bfa" },
};

export default function AgentStatusBadge({ agents }: AgentStatusBadgeProps) {
  if (!agents || agents.length === 0) return null;

  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", margin: "8px 0" }}>
      {agents.map((agentKey) => {
        const config = AGENT_CONFIG[agentKey] || {
          label: agentKey.replace("_", " "),
          icon: "🤖",
          color: "var(--text-muted)",
        };

        return (
          <span
            key={agentKey}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "4px",
              padding: "3px 8px",
              borderRadius: "6px",
              background: "rgba(30, 30, 66, 0.9)",
              border: `1px solid ${config.color}40`,
              color: config.color,
              fontSize: "0.72rem",
              fontWeight: 600,
            }}
          >
            <span>{config.icon}</span>
            <span>{config.label}</span>
          </span>
        );
      })}
    </div>
  );
}
