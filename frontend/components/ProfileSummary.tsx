"use client";

interface ProfileSummaryProps {
  profile: Record<string, any>;
}

export default function ProfileSummary({ profile }: ProfileSummaryProps) {
  if (!profile) return null;

  const summary = profile.summary || {};
  const qualityIssues = profile.data_quality_issues || [];
  const missingSummary = profile.missing_summary || {};
  const numericStats = profile.numeric_stats || {};
  const categoricalStats = profile.categorical_stats || {};

  return (
    <div className="profile-container" style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Overview Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "12px" }}>
        <div className="glass-card" style={{ padding: "16px" }}>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "4px" }}>Duplicate Rows</div>
          <div style={{ fontSize: "1.25rem", fontWeight: 700, color: summary.duplicate_rows > 0 ? "var(--warning)" : "var(--success)" }}>
            {summary.duplicate_rows ?? 0}
          </div>
        </div>
        <div className="glass-card" style={{ padding: "16px" }}>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "4px" }}>Memory Footprint</div>
          <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--text-primary)" }}>
            {summary.memory_usage_mb ?? 0} MB
          </div>
        </div>
        <div className="glass-card" style={{ padding: "16px" }}>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "4px" }}>Numeric Columns</div>
          <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--accent)" }}>
            {summary.numeric_columns_count ?? 0}
          </div>
        </div>
        <div className="glass-card" style={{ padding: "16px" }}>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "4px" }}>Categorical Columns</div>
          <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--primary-light)" }}>
            {summary.categorical_columns_count ?? 0}
          </div>
        </div>
      </div>

      {/* Data Quality Alerts */}
      {qualityIssues.length > 0 && (
        <div className="glass-card" style={{ padding: "20px" }}>
          <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "12px", display: "flex", alignItems: "center", gap: "8px" }}>
            <span>⚠️</span> Data Quality & Health Flags ({qualityIssues.length})
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            {qualityIssues.map((issue: any, idx: number) => {
              const bg = issue.severity === "high"
                ? "rgba(239, 68, 68, 0.15)"
                : issue.severity === "medium"
                ? "rgba(245, 158, 11, 0.15)"
                : "rgba(59, 130, 246, 0.15)";
              const color = issue.severity === "high"
                ? "#fca5a5"
                : issue.severity === "medium"
                ? "#fde047"
                : "#93c5fd";

              return (
                <div key={idx} style={{ padding: "10px 14px", borderRadius: "10px", background: bg, color: color, fontSize: "0.85rem", fontWeight: 500 }}>
                  {issue.message}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Numeric Columns Analysis */}
      {Object.keys(numericStats).length > 0 && (
        <div className="glass-card" style={{ padding: "20px" }}>
          <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "16px" }}>📈 Numeric Statistics</h3>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={thStyle}>Column</th>
                  <th style={thStyle}>Mean</th>
                  <th style={thStyle}>Std Dev</th>
                  <th style={thStyle}>Min</th>
                  <th style={thStyle}>Median</th>
                  <th style={thStyle}>Max</th>
                  <th style={thStyle}>Skew</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(numericStats).map(([col, stats]: [string, any]) => (
                  <tr key={col}>
                    <td style={tdStyle}><code style={{ color: "var(--accent)" }}>{col}</code></td>
                    <td style={tdStyle}>{stats.mean ?? "-"}</td>
                    <td style={tdStyle}>{stats.std ?? "-"}</td>
                    <td style={tdStyle}>{stats.min ?? "-"}</td>
                    <td style={tdStyle}>{stats.median ?? "-"}</td>
                    <td style={tdStyle}>{stats.max ?? "-"}</td>
                    <td style={tdStyle}>{stats.skewness ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Categorical Columns Analysis */}
      {Object.keys(categoricalStats).length > 0 && (
        <div className="glass-card" style={{ padding: "20px" }}>
          <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "16px" }}>🔤 Categorical Distributions</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "16px" }}>
            {Object.entries(categoricalStats).map(([col, stats]: [string, any]) => (
              <div key={col} style={{ background: "var(--bg-input)", padding: "14px", borderRadius: "12px", border: "1px solid var(--border)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                  <code style={{ color: "var(--primary-light)", fontWeight: 600 }}>{col}</code>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{stats.unique_count} distinct</span>
                </div>
                <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "8px" }}>
                  Top value: <strong style={{ color: "var(--text-primary)" }}>{stats.top_value}</strong> ({stats.top_freq} occurrences)
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                  {Object.entries(stats.value_counts || {}).slice(0, 5).map(([val, count]: [string, any]) => (
                    <div key={val} style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem" }}>
                      <span style={{ color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "160px" }}>{val}</span>
                      <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

const thStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "10px 12px",
  fontSize: "0.8rem",
  fontWeight: 600,
  color: "var(--text-muted)",
  borderBottom: "1px solid var(--border)",
};

const tdStyle: React.CSSProperties = {
  padding: "10px 12px",
  fontSize: "0.85rem",
  borderBottom: "1px solid rgba(45, 45, 94, 0.5)",
};
