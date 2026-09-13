"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { apiGetReports, getReportPdfUrl, getReportJsonUrl } from "@/lib/api";
import AgentStatusBadge from "@/components/AgentStatusBadge";

export default function ReportsPage() {
  const [reports, setReports] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadReports();
  }, []);

  const loadReports = async () => {
    try {
      const data = await apiGetReports();
      setReports(data);
    } catch (err) {
      console.error("Failed to load reports:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-fade-in">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "32px" }}>
        <div>
          <h1 style={{ fontSize: "1.5rem", fontWeight: 800, marginBottom: "4px" }}>Generated Reports</h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
            View and download multi-agent analysis reports
          </p>
        </div>
      </div>

      {loading ? (
        <div style={{ display: "flex", justifyContent: "center", padding: "60px" }}>
          <div className="spinner" />
        </div>
      ) : reports.length === 0 ? (
        <div className="glass-card" style={{ padding: "60px 24px", textAlign: "center" }}>
          <div style={{ fontSize: "3rem", marginBottom: "16px" }}>📋</div>
          <h3 style={{ fontWeight: 600, marginBottom: "8px" }}>No reports generated yet</h3>
          <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", marginBottom: "20px" }}>
            Reports are automatically generated when you ask questions about your datasets.
          </p>
          <Link href="/dashboard/datasets" className="btn-primary" style={{ textDecoration: "none" }}>
            View Datasets
          </Link>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {reports.map((rep) => (
            <div key={rep.id} className="glass-card" style={{ padding: "24px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "12px" }}>
                <div>
                  <div style={{ fontSize: "0.8rem", color: "var(--primary-light)", fontWeight: 600, marginBottom: "4px" }}>
                    📁 Dataset: {rep.dataset_name || "Unknown"}
                  </div>
                  <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-primary)" }}>
                    &quot;{rep.question}&quot;
                  </h3>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px" }}>
                    Generated on {new Date(rep.created_at).toLocaleString()}
                  </div>
                </div>

                {/* Export Action Buttons */}
                <div style={{ display: "flex", gap: "8px" }}>
                  <a
                    href={getReportPdfUrl(rep.id)}
                    target="_blank"
                    rel="noreferrer"
                    style={{
                      padding: "8px 14px",
                      borderRadius: "8px",
                      background: "rgba(99, 102, 241, 0.2)",
                      border: "1px solid var(--primary-light)",
                      color: "white",
                      fontSize: "0.8rem",
                      fontWeight: 600,
                      textDecoration: "none",
                    }}
                  >
                    📄 Export PDF
                  </a>
                  <a
                    href={getReportJsonUrl(rep.id)}
                    target="_blank"
                    rel="noreferrer"
                    style={{
                      padding: "8px 14px",
                      borderRadius: "8px",
                      background: "rgba(255, 255, 255, 0.05)",
                      border: "1px solid var(--border)",
                      color: "var(--text-secondary)",
                      fontSize: "0.8rem",
                      fontWeight: 600,
                      textDecoration: "none",
                    }}
                  >
                    💾 Export JSON
                  </a>
                </div>
              </div>

              {/* Agent execution badges */}
              <AgentStatusBadge agents={rep.agents_used || []} />

              {/* Executive Answer */}
              <div style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.6, margin: "12px 0" }}>
                {rep.answer}
              </div>

              {/* Key Insights */}
              {rep.insights && rep.insights.length > 0 && (
                <div style={{ background: "var(--bg-input)", padding: "12px 16px", borderRadius: "10px", marginTop: "12px" }}>
                  <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--accent)", marginBottom: "4px" }}>
                    Key Insights:
                  </div>
                  <ul style={{ paddingLeft: "18px", fontSize: "0.82rem", color: "var(--text-secondary)" }}>
                    {rep.insights.map((ins: string, i: number) => (
                      <li key={i}>{ins}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
