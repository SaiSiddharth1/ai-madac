"use client";

import React from "react";

interface KpiCardProps {
  label: string;
  value: string | number;
  explanation?: {
    what_this_shows?: string;
    parameters_used?: string[];
    key_insights?: string[];
  } | null;
  insights?: string[];
}

export default function KpiCard({
  label,
  value,
  explanation,
  insights = [],
}: KpiCardProps) {
  const whatThisShows = explanation?.what_this_shows;
  const params = explanation?.parameters_used || [];
  const allInsights = (explanation?.key_insights && explanation.key_insights.length > 0)
    ? explanation.key_insights
    : insights;

  return (
    <div
      style={{
        margin: "16px 0",
        borderRadius: "16px",
        background: "linear-gradient(135deg, rgba(20, 24, 48, 0.95), rgba(13, 16, 35, 0.98))",
        border: "1px solid rgba(99, 102, 241, 0.35)",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.37), inset 0 1px 0 rgba(255, 255, 255, 0.1)",
        overflow: "hidden",
      }}
    >
      {/* KPI Top Hero Header */}
      <div
        style={{
          padding: "24px 28px",
          borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
          background: "radial-gradient(ellipse at top left, rgba(99, 102, 241, 0.15), transparent 70%)",
        }}
      >
        <div
          style={{
            fontSize: "0.85rem",
            fontWeight: 700,
            textTransform: "uppercase",
            letterSpacing: "1px",
            color: "#a5b4fc",
            marginBottom: "8px",
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          <span>🎯</span> {label}
        </div>
        <div
          style={{
            fontSize: "2.6rem",
            fontWeight: 800,
            color: "#f8fafc",
            fontFamily: "Inter, system-ui, sans-serif",
            letterSpacing: "-0.5px",
            lineHeight: 1.1,
            textShadow: "0 2px 12px rgba(99, 102, 241, 0.4)",
          }}
        >
          {String(value)}
        </div>
      </div>

      {/* Structured Insights & Explanation Panels */}
      <div style={{ padding: "20px 28px", display: "flex", flexDirection: "column", gap: "16px" }}>
        {/* What this means */}
        {whatThisShows && (
          <div>
            <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "4px" }}>
              💡 What this means
            </div>
            <div style={{ fontSize: "0.9rem", color: "#e2e8f0", lineHeight: 1.5 }}>
              {whatThisShows}
            </div>
          </div>
        )}

        {/* Parameters used */}
        {params.length > 0 && (
          <div>
            <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "6px" }}>
              ⚙️ Parameters used
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
              {params.map((p, idx) => (
                <span
                  key={idx}
                  style={{
                    background: "rgba(99, 102, 241, 0.12)",
                    border: "1px solid rgba(99, 102, 241, 0.25)",
                    color: "#c7d2fe",
                    padding: "4px 10px",
                    borderRadius: "6px",
                    fontSize: "0.78rem",
                    fontWeight: 500,
                  }}
                >
                  {p}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Key insight */}
        {allInsights.length > 0 && (
          <div style={{ background: "rgba(16, 185, 129, 0.08)", border: "1px solid rgba(16, 185, 129, 0.2)", borderRadius: "10px", padding: "12px 16px" }}>
            <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "#6ee7b7", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "6px" }}>
              📌 Key Insight
            </div>
            <ul style={{ margin: 0, paddingLeft: "18px", color: "#d1fae5", fontSize: "0.85rem", lineHeight: 1.5 }}>
              {allInsights.map((ins, idx) => (
                <li key={idx} style={{ marginBottom: idx < allInsights.length - 1 ? "4px" : 0 }}>
                  {ins}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
