"use client";

import React, { useEffect, useRef, useState } from "react";

declare global {
  interface Window {
    Plotly?: any;
  }
}

interface PlotlyChartProps {
  chartPath: string;
  chartIndex?: number;
  chartJson?: Record<string, any> | null;
  parameters?: Record<string, any> | null;
  explanation?: any;
  baseUrl: string;
}

export default function PlotlyChart({
  chartPath,
  chartIndex = 0,
  chartJson,
  parameters,
  explanation,
  baseUrl,
}: PlotlyChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [viewMode, setViewMode] = useState<"interactive" | "static">("interactive");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [figureData, setFigureData] = useState<any>(chartJson || null);

  const imageSrc = `${baseUrl}${chartPath}`;
  const jsonSrc = `${baseUrl}${chartPath.replace(/\.png$/i, ".json")}`;

  // Fetch Plotly figure JSON if not provided directly
  useEffect(() => {
    let isMounted = true;
    if (chartJson) {
      setFigureData(chartJson);
      setLoading(false);
      return;
    }

    const fetchJson = async () => {
      try {
        setLoading(true);
        const res = await fetch(jsonSrc);
        if (!res.ok) {
          throw new Error(`Failed to load chart JSON: ${res.statusText}`);
        }
        const data = await res.json();
        if (isMounted) {
          setFigureData(data);
          setError(null);
        }
      } catch (err: any) {
        if (isMounted) {
          console.warn("Could not load Plotly JSON, falling back to static preview:", err);
          setError(err.message);
          setViewMode("static");
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchJson();

    return () => {
      isMounted = false;
    };
  }, [jsonSrc, chartJson]);

  // Render interactive Plotly chart when Plotly is ready and figure data exists
  useEffect(() => {
    if (viewMode !== "interactive" || !containerRef.current || !figureData) {
      return;
    }

    let intervalId: any = null;
    let attempts = 0;

    const renderChart = () => {
      if (typeof window !== "undefined" && window.Plotly && containerRef.current) {
        const layout = {
          ...figureData.layout,
          autosize: true,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { family: "Inter, sans-serif", color: "#f1f5f9" },
          margin: { l: 45, r: 35, t: 50, b: 45 },
        };

        const config = {
          responsive: true,
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ["sendDataToCloud"],
          toImageButtonOptions: {
            format: "png",
            filename: `ai_madac_chart_${chartIndex + 1}`,
            height: 600,
            width: 900,
            scale: 2,
          },
        };

        try {
          window.Plotly.newPlot(
            containerRef.current,
            figureData.data || [],
            layout,
            config
          );
        } catch (e) {
          console.error("Plotly render error:", e);
          setViewMode("static");
        }

        if (intervalId) clearInterval(intervalId);
      } else {
        attempts++;
        if (attempts > 30) {
          if (intervalId) clearInterval(intervalId);
          console.warn("Plotly script took too long, showing static image.");
          setViewMode("static");
        }
      }
    };

    renderChart();
    intervalId = setInterval(renderChart, 200);

    const handleResize = () => {
      if (window.Plotly && containerRef.current) {
        window.Plotly.Plots.resize(containerRef.current);
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      if (intervalId) clearInterval(intervalId);
      window.removeEventListener("resize", handleResize);
      if (window.Plotly && containerRef.current) {
        try {
          window.Plotly.purge(containerRef.current);
        } catch {}
      }
    };
  }, [viewMode, figureData]);

  // Extract explanation details if structured
  const whatThisShows =
    typeof explanation === "object" && explanation !== null
      ? explanation.what_this_shows
      : null;
  const parametersUsed =
    typeof explanation === "object" && explanation !== null
      ? explanation.parameters_used
      : [];

  return (
    <div
      style={{
        border: "1px solid var(--border)",
        borderRadius: "14px",
        background: "var(--bg-input)",
        overflow: "hidden",
        marginBottom: "16px",
      }}
    >
      {/* Chart Top Header & View Mode Switcher */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "10px 16px",
          borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
          background: "rgba(16, 20, 38, 0.6)",
        }}
      >
        <span
          style={{
            fontSize: "0.82rem",
            fontWeight: 700,
            color: "var(--accent)",
            display: "flex",
            alignItems: "center",
            gap: "6px",
          }}
        >
          📊 {figureData?.layout?.title?.text || `Visualization ${chartIndex + 1}`}
        </span>

        <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
          <button
            onClick={() => setViewMode("interactive")}
            style={{
              background:
                viewMode === "interactive"
                  ? "var(--accent)"
                  : "rgba(255, 255, 255, 0.06)",
              color: "white",
              border: "none",
              borderRadius: "6px",
              padding: "4px 10px",
              fontSize: "0.75rem",
              fontWeight: 600,
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
            title="Interactive zoom, pan, hover, and PNG download"
          >
            ⚡ Interactive
          </button>
          <button
            onClick={() => setViewMode("static")}
            style={{
              background:
                viewMode === "static"
                  ? "var(--accent)"
                  : "rgba(255, 255, 255, 0.06)",
              color: "white",
              border: "none",
              borderRadius: "6px",
              padding: "4px 10px",
              fontSize: "0.75rem",
              fontWeight: 600,
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
            title="High-resolution image view"
          >
            🖼️ Static
          </button>
        </div>
      </div>

      {/* Chart Display Area */}
      <div
        style={{
          minHeight: "360px",
          position: "relative",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#101426",
          padding: "8px",
        }}
      >
        {viewMode === "interactive" ? (
          <>
            <div
              ref={containerRef}
              style={{
                width: "100%",
                height: "380px",
                display: loading ? "none" : "block",
              }}
            />
            {loading && (
              <div style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
                ⏳ Rendering interactive Plotly chart...
              </div>
            )}
          </>
        ) : (
          <img
            src={imageSrc}
            alt={`Analysis chart ${chartIndex + 1}`}
            style={{
              display: "block",
              width: "100%",
              height: "auto",
              maxHeight: "440px",
              objectFit: "contain",
            }}
          />
        )}
      </div>

      {/* Parameters & Explanation Panel */}
      {parameters && Object.keys(parameters).length > 0 && (
        <div
          style={{
            padding: "10px 16px",
            background: "rgba(10, 13, 28, 0.8)",
            borderTop: "1px solid rgba(255, 255, 255, 0.06)",
            fontSize: "0.78rem",
          }}
        >
          <div
            style={{
              fontWeight: 700,
              color: "var(--text-secondary)",
              marginBottom: "6px",
              textTransform: "uppercase",
              letterSpacing: "0.5px",
            }}
          >
            ⚙️ Applied Parameters
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
            {parameters.chart_type && (
              <span
                style={{
                  background: "rgba(99, 102, 241, 0.15)",
                  color: "#a5b4fc",
                  border: "1px solid rgba(99, 102, 241, 0.3)",
                  padding: "3px 8px",
                  borderRadius: "6px",
                }}
              >
                Chart: <strong>{parameters.chart_type}</strong>
              </span>
            )}
            {(parameters.category || parameters.x_axis) && (
              <span
                style={{
                  background: "rgba(6, 182, 212, 0.15)",
                  color: "#67e8f9",
                  border: "1px solid rgba(6, 182, 212, 0.3)",
                  padding: "3px 8px",
                  borderRadius: "6px",
                }}
              >
                X-Axis / Category: <strong>{parameters.category || parameters.x_axis}</strong>
              </span>
            )}
            {(parameters.value || parameters.y_axis) && (
              <span
                style={{
                  background: "rgba(168, 85, 247, 0.15)",
                  color: "#d8b4fe",
                  border: "1px solid rgba(168, 85, 247, 0.3)",
                  padding: "3px 8px",
                  borderRadius: "6px",
                }}
              >
                Y-Axis / Value: <strong>{parameters.value || parameters.y_axis}</strong>
              </span>
            )}
            {parameters.aggregation && (
              <span
                style={{
                  background: "rgba(16, 185, 129, 0.15)",
                  color: "#6ee7b7",
                  border: "1px solid rgba(16, 185, 129, 0.3)",
                  padding: "3px 8px",
                  borderRadius: "6px",
                }}
              >
                Aggregation: <strong>{parameters.aggregation}</strong>
              </span>
            )}
            {parameters.filters && (
              <span
                style={{
                  background: "rgba(245, 158, 11, 0.15)",
                  color: "#fcd34d",
                  border: "1px solid rgba(245, 158, 11, 0.3)",
                  padding: "3px 8px",
                  borderRadius: "6px",
                }}
              >
                Filter: <strong>{parameters.filters}</strong>
              </span>
            )}
            {parameters.sorting && (
              <span
                style={{
                  background: "rgba(255, 255, 255, 0.08)",
                  color: "#e2e8f0",
                  border: "1px solid rgba(255, 255, 255, 0.15)",
                  padding: "3px 8px",
                  borderRadius: "6px",
                }}
              >
                Sorting: <strong>{parameters.sorting}</strong>
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
