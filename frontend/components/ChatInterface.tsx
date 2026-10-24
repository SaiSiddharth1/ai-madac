"use client";

import { useEffect, useRef, useState } from "react";
import { apiQueryDataset, apiGetDatasetQueries } from "@/lib/api";
import AgentStatusBadge from "./AgentStatusBadge";
import AgentWorkflow, { WorkflowState, AgentState } from "./AgentWorkflow";
import PlotlyChart from "./PlotlyChart";
import KpiCard from "./KpiCard";
import InteractiveTable from "./InteractiveTable";
import type { FullQueryResponse } from "@/types";

interface ChatInterfaceProps {
  datasetId: string;
  datasetName: string;
}

const SAMPLE_QUESTIONS = [
  "How many rows are in this dataset?",
  "How many columns are in this dataset?",
  "What are the column names?",
  "Show a summary of this dataset.",
  "Which columns contain missing values?",
  "Show the first 10 records.",
  "Show basic statistics for the numerical columns.",
  "What are the unique values in the categorical columns?",
];

export default function ChatInterface({ datasetId, datasetName }: ChatInterfaceProps) {
  const [history, setHistory] = useState<any[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [fetchingHistory, setFetchingHistory] = useState(true);
  const [workflowState, setWorkflowState] = useState<WorkflowState | null>(null);
  const [activeViews, setActiveViews] = useState<Record<string, "chart" | "table">>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const sessionIdRef = useRef("");

  useEffect(() => {
    const key = `madac_session_${datasetId}`;
    let sessionId = localStorage.getItem(key);
    if (!sessionId) {
      sessionId = crypto.randomUUID();
      localStorage.setItem(key, sessionId);
    }
    sessionIdRef.current = sessionId;
  }, [datasetId]);

  const getHistoricalWorkflowState = (query: any, report: any): WorkflowState => {
    const intent = query.intent || "DATA_RETRIEVAL";
    const plan = report.agents_used || ["sql_agent", "report_agent"];
    const agents: Record<string, AgentState> = {
      supervisor_agent: { status: 'completed', message: 'Intent detected: ' + intent }
    };
    plan.forEach((agent: string) => {
      agents[agent] = { status: 'completed', message: 'Task completed successfully' };
    });
    const allAgents = [
      "sql_agent", "data_analyst_agent", "ml_agent",
      "visualization_agent", "visualization_validator", "chart_renderer",
      "rag_agent", "report_agent"
    ];
    allAgents.forEach(a => {
      if (!plan.includes(a)) {
        agents[a] = { status: 'skipped', message: 'Not required' };
      }
    });
    return {
      intent,
      executionPlan: plan,
      agents
    };
  };

  useEffect(() => {
    loadQueryHistory();
  }, [datasetId]);

  useEffect(() => {
    scrollToBottom();
  }, [history, loading]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const loadQueryHistory = async () => {
    try {
      const data = await apiGetDatasetQueries(datasetId);
      // Backend returns newest first; reverse for chat chronological view
      setHistory(data.reverse());
    } catch (err) {
      console.error("Failed to load query history:", err);
    } finally {
      setFetchingHistory(false);
    }
  };

  const handleSubmit = async (e?: React.FormEvent, customQuestion?: string) => {
    if (e) e.preventDefault();
    const q = (customQuestion || question).trim();
    if (!q || loading) return;

    // Append optimistic user message
    const tempUserMsg = {
      id: "temp-" + Date.now(),
      question: q,
      isUser: true,
    };

    setHistory((prev) => [...prev, tempUserMsg]);
    if (!customQuestion) setQuestion("");
    setLoading(true);

    const currentWorkflow: WorkflowState = {
      intent: "",
      executionPlan: [],
      agents: {
        supervisor_agent: { status: 'running', message: 'Understanding query and routing plan...' },
        sql_agent: { status: 'waiting', message: 'Waiting' },
        data_analyst_agent: { status: 'waiting', message: 'Waiting' },
        ml_agent: { status: 'waiting', message: 'Waiting' },
        visualization_agent: { status: 'waiting', message: 'Waiting' },
        rag_agent: { status: 'waiting', message: 'Waiting' },
        report_agent: { status: 'waiting', message: 'Waiting' },
      }
    };
    setWorkflowState(currentWorkflow);

    try {
      const token = localStorage.getItem("madac_token");
      const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      
      const response = await fetch(`${API_BASE}/api/datasets/${datasetId}/query/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token || ""}`,
        },
        body: JSON.stringify({ question: q, session_id: sessionIdRef.current }),
      });

      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
          const errBody = await response.json();
          detail = errBody.detail || JSON.stringify(errBody);
        } catch { /* response wasn't JSON */ }
        throw new Error(`Failed to start workflow stream: ${detail}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("ReadableStream not supported by browser");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const dataStr = line.slice(6).trim();
          if (!dataStr) continue;

          try {
            const event = JSON.parse(dataStr);
            const { agent, status, message, intent: eventIntent, execution_plan } = event;

            if (agent === "supervisor_agent" && status === "completed") {
              currentWorkflow.intent = eventIntent || "DATA_RETRIEVAL";
              currentWorkflow.executionPlan = execution_plan || [];
              
              const allAgents = [
                "sql_agent", "data_analyst_agent", "ml_agent",
                "visualization_agent", "visualization_validator", "chart_renderer",
                "rag_agent", "report_agent"
              ];
              allAgents.forEach(a => {
                if (!execution_plan.includes(a)) {
                  currentWorkflow.agents[a] = { status: 'skipped', message: 'Not required for this query' };
                } else {
                  currentWorkflow.agents[a] = { status: 'waiting', message: 'Waiting' };
                }
              });
            }

            if (currentWorkflow.agents[agent]) {
              currentWorkflow.agents[agent] = { status, message };
            }

            setWorkflowState({ ...currentWorkflow });

            // A failed workflow is still a successful SSE connection. Surface
            // its server-side reason rather than waiting for the connection to
            // close and showing the misleading generic network error.
            if (status === "failed") {
              setHistory((prev) => [
                ...prev.filter((m) => m.id !== tempUserMsg.id),
                {
                  id: "err-" + Date.now(),
                  error: message || "The analysis workflow could not be completed.",
                  question: q,
                },
              ]);
              setWorkflowState(null);
            }

            if (event.result) {
              const res = event.result;
              res.workflow = { ...currentWorkflow };
              
              setHistory((prev) => [
                ...prev.filter((m) => m.id !== tempUserMsg.id),
                res,
              ]);
              setWorkflowState(null);
            }
          } catch (err) {
            console.error("Failed to parse event:", err);
          }
        }
      }
    } catch (err: any) {
      setHistory((prev) => [
        ...prev,
        {
          id: "err-" + Date.now(),
          error: err.message || "Failed to process query",
          question: q,
        },
      ]);
      setWorkflowState(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 160px)" }}>
      {/* Messages Scroll Area */}
      <div style={{ flex: 1, overflowY: "auto", paddingRight: "8px", display: "flex", flexDirection: "column", gap: "20px" }}>
        {fetchingHistory ? (
          <div style={{ display: "flex", justifyContent: "center", padding: "40px" }}>
            <div className="spinner" />
          </div>
        ) : history.length === 0 ? (
          <div className="glass-card" style={{ padding: "40px 24px", textAlign: "center", margin: "auto 0" }}>
            <div style={{ fontSize: "2.5rem", marginBottom: "12px" }}>💡</div>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: "8px" }}>
              Ask anything about &quot;{datasetName}&quot;
            </h3>
            <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginBottom: "20px", maxWidth: "500px", margin: "0 auto 20px" }}>
              The Autonomous Supervisor will analyze your question and route it to the appropriate agent.
            </p>

            <div style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "12px" }}>
              Suggested questions:
            </div>

            {/* Quick Sample Questions */}
            <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: "8px", maxWidth: "680px", margin: "0 auto" }}>
              {SAMPLE_QUESTIONS.map((sq, i) => (
                <button
                  key={i}
                  onClick={() => handleSubmit(undefined, sq)}
                  style={{
                    padding: "8px 14px",
                    borderRadius: "20px",
                    background: "var(--bg-input)",
                    border: "1px solid var(--border)",
                    color: "var(--primary-light)",
                    fontSize: "0.8rem",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                  }}
                  onMouseOver={(e) => (e.currentTarget.style.borderColor = "var(--primary)")}
                  onMouseOut={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
                >
                  ✨ {sq}
                </button>
              ))}
            </div>
          </div>
        ) : (
          history.map((item, idx) => {
            if (item.isUser) {
              return (
                <div key={item.id || idx} style={{ alignSelf: "flex-end", maxWidth: "80%" }}>
                  <div style={{ background: "var(--gradient-brand)", color: "white", padding: "12px 18px", borderRadius: "18px 18px 2px 18px", fontSize: "0.9rem", fontWeight: 500 }}>
                    {item.question}
                  </div>
                </div>
              );
            }

            if (item.error) {
              return (
                <div key={item.id || idx} style={{ alignSelf: "flex-start", maxWidth: "80%" }}>
                  <div style={{ background: "rgba(239, 68, 68, 0.15)", border: "1px solid rgba(239, 68, 68, 0.3)", color: "#fca5a5", padding: "12px 18px", borderRadius: "18px", fontSize: "0.85rem" }}>
                    ❌ {item.error}
                  </div>
                </div>
              );
            }

            const qInfo = item.query || {};
            const rInfo = item.report || {};
            const chartBaseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
            const chartPaths = (rInfo.chart_paths || []).filter((path: string) =>
              /\.(png|jpe?g|svg)$/i.test(path)
            );

            return (
              <div key={qInfo.id || idx} style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                {/* User Prompt */}
                <div style={{ alignSelf: "flex-end", maxWidth: "80%" }}>
                  <div style={{ background: "var(--gradient-brand)", color: "white", padding: "12px 18px", borderRadius: "18px 18px 2px 18px", fontSize: "0.9rem", fontWeight: 500 }}>
                    {qInfo.question}
                  </div>
                </div>

                {/* AI Response Card */}
                <div className="glass-card" style={{ alignSelf: "flex-start", maxWidth: "90%", width: "100%", padding: "20px" }}>
                  {/* Visual Agent Workflow Execution Pipeline */}
                  <AgentWorkflow state={item.workflow || getHistoricalWorkflowState(qInfo, rInfo)} />

                  {/* Agent execution badges */}
                  <AgentStatusBadge agents={rInfo.agents_used || []} />

                  {(() => {
                    const msgKey = String(qInfo.id || idx);
                    const isUnrelated = (
                      rInfo.response_type === "UNRELATED"
                      || rInfo.answer?.toLowerCase().includes("not related to the currently loaded dataset")
                    );
                    const hasChart = chartPaths.length > 0;
                    const isVisualization = (
                      rInfo.response_type === "VISUALIZATION"
                      || Boolean(rInfo.visualization?.should_visualize)
                      || hasChart
                    );
                    const isScalar = (
                      !isVisualization && (
                        rInfo.response_type === "SCALAR"
                        || Boolean(rInfo.kpi)
                        || (rInfo.data_used?.rows?.length === 1 && rInfo.data_used?.columns?.length === 1)
                      )
                    );
                    const hasTableData = Boolean(rInfo.data_used?.rows && rInfo.data_used.rows.length > 0);
                    const currentView = activeViews[msgKey] || (hasChart ? "chart" : "table");

                    if (isUnrelated) {
                      return (
                        <div
                          style={{
                            background: "rgba(245, 158, 11, 0.12)",
                            border: "1px solid rgba(245, 158, 11, 0.3)",
                            borderRadius: "14px",
                            padding: "18px 22px",
                            margin: "12px 0",
                          }}
                        >
                          <div
                            style={{
                              fontSize: "0.95rem",
                              fontWeight: 700,
                              color: "#fcd34d",
                              marginBottom: "8px",
                              display: "flex",
                              alignItems: "center",
                              gap: "8px",
                            }}
                          >
                            <span>⚠️</span> Question Not Related to Dataset
                          </div>
                          <div style={{ color: "#fef3c7", fontSize: "0.9rem", lineHeight: 1.6 }}>
                            {rInfo.answer}
                          </div>
                          <div style={{ color: "#fde68a", fontSize: "0.8rem", marginTop: "10px" }}>
                            Please ask a question related to columns and records in &quot;{datasetName}&quot;.
                          </div>
                        </div>
                      );
                    }

                    if (isScalar) {
                      return (
                        <KpiCard
                          label={rInfo.kpi?.label || rInfo.data_used?.columns?.[0] || "Total Result"}
                          value={rInfo.kpi?.value ?? (rInfo.data_used?.rows?.[0]?.[rInfo.data_used?.columns?.[0]] ?? rInfo.answer)}
                          explanation={rInfo.explanation}
                          insights={rInfo.insights || []}
                        />
                      );
                    }

                    return (
                      <div>
                        {/* Main Answer Header */}
                        <div style={{ fontSize: "0.95rem", fontWeight: 500, lineHeight: 1.6, color: "var(--text-primary)", margin: "12px 0" }}>
                          {rInfo.answer || "Analysis complete."}
                        </div>

                        {/* Chart View */}
                        {hasChart && currentView === "chart" && (
                          <section style={{ margin: "16px 0" }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
                              <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--accent)", display: "flex", alignItems: "center", gap: "6px" }}>
                                <span>📊</span> Visual Analysis & Interactive Plotly
                              </div>
                              {hasTableData && (
                                <button
                                  onClick={() => setActiveViews((prev) => ({ ...prev, [msgKey]: "table" }))}
                                  style={{
                                    padding: "5px 12px",
                                    borderRadius: "8px",
                                    background: "rgba(255, 255, 255, 0.06)",
                                    border: "1px solid rgba(255, 255, 255, 0.12)",
                                    color: "var(--primary-light)",
                                    fontSize: "0.78rem",
                                    fontWeight: 600,
                                    cursor: "pointer",
                                  }}
                                >
                                  📋 View as Table
                                </button>
                              )}
                            </div>
                            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                              {chartPaths.map((path: string, chartIndex: number) => {
                                const vizResult = rInfo.visualization_result || (rInfo.data_used && rInfo.data_used.visualization);
                                const params = rInfo.parameters || (rInfo.data_used && rInfo.data_used.parameters) || vizResult?.parameters;
                                const explanation = vizResult?.explanation || rInfo.explanation;
                                const chartJson = rInfo.chart_json || (rInfo.data_used && rInfo.data_used.chart_json);

                                return (
                                  <PlotlyChart
                                    key={path}
                                    chartPath={path}
                                    chartIndex={chartIndex}
                                    chartJson={chartJson}
                                    parameters={params}
                                    explanation={explanation}
                                    baseUrl={chartBaseUrl}
                                  />
                                );
                              })}
                            </div>
                          </section>
                        )}

                        {/* Table View */}
                        {hasTableData && (currentView === "table" || !hasChart) && (
                          <InteractiveTable
                            columns={rInfo.data_used.columns || []}
                            rows={rInfo.data_used.rows || []}
                            totalRowCount={rInfo.data_used.row_count}
                            explanation={rInfo.explanation}
                            isChartViewAvailable={hasChart}
                            isChartViewActive={currentView === "chart"}
                            onToggleView={() => setActiveViews((prev) => ({ ...prev, [msgKey]: "chart" }))}
                          />
                        )}

                        {/* Key Insights (when not shown inside chart) */}
                        {rInfo.insights && rInfo.insights.length > 0 && !hasChart && (
                          <div style={{ background: "var(--bg-input)", padding: "12px 16px", borderRadius: "10px", margin: "12px 0", border: "1px solid var(--border)" }}>
                            <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--accent)", marginBottom: "6px" }}>
                              📌 Key Insights
                            </div>
                            <ul style={{ paddingLeft: "20px", fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                              {rInfo.insights.map((ins: string, i: number) => (
                                <li key={i} style={{ marginBottom: "4px" }}>{ins}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    );
                  })()}

                  {/* SQL Query Disclosure */}
                  {rInfo.data_used && rInfo.data_used.query && (
                    <div style={{ marginTop: "14px" }}>
                      <details style={{ fontSize: "0.8rem" }}>
                        <summary style={{ cursor: "pointer", color: "var(--text-muted)", fontWeight: 500 }}>
                          🔍 View Executed SQL Query
                        </summary>
                        <pre style={{ background: "#0b0b18", padding: "10px 14px", borderRadius: "8px", color: "var(--accent-light)", marginTop: "6px", overflowX: "auto" }}>
                          {rInfo.data_used.query}
                        </pre>
                      </details>
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}

        {/* Loading Indicator with live Agent Workflow diagram */}
        {loading && workflowState && (
          <div style={{ alignSelf: "flex-start", width: "100%", maxWidth: "90%" }}>
            <AgentWorkflow state={workflowState} />
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Bar */}
      <form onSubmit={(e) => handleSubmit(e)} style={{ marginTop: "16px", display: "flex", gap: "10px" }}>
        <input
          type="text"
          className="input-field"
          placeholder="Ask a question about your dataset..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
        />
        <button type="submit" className="btn-primary" disabled={!question.trim() || loading} style={{ padding: "12px 24px" }}>
          {loading ? <div className="spinner" style={{ width: 18, height: 18 }} /> : "Ask AI"}
        </button>
      </form>
    </div>
  );
}
