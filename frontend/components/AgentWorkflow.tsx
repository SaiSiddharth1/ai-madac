"use client";

import React from "react";

export interface AgentState {
  status: "waiting" | "running" | "completed" | "failed" | "skipped";
  message: string;
}

export interface WorkflowState {
  intent: string;
  executionPlan: string[];
  agents: Record<string, AgentState>;
}

interface AgentWorkflowProps {
  state: WorkflowState;
}

// Mapping of agent IDs to icons and readable names
const AGENT_META: Record<string, { name: string; icon: string; color: string }> = {
  supervisor_agent: { name: "Supervisor Agent", icon: "🧠", color: "var(--accent)" },
  sql_agent: { name: "SQL Agent", icon: "🗄️", color: "#6366f1" },
  data_analyst_agent: { name: "Data Analyst Agent", icon: "📊", color: "#10b981" },
  ml_agent: { name: "ML Agent", icon: "🤖", color: "#8b5cf6" },
  visualization_agent: { name: "Visualization Agent", icon: "📈", color: "#f59e0b" },
  rag_agent: { name: "RAG Agent", icon: "📚", color: "#ec4899" },
  report_agent: { name: "Report Agent", icon: "📝", color: "#06b6d4" },
};

export default function AgentWorkflow({ state }: AgentWorkflowProps) {
  const { intent, executionPlan, agents } = state;

  // Helper to render node badge/card
  const renderNode = (agentId: string) => {
    const meta = AGENT_META[agentId];
    if (!meta) return null;

    const agentState = agents[agentId] || { status: "waiting", message: "Waiting" };
    const { status, message } = agentState;

    let borderClass = "border-zinc-800";
    let bgStyle = "rgba(15, 15, 27, 0.6)";
    let badgeColor = "bg-zinc-700 text-zinc-300";
    let statusText = "Waiting";
    let pulseClass = "";

    if (status === "running") {
      borderClass = "border-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.25)]";
      bgStyle = "rgba(30, 30, 60, 0.8)";
      badgeColor = "bg-indigo-600 text-white animate-pulse";
      statusText = "Running...";
      pulseClass = "animate-pulse";
    } else if (status === "completed") {
      borderClass = "border-emerald-500/50 shadow-[0_0_15px_rgba(16,185,129,0.15)]";
      bgStyle = "rgba(10, 35, 25, 0.7)";
      badgeColor = "bg-emerald-600/90 text-white";
      statusText = "Completed";
    } else if (status === "failed") {
      borderClass = "border-rose-500 shadow-[0_0_15px_rgba(239,68,68,0.25)]";
      bgStyle = "rgba(45, 15, 15, 0.7)";
      badgeColor = "bg-rose-600 text-white";
      statusText = "Failed";
    } else if (status === "skipped") {
      borderClass = "border-zinc-900/50 opacity-40";
      bgStyle = "rgba(5, 5, 10, 0.3)";
      badgeColor = "bg-zinc-800 text-zinc-500";
      statusText = "Skipped";
    }

    return (
      <div
        key={agentId}
        className={`flex flex-col p-4 rounded-xl border backdrop-blur-md transition-all duration-300 ${borderClass}`}
        style={{
          background: bgStyle,
          minWidth: "220px",
          flex: 1,
        }}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="text-xl">{meta.icon}</span>
            <span className="font-bold text-sm text-zinc-100">{meta.name}</span>
          </div>
          <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase ${badgeColor}`}>
            {statusText}
          </span>
        </div>
        <p className={`text-xs text-zinc-400 leading-relaxed mt-1 ${pulseClass}`}>
          {message}
        </p>
      </div>
    );
  };

  // Check if parallel layout should be used (both SQL and Analyst are in the execution plan)
  const isParallel = executionPlan.includes("sql_agent") && executionPlan.includes("data_analyst_agent");

  // Determine path nodes based on execution plan
  const activePlan = ["supervisor_agent", ...executionPlan.filter((a) => a !== "supervisor_agent")];

  return (
    <div className="glass-card w-full p-6 mb-6 overflow-hidden animate-fade-in">
      <div className="flex items-center justify-between border-b border-zinc-800 pb-4 mb-6">
        <div>
          <h2 className="text-lg font-extrabold text-zinc-100 flex items-center gap-2">
            <span className="animate-spin text-indigo-400">⚙️</span> Analysis Workflow Pipeline
          </h2>
          <p className="text-xs text-zinc-400 mt-1">
            Real-time multi-agent execution tracking via LangGraph
          </p>
        </div>
        {intent && (
          <div className="flex items-center gap-2 bg-indigo-500/10 border border-indigo-500/20 px-3 py-1.5 rounded-lg">
            <span className="text-xs font-bold text-zinc-400 uppercase">Intent:</span>
            <span className="text-xs font-black text-indigo-300 uppercase tracking-wider">{intent}</span>
          </div>
        )}
      </div>

      <div className="flex flex-col items-center gap-4 py-2">
        {/* Step 1: Supervisor */}
        {renderNode("supervisor_agent")}

        {/* Arrow below Supervisor */}
        <div className="flex flex-col items-center">
          <div className="w-0.5 h-6 bg-zinc-800 relative">
            <div className="absolute inset-0 bg-indigo-500 animate-pulse" />
          </div>
          <div className="text-zinc-600 text-xs mt-[-2px]">▼</div>
        </div>

        {/* Parallel Branches */}
        {isParallel ? (
          <div className="w-full flex flex-col items-center gap-4">
            {/* Split connectors */}
            <div className="w-3/4 flex justify-between relative px-8">
              <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-zinc-800 -translate-x-1/2" />
              <div className="w-1/2 border-t-2 border-l-2 border-zinc-800 h-6 rounded-tl-xl" />
              <div className="w-1/2 border-t-2 border-r-2 border-zinc-800 h-6 rounded-tr-xl" />
            </div>

            {/* Parallel nodes container */}
            <div className="w-full flex gap-4 justify-center flex-wrap px-4">
              {renderNode("sql_agent")}
              {renderNode("data_analyst_agent")}
            </div>

            {/* Merge connectors */}
            <div className="w-3/4 flex justify-between relative px-8 h-6">
              <div className="w-1/2 border-b-2 border-l-2 border-zinc-800 h-6 rounded-bl-xl" />
              <div className="w-1/2 border-b-2 border-r-2 border-zinc-800 h-6 rounded-br-xl" />
              <div className="absolute top-6 bottom-0 left-1/2 w-0.5 bg-zinc-800 -translate-x-1/2" />
            </div>

            {/* Arrow below Parallel */}
            <div className="flex flex-col items-center mt-4">
              <div className="text-zinc-600 text-xs">▼</div>
            </div>
          </div>
        ) : (
          /* Sequential display for remaining selected agents (except supervisor/report/parallel) */
          executionPlan
            .filter((agent) => agent !== "supervisor_agent" && agent !== "report_agent")
            .map((agent) => (
              <React.Fragment key={agent}>
                {renderNode(agent)}
                <div className="flex flex-col items-center">
                  <div className="w-0.5 h-6 bg-zinc-800" />
                  <div className="text-zinc-600 text-xs mt-[-2px]">▼</div>
                </div>
              </React.Fragment>
            ))
        )}

        {/* Step 3: Visualization (rendered if part of parallel plan) */}
        {isParallel && executionPlan.includes("visualization_agent") && (
          <>
            {renderNode("visualization_agent")}
            <div className="flex flex-col items-center">
              <div className="w-0.5 h-6 bg-zinc-800" />
              <div className="text-zinc-600 text-xs mt-[-2px]">▼</div>
            </div>
          </>
        )}

        {/* Step 4: Report Agent */}
        {executionPlan.includes("report_agent") && renderNode("report_agent")}
      </div>
    </div>
  );
}
