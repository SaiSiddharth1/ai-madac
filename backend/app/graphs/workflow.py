"""
LangGraph Multi-Agent Workflow Definition.
Builds and compiles the StateGraph orchestrating Supervisor, SQL, Analyst, ML, Visualization, RAG, and Report agents.
"""

import logging
from typing import Any

from langgraph.graph import StateGraph, START, END

from app.agents.state import AgentState
from app.agents.supervisor.agent import supervisor_agent
from app.agents.sql.agent import sql_agent
from app.agents.data_analyst.agent import data_analyst_agent
from app.agents.ml.agent import ml_agent
from app.agents.visualization.agent import visualization_agent
from app.agents.rag.agent import rag_agent
from app.agents.report.agent import report_agent

logger = logging.getLogger(__name__)


# Node functions
from app.core.workflow_events import workflow_event_manager


def _record_agent_completion(state: AgentState, result: dict[str, Any], agent_name: str) -> dict[str, Any]:
    """Mark an agent as visited even when it intentionally has no output.

    Routing uses ``agents_used`` to decide the next node.  Some agents can
    validly return no artifact (for example, a chart is skipped for one KPI).
    Without this marker the router selects that same agent indefinitely.
    """
    completed = result.get("agents_used", state.get("agents_used", []))
    result["agents_used"] = list(dict.fromkeys([*completed, agent_name]))
    return result

async def supervisor_node(state: AgentState) -> dict[str, Any]:
    """Execute Supervisor Agent for intent classification and plan routing."""
    query_id = state.get("query_id", "")
    await workflow_event_manager.emit(query_id, "supervisor_agent", "running", "Understanding query and routing plan")
    res = supervisor_agent.run(state)
    
    intent = res.get("intent", "DATA_RETRIEVAL")
    plan = res.get("execution_plan", [])
    name_map = {
        "sql_agent": "SQL Agent",
        "data_analyst_agent": "Data Analyst Agent",
        "ml_agent": "ML Agent",
        "visualization_agent": "Visualization Agent",
        "rag_agent": "RAG Agent",
        "report_agent": "Report Agent"
    }
    plan_readable = [name_map.get(a, a) for a in plan if a != "supervisor_agent"]
    selected_msg = f"Intent: {intent}. Selected: {', '.join(plan_readable)}"
    await workflow_event_manager.emit(
        query_id, "supervisor_agent", "completed", selected_msg, intent=intent, execution_plan=plan
    )
    return res


async def sql_node(state: AgentState) -> dict[str, Any]:
    """Execute SQL Agent for NL-to-SQL generation and execution."""
    query_id = state.get("query_id", "")
    await workflow_event_manager.emit(query_id, "sql_agent", "running", "Generating SQL and fetching records")
    res = _record_agent_completion(state, sql_agent.run(state), "sql_agent")
    errors = res.get("errors", [])
    if errors:
        await workflow_event_manager.emit(query_id, "sql_agent", "failed", f"SQL execution error: {errors[0]}")
    else:
        await workflow_event_manager.emit(query_id, "sql_agent", "completed", "Data retrieved successfully")
    return res


async def data_analyst_node(state: AgentState) -> dict[str, Any]:
    """Execute Data Analyst Agent node."""
    query_id = state.get("query_id", "")
    await workflow_event_manager.emit(query_id, "data_analyst_agent", "running", "Performing statistical and relationship analysis")
    res = _record_agent_completion(state, data_analyst_agent.run(state), "data_analyst_agent")
    errors = res.get("errors", [])
    if errors:
        await workflow_event_manager.emit(query_id, "data_analyst_agent", "failed", f"Analysis error: {errors[0]}")
    else:
        await workflow_event_manager.emit(query_id, "data_analyst_agent", "completed", "Statistical analysis completed")
    return res


async def ml_node(state: AgentState) -> dict[str, Any]:
    """Execute ML Agent node for model training or predictions."""
    query_id = state.get("query_id", "")
    await workflow_event_manager.emit(query_id, "ml_agent", "running", "Training predictive model and forecasting")
    res = _record_agent_completion(state, ml_agent.run(state), "ml_agent")
    errors = res.get("errors", [])
    if errors:
        await workflow_event_manager.emit(query_id, "ml_agent", "failed", f"ML execution error: {errors[0]}")
    else:
        await workflow_event_manager.emit(query_id, "ml_agent", "completed", "Prediction completed")
    return res


async def visualization_node(state: AgentState) -> dict[str, Any]:
    """Execute Visualization Agent node for chart generation."""
    query_id = state.get("query_id", "")
    await workflow_event_manager.emit(query_id, "visualization_agent", "running", "Generating interactive chart visualization")
    res = _record_agent_completion(state, visualization_agent.run(state), "visualization_agent")
    errors = res.get("errors", [])
    if errors:
        await workflow_event_manager.emit(query_id, "visualization_agent", "failed", f"Visualization error: {errors[0]}")
    else:
        await workflow_event_manager.emit(query_id, "visualization_agent", "completed", "Chart generated successfully")
    return res


async def rag_node(state: AgentState) -> dict[str, Any]:
    """Execute RAG Agent node for semantic knowledge retrieval."""
    query_id = state.get("query_id", "")
    await workflow_event_manager.emit(query_id, "rag_agent", "running", "Searching and retrieving semantic knowledge")
    res = _record_agent_completion(state, rag_agent.run(state), "rag_agent")
    errors = res.get("errors", [])
    if errors:
        await workflow_event_manager.emit(query_id, "rag_agent", "failed", f"RAG search error: {errors[0]}")
    else:
        await workflow_event_manager.emit(query_id, "rag_agent", "completed", "Knowledge retrieved successfully")
    return res


async def report_node(state: AgentState) -> dict[str, Any]:
    """Execute Report Agent for final answer and insight synthesis."""
    query_id = state.get("query_id", "")
    await workflow_event_manager.emit(query_id, "report_agent", "running", "Synthesizing answers and warnings into final report")
    res = _record_agent_completion(state, report_agent.run(state), "report_agent")
    errors = res.get("errors", [])
    if errors:
        await workflow_event_manager.emit(query_id, "report_agent", "failed", f"Synthesis error: {errors[0]}")
    else:
        await workflow_event_manager.emit(query_id, "report_agent", "completed", "Final report synthesized successfully")
    return res


# Conditional Router function
def route_next(state: AgentState) -> str:
    """
    Determine next node to execute based on execution plan.
    """
    plan = state.get("execution_plan", [])
    agents_used = state.get("agents_used", [])

    logger.debug(f"Routing check: plan={plan}, agents_used={agents_used}")

    for agent in plan:
        if agent not in agents_used and agent != "supervisor_agent":
            if agent == "rag_agent":
                return "rag_agent"
            elif agent == "sql_agent":
                return "sql_agent"
            elif agent == "data_analyst_agent":
                return "data_analyst_agent"
            elif agent == "ml_agent":
                return "ml_agent"
            elif agent == "visualization_agent":
                return "visualization_agent"
            elif agent == "report_agent":
                return "report_agent"

    if "report_agent" not in agents_used:
        return "report_agent"

    return END


def create_workflow() -> Any:
    """
    Build and compile the LangGraph StateGraph workflow.
    """
    builder = StateGraph(AgentState)

    # Add Nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("rag_agent", rag_node)
    builder.add_node("sql_agent", sql_node)
    builder.add_node("data_analyst_agent", data_analyst_node)
    builder.add_node("ml_agent", ml_node)
    builder.add_node("visualization_agent", visualization_node)
    builder.add_node("report_agent", report_node)

    # Entry point
    builder.add_edge(START, "supervisor")

    # Dynamic routing edges from nodes using route_next
    for node_name in ["supervisor", "rag_agent", "sql_agent", "data_analyst_agent", "ml_agent", "visualization_agent"]:
        builder.add_conditional_edges(
            node_name,
            route_next,
            {
                "rag_agent": "rag_agent",
                "sql_agent": "sql_agent",
                "data_analyst_agent": "data_analyst_agent",
                "ml_agent": "ml_agent",
                "visualization_agent": "visualization_agent",
                "report_agent": "report_agent",
                END: END,
            },
        )

    # Report agent leads to END
    builder.add_edge("report_agent", END)

    compiled_graph = builder.compile()
    logger.info("Complete 6-agent LangGraph workflow compiled successfully.")
    return compiled_graph


# Pre-compiled workflow graph
workflow_graph = create_workflow()
