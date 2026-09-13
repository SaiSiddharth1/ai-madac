"""
Query Service.
Orchestrates natural language query execution through the LangGraph agent workflow
and persists query, agent runs, and generated report to PostgreSQL.
"""

import logging
import time
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.state import AgentState
from app.db.models import Dataset, DatasetProfile, Query, AgentRun, Report
from app.graphs.workflow import workflow_graph

logger = logging.getLogger(__name__)


async def execute_query(
    user_id: UUID,
    dataset_id: UUID,
    question: str,
    db: AsyncSession,
) -> dict:
    """
    Execute a natural language query on a dataset through the multi-agent graph.

    Steps:
    1. Verify dataset ownership
    2. Retrieve dataset schema & cached profile
    3. Create Query record in database
    4. Invoke LangGraph workflow
    5. Save AgentRun execution metadata & Report in database
    """
    # 1. Fetch dataset & check ownership
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == user_id)
    )
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found or access denied",
        )

    # 2. Fetch dataset profile if available
    profile_result = await db.execute(
        select(DatasetProfile).where(DatasetProfile.dataset_id == dataset_id)
    )
    profile_record = profile_result.scalar_one_or_none()
    profile_json = profile_record.profile_json if profile_record else {}

    # 3. Create Query record
    query_record = Query(
        user_id=user_id,
        dataset_id=dataset_id,
        question=question,
    )
    db.add(query_record)
    await db.commit()
    await db.refresh(query_record)

    # 4. Construct initial AgentState
    initial_state: AgentState = {
        "user_id": str(user_id),
        "dataset_id": str(dataset_id),
        "query_id": str(query_record.id),
        "question": question,
        "table_name": dataset.table_name,
        "schema": dataset.schema_info or {},
        "dataset_profile": profile_json,
        "intent": "",
        "execution_plan": [],
        "sql_query": None,
        "sql_result": None,
        "analysis_result": None,
        "ml_result": None,
        "visualization_result": None,
        "rag_result": None,
        "final_report": None,
        "errors": [],
        "agents_used": [],
    }

    # 5. Execute LangGraph workflow
    start_time = time.time()
    logger.info(f"Invoking LangGraph workflow for query {query_record.id}...")

    try:
        final_state = await workflow_graph.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"LangGraph execution error for query {query_record.id}: {e}")
        final_state = initial_state
        final_state["errors"] = [f"Agent Workflow Execution Error: {str(e)}"]

    total_time_ms = round((time.time() - start_time) * 1000, 2)

    # Update Query record intent
    query_record.intent = final_state.get("intent", "DATA_RETRIEVAL")
    db.add(query_record)

    # 6. Save AgentRun records
    agents_invoked = final_state.get("agents_used", [])
    agent_run_models = []
    for agent_name in agents_invoked:
        ar = AgentRun(
            query_id=query_record.id,
            agent_name=agent_name,
            status="success" if not final_state.get("errors") else "completed_with_errors",
            execution_time_ms=total_time_ms / max(len(agents_invoked), 1),
        )
        db.add(ar)
        agent_run_models.append(ar)

    # 7. Save Report record
    report_dict = final_state.get("final_report") or {
        "answer": "Query executed, but no response was generated.",
        "insights": [],
        "chart_paths": [],
        "data_used": final_state.get("sql_result"),
        "agents_used": agents_invoked,
    }

    report_record = Report(
        query_id=query_record.id,
        answer=report_dict.get("answer", ""),
        insights=report_dict.get("insights", []),
        chart_paths=report_dict.get("chart_paths", []),
        data_used=report_dict.get("data_used"),
        agents_used=agents_invoked,
    )
    db.add(report_record)
    await db.commit()
    await db.refresh(report_record)

    logger.info(f"Query {query_record.id} completed successfully in {total_time_ms}ms")

    return {
        "query": query_record,
        "report": report_record,
        "agents": agent_run_models,
    }
