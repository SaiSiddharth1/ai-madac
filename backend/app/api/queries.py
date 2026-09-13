"""
Queries API endpoints — execute natural language queries and list query history.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_db, get_current_user
from app.db.models import User, Query, Report
from app.schemas.query import (
    QueryRequest,
    QueryResponse,
    ReportResponse,
    AgentRunResponse,
    FullQueryResponse,
)
from app.services.query_service import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/datasets", tags=["Queries"])


@router.post("/{dataset_id}/query", response_model=FullQueryResponse)
async def run_query(
    dataset_id: UUID,
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Execute a natural language query on a dataset.
    Routes query through Supervisor → Specialized Agents → Report.
    """
    result = await execute_query(
        user_id=current_user.id,
        dataset_id=dataset_id,
        question=body.question,
        db=db,
    )

    query_obj = result["query"]
    report_obj = result["report"]
    agents_list = result["agents"]

    return FullQueryResponse(
        query=QueryResponse.model_validate(query_obj),
        report=ReportResponse.model_validate(report_obj),
        agents=[AgentRunResponse.model_validate(a) for a in agents_list],
    )


@router.post("/{dataset_id}/query/stream")
async def run_query_stream(
    dataset_id: UUID,
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Execute a natural language query on a dataset and stream execution events in real time.
    """
    import json
    import asyncio
    from typing import AsyncGenerator
    from fastapi.responses import StreamingResponse
    from app.core.workflow_events import workflow_event_manager
    from app.graphs.workflow import workflow_graph
    from app.db.models import Dataset, DatasetProfile, AgentRun, Report

    # 1. Fetch dataset & check ownership
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == current_user.id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found or access denied",
        )

    # Fetch dataset profile
    profile_result = await db.execute(
        select(DatasetProfile).where(DatasetProfile.dataset_id == dataset_id)
    )
    profile_record = profile_result.scalar_one_or_none()
    profile_json = profile_record.profile_json if profile_record else {}

    # Create Query record
    query_record = Query(
        user_id=current_user.id,
        dataset_id=dataset_id,
        question=body.question,
    )
    db.add(query_record)
    await db.commit()
    await db.refresh(query_record)

    query_id_str = str(query_record.id)
    queue = workflow_event_manager.register(query_id_str)

    async def event_generator() -> AsyncGenerator[str, None]:
        async def run_workflow():
            try:
                # Construct initial AgentState
                initial_state = {
                    "user_id": str(current_user.id),
                    "dataset_id": str(dataset_id),
                    "query_id": query_id_str,
                    "question": body.question,
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
                
                # Execute LangGraph workflow
                final_state = await workflow_graph.ainvoke(initial_state)
                
                # Save agent runs & report
                query_record.intent = final_state.get("intent", "DATA_RETRIEVAL")
                db.add(query_record)
                
                agents_invoked = final_state.get("agents_used", [])
                agent_run_models = []
                for agent_name in agents_invoked:
                    ar = AgentRun(
                        query_id=query_record.id,
                        agent_name=agent_name,
                        status="success" if not final_state.get("errors") else "completed_with_errors",
                        execution_time_ms=0.0,
                    )
                    db.add(ar)
                    agent_run_models.append(ar)
                
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
                await db.refresh(query_record)
                await db.refresh(report_record)
                
                # Format final response
                response_obj = FullQueryResponse(
                    query=QueryResponse.model_validate(query_record),
                    report=ReportResponse.model_validate(report_record) if report_record else None,
                    agents=[AgentRunResponse.model_validate(ar) for ar in agent_run_models],
                )
                
                # Emit final completed event containing result
                await workflow_event_manager.emit(
                    query_id_str,
                    "report_agent",
                    "completed",
                    "Final report synthesized successfully",
                    # SSE events are encoded with ``json.dumps`` below.  Use
                    # Pydantic's JSON mode here so UUIDs and datetimes from the
                    # persisted query/report are converted before they reach
                    # the stream.  Without this, the workflow completes but
                    # the response stream disconnects while sending its final
                    # result (``Object of type UUID is not JSON serializable``).
                    result=response_obj.model_dump(mode="json")
                )
            except Exception as e:
                logger.error(f"Error in streaming workflow execution: {e}", exc_info=True)
                await workflow_event_manager.emit(
                    query_id_str,
                    "report_agent",
                    "failed",
                    f"Workflow execution failed: {str(e)}"
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_workflow())
        
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                # Keep SSE payloads JSON-safe even if a future agent emits a
                # richer Python value in its progress metadata.
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except asyncio.CancelledError:
            task.cancel()
            logger.warning(f"Streaming query {query_id_str} was cancelled by client.")
        finally:
            workflow_event_manager.unregister(query_id_str)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{dataset_id}/queries", response_model=list[FullQueryResponse])
async def list_dataset_queries(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get all past queries for a specific dataset.
    """
    result = await db.execute(
        select(Query)
        .options(selectinload(Query.report), selectinload(Query.agent_runs))
        .where(Query.dataset_id == dataset_id, Query.user_id == current_user.id)
        .order_by(Query.created_at.desc())
    )
    queries = result.scalars().all()

    response_list = []
    for q in queries:
        response_list.append(
            FullQueryResponse(
                query=QueryResponse.model_validate(q),
                report=ReportResponse.model_validate(q.report) if q.report else None,
                agents=[AgentRunResponse.model_validate(a) for a in q.agent_runs],
            )
        )

    return response_list
