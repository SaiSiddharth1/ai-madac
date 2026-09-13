"""
Pydantic schemas for query and report endpoints.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Natural language query request."""
    question: str = Field(..., min_length=3, max_length=2000)


class QueryResponse(BaseModel):
    """Query metadata."""
    id: UUID
    dataset_id: UUID
    question: str
    intent: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AgentRunResponse(BaseModel):
    """Individual agent execution info."""
    agent_name: str
    status: str
    execution_time_ms: Optional[float] = None

    model_config = {"from_attributes": True}


class ReportResponse(BaseModel):
    """Final report returned to client."""
    id: UUID
    query_id: UUID
    answer: str
    insights: Optional[list[str]] = None
    chart_paths: Optional[list[str]] = None
    data_used: Optional[dict[str, Any]] = None
    agents_used: Optional[list[str]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class FullQueryResponse(BaseModel):
    """Complete response including report and agent info."""
    query: QueryResponse
    report: Optional[ReportResponse] = None
    agents: list[AgentRunResponse] = []
