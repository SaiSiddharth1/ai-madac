"""
Pydantic schemas for query and report endpoints.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class QueryRequest(BaseModel):
    """Natural language query request."""
    question: str = Field(..., min_length=3, max_length=2000)
    session_id: Optional[str] = Field(None, min_length=1, max_length=128)


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
    chart_json: Optional[dict[str, Any]] = None
    parameters: Optional[dict[str, Any]] = None
    visualization_result: Optional[dict[str, Any]] = None
    visualization: Optional[dict[str, Any]] = None
    response_type: Optional[str] = None
    kpi: Optional[dict[str, Any]] = None
    explanation: Optional[dict[str, Any]] = None
    data_used: Optional[dict[str, Any]] = None
    agents_used: Optional[list[str]] = None
    created_at: datetime

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def populate_visualization_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            d_used = data.get("data_used") or {}
            if isinstance(d_used, dict):
                data.setdefault("chart_json", d_used.get("chart_json"))
                data.setdefault("parameters", d_used.get("parameters"))
                data.setdefault("visualization_result", d_used.get("visualization"))
                data.setdefault("visualization", d_used.get("visualization_meta") or data.get("visualization"))
                data.setdefault("response_type", d_used.get("response_type") or data.get("response_type"))
                data.setdefault("kpi", d_used.get("kpi"))
                data.setdefault("explanation", d_used.get("explanation"))
            return data

        d_used = getattr(data, "data_used", None) or {}
        if isinstance(d_used, dict):
            chart_json = getattr(data, "chart_json", None) or d_used.get("chart_json")
            parameters = getattr(data, "parameters", None) or d_used.get("parameters")
            visualization_result = getattr(data, "visualization_result", None) or d_used.get("visualization")
            visualization = getattr(data, "visualization", None) or d_used.get("visualization_meta")
            response_type = getattr(data, "response_type", None) or d_used.get("response_type")
            kpi = getattr(data, "kpi", None) or d_used.get("kpi")
            explanation = getattr(data, "explanation", None) or d_used.get("explanation")
            return {
                "id": getattr(data, "id", None),
                "query_id": getattr(data, "query_id", None),
                "answer": getattr(data, "answer", ""),
                "insights": getattr(data, "insights", []),
                "chart_paths": getattr(data, "chart_paths", []),
                "chart_json": chart_json,
                "parameters": parameters,
                "visualization_result": visualization_result,
                "visualization": visualization,
                "response_type": response_type,
                "kpi": kpi,
                "explanation": explanation,
                "data_used": d_used,
                "agents_used": getattr(data, "agents_used", []),
                "created_at": getattr(data, "created_at", None),
            }
        return data


class FullQueryResponse(BaseModel):
    """Complete response including report and agent info."""
    query: QueryResponse
    report: Optional[ReportResponse] = None
    agents: list[AgentRunResponse] = []
