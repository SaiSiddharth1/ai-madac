"""
Agent State definition for LangGraph workflow.
Shared typed state passed through all agent nodes.
"""

from typing import Any, Optional
from typing_extensions import TypedDict


class DatasetContext(TypedDict, total=False):
    """Persistent dataset metadata, schema, and configuration."""
    dataset_id: str
    table_name: str
    schema: dict[str, Any]
    dataset_profile: dict[str, Any]


class QuestionAnalysisContext(TypedDict, total=False):
    """Transient question-analysis fields. Reset completely for every new question."""
    query_id: str
    question: str
    intent: str
    query_contract: Optional[dict[str, Any]]
    analysis_spec: Optional[dict[str, Any]]
    structured_intent: dict[str, Any]
    execution_plan: list[str]
    sql_query: Optional[str]
    sql_result: Optional[dict[str, Any]]
    analysis_result: Optional[dict[str, Any]]
    ml_result: Optional[dict[str, Any]]
    visualization_result: Optional[dict[str, Any]]
    rag_result: Optional[dict[str, Any]]
    final_report: Optional[dict[str, Any]]
    errors: list[str]
    agents_used: list[str]


class AgentState(TypedDict, total=False):
    """
    Shared state for the LangGraph multi-agent workflow.
    Composed of persistent DatasetContext and transient QuestionAnalysisContext.
    """

    # --- Persistent Dataset Context ---
    user_id: str
    dataset_id: str
    session_id: str
    table_name: str
    schema: dict[str, Any]          # Dataset column info
    dataset_profile: dict[str, Any]  # Cached profile from Data Analyst
    previous_validated_result: Optional[dict[str, Any]]
    validated_result: Optional[dict[str, Any]]

    # --- Transient Question Context ---
    query_id: str
    question: str
    intent: str                      # DATA_RETRIEVAL, DATA_ANALYSIS, PREDICTION, etc.
    query_contract: Optional[dict[str, Any]] # Fresh structured semantic Query Contract / AnalysisSpec
    analysis_spec: Optional[dict[str, Any]]
    structured_intent: dict[str, Any] # Parsed operation, dimensions, filters and ranking intent
    execution_plan: list[str]        # Ordered list of agent names to invoke

    # --- Agent outputs ---
    sql_query: Optional[str]
    sql_result: Optional[dict[str, Any]]
    analysis_result: Optional[dict[str, Any]]
    ml_result: Optional[dict[str, Any]]
    visualization_result: Optional[dict[str, Any]]
    rag_result: Optional[dict[str, Any]]

    # --- Final output ---
    final_report: Optional[dict[str, Any]]
    errors: list[str]
    agents_used: list[str]

    # --- Intermediate Visualization Workflow state ---
    _query_contract: Optional[Any]
    _raw_spec: Optional[dict[str, Any]]
    _validated_spec: Optional[dict[str, Any]]
    _available_cols: Optional[list[str]]
    _query_df: Optional[Any]


def create_fresh_question_state(
    user_id: str,
    dataset_id: str,
    query_id: str,
    question: str,
    table_name: str,
    schema: dict[str, Any],
    dataset_profile: Optional[dict[str, Any]] = None,
    session_id: str = "",
    previous_validated_result: Optional[dict[str, Any]] = None,
) -> AgentState:
    """
    Factory creating a guaranteed fresh state for every user question.
    Zero transient state from prior questions can survive.
    """
    return {
        "user_id": user_id,
        "dataset_id": dataset_id,
        "query_id": query_id,
        "session_id": session_id or "",
        "question": question,
        "table_name": table_name,
        "schema": schema or {},
        "dataset_profile": dataset_profile or {},
        "previous_validated_result": previous_validated_result,
        "validated_result": None,
        "intent": "",
        "query_contract": None,
        "analysis_spec": None,
        "structured_intent": {},
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
        "_query_contract": None,
        "_raw_spec": None,
        "_validated_spec": None,
        "_available_cols": None,
        "_query_df": None,
    }
