"""
Agent State definition for LangGraph workflow.
Shared typed state passed through all agent nodes.
"""

from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state for the LangGraph multi-agent workflow.
    All agent nodes read from and write to this state.
    """

    # --- Context (set by query service before graph invocation) ---
    user_id: str
    dataset_id: str
    query_id: str
    session_id: str
    question: str
    table_name: str
    schema: dict[str, Any]          # Dataset column info
    dataset_profile: dict[str, Any]  # Cached profile from Data Analyst
    previous_validated_result: Optional[dict[str, Any]]
    validated_result: Optional[dict[str, Any]]

    # --- Supervisor output ---
    intent: str                      # DATA_RETRIEVAL, DATA_ANALYSIS, PREDICTION, etc.
    query_contract: Optional[dict[str, Any]] # Fresh structured semantic Query Contract
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
