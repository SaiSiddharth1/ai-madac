"""
Query Contract Schema.
Defines the formal structured semantic contract produced by Gemini understanding
and validated before SQL/Pandas deterministic execution.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class FilterSpec(BaseModel):
    column: str
    operator: str = "="  # '=', '!=', '>', '<', '>=', '<=', 'LIKE', 'IN', 'BETWEEN'
    value: Any


class SortSpec(BaseModel):
    field: str
    direction: str = "desc"  # 'asc' or 'desc'


class QueryContract(BaseModel):
    """
    Fresh structured semantic contract for a user query.
    Source of truth for semantic validation, execution, and visualization planning.
    """
    query_id: str
    dataset_id: str
    original_query: str
    is_related_to_dataset: bool = True
    unrelated_reason: Optional[str] = None

    intent: str = "AGGREGATION"  # AGGREGATION, DATA_RETRIEVAL, DISTRIBUTION, RELATIONSHIP, PREDICTION, UNRELATED
    answer_type: str = "SCALAR"   # SCALAR, GROUPED, DISTRIBUTION, RELATIONSHIP, RECORDS, UNRELATED

    # Semantic measure information
    measure: Optional[str] = None             # Semantic concept e.g. "units sold", "revenue"
    measure_column: Optional[str] = None      # Dataset column e.g. "Quantity", "UnitPrice"
    measure_expression: Optional[str] = None  # SQL expression e.g. "Quantity", "Quantity * UnitPrice"
    aggregation: Optional[str] = None         # SUM, AVG, COUNT, COUNT_DISTINCT, MIN, MAX, or None

    # Grouping and dimensions: GROUPING MUST NEVER BE INVENTED
    dimensions: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)

    # Filters, sort, limit
    filters: list[FilterSpec] = Field(default_factory=list)
    sort: Optional[SortSpec] = None
    limit: Optional[int] = None

    # Visualization fields
    requested_visualization: bool = False
    should_visualize: bool = False
    suggested_chart_type: Optional[str] = None  # bar, horizontal_bar, line, scatter, histogram, box, pie, heatmap
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    color_by: Optional[str] = None
    chart_title: Optional[str] = None
    bin_width: Optional[float] = None

    is_follow_up: bool = False
