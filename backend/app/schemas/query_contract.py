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


class HavingSpec(BaseModel):
    column: Optional[str] = None
    aggregation: Optional[str] = None  # COUNT, AVG, SUM, MIN, MAX
    expression: Optional[str] = None   # e.g. "COUNT(*)" or 'AVG("Profit")'
    operator: str = ">"                # '>', '<', '>=', '<=', '=', '!='
    value: Any


class OperandSpec(BaseModel):
    name: str = "A"
    measure: Optional[str] = None
    measure_column: Optional[str] = None
    aggregation: Optional[str] = "SUM"
    filters: list[FilterSpec] = Field(default_factory=list)


class CalculationSpec(BaseModel):
    operator: str  # SUBTRACT, ADD, MULTIPLY, DIVIDE, RATIO, PERCENTAGE, PERCENTAGE_CHANGE, MARGIN
    operands: list[OperandSpec] = Field(default_factory=list)
    expression: Optional[str] = None
    label: Optional[str] = None
    target_column_name: Optional[str] = None


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

    # Filters, having, sort, limit
    filters: list[FilterSpec] = Field(default_factory=list)
    having: list[HavingSpec] = Field(default_factory=list)
    sort: Optional[SortSpec] = None
    limit: Optional[int] = None
    calculation: Optional[CalculationSpec] = None

    # Visualization fields
    requested_visualization: bool = False
    should_visualize: bool = False
    suggested_chart_type: Optional[str] = None  # bar, horizontal_bar, line, scatter, histogram, box, pie, heatmap
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    color_by: Optional[str] = None
    chart_title: Optional[str] = None
    bin_width: Optional[float] = None
    time_column: Optional[str] = None
    time_granularity: Optional[str] = None  # year, quarter, month, week, day, hour
    series_dimension: Optional[str] = None  # Category dimension for multi-series line/bar
    secondary_measure: Optional[str] = None  # Secondary measure for clustered multi-metric bar
    secondary_measure_column: Optional[str] = None

    is_follow_up: bool = False

    @property
    def explicit_chart_type(self) -> Optional[str]:
        return self.suggested_chart_type

    @property
    def requested_fields(self) -> list[str]:
        fields = list(self.dimensions)
        for g in self.group_by:
            if g not in fields:
                fields.append(g)
        if self.series_dimension and self.series_dimension not in fields:
            fields.append(self.series_dimension)
        if self.measure_column and self.measure_column not in fields:
            fields.append(self.measure_column)
        if self.x_axis and self.x_axis not in fields:
            fields.append(self.x_axis)
        if self.y_axis and self.y_axis not in fields:
            fields.append(self.y_axis)
        return fields


    @property
    def sorting(self) -> dict[str, Any]:
        if self.sort:
            return {"field": self.sort.field, "direction": self.sort.direction}
        return {"field": None, "direction": None}

    @property
    def binning(self) -> Optional[dict[str, Any]]:
        if self.bin_width:
            return {"mode": "fixed", "size": self.bin_width, "bin_size": self.bin_width}
        return None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# Canonical alias
AnalysisSpec = QueryContract

