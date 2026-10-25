"""
Semantic Planner for AI-MADAC.
Produces a formal, unambiguous Query Contract from the user's natural language question
and dataset schema using Gemini with rigorous schema-aware fallback and validation.
Enforces the Query Contract as the single authoritative semantic source of truth.
"""

import json
import logging
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_provider import get_llm
from app.schemas.query_contract import CalculationSpec, FilterSpec, HavingSpec, OperandSpec, QueryContract, SortSpec
from app.utils.schema_resolver import SchemaResolver, normalize_term
from app.utils.temporal import find_temporal_column, is_temporal_column, is_temporal_intent

logger = logging.getLogger(__name__)

SEMANTIC_PLANNER_SYSTEM_PROMPT = """You are the Senior Semantic Data Analyst and Query Intelligence Agent for AI-MADAC.
Your job is to analyze the user's natural language question and the active dataset schema, and produce a formal, unambiguous Query Contract in JSON.

CORE ARCHITECTURE PRINCIPLES:
1. SEMANTIC PRECISION: Map the user's language accurately to active schema columns.
   - Match columns ONLY against the active dataset schema provided.
   - NEVER substitute an unrelated column (e.g. MonthlyIncome must NEVER become Age or Sales).
   - "units sold" or "items sold" -> Quantity column (e.g., 'Quantity', 'UnitsSold'). NEVER swap with unit price!
   - "revenue", "sales volume", or "turnover" -> Sales or Revenue column. If missing and Quantity & UnitPrice exist: 'Quantity * UnitPrice'.
   - "unique <entity>" -> COUNT_DISTINCT on that specific column (e.g. 'JobRole', 'CustomerID').
   - Do NOT select columns based on positional order or assume the first numeric column is the requested metric.

2. CRITICAL RULE ON RATE/SALARY NUMERIC COLUMNS:
   - Columns like 'MonthlyIncome', 'MonthlyRate', 'DailyRate', 'HourlyRate', 'AnnualSalary', 'Age' are NUMERIC measures.
   - Questions like 'What is the maximum monthly income?' or 'What is the average daily rate?' are SCALAR AGGREGATIONS:
     intent: 'AGGREGATION', answer_type: 'SCALAR', measure: 'MonthlyIncome', aggregation: 'MAX', group_by: [], should_visualize: false.
   - The word 'monthly', 'daily', or 'hourly' in these questions is part of the metric name, NEVER a time-series or trend!

3. MULTI-LINE AND GROUPING INTENT:
   - If user asks: "Show average monthly income by job level and gender as a multi-line chart":
     chart_type: "multi_line", x_axis: "JobLevel", y_axis: "MonthlyIncome", measure: "MonthlyIncome",
     aggregation: "AVG", dimensions: ["JobLevel"], group_by: ["JobLevel", "Gender"], series_dimension: "Gender".
     DO NOT treat this as a time series or use OverTime! OverTime is a categorical overtime indicator, not a timestamp.

4. STRICT AGGREGATION RULES:
   - "total", "sum", "overall", "aggregate" -> SUM
   - "average", "mean", "avg" -> AVG
   - "minimum", "min", "lowest" -> MIN
   - "maximum", "max", "highest" -> MAX
   - "how many rows", "number of records", "count", "how many employees" -> COUNT (measure: 'total_records')
   - "unique <entity>", "distinct <entity>" -> COUNT_DISTINCT

5. GROUPING MUST NEVER BE INVENTED:
   - If the user asks a scalar question without "by", "per", "across", group_by MUST be empty [].
   - If user asks "by X and Y", group_by MUST contain both X and Y.

6. DERIVED MATHEMATICAL CALCULATIONS:
   If the question asks to compute difference, sum, ratio, profit margin, or percentage between metrics:
   - intent MUST be "CALCULATION"
   - populate "calculation" object with operator: SUBTRACT, ADD, MULTIPLY, DIVIDE, RATIO, MARGIN, PERCENTAGE, PERCENTAGE_CHANGE
   - Verify that all operand columns actually exist in the schema! If not, mark question as UNRELATED.

7. DATASET RELEVANCE:
   - If the question asks for columns not in the active dataset schema (e.g. "top 10 products by total sales" on IBM HR which has no products or sales), set:
     "is_related_to_dataset": false,
     "unrelated_reason": "The active dataset does not contain the required fields. Please select the appropriate dataset.",
     "intent": "UNRELATED",
     "answer_type": "UNRELATED",
     "should_visualize": false

OUTPUT FORMAT:
Respond with ONLY a valid JSON object matching this schema:
{
  "is_related_to_dataset": true,
  "unrelated_reason": null,
  "intent": "AGGREGATION",
  "answer_type": "SCALAR",
  "measure": "MonthlyIncome",
  "measure_column": "MonthlyIncome",
  "measure_expression": "MonthlyIncome",
  "aggregation": "AVG",
  "dimensions": ["JobLevel"],
  "group_by": ["JobLevel", "Gender"],
  "series_dimension": "Gender",
  "filters": [],
  "having": [],
  "sort": null,
  "limit": null,
  "calculation": null,
  "should_visualize": true,
  "suggested_chart_type": "multi_line",
  "x_axis": "JobLevel",
  "y_axis": "MonthlyIncome",
  "color_by": "Gender",
  "chart_title": "Average Monthly Income by Job Level and Gender",
  "time_column": null,
  "time_granularity": null,
  "bin_width": null,
  "reasoning": "Direct explanation of mapping"
}
"""


class SemanticPlanner:
    """Interprets user queries against dataset schema using Gemini."""

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm(temperature=0.0)
        return self._llm

    def plan_query(
        self,
        question: str,
        schema: dict[str, Any],
        dataset_id: str,
        query_id: str,
        dataset_profile: Optional[dict[str, Any]] = None,
    ) -> QueryContract:
        """Formulate a fresh authoritative Query Contract for the query."""
        cols_info = []
        schema_cols = schema.get("columns", []) if isinstance(schema, dict) else []
        for col in schema_cols:
            name = col.get("name", "") if isinstance(col, dict) else str(col)
            dtype = col.get("dtype", "") if isinstance(col, dict) else ""
            sample = col.get("sample_values") or col.get("sample", []) if isinstance(col, dict) else []
            cols_info.append(f"- {name} ({dtype}) [samples: {sample[:3]}]")
        schema_desc = "\n".join(cols_info) if cols_info else "No column details."

        profile_summary = (dataset_profile or {}).get("summary", {})
        profile_str = json.dumps(profile_summary, default=str) if profile_summary else ""

        user_prompt = f"""ACTIVE DATASET SCHEMA:
{schema_desc}

DATASET SUMMARY:
{profile_str}

USER QUESTION:
"{question}"
"""
        try:
            messages = [
                SystemMessage(content=SEMANTIC_PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = self.llm.invoke(messages)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            data = json.loads(raw)
            logger.info("Gemini Semantic Planner produced contract: %s", json.dumps(data))

            # Build contract
            contract = self._build_contract_from_data(data, question, dataset_id, query_id, schema)
            # Apply semantic guardrails
            contract = self._validate_and_sanitize_contract(contract, schema)
            return contract

        except Exception as e:
            logger.error("Gemini semantic planning failed: %s. Using deterministic fallback.", e, exc_info=True)
            fallback = self._heuristic_contract_fallback(question, schema, dataset_id, query_id)
            return self._validate_and_sanitize_contract(fallback, schema)

    def _build_contract_from_data(
        self,
        data: dict[str, Any],
        question: str,
        dataset_id: str,
        query_id: str,
        schema: dict[str, Any],
    ) -> QueryContract:
        """Construct QueryContract from JSON data."""
        is_related = data.get("is_related_to_dataset", True)
        unrelated_reason = data.get("unrelated_reason")

        intent = data.get("intent", "AGGREGATION")
        answer_type = data.get("answer_type", "SCALAR")

        measure = data.get("measure")
        measure_column = data.get("measure_column")
        measure_expression = data.get("measure_expression")
        aggregation = data.get("aggregation")

        dimensions = data.get("dimensions", [])
        group_by = data.get("group_by", [])

        # Filters
        raw_filters = data.get("filters", [])
        filters = []
        for f in raw_filters:
            if isinstance(f, dict) and "column" in f:
                filters.append(FilterSpec(
                    column=f["column"],
                    operator=f.get("operator", "="),
                    value=f.get("value"),
                ))

        # Having
        raw_having = data.get("having", [])
        having = []
        for h in raw_having:
            if isinstance(h, dict):
                having.append(HavingSpec(
                    column=h.get("column"),
                    aggregation=h.get("aggregation"),
                    expression=h.get("expression"),
                    operator=h.get("operator", ">"),
                    value=h.get("value"),
                ))

        # Sort
        sort = None
        raw_sort = data.get("sort")
        if isinstance(raw_sort, dict) and "field" in raw_sort:
            sort = SortSpec(
                field=raw_sort["field"],
                direction=raw_sort.get("direction", "desc"),
            )

        limit = data.get("limit")
        try:
            limit = int(limit) if limit is not None else None
        except (ValueError, TypeError):
            limit = None

        should_visualize = bool(data.get("should_visualize", False))
        suggested_chart_type = data.get("suggested_chart_type")
        x_axis = data.get("x_axis")
        y_axis = data.get("y_axis")
        color_by = data.get("color_by")
        chart_title = data.get("chart_title")
        bin_width = data.get("bin_width")
        try:
            bin_width = float(bin_width) if bin_width is not None else None
        except (ValueError, TypeError):
            bin_width = None

        time_column = data.get("time_column")
        time_granularity = str(data["time_granularity"]).lower() if data.get("time_granularity") else None
        series_dimension = data.get("series_dimension")
        secondary_measure = data.get("secondary_measure")
        secondary_measure_column = data.get("secondary_measure_column")

        # Parse calculation if present
        calculation = None
        raw_calc = data.get("calculation")
        if isinstance(raw_calc, dict) and "operator" in raw_calc:
            operands = []
            for op in raw_calc.get("operands", []):
                if isinstance(op, dict):
                    op_filters = []
                    for f in op.get("filters", []):
                        if isinstance(f, dict) and "column" in f:
                            op_filters.append(FilterSpec(column=f["column"], operator=f.get("operator", "="), value=f.get("value")))
                    operands.append(OperandSpec(
                        name=op.get("name", "A"),
                        measure=op.get("measure"),
                        measure_column=op.get("measure_column") or op.get("measure"),
                        aggregation=op.get("aggregation", "SUM"),
                        filters=op_filters,
                    ))
            calculation = CalculationSpec(
                operator=raw_calc["operator"],
                operands=operands,
                label=raw_calc.get("label"),
                expression=raw_calc.get("expression"),
                target_column_name=raw_calc.get("target_column_name"),
            )

        q_lower = question.lower()
        explicit_vis = bool(re.search(r"\b(?:chart|plot|graph|visualiz\w*|histogram|scatter|box\s*plot|pie|bar)\b", q_lower))
        if explicit_vis and is_related:
            should_visualize = True

        return QueryContract(
            query_id=query_id,
            dataset_id=dataset_id,
            original_query=question,
            is_related_to_dataset=is_related,
            unrelated_reason=unrelated_reason,
            intent=intent,
            answer_type=answer_type,
            measure=measure,
            measure_column=measure_column,
            measure_expression=measure_expression,
            aggregation=aggregation,
            dimensions=dimensions,
            group_by=group_by,
            filters=filters,
            having=having,
            sort=sort,
            limit=limit,
            calculation=calculation,
            requested_visualization=explicit_vis,
            should_visualize=should_visualize,
            suggested_chart_type=suggested_chart_type,
            x_axis=x_axis,
            y_axis=y_axis,
            color_by=color_by,
            chart_title=chart_title,
            bin_width=bin_width,
            time_column=time_column,
            time_granularity=time_granularity,
            series_dimension=series_dimension,
            secondary_measure=secondary_measure,
            secondary_measure_column=secondary_measure_column,
            is_follow_up=False,
        )

    def _validate_and_sanitize_contract(
        self,
        contract: QueryContract,
        schema: dict[str, Any],
    ) -> QueryContract:
        """
        Enforce strict semantic safety:
        1. Never substitute an unrelated column.
        2. Prevent non-temporal columns (e.g. OverTime, MonthlyIncome) from becoming timestamps.
        3. Preserve multi-line dimensions (e.g. JobLevel + Gender).
        4. Preserve aggregations (AVG remains AVG, MAX remains MAX).
        5. Catch cross-dataset mismatches (e.g. Product/Sales on IBM HR).
        """
        raw_cols = schema.get("columns", []) if isinstance(schema, dict) else []
        schema_col_names = [c.get("name", "") if isinstance(c, dict) else str(c) for c in raw_cols]
        cols_lower = {c.lower(): c for c in schema_col_names if c}
        q_lower = contract.original_query.lower()

        # 1. CROSS-DATASET PROTECTION:
        # Check if question is asking for domain fields not in active dataset
        asks_sales = bool(re.search(r"\b(?:sales|total sales|gross sales|revenue|turnover)\b", q_lower))
        asks_products = bool(re.search(r"\b(?:products?|item|sku)\b", q_lower))
        has_sales = bool(SchemaResolver.resolve_column("sales", schema_col_names, schema))
        has_product = bool(SchemaResolver.resolve_column("product", schema_col_names, schema))

        if (asks_sales and not has_sales) or (asks_products and not has_product):
            contract.is_related_to_dataset = False
            contract.intent = "UNRELATED"
            contract.answer_type = "UNRELATED"
            contract.unrelated_reason = (
                "The active dataset does not contain the required Product/Sales fields. "
                "Please select the appropriate dataset."
            )
            contract.should_visualize = False
            return contract

        if not contract.is_related_to_dataset:
            return contract

        # 2. EXPLICIT CHART TYPE LOCK:
        explicit_chart = None
        if re.search(r"\bmulti[-_\s]line\b", q_lower):
            explicit_chart = "multi_line"
        elif re.search(r"\bgrouped\s*bar\b|\bclustered\s*bar\b", q_lower):
            explicit_chart = "grouped_bar"
        elif re.search(r"\bhorizontal\s*bar\b", q_lower):
            explicit_chart = "horizontal_bar"
        elif re.search(r"\bpie\b|\bdonut\b", q_lower):
            explicit_chart = "pie"
        elif re.search(r"\bhistogram\b", q_lower):
            explicit_chart = "histogram"
        elif re.search(r"\bscatter\b", q_lower):
            explicit_chart = "scatter"
        elif re.search(r"\bbox\s*plot\b|\bbox\b", q_lower):
            explicit_chart = "box"
        elif re.search(r"\bbar\s*chart\b|\bbar\s*graph\b|\bas a bar\b|\bin a bar\b|\bbar\b", q_lower):
            explicit_chart = "bar"
        elif re.search(r"\bline\s*chart\b|\bline\s*graph\b|\bas a line\b|\bin a line\b|\bline\b", q_lower):
            explicit_chart = "line"
        elif re.search(r"\barea\b", q_lower):
            explicit_chart = "area"

        if explicit_chart:
            contract.requested_visualization = True
            contract.should_visualize = True
            contract.suggested_chart_type = explicit_chart

        # 3. AGGREGATION SEMANTICS:
        is_explicit_min = bool(re.search(r"\b(?:min|minimum|lowest|smallest|least)\b", q_lower))
        is_explicit_max = bool(re.search(r"\b(?:max|maximum|highest value|peak|greatest)\b", q_lower)) or (
            bool(re.search(r"\bhighest\b", q_lower)) and not bool(re.search(r"\b(?:which|what)\s+[a-zA-Z_]+\s+has\s+the\s+highest\b", q_lower))
        )
        is_explicit_avg = bool(re.search(r"\b(?:average|mean|avg)\b", q_lower))
        is_explicit_count = bool(re.search(r"\b(?:count|number of|how many)\b", q_lower))
        is_explicit_sum = bool(re.search(r"\b(?:total|sum|combined|overall total)\b", q_lower))

        if is_explicit_min:
            contract.aggregation = "MIN"
        elif is_explicit_max:
            contract.aggregation = "MAX"
        elif is_explicit_avg:
            contract.aggregation = "AVG"
        elif is_explicit_count:
            if not contract.aggregation or contract.aggregation != "COUNT_DISTINCT":
                contract.aggregation = "COUNT"
        elif is_explicit_sum:
            contract.aggregation = "SUM"

        # 4. TEMPORAL DETECTION & TIME-SERIES INTENT:
        temporal_col = find_temporal_column(schema)
        is_genuine_time_series = is_temporal_intent(q_lower, has_temporal_column=bool(temporal_col))

        if is_genuine_time_series and temporal_col:
            contract.intent = "TIME_SERIES"
            contract.answer_type = "GROUPED"
            contract.time_column = temporal_col
            contract.x_axis = temporal_col

            # Time granularity
            if any(w in q_lower for w in ("yearly", "year by year", "annually", "annual")):
                time_grain = "year"
            elif any(w in q_lower for w in ("quarterly", "per quarter")):
                time_grain = "quarter"
            elif any(w in q_lower for w in ("weekly", "per week")):
                time_grain = "week"
            elif any(w in q_lower for w in ("daily", "day by day", "per day")):
                time_grain = "day"
            else:
                time_grain = contract.time_granularity or "month"
            contract.time_granularity = time_grain

            # Check if multi-series (e.g. for each category)
            series_dim = None
            if re.search(r"\b(?:multi[-_\s]line|for each|each|by)\b", q_lower):
                by_cat_m = re.search(r"\b(?:for each|each|by|across)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\b", q_lower)
                if by_cat_m:
                    cand_dim = SchemaResolver.resolve_column(by_cat_m.group(1).strip(), schema_col_names, schema)
                    if cand_dim and cand_dim != temporal_col:
                        series_dim = cand_dim

            if series_dim:
                contract.series_dimension = series_dim
                contract.group_by = [temporal_col, series_dim]
                contract.suggested_chart_type = "line"
            else:
                contract.series_dimension = None
                contract.group_by = [temporal_col]
                contract.suggested_chart_type = "line"

            # Date range filters
            yr_match = re.search(r"\b(?:from|between)\s+(\d{4})\s+(?:to|and|-)\s+(\d{4})\b", q_lower)
            if yr_match:
                y1, y2 = yr_match.group(1), yr_match.group(2)
                contract.filters = [
                    FilterSpec(column=temporal_col, operator=">=", value=f"{y1}-01-01"),
                    FilterSpec(column=temporal_col, operator="<=", value=f"{y2}-12-31 23:59:59"),
                ]
            contract.sort = SortSpec(field=temporal_col, direction="asc")
            contract.should_visualize = True
        else:
            # Not a time series: ensure time_column is clear
            contract.time_column = None
            contract.time_granularity = None
            if contract.intent == "TIME_SERIES":
                contract.intent = "AGGREGATION"

        # 5. LINE / MULTI-LINE ACROSS DIMENSIONS (e.g. JobLevel and Gender):
        if explicit_chart in ("multi_line", "line") and not is_genuine_time_series:
            # Parse dimensions after "by"
            by_and_m = re.search(r"\bby\s+([a-zA-Z0-9_\s]+?)\s+(?:and|&)\s+([a-zA-Z0-9_\s]+?)(?:\s+as\s+|\s+in\s+|$)", q_lower)
            by_single_m = re.search(r"\bby\s+([a-zA-Z0-9_\s]+?)(?:\s+as\s+|\s+in\s+|$)", q_lower)

            x_col = None
            series_dim = None

            if by_and_m:
                cand1 = by_and_m.group(1).strip()
                cand2 = by_and_m.group(2).strip()
                res1 = SchemaResolver.resolve_column(cand1, schema_col_names, schema)
                res2 = SchemaResolver.resolve_column(cand2, schema_col_names, schema)
                if res1 and res2:
                    x_col = res1
                    series_dim = res2
                elif res1:
                    x_col = res1
            elif by_single_m:
                cand = by_single_m.group(1).strip()
                res = SchemaResolver.resolve_column(cand, schema_col_names, schema)
                if res:
                    x_col = res

            # Resolve measure: e.g. "average monthly income by ..."
            meas_col = None
            for phrase in ("monthly income", "monthly salary", "daily rate", "hourly rate", "monthly rate", "expected return", "sales", "revenue", "profit", "quantity", "price", "age"):
                if phrase in q_lower:
                    res = SchemaResolver.resolve_column(phrase, schema_col_names, schema)
                    if res and res not in (x_col, series_dim):
                        meas_col = res
                        break

            if x_col:
                contract.x_axis = x_col
                contract.dimensions = [x_col]
                if series_dim:
                    contract.series_dimension = series_dim
                    contract.group_by = [x_col, series_dim]
                    contract.suggested_chart_type = "multi_line"
                else:
                    contract.series_dimension = None
                    contract.group_by = [x_col]
                    contract.suggested_chart_type = "line" if explicit_chart == "line" else (contract.suggested_chart_type or "line")

                if meas_col:
                    contract.measure = meas_col
                    contract.measure_column = meas_col
                    contract.measure_expression = meas_col

                contract.sort = SortSpec(field=x_col, direction="asc")
                contract.should_visualize = True
                contract.requested_visualization = True

        # 6. SCATTER PLOT SEMANTICS: "X versus Y" -> X is first, Y is second
        is_scatter = bool(re.search(r"\b(?:relationship|versus|vs\.?|against|correlation|scatter)\b", q_lower))
        if is_scatter:
            contract.intent = "RELATIONSHIP"
            contract.answer_type = "RELATIONSHIP"
            contract.aggregation = None
            contract.group_by = []
            contract.should_visualize = True
            contract.requested_visualization = True
            contract.suggested_chart_type = "scatter"

            vs_m = re.search(r"\b([a-zA-Z0-9_\s]+?)\s+(?:versus|vs\.?|against)\s+([a-zA-Z0-9_\s]+?)(?:\s+as\s+|\s+in\s+|$)", q_lower)
            if vs_m:
                var1_str = vs_m.group(1).strip()
                var2_str = vs_m.group(2).strip()
                col1 = SchemaResolver.resolve_column(var1_str, schema_col_names, schema)
                col2 = SchemaResolver.resolve_column(var2_str, schema_col_names, schema)
                if col1 and col2:
                    contract.x_axis = col1
                    contract.y_axis = col2
                    contract.measure = col1
                    contract.measure_column = col1

        # 7. HISTOGRAM SEMANTICS: Raw observations of requested measure
        is_dist = bool(re.search(r"\b(?:distribution|histogram|spread)\b", q_lower))
        if is_dist:
            contract.intent = "DISTRIBUTION"
            contract.answer_type = "DISTRIBUTION"
            contract.aggregation = None
            contract.group_by = []
            contract.should_visualize = True
            contract.requested_visualization = True
            contract.suggested_chart_type = "histogram"

            dist_m = re.search(r"\b(?:distribution|histogram|spread)\s+of\s+([a-zA-Z0-9_\s]+?)(?:\s+using|\s+as|\s+in|$)", q_lower)
            if dist_m:
                target_str = dist_m.group(1).strip()
                target_c = SchemaResolver.resolve_column(target_str, schema_col_names, schema)
                if target_c:
                    contract.measure = target_c
                    contract.measure_column = target_c
                    contract.x_axis = target_c

            bin_match = re.search(r"\b(?:bin\s*width|bin\s*size|bins?\s*of)\s*(\d+(?:\.\d+)?)\b", q_lower)
            if bin_match:
                contract.bin_width = float(bin_match.group(1))

        # 8. DISTINCT COUNTING:
        is_distinct = bool(re.search(r"\b(?:unique|distinct)\b", q_lower))
        if is_distinct:
            contract.aggregation = "COUNT_DISTINCT"
            dist_m = re.search(r"\b(?:unique|distinct)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\b", q_lower)
            if dist_m:
                entity_cand = dist_m.group(1).strip()
                entity_col = SchemaResolver.resolve_column(entity_cand, schema_col_names, schema)
                if entity_col:
                    contract.measure = entity_col
                    contract.measure_column = entity_col
                    contract.measure_expression = entity_col

        # 9. ROW COUNTING:
        is_row_count = bool(re.search(r"\b(?:how many\s+(?:employees|records|rows|orders|transactions|invoices|people|items|customers|investors)?\s*(?:are there|in total|total)?|count of records|total count)\b", q_lower))
        if is_row_count and not is_distinct and not any(k in q_lower for k in ("sales", "profit", "revenue", "income", "salary", "price", "quantity", "discount", "margin", "cost", "return")):
            contract.intent = "AGGREGATION"
            contract.answer_type = "SCALAR"
            contract.aggregation = "COUNT"
            contract.measure = "total_records"
            contract.measure_column = None
            contract.measure_expression = None
            contract.group_by = []
            contract.should_visualize = False

        # 10. ROW-LEVEL FILTERS (e.g. employees who work overtime):
        if bool(re.search(r"\b(?:who work overtime|working overtime|with overtime|who have overtime)\b", q_lower)):
            overtime_col = SchemaResolver.resolve_column("overtime", schema_col_names, schema)
            if overtime_col and not any(f.column == overtime_col for f in contract.filters):
                contract.filters.append(FilterSpec(column=overtime_col, operator="=", value="Yes"))

        # 11. BAR CHART SEMANTICS:
        if explicit_chart in ("bar", "horizontal_bar") and not is_genuine_time_series:
            top_m = re.search(r"\b(?:top|bottom|first|last)\s+\d+\s+([a-zA-Z0-9_\s]+?)\s+by\s+([a-zA-Z0-9_\s]+?)(?:\s+as\s+|\s+in\s+|$)", q_lower)
            if top_m:
                dim_cand = SchemaResolver.resolve_column(top_m.group(1).strip(), schema_col_names, schema)
                meas_cand = SchemaResolver.resolve_column(top_m.group(2).strip(), schema_col_names, schema)
                if dim_cand:
                    contract.dimensions = [dim_cand]
                    contract.group_by = [dim_cand]
                    contract.x_axis = dim_cand if explicit_chart != "horizontal_bar" else (meas_cand or contract.measure_column)
                    contract.y_axis = (meas_cand or contract.measure_column) if explicit_chart != "horizontal_bar" else dim_cand
                if meas_cand:
                    contract.measure = meas_cand
                    contract.measure_column = meas_cand
                    contract.measure_expression = meas_cand
                contract.suggested_chart_type = explicit_chart
                contract.should_visualize = True
                contract.requested_visualization = True
            else:
                by_m = re.search(r"\bby\s+([a-zA-Z0-9_\s]+?)(?:\s+as\s+|\s+in\s+|$)", q_lower)
                if by_m:
                    cand_dim = SchemaResolver.resolve_column(by_m.group(1).strip(), schema_col_names, schema)
                    if cand_dim and cand_dim != contract.measure_column:
                        contract.dimensions = [cand_dim]
                        contract.group_by = [cand_dim]
                        contract.x_axis = cand_dim
                        contract.suggested_chart_type = explicit_chart
                        contract.should_visualize = True
                        contract.requested_visualization = True

        # 12. PREVENT UNWANTED GROUPING:
        asks_grouping = bool(re.search(r"\b(?:by|per|across|for each|each|breakdown by)\b", q_lower)) or is_genuine_time_series
        asks_ranking = bool(re.search(r"\b(?:top|bottom|highest|lowest)\s+\d*\b", q_lower))
        if not explicit_chart and not asks_grouping and not asks_ranking and not is_genuine_time_series:
            contract.group_by = []
            contract.dimensions = []
            contract.answer_type = "SCALAR"
            contract.should_visualize = False
            contract.suggested_chart_type = None

        return contract

    def _detect_calculation(
        self,
        question: str,
        q_lower: str,
        schema: dict[str, Any],
        dataset_id: str,
        query_id: str,
    ) -> Optional[QueryContract]:
        """Detect mathematical / derived metric queries with strict schema validation."""
        is_diff = bool(re.search(r"\b(?:difference between|how much higher|how much more|gap between)\b", q_lower))
        is_margin = bool(re.search(r"\b(?:profit margin|margin based on)\b", q_lower))
        is_pct = bool(re.search(r"\b(?:what percentage of|what percent of|percentage of total|share of total)\b", q_lower))
        is_ratio = bool(re.search(r"\b(?:ratio of)\b", q_lower))
        is_sum = bool(re.search(r"\b(?:sum of total)\b", q_lower))

        if not (is_diff or is_margin or is_pct or is_ratio or is_sum):
            return None

        schema_cols = [c.get("name", "") if isinstance(c, dict) else str(c) for c in schema.get("columns", [])]

        if is_pct:
            sales_col = SchemaResolver.resolve_column("sales", schema_cols, schema)
            cat_col = SchemaResolver.resolve_column("category", schema_cols, schema)
            if not sales_col or not cat_col:
                # Active dataset does not have required fields
                return QueryContract(
                    query_id=query_id,
                    dataset_id=dataset_id,
                    original_query=question,
                    is_related_to_dataset=False,
                    unrelated_reason="The active dataset does not contain the required Product/Sales/Category fields. Please select the appropriate dataset.",
                    intent="UNRELATED",
                    answer_type="UNRELATED",
                    should_visualize=False,
                )

            cat_val = None
            for cand_val in ("Technology", "Furniture", "Office Supplies", "Consumer", "Corporate", "Home Office"):
                if cand_val.lower() in q_lower:
                    cat_val = cand_val
                    break
            if not cat_val:
                cat_val = "Technology"

            calc_spec = CalculationSpec(
                operator="PERCENTAGE",
                operands=[
                    OperandSpec(name=f"{cat_val} Sales", measure=sales_col, measure_column=sales_col, aggregation="SUM", filters=[FilterSpec(column=cat_col, operator="=", value=cat_val)]),
                    OperandSpec(name="Total Sales", measure=sales_col, measure_column=sales_col, aggregation="SUM", filters=[]),
                ],
                label=f"{cat_val} Percentage of Total {sales_col}",
            )
            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="CALCULATION",
                answer_type="SCALAR",
                measure=f"{cat_val} Percentage",
                calculation=calc_spec,
                should_visualize=False,
            )

        if is_margin:
            sales_col = SchemaResolver.resolve_column("sales", schema_cols, schema)
            profit_col = SchemaResolver.resolve_column("profit", schema_cols, schema)
            if not sales_col or not profit_col:
                return QueryContract(
                    query_id=query_id,
                    dataset_id=dataset_id,
                    original_query=question,
                    is_related_to_dataset=False,
                    unrelated_reason="The active dataset does not contain Sales or Profit fields.",
                    intent="UNRELATED",
                    answer_type="UNRELATED",
                    should_visualize=False,
                )
            calc_spec = CalculationSpec(
                operator="MARGIN",
                operands=[
                    OperandSpec(name="Total Sales", measure=sales_col, measure_column=sales_col, aggregation="SUM"),
                    OperandSpec(name="Total Profit", measure=profit_col, measure_column=profit_col, aggregation="SUM"),
                ],
                label="Profit Margin",
            )
            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="CALCULATION",
                answer_type="SCALAR",
                measure="Profit Margin",
                calculation=calc_spec,
                should_visualize=False,
            )

        if is_diff or is_sum or is_ratio:
            m_cand = []
            for phrase in ("sales", "profit", "revenue", "income", "monthly income", "salary", "quantity", "cost", "discount", "return"):
                if phrase in q_lower:
                    res = SchemaResolver.resolve_column(phrase, schema_cols, schema)
                    if res and res not in m_cand:
                        m_cand.append(res)

            if len(m_cand) < 2:
                # Cannot resolve 2 operands safely
                return QueryContract(
                    query_id=query_id,
                    dataset_id=dataset_id,
                    original_query=question,
                    is_related_to_dataset=False,
                    unrelated_reason="The active dataset does not contain the required fields for this calculation.",
                    intent="UNRELATED",
                    answer_type="UNRELATED",
                    should_visualize=False,
                )

            meas_a, meas_b = m_cand[0], m_cand[1]
            op = "SUBTRACT" if is_diff else ("ADD" if is_sum else "RATIO")
            lbl = f"Difference between {meas_a} and {meas_b}" if is_diff else (f"Sum of {meas_a} and {meas_b}" if is_sum else f"Ratio of {meas_a} to {meas_b}")

            calc_spec = CalculationSpec(
                operator=op,
                operands=[
                    OperandSpec(name=f"Total {meas_a}", measure=meas_a, measure_column=meas_a, aggregation="SUM"),
                    OperandSpec(name=f"Total {meas_b}", measure=meas_b, measure_column=meas_b, aggregation="SUM"),
                ],
                label=lbl,
            )
            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="CALCULATION",
                answer_type="SCALAR",
                measure=lbl,
                calculation=calc_spec,
                should_visualize=False,
            )

        return None

    def _heuristic_contract_fallback(
        self,
        question: str,
        schema: dict[str, Any],
        dataset_id: str,
        query_id: str,
    ) -> QueryContract:
        """Deterministic schema-aware fallback without positional numeric guesses."""
        q_lower = question.lower()
        schema_cols = [c.get("name", "") if isinstance(c, dict) else str(c) for c in schema.get("columns", [])]

        # Check for calculation query
        calc_res = self._detect_calculation(question, q_lower, schema, dataset_id, query_id)
        if calc_res:
            return calc_res

        # Resolve explicit chart request
        explicit_chart = None
        if re.search(r"\bmulti[-_\s]line\b", q_lower):
            explicit_chart = "multi_line"
        elif re.search(r"\bhistogram\b", q_lower):
            explicit_chart = "histogram"
        elif re.search(r"\bscatter\b", q_lower):
            explicit_chart = "scatter"
        elif re.search(r"\bhorizontal\s*bar\b", q_lower):
            explicit_chart = "horizontal_bar"
        elif re.search(r"\bpie\b|\bdonut\b", q_lower):
            explicit_chart = "pie"
        elif re.search(r"\bbar\b", q_lower):
            explicit_chart = "bar"
        elif re.search(r"\bline\b", q_lower):
            explicit_chart = "line"

        # Limit and ranking
        limit_m = re.search(r"\b(?:top|bottom|first|last|limit)\s+(\d+)\b", q_lower)
        limit_val = int(limit_m.group(1)) if limit_m else None

        # Resolve measure from query
        meas_col = None
        for phrase in (
            "monthly income", "monthly salary", "daily rate", "hourly rate", "monthly rate",
            "annual income", "salary", "age", "sales", "total sales", "revenue", "profit",
            "quantity", "units sold", "price", "unit price", "cost", "expected return"
        ):
            if phrase in q_lower:
                res = SchemaResolver.resolve_column(phrase, schema_cols, schema)
                if res:
                    meas_col = res
                    break

        # Resolve dimensions
        x_col = None
        series_dim = None

        # Pattern: top N <entity> by <measure>
        top_m = re.search(r"\b(?:top|bottom|first|last)\s+\d+\s+([a-zA-Z0-9_\s]+?)\s+by\s+([a-zA-Z0-9_\s]+?)(?:\s+as\s+|\s+in\s+|$)", q_lower)
        if top_m:
            entity_phrase = top_m.group(1).strip()
            measure_phrase = top_m.group(2).strip()
            dim_res = SchemaResolver.resolve_column(entity_phrase, schema_cols, schema)
            meas_res = SchemaResolver.resolve_column(measure_phrase, schema_cols, schema)
            if dim_res:
                x_col = dim_res
            if meas_res:
                meas_col = meas_res

        # Pattern: by <dim1> and <dim2> or by <dim>
        if not x_col:
            by_and_m = re.search(r"\bby\s+([a-zA-Z0-9_\s]+?)\s+(?:and|&)\s+([a-zA-Z0-9_\s]+?)(?:\s+as\s+|\s+in\s+|$)", q_lower)
            by_single_m = re.search(r"\bby\s+([a-zA-Z0-9_\s]+?)(?:\s+as\s+|\s+in\s+|$)", q_lower)

            if by_and_m:
                cand_x = SchemaResolver.resolve_column(by_and_m.group(1).strip(), schema_cols, schema)
                cand_s = SchemaResolver.resolve_column(by_and_m.group(2).strip(), schema_cols, schema)
                if cand_x and cand_x != meas_col:
                    x_col = cand_x
                if cand_s and cand_s != meas_col:
                    series_dim = cand_s
            elif by_single_m:
                cand_x = SchemaResolver.resolve_column(by_single_m.group(1).strip(), schema_cols, schema)
                if cand_x and cand_x != meas_col:
                    x_col = cand_x

        # If dimension still not found, check known categorical phrases
        if not x_col:
            for phrase in ("product", "products", "category", "region", "department", "gender", "job role", "job level", "segment"):
                if re.search(rf"\b{phrase}\b", q_lower):
                    cand_dim = SchemaResolver.resolve_column(phrase, schema_cols, schema)
                    if cand_dim and cand_dim != meas_col:
                        x_col = cand_dim
                        break

        # Aggregation
        agg = "SUM"
        if bool(re.search(r"\b(?:average|mean|avg)\b", q_lower)):
            agg = "AVG"
        elif bool(re.search(r"\b(?:maximum|max|highest value|peak)\b", q_lower)):
            agg = "MAX"
        elif bool(re.search(r"\b(?:minimum|min|lowest)\b", q_lower)):
            agg = "MIN"
        elif bool(re.search(r"\b(?:unique|distinct)\b", q_lower)):
            agg = "COUNT_DISTINCT"
        elif bool(re.search(r"\b(?:count|how many|number of)\b", q_lower)):
            agg = "COUNT"

        group_by = []
        if x_col:
            group_by.append(x_col)
        if series_dim:
            group_by.append(series_dim)

        # Sort direction
        if limit_val or bool(re.search(r"\b(?:top|highest|most|greatest|best)\b", q_lower)):
            sort_dir = "desc"
            sort_col = meas_col or (x_col or "metric")
        elif bool(re.search(r"\b(?:bottom|lowest|least|worst)\b", q_lower)):
            sort_dir = "asc"
            sort_col = meas_col or (x_col or "metric")
        elif explicit_chart in ("line", "multi_line"):
            sort_dir = "asc"
            sort_col = x_col or (meas_col or "metric")
        else:
            sort_dir = "desc"
            sort_col = x_col or (meas_col or "metric")

        return QueryContract(
            query_id=query_id,
            dataset_id=dataset_id,
            original_query=question,
            intent="AGGREGATION",
            answer_type="GROUPED" if group_by else "SCALAR",
            measure=meas_col or "total_records",
            measure_column=meas_col,
            measure_expression=meas_col,
            aggregation=agg,
            dimensions=[x_col] if x_col else [],
            group_by=group_by,
            series_dimension=series_dim,
            x_axis=x_col if explicit_chart != "horizontal_bar" else meas_col,
            y_axis=meas_col if explicit_chart != "horizontal_bar" else x_col,
            suggested_chart_type=explicit_chart,
            should_visualize=bool(explicit_chart or group_by),
            requested_visualization=bool(explicit_chart),
            limit=limit_val,
            sort=SortSpec(field=sort_col, direction=sort_dir),
        )


semantic_planner = SemanticPlanner()
