"""
Semantic Planner for AI-MADAC.
Uses Gemini runtime intelligence to extract a rigorous Query Contract from the
user's question and dataset schema.
"""

import json
import logging
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_provider import get_llm
from app.schemas.query_contract import CalculationSpec, FilterSpec, HavingSpec, OperandSpec, QueryContract, SortSpec
from app.agents.visualization.contract import resolve_column_semantically

logger = logging.getLogger(__name__)

SEMANTIC_PLANNER_SYSTEM_PROMPT = """You are the Senior Semantic Data Analyst and Query Intelligence Agent for AI-MADAC.
Your job is to analyze the user's natural language question and the dataset schema, and produce a formal, unambiguous Query Contract in JSON.

CORE ARCHITECTURE PRINCIPLES:
1. SEMANTIC PRECISION: Map the user's language accurately to schema columns.
   - "units sold" or "items sold" -> Quantity column (e.g., 'Quantity', 'UnitsSold', 'SalesQuantity'). NEVER swap with unit price!
   - "unit price" or "price per unit" -> Unit price column (e.g., 'UnitPrice', 'Price').
   - "revenue", "sales volume", or "turnover" -> If a dedicated revenue/sales column exists, use it. If NOT, but quantity and price columns exist (e.g. 'Quantity' and 'UnitPrice'), define derived metric: 'Quantity * UnitPrice'.
   - "customers" or "clients" -> Customer identifier (e.g., 'CustomerID', 'Client_ID').
   - "transactions", "orders", or "invoices" -> Transaction/Invoice identifier (e.g., 'InvoiceNo', 'OrderID').
   - "country", "nation", "region" -> Country / Region column (e.g., 'Country').
   - Do NOT select columns based on positional order or assume the first numeric column is the requested metric.

2. CRITICAL RULE ON RATE/SALARY NUMERIC COLUMNS:
   - Columns like 'MonthlyIncome', 'MonthlyRate', 'DailyRate', 'HourlyRate', 'AnnualSalary', 'Age' are NUMERIC measures.
   - Questions like 'What is the maximum monthly income?' or 'What is the average daily rate?' are SCALAR AGGREGATIONS:
     intent: 'AGGREGATION', answer_type: 'SCALAR', measure: 'MonthlyIncome', aggregation: 'MAX', group_by: [], should_visualize: false.
   - The word 'monthly', 'daily', or 'hourly' in these questions is part of the metric name, NEVER a time-series or trend!

3. STRICT AGGREGATION RULES:
   - "total", "sum", "overall", "aggregate" -> SUM
   - "average", "mean", "avg" -> AVG
   - "minimum", "min", "lowest" -> MIN
   - "maximum", "max", "highest" -> MAX
   - "how many rows", "number of records", "count", "how many employees/orders" -> COUNT (measure: 'total_records')
   - "unique <entity>", "distinct <entity>" (e.g. "unique job roles", "distinct customers") -> COUNT_DISTINCT on that specific column (e.g. 'JobRole', 'CustomerID'). NEVER default to Age or another numeric column!

4. GROUPING MUST NEVER BE INVENTED:
   - If the user asks: "What is the total revenue?" or "What is the maximum monthly income?", NO GROUPING IS REQUESTED!
     answer_type MUST be "SCALAR", group_by MUST be empty [].
     NEVER introduce grouping by country, segment, category, or date unless the user explicitly asks for it!
   - If the user asks: "What is the total revenue by country?", THEN AND ONLY THEN:
     answer_type MUST be "GROUPED", group_by MUST be ["Country"].
   - If the user asks: "Top 10 countries by revenue", group_by MUST be ["Country"], limit MUST be 10, sort MUST be {"field": "revenue", "direction": "desc"}.

5. DERIVED MATHEMATICAL CALCULATIONS:
   If the question asks to compute the difference, sum, ratio, profit margin, or percentage between metrics:
   - intent MUST be "CALCULATION"
   - populate "calculation" object with operator: SUBTRACT, ADD, MULTIPLY, DIVIDE, RATIO, MARGIN, PERCENTAGE, PERCENTAGE_CHANGE
   - Operands define base metrics:
     - "difference between total sales and total profit" -> operator: "SUBTRACT", operands: [{"measure": "Sales", "aggregation": "SUM"}, {"measure": "Profit", "aggregation": "SUM"}], label: "Difference"
     - "profit margin based on total sales and total profit" -> operator: "MARGIN", operands: [{"measure": "Sales", "aggregation": "SUM"}, {"measure": "Profit", "aggregation": "SUM"}], label: "Profit Margin"
     - "percentage of total sales comes from the Technology category" -> operator: "PERCENTAGE", operands: [{"measure": "Sales", "aggregation": "SUM", "filters": [{"column": "Category", "operator": "=", "value": "Technology"}]}, {"measure": "Sales", "aggregation": "SUM", "filters": []}], label: "Technology Sales Percentage"
     - "ratio of total sales to total profit" -> operator: "RATIO", operands: [{"measure": "Sales", "aggregation": "SUM"}, {"measure": "Profit", "aggregation": "SUM"}], label: "Ratio"
     - "difference between sales and profit by category" -> operator: "SUBTRACT", group_by: ["Category"], answer_type: "GROUPED", operands: [{"measure": "Sales", "aggregation": "SUM"}, {"measure": "Profit", "aggregation": "SUM"}], label: "Difference by Category"

6. VISUALIZATION RULES:
   - If the user explicitly asks for a chart type ("as a bar chart", "as a line chart", "as a multi-line chart", "as a scatter plot", "as a histogram", "as a pie chart", "as a grouped bar chart"), that chart type is LOCKED:
     should_visualize: true, requested_visualization: true, suggested_chart_type: "<chart_type>".
   - should_visualize should be false for simple scalar metrics ("What is the total revenue?", "What is the maximum monthly income?") unless explicitly requested to plot.
   - Chart types:
     - Distribution of one numeric column -> "histogram" (RAW OBSERVATIONS, no aggregation!)
     - Relationship between two numeric columns -> "scatter" (RAW OBSERVATIONS, no aggregation!)
     - Category + numeric metric -> "bar" (or "horizontal_bar" if long labels or top-N ranking)
     - Time + numeric metric -> "line"
     - Multi-series line -> "multi_line"
     - Small part-to-whole -> "pie"

7. DATASET RELEVANCE:
   - If the question is completely unrelated to the dataset (e.g. "What is the weather in Hyderabad?" when dataset has sales or finance data), set:
     "is_related_to_dataset": false,
     "unrelated_reason": "The question is not related to the currently loaded dataset schema.",
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
  "measure": "Quantity",
  "measure_column": "Quantity",
  "measure_expression": "Quantity",
  "aggregation": "SUM",
  "dimensions": [],
  "group_by": [],
  "filters": [],
  "sort": null,
  "limit": null,
  "calculation": null,
  "should_visualize": false,
  "suggested_chart_type": null,
  "x_axis": null,
  "y_axis": null,
  "color_by": null,
  "chart_title": null,
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
        """Formulate a fresh Query Contract for the query."""
        cols_info = []
        for col in schema.get("columns", []):
            name = col.get("name", "")
            dtype = col.get("dtype", "")
            sample = col.get("sample_values") or col.get("sample", [])
            cols_info.append(f"- {name} ({dtype}) [samples: {sample[:3]}]")
        schema_desc = "\n".join(cols_info) if cols_info else "No column details."

        profile_summary = (dataset_profile or {}).get("summary", {})
        profile_str = json.dumps(profile_summary, default=str) if profile_summary else ""

        user_prompt = f"""DATASET SCHEMA:
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

            # Build and validate contract
            contract = self._build_contract_from_data(data, question, dataset_id, query_id, schema)
            # Apply semantic guardrails
            contract = self._validate_and_sanitize_contract(contract, schema)
            return contract

        except Exception as e:
            logger.error("Gemini semantic planning failed: %s. Using heuristic fallback.", e, exc_info=True)
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
        time_granularity = data.get("time_granularity")
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

        # Check explicit visualization terms in question
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
        1. Check if column exists in schema.
        2. Prevent measure swapping (e.g. units sold -> UnitPrice).
        3. Prevent unwanted grouping (e.g. total revenue -> group by Country).
        4. Detect derived revenue if missing.
        """
        cols = {str(c.get("name", "")).lower(): str(c.get("name", "")) for c in schema.get("columns", [])}
        q_lower = contract.original_query.lower()

        if not contract.is_related_to_dataset:
            return contract

        # 0. Explicit Chart Request Lock:
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

        # 1. Prevent unwanted grouping, EXCEPT for time-series / temporal queries:
        is_temporal = bool(re.search(
            r"\b(?:over time|changed over time|change over time|trend|trend over time|historical trend|month by month|day by day|year by year|sales trend|revenue trend|quantity trend)\b",
            q_lower
        )) or (
            bool(re.search(r"\b(?:daily|weekly|monthly|quarterly|yearly|annually)\b", q_lower))
            and not any(m_word in q_lower for m_word in ("monthly income", "monthly salary", "monthly charges", "monthly rate", "monthly expense", "monthly cost", "daily rate", "hourly rate"))
        )
        is_explicit_line = bool(re.search(r"\b(?:line chart|line graph|plot as a line|in line chart|as a line chart|as line)\b", q_lower))

        asks_grouping = bool(re.search(r"\b(?:by|per|across|for each|each|breakdown by)\b", q_lower)) or is_temporal
        asks_ranking = bool(re.search(r"\b(?:top|bottom|highest|lowest)\s+\d*\b", q_lower))
        if not explicit_chart and not asks_grouping and not asks_ranking and not is_temporal:
            contract.group_by = []
            contract.dimensions = []
            contract.answer_type = "SCALAR"
            contract.should_visualize = False
            contract.suggested_chart_type = None
            contract.time_column = None
            contract.time_granularity = None
            contract.series_dimension = None
            if contract.intent in ("TIME_SERIES", "TREND", "VISUALIZATION"):
                contract.intent = "AGGREGATION"

        # 2. Prevent Measure Swapping:
        is_price_question = bool(re.search(r"\b(?:unit\s*price|price per unit|unitprice|price|cost|rate)\b", q_lower))
        is_units_question = (
            bool(re.search(r"\b(?:quantity|items?|units?\s+sold|items?\s+sold)\b", q_lower))
            or (bool(re.search(r"\bunits?\b", q_lower)) and not is_price_question)
        )
        is_revenue_question = bool(re.search(r"\b(?:revenue|turnover)\b", q_lower))
        is_sales_question = bool(re.search(r"\b(?:sales|total sales|gross sales)\b", q_lower))
        sales_col = next((real for low, real in cols.items() if low in ("sales", "gross_sales")), None)
        if not sales_col:
            sales_col = resolve_column_semantically("sales", list(cols.values()))

        # Check explicit aggregation keyword in query
        is_explicit_min = bool(re.search(r"\b(?:min|minimum|lowest|smallest|least)\b", q_lower))
        is_explicit_max = bool(re.search(r"\b(?:max|maximum|highest|largest|peak|greatest)\b", q_lower))
        is_explicit_avg = bool(re.search(r"\b(?:average|mean|avg)\b", q_lower))
        is_explicit_count = bool(re.search(r"\b(?:count|number of|how many)\b", q_lower))

        if is_explicit_min:
            contract.aggregation = "MIN"
        elif is_explicit_max:
            contract.aggregation = "MAX"
        elif is_explicit_avg:
            contract.aggregation = "AVG"
        elif is_explicit_count:
            contract.aggregation = "COUNT"

        # If user asked for sales and dataset has a direct Sales column, direct column MUST take precedence
        if is_sales_question and sales_col:
            contract.measure = sales_col
            contract.measure_column = sales_col
            contract.measure_expression = sales_col
            if not contract.aggregation:
                contract.aggregation = "MIN" if is_explicit_min else ("MAX" if is_explicit_max else ("AVG" if is_explicit_avg else "SUM"))

        if is_units_question and not is_price_question and not is_revenue_question and not is_sales_question:
            # Measure MUST NOT be UnitPrice or any price column!
            if contract.measure_column and any(p in contract.measure_column.lower() for p in ("price", "cost", "unitprice")):
                # Find the true quantity column
                true_qty = next((real for low, real in cols.items() if any(k in low for k in ("quantity", "qty", "units", "sales_volume"))), None)
                if true_qty:
                    contract.measure_column = true_qty
                    contract.measure_expression = true_qty
                    contract.measure = "Quantity"

        if is_price_question and not is_revenue_question and not is_sales_question:
            # Measure MUST NOT be Quantity!
            if contract.measure_column and any(q in contract.measure_column.lower() for q in ("quantity", "qty", "units")):
                true_price = next((real for low, real in cols.items() if any(k in low for k in ("unitprice", "price", "unit_price", "rate"))), None)
                if true_price:
                    contract.measure_column = true_price
                    contract.measure_expression = true_price
                    contract.measure = "UnitPrice"

        # 3. Handle Derived Revenue (only when no direct sales/revenue column exists):
        if is_revenue_question or (is_sales_question and not sales_col):
            # Check if dataset has a direct revenue column
            direct_rev = next((real for low, real in cols.items() if low in ("revenue", "turnover", "sales", "gross_sales")), None)
            if not direct_rev:
                direct_rev = resolve_column_semantically("revenue", list(cols.values()))
            if direct_rev:
                contract.measure = direct_rev
                contract.measure_column = direct_rev
                contract.measure_expression = direct_rev
                if not contract.aggregation:
                    contract.aggregation = "SUM"
            else:
                # Derived from Quantity * UnitPrice
                qty_col = next((real for low, real in cols.items() if any(k in low for k in ("quantity", "qty", "units"))), None)
                price_col = next((real for low, real in cols.items() if any(k in low for k in ("unitprice", "price", "unit_price"))), None)
                if qty_col and price_col:
                    contract.measure = "revenue"
                    contract.measure_column = f"{qty_col} * {price_col}"
                    contract.measure_expression = f'"{qty_col}" * "{price_col}"'
                    if not contract.aggregation:
                        contract.aggregation = "SUM"

        # 4. Handle Scatter / Relationship:
        if bool(re.search(r"\b(?:relationship|versus|vs\.?|against|correlation|scatter)\b", q_lower)):
            numeric_cols = [
                real for low, real in cols.items()
                if any(k in low for k in ("quantity", "price", "unitprice", "age", "income", "amount", "return", "sales", "profit", "cost", "value"))
            ]
            if len(numeric_cols) >= 2:
                contract.intent = "RELATIONSHIP"
                contract.answer_type = "RELATIONSHIP"
                contract.aggregation = None  # raw observations!
                contract.group_by = []
                contract.should_visualize = True
                contract.suggested_chart_type = "scatter"
                contract.x_axis = numeric_cols[0]
                contract.y_axis = numeric_cols[1]

                # Check if colored by category (e.g. "Sales vs Profit by Category")
                for low, real in cols.items():
                    if real not in numeric_cols and any(w in q_lower for w in (low, low.replace("_", " "), low.rstrip("s"))):
                        contract.color_by = real
                        break

        # 5. Handle Distribution / Histogram:
        if bool(re.search(r"\b(?:distribution|histogram|spread)\b", q_lower)):
            contract.intent = "DISTRIBUTION"
            contract.answer_type = "DISTRIBUTION"
            contract.aggregation = None  # raw observations!
            contract.group_by = []
            contract.should_visualize = True
            contract.suggested_chart_type = "histogram"
            # check bin width
            bin_match = re.search(r"\b(?:bin\s*width|bin\s*size|bins?\s*of)\s*(\d+(?:\.\d+)?)\b", q_lower)
            if bin_match:
                contract.bin_width = float(bin_match.group(1))

        # 6. Distinct counting & Row counting:
        is_distinct = bool(re.search(r"\b(?:unique|distinct)\b", q_lower))
        if is_distinct:
            contract.aggregation = "COUNT_DISTINCT"
            dist_m = re.search(r"\b(?:unique|distinct)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\b", q_lower)
            if dist_m:
                entity_cand = dist_m.group(1).strip()
                entity_col = resolve_column_semantically(entity_cand, list(cols.values()))
                if not entity_col:
                    sing = (entity_cand[:-3] + "y") if entity_cand.endswith("ies") else (entity_cand[:-1] if entity_cand.endswith("s") and not entity_cand.endswith("ss") else entity_cand)
                    entity_col = resolve_column_semantically(sing, list(cols.values()))
                if entity_col:
                    contract.measure = entity_col
                    contract.measure_column = entity_col
                    contract.measure_expression = entity_col

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

        # 7. Handle Time Series / Trend / Line Chart:
        if is_temporal or is_explicit_line:
            # Generic time column resolution from dataset schema
            time_col = next((
                real for low, real in cols.items()
                if not any(ex in low for ex in ("income", "salary", "rate", "cost", "price", "fee", "expense", "charges", "sales", "revenue", "amount", "profit", "balance", "limit"))
                and any(k in low for k in ("invoicedate", "order_date", "orderdate", "date", "timestamp", "time", "year", "created_at", "saledate", "purchasedate"))
            ), None)
            if time_col:
                contract.intent = "TIME_SERIES"
                contract.answer_type = "GROUPED"
                contract.time_column = time_col

                # Determine time granularity
                if any(w in q_lower for w in ("monthly", "month by month", "per month")):
                    time_grain = "month"
                elif any(w in q_lower for w in ("yearly", "year by year", "annually", "annual")):
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

                # Check if multi-series line chart (e.g. "monthly sales for each category" or "by category")
                series_dim = None
                is_explicit_multi = (
                    bool(re.search(r"\b(?:multi[-_\s]line|multiple lines?)\b", q_lower))
                    or any(w in q_lower for w in ("male and female", "female and male", "both genders"))
                    or bool(re.search(r"\b(?:for each|each)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\b", q_lower))
                )
                if is_explicit_multi:
                    # Only search non-numeric, non-temporal categorical columns for series dimension
                    cat_cols = [
                        c.get("name") for c in schema.get("columns", [])
                        if not any(nt in str(c.get("dtype", "")).lower() for nt in ("int", "float", "double", "real", "num", "dec"))
                        and not any(k in str(c.get("name", "")).lower() for k in ("sales", "profit", "revenue", "quantity", "price", "unitprice", "discount", "cost", "date", "time", "year", "month", "day", "id", "row_id"))
                    ]
                    by_cat_m = re.search(r"\b(?:for each|each|by|across)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\b", q_lower)
                    if by_cat_m:
                        cand_dim = resolve_column_semantically(by_cat_m.group(1).strip(), cat_cols)
                        if cand_dim and cand_dim != time_col and cand_dim != contract.measure_column:
                            series_dim = cand_dim

                    if not series_dim:
                        for low, real in cols.items():
                            if real in cat_cols and real != time_col and real != contract.measure_column:
                                if low in q_lower or low.replace("_", " ") in q_lower or any(alias in q_lower for alias in ("male", "female", "men", "women") if "gender" in low or "sex" in low):
                                    series_dim = real
                                    break

                if series_dim:
                    contract.series_dimension = series_dim
                    contract.group_by = [time_col, series_dim]
                    contract.suggested_chart_type = "line"
                else:
                    contract.series_dimension = None
                    contract.group_by = [time_col]
                    contract.suggested_chart_type = "line"

                # Extract date range filters e.g. "from 2011 to 2015" or "between 2011 and 2015"
                yr_match = re.search(r"\b(?:from|between)\s+(\d{4})\s+(?:to|and|-)\s+(\d{4})\b", q_lower)
                if yr_match:
                    y1, y2 = yr_match.group(1), yr_match.group(2)
                    contract.filters = [
                        FilterSpec(column=time_col, operator=">=", value=f"{y1}-01-01"),
                        FilterSpec(column=time_col, operator="<=", value=f"{y2}-12-31 23:59:59"),
                    ]
                else:
                    yr_single = re.search(r"\b(?:in|for)\s+(\d{4})\b", q_lower)
                    if yr_single:
                        y = yr_single.group(1)
                        contract.filters = [
                            FilterSpec(column=time_col, operator=">=", value=f"{y}-01-01"),
                            FilterSpec(column=time_col, operator="<=", value=f"{y}-12-31 23:59:59"),
                        ]

                contract.should_visualize = True
                contract.x_axis = time_col
                contract.sort = SortSpec(field=time_col, direction="asc")
            else:
                # Continuous / ordinal numeric or categorical line chart (e.g. Age)
                # Find X dimension after "by": e.g. "by age"
                by_m = re.search(r"\bby\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\b", q_lower)
                x_cand = by_m.group(1).strip() if by_m else None
                x_col = resolve_column_semantically(x_cand, list(cols.values())) if x_cand else None
                if not x_col:
                    x_col = next((real for low, real in cols.items() if low in q_lower or low.replace("_", " ") in q_lower), None)

                # Check for series dimension (e.g. "for male and female investors" or "by gender")
                series_dim = None
                for low, real in cols.items():
                    if real != x_col and (low in q_lower or low.replace("_", " ") in q_lower or any(alias in q_lower for alias in ("male", "female", "men", "women") if "gender" in low or "sex" in low)):
                        if contract.measure_column and real == contract.measure_column:
                            continue
                        series_dim = real
                        break

                # Resolve measure if "average <measure> by <dim>"
                avg_m = re.search(r"\b(?:average|mean|avg)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\s+by\b", q_lower)
                if avg_m:
                    m_phrase = avg_m.group(1).strip()
                    m_col = resolve_column_semantically(m_phrase, list(cols.values()))
                    if m_col:
                        contract.measure = m_col
                        contract.measure_column = m_col
                        contract.measure_expression = m_col
                        contract.aggregation = "AVG"

                if x_col:
                    contract.intent = "TREND"
                    contract.answer_type = "GROUPED"
                    contract.x_axis = x_col
                    if series_dim:
                        contract.series_dimension = series_dim
                        contract.group_by = [x_col, series_dim]
                        contract.suggested_chart_type = "line"
                    else:
                        contract.group_by = [x_col]
                        contract.suggested_chart_type = "line"
                    contract.sort = SortSpec(field=x_col, direction="asc")
                    contract.should_visualize = True

        # 8. Handle Part-to-whole / Pie / Donut
        if bool(re.search(r"\b(?:contribution|share|proportion|part of|pie|donut)\b", q_lower)):
            contract.should_visualize = True
            contract.suggested_chart_type = "pie"

        # 9. Handle Top-N Ranking -> Horizontal Bar
        if asks_ranking or re.search(r"\btop\s+\d+\b", q_lower):
            contract.should_visualize = True
            contract.suggested_chart_type = "horizontal_bar"

        # 10. Handle Clustered Multi-Measure Bar: e.g. "sales and profit by category"
        is_and_question = bool(re.search(r"\band\b", q_lower))
        if is_and_question and not is_temporal and len(contract.group_by) == 1:
            matching_numeric = [
                real for low, real in cols.items()
                if any(k in low for k in ("sales", "profit", "revenue", "quantity", "price", "cost"))
                and (low in q_lower or low.replace("_", " ") in q_lower)
            ]
            if len(matching_numeric) >= 2:
                contract.measure = matching_numeric[0]
                contract.measure_column = matching_numeric[0]
                contract.secondary_measure = matching_numeric[1]
                contract.secondary_measure_column = matching_numeric[1]
                contract.suggested_chart_type = "clustered_bar"
                contract.should_visualize = True

        return contract

    def _detect_calculation(
        self,
        question: str,
        q_lower: str,
        cols: dict[str, str],
        schema: dict[str, Any],
        dataset_id: str,
        query_id: str,
    ) -> Optional[QueryContract]:
        """Detect mathematical / derived metric queries deterministically."""
        is_diff = bool(re.search(r"\b(?:difference between|how much higher|how much more|gap between)\b", q_lower))
        is_margin = bool(re.search(r"\b(?:profit margin|margin based on)\b", q_lower))
        is_pct = bool(re.search(r"\b(?:what percentage of|what percent of|percentage of total|share of total)\b", q_lower))
        is_ratio = bool(re.search(r"\b(?:ratio of)\b", q_lower))
        is_sum = bool(re.search(r"\b(?:sum of total)\b", q_lower))
        is_sales_per_order = bool(re.search(r"\baverage\s+sales\s+per\s+order\b", q_lower))

        if not (is_diff or is_margin or is_pct or is_ratio or is_sum or is_sales_per_order):
            return None

        group_by = []
        by_m = re.search(r"\bby\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\b", q_lower)
        if by_m:
            grp_col = resolve_column_semantically(by_m.group(1).strip(), list(cols.values()))
            if grp_col:
                group_by = [grp_col]

        if is_sales_per_order:
            sales_col = resolve_column_semantically("sales", list(cols.values())) or "Sales"
            order_col = next((c for c in cols.values() if any(k in c.lower() for k in ("order_id", "orderid", "invoiceno", "transaction"))), "Order_ID")
            calc_spec = CalculationSpec(
                operator="DIVIDE",
                operands=[
                    OperandSpec(name="Total Sales", measure=sales_col, measure_column=sales_col, aggregation="SUM"),
                    OperandSpec(name="Total Orders", measure=order_col, measure_column=order_col, aggregation="COUNT_DISTINCT"),
                ],
                label="Average Sales Per Order",
            )
            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="CALCULATION",
                answer_type="SCALAR",
                measure=sales_col,
                aggregation="AVG",
                group_by=[],
                calculation=calc_spec,
                should_visualize=False,
            )

        if is_margin:
            sales_col = resolve_column_semantically("sales", list(cols.values())) or "Sales"
            profit_col = resolve_column_semantically("profit", list(cols.values())) or "Profit"
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
                answer_type="GROUPED" if group_by else "SCALAR",
                measure="Profit Margin",
                group_by=group_by,
                calculation=calc_spec,
                should_visualize=bool(group_by),
                suggested_chart_type="bar" if group_by else None,
            )

        if is_pct:
            sales_col = resolve_column_semantically("sales", list(cols.values())) or "Sales"
            cat_col = resolve_column_semantically("category", list(cols.values())) or "Category"
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

        if is_diff or is_sum or is_ratio:
            m_cand = []
            for phrase in ("sales", "profit", "revenue", "income", "salary", "quantity", "cost", "discount", "return"):
                if phrase in q_lower:
                    res = resolve_column_semantically(phrase, list(cols.values()))
                    if res and res not in m_cand:
                        m_cand.append(res)

            meas_a = m_cand[0] if len(m_cand) >= 1 else "Sales"
            meas_b = m_cand[1] if len(m_cand) >= 2 else ("Profit" if meas_a != "Profit" else "Sales")

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
                answer_type="GROUPED" if group_by else "SCALAR",
                measure=lbl,
                group_by=group_by,
                calculation=calc_spec,
                should_visualize=bool(group_by),
                suggested_chart_type="bar" if group_by else None,
            )

        return None

    def _heuristic_contract_fallback(
        self,
        question: str,
        schema: dict[str, Any],
        dataset_id: str,
        query_id: str,
    ) -> QueryContract:
        """Deterministic fallback if LLM planner fails or is unavailable."""
        q_lower = question.lower()
        cols = {str(c.get("name", "")).lower(): str(c.get("name", "")) for c in schema.get("columns", [])}

        # Check for genuinely unrelated query
        unrelated_triggers = ("weather", "temperature", "forecast", "capital of", "president", "who wrote", "horoscope", "recipe", "fifa", "cricket")
        if any(w in q_lower for w in unrelated_triggers):
            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                is_related_to_dataset=False,
                unrelated_reason="The question is not related to the currently loaded dataset.",
                intent="UNRELATED",
                answer_type="UNRELATED",
                should_visualize=False,
            )

        # Check for calculation queries first
        calc_contract = self._detect_calculation(question, q_lower, cols, schema, dataset_id, query_id)
        if calc_contract:
            return calc_contract

        # Generic column type classification based on schema dtypes and names
        datetime_cols = []
        numeric_cols = []
        cat_cols = []
        raw_cols_list = schema.get("columns", []) if isinstance(schema, dict) else schema or []
        for c in raw_cols_list:
            c_name = c.get("name", "") if isinstance(c, dict) else str(c)
            c_dtype = str(c.get("dtype", "")).lower() if isinstance(c, dict) else ""
            if not c_name:
                continue
            name_low = c_name.lower()
            is_currency_or_rate = any(k in name_low for k in ("income", "salary", "rate", "cost", "price", "fee", "expense", "charges", "sales", "revenue", "amount", "profit", "balance", "limit", "age", "quantity"))
            if is_currency_or_rate or any(nt in c_dtype for nt in ("int", "float", "real", "double", "num", "dec", "bigint")):
                numeric_cols.append(c_name)
            elif any(dt in c_dtype for dt in ("date", "time", "timestamp")) or any(k in name_low for k in ("invoicedate", "order_date", "orderdate", "date", "timestamp", "time", "year", "created_at", "saledate", "purchasedate")):
                datetime_cols.append(c_name)
            elif any(k in name_low for k in ("_id", "row_id", "customer_id", "product_id", "order_id", "code", "_no", "key", "uuid", "guid")) and not any(k in name_low for k in ("category", "sub_category", "segment", "region", "country", "state", "city")):
                pass
            else:
                cat_cols.append(c_name)

        def _get_word_variants(s: str) -> set[str]:
            v = {s}
            if s.endswith("ies"):
                v.add(s[:-3] + "y")
            elif s.endswith("es") and any(s.endswith(x) for x in ("shes", "ches", "xes", "sses", "zes")):
                v.add(s[:-2])
            elif s.endswith("s") and not s.endswith("ss"):
                v.add(s[:-1])
            elif s.endswith("y"):
                v.add(s[:-1] + "ies")
            else:
                v.add(s + "s")
            return v

        q_tokens = set(re.findall(r"\b[a-zA-Z_]+\b", q_lower))

        # Matched columns in order of appearance in question
        matched_numeric = [col for col in numeric_cols if col.lower() in q_lower or col.lower().replace("_", " ") in q_lower]
        matched_cat = [
            col for col in cat_cols
            if col.lower() in q_lower
            or col.lower().replace("_", " ") in q_lower
            or any(var in q_tokens for var in _get_word_variants(col.lower()))
        ]

        qty_col = next((real for low, real in cols.items() if any(k in low for k in ("quantity", "qty", "units", "sales_volume"))), None)
        price_col = next((real for low, real in cols.items() if any(k in low for k in ("unitprice", "unit_price", "price"))), None)
        cust_col = next((real for low, real in cols.items() if any(k in low for k in ("customer", "client", "user_id"))), None)
        country_col = next((real for low, real in cols.items() if any(k in low for k in ("country", "nation", "region", "location"))), None)
        date_col = datetime_cols[0] if datetime_cols else next((real for low, real in cols.items() if not any(ex in low for ex in ("income", "salary", "rate", "cost", "price", "fee", "expense", "charges", "sales", "revenue", "amount", "profit", "balance", "limit")) and any(k in low for k in ("invoicedate", "order_date", "orderdate", "date", "timestamp", "time", "year", "created_at", "saledate", "purchasedate"))), None)
        inv_col = next((real for low, real in cols.items() if any(k in low for k in ("invoiceno", "invoice", "orderid", "order_id", "transaction", "id"))), None)

        matched_cat_col = matched_cat[0] if matched_cat else None

        # Check ranking: top N
        top_match = re.search(r"\btop\s+(\d+)\b", q_lower)
        limit_val = int(top_match.group(1)) if top_match else None

        # Check grouping
        asks_grouping = bool(re.search(r"\b(?:by|per|across|for each|each|breakdown by)\b", q_lower)) or bool(top_match)
        group_by = []
        if (asks_grouping or top_match) and (country_col and ("country" in q_lower or "countries" in q_lower)):
            group_by = [country_col]
        elif (asks_grouping or top_match) and matched_cat_col:
            group_by = [matched_cat_col]

        # Time grain detection
        time_grain = "month"
        if any(w in q_lower for w in ("yearly", "year by year", "annually", "annual", "year")):
            time_grain = "year"
        elif any(w in q_lower for w in ("quarterly", "per quarter", "quarter")):
            time_grain = "quarter"
        elif any(w in q_lower for w in ("weekly", "per week", "week")):
            time_grain = "week"
        elif any(w in q_lower for w in ("daily", "day by day", "per day", "day")):
            time_grain = "day"

        # 0. GROUPED ENTITIES WITH AGGREGATE THRESHOLDS (HAVING queries)
        # e.g. "Show categories with more than 100 orders and average profit above 20"
        having_pattern = re.search(r"\b(?:show|find|list|get|which)\s+([a-zA-Z_]+)\s+(?:with|having|where|that have)\s+(.+)", q_lower)
        if having_pattern:
            entity_cand = having_pattern.group(1).strip()
            cond_str = having_pattern.group(2).strip()
            entity_dim = resolve_column_semantically(entity_cand, list(cols.values()))
            if not entity_dim:
                sing_cand = (entity_cand[:-3] + "y") if entity_cand.endswith("ies") else (entity_cand[:-1] if entity_cand.endswith("s") and not entity_cand.endswith("ss") else entity_cand)
                entity_dim = resolve_column_semantically(sing_cand, list(cols.values()))
            if not entity_dim and matched_cat:
                entity_dim = matched_cat[0]
            if not entity_dim and cat_cols:
                entity_dim = cat_cols[0]

            if entity_dim:
                having_list = []
                cond_parts = [p.strip() for p in re.split(r"\b(?:and|,)\b", cond_str) if p.strip()]
                meas_found = None
                meas_agg = "AVG"
                has_count = False

                for part in cond_parts:
                    # Check count condition: "more than 100 orders", "> 100 orders"
                    cnt_m = re.search(r"\b(?:more than|greater than|above|>|>=)\s*(\d+)\s*(?:orders|transactions|invoices|records|items|rows|sales|customers)?\b", part)
                    if not cnt_m:
                        cnt_m = re.search(r"\b(?:orders|transactions|invoices|records|items|rows|sales|customers)\s*(?:more than|greater than|above|>|>=)\s*(\d+)\b", part)
                    if cnt_m and not any(k in part for k in ("profit", "sales", "revenue", "price", "discount", "margin", "cost", "salary")):
                        op = ">=" if (">=" in cnt_m.group(0) or "at least" in cnt_m.group(0)) else ">"
                        val = int(cnt_m.group(1))
                        having_list.append(HavingSpec(expression="COUNT(*)", operator=op, value=val))
                        has_count = True
                        continue

                    # Check metric condition: "average profit above 20"
                    for phrase in ("profit", "sales", "revenue", "quantity", "discount", "margin", "cost", "price", "unitprice"):
                        if phrase in part:
                            res_meas = resolve_column_semantically(phrase, list(cols.values()))
                            if res_meas and res_meas != entity_dim:
                                meas_found = res_meas
                                break

                    part_agg = "AVG" if ("average" in part or "avg" in part or "mean" in part) else "SUM"
                    num_m = re.search(r"\b(?:above|greater than|more than|>|>=|<|<=|below|less than)\s*(-?\d+(?:\.\d+)?)\b", part)
                    if num_m and meas_found:
                        op = ">=" if ">=" in num_m.group(0) else ("<=" if "<=" in num_m.group(0) else (">" if any(w in num_m.group(0) for w in ("above", "greater", "more", ">")) else "<"))
                        val = float(num_m.group(1))
                        having_list.append(HavingSpec(column=meas_found, aggregation=part_agg, operator=op, value=val))
                        meas_agg = part_agg

                if having_list:
                    primary_m = meas_found or (matched_numeric[0] if matched_numeric else (numeric_cols[0] if numeric_cols else list(cols.values())[0]))
                    return QueryContract(
                        query_id=query_id,
                        dataset_id=dataset_id,
                        original_query=question,
                        intent="AGGREGATION",
                        answer_type="GROUPED",
                        measure=primary_m,
                        measure_column=primary_m,
                        measure_expression=primary_m,
                        aggregation=meas_agg,
                        dimensions=[entity_dim],
                        group_by=[entity_dim],
                        having=having_list,
                        secondary_measure=None,
                        secondary_measure_column=None,
                        sort=SortSpec(field=f"{meas_agg.lower()}_{primary_m.lower()}", direction="desc"),
                        limit=15,
                        should_visualize=False,
                    )

        # 1. SCATTER / RELATIONSHIP
        is_scatter = bool(re.search(r"\b(?:relationship|versus|vs\.?|against|correlation|scatter)\b", q_lower))
        if is_scatter:
            cand_cols = []
            for phrase in ("expected return", "expected_return", "investment avenue", "investment objective", "annual income", "unit price", "unit_price", "sales", "revenue", "profit", "quantity"):
                if phrase in q_lower:
                    res = resolve_column_semantically(phrase, list(cols.values()))
                    if res and res not in cand_cols:
                        cand_cols.append(res)
            for w in re.findall(r'[a-zA-Z_]+', q_lower):
                res = resolve_column_semantically(w, list(cols.values()))
                if res and res not in cand_cols:
                    cand_cols.append(res)

            # Check color_by (e.g. by gender)
            color_col = None
            for low, real in cols.items():
                if real not in cand_cols and (low in q_lower or low.replace("_", " ") in q_lower or any(alias in q_lower for alias in ("male", "female", "men", "women") if "gender" in low or "sex" in low)):
                    color_col = real
                    break

            # Check filters (e.g. "Mutual Funds investors")
            filters_list = []
            for val_phrase in ("mutual funds", "equity", "fixed deposits", "gold", "debentures", "government bonds", "male", "female"):
                if val_phrase in q_lower:
                    # find matching column
                    for low, real in cols.items():
                        if any(w in val_phrase for w in ("mutual", "fund", "equity", "gold", "deposit", "bond")) and any(k in low for k in ("avenue", "investment")):
                            filters_list.append(FilterSpec(column=real, operator="=", value=val_phrase.title() if "mutual" in val_phrase else val_phrase.capitalize()))
                            break

            x_c = cand_cols[0] if len(cand_cols) >= 1 else (matched_numeric[0] if matched_numeric else (numeric_cols[0] if numeric_cols else list(cols.values())[0]))
            y_c = cand_cols[1] if len(cand_cols) >= 2 else (matched_numeric[1] if len(matched_numeric) > 1 else (numeric_cols[1] if len(numeric_cols) > 1 else x_c))

            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="RELATIONSHIP",
                answer_type="RELATIONSHIP",
                aggregation=None,
                filters=filters_list,
                should_visualize=True,
                suggested_chart_type="scatter",
                x_axis=x_c,
                y_axis=y_c,
                color_by=color_col,
            )

        # 2. DISTRIBUTION / HISTOGRAM
        is_dist = bool(re.search(r"\b(?:distribution|histogram|spread)\b", q_lower))
        if is_dist:
            target_c = None
            for phrase in ("expected return", "expected_return", "investment avenue", "investment objective", "age", "income", "salary", "sales", "quantity", "price", "unitprice"):
                if phrase in q_lower:
                    res = resolve_column_semantically(phrase, list(cols.values()))
                    if res:
                        target_c = res
                        break
            if not target_c:
                for real in cols.values():
                    if real.lower() in q_lower or real.lower().replace("_", " ") in q_lower:
                        target_c = real
                        break
            if not target_c:
                target_c = matched_numeric[0] if matched_numeric else (numeric_cols[0] if numeric_cols else list(cols.values())[0])

            bin_match = re.search(r"\b(?:bin\s*width|bin\s*size|bins?\s*of)\s*(\d+(?:\.\d+)?)\b", q_lower)
            b_width = float(bin_match.group(1)) if bin_match else None

            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="DISTRIBUTION",
                answer_type="DISTRIBUTION",
                measure=target_c,
                measure_column=target_c,
                measure_expression=target_c,
                aggregation=None,
                should_visualize=True,
                suggested_chart_type="histogram",
                x_axis=target_c,
                bin_width=b_width,
            )

        # 3. LINE & MULTI-LINE (Temporal or Numeric X)
        is_trend = (
            bool(re.search(r"\b(?:over time|changed over time|change over time|trend|trend over time|historical trend|month by month|day by day|year by year|sales trend|revenue trend|quantity trend|timeline|trajectory)\b", q_lower))
            or (
                bool(re.search(r"\b(?:daily|weekly|monthly|quarterly|yearly|annually)\b", q_lower))
                and not any(m_word in q_lower for m_word in ("monthly income", "monthly salary", "monthly charges", "monthly rate", "monthly expense", "monthly cost", "daily rate", "hourly rate"))
            )
        )
        is_line_q = bool(re.search(r"\b(?:multi[-_\s]line|line chart|line graph|plot as a line|as a line chart|as line)\b", q_lower))
        is_multi_q = bool(re.search(r"\bmulti[-_\s]line\b", q_lower)) or any(w in q_lower for w in ("male and female", "female and male", "both genders"))
        if is_line_q or (is_trend and (date_col or group_by)):
            # Check for X column
            by_m = re.search(r"\bby\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\b", q_lower)
            x_cand = by_m.group(1).strip() if by_m else None
            x_col = resolve_column_semantically(x_cand, list(cols.values())) if x_cand else None
            if not x_col and date_col:
                x_col = date_col
            if not x_col:
                x_col = matched_numeric[0] if matched_numeric else (numeric_cols[0] if numeric_cols else list(cols.values())[0])

            # 1. Resolve measure FIRST (so it is not mistaken for a series dimension)
            meas_col = None
            avg_m = re.search(r"\b(?:average|mean|avg)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\s+by\b", q_lower)
            if avg_m:
                meas_col = resolve_column_semantically(avg_m.group(1).strip(), list(cols.values()))
            if not meas_col:
                for phrase in ("expected return", "expected_return", "total sales", "revenue", "sales", "profit", "quantity", "unitprice", "unit price", "price", "amount", "income"):
                    if phrase in q_lower:
                        res = resolve_column_semantically(phrase, list(cols.values()))
                        if res and res != x_col:
                            meas_col = res
                            break
            if not meas_col and matched_numeric:
                meas_col = next((c for c in matched_numeric if c != x_col), None)
            if not meas_col and numeric_cols:
                meas_col = next((c for c in numeric_cols if c != x_col), None)
            if not meas_col:
                meas_col = next((c for c in cols.values() if c != x_col), x_col)

            # 2. Check for series dimension ONLY if multi-line is explicitly requested or breakdown dimension is mentioned
            series_dim = None
            is_explicit_multi = (
                is_multi_q
                or bool(re.search(r"\b(?:multi[-_\s]line|multiple lines?)\b", q_lower))
                or bool(re.search(r"\b(?:for each|each)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\b", q_lower))
            )
            if is_explicit_multi and cat_cols:
                by_cat_m = re.search(r"\b(?:for each|each|by|across)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\b", q_lower)
                if by_cat_m:
                    cand_dim = resolve_column_semantically(by_cat_m.group(1).strip(), cat_cols)
                    if cand_dim and cand_dim not in (x_col, meas_col):
                        series_dim = cand_dim

                if not series_dim and (is_multi_q or bool(re.search(r"\b(?:multi[-_\s]line|multiple lines?)\b", q_lower))):
                    for low, real in cols.items():
                        if real in cat_cols and real not in (x_col, meas_col) and (low in q_lower or low.replace("_", " ") in q_lower or any(alias in q_lower for alias in ("male", "female", "men", "women") if "gender" in low or "sex" in low)):
                            series_dim = real
                            break

            # 3. Detect year range filters: "from 2011 to 2015" or "between 2011 and 2015"
            line_filters = []
            yr_match = re.search(r"\b(?:from|between)\s+(\d{4})\s+(?:to|and|-)\s+(\d{4})\b", q_lower)
            if yr_match and x_col == date_col:
                y1, y2 = yr_match.group(1), yr_match.group(2)
                line_filters.append(FilterSpec(column=x_col, operator=">=", value=f"{y1}-01-01"))
                line_filters.append(FilterSpec(column=x_col, operator="<=", value=f"{y2}-12-31 23:59:59"))
            else:
                yr_single = re.search(r"\b(?:in|for)\s+(\d{4})\b", q_lower)
                if yr_single and x_col == date_col:
                    y = yr_single.group(1)
                    line_filters.append(FilterSpec(column=x_col, operator=">=", value=f"{y}-01-01"))
                    line_filters.append(FilterSpec(column=x_col, operator="<=", value=f"{y}-12-31 23:59:59"))

            is_avg = bool(re.search(r"\b(?:average|mean|avg)\b", q_lower))
            agg = "AVG" if is_avg else "SUM"
            ans_groups = [x_col, series_dim] if series_dim else [x_col]
            suggested_type = "line"

            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="TIME_SERIES" if x_col == date_col else "TREND",
                answer_type="GROUPED",
                measure=meas_col,
                measure_column=meas_col,
                measure_expression=meas_col,
                aggregation=agg,
                dimensions=ans_groups,
                group_by=ans_groups,
                filters=line_filters,
                series_dimension=series_dim,
                should_visualize=True,
                suggested_chart_type=suggested_type,
                x_axis=x_col,
                y_axis=f"average_{meas_col.lower()}" if is_avg else f"total_{meas_col.lower()}",
                sort=SortSpec(field=x_col, direction="asc"),
            )

        # 4. GROUPED / CLUSTERED BAR (Multi-Measure or 2 Dimensions)
        is_grouped_bar = bool(re.search(r"\b(?:grouped|clustered)\b", q_lower)) or (len(matched_numeric) >= 2 and (matched_cat or "each" in q_lower or "by" in q_lower))
        if is_grouped_bar and len(cols) >= 3:
            cat_dim = None
            by_m = re.search(r"\b(?:for each|by|each)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\b", q_lower)
            if by_m:
                cat_dim = resolve_column_semantically(by_m.group(1).strip(), list(cols.values()))
            if not cat_dim and matched_cat:
                cat_dim = matched_cat[0]
            if not cat_dim:
                cat_dim = cat_cols[0] if cat_cols else list(cols.values())[0]

            m_candidates = []
            for phrase in ("age", "expected return", "expected_return", "sales", "profit", "revenue", "quantity", "unitprice"):
                if phrase in q_lower:
                    res = resolve_column_semantically(phrase, list(cols.values()))
                    if res and res != cat_dim and res not in m_candidates:
                        m_candidates.append(res)
            if len(m_candidates) < 2:
                for c in cols.values():
                    if c != cat_dim and c not in m_candidates:
                        m_candidates.append(c)
                    if len(m_candidates) == 2:
                        break

            m1 = m_candidates[0] if len(m_candidates) >= 1 else "Metric1"
            m2 = m_candidates[1] if len(m_candidates) >= 2 else "Metric2"
            is_avg = bool(re.search(r"\b(?:average|mean|avg)\b", q_lower))
            agg = "AVG" if is_avg else "SUM"

            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="AGGREGATION",
                answer_type="GROUPED",
                measure=m1,
                measure_column=m1,
                measure_expression=m1,
                secondary_measure=m2,
                secondary_measure_column=m2,
                aggregation=agg,
                dimensions=[cat_dim],
                group_by=[cat_dim],
                should_visualize=True,
                suggested_chart_type="grouped_bar",
                x_axis=cat_dim,
                y_axis=m1,
            )

        # 5. PROPORTION / CONTRIBUTION / PIE
        is_pie = bool(re.search(r"\b(?:contribution|proportion|share|ratio|pie|donut|percentage)\b", q_lower))
        if is_pie:
            target_cat = None
            by_m = re.search(r"\bby\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\b", q_lower)
            if by_m:
                target_cat = resolve_column_semantically(by_m.group(1).strip(), list(cols.values()))
            if not target_cat and matched_cat:
                target_cat = matched_cat[0]
            if not target_cat:
                target_cat = cat_cols[0] if cat_cols else list(cols.values())[0]

            # Check if there is an explicit numeric measure (e.g. "share of total sales")
            meas_col = None
            if matched_numeric:
                meas_col = matched_numeric[0]
            else:
                for phrase in ("sales", "profit", "revenue", "quantity", "unitprice", "cost", "amount"):
                    if phrase in q_lower:
                        cand = resolve_column_semantically(phrase, list(cols.values()))
                        if cand and cand != target_cat:
                            meas_col = cand
                            break

            if meas_col:
                is_avg = bool(re.search(r"\b(?:average|mean|avg)\b", q_lower))
                agg = "AVG" if is_avg else "SUM"
                y_label = f"total_{meas_col.lower()}" if agg == "SUM" else f"average_{meas_col.lower()}"
                return QueryContract(
                    query_id=query_id,
                    dataset_id=dataset_id,
                    original_query=question,
                    intent="PROPORTION",
                    answer_type="GROUPED",
                    measure=meas_col,
                    measure_column=meas_col,
                    measure_expression=meas_col,
                    aggregation=agg,
                    dimensions=[target_cat],
                    group_by=[target_cat],
                    should_visualize=True,
                    suggested_chart_type="pie",
                    x_axis=target_cat,
                    y_axis=y_label,
                )

            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="PROPORTION",
                answer_type="GROUPED",
                measure=target_cat,
                measure_column=target_cat,
                measure_expression=target_cat,
                aggregation="COUNT",
                dimensions=[target_cat],
                group_by=[target_cat],
                should_visualize=True,
                suggested_chart_type="pie",
                x_axis=target_cat,
                y_axis="investor_count",
            )

        # 6. TOP-N RANKING / HORIZONTAL BAR
        is_horiz = bool(re.search(r"\b(?:horizontal bar|horizontal)\b", q_lower))
        if (top_match or is_horiz) and (matched_cat or country_col or cat_cols):
            cat_dim = None
            dim_match = re.search(r"\btop\s*\d*\s*([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\s+by\b", q_lower)
            if dim_match:
                cat_dim = resolve_column_semantically(dim_match.group(1).strip(), list(cols.values()))
            if not cat_dim:
                by_m = re.search(r"\bby\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\b", q_lower)
                if by_m:
                    cand = resolve_column_semantically(by_m.group(1).strip(), list(cols.values()))
                    if cand and cand in cat_cols:
                        cat_dim = cand
            if not cat_dim and country_col:
                cat_dim = country_col
            if not cat_dim:
                cat_dim = matched_cat[0] if matched_cat else (cat_cols[0] if cat_cols else list(cols.values())[0])

            is_avg = bool(re.search(r"\b(?:average|mean|avg)\b", q_lower))
            meas_col = None
            meas_expr = None
            if qty_col and price_col and ("revenue" in q_lower or "turnover" in q_lower):
                meas_col = "total_revenue"
                meas_expr = f'"{qty_col}" * "{price_col}"'
            else:
                for phrase in ("revenue", "expected return", "expected_return", "age", "sales", "profit", "return", "quantity", "unitprice"):
                    if phrase in q_lower:
                        res = resolve_column_semantically(phrase, list(cols.values()))
                        if res and res != cat_dim:
                            meas_col = res
                            meas_expr = res
                            break

            agg = ("AVG" if is_avg else "SUM") if meas_col else "COUNT"
            meas_name = meas_col or cat_dim
            meas_expression = meas_expr or meas_name

            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="RANKING",
                answer_type="GROUPED",
                measure=meas_name,
                measure_column=meas_name,
                measure_expression=meas_expression,
                aggregation=agg,
                dimensions=[cat_dim],
                group_by=[cat_dim],
                limit=limit_val or 5,
                sort=SortSpec(field=f"ranking_metric", direction="desc"),
                should_visualize=True,
                suggested_chart_type="horizontal_bar",
                x_axis=cat_dim,
                y_axis=meas_name,
            )

        # 7. STANDARD BAR / GROUPED CATEGORICAL
        if "bar" in q_lower or asks_grouping:
            has_grouping_kw = bool(re.search(r"\b(?:by|for each|across)\b", q_lower))
            cat_dim = None
            if has_grouping_kw:
                by_m = re.search(r"\b(?:by|for each|across)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\b", q_lower)
                if by_m:
                    cat_dim = resolve_column_semantically(by_m.group(1).strip(), list(cols.values()))
                if not cat_dim and matched_cat:
                    cat_dim = matched_cat[0]
                if not cat_dim:
                    cat_dim = cat_cols[0] if cat_cols else list(cols.values())[0]

            is_min = bool(re.search(r"\b(?:min|minimum|lowest|smallest|least)\b", q_lower))
            is_max = bool(re.search(r"\b(?:max|maximum|highest|largest|peak|greatest)\b", q_lower))
            is_count = bool(re.search(r"\b(?:number of|count of|how many|records|investors)\b", q_lower))
            is_avg = bool(re.search(r"\b(?:average|mean|avg)\b", q_lower))

            meas_col = None
            for phrase in ("monthly income", "monthly salary", "daily rate", "hourly rate", "income", "salary", "age", "expected return", "expected_return", "sales", "revenue", "profit", "price", "unitprice", "quantity"):
                if phrase in q_lower:
                    res = resolve_column_semantically(phrase, list(cols.values()))
                    if res and res != cat_dim:
                        meas_col = res
                        break
            if not meas_col and matched_numeric:
                meas_col = matched_numeric[0]
            if not meas_col and numeric_cols:
                meas_col = numeric_cols[0]

            if is_min:
                agg = "MIN"
            elif is_max:
                agg = "MAX"
            elif is_avg:
                agg = "AVG"
            elif is_count or not meas_col:
                agg = "COUNT"
            else:
                agg = "SUM"

            if cat_dim and has_grouping_kw:
                meas = cat_dim if agg == "COUNT" else (meas_col or cat_dim)
                return QueryContract(
                    query_id=query_id,
                    dataset_id=dataset_id,
                    original_query=question,
                    intent="AGGREGATION",
                    answer_type="GROUPED",
                    measure=meas,
                    measure_column=meas,
                    measure_expression=meas,
                    aggregation=agg,
                    dimensions=[cat_dim],
                    group_by=[cat_dim],
                    requested_visualization=True,
                    should_visualize=True,
                    suggested_chart_type="bar",
                    x_axis=cat_dim,
                    y_axis=f"{agg.lower()}_{meas.lower()}",
                )
            else:
                # Explicit single-metric bar chart (no grouping)
                meas = meas_col or "metric"
                return QueryContract(
                    query_id=query_id,
                    dataset_id=dataset_id,
                    original_query=question,
                    intent="AGGREGATION",
                    answer_type="SCALAR",
                    measure=meas,
                    measure_column=meas,
                    measure_expression=meas,
                    aggregation=agg,
                    dimensions=[],
                    group_by=[],
                    requested_visualization=True,
                    should_visualize=True,
                    suggested_chart_type="bar",
                    x_axis=meas,
                    y_axis=f"{agg.lower()}_{meas.lower()}",
                )

        # 8. DERIVED REVENUE ONLY IF NO DIRECT SALES/REVENUE COLUMN EXISTS IN SCHEMA
        direct_sales_col = next((real for low, real in cols.items() if low in ("sales", "revenue", "turnover", "gross_sales")), None)
        if not direct_sales_col:
            direct_sales_col = resolve_column_semantically("sales", list(cols.values()))

        is_revenue = bool(re.search(r"\b(?:revenue|turnover)\b", q_lower)) or (not direct_sales_col and "sales" in q_lower)
        if is_revenue and not direct_sales_col and qty_col and price_col:
            expr = f'"{qty_col}" * "{price_col}"'
            sort = SortSpec(field="total_revenue", direction="desc") if (top_match or "highest" in q_lower) else None
            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="AGGREGATION",
                answer_type="GROUPED" if group_by else "SCALAR",
                measure="revenue",
                measure_column=f"{qty_col} * {price_col}",
                measure_expression=expr,
                aggregation="SUM",
                group_by=group_by,
                sort=sort,
                limit=limit_val,
                should_visualize=bool(group_by),
                suggested_chart_type="bar" if group_by else None,
                x_axis=group_by[0] if group_by else None,
                y_axis="total_revenue" if group_by else None,
            )

        # 9. DISTINCT COUNT
        is_distinct = bool(re.search(r"\b(?:unique|distinct)\b", q_lower))
        if is_distinct:
            dist_col = None
            if any(k in q_lower for k in ("customer", "client", "user")):
                dist_col = next((c for c in cols.values() if any(k in c.lower() for k in ("customer", "client", "user"))), None)
            if not dist_col:
                dist_col = next((c for c in cols.values() if any(k in c.lower() for k in ("customer", "user", "client", "id", "order", "invoice"))), None)
            if not dist_col:
                dist_col = list(cols.values())[0]
            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="AGGREGATION",
                answer_type="SCALAR",
                measure=dist_col,
                measure_column=dist_col,
                measure_expression=dist_col,
                aggregation="COUNT_DISTINCT",
                group_by=[],
                limit=None,
                should_visualize=False,
            )

        # 10. GENERIC METRIC IN QUESTION
        metric_col = None
        if "units sold" in q_lower and qty_col:
            metric_col = qty_col
        else:
            # Check semantic resolution for question keywords first
            for phrase in ("sales", "profit", "revenue", "quantity", "discount", "price"):
                if phrase in q_lower:
                    res = resolve_column_semantically(phrase, list(cols.values()))
                    if res:
                        metric_col = res
                        break
            if not metric_col:
                q_nospace = q_lower.replace(" ", "")
                for low, real in cols.items():
                    if low in q_lower or low.replace("_", " ") in q_lower or low in q_nospace:
                        metric_col = real
                        break
        if metric_col:
            is_min = bool(re.search(r"\b(?:min|minimum|lowest|smallest|least)\b", q_lower))
            is_max = bool(re.search(r"\b(?:max|maximum|highest|largest|peak|greatest)\b", q_lower))
            is_avg = bool(re.search(r"\b(?:average|mean|avg)\b", q_lower))
            is_count = bool(re.search(r"\b(?:count|number of|how many)\b", q_lower))
            if is_min:
                agg = "MIN"
            elif is_max:
                agg = "MAX"
            elif is_avg:
                agg = "AVG"
            elif is_count:
                agg = "COUNT"
            else:
                agg = "SUM"
            ans_type = "GROUPED" if group_by else "SCALAR"
            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="AGGREGATION",
                answer_type=ans_type,
                measure=metric_col,
                measure_column=metric_col,
                measure_expression=metric_col,
                aggregation=agg,
                group_by=group_by,
                limit=limit_val,
                should_visualize=bool(group_by),
            )

        return QueryContract(
            query_id=query_id,
            dataset_id=dataset_id,
            original_query=question,
            intent="DATA_RETRIEVAL",
            answer_type="RECORDS",
            limit=50,
        )


# Singleton
semantic_planner = SemanticPlanner()
