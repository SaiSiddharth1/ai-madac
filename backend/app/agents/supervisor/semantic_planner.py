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
from app.schemas.query_contract import FilterSpec, QueryContract, SortSpec
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

2. STRICT AGGREGATION RULES:
   - "total", "sum", "overall", "aggregate" -> SUM
   - "average", "mean", "avg" -> AVG
   - "minimum", "min", "lowest" -> MIN
   - "maximum", "max", "highest" -> MAX
   - "how many rows", "number of records", "count" -> COUNT
   - "unique customers", "distinct customers", "how many different customers" -> COUNT_DISTINCT on the customer identifier column. NEVER replace distinct counting with row counting!

3. GROUPING MUST NEVER BE INVENTED:
   - If the user asks: "What is the total revenue?" or "What are the units sold?", NO GROUPING IS REQUESTED!
     answer_type MUST be "SCALAR", group_by MUST be empty [].
     NEVER introduce grouping by country, segment, category, or date unless the user explicitly asks for it!
   - If the user asks: "What is the total revenue by country?", THEN AND ONLY THEN:
     answer_type MUST be "GROUPED", group_by MUST be ["Country"].
   - If the user asks: "Top 10 countries by revenue", group_by MUST be ["Country"], limit MUST be 10, sort MUST be {"field": "revenue", "direction": "desc"}.

4. VISUALIZATION RULES:
   - should_visualize should be true if:
     a) Explicitly requested: "chart", "plot", "visualize", "graph", "histogram", "scatter", "distribution", "pie", "bar".
     b) Naturally visual: distributions, multi-category comparisons, rankings, time-series trends, relationships/correlations.
   - should_visualize should be false for simple scalar metrics ("What is the total revenue?", "How many rows?") unless explicitly requested to plot.
   - Chart types:
     - Distribution of one numeric column -> "histogram" (RAW OBSERVATIONS, no aggregation!)
     - Relationship between two numeric columns -> "scatter" (RAW OBSERVATIONS, no aggregation!)
     - Category + numeric metric -> "bar" (or "horizontal_bar" if long labels or top-N ranking)
     - Time + numeric metric -> "line"
     - Small part-to-whole -> "pie"

5. DATASET RELEVANCE:
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
            sort=sort,
            limit=limit,
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

        # 1. Prevent unwanted grouping, EXCEPT for time-series / temporal queries:
        is_temporal = bool(re.search(
            r"\b(?:over time|changed over time|change over time|trend|trend over time|historical trend|daily|weekly|monthly|quarterly|yearly|annually|month by month|day by day|year by year|sales trend|revenue trend|quantity trend)\b",
            q_lower
        ))
        is_explicit_line = bool(re.search(r"\b(?:line chart|line graph|plot as a line|in line chart|as a line chart|as line)\b", q_lower))

        asks_grouping = bool(re.search(r"\b(?:by|per|across|for each|each|breakdown by)\b", q_lower)) or is_temporal
        asks_ranking = bool(re.search(r"\b(?:top|bottom|highest|lowest)\s+\d*\b", q_lower))
        if not asks_grouping and not asks_ranking and not is_temporal:
            if any(term in q_lower for term in ("total revenue", "total units", "total quantity", "average unit price", "average age", "unique customers", "total sales")):
                contract.group_by = []
                contract.answer_type = "SCALAR"
                contract.should_visualize = contract.requested_visualization  # only visualize if user explicitly asked

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

        # If user asked for sales and dataset has a direct Sales column, direct column MUST take precedence
        if is_sales_question and sales_col:
            contract.measure = sales_col
            contract.measure_column = sales_col
            contract.measure_expression = sales_col
            if not contract.aggregation:
                contract.aggregation = "SUM"

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

        # 6. Distinct counting:
        if bool(re.search(r"\b(?:unique|distinct)\b", q_lower)):
            contract.aggregation = "COUNT_DISTINCT"

        # 7. Handle Time Series / Trend / Line Chart:
        if is_temporal or is_explicit_line:
            # Generic time column resolution from dataset schema
            time_col = next((
                real for low, real in cols.items()
                if any(k in low for k in ("invoicedate", "order_date", "orderdate", "date", "timestamp", "time", "year", "month", "day", "created_at", "saledate", "purchasedate"))
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
            if any(dt in c_dtype for dt in ("date", "time", "timestamp")) or any(k in name_low for k in ("invoicedate", "order_date", "orderdate", "date", "timestamp", "time", "year", "month", "day")):
                datetime_cols.append(c_name)
            elif any(nt in c_dtype for nt in ("int", "float", "real", "double", "num", "dec")) or any(k in name_low for k in ("sales", "profit", "revenue", "quantity", "unitprice", "price", "amount", "margin", "cost", "discount", "age", "salary")):
                numeric_cols.append(c_name)
            else:
                cat_cols.append(c_name)

        # Matched columns in order of appearance in question
        matched_numeric = [col for col in numeric_cols if col.lower() in q_lower or col.lower().replace("_", " ") in q_lower]
        matched_cat = [col for col in cat_cols if col.lower() in q_lower or col.lower().replace("_", " ") in q_lower or col.lower().rstrip("s") in q_lower]

        qty_col = next((real for low, real in cols.items() if any(k in low for k in ("quantity", "qty", "units", "sales_volume"))), None)
        price_col = next((real for low, real in cols.items() if any(k in low for k in ("unitprice", "unit_price", "price"))), None)
        cust_col = next((real for low, real in cols.items() if any(k in low for k in ("customer", "client", "user_id"))), None)
        country_col = next((real for low, real in cols.items() if any(k in low for k in ("country", "nation", "region", "location"))), None)
        date_col = datetime_cols[0] if datetime_cols else next((real for low, real in cols.items() if any(k in low for k in ("date", "time", "year", "month", "invoicedate"))), None)
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
        is_trend = bool(re.search(
            r"\b(?:over time|changed over time|change over time|trend|trend over time|historical trend|daily|weekly|monthly|quarterly|yearly|annually|month by month|day by day|year by year|sales trend|revenue trend|quantity trend|timeline|trajectory)\b",
            q_lower
        ))
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
            cat_dim = None
            by_m = re.search(r"\b(?:by|for each|across)\s+([a-zA-Z_]+(?:\s+[a-zA-Z_]+)?)\b", q_lower)
            if by_m:
                cat_dim = resolve_column_semantically(by_m.group(1).strip(), list(cols.values()))
            if not cat_dim and matched_cat:
                cat_dim = matched_cat[0]
            if not cat_dim:
                cat_dim = cat_cols[0] if cat_cols else list(cols.values())[0]

            is_count = bool(re.search(r"\b(?:number of|count of|how many|records|investors)\b", q_lower))
            is_avg = bool(re.search(r"\b(?:average|mean|avg)\b", q_lower))
            meas_col = None
            for phrase in ("age", "expected return", "expected_return", "sales", "revenue", "profit", "price", "unitprice"):
                if phrase in q_lower:
                    res = resolve_column_semantically(phrase, list(cols.values()))
                    if res and res != cat_dim:
                        meas_col = res
                        break

            agg = "COUNT" if is_count or not meas_col else ("AVG" if is_avg else "SUM")
            meas = cat_dim if agg == "COUNT" else meas_col

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
                should_visualize=True,
                suggested_chart_type="bar",
                x_axis=cat_dim,
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
            is_avg = bool(re.search(r"\b(?:average|mean|avg)\b", q_lower))
            agg = "AVG" if is_avg else ("COUNT" if "count" in q_lower else "SUM")
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
