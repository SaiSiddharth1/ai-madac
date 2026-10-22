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
        is_revenue_question = bool(re.search(r"\b(?:revenue|turnover|total sales)\b", q_lower))

        if is_units_question and not is_price_question and not is_revenue_question:
            # Measure MUST NOT be UnitPrice or any price column!
            if contract.measure_column and any(p in contract.measure_column.lower() for p in ("price", "cost", "unitprice")):
                # Find the true quantity column
                true_qty = next((real for low, real in cols.items() if any(k in low for k in ("quantity", "qty", "units", "sales_volume"))), None)
                if true_qty:
                    contract.measure_column = true_qty
                    contract.measure_expression = true_qty
                    contract.measure = "Quantity"

        if is_price_question and not is_revenue_question:
            # Measure MUST NOT be Quantity!
            if contract.measure_column and any(q in contract.measure_column.lower() for q in ("quantity", "qty", "units")):
                true_price = next((real for low, real in cols.items() if any(k in low for k in ("unitprice", "price", "unit_price", "rate"))), None)
                if true_price:
                    contract.measure_column = true_price
                    contract.measure_expression = true_price
                    contract.measure = "UnitPrice"

        # 3. Handle Derived Revenue:
        if is_revenue_question:
            # Check if dataset has a direct revenue column
            direct_rev = next((real for low, real in cols.items() if "revenue" in low or "turnover" in low), None)
            if not direct_rev:
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
                if any(k in low for k in ("quantity", "price", "unitprice", "age", "income", "amount", "return", "sales"))
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
                contract.group_by = [time_col]
                contract.should_visualize = True
                contract.suggested_chart_type = "line"
                contract.x_axis = time_col
                contract.sort = SortSpec(field=time_col, direction="asc")

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

        # Detect columns in question
        qty_col = next((real for low, real in cols.items() if any(k in low for k in ("quantity", "qty", "units", "sales_volume"))), None)
        price_col = next((real for low, real in cols.items() if any(k in low for k in ("unitprice", "price", "unit_price", "rate", "cost"))), None)
        cust_col = next((real for low, real in cols.items() if any(k in low for k in ("customer", "client", "user_id"))), None)
        country_col = next((real for low, real in cols.items() if any(k in low for k in ("country", "nation", "region", "location"))), None)
        date_col = next((real for low, real in cols.items() if any(k in low for k in ("date", "time", "year", "month", "invoicedate"))), None)
        inv_col = next((real for low, real in cols.items() if any(k in low for k in ("invoiceno", "invoice", "orderid", "order_id", "transaction", "id"))), None)

        # Find any matching categorical column in question
        matched_cat_col = None
        for low, real in cols.items():
            clean_low = low.replace("_", " ")
            clean_stem = clean_low.rstrip("s")
            words = [w.rstrip("s") for w in re.findall(r"\b\w+\b", q_lower)]
            if any(w == clean_stem or clean_stem in w or w == clean_low for w in words):
                matched_cat_col = real
                break

        # Check ranking: top N
        top_match = re.search(r"\btop\s+(\d+)\b", q_lower)
        limit_val = int(top_match.group(1)) if top_match else None
        sort_spec = SortSpec(field="metric", direction="desc") if (top_match or "most" in q_lower or "highest" in q_lower) else None

        # Check grouping
        asks_grouping = bool(re.search(r"\b(?:by|per|across|for each|each|breakdown by)\b", q_lower)) or bool(top_match)
        group_by = []
        if (asks_grouping or top_match) and (country_col and ("country" in q_lower or "countries" in q_lower)):
            group_by = [country_col]
        elif (asks_grouping or top_match) and matched_cat_col and matched_cat_col != qty_col and matched_cat_col != price_col:
            group_by = [matched_cat_col]

        # 1. SCATTER / RELATIONSHIP
        is_scatter = bool(re.search(r"\b(?:relationship|versus|vs\.?)\b", q_lower))
        if is_scatter and qty_col and price_col:
            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="RELATIONSHIP",
                answer_type="RELATIONSHIP",
                should_visualize=True,
                suggested_chart_type="scatter",
                x_axis=qty_col,
                y_axis=price_col,
            )

        # 2. DISTRIBUTION / HISTOGRAM
        is_dist = bool(re.search(r"\b(?:distribution|histogram)\b", q_lower))
        if is_dist and qty_col:
            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="DISTRIBUTION",
                answer_type="DISTRIBUTION",
                measure=qty_col,
                measure_column=qty_col,
                measure_expression=qty_col,
                should_visualize=True,
                suggested_chart_type="histogram",
                x_axis=qty_col,
            )

        # 3. TIME-BASED TREND
        is_trend = bool(re.search(
            r"\b(?:over time|changed over time|change over time|trend|trend over time|historical trend|daily|weekly|monthly|quarterly|yearly|annually|month by month|day by day|year by year|sales trend|revenue trend|quantity trend|timeline|trajectory)\b",
            q_lower
        ))
        if is_trend and (date_col or group_by):
            trend_dim = date_col or (group_by[0] if group_by else None)
            expr = f'"{qty_col}" * "{price_col}"' if (qty_col and price_col and "revenue" in q_lower) else (qty_col or "metric")
            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="TIME_SERIES",
                answer_type="GROUPED",
                measure="revenue" if "revenue" in q_lower else "Quantity",
                measure_column=f"{qty_col} * {price_col}" if (qty_col and price_col and "revenue" in q_lower) else qty_col,
                measure_expression=expr,
                aggregation="SUM",
                time_column=trend_dim,
                group_by=[trend_dim] if trend_dim else [],
                should_visualize=True,
                suggested_chart_type="line",
                x_axis=trend_dim,
                y_axis="total_revenue" if "revenue" in q_lower else "total_quantity",
                sort=SortSpec(field=trend_dim, direction="asc") if trend_dim else None,
            )

        # 4. REVENUE
        is_revenue = bool(re.search(r"\b(?:revenue|turnover|total sales)\b", q_lower))
        if is_revenue:
            expr = f'"{qty_col}" * "{price_col}"' if qty_col and price_col else "revenue"
            sort = SortSpec(field="total_revenue", direction="desc") if (top_match or "highest" in q_lower) else None
            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="AGGREGATION",
                answer_type="GROUPED" if group_by else "SCALAR",
                measure="revenue",
                measure_column=f"{qty_col} * {price_col}" if qty_col and price_col else "revenue",
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

        # 5. TRANSACTIONS / ORDERS / INVOICES
        is_trans = bool(re.search(r"\b(?:transactions?|orders?|invoices?)\b", q_lower))
        if is_trans:
            agg_type = "COUNT_DISTINCT" if (inv_col and "unique" in q_lower) else "COUNT"
            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="AGGREGATION",
                answer_type="GROUPED" if group_by else "SCALAR",
                measure=inv_col or "InvoiceNo",
                measure_column=inv_col or "InvoiceNo",
                measure_expression=inv_col or "InvoiceNo",
                aggregation=agg_type,
                group_by=group_by,
                limit=limit_val,
                should_visualize=bool(group_by),
                suggested_chart_type="bar" if group_by else None,
            )

        # 6. PRICE
        is_price = bool(re.search(r"\b(?:unit\s*price|price per unit|unitprice|price)\b", q_lower))
        is_avg = bool(re.search(r"\b(?:average|mean|avg)\b", q_lower))
        if is_price and price_col and not is_revenue:
            agg = "AVG" if is_avg else "SUM"
            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="AGGREGATION",
                answer_type="GROUPED" if group_by else "SCALAR",
                measure="UnitPrice",
                measure_column=price_col,
                measure_expression=price_col,
                aggregation=agg,
                group_by=group_by,
                limit=limit_val,
                should_visualize=bool(group_by),
            )

        # 7. UNITS / QUANTITY
        is_units = (
            bool(re.search(r"\b(?:quantity|items?|units?\s+sold|items?\s+sold)\b", q_lower))
            or (bool(re.search(r"\bunits?\b", q_lower)) and not is_price)
        )
        if is_units and qty_col:
            agg = "AVG" if is_avg else "SUM"
            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="AGGREGATION",
                answer_type="GROUPED" if group_by else "SCALAR",
                measure="Quantity",
                measure_column=qty_col,
                measure_expression=qty_col,
                aggregation=agg,
                group_by=group_by,
                limit=limit_val,
                should_visualize=bool(group_by),
                suggested_chart_type="bar" if group_by else None,
            )

        # 8. UNIQUE CUSTOMERS
        is_unique = bool(re.search(r"\b(?:unique|distinct)\b", q_lower))
        if is_unique and cust_col:
            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="AGGREGATION",
                answer_type="SCALAR",
                measure="CustomerID",
                measure_column=cust_col,
                measure_expression=cust_col,
                aggregation="COUNT_DISTINCT",
            )

        # 9. POPULAR / RANKED CATEGORY (e.g. Most popular investment avenue)
        is_popular = bool(re.search(r"\b(?:most popular|popular|highest count|top avenue|favorite)\b", q_lower))
        if is_popular and matched_cat_col:
            return QueryContract(
                query_id=query_id,
                dataset_id=dataset_id,
                original_query=question,
                intent="AGGREGATION",
                answer_type="GROUPED",
                measure=matched_cat_col,
                measure_column=matched_cat_col,
                measure_expression=matched_cat_col,
                aggregation="COUNT",
                group_by=[matched_cat_col],
                sort=SortSpec(field="investor_count", direction="desc"),
                limit=1,
                should_visualize=False,
            )

        # 10. GENERIC METRIC IN QUESTION
        metric_col = next((real for low, real in cols.items() if low in q_lower or low.replace("_", " ") in q_lower), None)
        if metric_col:
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
