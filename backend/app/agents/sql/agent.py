"""
SQL Agent for AI-MADAC.
Translates structured Query Contracts and natural language questions into safe,
deterministic SQL queries, executes them against PostgreSQL/SQLite, and validates the result shape.
"""

import datetime
from decimal import Decimal
import json
import logging
import re
from typing import Any, Optional

import pandas as pd
import sqlalchemy as sa
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import AgentState
from app.core.llm_provider import get_llm
from app.db.database import sync_engine
from app.schemas.query_contract import QueryContract
from app.utils.serialization import to_json_safe
from app.utils.sql_validator import sanitize_table_name, validate_sql
from app.utils.temporal import is_temporal_column

logger = logging.getLogger(__name__)

SQL_AGENT_PROMPT = """You are an expert SQL Data Analyst Agent.
Your task is to write a PostgreSQL/SQLite compatible SELECT query to answer the user's question based on the provided table schema.

CRITICAL RULES:
1. ONLY generate SELECT statements. Never write INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, or CREATE.
2. Use the EXACT table name provided in double quotes: "{table_name}".
3. Use the EXACT column names provided in double quotes.
4. If the question asks for a single scalar metric (e.g. "total revenue", "total quantity", "average price") with NO grouping requested, NEVER use GROUP BY! Return a single scalar row.
5. If the user asks for grouping (e.g. "by country"), GROUP BY the requested dimension.
6. For unique/distinct counting (e.g. "unique customers"), use COUNT(DISTINCT "column_name").
7. If calculating revenue and no 'revenue' column exists, but 'Quantity' and 'UnitPrice' exist, use SUM("Quantity" * "UnitPrice").
8. Apply NOT NULL filters to grouping dimensions and numeric measures.
9. Output ONLY the raw SQL query. Do NOT include markdown code blocks or explanations.

Schema details for table "{table_name}":
{schema_description}
"""


class SQLAgent:
    """
    SQL Agent executing deterministic and validated queries based on the Query Contract.
    """

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm(temperature=0.0)
        return self._llm

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Execute SQL query based on Query Contract and return validated tabular data.
        """
        table_name = state.get("table_name", "")
        question = state.get("question", "")
        schema = state.get("schema", {})
        query_id = state.get("query_id", "")
        dataset_id = state.get("dataset_id", "")

        safe_table = sanitize_table_name(table_name)
        if not safe_table:
            return {
                "errors": ["Invalid table name provided for SQL execution."],
                "sql_result": None,
            }

        # Retrieve or construct Query Contract
        contract_data = state.get("query_contract")
        if not contract_data:
            from app.agents.supervisor.semantic_planner import semantic_planner
            contract = semantic_planner.plan_query(
                question=question,
                schema=schema,
                dataset_id=dataset_id,
                query_id=query_id,
                dataset_profile=state.get("dataset_profile"),
            )
            contract_data = contract.model_dump()
        else:
            contract = QueryContract(**contract_data)

        # Check for unrelated question
        if not contract.is_related_to_dataset or contract.intent == "UNRELATED":
            unrelated_msg = contract.unrelated_reason or (
                "I can't generate an answer or chart for this question because it is not related to the currently loaded dataset."
            )
            return {
                "sql_query": "-- Query not related to dataset",
                "sql_result": {
                    "query": None,
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "displayed_row_count": 0,
                    "message": unrelated_msg,
                    "response_type": "UNRELATED",
                },
                "validated_result": None,
                "agents_used": state.get("agents_used", []) + ["sql_agent"],
            }

        # Handle metadata questions directly
        q_clean = question.lower().strip()
        schema_cols = schema.get("columns", [])
        if re.search(r"\b(?:how many|number of|count of)\s+columns\b", q_clean):
            col_count = len(schema_cols)
            return {
                "sql_query": f"-- Schema inspection\nSELECT {col_count} AS total_columns",
                "sql_result": {
                    "query": f"SELECT {col_count} AS total_columns",
                    "source_question": question,
                    "columns": ["total_columns"],
                    "rows": [{"total_columns": col_count}],
                    "row_count": 1,
                    "displayed_row_count": 1,
                    "response_type": "SCALAR",
                    "scalar_value": col_count,
                    "formatted_value": str(col_count),
                    "metric_name": "Total Columns",
                },
                "validated_result": {
                    "question_id": query_id,
                    "dataset_id": dataset_id,
                    "operation": "METADATA",
                    "columns_used": ["total_columns"],
                    "result": [{"total_columns": col_count}],
                    "row_count": 1,
                    "validated": True,
                    "query": f"SELECT {col_count} AS total_columns",
                },
                "agents_used": state.get("agents_used", []) + ["sql_agent"],
            }

        if re.search(r"\b(?:what are the column names|column names|list (?:the )?columns|what columns)\b", q_clean):
            rows = [{"column_name": c.get("name"), "data_type": str(c.get("dtype"))} for c in schema_cols]
            return {
                "sql_query": "-- Schema inspection\nSELECT column_name, data_type FROM schema",
                "sql_result": {
                    "query": "SELECT column_name, data_type FROM schema",
                    "source_question": question,
                    "columns": ["column_name", "data_type"],
                    "rows": rows,
                    "row_count": len(rows),
                    "displayed_row_count": len(rows),
                    "response_type": "GROUPED",
                },
                "validated_result": {
                    "question_id": query_id,
                    "dataset_id": dataset_id,
                    "operation": "METADATA",
                    "columns_used": ["column_name", "data_type"],
                    "result": rows,
                    "row_count": len(rows),
                    "validated": True,
                    "query": "SELECT column_name, data_type FROM schema",
                },
                "agents_used": state.get("agents_used", []) + ["sql_agent"],
            }

        # Deterministic Query Generation from validated contract
        sql_query = self._build_query_from_contract(contract, schema, safe_table)

        # Validate SQL
        is_valid, error_msg = validate_sql(sql_query)
        if not is_valid:
            logger.warning(f"Contract SQL invalid: {error_msg}. Using fallback query builder.")
            sql_query = self._build_fallback_query(question, schema, safe_table)

        logger.info("========== AI-MADAC DEBUG ==========\nGENERATED SQL:\n%s", sql_query)

        # Execute Query
        try:
            escaped_sql = sql_query.replace("%", "%%") if ("%" in sql_query and not re.search(r"%\([a-zA-Z0-9_]+\)s", sql_query)) else sql_query
            df = pd.read_sql(sa.text(escaped_sql), con=sync_engine)

            # Phase 6: Strict Result Validation against Query Contract
            df, sql_query = self._validate_and_repair_result(df, contract, schema, safe_table, sql_query)

            columns = df.columns.tolist()
            # Clean missing values and deeply convert datetimes/timestamps/decimals to JSON-serializable primitives
            rows = to_json_safe(df)

            # Extract scalar properties if scalar result
            scalar_value = None
            formatted_value = None
            metric_name = None

            if contract.answer_type == "SCALAR" and len(rows) >= 1 and len(columns) >= 1:
                col = columns[0]
                raw_val = rows[0].get(col)
                scalar_value = to_json_safe(raw_val)
                metric_name = col.replace("_", " ").title()
                if isinstance(raw_val, (int, float, Decimal)):
                    num_val = float(raw_val)
                    if "revenue" in col.lower() or "price" in col.lower() or "sales" in col.lower():
                        formatted_value = f"{num_val:,.2f}"
                    elif isinstance(raw_val, (float, Decimal)):
                        formatted_value = f"{num_val:,.2f}"
                    else:
                        formatted_value = f"{int(num_val):,}"
                else:
                    formatted_value = str(raw_val)

            # Deterministic calculation layer
            calc_result_dict = None
            if contract.calculation:
                from app.analytics.calculation_engine import CalculationEngine
                df, calc_res = CalculationEngine.compute_from_dataframe(df, contract.calculation)
                if calc_res:
                    calc_result_dict = calc_res.model_dump()
                    if not calc_res.grouped and calc_res.value is not None:
                        scalar_value = calc_res.value
                        formatted_value = calc_res.formatted_value
                        metric_name = calc_res.label
                columns = df.columns.tolist()
                rows = to_json_safe(df)

            result_data = {
                "query": sql_query,
                "source_question": question,
                "analysis_plan": contract.model_dump(),
                "columns": columns,
                "rows": rows[:1000],
                "row_count": len(df),
                "displayed_row_count": len(rows[:1000]),
                "response_type": contract.answer_type,
                "scalar_value": scalar_value,
                "formatted_value": formatted_value,
                "metric_name": metric_name,
                "calculation_result": calc_result_dict,
            }

            validated_result = {
                "question_id": query_id,
                "dataset_id": dataset_id,
                "operation": contract.intent,
                "columns_used": columns,
                "filters": [f.model_dump() for f in contract.filters],
                "group_by": contract.group_by,
                "result": rows[:1000],
                "row_count": len(df),
                "validated": True,
                "query": sql_query,
                "response_type": contract.answer_type,
                "scalar_value": scalar_value,
                "formatted_value": formatted_value,
                "metric_name": metric_name,
                "calculation_result": calc_result_dict,
            }

            return {
                "sql_query": sql_query,
                "sql_result": result_data,
                "validated_result": validated_result,
                "agents_used": state.get("agents_used", []) + ["sql_agent"],
            }

        except Exception as e:
            logger.error(f"SQL execution error for '{sql_query}': {e}", exc_info=True)
            return {
                "sql_query": sql_query,
                "sql_result": None,
                "errors": state.get("errors", []) + [f"SQL Execution Error: {str(e)}"],
                "agents_used": state.get("agents_used", []) + ["sql_agent"],
            }

    def _build_query_from_contract(
        self,
        contract: QueryContract,
        schema: dict[str, Any],
        safe_table: str,
    ) -> str:
        """
        Build exact deterministic SQL matching the Query Contract without guessing or unwanted grouping.
        """
        cols_map = {}
        for c in (schema.get("columns", []) if isinstance(schema, dict) else schema or []):
            name = c.get("name", "") if isinstance(c, dict) else str(c)
            if name:
                cols_map[name.lower()] = name

        def quote_col(col_name: str) -> str:
            # Check if column is in schema
            exact = cols_map.get(col_name.lower(), col_name)
            return f'"{exact}"'

        def quote_expr(expr: Optional[str]) -> str:
            if not expr:
                return "*"
            # If expression has arithmetic e.g. Quantity * UnitPrice, quote identifiers
            parts = re.split(r"(\s*[\*\+\-\/]\s*)", expr)
            res = []
            for part in parts:
                cleaned = part.strip()
                if cleaned in ("*", "+", "-", "/"):
                    res.append(f" {cleaned} ")
                elif cleaned:
                    clean_id = cleaned.replace('"', '').strip()
                    if clean_id.lower() in cols_map:
                        res.append(f'"{cols_map[clean_id.lower()]}"')
                    else:
                        res.append(f'"{clean_id}"')
            return "".join(res)

        # Filters
        where_conditions = []
        op_map = {
            "=": "=", "==": "=", "equals": "=", "equal": "=",
            "!=": "!=", "<>": "!=", "not_equal": "!=", "not_equals": "!=",
            ">": ">", "greater_than": ">",
            ">=": ">=", "greater_than_or_equal": ">=", "greater_or_equal": ">=",
            "<": "<", "less_than": "<",
            "<=": "<=", "less_than_or_equal": "<=", "less_or_equal": "<=",
            "like": "LIKE", "contains": "LIKE",
        }
        for f in contract.filters:
            col_q = quote_col(f.column)
            val = f.value
            op = getattr(f, "operator", "=") or "="
            sql_op = op_map.get(str(op).lower(), "=")
            if isinstance(val, str):
                safe_val = val.replace("'", "''")
                if sql_op == "LIKE":
                    where_conditions.append(f"{col_q} LIKE '%{safe_val}%'")
                else:
                    where_conditions.append(f"{col_q} {sql_op} '{safe_val}'")
            elif val is not None:
                where_conditions.append(f"{col_q} {sql_op} {val}")

        # 0. CALCULATION SPECIFICATION
        if contract.calculation and contract.calculation.operands:
            calc = contract.calculation
            select_items = []
            not_nulls = []

            # Add group dimensions if grouped
            if contract.group_by:
                for g in contract.group_by:
                    g_q = quote_col(g)
                    select_items.append(g_q)
                    not_nulls.append(f"{g_q} IS NOT NULL")

            for i, op in enumerate(calc.operands):
                op_meas = op.measure_column or op.measure
                op_q = quote_col(op_meas) if op_meas else "*"
                op_agg = (op.aggregation or "SUM").upper()
                op_alias = self._sanitize_alias(op.name or f"operand_{i+1}")

                # Check if this operand has specific filters (e.g. Technology category)
                if op.filters:
                    conds = []
                    for f in op.filters:
                        fq = quote_col(f.column)
                        safe_v = str(f.value).replace("'", "''")
                        conds.append(f"{fq} = '{safe_v}'")
                    cond_str = " AND ".join(conds)
                    case_expr = f"CASE WHEN {cond_str} THEN {op_q} ELSE 0 END"
                    select_items.append(f"ROUND(CAST(SUM({case_expr}) AS NUMERIC), 2) AS {op_alias}")
                else:
                    if op_agg == "COUNT_DISTINCT":
                        select_items.append(f"COUNT(DISTINCT {op_q}) AS {op_alias}")
                    elif op_agg == "COUNT":
                        select_items.append(f"COUNT(*) AS {op_alias}")
                    elif op_agg == "AVG":
                        select_items.append(f"ROUND(CAST(AVG({op_q}) AS NUMERIC), 2) AS {op_alias}")
                    else:
                        select_items.append(f"ROUND(CAST(SUM({op_q}) AS NUMERIC), 2) AS {op_alias}")
                    if op_meas and op_agg != "COUNT":
                        not_nulls.append(f"{op_q} IS NOT NULL")

            all_where = where_conditions + not_nulls
            where_sql = f" WHERE {' AND '.join(all_where)}" if all_where else ""

            if contract.group_by:
                group_sql = ", ".join([quote_col(g) for g in contract.group_by])
                return f'SELECT {", ".join(select_items)} FROM "{safe_table}"{where_sql} GROUP BY {group_sql} ORDER BY {group_sql} ASC'
            else:
                return f'SELECT {", ".join(select_items)} FROM "{safe_table}"{where_sql}'

        # 1. SCALAR AGGREGATION: GROUPING MUST NEVER BE INVENTED
        if contract.answer_type == "SCALAR":
            agg = (contract.aggregation or "SUM").upper()
            expr = quote_expr(contract.measure_expression or contract.measure_column)

            # Not-null check on measure columns
            not_null_cols = []
            if contract.measure_column:
                # check if multi-column expression
                for part in re.findall(r'[a-zA-Z_]\w*', contract.measure_column):
                    if part.lower() in cols_map:
                        not_null_cols.append(f'"{cols_map[part.lower()]}" IS NOT NULL')

            all_where = where_conditions + not_null_cols
            where_sql = f" WHERE {' AND '.join(all_where)}" if all_where else ""

            alias_name = self._sanitize_alias(contract.measure or "metric")
            if agg == "COUNT_DISTINCT":
                col_name = contract.measure_column or contract.measure or "id"
                col_q = quote_col(col_name)
                return f'SELECT COUNT(DISTINCT {col_q}) AS unique_{alias_name} FROM "{safe_table}"{where_sql}'
            elif agg == "COUNT":
                return f'SELECT COUNT(*) AS total_records FROM "{safe_table}"{where_sql}'
            elif agg == "AVG":
                return f'SELECT ROUND(CAST(AVG({expr}) AS NUMERIC), 2) AS average_{alias_name} FROM "{safe_table}"{where_sql}'
            elif agg == "MIN":
                return f'SELECT ROUND(CAST(MIN({expr}) AS NUMERIC), 2) AS minimum_{alias_name} FROM "{safe_table}"{where_sql}'
            elif agg == "MAX":
                return f'SELECT ROUND(CAST(MAX({expr}) AS NUMERIC), 2) AS maximum_{alias_name} FROM "{safe_table}"{where_sql}'
            else:  # SUM
                return f'SELECT ROUND(CAST(SUM({expr}) AS NUMERIC), 2) AS total_{alias_name} FROM "{safe_table}"{where_sql}'

        # 2. GROUPED AGGREGATION: Grouping strictly by requested group_by columns
        if contract.answer_type == "GROUPED" or contract.group_by:
            group_cols = [quote_col(g) for g in contract.group_by]
            group_sql = ", ".join(group_cols)

            agg = (contract.aggregation or "SUM").upper()
            expr = quote_expr(contract.measure_expression or contract.measure_column)

            not_null_conds = [f"{g} IS NOT NULL" for g in group_cols]
            if contract.measure_column:
                for part in re.findall(r'[a-zA-Z_]\w*', contract.measure_column):
                    if part.lower() in cols_map:
                        not_null_conds.append(f'"{cols_map[part.lower()]}" IS NOT NULL')

            all_where = where_conditions + not_null_conds
            where_sql = f" WHERE {' AND '.join(all_where)}" if all_where else ""

            alias_name = self._sanitize_alias(contract.measure or "metric")
            if agg == "COUNT_DISTINCT":
                col_name = contract.measure_column or contract.measure or "id"
                col_q = quote_col(col_name)
                agg_select = f"COUNT(DISTINCT {col_q}) AS unique_{alias_name}"
                order_col = f"unique_{alias_name}"
            elif agg == "COUNT":
                agg_select = "COUNT(*) AS total_records"
                order_col = "total_records"
            elif agg == "AVG":
                agg_select = f"ROUND(CAST(AVG({expr}) AS NUMERIC), 2) AS average_{alias_name}"
                order_col = f"average_{alias_name}"
            elif agg == "MIN":
                agg_select = f"ROUND(CAST(MIN({expr}) AS NUMERIC), 2) AS minimum_{alias_name}"
                order_col = f"minimum_{alias_name}"
            elif agg == "MAX":
                agg_select = f"ROUND(CAST(MAX({expr}) AS NUMERIC), 2) AS maximum_{alias_name}"
                order_col = f"maximum_{alias_name}"
            else:  # SUM
                agg_select = f"ROUND(CAST(SUM({expr}) AS NUMERIC), 2) AS total_{alias_name}"
                order_col = f"total_{alias_name}"

            if contract.secondary_measure_column:
                sec_expr = quote_expr(contract.secondary_measure_column)
                sec_alias = self._sanitize_alias(contract.secondary_measure or "secondary_metric")
                if any(k in contract.secondary_measure_column.lower() for k in ("id", "code", "no", "number", "name")):
                    agg_select += f', COUNT(DISTINCT {sec_expr}) AS total_{sec_alias}'
                else:
                    agg_select += f', ROUND(CAST(SUM({sec_expr}) AS NUMERIC), 2) AS total_{sec_alias}'

            # Ordering and grouping:
            # 1. Genuine temporal time-series queries
            temporal_col = next((
                g for g in contract.group_by
                if is_temporal_column(g)
            ), None)

            is_line_or_trend = (
                contract.intent in ("TIME_SERIES", "TREND")
                or contract.suggested_chart_type in ("line", "multi_line")
            )

            is_pg = (sync_engine.dialect.name == "postgresql")

            # Build HAVING clause if specified
            having_conditions = []
            if hasattr(contract, "having") and contract.having:
                for h in contract.having:
                    op = getattr(h, "operator", ">") or ">"
                    val = h.value
                    if getattr(h, "expression", None):
                        having_conditions.append(f"{h.expression} {op} {val}")
                    elif getattr(h, "column", None):
                        col_q = quote_col(h.column)
                        h_agg = (getattr(h, "aggregation", None) or "COUNT").upper()
                        if h_agg == "COUNT":
                            having_conditions.append(f"COUNT(*) {op} {val}")
                        elif h_agg == "COUNT_DISTINCT":
                            having_conditions.append(f"COUNT(DISTINCT {col_q}) {op} {val}")
                        elif h_agg in ("AVG", "SUM", "MIN", "MAX"):
                            having_conditions.append(f"{h_agg}({col_q}) {op} {val}")
                    elif getattr(h, "aggregation", None):
                        having_conditions.append(f"COUNT(*) {op} {val}")

            having_sql = f" HAVING {' AND '.join(having_conditions)}" if having_conditions else ""

            if temporal_col:
                grain = (contract.time_granularity or "month").lower()
                if is_pg:
                    if grain == "year":
                        time_sql_expr = f"TO_CHAR(CAST({quote_col(temporal_col)} AS TIMESTAMP), 'YYYY')"
                    elif grain == "quarter":
                        time_sql_expr = f"TO_CHAR(CAST({quote_col(temporal_col)} AS TIMESTAMP), 'YYYY-\"Q\"Q')"
                    elif grain == "week":
                        time_sql_expr = f"TO_CHAR(CAST({quote_col(temporal_col)} AS TIMESTAMP), 'IYYY-\"W\"IW')"
                    elif grain == "day":
                        time_sql_expr = f"TO_CHAR(CAST({quote_col(temporal_col)} AS TIMESTAMP), 'YYYY-MM-DD')"
                    else:  # month default
                        time_sql_expr = f"TO_CHAR(CAST({quote_col(temporal_col)} AS TIMESTAMP), 'YYYY-MM')"
                else:
                    if grain == "year":
                        time_sql_expr = f"COALESCE(strftime('%Y', {quote_col(temporal_col)}), SUBSTR({quote_col(temporal_col)}, 1, 4))"
                    elif grain == "quarter":
                        time_sql_expr = f"printf('%s-Q%d', SUBSTR({quote_col(temporal_col)}, 1, 4), (CAST(SUBSTR({quote_col(temporal_col)}, 6, 2) AS INTEGER) + 2) / 3)"
                    elif grain == "week":
                        time_sql_expr = f"strftime('%Y-W%W', {quote_col(temporal_col)})"
                    elif grain == "day":
                        time_sql_expr = f"COALESCE(strftime('%Y-%m-%d', {quote_col(temporal_col)}), SUBSTR({quote_col(temporal_col)}, 1, 10))"
                    else:  # month default
                        time_sql_expr = f"COALESCE(strftime('%Y-%m', {quote_col(temporal_col)}), SUBSTR({quote_col(temporal_col)}, 1, 7))"

                other_group_cols = [quote_col(g) for g in contract.group_by if g != temporal_col]
                if other_group_cols:
                    group_select_sql = f'{time_sql_expr} AS {quote_col(temporal_col)}, {", ".join(other_group_cols)}'
                    group_clause_sql = f'{time_sql_expr}, {", ".join(other_group_cols)}'
                else:
                    group_select_sql = f'{time_sql_expr} AS {quote_col(temporal_col)}'
                    group_clause_sql = time_sql_expr

                order_col = time_sql_expr
                direction = "ASC"
                if contract.sort:
                    direction = contract.sort.direction.upper()
                limit_sql = f" LIMIT {contract.limit}" if contract.limit else " LIMIT 500"

                return (
                    f'SELECT {group_select_sql}, {agg_select} FROM "{safe_table}"'
                    f'{where_sql} GROUP BY {group_clause_sql}{having_sql} ORDER BY {order_col} {direction}{limit_sql}'
                )
            elif is_line_or_trend:
                # Non-temporal numeric or ordinal line chart (e.g. Age, or Age + Gender)
                x_col = contract.group_by[0]
                other_group_cols = [quote_col(g) for g in contract.group_by[1:]]
                order_expr = quote_col(x_col)
                direction = "ASC"
                if contract.sort:
                    direction = contract.sort.direction.upper()
                order_clause = f"{order_expr} {direction}" + (f", {', '.join(other_group_cols)}" if other_group_cols else "")
                limit_sql = f" LIMIT {contract.limit}" if contract.limit else " LIMIT 500"

                return (
                    f'SELECT {group_sql}, {agg_select} FROM "{safe_table}"'
                    f'{where_sql} GROUP BY {group_sql}{having_sql} ORDER BY {order_clause}{limit_sql}'
                )
            else:
                direction = "DESC"
                if contract.sort:
                    direction = contract.sort.direction.upper()
                limit_sql = f" LIMIT {contract.limit}" if contract.limit else " LIMIT 15"

                if any("COUNT" in hc for hc in having_conditions) and "total_orders" not in agg_select and "total_records" not in agg_select:
                    agg_select += ", COUNT(*) AS total_orders"

                return (
                    f'SELECT {group_sql}, {agg_select} FROM "{safe_table}"'
                    f'{where_sql} GROUP BY {group_sql}{having_sql} ORDER BY {order_col} {direction}{limit_sql}'
                )

        # 3. DISTRIBUTION / HISTOGRAM (Raw observations)
        if contract.answer_type == "DISTRIBUTION":
            target_col = contract.measure_column or (contract.x_axis or "Quantity")
            target_q = quote_col(target_col)
            where_sql = f" WHERE {target_q} IS NOT NULL"
            return f'SELECT {target_q} FROM "{safe_table}"{where_sql} LIMIT 5000'

        # 4. RELATIONSHIP / SCATTER (Raw observations)
        if contract.answer_type == "RELATIONSHIP":
            x_col = contract.x_axis or "Quantity"
            y_col = contract.y_axis or "UnitPrice"
            x_q, y_q = quote_col(x_col), quote_col(y_col)
            where_conds = [f"{x_q} IS NOT NULL", f"{y_q} IS NOT NULL"]
            select_cols = [x_q, y_q]
            if contract.color_by and contract.color_by.lower() in cols_map:
                color_q = quote_col(cols_map[contract.color_by.lower()])
                select_cols.append(color_q)
                where_conds.append(f"{color_q} IS NOT NULL")
            where_sql = f" WHERE {' AND '.join(where_conds)}"
            return f'SELECT {", ".join(select_cols)} FROM "{safe_table}"{where_sql} LIMIT 5000'

        # 5. DATA_RETRIEVAL / RECORDS (Raw preview)
        limit_sql = f" LIMIT {contract.limit or 50}"
        return f'SELECT * FROM "{safe_table}"{limit_sql}'

    def _validate_and_repair_result(
        self,
        df: pd.DataFrame,
        contract: QueryContract,
        schema: dict[str, Any],
        safe_table: str,
        current_sql: str,
    ) -> tuple[pd.DataFrame, str]:
        """
        Validate executed DataFrame against Query Contract.
        If result contains unwanted grouping or incorrect shape, repair deterministically.
        """
        if contract.answer_type == "SCALAR":
            # SCALAR must have exactly 1 row (multiple columns allowed for multi-operand calculations)
            if (len(df) > 1) or (len(df.columns) > 1 and not contract.calculation):
                logger.warning(
                    "Result validation failed for SCALAR: got %d rows, %d columns. Re-executing pure scalar query.",
                    len(df), len(df.columns),
                )
                # Force pure scalar query
                contract.group_by = []
                scalar_sql = self._build_query_from_contract(contract, schema, safe_table)
                repaired_df = pd.read_sql(scalar_sql, con=sync_engine)
                return repaired_df, scalar_sql

        return df, current_sql

    def _build_fallback_query(
        self,
        question: str,
        schema: dict[str, Any],
        safe_table: str,
    ) -> str:
        """Fallback query builder when standard generation encounters issues."""
        q_lower = question.lower()
        cols = schema.get("columns", [])
        names = [str(c.get("name", "")) for c in cols]

        # Check for count
        if re.search(r"\b(?:count|number of rows|how many rows|records)\b", q_lower):
            return f'SELECT COUNT(*) AS total_records FROM "{safe_table}"'

        # Check for revenue
        if "revenue" in q_lower:
            qty_col = next((n for n in names if any(t in n.lower() for t in ("quantity", "qty", "units"))), None)
            price_col = next((n for n in names if any(t in n.lower() for t in ("unitprice", "price", "unit_price"))), None)
            if qty_col and price_col:
                if "by country" in q_lower:
                    country_col = next((n for n in names if "country" in n.lower()), "Country")
                    return (
                        f'SELECT "{country_col}", ROUND(CAST(SUM("{qty_col}" * "{price_col}") AS NUMERIC), 2) AS total_revenue '
                        f'FROM "{safe_table}" WHERE "{country_col}" IS NOT NULL '
                        f'GROUP BY "{country_col}" ORDER BY total_revenue DESC LIMIT 10'
                    )
                return f'SELECT ROUND(CAST(SUM("{qty_col}" * "{price_col}") AS NUMERIC), 2) AS total_revenue FROM "{safe_table}"'

        return f'SELECT * FROM "{safe_table}" LIMIT 50'

    @staticmethod
    def _sanitize_alias(alias: str) -> str:
        """Create valid clean SQL identifier alias."""
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", alias.lower().strip())
        return re.sub(r"_+", "_", clean).strip("_")


# Singleton instance
sql_agent = SQLAgent()
