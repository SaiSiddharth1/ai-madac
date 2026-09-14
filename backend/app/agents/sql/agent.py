"""
SQL Agent.
Generates safe SELECT queries from natural language questions and executes them against PostgreSQL.
"""

import json
import logging
import re
from typing import Any

import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy import text

from app.agents.state import AgentState
from app.core.llm_provider import get_llm
from app.db.database import sync_engine
from app.utils.sql_validator import validate_sql, sanitize_table_name

logger = logging.getLogger(__name__)

SQL_AGENT_PROMPT = """You are an expert SQL Data Analyst Agent.
Your task is to write a PostgreSQL SELECT query to answer the user's question based on the provided table schema.

CRITICAL RULES:
1. ONLY generate SELECT statements. Never write INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, or CREATE.
2. Use the EXACT table name provided in quotes: "{table_name}".
3. Use the EXACT column names provided in quotes or as lowercase identifiers.
4. Keep queries efficient. Add LIMIT 1000 unless performing an aggregation (COUNT, SUM, AVG, GROUP BY).
5. Determine the metric before writing SQL: total/funding/commitment/amount = SUM; average/mean = AVG; highest/lowest = MAX/MIN; count/how many = COUNT.
6. Never answer a requested financial total with COUNT(*). For a top-N category total, GROUP BY the category, SUM the relevant numeric column, ORDER BY that SUM, then LIMIT N.
7. Apply filters before aggregation. Exclude NULL grouping values and NULL numeric measures from grouped calculations unless the user asks otherwise.
8. Use descriptive aliases such as total_commitment, average_commitment, maximum_commitment, minimum_commitment, and project_count.
9. If the question asks about impact, relationship, effect, or comparison (e.g. "Did events affect stock impact?"), write an AGGREGATED query using GROUP BY or conditional averages/counts (e.g. grouping by whether event is NULL or event categories) to compute exact comparison metrics.
10. Output ONLY the raw SQL query. Do NOT include markdown code blocks (```sql ... ```) or explanation.

Schema details for table "{table_name}":
{schema_description}
"""


class SQLAgent:
    """
    SQL Agent responsible for NL-to-SQL translation, security validation, and query execution.
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
        Generate SQL query, validate, execute against database, and update state.
        """
        table_name = state.get("table_name", "")
        question = state.get("question", "")
        schema = state.get("schema", {})

        safe_table = sanitize_table_name(table_name)
        if not safe_table:
            return {
                "errors": ["Invalid table name provided for SQL execution."],
                "sql_result": None,
            }

        unsupported_metric = self._unsupported_metric_message(question, schema)
        if unsupported_metric:
            return {
                "sql_result": {
                    "query": None,
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "displayed_row_count": 0,
                    "message": unsupported_metric,
                },
                "agents_used": state.get("agents_used", []) + ["sql_agent"],
            }

        logger.info(f"SQL Agent generating query for table '{safe_table}', question: '{question}'")

        # Format column details
        columns_info = []
        for col in schema.get("columns", []):
            columns_info.append(f"  - {col.get('name')} ({col.get('dtype')})")
        schema_desc = "\n".join(columns_info)

        # Metric questions are handled by deterministic rules. This protects
        # against returning a row count for a requested financial total.
        sql_query = ""
        if self._requires_metric_guardrail(question):
            sql_query = self._build_heuristic_query(question, schema, safe_table)
        else:
            try:
                sys_msg = SQL_AGENT_PROMPT.format(
                    table_name=safe_table,
                    schema_description=schema_desc,
                )
                response = self.llm.invoke([
                    SystemMessage(content=sys_msg),
                    HumanMessage(content=question),
                ])
                raw_sql = response.content.strip()
                raw_sql = re.sub(r"^```(sql)?\s*", "", raw_sql, flags=re.IGNORECASE)
                raw_sql = re.sub(r"\s*```$", "", raw_sql)
                sql_query = raw_sql.strip()
            except Exception as e:
                logger.warning(f"LLM SQL generation failed: {e}. Using schema-aware query rules.")
                sql_query = self._build_heuristic_query(question, schema, safe_table)

        # 2. Validate SQL
        is_valid, error_msg = validate_sql(sql_query)
        if not is_valid:
            logger.error(f"SQL validation failed for '{sql_query}': {error_msg}")
            sql_query = self._build_heuristic_query(question, schema, safe_table)

        # 3. Execute Query
        logger.info(f"Executing SQL: {sql_query}")
        try:
            df = pd.read_sql(sql_query, con=sync_engine)
            df_truncated = df.head(1000)

            columns = df_truncated.columns.tolist()
            rows = df_truncated.fillna("").to_dict(orient="records")

            result_data = {
                "query": sql_query,
                "columns": columns,
                "rows": rows,
                "row_count": len(df),
                "displayed_row_count": len(rows),
            }

            agents_used = state.get("agents_used", []) + ["sql_agent"]

            return {
                "sql_query": sql_query,
                "sql_result": result_data,
                "agents_used": agents_used,
            }

        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return {
                "sql_query": sql_query,
                "sql_result": None,
                "errors": state.get("errors", []) + [f"SQL Execution Error: {str(e)}"],
            }

    @staticmethod
    def _build_heuristic_query(question: str, schema: dict, table_name: str) -> str:
        """Create a useful, safe query when an LLM provider is unavailable.

        This deliberately covers the common questions users ask immediately
        after uploading a CSV: unique counts, totals/averages, and a measure
        grouped by a category.  It is preferable to returning the same first
        50 rows for every question.
        """
        q = question.lower()
        columns = schema.get("columns", [])
        names = [str(column.get("name", "")) for column in columns]

        def quoted(name: str) -> str:
            return '"' + name.replace('"', '""') + '"'

        def matches_question(name: str) -> bool:
            normalized = name.lower().replace("_", " ")
            plural = name.lower()[:-1] + "ies" if name.lower().endswith("y") else name.lower() + "s"
            return name.lower() in q or normalized in q or plural in q

        def asks_for_count() -> bool:
            return bool(re.search(r"\b(?:how many|number of|count)\b", q))

        categories = [
            name for name in names
            if any(token in name.lower() for token in ("country", "sector", "status", "type", "category", "source", "region", "project", "year", "date"))
        ]
        numeric = [
            str(column.get("name", "")) for column in columns
            if str(column.get("dtype", "")).lower() in {"int", "int64", "integer", "float", "float64", "double", "number"}
            or any(token in str(column.get("name", "")).lower() for token in ("amount", "commitment", "revenue", "profit", "value", "cost", "price", "capital"))
        ]

        requested_categories = [
            name for name in categories
            if matches_question(name) or any(token in q for token in name.lower().split("_") if len(token) > 2)
        ]
        category = requested_categories[0] if requested_categories else (categories[0] if categories else None)
        requested_numeric = [
            name for name in numeric
            if matches_question(name) or any(token in q for token in name.lower().split("_") if len(token) > 2)
        ]
        measure = requested_numeric[0] if requested_numeric else (numeric[0] if numeric else None)

        # "How many countries/sectors ..." means the number of unique values,
        # not the number of raw records.
        if asks_for_count() and requested_categories:
            selected = requested_categories[0]
            return f"SELECT COUNT(DISTINCT {quoted(selected)}) AS {selected}_count FROM {quoted(table_name)}"

        numeric_metric_requested = any(word in q for word in (
            "total", "sum", "average", "mean", "highest", "lowest", "largest", "smallest",
            "commitment", "revenue", "funding", "amount", "value",
        ))

        # A top-project ranking orders project records by their numeric measure.
        if category and "project" in category.lower() and requested_categories and measure and any(word in q for word in ("top", "bottom")):
            order = "ASC" if "bottom" in q else "DESC"
            top_match = re.search(r"\b(?:top|bottom)\s+(\d+)", q)
            limit = f" LIMIT {top_match.group(1)}" if top_match else " LIMIT 10"
            return f"SELECT {quoted(category)}, {quoted(measure)} FROM {quoted(table_name)} WHERE {quoted(category)} IS NOT NULL AND {quoted(measure)} IS NOT NULL ORDER BY {quoted(measure)} {order}{limit}"

        # Numeric aggregation comes before the visualization branch so a chart
        # of total commitment remains a SUM rather than a project count.
        if requested_categories and category and measure and any(word in q for word in ("by ", "each ", "per ", "highest", "lowest", "top", "bottom", "compare")) and numeric_metric_requested:
            if any(word in q for word in ("average", "mean")):
                function, alias_prefix = "AVG", "average"
            elif any(word in q for word in ("highest", "largest")):
                function, alias_prefix = "MAX", "maximum"
            elif any(word in q for word in ("lowest", "smallest")):
                function, alias_prefix = "MIN", "minimum"
            else:
                function, alias_prefix = "SUM", "total"
            alias = f"{alias_prefix}_{measure}"
            order = "ASC" if any(word in q for word in ("lowest", "smallest", "bottom")) else "DESC"
            top_match = re.search(r"\b(?:top|bottom)\s+(\d+)", q)
            limit = f" LIMIT {top_match.group(1)}" if top_match else (" LIMIT 10" if any(word in q for word in ("top", "bottom")) else "")
            return (
                f"SELECT {quoted(category)}, {function}({quoted(measure)}) AS {quoted(alias)} "
                f"FROM {quoted(table_name)} WHERE {quoted(category)} IS NOT NULL "
                f"AND {quoted(measure)} IS NOT NULL GROUP BY {quoted(category)} "
                f"ORDER BY {quoted(alias)} {order}{limit}"
            )

        # A frequency table is appropriate only when no numeric metric was requested.
        if category and not numeric_metric_requested and any(word in q for word in ("chart", "plot", "graph", "visualize", "distribution")):
            return (
                f"SELECT {quoted(category)}, COUNT(*) AS project_count FROM {quoted(table_name)} "
                f"WHERE {quoted(category)} IS NOT NULL GROUP BY {quoted(category)} ORDER BY project_count DESC"
            )

        if measure and any(word in q for word in ("total", "sum", "average", "mean", "highest", "largest", "lowest", "smallest")):
            if any(word in q for word in ("average", "mean")):
                function, alias = "AVG", "average"
            elif any(word in q for word in ("highest", "largest")):
                function, alias = "MAX", "maximum"
            elif any(word in q for word in ("lowest", "smallest")):
                function, alias = "MIN", "minimum"
            else:
                function, alias = "SUM", "total"
            return f"SELECT {function}({quoted(measure)}) AS {quoted(alias + '_' + measure)} FROM {quoted(table_name)} WHERE {quoted(measure)} IS NOT NULL"

        if category and (asks_for_count() or "breakdown" in q):
            return (
                f"SELECT {quoted(category)}, COUNT(*) AS project_count FROM {quoted(table_name)} "
                f"WHERE {quoted(category)} IS NOT NULL GROUP BY {quoted(category)} ORDER BY project_count DESC"
            )

        return f"SELECT * FROM {quoted(table_name)} LIMIT 50"

    @staticmethod
    def _requires_metric_guardrail(question: str) -> bool:
        """Return true when a wrong aggregation would change the answer's meaning."""
        return bool(re.search(
            r"\b(?:total|sum|average|mean|highest|lowest|largest|smallest|count|how many|number of|commitment|funding|revenue|amount)\b",
            question.lower(),
        ))

    @staticmethod
    def _unsupported_metric_message(question: str, schema: dict) -> str | None:
        """Prevent a made-up financial metric from being answered with another column."""
        q = question.lower()
        column_text = " ".join(str(column.get("name", "")).lower().replace("_", " ") for column in schema.get("columns", []))
        requested = {
            "profit": ("profit",),
            "market capitalization": ("market capitalization", "market cap"),
            "market cap": ("market capitalization", "market cap"),
        }
        for label, terms in requested.items():
            if label in q and not any(term in column_text for term in terms):
                return f"This dataset does not contain a '{label}' column, so it cannot calculate that value."
        return None


# Singleton instance
sql_agent = SQLAgent()
