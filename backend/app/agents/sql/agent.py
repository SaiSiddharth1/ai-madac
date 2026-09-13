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
5. If the question asks about impact, relationship, effect, or comparison (e.g. "Did events affect stock impact?"), write an AGGREGATED query using GROUP BY or conditional averages/counts (e.g. grouping by whether event is NULL or event categories) to compute exact comparison metrics.
6. Output ONLY the raw SQL query. Do NOT include markdown code blocks (```sql ... ```) or explanation.

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

        # 1. Generate SQL via LLM
        sql_query = ""
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

            # Clean markdown code blocks if present
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
            if any(token in name.lower() for token in ("country", "sector", "status", "type", "category", "source"))
        ]
        numeric = [
            str(column.get("name", "")) for column in columns
            if str(column.get("dtype", "")).lower() in {"int", "int64", "integer", "float", "float64", "double", "number"}
            or any(token in str(column.get("name", "")).lower() for token in ("amount", "commitment", "revenue", "profit", "value", "cost", "price", "capital"))
        ]

        requested_categories = [name for name in categories if matches_question(name)]
        category = requested_categories[0] if requested_categories else (categories[0] if categories else None)
        requested_numeric = [name for name in numeric if matches_question(name)]
        measure = requested_numeric[0] if requested_numeric else (numeric[0] if numeric else None)

        # "How many countries/sectors ..." means the number of unique values,
        # not the number of raw records.
        if asks_for_count() and requested_categories:
            selected = requested_categories[0]
            return f"SELECT COUNT(DISTINCT {quoted(selected)}) AS {selected}_count FROM {quoted(table_name)}"

        # For an explicitly requested categorical chart, a frequency table is
        # the appropriate source data (rather than an arbitrary raw preview).
        if category and any(word in q for word in ("chart", "plot", "graph", "visualize", "distribution")):
            return (
                f"SELECT {quoted(category)}, COUNT(*) AS record_count FROM {quoted(table_name)} "
                f"GROUP BY {quoted(category)} ORDER BY record_count DESC"
            )

        # Aggregated comparisons, e.g. total commitment by country/sector.
        if category and measure and any(word in q for word in ("by ", "each ", "per ", "highest", "lowest", "top")) and any(
            word in q for word in ("total", "sum", "average", "mean", "highest", "lowest", "commitment", "revenue", "amount", "value")
        ):
            function = "AVG" if any(word in q for word in ("average", "mean")) else "SUM"
            alias = f"{function.lower()}_{measure}"
            order = "ASC" if any(word in q for word in ("lowest", "smallest")) else "DESC"
            limit = " LIMIT 10" if any(word in q for word in ("highest", "lowest", "top")) else ""
            return (
                f"SELECT {quoted(category)}, {function}({quoted(measure)}) AS {quoted(alias)} "
                f"FROM {quoted(table_name)} GROUP BY {quoted(category)} "
                f"ORDER BY {quoted(alias)} {order}{limit}"
            )

        if measure and any(word in q for word in ("total", "sum", "average", "mean")):
            function = "AVG" if any(word in q for word in ("average", "mean")) else "SUM"
            return f"SELECT {function}({quoted(measure)}) AS {quoted(function.lower() + '_' + measure)} FROM {quoted(table_name)}"

        if category and (asks_for_count() or "breakdown" in q):
            return (
                f"SELECT {quoted(category)}, COUNT(*) AS record_count FROM {quoted(table_name)} "
                f"GROUP BY {quoted(category)} ORDER BY record_count DESC"
            )

        return f"SELECT * FROM {quoted(table_name)} LIMIT 50"

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
