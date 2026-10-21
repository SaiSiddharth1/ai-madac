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

        # Follow-up filters recompute the original analysis against the table.
        # They never filter an already aggregated display result in memory.
        prior = state.get("previous_validated_result") or {}
        if (state.get("structured_intent") or {}).get("mode") == "FOLLOW_UP_FILTER" and prior.get("source_question"):
            question = f"{prior['source_question']} {question}"

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
        structured_intent = state.get("structured_intent") or self._parse_intent(question, schema)
        if structured_intent.get("mode") == "FOLLOW_UP_FILTER" and prior.get("analysis_plan"):
            structured_intent = {**prior["analysis_plan"], "mode": "FOLLOW_UP_FILTER"}
        self._debug_log(
            "SQL INPUT",
            question=question,
            table=safe_table,
            structured_intent=structured_intent,
        )
        if self._requires_metric_guardrail(question) or structured_intent.get("operation") in {"FREQUENCY", "PERCENTAGE"}:
            sql_query = self._build_heuristic_query(question, schema, safe_table, structured_intent)
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
                sql_query = self._build_heuristic_query(question, schema, safe_table, structured_intent)

        # 2. Validate SQL
        is_valid, error_msg = validate_sql(sql_query)
        if not is_valid:
            logger.error(f"SQL validation failed for '{sql_query}': {error_msg}")
            sql_query = self._build_heuristic_query(question, schema, safe_table, structured_intent)

        # 3. Execute Query
        logger.info("========== AI-MADAC DEBUG ==========\nGENERATED SQL:\n%s", sql_query)
        try:
            df = pd.read_sql(sql_query, con=sync_engine)
            # Do not forward a raw preview to the Report Agent when it cannot
            # satisfy the shared analytical intent.  One deterministic retry
            # protects against an invalid LLM/legacy fallback query.
            if not self._result_answers_intent(df.columns.tolist(), structured_intent):
                retry_query = self._build_heuristic_query(question, schema, safe_table, structured_intent)
                logger.warning(
                    "========== AI-MADAC DEBUG ==========\nRESULT RELEVANCE FAILED\n"
                    "result_columns=%s\nretry_sql=%s", df.columns.tolist(), retry_query,
                )
                if retry_query != sql_query:
                    sql_query = retry_query
                    df = pd.read_sql(sql_query, con=sync_engine)
                if not self._result_answers_intent(df.columns.tolist(), structured_intent):
                    return {
                        "sql_query": sql_query,
                        "sql_result": None,
                        "errors": state.get("errors", []) + [
                            "SQL result columns do not satisfy the requested analytical intent."
                        ],
                    }
            df_truncated = df.head(1000)

            columns = df_truncated.columns.tolist()
            # Keep missing numeric values as JSON null, not an empty string.
            # Mixing empty strings with floats later breaks ordering/ranking.
            serializable_df = df_truncated.astype(object).where(pd.notna(df_truncated), None)
            rows = serializable_df.to_dict(orient="records")

            result_data = {
                "query": sql_query,
                "source_question": question,
                "analysis_plan": structured_intent,
                "columns": columns,
                "rows": rows,
                "row_count": len(df),
                "displayed_row_count": len(rows),
            }
            validated_result = {
                "question_id": state.get("query_id"),
                "dataset_id": state.get("dataset_id"),
                "operation": structured_intent.get("operation"),
                "columns_used": columns,
                "filters": self._infer_filters(question, [str(c.get("name", "")) for c in schema.get("columns", [])]),
                "group_by": [structured_intent.get("target_column")] if structured_intent.get("group_by_needed") else [],
                "result": rows,
                "row_count": len(df),
                "validated": True,
                "query": sql_query,
            }
            self._debug_log(
                "SQL OUTPUT",
                executed_sql=sql_query,
                result_columns=columns,
                result_row_count=len(df),
                result_preview=rows[:10],
            )

            agents_used = state.get("agents_used", []) + ["sql_agent"]

            return {
                "sql_query": sql_query,
                "sql_result": result_data,
                "validated_result": validated_result,
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
    def _result_answers_intent(columns: list[str], structured_intent: dict[str, Any]) -> bool:
        """Reject result shapes that cannot answer a structured aggregation."""
        operation = structured_intent.get("operation")
        target = structured_intent.get("target_column")
        if operation in {"FREQUENCY", "PERCENTAGE"}:
            return bool(target and target in columns and any("count" in c.lower() or "percentage" in c.lower() or c in ("investor_count", "count") for c in columns))
        if operation == "AGGREGATE":
            if structured_intent.get("group_by_needed"):
                return bool(target and target in columns and (len(columns) >= 2 or any("count" in c.lower() or "total" in c.lower() for c in columns)))
            return (
                "investor_count" in columns
                or "total_rows" in columns
                or any("count" in c.lower() or "total" in c.lower() for c in columns)
                or bool(target and target in columns)
                or len(columns) == 1
            )
        return True

    @staticmethod
    def _debug_log(stage: str, **payload: Any) -> None:
        """Emit query trace data without credentials or connection settings."""
        logger.info(
            "========== AI-MADAC DEBUG ==========\n%s:\n%s\n====================================",
            stage,
            json.dumps(payload, default=str),
        )

    @staticmethod
    def _build_heuristic_query(question: str, schema: dict, table_name: str, structured_intent: dict | None = None) -> str:
        """Create a useful, safe query when an LLM provider is unavailable.

        This deliberately covers the common questions users ask immediately
        after uploading a CSV: unique counts, totals/averages, and a measure
        grouped by a category.  It is preferable to returning the same first
        50 rows for every question.
        """
        q = question.lower()
        columns = schema.get("columns", [])
        names = [str(column.get("name", "")) for column in columns]

        structured_intent = structured_intent or SQLAgent._parse_intent(question, schema)

        def quoted(name: str) -> str:
            return '"' + name.replace('"', '""') + '"'

        def matches_question(name: str) -> bool:
            normalized = name.lower().replace("_", " ")
            plural = name.lower()[:-1] + "ies" if name.lower().endswith("y") else name.lower() + "s"
            return name.lower() in q or normalized in q or plural in q

        def asks_for_count() -> bool:
            return bool(re.search(r"\b(?:how many|number of|count)\b", q))

        # Frequency, percentage and filtered-count questions are deterministic
        # because selecting the wrong dimension changes their meaning entirely.
        target = structured_intent.get("target_column")
        operation = structured_intent.get("operation")
        filters = SQLAgent._infer_filters(question, names)
        where_parts = []
        for column, value in filters.items():
            where_parts.append(f"{quoted(column)} = '{value.replace(chr(39), chr(39) * 2)}'")
        where_sql = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        non_null = f"{quoted(target)} IS NOT NULL" if target else ""
        group_where = f" WHERE {non_null}" if non_null else ""
        if where_parts and non_null:
            group_where = f" WHERE {' AND '.join(where_parts + [non_null])}"

        if operation == "PERCENTAGE" and target:
            return (
                f"SELECT {quoted(target)}, COUNT(*) AS investor_count, "
                f"ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage "
                f"FROM {quoted(table_name)}{group_where} GROUP BY {quoted(target)} "
                f"ORDER BY investor_count DESC"
            )
        # A multi-dimensional question must preserve every requested
        # dimension. This is a deterministic cross-tab rather than a raw
        # preview or a single-column popularity result.
        avenue = next((name for name in names if name.lower() == "avenue"), None)
        objective = next((name for name in names if name.lower() == "objective"), None)
        expect = next((name for name in names if name.lower() in {"expect", "expected_return", "expected return"}), None)
        range_match = re.search(r"(\d+)\s*[-–—]\s*(\d+)\s*%", question)
        asks_avenue = "avenue" in q
        asks_objective = "objective" in q
        if operation == "FREQUENCY" and asks_avenue and asks_objective and avenue and objective:
            clauses = [f"{quoted(avenue)} IS NOT NULL", f"{quoted(objective)} IS NOT NULL"]
            if expect and range_match:
                # Categorical percentage bands are matched as values. A
                # numeric expected-return column still gets an inclusive range.
                low, high = range_match.groups()
                expect_dtype = next((str(c.get("dtype", "")).lower() for c in columns if c.get("name") == expect), "")
                if expect_dtype in {"int", "int64", "float", "float64", "integer", "number"}:
                    clauses.append(f"{quoted(expect)} BETWEEN {low} AND {high}")
                else:
                    clauses.append(f"{quoted(expect)} = '{low}-{high}%'")
            return (
                f"SELECT {quoted(avenue)}, {quoted(objective)}, COUNT(*) AS investor_count "
                f"FROM {quoted(table_name)} WHERE {' AND '.join(clauses)} "
                f"GROUP BY {quoted(avenue)}, {quoted(objective)} ORDER BY investor_count DESC"
            )
        asks_visual = bool(re.search(r"\b(?:chart|plot|graph|visualize|pie|bar|bargraph|distribution|histogram)\b", q))
        numeric_metric_requested = any(word in q for word in (
            "total", "sum", "average", "mean", "highest", "lowest", "largest", "smallest",
            "commitment", "revenue", "funding", "amount", "value", "price", "sales", "cost"
        ))
        is_numeric_aggregation = (
            numeric_metric_requested
            or (structured_intent and structured_intent.get("aggregation") in {"SUM", "AVG"})
        )
        if asks_visual and target and not is_numeric_aggregation:
            limit = " LIMIT 1" if any(phrase in q for phrase in ("most popular", "most common", "least popular", "least common")) else ""
            return (
                f"SELECT {quoted(target)}, COUNT(*) AS investor_count FROM {quoted(table_name)}"
                f"{group_where} GROUP BY {quoted(target)} ORDER BY investor_count DESC{limit}"
            )
        if operation == "AGGREGATE" and asks_for_count() and target and not structured_intent.get("group_by_needed") and ("unique" in q or not filters):
            return f"SELECT COUNT(DISTINCT {quoted(target)}) AS {target}_count FROM {quoted(table_name)}{where_sql}"
        if operation == "FREQUENCY" and target:
            limit = " LIMIT 1" if any(phrase in q for phrase in ("most popular", "most common", "least popular", "least common")) else ""
            return (
                f"SELECT {quoted(target)}, COUNT(*) AS investor_count FROM {quoted(table_name)}"
                f"{group_where} GROUP BY {quoted(target)} ORDER BY investor_count DESC{limit}"
            )
        if operation == "RANK":
            measure = next((name for name in names if str(name).lower().replace("_", " ") in q), None)
            # Use the final matching numeric-ish column for phrases such as
            # "transactions by sales amount".
            numeric_matches = [name for name in names if any(token in name.lower() for token in ("amount", "sales", "value", "commitment", "revenue", "price"))]
            measure = numeric_matches[0] if numeric_matches else measure
            if measure:
                order = "ASC" if any(word in q for word in ("bottom", "lowest", "least")) else "DESC"
                limit_match = re.search(r"\b(?:top|bottom)\s+(\d+)", q)
                limit = f" LIMIT {limit_match.group(1)}" if limit_match else " LIMIT 10"
                if "project" in q:
                    project_column = next((name for name in names if "project" in name.lower()), None)
                    if project_column:
                        return (f"SELECT {quoted(project_column)}, {quoted(measure)} FROM {quoted(table_name)} "
                                f"WHERE {quoted(project_column)} IS NOT NULL AND {quoted(measure)} IS NOT NULL "
                                f"ORDER BY {quoted(measure)} {order}{limit}")
                return f"SELECT * FROM {quoted(table_name)} WHERE {quoted(measure)} IS NOT NULL ORDER BY {quoted(measure)} {order}{limit}"
        if operation == "AGGREGATE" and structured_intent.get("aggregation") in {"SUM", "AVG"} and target:
            measure = next((name for name in names if any(token in name.lower() for token in ("amount", "sales", "value", "commitment", "revenue", "price"))), None)
            function = structured_intent["aggregation"]
            alias = ("total" if function == "SUM" else "average") + "_" + (measure or "value")
            if measure:
                limit_match = re.search(r"\b(?:top|bottom)\s+(\d+)", q)
                limit = f" LIMIT {limit_match.group(1)}" if limit_match else ""
                return (f"SELECT {quoted(target)}, {function}({quoted(measure)}) AS {quoted(alias)} FROM {quoted(table_name)} "
                        f"WHERE {quoted(target)} IS NOT NULL AND {quoted(measure)} IS NOT NULL GROUP BY {quoted(target)} ORDER BY {quoted(alias)} DESC{limit}")
        if operation == "AGGREGATE" and asks_for_count() and not target:
            return f"SELECT COUNT(*) AS total_rows FROM {quoted(table_name)}{where_sql}"
        if operation == "AGGREGATE" and asks_for_count() and target and not structured_intent.get("group_by_needed"):
            return f"SELECT COUNT(*) AS investor_count FROM {quoted(table_name)}{where_sql}"
        if operation == "AGGREGATE" and target and structured_intent.get("group_by_needed"):
            return (
                f"SELECT {quoted(target)}, COUNT(*) AS investor_count FROM {quoted(table_name)}"
                f"{group_where} GROUP BY {quoted(target)} ORDER BY investor_count DESC"
            )

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
    def _parse_intent(question: str, schema: dict) -> dict[str, Any]:
        """Local counterpart for direct SQL-agent calls outside the graph."""
        q = question.lower()
        names = [str(c.get("name", "")) for c in schema.get("columns", [])]
        def score(name: str) -> int:
            words = name.lower().replace("_", " ")
            singular = words[:-1] if words.endswith("s") else words
            plural = words[:-1] + "ies" if words.endswith("y") else words + "s"
            if words in q or singular in q or plural in q:
                return 100 + len(words)
            return sum(1 for part in words.split() if part not in {"id", "of"} and part in q)
        candidates = sorted(((score(name), name) for name in names), reverse=True)
        percentage = any(word in q for word in ("percentage", "percent", "proportion", "share"))
        frequency = any(word in q for word in ("most popular", "most common", "for each", "each ", "frequency"))
        count = bool(re.search(r"\b(?:how many|count|number of)\b", q))
        rank = bool(re.search(r"\b(?:top|bottom|highest|lowest)\b", q))
        requested_aggregate = "AVG" if any(word in q for word in ("average", "mean")) else "SUM" if any(word in q for word in ("total", "sum")) else None
        numeric_names = {
            str(c.get("name", "")) for c in schema.get("columns", [])
            if str(c.get("dtype", "")).lower() in {"int", "int64", "integer", "float", "float64", "double", "number"}
        }
        dimension_candidates = [(value, name) for value, name in candidates if name not in numeric_names]
        target = (dimension_candidates[0][1] if requested_aggregate and dimension_candidates and dimension_candidates[0][0]
                  else candidates[0][1] if candidates and candidates[0][0] else None)
        avenue_column = next((name for name in names if name.lower() == "avenue"), None)
        if "avenue" in q and avenue_column:
            target = avenue_column
        entity_rank = bool(re.search(r"\b(?:transaction|record|project)s?\b", q))
        grouped = bool(target and (frequency or percentage or requested_aggregate or (rank and not entity_rank) or "each" in q))
        return {
            "operation": "PERCENTAGE" if percentage else "FREQUENCY" if frequency else "RANK" if rank and not requested_aggregate else "AGGREGATE" if count or requested_aggregate else "RETRIEVE",
            "target_column": target,
            "group_by_needed": grouped,
            "aggregation": requested_aggregate or ("COUNT" if count else None),
        }

    @staticmethod
    def _infer_filters(question: str, column_names: list[str]) -> dict[str, str]:
        """Recognize explicitly named categorical values for safe count filters."""
        q = question.lower()
        aliases = {name.lower().replace("_", " "): name for name in column_names}
        filters: dict[str, str] = {}
        has_male = bool(re.search(r"\bmales?\b", q))
        has_female = bool(re.search(r"\bfemales?\b", q))
        if has_female and has_male:
            pass
        elif has_female and "gender" in aliases:
            filters[aliases["gender"]] = "Female"
        elif has_male and "gender" in aliases:
            filters[aliases["gender"]] = "Male"
        objective = aliases.get("objective")
        if objective:
            for value in ("Wealth Creation", "Capital Preservation", "Growth", "Income"):
                if value.lower() in q:
                    filters[objective] = value
                    break
        expect = next((name for name in column_names if name.lower() in {"expect", "expected_return"}), None)
        range_match = re.search(r"(\d+)\s*[-–—]\s*(\d+)\s*%", question)
        if expect and range_match:
            filters[expect] = f"{range_match.group(1)}-{range_match.group(2)}%"
        return filters

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
