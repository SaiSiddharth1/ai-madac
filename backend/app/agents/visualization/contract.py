"""
Query Contract and Semantic Column Resolution for AI-MADAC Visualization System.

Enforces:
1. THE CURRENT USER QUERY IS THE PRIMARY SOURCE OF TRUTH.
2. Explicit Field Rule: Explicitly mentioned dataset fields MUST appear in the contract.
3. Deterministic semantic column resolution using aliases, normalization, and schema inspection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd


# Known semantic aliases mapped to common analytical concepts
SEMANTIC_ALIASES: dict[str, list[str]] = {
    "gender": ["gender", "sex", "male", "female", "males", "females", "men", "women"],
    "investment_objective": [
        "investment_objective", "investment objective", "investment objectives",
        "objective", "objectives", "goal", "goals", "investment goal", "investment goals",
        "purpose", "aim", "target objective"
    ],
    "investment_avenue": [
        "investment_avenue", "investment avenue", "investment avenues", "investment_avenues",
        "avenue", "avenues", "investment type", "investment types", "avenue of investment",
        "instrument", "instruments", "preferred avenue"
    ],
    "expected_return": [
        "expected_return", "expected return", "expected returns", "expected_returns",
        "return", "returns", "roi", "expected roi", "expect", "expected rate",
        "rate of return", "return expectations"
    ],
    "duration": [
        "duration", "investment duration", "tenure", "term", "horizon", "time period",
        "holding period", "investment period", "investment horizon"
    ],
    "age": ["age", "investor age", "investor's age", "years old", "age group", "age distribution"],
    "income": [
        "income", "annual income", "annual_income", "salary", "earnings", "revenue",
        "household income", "monthly income"
    ],
    "department": ["department", "dept", "division", "team", "unit"],
    "status": ["status", "state", "condition", "stage"],
    "product": ["product", "item", "sku", "product line", "product category"],
    "region": ["region", "area", "zone", "location", "territory", "geography"],
    "city": ["city", "town", "metro", "municipality"],
    "state": ["state", "province"],
    "country": ["country", "nation"],
    "date": ["date", "year", "month", "day", "quarter", "period", "timestamp", "time"],
    "sales": ["sales", "revenue", "turnover", "total sales", "gross sales", "amount"],
}


@dataclass
class QueryContract:
    """
    Structured query contract created before chart selection.
    Preserved throughout the pipeline and used to strictly validate the final specification.
    """
    raw_question: str
    requested_fields: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    measure: dict[str, Any] = field(default_factory=lambda: {"type": "count", "column": None})
    intent: str = "categorical_comparison"
    explicit_chart_type: Optional[str] = None
    filters: list[dict[str, Any]] = field(default_factory=list)
    sorting: dict[str, Any] = field(default_factory=lambda: {"field": None, "direction": None})
    limit: Optional[int] = None
    binning: Optional[dict[str, Any]] = None
    is_follow_up: bool = False
    visualization_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_question": self.raw_question,
            "visualization_required": self.visualization_required,
            "requested_fields": self.requested_fields,
            "dimensions": self.dimensions,
            "measure": self.measure,
            "intent": self.intent,
            "explicit_chart_type": self.explicit_chart_type,
            "filters": self.filters,
            "sorting": self.sorting,
            "limit": self.limit,
            "binning": self.binning,
            "is_follow_up": self.is_follow_up,
        }


def is_visualization_requested(question: str, intent: str = "") -> bool:
    """Return True if the query explicitly or implicitly requires a visual chart."""
    q = question.lower().strip()
    if intent in {"RELATIONSHIP_ANALYSIS", "VISUALIZATION"}:
        return True

    visual_terms = (
        "visual", "visually", "visualize", "visualization", "visualizing",
        "chart", "charts", "plot", "plots", "plotting", "graph", "graphs",
        "graphical", "graphically", "diagram",
        "bar chart", "pie chart", "line chart", "scatter plot", "scatter",
        "histogram", "box plot", "donut chart", "donut", "heatmap", "area chart",
        "clustered column", "clustered bar", "stacked column", "stacked bar",
        "distribution", "trend", "trends"
    )
    if any(re.search(r"\b" + re.escape(t) + r"\b", q) for t in visual_terms):
        return True

    if re.search(r"\b(?:versus|vs\.?)\b", q):
        return True

    if any(phrase in q for phrase in ("by avenue and objective", "by gender and objective", "compared to", "breakdown by")):
        return True

    return False


def normalize_term(term: str) -> str:
    """Normalize a column name or query phrase for robust semantic matching."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", term.lower())
    return " ".join(cleaned.split())


def resolve_column_semantically(term: str, available_columns: list[str]) -> Optional[str]:
    """
    Resolve a user term to an actual dataset column using strict priority:
    1. Exact column match (case-insensitive)
    2. Normalized column match (spaces, underscores, hyphens)
    3. Singular / plural variation match
    4. Known semantic alias mapping
    5. Return None if no confident match exists (NEVER guess or invent).
    """
    t_norm = normalize_term(term)

    # 1. Exact match
    for col in available_columns:
        if col.lower() == term.lower() or normalize_term(col) == t_norm:
            return col

    # 2. Singular / plural variations
    singular_t = t_norm[:-1] if t_norm.endswith("s") else t_norm
    plural_t = (t_norm[:-1] + "ies") if t_norm.endswith("y") else (t_norm + "s")
    for col in available_columns:
        col_norm = normalize_term(col)
        if col_norm in (singular_t, plural_t):
            return col

    # 3. Known semantic aliases
    for alias_key, aliases in SEMANTIC_ALIASES.items():
        if t_norm == alias_key or t_norm in aliases:
            # Check if any available column matches alias_key or any alias in the group
            for col in available_columns:
                col_norm = normalize_term(col)
                if col_norm == alias_key or col_norm in aliases:
                    return col

    # 4. Partial exact word matches if column name has multiple words (e.g. "Investment Avenue" vs "Avenue")
    for col in available_columns:
        col_words = normalize_term(col).split()
        if t_norm in col_words:
            return col

    return None


def extract_query_contract(
    question: str,
    schema: dict[str, Any] | list[str],
    previous_result: Optional[dict[str, Any]] = None,
) -> QueryContract:
    """
    Extract a formal QueryContract from the current user query and dataset schema.
    """
    q_raw = question.strip()
    q = f" {q_raw.lower()} "

    # Extract schema column names and types
    available_cols: list[str] = []
    dtypes_map: dict[str, str] = {}
    if isinstance(schema, dict):
        for c in schema.get("columns", []):
            name = str(c.get("name", ""))
            if name:
                available_cols.append(name)
                dtypes_map[name] = str(c.get("dtype", "")).lower()
    elif isinstance(schema, list):
        available_cols = [str(c) for c in schema if c]

    # Check for explicit follow-up reference (e.g. "show it as a bar chart", "show the same as pie")
    anaphoric_tokens = (" it ", " that ", " this ", " the same ", " same result ", " same data ")
    has_anaphoric = any(tok in q for tok in anaphoric_tokens)

    # Identify requested column matches from the query
    matched_cols: list[str] = []
    matched_positions: list[tuple[int, str]] = []

    # Score each column against the query
    for col in available_cols:
        col_norm = normalize_term(col)
        col_singular = col_norm[:-1] if col_norm.endswith("s") else col_norm
        col_plural = (col_norm[:-1] + "ies") if col_norm.endswith("y") else (col_norm + "s")

        # Check exact phrases
        patterns = [
            r"\b" + re.escape(col_norm) + r"\b",
            r"\b" + re.escape(col_singular) + r"\b",
            r"\b" + re.escape(col_plural) + r"\b",
        ]
        # Also check semantic aliases
        for alias_key, aliases in SEMANTIC_ALIASES.items():
            if col_norm == alias_key or col_norm in aliases:
                for alias in aliases:
                    patterns.append(r"\b" + re.escape(alias) + r"\b")

        pos = -1
        for pat in patterns:
            m = re.search(pat, q)
            if m:
                pos = m.start()
                break

        if pos != -1 and col not in matched_cols:
            matched_positions.append((pos, col))

    # Sort matched columns by order of appearance in the user query
    matched_positions.sort(key=lambda x: x[0])
    matched_cols = [col for _, col in matched_positions]

    # Special handling for "gender" if "male" or "female" or "gender" appears in question
    gender_col = next((c for c in available_cols if c.lower() == "gender" or "gender" in c.lower()), None)
    if gender_col and gender_col not in matched_cols:
        if any(term in q for term in (" male ", " female ", " men ", " women ", " gender ", " males ", " females ")):
            matched_cols.append(gender_col)

    # Special handling for "objective" if "objective" or "goal" in question
    obj_col = next((c for c in available_cols if "objective" in c.lower() or "goal" in c.lower()), None)
    if obj_col and obj_col not in matched_cols:
        if any(term in q for term in (" objective ", " objectives ", " goal ", " goals ")):
            matched_cols.append(obj_col)

    # Special handling for "avenue" if "avenue" in question
    ave_col = next((c for c in available_cols if c.lower() == "avenue" or ("avenue" in c.lower() and "investment" not in c.lower())), None)
    if not ave_col:
        ave_col = next((c for c in available_cols if "avenue" in c.lower()), None)
    if ave_col and ave_col not in matched_cols:
        if any(term in q for term in (" avenue ", " avenues ", " investment avenue ")):
            matched_cols.append(ave_col)

    # An explicit follow-up is ONLY valid if NO new dataset columns are requested
    is_follow_up = False
    if has_anaphoric and previous_result and len(matched_cols) == 0:
        is_follow_up = True
        # Inherit requested fields from previous result if valid
        prev_cols = previous_result.get("columns_used") or []
        for col in prev_cols:
            if col in available_cols and col not in matched_cols:
                matched_cols.append(col)

    # Detect explicit chart requests
    explicit_chart = None
    if re.search(r"\bpie\b|\bdonut\b", q):
        explicit_chart = "pie"
    elif re.search(r"\b(?:bar chart|bar graph|horizontal bar|grouped bar|stacked bar)\b", q) or re.search(r"\bbar\b", q):
        explicit_chart = "bar"
    elif re.search(r"\bhistogram\b", q):
        explicit_chart = "histogram"
    elif re.search(r"\bscatter\b|\bscatter plot\b", q):
        explicit_chart = "scatter"
    elif re.search(r"\bbox plot\b|\bboxplot\b", q):
        explicit_chart = "box"
    elif re.search(r"\bheatmap\b", q):
        explicit_chart = "heatmap"
    elif re.search(r"\bline chart\b|\bline graph\b|\bline\b", q):
        explicit_chart = "line"
    elif re.search(r"\barea chart\b|\barea\b", q):
        explicit_chart = "area"

    # Identify numerical columns among matched columns
    numeric_dtypes = {"int", "int64", "integer", "float", "float64", "double", "number"}
    numeric_matched = [c for c in matched_cols if dtypes_map.get(c, "") in numeric_dtypes or "age" in c.lower() or "return" in c.lower() or "income" in c.lower() or "salary" in c.lower()]
    categorical_matched = [c for c in matched_cols if c not in numeric_matched]

    # Detect measure requested (count, average, sum, min, max)
    measure_type = "count"
    measure_col = None
    if re.search(r"\b(?:average|mean|avg)\b", q):
        measure_type = "mean"
        measure_col = numeric_matched[0] if numeric_matched else None
    elif re.search(r"\b(?:total|sum)\b", q):
        measure_type = "sum"
        measure_col = numeric_matched[0] if numeric_matched else None
    elif re.search(r"\b(?:highest|maximum|max)\b", q) and not re.search(r"\b(?:most popular|highest count)\b", q):
        measure_type = "max"
        measure_col = numeric_matched[0] if numeric_matched else None
    elif re.search(r"\b(?:lowest|minimum|min)\b", q) and not re.search(r"\b(?:least popular|lowest count)\b", q):
        measure_type = "min"
        measure_col = numeric_matched[0] if numeric_matched else None

    # Detect Intent
    intent = "categorical_comparison"
    if explicit_chart == "histogram" or re.search(r"\b(?:distribution|histogram|spread)\b", q):
        intent = "distribution"
    elif explicit_chart == "scatter" or (len(numeric_matched) >= 2 and any(tok in q for tok in ("versus", "vs", "relationship", "correlation", "scatter", "against", "and"))):
        intent = "relationship"
        if not explicit_chart:
            explicit_chart = "scatter"
    elif len(categorical_matched) >= 2 and any(tok in q for tok in ("versus", "vs", "by", "across", "and")):
        intent = "cross_tabulation"
    elif explicit_chart == "pie" or re.search(r"\b(?:proportion|percentage|share|ratio)\b", q):
        intent = "proportion"
    elif re.search(r"\b(?:trend|over time|growth|over the years|monthly|yearly)\b", q):
        intent = "trend"
    elif re.search(r"\b(?:rank|ranking|top|bottom|most popular|least popular)\b", q):
        intent = "ranking"
    elif len(matched_cols) == 0 and not is_follow_up:
        intent = "single_metric"

    # Detect Filters
    filters: list[dict[str, Any]] = []
    # 1. Age filters: "above 40", "> 40", "older than 40"
    age_filter_match = re.search(r"\b(?:above|older than|greater than|>)\s*(\d+)", q)
    if age_filter_match:
        age_col = next((c for c in available_cols if c.lower() == "age" or "age" in c.lower()), "age")
        filters.append({"column": age_col, "op": ">", "value": int(age_filter_match.group(1))})
    age_under_match = re.search(r"\b(?:under|younger than|less than|<)\s*(\d+)", q)
    if age_under_match:
        age_col = next((c for c in available_cols if c.lower() == "age" or "age" in c.lower()), "age")
        filters.append({"column": age_col, "op": "<", "value": int(age_under_match.group(1))})

    # 2. Return percentage range: "between 30% and 40%", "30-40%"
    return_range_match = re.search(r"(\d+)\s*(?:%|-|to|and)\s*(\d+)\s*%", q)
    if return_range_match:
        ret_col = next((c for c in available_cols if "return" in c.lower() or "expect" in c.lower()), "expected_return")
        low, high = int(return_range_match.group(1)), int(return_range_match.group(2))
        filters.append({"column": ret_col, "op": "between", "value": (low, high)})

    # 3. Categorical filters: female only, male only
    if re.search(r"\b(?:female|women)\b", q) and not re.search(r"\b(?:gender|male and female|both)\b", q):
        if gender_col and gender_col not in matched_cols:
            filters.append({"column": gender_col, "op": "==", "value": "Female"})
    elif re.search(r"\bmale\b|\bmen\b", q) and not re.search(r"\b(?:gender|male and female|both)\b", q):
        if gender_col and gender_col not in matched_cols:
            filters.append({"column": gender_col, "op": "==", "value": "Male"})

    # Detect Sorting & Limits
    sorting: dict[str, Any] = {"field": None, "direction": None}
    limit: Optional[int] = None
    if re.search(r"\b(?:top|highest|most|best)\b", q):
        sorting = {"field": "count" if measure_type == "count" else measure_col, "direction": "desc"}
    elif re.search(r"\b(?:bottom|lowest|least|worst)\b", q):
        sorting = {"field": "count" if measure_type == "count" else measure_col, "direction": "asc"}

    limit_match = re.search(r"\b(?:top|bottom|first|last|highest|lowest)\s+(\d+)\b", q)
    if limit_match:
        limit = int(limit_match.group(1))

    # Detect Binning (e.g., "bins of 5 years", "bin size 5")
    binning: Optional[dict[str, Any]] = None
    bin_match = re.search(r"\b(?:bins?\s*(?:of|size)?\s*(\d+)|(\d+)\s*(?:years?|year)?\s*bins?)\b", q)
    if bin_match:
        size = int(bin_match.group(1) or bin_match.group(2))
        binning = {"bin_size": size}

    vis_required = is_visualization_requested(q_raw, intent) or bool(explicit_chart) or intent in ("cross_tabulation", "distribution", "relationship")

    return QueryContract(
        raw_question=q_raw,
        requested_fields=matched_cols,
        dimensions=categorical_matched if categorical_matched else ([] if intent in ("relationship", "distribution") else matched_cols),
        measure={"type": measure_type, "column": measure_col},
        intent=intent,
        explicit_chart_type=explicit_chart,
        filters=filters,
        sorting=sorting,
        limit=limit,
        binning=binning,
        is_follow_up=is_follow_up,
        visualization_required=vis_required,
    )
