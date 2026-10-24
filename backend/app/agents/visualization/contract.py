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
        "household income", "monthly income", "monthly_income", "monthly salary", "pay", "compensation"
    ],
    "job_role": [
        "job_role", "job role", "job roles", "role", "roles", "position", "positions",
        "title", "job title", "designation", "profession", "occupation"
    ],
    "job_level": ["job_level", "job level", "level", "seniority", "grade", "rank"],
    "department": ["department", "dept", "division", "team", "unit", "business unit"],
    "status": ["status", "state", "condition", "stage"],
    "product": ["product", "item", "sku", "product line", "product category", "product name"],
    "category": ["category", "categories", "segment", "type", "classification", "group"],
    "region": ["region", "area", "zone", "location", "territory", "geography"],
    "city": ["city", "town", "metro", "municipality"],
    "state": ["state", "province"],
    "country": ["country", "nation"],
    "customer": ["customer", "customers", "client", "clients", "buyer", "buyers", "customer id", "customer name"],
    "order": ["order", "orders", "transaction", "transactions", "invoice", "invoices", "order id"],
    "sales": ["sales", "revenue", "turnover", "total sales", "gross sales", "amount"],
    "profit": ["profit", "earnings", "net profit", "margin", "gain"],
    "quantity": ["quantity", "units", "units sold", "volume", "qty", "count"],
    "discount": ["discount", "markdown", "rebate", "concession"],
    "date": ["date", "year", "month", "day", "quarter", "period", "timestamp", "time", "order date", "ship date"],
}


RANGE_REGEX = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:%|-|–|—|to)\s*(\d+(?:\.\d+)?)\s*%?\s*$", re.IGNORECASE)


class DataTypeClassifier:
    """Classifies dataset columns into precise semantic data types."""

    @classmethod
    def classify(cls, series: Optional[pd.Series], col_name: str = "") -> str:
        """
        Classifies series into:
        - 'numerical': continuous or discrete numbers
        - 'range_category': discrete brackets like '10-20%', '20-30%'
        - 'categorical': nominal/ordinal text categories
        - 'boolean': boolean True/False or 0/1
        - 'datetime': temporal dates/timestamps
        - 'identifier': unique IDs/keys
        """
        if series is None or len(series) == 0:
            name_lower = col_name.lower()
            if any(k in name_lower for k in ("id", "uuid", "key", "code")):
                return "identifier"
            if any(k in name_lower for k in ("date", "time", "year", "month", "day", "quarter")):
                return "datetime"
            if any(k in name_lower for k in ("age", "salary", "income", "amount", "price", "count", "sales")):
                return "numerical"
            if any(k in name_lower for k in ("return", "expect", "range", "bracket")):
                return "range_category"
            return "categorical"

        # Check boolean
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        non_null = series.dropna()
        if len(non_null) == 0:
            return "categorical"

        # Check datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        # Check numerical dtype
        if pd.api.types.is_numeric_dtype(series):
            # Check if identifier
            if series.nunique() == len(series) and any(k in col_name.lower() for k in ("id", "key", "code")):
                return "identifier"
            return "numerical"

        # Check range category (e.g. '10-20%', '20-30%', '30-40%')
        str_vals = non_null.astype(str).str.strip()
        range_matches = str_vals.apply(lambda v: bool(RANGE_REGEX.match(v)))
        if range_matches.mean() >= 0.6:
            return "range_category"

        # Check if can convert to float
        numeric_series = pd.to_numeric(str_vals, errors="coerce")
        if numeric_series.notna().mean() >= 0.85:
            return "numerical"

        # Check identifier
        if series.nunique() == len(series) and (
            any(k in col_name.lower() for k in ("id", "key", "code", "ssn", "uuid")) or len(series) > 100
        ):
            return "identifier"

        return "categorical"


@dataclass
class QueryContract:
    """
    Structured query contract created before chart selection.
    Source of truth for validation, chart selection, data calculation, and rendering.
    """
    raw_question: str
    visualization_required: bool = False
    intent: str = "categorical_comparison"
    requested_dimensions: list[str] = field(default_factory=list)
    requested_measures: list[dict[str, Any] | str] = field(default_factory=lambda: ["count"])
    requested_fields: list[str] = field(default_factory=list)
    filters: list[dict[str, Any]] = field(default_factory=list)
    bin: Optional[dict[str, Any]] = None
    requested_chart: Optional[str] = None
    sorting: dict[str, Any] = field(default_factory=lambda: {"field": None, "direction": None})
    limit: Optional[int] = None
    time_granularity: Optional[str] = None
    series_dimension: Optional[str] = None
    secondary_measure: Optional[str] = None
    color_by: Optional[str] = None
    is_follow_up: bool = False

    # Backward compatibility properties
    @property
    def dimensions(self) -> list[str]:
        return self.requested_dimensions

    @dimensions.setter
    def dimensions(self, val: list[str]):
        self.requested_dimensions = val

    @property
    def measure(self) -> dict[str, Any]:
        if self.requested_measures and isinstance(self.requested_measures[0], dict):
            return self.requested_measures[0]
        return {"type": "count", "column": None}

    @measure.setter
    def measure(self, val: dict[str, Any]):
        self.requested_measures = [val]

    @property
    def binning(self) -> Optional[dict[str, Any]]:
        return self.bin

    @binning.setter
    def binning(self, val: Optional[dict[str, Any]]):
        self.bin = val

    @property
    def explicit_chart_type(self) -> Optional[str]:
        return self.requested_chart

    @explicit_chart_type.setter
    def explicit_chart_type(self, val: Optional[str]):
        self.requested_chart = val

    @property
    def suggested_chart_type(self) -> Optional[str]:
        return self.requested_chart

    @suggested_chart_type.setter
    def suggested_chart_type(self, val: Optional[str]):
        self.requested_chart = val

    @property
    def x_axis(self) -> Optional[str]:
        return self.requested_dimensions[0] if self.requested_dimensions else None

    @x_axis.setter
    def x_axis(self, val: Optional[str]):
        if val:
            if val not in self.requested_dimensions:
                self.requested_dimensions.insert(0, val)

    @property
    def y_axis(self) -> Optional[str]:
        meas = self.measure
        return meas.get("column") if isinstance(meas, dict) else None

    @y_axis.setter
    def y_axis(self, val: Optional[str]):
        if val:
            self.requested_measures = [{"type": "avg", "column": val}]

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_question": self.raw_question,
            "visualization_required": self.visualization_required,
            "intent": self.intent,
            "requested_dimensions": self.requested_dimensions,
            "requested_measures": self.requested_measures,
            "requested_fields": self.requested_fields,
            "filters": self.filters,
            "bin": self.bin,
            "binning": self.bin,
            "requested_chart": self.requested_chart,
            "explicit_chart_type": self.requested_chart,
            "sorting": self.sorting,
            "limit": self.limit,
            "time_granularity": self.time_granularity,
            "series_dimension": self.series_dimension,
            "secondary_measure": self.secondary_measure,
            "is_follow_up": self.is_follow_up,
            # Legacy field aliases
            "dimensions": self.requested_dimensions,
            "measure": self.measure,
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
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", term)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", s.lower())
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
    candidates = {t_norm}
    if t_norm.endswith("ies"):
        candidates.add(t_norm[:-3] + "y")
    elif t_norm.endswith("es") and any(t_norm.endswith(x) for x in ("shes", "ches", "xes", "sses", "zes")):
        candidates.add(t_norm[:-2])
    elif t_norm.endswith("s") and not t_norm.endswith("ss"):
        candidates.add(t_norm[:-1])

    if t_norm.endswith("y"):
        candidates.add(t_norm[:-1] + "ies")
    elif not t_norm.endswith("s"):
        candidates.add(t_norm + "s")
        candidates.add(t_norm + "es")

    for col in available_columns:
        col_norm = normalize_term(col)
        if col_norm in candidates:
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

    # Identify requested column matches with token spans to avoid duplicate column collisions
    candidate_spans: list[tuple[int, int, str]] = []

    for col in available_cols:
        col_norm = normalize_term(col)
        col_singular = col_norm[:-1] if col_norm.endswith("s") else col_norm
        col_plural = (col_norm[:-1] + "ies") if col_norm.endswith("y") else (col_norm + "s")

        patterns = [
            r"\b" + re.escape(col_norm) + r"\b",
            r"\b" + re.escape(col_singular) + r"\b",
            r"\b" + re.escape(col_plural) + r"\b",
        ]
        for alias_key, aliases in SEMANTIC_ALIASES.items():
            if col_norm == alias_key or col_norm in aliases:
                for alias in aliases:
                    patterns.append(r"\b" + re.escape(alias) + r"\b")

        for pat in patterns:
            for m in re.finditer(pat, q):
                candidate_spans.append((m.start(), m.end(), col))

    # Span deduplication: if two column matches overlap on the same token span in user query,
    # keep only one column (preventing collision like 'investment_avenues' + 'avenue' on "investment avenues")
    accepted_spans: list[tuple[int, int, str]] = []
    for start, end, col in sorted(candidate_spans, key=lambda x: (x[1] - x[0]), reverse=True):
        overlap = any(max(start, s[0]) < min(end, s[1]) for s in accepted_spans)
        if not overlap:
            accepted_spans.append((start, end, col))

    # Sort accepted spans by appearance order in query
    accepted_spans.sort(key=lambda x: x[0])
    matched_cols: list[str] = []
    for _, _, col in accepted_spans:
        if col not in matched_cols:
            matched_cols.append(col)

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
    if re.search(r"\bmulti[-_\s]line\b", q):
        explicit_chart = "multi_line"
    elif re.search(r"\bpie\b|\bdonut\b", q):
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

    # Identify numerical columns and data types
    numeric_matched = []
    range_matched = []
    categorical_matched = []
    datetime_matched = []

    for c in matched_cols:
        col_type = dtypes_map.get(c, "")
        is_currency_or_rate = any(k in c.lower() for k in ("income", "salary", "rate", "cost", "price", "fee", "expense", "charges", "sales", "revenue", "amount", "profit", "balance", "limit", "age", "quantity"))
        if is_currency_or_rate or any(nt in col_type for nt in ("int", "float", "double", "number", "real", "numeric", "decimal", "bigint")):
            numeric_matched.append(c)
        elif any(dt in col_type for dt in ("date", "time", "timestamp")) or any(k in c.lower() for k in ("invoicedate", "order_date", "orderdate", "date", "timestamp", "time", "year", "created_at", "saledate", "purchasedate")):
            datetime_matched.append(c)
        elif any(rt in c.lower() for rt in ("expect", "return", "range", "bracket")) and not any(nt in col_type for nt in ("int", "float", "double")):
            range_matched.append(c)
        else:
            categorical_matched.append(c)

    # Detect Filters
    filters: list[dict[str, Any]] = []

    # 1. Return percentage range / category: "between 30% and 40%", "30-40%"
    return_range_match = re.search(r"(\d+)\s*(?:%|-|–|—|to|and)\s*(\d+)\s*%", q)
    if return_range_match:
        ret_col = next((c for c in available_cols if "expect" in c.lower() or "return" in c.lower()), None)
        if ret_col:
            low, high = int(return_range_match.group(1)), int(return_range_match.group(2))
            filters.append({
                "column": ret_col,
                "field": ret_col,
                "op": "==",
                "operator": "equals",
                "value": f"{low}-{high}%",
                "range_val": (low, high),
            })

    # 2. Numeric comparison filters: "above 40", "> 40", "under 20", etc.
    # Associate comparison with the relevant numeric column ONLY if it exists in available_cols
    comp_match = re.search(r"\b(?:above|older than|greater than|>)\s*(\d+(?:\.\d+)?)", q)
    if comp_match:
        target_num_col = None
        # Check if "age" or another column is explicitly mentioned
        if "age" in q.lower():
            target_num_col = next((c for c in available_cols if c.lower() == "age" or "age" in c.lower()), None)
        elif numeric_matched:
            target_num_col = numeric_matched[0]
        elif matched_cols:
            target_num_col = next((c for c in matched_cols if c in available_cols), None)
        if target_num_col:
            val_num = float(comp_match.group(1)) if "." in comp_match.group(1) else int(comp_match.group(1))
            filters.append({"column": target_num_col, "field": target_num_col, "op": ">", "operator": "greater_than", "value": val_num})

    comp_under_match = re.search(r"\b(?:under|younger than|less than|<)\s*(\d+(?:\.\d+)?)", q)
    if comp_under_match:
        target_num_col = None
        if "age" in q.lower():
            target_num_col = next((c for c in available_cols if c.lower() == "age" or "age" in c.lower()), None)
        elif numeric_matched:
            target_num_col = numeric_matched[0]
        elif matched_cols:
            target_num_col = next((c for c in matched_cols if c in available_cols), None)
        if target_num_col:
            val_num = float(comp_under_match.group(1)) if "." in comp_under_match.group(1) else int(comp_under_match.group(1))
            filters.append({"column": target_num_col, "field": target_num_col, "op": "<", "operator": "less_than", "value": val_num})

    # 3. Categorical filters: female only, male only (ONLY if gender column exists in available_cols!)
    gender_col = next((c for c in available_cols if c.lower() == "gender" or "gender" in c.lower() or c.lower() == "sex"), None)
    if gender_col:
        if re.search(r"\b(?:female|women)\b", q) and not re.search(r"\b(?:gender|male and female|both)\b", q):
            if gender_col not in matched_cols:
                filters.append({"column": gender_col, "field": gender_col, "op": "==", "operator": "equals", "value": "Female"})
        elif re.search(r"\bmale\b|\bmen\b", q) and not re.search(r"\b(?:gender|male and female|both)\b", q):
            if gender_col not in matched_cols:
                filters.append({"column": gender_col, "field": gender_col, "op": "==", "operator": "equals", "value": "Male"})

    # Separate filter columns from dimensions to be plotted!
    filter_cols = {f.get("column") for f in filters if f.get("column")}
    plot_dimensions = [c for c in categorical_matched if c not in filter_cols]
    plot_matched_cols = [c for c in matched_cols if c not in filter_cols]

    # Detect measure requested (count, average, sum, min, max)
    measure_type = "count"
    measure_col = None
    if re.search(r"\b(?:average|mean|avg)\b", q):
        measure_type = "mean"
        # Extract target measure phrase e.g. "average expected return by age"
        m_avg = re.search(r"\b(?:average|mean|avg)\s+(?:of\s+)?([a-zA-Z0-9_\s%]+?)(?:\s+by|\s+for|\s+across|\s+as|\s+in|\s+and|$)", q)
        target_measure = None
        if m_avg:
            phrase = m_avg.group(1).strip()
            target_measure = resolve_column_semantically(phrase, available_cols)
        if target_measure and target_measure in matched_cols:
            measure_col = target_measure
        elif numeric_matched:
            by_match = re.search(r"\b(?:by|across|per)\s+([a-zA-Z0-9_]+)", q)
            by_col = resolve_column_semantically(by_match.group(1), available_cols) if by_match else None
            meas_candidates = [c for c in numeric_matched + range_matched if c != by_col]
            measure_col = meas_candidates[0] if meas_candidates else numeric_matched[0]
        else:
            measure_col = range_matched[0] if range_matched else None
    elif re.search(r"\b(?:total|sum)\b", q):
        measure_type = "sum"
        measure_col = numeric_matched[0] if numeric_matched else None
    elif re.search(r"\b(?:highest|maximum|max)\b", q) and not re.search(r"\b(?:most popular|highest count)\b", q):
        measure_type = "max"
        measure_col = numeric_matched[0] if numeric_matched else None
    elif re.search(r"\b(?:lowest|minimum|min)\b", q) and not re.search(r"\b(?:least popular|lowest count)\b", q):
        measure_type = "min"
        measure_col = numeric_matched[0] if numeric_matched else None
    elif numeric_matched and not (explicit_chart in ("scatter", "histogram")):
        measure_type = "sum"
        measure_col = numeric_matched[0]

    # Detect Binning (e.g., "bins of 5", "bins of 5 years", "bin size 5")
    bin_spec: Optional[dict[str, Any]] = None
    bin_match = re.search(r"\b(?:bins?\s*(?:of|size)?\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:years?|year)?\s*bins?)\b", q)
    if bin_match:
        size_raw = bin_match.group(1) or bin_match.group(2)
        size = float(size_raw) if "." in size_raw else int(size_raw)
        bin_spec = {"mode": "fixed", "size": size, "bin_size": size}

    # Detect Intent
    intent = "categorical_comparison"
    if explicit_chart == "histogram" or re.search(r"\b(?:distribution|histogram|spread)\b", q):
        intent = "distribution"
        explicit_chart = "histogram"
        if not bin_spec:
            bin_spec = {"mode": "auto"}
    elif explicit_chart in ("scatter", "relationship"):
        intent = "relationship"
        explicit_chart = "scatter"
    elif len(numeric_matched) >= 2 and any(tok in q for tok in ("versus", "vs", "relationship", "correlation", "against")):
        intent = "relationship"
        if not explicit_chart:
            explicit_chart = "scatter"
    elif len(numeric_matched) == 1 and (len(range_matched) == 1 or len(categorical_matched) == 1) and any(tok in q for tok in ("versus", "vs", "relationship", "against")):
        if not explicit_chart:
            explicit_chart = "scatter" if "scatter" in q else "box"
            intent = "relationship" if explicit_chart == "scatter" else "distribution"
    elif len(plot_dimensions) >= 2 and any(tok in q for tok in ("versus", "vs", "by", "across", "and")):
        intent = "cross_tabulation"
    elif filters and len(plot_dimensions) == 1:
        intent = "filtered_categorical_distribution"
    elif explicit_chart in ("pie", "donut") or re.search(r"\b(?:proportion|percentage|share|ratio|contribution|contributions)\b", q):
        intent = "proportion"
        explicit_chart = "pie"
    elif explicit_chart in ("line", "multi_line") or (
        bool(re.search(r"\b(?:trend|over time|growth|over the years)\b", q))
        or (
            bool(re.search(r"\b(?:monthly|yearly)\b", q))
            and not any(m_word in q.lower() for m_word in ("monthly income", "monthly salary", "monthly charges", "monthly rate", "monthly expense", "monthly cost", "daily rate", "hourly rate"))
        )
    ):
        intent = "trend"
        explicit_chart = explicit_chart or "line"
    elif re.search(r"\b(?:rank|ranking|top|bottom|most popular|least popular)\b", q):
        intent = "ranking"
    elif len(plot_matched_cols) == 0 and not is_follow_up:
        intent = "single_metric"

    # Detect time granularity and series dimension
    time_grain = None
    series_dim = None
    sec_measure = None

    is_temporal = (
        bool(re.search(r"\b(?:trend|over time|growth|over the years|timeline|trajectory|month by month|year by year)\b", q))
        or (
            bool(re.search(r"\b(?:daily|weekly|monthly|quarterly|yearly|annually)\b", q))
            and not any(k in q.lower() for k in ("monthly income", "monthly salary", "monthly charges", "monthly rate", "monthly expense", "monthly cost", "daily rate", "hourly rate"))
        )
        or (datetime_matched and len(datetime_matched) > 0 and not any(k in q.lower() for k in ("min", "max", "average", "avg", "sum", "total") if not any(w in q.lower() for w in ("by", "per", "across", "trend"))))
    )
    if is_temporal:
        intent = "trend"
        explicit_chart = explicit_chart or "line"
        if any(w in q for w in ("monthly", "month by month", "per month")):
            time_grain = "month"
        elif any(w in q for w in ("yearly", "year by year", "annually", "annual")):
            time_grain = "year"
        elif any(w in q for w in ("quarterly", "per quarter")):
            time_grain = "quarter"
        elif any(w in q for w in ("weekly", "per week")):
            time_grain = "week"
        elif any(w in q for w in ("daily", "day by day", "per day")):
            time_grain = "day"
        else:
            time_grain = "month"

        if plot_dimensions:
            series_dim = plot_dimensions[0]

    # Line and Multi-line handling
    color_by_col = None
    if explicit_chart in ("line", "multi_line"):
        intent = "trend"
        if explicit_chart == "multi_line" or (plot_dimensions and len(plot_dimensions) >= 1):
            if plot_dimensions:
                series_dim = plot_dimensions[0]
                color_by_col = series_dim

    # Scatter vs Clustered bar
    is_scatter = (
        explicit_chart == "scatter"
        or (len(numeric_matched) >= 2 and bool(re.search(r"\b(?:versus|vs\.?|relationship|against|correlation)\b", q)))
    )
    if is_scatter:
        intent = "relationship"
        explicit_chart = "scatter"
        if plot_dimensions:
            color_by_col = plot_dimensions[0]
            series_dim = plot_dimensions[0]
    elif len(numeric_matched) >= 2 and len(plot_dimensions) == 1 and not is_temporal:
        measure_col = numeric_matched[0]
        sec_measure = numeric_matched[1]
        measure_type = "sum"
        explicit_chart = explicit_chart or "clustered_bar"

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
        if sorting.get("direction") == "desc" and not explicit_chart:
            explicit_chart = "horizontal_bar"

    # Dimensions calculation
    if explicit_chart in ("line", "multi_line"):
        # Resolve X dimension
        x_dim = None
        if is_temporal:
            x_dim = next((
                c for c in datetime_matched + available_cols
                if any(k in c.lower() for k in ("invoicedate", "order_date", "orderdate", "date", "timestamp", "time", "year", "month", "day", "created_at", "saledate"))
            ), None)

        if not x_dim:
            by_match = re.search(r"\b(?:by|across|per)\s+([a-zA-Z0-9_]+)", q)
            if by_match:
                x_dim = resolve_column_semantically(by_match.group(1), available_cols)
        if not x_dim:
            x_dim = next((c for c in datetime_matched + numeric_matched + matched_cols if c != measure_col and c != series_dim), None)
        dimensions = [x_dim] if x_dim else []
        if series_dim and series_dim not in dimensions:
            dimensions.append(series_dim)
        sorting = {"field": x_dim, "direction": "asc"}
    elif intent in ("relationship", "distribution"):
        dimensions = []
    elif plot_dimensions:
        dimensions = [d for d in plot_dimensions if d != measure_col]
    else:
        dimensions = [c for c in plot_matched_cols if c != measure_col]

    vis_required = (
        is_visualization_requested(q_raw, intent)
        or bool(explicit_chart)
        or intent in ("cross_tabulation", "distribution", "relationship", "filtered_categorical_distribution", "trend", "proportion")
    )

    return QueryContract(
        raw_question=q_raw,
        visualization_required=vis_required,
        intent=intent,
        requested_dimensions=dimensions,
        requested_measures=[{"type": measure_type, "column": measure_col}],
        requested_fields=plot_matched_cols if plot_matched_cols else matched_cols,
        filters=filters,
        bin=bin_spec,
        requested_chart=explicit_chart,
        sorting=sorting,
        limit=limit,
        time_granularity=time_grain,
        series_dimension=series_dim,
        secondary_measure=sec_measure,
        color_by=color_by_col,
        is_follow_up=is_follow_up,
    )
