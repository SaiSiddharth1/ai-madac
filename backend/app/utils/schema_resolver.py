"""
Centralized Schema-Aware Semantic Resolver for AI-MADAC.
Resolves measures, dimensions, filters, grouping columns, and series dimensions
strictly against the active dataset schema without guessing or substituting unrelated columns.
"""

import re
from typing import Any, Optional

from app.schemas.query_contract import FilterSpec
from app.utils.temporal import is_temporal_column, find_temporal_column


SEMANTIC_ALIASES: dict[str, list[str]] = {
    "gender": ["gender", "sex", "male", "female", "males", "females", "men", "women"],
    "monthly_income": [
        "monthlyincome", "monthly_income", "monthly income", "monthly salary",
        "monthly_salary", "monthly pay", "monthly earnings"
    ],
    "income": [
        "income", "monthlyincome", "monthly_income", "annual_income", "annualincome",
        "salary", "earnings", "monthly salary", "pay", "compensation"
    ],
    "job_level": ["joblevel", "job_level", "job level", "level", "seniority", "grade", "rank"],
    "job_role": [
        "jobrole", "job_role", "job role", "job roles", "role", "roles", "position",
        "positions", "title", "job title", "designation", "profession", "occupation"
    ],
    "overtime": ["overtime", "over_time", "over time"],
    "age": ["age", "employee age", "investor age", "years old", "age group", "age bracket", "ages"],
    "department": ["department", "dept", "division", "team", "unit", "business unit", "departments"],
    "attrition": ["attrition", "turnover", "churn", "resignation", "left"],
    "gender": ["gender", "sex", "male and female", "male or female", "men and women", "male", "female", "males", "females"],
    "overtime": ["overtime", "over time", "work overtime", "working overtime"],
    "sales": ["sales", "gross_sales", "total_sales", "sales_amount", "revenue", "turnover"],
    "profit": ["profit", "net_profit", "earnings", "margin", "gain"],
    "quantity": ["quantity", "units", "units_sold", "volume", "qty", "items_sold"],
    "product": ["product", "product_name", "product_id", "product_description", "item", "sku"],
    "category": ["category", "categories", "segment", "classification", "group", "product_category"],
    "sub_category": ["sub_category", "sub category", "subcategory", "sub-category"],
    "region": ["region", "area", "zone", "location", "territory", "geography"],
    "country": ["country", "nation"],
    "city": ["city", "town", "metro", "municipality"],
    "state": ["state", "province"],
    "order": ["order", "orders", "order_id", "orderid", "transaction", "invoice"],
    "customer": ["customer", "customers", "customer_name", "customer_id", "client"],
    "discount": ["discount", "rebate", "markdown"],
    "education": ["education", "education_level", "qualification", "degree"],
    "education_field": ["educationfield", "education_field", "education field", "major", "field_of_study"],
    "environment_satisfaction": ["environmentsatisfaction", "environment_satisfaction", "environment satisfaction"],
    "job_satisfaction": ["jobsatisfaction", "job_satisfaction", "job satisfaction"],
    "work_life_balance": ["worklifebalance", "work_life_balance", "work life balance"],
    "years_at_company": ["yearsatcompany", "years_at_company", "years at company", "years at the company", "tenure"],
    "investment_objective": [
        "investment_objective", "investment objective", "investment objectives",
        "objective", "objectives", "goal", "goals"
    ],
    "investment_avenue": [
        "investment_avenue", "investment avenue", "investment avenues", "avenue", "avenues"
    ],
    "expected_return": [
        "expected_return", "expected return", "expected returns", "expect", "return", "returns"
    ],
}


def normalize_term(term: str) -> str:
    """Normalize a column name or query phrase for robust semantic matching."""
    if not term:
        return ""
    # Split camelCase
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(term))
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", s.lower())
    return " ".join(cleaned.split())


class SchemaResolver:
    """Centralized schema-aware column and intent resolver."""

    def __init__(self, schema: Optional[dict[str, Any] | list[str]] = None):
        self.schema = schema if isinstance(schema, dict) else None
        if isinstance(schema, dict):
            self.available_columns = [c.get("name", "") for c in schema.get("columns", []) if c.get("name")]
        elif isinstance(schema, list):
            self.available_columns = [str(c) for c in schema if c]
        else:
            self.available_columns = []

    def resolve_column(
        *args,
        **kwargs,
    ) -> Optional[str]:
        """
        Supports both instance and classmethod call styles:
        - resolver.resolve_column("term")
        - SchemaResolver.resolve_column("term", available_columns, schema)
        """
        if not args:
            return None

        if isinstance(args[0], SchemaResolver):
            self = args[0]
            term = args[1] if len(args) > 1 else kwargs.get("term", "")
            cols = args[2] if len(args) > 2 else kwargs.get("available_columns", self.available_columns)
            sch = args[3] if len(args) > 3 else kwargs.get("schema", self.schema)
            pref_num = args[4] if len(args) > 4 else kwargs.get("prefer_numeric", False)
        else:
            term = args[0]
            cols = args[1] if len(args) > 1 else kwargs.get("available_columns", [])
            sch = args[2] if len(args) > 2 else kwargs.get("schema", None)
            pref_num = args[3] if len(args) > 3 else kwargs.get("prefer_numeric", False)

        return SchemaResolver._resolve_column_impl(term, cols, sch, pref_num)

    @classmethod
    def _resolve_column_impl(
        cls,
        term: str,
        available_columns: list[str],
        schema: Optional[dict[str, Any]] = None,
        prefer_numeric: bool = False,
    ) -> Optional[str]:
        """
        Resolve a term to an actual dataset column using strict priority order:
        1. Exact case-sensitive match
        2. Exact case-insensitive match
        3. Normalized match (spaces and underscores equivalent)
        4. Compound phrase match (e.g. 'monthly income' -> 'MonthlyIncome')
        5. Singular/plural word variations
        6. Known semantic alias mapping
        7. Partial exact word tokens
        8. Ambiguity check: return None if ambiguous
        NEVER substitute an unrelated column.
        """
        if not term or not available_columns:
            return None

        clean_term = term.strip()
        t_lower = clean_term.lower()
        t_norm = normalize_term(clean_term)

        # 1. Exact case-sensitive match
        if clean_term in available_columns:
            return clean_term

        # 2. Case-insensitive match
        for col in available_columns:
            if col.lower() == t_lower:
                return col

        # 3. Normalized match
        for col in available_columns:
            if normalize_term(col) == t_norm:
                return col

        # 4. Check compound phrase without spaces (e.g., 'monthly income' vs 'monthlyincome')
        t_nospaces = t_norm.replace(" ", "")
        for col in available_columns:
            if normalize_term(col).replace(" ", "") == t_nospaces:
                return col

        # 5. Singular / plural variations
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

        # 6. Known semantic aliases
        matching_alias_cols = []
        for alias_key, aliases in SEMANTIC_ALIASES.items():
            norm_alias_key = normalize_term(alias_key)
            norm_aliases = {normalize_term(a) for a in aliases}
            norm_aliases.add(norm_alias_key)

            if any(cand in norm_aliases for cand in candidates) or any(a in t_norm for a in norm_aliases if len(a) > 3):
                for col in available_columns:
                    col_norm = normalize_term(col)
                    col_nospaces = col_norm.replace(" ", "")
                    if col_norm in norm_aliases or any(col_nospaces == a.replace(" ", "") for a in norm_aliases):
                        if col not in matching_alias_cols:
                            matching_alias_cols.append(col)

        if len(matching_alias_cols) == 1:
            return matching_alias_cols[0]
        elif len(matching_alias_cols) > 1:
            # If multiple columns match (e.g. 'Product Name' and 'Product ID'),
            # prefer the descriptive/name column over an ID column unless user explicitly asked for 'id'
            if "id" not in t_norm.split():
                non_id_cols = [c for c in matching_alias_cols if "id" not in normalize_term(c).split()]
                if len(non_id_cols) == 1:
                    return non_id_cols[0]
                elif len(non_id_cols) > 1:
                    name_cols = [c for c in non_id_cols if "name" in normalize_term(c).split()]
                    if len(name_cols) == 1:
                        return name_cols[0]

            # Ambiguity check: e.g. "rate" matching DailyRate, HourlyRate, MonthlyRate
            # Prefer exact phrase match if exists
            for col in matching_alias_cols:
                if t_norm in normalize_term(col):
                    return col
            # Multiple columns matched equally: flag as ambiguous
            return None

        # 7. Exact multi-word subset (e.g., 'job role' in 'job role description' or 'job role' matching 'JobRole')
        for col in available_columns:
            col_norm = normalize_term(col)
            if t_norm and t_norm == col_norm:
                return col

        # 8. Word-level match
        t_words = t_norm.split()
        if len(t_words) > 1:
            for col in available_columns:
                col_words = normalize_term(col).split()
                if all(w in col_words for w in t_words):
                    return col

        # NEVER return first numeric column or a random column!
        return None

    @classmethod
    def validate_contract_against_schema(
        cls,
        contract: Any,
        schema: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """
        Strict validation of the QueryContract against the active dataset schema.
        Verifies:
        - Measure column exists if required
        - Group by columns exist
        - Series dimension exists if requested
        - Filter columns exist
        - Unrelated dataset requests are caught immediately
        Returns: (is_valid, list_of_error_reasons)
        """
        cols_in_schema = [str(c.get("name", "")) for c in schema.get("columns", []) if c.get("name")]
        cols_lower = {c.lower(): c for c in cols_in_schema}

        errors = []

        # 1. Validate Measure
        if contract.measure_column:
            # Handle multi-column expressions like "Quantity * UnitPrice"
            sub_identifiers = re.findall(r'[a-zA-Z_]\w*', contract.measure_column)
            for sub_id in sub_identifiers:
                if sub_id.lower() not in cols_lower and sub_id != "total_records":
                    # Check semantic match
                    res = cls.resolve_column(sub_id, cols_in_schema, schema)
                    if not res:
                        errors.append(
                            f"The requested measure column '{sub_id}' does not exist in the active dataset schema. "
                            f"Available columns: {', '.join(cols_in_schema[:8])}..."
                        )

        # 2. Validate Group By / Dimensions
        for dim in getattr(contract, "group_by", []) or []:
            if dim.lower() not in cols_lower:
                res = cls.resolve_column(dim, cols_in_schema, schema)
                if not res:
                    errors.append(f"The grouping dimension '{dim}' does not exist in the active dataset.")

        # 3. Validate Series Dimension
        series_dim = getattr(contract, "series_dimension", None)
        if series_dim:
            if series_dim.lower() not in cols_lower:
                res = cls.resolve_column(series_dim, cols_in_schema, schema)
                if not res:
                    errors.append(f"The series dimension '{series_dim}' does not exist in the active dataset.")

        # 4. Validate Filters
        for f in getattr(contract, "filters", []) or []:
            f_col = getattr(f, "column", None) or (f.get("column") if isinstance(f, dict) else None)
            if f_col and f_col.lower() not in cols_lower:
                res = cls.resolve_column(f_col, cols_in_schema, schema)
                if not res:
                    errors.append(f"Filter column '{f_col}' does not exist in the active dataset.")

        # 5. Validate Temporal column if intent is TIME_SERIES
        if getattr(contract, "intent", None) == "TIME_SERIES" or getattr(contract, "time_column", None):
            t_col = getattr(contract, "time_column", None)
            if t_col:
                if t_col.lower() not in cols_lower:
                    errors.append(f"Time column '{t_col}' does not exist in the active dataset.")
                elif not is_temporal_column(t_col, schema=schema):
                    errors.append(f"Column '{t_col}' is not a valid date or timestamp.")

        return (len(errors) == 0, errors)

    @classmethod
    def resolve_filters(
        cls,
        question: str,
        available_columns: list[str],
        schema: Optional[dict[str, Any]] = None,
        existing_group_by: Optional[list[str]] = None,
    ) -> list[FilterSpec]:
        """
        Dynamically resolve semantic row-level filters from natural language query
        strictly against the active dataset schema without guessing.
        Handles:
        - Attrition / Turnover / Leaving status (Yes / No)
        - Overtime status (Yes / No)
        - Categorical dimension values (e.g. Sales department)
        - Gender filtering (Male / Female)
        """
        if not question or not available_columns:
            return []

        q_lower = question.lower().strip()
        existing_gb = [str(g).lower() for g in (existing_group_by or [])]
        filters: list[FilterSpec] = []

        def get_bool_values(col: str) -> tuple[str, str]:
            if isinstance(schema, dict):
                for c in schema.get("columns", []):
                    if isinstance(c, dict) and c.get("name", "").lower() == col.lower():
                        samples = c.get("sample_values") or c.get("sample") or []
                        for s in samples:
                            s_str = str(s).strip()
                            if s_str.lower() in ("yes", "y", "true", "1"):
                                return s_str, "No" if s_str.lower() in ("yes", "y") else "False"
                            elif s_str.lower() in ("no", "n", "false", "0"):
                                return "Yes" if s_str.lower() in ("no", "n") else "True", s_str
            return "Yes", "No"

        # 1. Attrition / Leaving Filter
        attr_col = cls.resolve_column("attrition", available_columns, schema)
        if attr_col and attr_col.lower() not in existing_gb and not re.search(r"\b(?:by|for each|each|across)\s+attrition\b", q_lower):
            yes_val, no_val = get_bool_values(attr_col)
            is_neg = bool(re.search(
                r"\b(?:not\s+left|didn't\s+leave|did\s+not\s+leave|have\s+not\s+left|haven't\s+left|stayed|remain(?:ed|ing)?|current\s+employees|active\s+employees|no\s+attrition|non[-_\s]attrition)\b",
                q_lower
            ))
            is_aff = bool(re.search(
                r"\b(?:left\s+(?:the\s+)?company|left|employees\s+who\s+left|have\s+left|who\s+have\s+left|who\s+left|resigned|churned|attrition(?:\s+count)?|attrited)\b",
                q_lower
            ))
            if is_neg:
                filters.append(FilterSpec(column=attr_col, operator="=", value=no_val))
            elif is_aff:
                filters.append(FilterSpec(column=attr_col, operator="=", value=yes_val))

        # 2. Overtime Filter
        ot_col = cls.resolve_column("overtime", available_columns, schema)
        if ot_col and ot_col.lower() not in existing_gb and not re.search(r"\b(?:by|for each|each|across)\s+overtime\b", q_lower):
            yes_val, no_val = get_bool_values(ot_col)
            is_neg = bool(re.search(
                r"\b(?:do\s+not\s+work\s+overtime|don't\s+work\s+overtime|without\s+overtime|no\s+overtime|not\s+working\s+overtime|who\s+do\s+not\s+work\s+overtime)\b",
                q_lower
            ))
            is_aff = bool(re.search(
                r"\b(?:who\s+work\s+overtime|work\s+overtime|working\s+overtime|with\s+overtime|have\s+overtime|overtime)\b",
                q_lower
            ))
            if is_neg:
                filters.append(FilterSpec(column=ot_col, operator="=", value=no_val))
            elif is_aff:
                filters.append(FilterSpec(column=ot_col, operator="=", value=yes_val))

        # 3. Department Filter (e.g. Sales department)
        dept_col = cls.resolve_column("department", available_columns, schema)
        if dept_col and dept_col.lower() not in existing_gb:
            matched_val = None
            if isinstance(schema, dict):
                for c in schema.get("columns", []):
                    if isinstance(c, dict) and c.get("name", "").lower() == dept_col.lower():
                        for s in (c.get("sample_values") or c.get("sample") or []):
                            s_clean = str(s).strip()
                            if len(s_clean) > 1 and re.search(rf"\b{re.escape(s_clean.lower())}\b", q_lower):
                                matched_val = s_clean
                                break
            if not matched_val:
                dept_m = re.search(r"\b([a-zA-Z\s&]+?)\s+(?:department|dept)\b", q_lower)
                if dept_m:
                    cand_raw = dept_m.group(1).strip()
                    cand_words = [w for w in cand_raw.split() if w.lower() not in ("how", "many", "the", "in", "from", "for", "of", "are", "there", "show", "what", "is")]
                    cand_val = " ".join(cand_words).strip()
                    if cand_val.lower() == "sales":
                        matched_val = "Sales"
                    elif cand_val.lower() in ("r&d", "research & development", "research and development"):
                        matched_val = "Research & Development"
                    elif cand_val.lower() in ("hr", "human resources"):
                        matched_val = "Human Resources"
                    elif cand_val:
                        matched_val = cand_val.title()
            if matched_val:
                filters.append(FilterSpec(column=dept_col, operator="=", value=matched_val))

        # 4. Gender Filter (e.g. male employees, female employees)
        gender_col = cls.resolve_column("gender", available_columns, schema)
        if gender_col and gender_col.lower() not in existing_gb and not re.search(r"\b(?:by|for each|each|across)\s+gender\b", q_lower):
            if re.search(r"\b(?:male|males|men)\b", q_lower) and not re.search(r"\b(?:female|females|women)\b", q_lower):
                filters.append(FilterSpec(column=gender_col, operator="=", value="Male"))
            elif re.search(r"\b(?:female|females|women)\b", q_lower) and not re.search(r"\b(?:male|males|men)\b", q_lower):
                filters.append(FilterSpec(column=gender_col, operator="=", value="Female"))

        return filters

    @classmethod
    def resolve_grouping_columns(
        cls,
        question: str,
        available_columns: list[str],
        schema: Optional[dict[str, Any]] = None,
        exclude_columns: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Dynamically identify grouping/dimension columns from natural language query
        using semantic grouping indicators strictly against the active schema:
        - "by X"
        - "in each X"
        - "for each X"
        - "each X"
        - "per X"
        - "across X"
        - "broken down by X"
        - "grouped by X"
        """
        if not question or not available_columns:
            return []

        q_clean = re.sub(r"[?!.,]", " ", question.lower())
        exclude = {e.lower() for e in (exclude_columns or []) if e}
        group_cols: list[str] = []

        # Multi-dimension comparison patterns:
        # Pattern 1: "compare ... for employees who work overtime and those who do not across age"
        comp_ot_m = re.search(
            r"\b(?:compare|comparison|versus|vs)\b.*?\bfor\s+(?:employees\s+who\s+)?([a-zA-Z0-9_\s]+?)\s+(?:and|&)\s+(?:those\s+who\s+do\s+not|[a-zA-Z0-9_\s]+?)\s+across\s+([a-zA-Z0-9_\s]+)",
            q_clean
        )
        if comp_ot_m:
            c1_str = comp_ot_m.group(1).strip()
            c2_str = comp_ot_m.group(2).strip()
            col1 = cls.resolve_column(c1_str, available_columns, schema)
            col2 = cls.resolve_column(c2_str, available_columns, schema)
            if col1 and col1.lower() not in exclude and col1 not in group_cols:
                group_cols.append(col1)
            if col2 and col2.lower() not in exclude and col2 not in group_cols:
                group_cols.append(col2)

        # Pattern 2: "compare ... for male and female employees across age"
        comp_mf_m = re.search(
            r"\b(?:compare|comparison|versus|vs)\b.*?\bfor\s+([a-zA-Z0-9_\s]+?)\s+(?:and|&|versus|vs)\s+([a-zA-Z0-9_\s]+?)(?:\s+employees)?\s+across\s+([a-zA-Z0-9_\s]+)",
            q_clean
        )
        if comp_mf_m:
            g1_str = f"{comp_mf_m.group(1).strip()} and {comp_mf_m.group(2).strip()}"
            c2_str = comp_mf_m.group(3).strip()
            col1 = cls.resolve_column(g1_str, available_columns, schema) or cls.resolve_column("gender", available_columns, schema)
            col2 = cls.resolve_column(c2_str, available_columns, schema)
            if col1 and col1.lower() not in exclude and col1 not in group_cols:
                group_cols.append(col1)
            if col2 and col2.lower() not in exclude and col2 not in group_cols:
                group_cols.append(col2)

        # Pattern 3: "across X for each Y" or "for each X across Y"
        across_each_m = re.search(
            r"\b(?:across|for\s+each|in\s+each|by)\s+([a-zA-Z0-9_\s]+?)\s+(?:for\s+each|in\s+each|across|by)\s+([a-zA-Z0-9_\s]+)",
            q_clean
        )
        if across_each_m:
            c1_str = across_each_m.group(1).strip()
            c2_str = across_each_m.group(2).strip()
            col1 = cls.resolve_column(c1_str, available_columns, schema)
            col2 = cls.resolve_column(c2_str, available_columns, schema)
            if col1 and col1.lower() not in exclude and col1 not in group_cols:
                group_cols.append(col1)
            if col2 and col2.lower() not in exclude and col2 not in group_cols:
                group_cols.append(col2)

        # Multi-dimension pattern: "by X and Y" or "for each X and Y"
        multi_m = re.search(
            r"\b(?:broken\s+down\s+by|grouped\s+by|by|in\s+each|for\s+each|each|per|across)\s+([a-zA-Z0-9_\s&]+?)\s+(?:and|&)\s+([a-zA-Z0-9_\s&]+?)(?:\s+as\s+|\s+in\s+|$|\s+using\b)",
            q_clean
        )
        if multi_m:
            c1_str = re.sub(r"\b(?:group|bracket|range|category)\b", "", multi_m.group(1)).strip()
            c2_str = re.sub(r"\b(?:group|bracket|range|category)\b", "", multi_m.group(2)).strip()
            col1 = cls.resolve_column(c1_str, available_columns, schema)
            col2 = cls.resolve_column(c2_str, available_columns, schema)
            if col1 and col1.lower() not in exclude and col1 not in group_cols:
                group_cols.append(col1)
            if col2 and col2.lower() not in exclude and col2 not in group_cols:
                group_cols.append(col2)

        # Single-dimension patterns:
        single_matches = re.finditer(
            r"\b(?:broken\s+down\s+by|grouped\s+by|by|in\s+each|for\s+each|each|per|across)\s+([a-zA-Z0-9_\s&]+?)(?:\s+for\s+each\b|\s+in\s+each\b|\s+across\b|\s+by\b|\s+per\b|\s+broken\s+down\s+by\b|\s+grouped\s+by\b|\s+as\s+|\s+in\s+|$|\s+using\b|\s+with\b)",
            q_clean
        )
        for m in single_matches:
            raw_cand = m.group(1).strip()
            cand = re.sub(r"\b(?:group|bracket|range|category|level)\b", "", raw_cand).strip()
            col = cls.resolve_column(cand, available_columns, schema) or cls.resolve_column(raw_cand, available_columns, schema)
            if col and col.lower() not in exclude and col not in group_cols:
                group_cols.append(col)

        # Fallback check for exact column mentions following grouping keywords
        if not group_cols:
            for c in available_columns:
                if c.lower() in exclude:
                    continue
                c_norm = normalize_term(c)
                if re.search(rf"\b(?:by|each|per)\s+{re.escape(c_norm)}\b", q_clean):
                    if c not in group_cols:
                        group_cols.append(c)

        # Prioritize continuous/numeric or temporal column as X-axis (first) and categorical as series (second)
        if len(group_cols) >= 2:
            num_or_temp = []
            cat = []
            for c in group_cols:
                if cls.is_numeric_column(c, schema) or is_temporal_column(c, schema=schema):
                    num_or_temp.append(c)
                else:
                    cat.append(c)
            if num_or_temp and cat:
                group_cols = num_or_temp + cat

        return group_cols

    @classmethod
    def resolve_measure_column(
        cls,
        question: str,
        available_columns: list[str],
        schema: Optional[dict[str, Any]] = None,
        exclude_columns: Optional[list[str]] = None,
    ) -> Optional[str]:
        """
        Dynamically identify the numeric measure requested by the question
        strictly against the active dataset schema without guessing.
        """
        if not question or not available_columns:
            return None

        q_clean = question.lower().strip()
        exclude = {e.lower() for e in (exclude_columns or []) if e}

        numeric_cols = []
        if isinstance(schema, dict):
            for c in schema.get("columns", []):
                if isinstance(c, dict):
                    name = c.get("name", "")
                    dtype = str(c.get("dtype", "")).lower()
                    if name and any(num_t in dtype for num_t in ("int", "float", "real", "num", "double", "decimal")):
                        numeric_cols.append(name)
        if not numeric_cols:
            numeric_cols = list(available_columns)

        candidates = []
        for col in numeric_cols:
            if col.lower() in exclude:
                continue
            col_norm = normalize_term(col)
            if len(col_norm) > 2 and re.search(rf"\b{re.escape(col_norm)}\b", q_clean):
                candidates.append((len(col_norm), col))
            for alias_key, aliases in SEMANTIC_ALIASES.items():
                if any(normalize_term(a) == col_norm for a in aliases) or normalize_term(alias_key) == col_norm:
                    for a in aliases:
                        a_norm = normalize_term(a)
                        if len(a_norm) > 2 and re.search(rf"\b{re.escape(a_norm)}\b", q_clean):
                            candidates.append((len(a_norm), col))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return None

    @classmethod
    def get_column_dtype(
        cls,
        col_name: str,
        schema: Optional[dict[str, Any]] = None,
    ) -> str:
        """Return the lowercase dtype of a column from schema."""
        if not col_name or not isinstance(schema, dict):
            return ""
        for c in schema.get("columns", []):
            if isinstance(c, dict) and c.get("name", "").lower() == col_name.lower():
                return str(c.get("dtype", "")).lower()
        return ""

    @classmethod
    def is_numeric_column(
        cls,
        col_name: str,
        schema: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Check if column is continuous numeric (not categorical or ID)."""
        if not col_name:
            return False
        col_lower = col_name.lower()
        if col_lower.endswith("_id") or col_lower.endswith("id") or col_lower.endswith("_code"):
            return False

        dtype = cls.get_column_dtype(col_name, schema)
        if dtype:
            if any(t in dtype for t in ("char", "str", "text", "object", "bool")):
                return False
            if any(t in dtype for t in ("int", "float", "real", "num", "double", "decimal")):
                return True

        # Check sample values if available
        if isinstance(schema, dict):
            for c in schema.get("columns", []):
                if isinstance(c, dict) and c.get("name", "").lower() == col_lower:
                    samples = c.get("sample_values") or c.get("sample") or []
                    if samples:
                        num_count = 0
                        for s in samples:
                            try:
                                float(str(s).replace(",", ""))
                                num_count += 1
                            except (ValueError, TypeError):
                                pass
                        if num_count / len(samples) >= 0.7:
                            return True
                        return False

        # Fallback common numeric column name keywords
        if any(k in col_lower for k in ("age", "income", "salary", "rate", "cost", "price", "sales", "revenue", "amount", "profit", "years", "distance", "duration", "score")):
            return True

        return False

    @classmethod
    def is_categorical_column(
        cls,
        col_name: str,
        schema: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Check if column is categorical (text, object, low cardinality string)."""
        return not cls.is_numeric_column(col_name, schema)

    @classmethod
    def get_column_cardinality(
        cls,
        col_name: str,
        schema: Optional[dict[str, Any]] = None,
    ) -> Optional[int]:
        """Return unique count or sample value count if available."""
        if not col_name or not isinstance(schema, dict):
            return None
        for c in schema.get("columns", []):
            if isinstance(c, dict) and c.get("name", "").lower() == col_name.lower():
                uc = c.get("unique_count")
                if uc is not None:
                    return int(uc)
                samples = c.get("sample_values") or c.get("sample") or []
                if samples:
                    return len(set(str(s) for s in samples))
        return None

    @classmethod
    def resolve_distribution_column(
        cls,
        question: str,
        available_columns: list[str],
        schema: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Identify the column targeted by a distribution question strictly against schema.
        Handles:
        - "Show the distribution of employees by gender." -> Gender
        - "Show the distribution of employee ages." -> Age
        - "Show the distribution of monthly income." -> MonthlyIncome
        - "Show the distribution of job roles." -> JobRole
        """
        if not question or not available_columns:
            return None

        q_clean = re.sub(r"[?!.,]", " ", question.lower()).strip()

        # 1. Check grouping resolution (e.g. "by gender" or "by job role")
        group_cols = cls.resolve_grouping_columns(question, available_columns, schema)
        if group_cols:
            return group_cols[0]

        # 2. Extract after "distribution of" or "histogram of"
        dist_m = re.search(
            r"\b(?:distribution|histogram|spread)\s+of\s+([a-zA-Z0-9_\s&]+?)(?:\s+by|\s+across|\s+using|\s+as|\s+in|$)",
            q_clean
        )
        if dist_m:
            cand_raw = dist_m.group(1).strip()
            # Clean common filler words like "employee", "employees", "all", "the"
            cand_words = [w for w in cand_raw.split() if w not in ("employee", "employees", "all", "the", "our", "company")]
            cand_clean = " ".join(cand_words).strip()

            col = (
                cls.resolve_column(cand_clean, available_columns, schema)
                or cls.resolve_column(cand_raw, available_columns, schema)
            )
            if col:
                return col

        # 3. Check measure column (e.g. monthly income)
        meas_col = cls.resolve_measure_column(question, available_columns, schema)
        if meas_col:
            return meas_col

        # 4. Check any column name mentioned in question
        for c in available_columns:
            c_norm = normalize_term(c)
            if len(c_norm) > 2 and re.search(rf"\b{re.escape(c_norm)}\b", q_lower):
                return c

        return None

