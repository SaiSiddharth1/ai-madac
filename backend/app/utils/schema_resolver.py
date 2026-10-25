"""
Centralized Schema-Aware Semantic Resolver for AI-MADAC.
Resolves measures, dimensions, filters, grouping columns, and series dimensions
strictly against the active dataset schema without guessing or substituting unrelated columns.
"""

import re
from typing import Any, Optional

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
    "age": ["age", "employee age", "investor age", "years old"],
    "department": ["department", "dept", "division", "team", "unit", "business unit"],
    "attrition": ["attrition", "turnover", "churn", "resignation", "left"],
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
    "years_at_company": ["yearsatcompany", "years_at_company", "years at company", "tenure"],
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
