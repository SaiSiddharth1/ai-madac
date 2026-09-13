"""
SQL validation utility — ensures generated SQL is safe to execute.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Dangerous SQL keywords that must never appear in generated queries
FORBIDDEN_KEYWORDS = [
    r"\bDROP\b",
    r"\bDELETE\b",
    r"\bUPDATE\b",
    r"\bINSERT\b",
    r"\bALTER\b",
    r"\bTRUNCATE\b",
    r"\bCREATE\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
    r"\bEXEC\b",
    r"\bEXECUTE\b",
    r"\bMERGE\b",
    r"\bCALL\b",
    r"\bSET\b",
    r"--",  # SQL comments (potential injection)
    r"/\*",  # Block comments
    r";\s*\w",  # Multiple statements
]


def validate_sql(sql: str) -> tuple[bool, Optional[str]]:
    """
    Validate that a SQL query is safe to execute.

    Returns:
        (is_valid, error_message) — error_message is None if valid.
    """
    if not sql or not sql.strip():
        return False, "Empty SQL query"

    sql_upper = sql.strip().upper()

    # Must start with SELECT
    if not sql_upper.startswith("SELECT"):
        return False, "Only SELECT queries are allowed"

    # Check for forbidden keywords
    for pattern in FORBIDDEN_KEYWORDS:
        if re.search(pattern, sql_upper):
            keyword = re.search(pattern, sql_upper).group()
            return False, f"Forbidden SQL keyword detected: {keyword}"

    # Check for multiple statements
    # Remove string literals before checking for semicolons
    sql_no_strings = re.sub(r"'[^']*'", "", sql)
    statements = [s.strip() for s in sql_no_strings.split(";") if s.strip()]
    if len(statements) > 1:
        return False, "Multiple SQL statements are not allowed"

    return True, None


def sanitize_table_name(table_name: str) -> str:
    """Sanitize a table name to prevent injection."""
    return re.sub(r"[^a-zA-Z0-9_]", "", table_name)
