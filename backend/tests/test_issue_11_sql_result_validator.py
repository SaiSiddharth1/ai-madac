"""
Tests for Issue 11: SQL Result Validator.
Verifies that after SQL execution, the result DataFrame is strictly validated against AnalysisSpec:
- Grouped queries reject scalar results (e.g. 1 row total_records=1470 instead of department-level rows)
- Grouped queries reject results missing group_by columns
- Scalar queries reject multiple rows
- Distribution queries reject scalar aggregates instead of raw observations
- Auto-repair and re-execution in SQLAgent
"""

import pandas as pd
import pytest

from app.agents.sql.result_validator import SQLResultValidator
from app.agents.sql.agent import SQLAgent
from app.schemas.query_contract import QueryContract


@pytest.fixture
def hr_schema():
    return {
        "columns": [
            {"name": "Age", "dtype": "int64", "sample_values": [41, 49, 37]},
            {"name": "Attrition", "dtype": "object", "sample_values": ["Yes", "No"]},
            {"name": "Department", "dtype": "object", "sample_values": ["Sales", "Research & Development", "Human Resources"]},
            {"name": "MonthlyIncome", "dtype": "int64", "sample_values": [5993, 5130, 2090]},
        ],
        "row_count": 1470,
    }


def test_reject_scalar_when_grouped_expected(hr_schema):
    """If AnalysisSpec expects group_by=['Department'], a 1-row scalar total must be rejected."""
    spec = QueryContract(
        query_id="q1",
        dataset_id="d1",
        original_query="How many employees are there in each department?",
        aggregation="COUNT",
        measure="total_records",
        group_by=["Department"],
        answer_type="GROUPED",
    )
    # Simulated invalid SQL output: 1 row global count
    bad_df = pd.DataFrame([{"total_records": 1470}])
    is_valid, reason = SQLResultValidator.validate(bad_df, spec, hr_schema, dataset_row_count=1470)

    assert is_valid is False
    assert "EXPECTED GROUPED RESULT" in reason


def test_reject_missing_group_column(hr_schema):
    """If result has multiple rows but lacks the grouping column, it must be rejected."""
    spec = QueryContract(
        query_id="q2",
        dataset_id="d1",
        original_query="Show average monthly income by department",
        aggregation="AVG",
        measure_column="MonthlyIncome",
        group_by=["Department"],
        answer_type="GROUPED",
    )
    bad_df = pd.DataFrame([{"avg_income": 6500.0}, {"avg_income": 7000.0}])
    is_valid, reason = SQLResultValidator.validate(bad_df, spec, hr_schema)

    assert is_valid is False
    assert "Missing grouping column 'Department'" in reason


def test_pass_valid_grouped_result(hr_schema):
    """Valid grouped DataFrame passes."""
    spec = QueryContract(
        query_id="q3",
        dataset_id="d1",
        original_query="How many employees are there in each department?",
        aggregation="COUNT",
        measure="total_records",
        group_by=["Department"],
        answer_type="GROUPED",
    )
    good_df = pd.DataFrame([
        {"Department": "Sales", "total_records": 446},
        {"Department": "Research & Development", "total_records": 961},
        {"Department": "Human Resources", "total_records": 63},
    ])
    is_valid, reason = SQLResultValidator.validate(good_df, spec, hr_schema)
    assert is_valid is True
    assert reason is None


def test_reject_multiple_rows_when_scalar_expected(hr_schema):
    """If AnalysisSpec expects SCALAR, multiple rows must be rejected."""
    spec = QueryContract(
        query_id="q4",
        dataset_id="d1",
        original_query="What is the average age?",
        aggregation="AVG",
        measure_column="Age",
        answer_type="SCALAR",
        group_by=[],
    )
    bad_df = pd.DataFrame([{"Age": 41}, {"Age": 49}, {"Age": 37}])
    is_valid, reason = SQLResultValidator.validate(bad_df, spec, hr_schema)

    assert is_valid is False
    assert "EXPECTED SCALAR RESULT BUT RECEIVED MULTIPLE ROWS" in reason


def test_reject_scalar_when_distribution_expected(hr_schema):
    """Distribution queries must reject 1-row scalar aggregates."""
    spec = QueryContract(
        query_id="q5",
        dataset_id="d1",
        original_query="Show the distribution of employee ages",
        dimensions=["Age"],
        x_axis="Age",
        measure_column="Age",
        intent="DISTRIBUTION",
        answer_type="DISTRIBUTION",
        suggested_chart_type="histogram",
    )
    bad_df = pd.DataFrame([{"Age": 54279}])  # Premature SUM(Age)
    is_valid, reason = SQLResultValidator.validate(bad_df, spec, hr_schema, dataset_row_count=1470)

    assert is_valid is False
    assert "EXPECTED DISTRIBUTION OBSERVATIONS BUT RECEIVED SCALAR RESULT" in reason


def test_end_to_end_repair_scalar_to_grouped():
    """Test SQLAgent._validate_and_repair_result recovers correct grouped results against live table."""
    agent = SQLAgent()
    spec = QueryContract(
        query_id="q6",
        dataset_id="631ad5c504",
        original_query="How many employees are there in each department?",
        aggregation="COUNT",
        measure="total_records",
        group_by=["Department"],
        answer_type="GROUPED",
    )
    schema = {
        "columns": [
            {"name": "Department", "dtype": "object", "sample_values": ["Sales", "Research & Development"]},
            {"name": "Age", "dtype": "int64", "sample_values": [41, 49]},
        ],
        "row_count": 1470,
    }
    table = "dataset_631ad5c504"

    # Feed agent a bad scalar result
    bad_df = pd.DataFrame([{"total_records": 1470}])
    bad_sql = f'SELECT COUNT(*) AS total_records FROM "{table}"'

    repaired_df, repaired_sql = agent._validate_and_repair_result(bad_df, spec, schema, table, bad_sql)

    assert len(repaired_df) > 1
    assert "Department" in repaired_df.columns
    assert "GROUP BY" in repaired_sql.upper()
