import os
import sys
import sqlite3
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.sql.agent import sql_agent
from app.agents.supervisor.semantic_planner import semantic_planner

IBM_HR_SCHEMA = {
    "columns": [
        {"name": "Age", "dtype": "INTEGER"},
        {"name": "Attrition", "dtype": "TEXT", "sample_values": ["Yes", "No"]},
        {"name": "Department", "dtype": "TEXT", "sample_values": ["Sales", "Research & Development", "Human Resources"]},
        {"name": "Gender", "dtype": "TEXT", "sample_values": ["Male", "Female"], "unique_count": 2},
        {"name": "JobRole", "dtype": "TEXT", "unique_count": 9},
        {"name": "MonthlyIncome", "dtype": "INTEGER"},
        {"name": "OverTime", "dtype": "TEXT", "sample_values": ["Yes", "No"]},
        {"name": "YearsAtCompany", "dtype": "INTEGER"},
    ]
}


def get_live_table():
    conn = sqlite3.connect("ai_madac.db")
    c = conn.cursor()
    c.execute("SELECT table_name FROM datasets WHERE table_name LIKE '%631ad5c504%'")
    res = c.fetchone()
    conn.close()
    assert res is not None
    return res[0]


def test_distribution_by_gender_is_categorical():
    """
    Q1: 'Show the distribution of employees by gender.'
    Gender is categorical: must produce COUNT(*) GROUP BY Gender with Bar/Pie chart, NOT Histogram.
    """
    table = get_live_table()
    question = "Show the distribution of employees by gender."
    contract = semantic_planner.plan_query(
        question=question,
        schema=IBM_HR_SCHEMA,
        dataset_id="dataset_631ad5c504",
        query_id="test_dist_gender",
    )
    assert contract.answer_type == "GROUPED", f"Expected GROUPED answer_type, got {contract.answer_type}"
    assert "Gender" in contract.group_by, f"Expected Gender in group_by, got {contract.group_by}"
    assert contract.aggregation == "COUNT", f"Expected COUNT aggregation, got {contract.aggregation}"
    assert contract.suggested_chart_type in ("bar", "pie"), f"Expected Bar/Pie chart, got {contract.suggested_chart_type}"

    # Verify SQL execution
    state = {
        "question": question,
        "query_contract": contract.model_dump(),
        "schema": IBM_HR_SCHEMA,
        "dataset_id": "dataset_631ad5c504",
        "table_name": table,
    }
    result = sql_agent.run(state)
    assert result["sql_result"] is not None
    sql = result["sql_query"]
    assert "GROUP BY" in sql.upper(), f"Expected GROUP BY in SQL: {sql}"
    assert "Gender" in sql, f"Expected Gender in SQL: {sql}"
    assert "HISTOGRAM" not in sql.upper()

    rows = result["sql_result"]["rows"]
    assert len(rows) == 2, f"Expected 2 rows (Female, Male), got {len(rows)}"
    genders = {r.get("Gender") for r in rows}
    assert genders == {"Female", "Male"}, f"Expected Female and Male, got {genders}"


def test_distribution_of_ages_is_numeric_histogram():
    """
    Q2: 'Show the distribution of employee ages.'
    Age is continuous numeric: must produce DISTRIBUTION intent, Histogram chart, and raw observations.
    """
    table = get_live_table()
    question = "Show the distribution of employee ages."
    contract = semantic_planner.plan_query(
        question=question,
        schema=IBM_HR_SCHEMA,
        dataset_id="dataset_631ad5c504",
        query_id="test_dist_age",
    )
    assert contract.answer_type == "DISTRIBUTION", f"Expected DISTRIBUTION answer_type, got {contract.answer_type}"
    assert contract.suggested_chart_type == "histogram", f"Expected histogram chart, got {contract.suggested_chart_type}"
    assert contract.measure == "Age" or contract.measure_column == "Age"
    assert contract.group_by == [], f"Expected empty group_by for raw observations, got {contract.group_by}"

    # Verify SQL execution retrieves raw Age observations
    state = {
        "question": question,
        "query_contract": contract.model_dump(),
        "schema": IBM_HR_SCHEMA,
        "dataset_id": "dataset_631ad5c504",
        "table_name": table,
    }
    result = sql_agent.run(state)
    assert result["sql_result"] is not None
    sql = result["sql_query"]
    assert "SUM" not in sql.upper(), f"SQL must NOT calculate SUM(Age): {sql}"

    rows = result["sql_result"]["rows"]
    assert len(rows) > 100, f"Expected raw observations, got {len(rows)} rows"
    first_age = rows[0].get("Age")
    assert isinstance(first_age, (int, float)) and 18 <= first_age <= 65


def test_distribution_of_monthly_income_is_numeric_histogram():
    """
    Q3: 'Show the distribution of monthly income.'
    MonthlyIncome is continuous numeric: must produce DISTRIBUTION intent, Histogram chart, and raw observations.
    """
    table = get_live_table()
    question = "Show the distribution of monthly income."
    contract = semantic_planner.plan_query(
        question=question,
        schema=IBM_HR_SCHEMA,
        dataset_id="dataset_631ad5c504",
        query_id="test_dist_income",
    )
    assert contract.answer_type == "DISTRIBUTION", f"Expected DISTRIBUTION answer_type, got {contract.answer_type}"
    assert contract.suggested_chart_type == "histogram", f"Expected histogram chart, got {contract.suggested_chart_type}"
    assert contract.measure == "MonthlyIncome" or contract.measure_column == "MonthlyIncome"
    assert contract.group_by == [], f"Expected empty group_by for raw observations, got {contract.group_by}"

    # Verify SQL execution
    state = {
        "question": question,
        "query_contract": contract.model_dump(),
        "schema": IBM_HR_SCHEMA,
        "dataset_id": "dataset_631ad5c504",
        "table_name": table,
    }
    result = sql_agent.run(state)
    assert result["sql_result"] is not None
    sql = result["sql_query"]
    assert "SUM" not in sql.upper(), f"SQL must NOT calculate SUM(MonthlyIncome): {sql}"

    rows = result["sql_result"]["rows"]
    assert len(rows) > 100, f"Expected raw observations, got {len(rows)} rows"
    first_income = rows[0].get("MonthlyIncome")
    assert isinstance(first_income, (int, float)) and first_income > 0


def test_distribution_of_job_roles_is_categorical():
    """
    Q4: 'Show the distribution of employees by job role.' and 'Show the distribution of job roles.'
    JobRole is categorical: must produce COUNT(*) GROUP BY JobRole with Bar/Pie chart, NOT Histogram.
    """
    table = get_live_table()
    for question in ("Show the distribution of employees by job role.", "Show the distribution of job roles."):
        contract = semantic_planner.plan_query(
            question=question,
            schema=IBM_HR_SCHEMA,
            dataset_id="dataset_631ad5c504",
            query_id="test_dist_role",
        )
        assert contract.answer_type == "GROUPED", f"Expected GROUPED answer_type for '{question}', got {contract.answer_type}"
        assert "JobRole" in contract.group_by, f"Expected JobRole in group_by for '{question}', got {contract.group_by}"
        assert contract.aggregation == "COUNT", f"Expected COUNT aggregation for '{question}', got {contract.aggregation}"
        assert contract.suggested_chart_type in ("bar", "pie"), f"Expected Bar/Pie chart for '{question}', got {contract.suggested_chart_type}"

        # Verify SQL execution
        state = {
            "question": question,
            "query_contract": contract.model_dump(),
            "schema": IBM_HR_SCHEMA,
            "dataset_id": "dataset_631ad5c504",
            "table_name": table,
        }
        result = sql_agent.run(state)
        assert result["sql_result"] is not None
        sql = result["sql_query"]
        assert "GROUP BY" in sql.upper(), f"Expected GROUP BY in SQL: {sql}"
        assert "JobRole" in sql

        rows = result["sql_result"]["rows"]
        assert len(rows) == 9, f"Expected 9 job roles in IBM HR, got {len(rows)}"
