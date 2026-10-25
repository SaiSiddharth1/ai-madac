"""
Test Phase 3: Semantic Grouping / "each" / "by" Resolution.
Runs:
1. How many employees are there in each department?
2. Show the number of employees in each department.
3. Show the average monthly income by age.
4. Show the average years at the company for each age.
5. Show the average job satisfaction by department.

Verifies:
- Grouping dimensions are preserved through semantic planner into group_by
- SQL Agent preserves group_by without stripping it
- Each query produces grouped rows from dataset_631ad5c504
"""
import os
import sqlite3
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.sql.agent import sql_agent
from app.agents.supervisor.semantic_planner import semantic_planner

IBM_HR_SCHEMA = {
    "columns": [
        {"name": "Age", "dtype": "INTEGER"},
        {"name": "Attrition", "dtype": "TEXT", "sample_values": ["Yes", "No"]},
        {"name": "Department", "dtype": "TEXT", "sample_values": ["Sales", "Research & Development", "Human Resources"]},
        {"name": "Gender", "dtype": "TEXT", "sample_values": ["Male", "Female"]},
        {"name": "JobRole", "dtype": "TEXT"},
        {"name": "JobSatisfaction", "dtype": "INTEGER"},
        {"name": "MonthlyIncome", "dtype": "INTEGER"},
        {"name": "OverTime", "dtype": "TEXT", "sample_values": ["Yes", "No"]},
        {"name": "YearsAtCompany", "dtype": "INTEGER"},
    ]
}


def test_issue_3_q1_employees_in_each_department():
    q = "How many employees are there in each department?"
    contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q1")
    assert "Department" in contract.group_by
    assert contract.aggregation == "COUNT"
    assert contract.answer_type == "GROUPED"

    sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "dataset_631ad5c504")
    assert 'COUNT(*)' in sql
    assert 'GROUP BY "Department"' in sql

    conn = sqlite3.connect("ai_madac.db")
    rows = conn.execute(sql).fetchall()
    conn.close()

    assert len(rows) == 3, f"Expected 3 departments, got {len(rows)}"
    counts = dict(rows)
    assert counts.get("Human Resources") == 63
    assert counts.get("Research & Development") == 961
    assert counts.get("Sales") == 446
    assert sum(counts.values()) == 1470


def test_issue_3_q2_number_of_employees_in_each_department():
    q = "Show the number of employees in each department."
    contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q2")
    assert "Department" in contract.group_by
    assert contract.aggregation == "COUNT"
    assert contract.answer_type == "GROUPED"

    sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "dataset_631ad5c504")
    assert 'COUNT(*)' in sql
    assert 'GROUP BY "Department"' in sql

    conn = sqlite3.connect("ai_madac.db")
    rows = conn.execute(sql).fetchall()
    conn.close()

    assert len(rows) == 3, f"Expected 3 departments, got {len(rows)}"


def test_issue_3_q3_average_monthly_income_by_age():
    q = "Show the average monthly income by age."
    contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q3")
    assert "Age" in contract.group_by
    assert contract.measure_column == "MonthlyIncome"
    assert contract.aggregation == "AVG"
    assert contract.answer_type == "GROUPED"

    sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "dataset_631ad5c504")
    assert 'AVG("MonthlyIncome")' in sql
    assert 'GROUP BY "Age"' in sql

    conn = sqlite3.connect("ai_madac.db")
    rows = conn.execute(sql).fetchall()
    conn.close()

    # Ages range from 18 to 60 -> 43 distinct age groups
    assert len(rows) > 30, f"Expected grouped rows across ages, got {len(rows)}"
    assert rows[0][0] is not None
    assert rows[0][1] is not None


def test_issue_3_q4_average_years_at_company_for_each_age():
    q = "Show the average years at the company for each age."
    contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q4")
    assert "Age" in contract.group_by
    assert contract.measure_column == "YearsAtCompany"
    assert contract.aggregation == "AVG"
    assert contract.answer_type == "GROUPED"

    sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "dataset_631ad5c504")
    assert 'AVG("YearsAtCompany")' in sql
    assert 'GROUP BY "Age"' in sql

    conn = sqlite3.connect("ai_madac.db")
    rows = conn.execute(sql).fetchall()
    conn.close()

    assert len(rows) > 30, f"Expected grouped rows across ages, got {len(rows)}"


def test_issue_3_q5_average_job_satisfaction_by_department():
    q = "Show the average job satisfaction by department."
    contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q5")
    assert "Department" in contract.group_by
    assert contract.measure_column == "JobSatisfaction"
    assert contract.aggregation == "AVG"
    assert contract.answer_type == "GROUPED"

    sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "dataset_631ad5c504")
    assert 'AVG("JobSatisfaction")' in sql
    assert 'GROUP BY "Department"' in sql

    conn = sqlite3.connect("ai_madac.db")
    rows = conn.execute(sql).fetchall()
    conn.close()

    assert len(rows) == 3, f"Expected 3 departments, got {len(rows)}"
    # Values between 1 and 4
    for dept, avg_sat in rows:
        assert 1.0 <= avg_sat <= 4.0
