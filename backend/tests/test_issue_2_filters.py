"""
Test Phase 2: Semantic Filter Resolution ("left the company", "not left the company", "attrition count").
Runs:
1. How many employees have left the company?
2. How many employees have not left the company?
3. What is the attrition count?
4. What is the attrition count by department?

Verifies:
- Attrition filter is correctly resolved (Yes/No)
- SQL generates COUNT(*) with WHERE "Attrition" = 'Yes' / 'No'
- For question 4: Department + COUNT(*) + Attrition = Yes + GROUP BY Department
- Result is NOT total employee counts by department
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
        {"name": "MonthlyIncome", "dtype": "INTEGER"},
        {"name": "OverTime", "dtype": "TEXT", "sample_values": ["Yes", "No"]},
        {"name": "YearsAtCompany", "dtype": "INTEGER"},
    ]
}


def test_issue_2_left_the_company():
    q = "How many employees have left the company?"
    contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q1")
    assert contract.aggregation == "COUNT"
    assert any(f.column == "Attrition" and f.value == "Yes" for f in contract.filters), f"Filters: {contract.filters}"
    assert contract.group_by == []

    sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "dataset_631ad5c504")
    assert 'COUNT(*)' in sql
    assert '"Attrition" = \'Yes\'' in sql
    assert 'GROUP BY' not in sql

    # Execute against database
    conn = sqlite3.connect("ai_madac.db")
    cursor = conn.cursor()
    count = cursor.execute(sql).fetchone()[0]
    conn.close()
    assert count == 237, f"Expected 237 employees who left, got {count}"


def test_issue_2_not_left_the_company():
    q = "How many employees have not left the company?"
    contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q2")
    assert contract.aggregation == "COUNT"
    assert any(f.column == "Attrition" and f.value == "No" for f in contract.filters), f"Filters: {contract.filters}"
    assert contract.group_by == []

    sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "dataset_631ad5c504")
    assert 'COUNT(*)' in sql
    assert '"Attrition" = \'No\'' in sql
    assert 'GROUP BY' not in sql

    # Execute against database
    conn = sqlite3.connect("ai_madac.db")
    cursor = conn.cursor()
    count = cursor.execute(sql).fetchone()[0]
    conn.close()
    assert count == 1233, f"Expected 1233 employees who have not left, got {count}"


def test_issue_2_attrition_count():
    q = "What is the attrition count?"
    contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q3")
    assert contract.aggregation == "COUNT"
    assert any(f.column == "Attrition" and f.value == "Yes" for f in contract.filters), f"Filters: {contract.filters}"
    assert contract.group_by == []

    sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "dataset_631ad5c504")
    assert 'COUNT(*)' in sql
    assert '"Attrition" = \'Yes\'' in sql

    # Execute against database
    conn = sqlite3.connect("ai_madac.db")
    cursor = conn.cursor()
    count = cursor.execute(sql).fetchone()[0]
    conn.close()
    assert count == 237, f"Expected 237 attrition count, got {count}"


def test_issue_2_attrition_count_by_department():
    q = "What is the attrition count by department?"
    contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q4")
    assert "Department" in contract.group_by
    assert contract.aggregation == "COUNT"
    assert any(f.column == "Attrition" and f.value == "Yes" for f in contract.filters), f"Filters: {contract.filters}"

    sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "dataset_631ad5c504")
    assert 'COUNT(*)' in sql
    assert '"Attrition" = \'Yes\'' in sql
    assert 'GROUP BY "Department"' in sql

    # Execute against database
    conn = sqlite3.connect("ai_madac.db")
    cursor = conn.cursor()
    rows = cursor.execute(sql).fetchall()
    conn.close()

    dept_counts = dict(rows)
    assert dept_counts.get("Human Resources") == 12
    assert dept_counts.get("Research & Development") == 133
    assert dept_counts.get("Sales") == 92
    total = sum(dept_counts.values())
    assert total == 237, f"Expected total 237 attrition across departments, got {total}"
    assert total != 1470, "Should not return total employee count"
