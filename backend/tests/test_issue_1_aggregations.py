"""
Test Phase 1: Fix Invalid SQL Aggregations.
Verifies:
1. What is the total number of employees? -> COUNT(*)
2. Which job role has the highest number of employees? -> COUNT(*) GROUP BY JobRole
3. What is the highest monthly income? -> MAX(MonthlyIncome)
4. What is the average monthly income? -> AVG(MonthlyIncome)
5. Pre-execution SQL validation rejects SUM(*), AVG(*), MAX(*), MIN(*).
"""
import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.supervisor.semantic_planner import semantic_planner
from app.agents.sql.agent import sql_agent
from app.utils.sql_validator import validate_sql


IBM_HR_SCHEMA = {
    "columns": [
        {"name": "Age", "dtype": "INTEGER"},
        {"name": "Attrition", "dtype": "TEXT"},
        {"name": "Department", "dtype": "TEXT"},
        {"name": "Gender", "dtype": "TEXT"},
        {"name": "JobRole", "dtype": "TEXT"},
        {"name": "MonthlyIncome", "dtype": "INTEGER"},
        {"name": "OverTime", "dtype": "TEXT"},
        {"name": "YearsAtCompany", "dtype": "INTEGER"},
    ]
}


def test_issue_1_aggregations():
    # 1. Total number of employees
    q1 = "What is the total number of employees?"
    c1 = semantic_planner.plan_query(q1, IBM_HR_SCHEMA, "dataset_ibm_hr", "q1")
    sql1 = sql_agent._build_query_from_contract(c1, IBM_HR_SCHEMA, "ibm_hr")
    assert "COUNT(*)" in sql1.upper()
    assert "SUM(*)" not in sql1.upper()
    is_valid, _ = validate_sql(sql1)
    assert is_valid

    # 2. Which job role has the highest number of employees?
    q2 = "Which job role has the highest number of employees?"
    c2 = semantic_planner.plan_query(q2, IBM_HR_SCHEMA, "dataset_ibm_hr", "q2")
    sql2 = sql_agent._build_query_from_contract(c2, IBM_HR_SCHEMA, "ibm_hr")
    assert "COUNT(*)" in sql2.upper()
    assert "MAX(*)" not in sql2.upper()
    assert '"JobRole"' in sql2
    assert 'GROUP BY "JobRole"' in sql2 or 'GROUP BY "JobRole"' in sql2.upper()
    is_valid, _ = validate_sql(sql2)
    assert is_valid

    # 3. What is the highest monthly income?
    q3 = "What is the highest monthly income?"
    c3 = semantic_planner.plan_query(q3, IBM_HR_SCHEMA, "dataset_ibm_hr", "q3")
    sql3 = sql_agent._build_query_from_contract(c3, IBM_HR_SCHEMA, "ibm_hr")
    assert "MAX" in sql3.upper()
    assert '"MonthlyIncome"' in sql3
    is_valid, _ = validate_sql(sql3)
    assert is_valid

    # 4. What is the average monthly income?
    q4 = "What is the average monthly income?"
    c4 = semantic_planner.plan_query(q4, IBM_HR_SCHEMA, "dataset_ibm_hr", "q4")
    sql4 = sql_agent._build_query_from_contract(c4, IBM_HR_SCHEMA, "ibm_hr")
    assert "AVG" in sql4.upper()
    assert '"MonthlyIncome"' in sql4
    is_valid, _ = validate_sql(sql4)
    assert is_valid

    # 5. Pre-execution SQL validator rejects illegal aggregations
    assert not validate_sql("SELECT SUM(*) FROM table")[0]
    assert not validate_sql("SELECT MAX(*) FROM table")[0]
    assert not validate_sql("SELECT AVG(*) FROM table")[0]
    assert not validate_sql("SELECT MIN(*) FROM table")[0]
    assert validate_sql("SELECT COUNT(*) FROM table")[0]
