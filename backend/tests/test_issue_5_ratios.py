"""
Test Phase 5: Ratio and Percentage Calculations (Attrition Rate & Overtime Attrition).
Runs:
1. What is the overall attrition rate?
2. What percentage of employees who work overtime have left?

Verifies:
- Attrition rate is NOT interpreted as SUM(*)
- Calculation has numerator (Attrition = Yes) and denominator (all employees)
- Overtime percentage has numerator (OverTime = Yes AND Attrition = Yes) and denominator (OverTime = Yes)
- Deterministic math produces exact verified numbers against dataset_631ad5c504
"""
import os
import sqlite3
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.sql.agent import sql_agent
from app.agents.supervisor.semantic_planner import semantic_planner
from app.analytics.calculation_engine import CalculationEngine

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


def test_issue_5_overall_attrition_rate():
    q = "What is the overall attrition rate?"
    contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q1")

    assert contract.intent == "CALCULATION"
    assert contract.calculation is not None
    assert contract.calculation.operator == "PERCENTAGE"
    assert len(contract.calculation.operands) == 2

    # Numerator: Attrition = Yes
    op_num = contract.calculation.operands[0]
    num_cols = {f.column: f.value for f in op_num.filters}
    assert num_cols.get("Attrition") == "Yes"

    # Denominator: All employees (no filters)
    op_den = contract.calculation.operands[1]
    assert len(op_den.filters) == 0

    # SQL execution
    sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "dataset_631ad5c504")
    assert "SUM(*)" not in sql
    assert "MAX(*)" not in sql

    conn = sqlite3.connect("ai_madac.db")
    row = conn.execute(sql).fetchone()
    conn.close()

    op1, op2 = row[0], row[1]
    assert op1 == 237.0, f"Expected 237 employees who left, got {op1}"
    assert op2 == 1470.0, f"Expected 1470 total employees, got {op2}"

    calc_res = CalculationEngine.compute_scalar(op1, op2, "PERCENTAGE")
    assert round(calc_res.value, 2) == 16.12


def test_issue_5_overtime_employees_left_percentage():
    q = "What percentage of employees who work overtime have left?"
    contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q2")

    assert contract.intent == "CALCULATION"
    assert contract.calculation is not None
    assert contract.calculation.operator == "PERCENTAGE"
    assert len(contract.calculation.operands) == 2

    # Numerator: OverTime = Yes and Attrition = Yes
    op_num = contract.calculation.operands[0]
    num_cols = {f.column: f.value for f in op_num.filters}
    assert num_cols.get("OverTime") == "Yes"
    assert num_cols.get("Attrition") == "Yes"

    # Denominator: OverTime = Yes
    op_den = contract.calculation.operands[1]
    den_cols = {f.column: f.value for f in op_den.filters}
    assert den_cols.get("OverTime") == "Yes"

    # SQL execution
    sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "dataset_631ad5c504")
    conn = sqlite3.connect("ai_madac.db")
    row = conn.execute(sql).fetchone()
    conn.close()

    op1, op2 = row[0], row[1]
    assert op1 == 127.0, f"Expected 127 overtime employees who left, got {op1}"
    assert op2 == 416.0, f"Expected 416 total overtime employees, got {op2}"

    calc_res = CalculationEngine.compute_scalar(op1, op2, "PERCENTAGE")
    assert round(calc_res.value, 2) == 30.53
