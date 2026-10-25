"""
Test Phase 4: Fix Sales False-Positive Dataset Protection.
Runs:
1. How many Sales department employees work overtime?
2. How many employees work overtime?
3. What percentage of employees who work overtime have left the company?
4. Show top 10 products by total sales (still properly rejected on IBM HR)

Verifies:
- 'Sales department' resolves to Department = 'Sales' without false-positive dataset rejection
- 'work overtime' resolves to OverTime = 'Yes'
- Percentage question uses OverTime=Yes AND Attrition=Yes as numerator, OverTime=Yes as denominator
- Unrelated product/sales queries on IBM HR remain protected
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
        {"name": "JobSatisfaction", "dtype": "INTEGER"},
        {"name": "MonthlyIncome", "dtype": "INTEGER"},
        {"name": "OverTime", "dtype": "TEXT", "sample_values": ["Yes", "No"]},
        {"name": "YearsAtCompany", "dtype": "INTEGER"},
    ]
}


def test_issue_4_q1_sales_department_overtime():
    q = "How many Sales department employees work overtime?"
    contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q1")

    # Must be related to dataset!
    assert contract.is_related_to_dataset is True
    assert contract.intent == "AGGREGATION"

    # Filters must contain Department = 'Sales' and OverTime = 'Yes'
    dept_filter = next((f for f in contract.filters if f.column == "Department"), None)
    ot_filter = next((f for f in contract.filters if f.column == "OverTime"), None)
    assert dept_filter is not None, f"Filters: {contract.filters}"
    assert dept_filter.value == "Sales"
    assert ot_filter is not None, f"Filters: {contract.filters}"
    assert ot_filter.value == "Yes"

    sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "dataset_631ad5c504")
    assert 'COUNT(*)' in sql
    assert '"Department" = \'Sales\'' in sql
    assert '"OverTime" = \'Yes\'' in sql

    conn = sqlite3.connect("ai_madac.db")
    count = conn.execute(sql).fetchone()[0]
    conn.close()
    assert count == 128, f"Expected 128 Sales employees with overtime, got {count}"


def test_issue_4_q2_employees_work_overtime():
    q = "How many employees work overtime?"
    contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q2")

    assert contract.is_related_to_dataset is True
    assert contract.intent == "AGGREGATION"

    ot_filter = next((f for f in contract.filters if f.column == "OverTime"), None)
    assert ot_filter is not None, f"Filters: {contract.filters}"
    assert ot_filter.value == "Yes"

    sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "dataset_631ad5c504")
    assert 'COUNT(*)' in sql
    assert '"OverTime" = \'Yes\'' in sql

    conn = sqlite3.connect("ai_madac.db")
    count = conn.execute(sql).fetchone()[0]
    conn.close()
    assert count == 416, f"Expected 416 overtime employees, got {count}"


def test_issue_4_q3_overtime_attrition_percentage():
    q = "What percentage of employees who work overtime have left the company?"
    contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q3")

    assert contract.is_related_to_dataset is True
    assert contract.intent == "CALCULATION"
    assert contract.calculation is not None
    assert contract.calculation.operator == "PERCENTAGE"
    assert len(contract.calculation.operands) == 2

    # Numerator operand: OverTime = Yes and Attrition = Yes
    op_num = contract.calculation.operands[0]
    num_cols = {f.column: f.value for f in op_num.filters}
    assert num_cols.get("OverTime") == "Yes"
    assert num_cols.get("Attrition") == "Yes"

    # Denominator operand: OverTime = Yes
    op_den = contract.calculation.operands[1]
    den_cols = {f.column: f.value for f in op_den.filters}
    assert den_cols.get("OverTime") == "Yes"

    # SQL execution
    sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "dataset_631ad5c504")
    conn = sqlite3.connect("ai_madac.db")
    row = conn.execute(sql).fetchone()
    conn.close()

    op1, op2 = row[0], row[1]
    assert op1 == 127.0, f"Expected numerator 127, got {op1}"
    assert op2 == 416.0, f"Expected denominator 416, got {op2}"

    calc_res = CalculationEngine.compute_scalar(op1, op2, "PERCENTAGE")
    assert round(calc_res.value, 2) == 30.53


def test_issue_4_q4_superstore_query_still_rejected_on_ibm_hr():
    q = "Show top 10 products by total sales."
    contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q4")
    assert contract.is_related_to_dataset is False or contract.intent == "UNRELATED"
