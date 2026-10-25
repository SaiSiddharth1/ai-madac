"""
Test Phase 8: Multi-Series / Multi-Line Questions.

Verifies:
1. Compare average monthly income for employees who work overtime and those who do not across age.
   - dimension: Age
   - measure: MonthlyIncome
   - aggregation: AVG
   - group_by: ['Age', 'OverTime']
   - series_dimension: OverTime
   - suggested_chart_type: multi_line

2. Compare attrition rates for male and female employees across age.
   - dimension: Age
   - group_by: ['Age', 'Gender']
   - series_dimension: Gender
   - suggested_chart_type: multi_line
   - question_type: CALCULATION / RATIO with numerator/denominator

3. Compare average monthly income across departments for each age.
   - dimension: Age
   - measure: MonthlyIncome
   - aggregation: AVG
   - group_by: ['Age', 'Department']
   - series_dimension: Department
   - suggested_chart_type: multi_line
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
        {"name": "JobSatisfaction", "dtype": "INTEGER"},
        {"name": "MonthlyIncome", "dtype": "INTEGER"},
        {"name": "OverTime", "dtype": "TEXT"},
        {"name": "YearsAtCompany", "dtype": "INTEGER"},
    ]
}


def test_issue_8_multiseries_overtime_across_age():
    q = "Compare average monthly income for employees who work overtime and those who do not across age."
    c = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q8_1")

    assert "Age" in c.group_by
    assert "OverTime" in c.group_by
    assert c.series_dimension == "OverTime"
    assert c.x_axis == "Age"
    assert c.suggested_chart_type == "multi_line"
    assert c.measure_column == "MonthlyIncome"
    assert c.aggregation == "AVG"

    sql = sql_agent._build_query_from_contract(c, IBM_HR_SCHEMA, "dataset_631ad5c504")
    assert '"Age"' in sql
    assert '"OverTime"' in sql
    assert 'AVG(' in sql.upper()
    assert 'GROUP BY' in sql.upper()
    is_valid, _ = validate_sql(sql)
    assert is_valid


def test_issue_8_multiseries_attrition_rates_across_age():
    q = "Compare attrition rates for male and female employees across age."
    c = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q8_2")

    assert "Age" in c.group_by
    assert "Gender" in c.group_by
    assert c.series_dimension == "Gender"
    assert c.x_axis == "Age"
    assert c.suggested_chart_type == "multi_line"
    assert c.intent == "CALCULATION"
    assert c.calculation is not None

    sql = sql_agent._build_query_from_contract(c, IBM_HR_SCHEMA, "dataset_631ad5c504")
    assert '"Age"' in sql
    assert '"Gender"' in sql
    assert 'GROUP BY' in sql.upper()
    is_valid, _ = validate_sql(sql)
    assert is_valid


def test_issue_8_multiseries_monthly_income_across_departments_for_each_age():
    q = "Compare average monthly income across departments for each age."
    c = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q8_3")

    assert "Age" in c.group_by
    assert "Department" in c.group_by
    assert c.series_dimension == "Department"
    assert c.x_axis == "Age"
    assert c.suggested_chart_type == "multi_line"
    assert c.measure_column == "MonthlyIncome"
    assert c.aggregation == "AVG"

    sql = sql_agent._build_query_from_contract(c, IBM_HR_SCHEMA, "dataset_631ad5c504")
    assert '"Age"' in sql
    assert '"Department"' in sql
    assert 'AVG(' in sql.upper()
    assert 'GROUP BY' in sql.upper()
    is_valid, _ = validate_sql(sql)
    assert is_valid
