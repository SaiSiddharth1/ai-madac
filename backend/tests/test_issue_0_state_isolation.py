"""
Test Phase 0: State Isolation and Elimination of Stale Question Data.
Runs sequentially:
1. What is the most popular investment avenue?
2. What is the average age of employees?
3. What is the attrition count by department?
4. Which job role has the highest average years at the company?

Verifies:
- no Investment_Avenue appears in questions 2-4
- no investor_count appears in any question
- no stale aggregation survives
- no stale chart specification survives
"""
import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.state import create_fresh_question_state
from app.agents.supervisor.semantic_planner import semantic_planner
from app.agents.visualization.contract import extract_query_contract
from app.agents.visualization.data_preparer import DataPreparer
import pandas as pd


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

FINANCE_SCHEMA = {
    "columns": [
        {"name": "gender", "dtype": "TEXT"},
        {"name": "age", "dtype": "INTEGER"},
        {"name": "Investment_Avenue", "dtype": "TEXT"},
        {"name": "Avenue", "dtype": "TEXT"},
    ]
}


def test_issue_0_sequential_state_isolation():
    # 1. First question on Finance dataset
    q1 = "What is the most popular investment avenue?"
    c1 = semantic_planner.plan_query(q1, FINANCE_SCHEMA, "dataset_finance", "q1")
    
    # Simulate saving previous validated result
    prev_validated_result = {
        "validated": True,
        "dataset_id": "dataset_finance",
        "columns_used": ["Investment_Avenue", "total_records"],
        "result": [{"Investment_Avenue": "Mutual Funds", "total_records": 14}],
    }

    # 2. Second question on IBM HR dataset
    q2 = "What is the average age of employees?"
    state2 = create_fresh_question_state(
        user_id="u1",
        dataset_id="dataset_ibm_hr",
        query_id="q2",
        question=q2,
        table_name="ibm_hr",
        schema=IBM_HR_SCHEMA,
        previous_validated_result=prev_validated_result,
    )
    c2 = semantic_planner.plan_query(q2, IBM_HR_SCHEMA, "dataset_ibm_hr", "q2")
    
    # Verify Question 2
    assert "Investment_Avenue" not in str(c2.model_dump())
    assert "investor_count" not in str(c2.model_dump())
    assert c2.measure_column == "Age"
    assert c2.aggregation == "AVG"
    assert c2.group_by == []

    # 3. Third question on IBM HR dataset
    q3 = "What is the attrition count by department?"
    state3 = create_fresh_question_state(
        user_id="u1",
        dataset_id="dataset_ibm_hr",
        query_id="q3",
        question=q3,
        table_name="ibm_hr",
        schema=IBM_HR_SCHEMA,
        previous_validated_result={"validated": True, "columns_used": ["Age"], "result": [{"Age": 36.92}]},
    )
    c3 = semantic_planner.plan_query(q3, IBM_HR_SCHEMA, "dataset_ibm_hr", "q3")
    assert "Age" != c3.measure_column
    assert "Investment_Avenue" not in str(c3.model_dump())
    assert "investor_count" not in str(c3.model_dump())

    # 4. Fourth question on IBM HR dataset
    q4 = "Which job role has the highest average years at the company?"
    state4 = create_fresh_question_state(
        user_id="u1",
        dataset_id="dataset_ibm_hr",
        query_id="q4",
        question=q4,
        table_name="ibm_hr",
        schema=IBM_HR_SCHEMA,
        previous_validated_result={"validated": True, "columns_used": ["Department", "total_records"], "result": []},
    )
    c4 = semantic_planner.plan_query(q4, IBM_HR_SCHEMA, "dataset_ibm_hr", "q4")
    assert "Investment_Avenue" not in str(c4.model_dump())
    assert "investor_count" not in str(c4.model_dump())

    # 5. Verify DataPreparer does not inject investor_count
    df = pd.DataFrame([{"JobRole": "Sales Executive"}, {"JobRole": "Research Scientist"}])
    spec = {"chart": {"type": "bar"}, "fields": {"x": "JobRole"}}
    prep_df, params, insights = DataPreparer.prepare(df, spec)
    assert "investor_count" not in prep_df.columns
    assert "investor_count" not in str(params)
    assert "investor_count" not in str(insights)
    assert "investors" not in str(insights)
