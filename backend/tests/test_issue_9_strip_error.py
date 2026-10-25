"""
Test Phase 9: Fix the Python .strip() Error.

Verifies:
1. "Compare the average job satisfaction across departments for each age."
   - No AttributeError: 'property' object has no attribute 'strip'
   - No TypeError: unhashable type: 'list'
   - Semantic structure:
     X = Age
     Y = AVG(JobSatisfaction)
     Series = Department
     group_by = ['Age', 'Department']
     chart_type = multi_line
2. End-to-end visualization execution succeeds without exception.
"""
import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.supervisor.semantic_planner import semantic_planner
from app.agents.sql.agent import sql_agent
from app.agents.visualization.agent import visualization_agent
from app.agents.visualization.plotly_renderer import PlotlyRenderer
from app.agents.visualization.spec_validator import SpecValidator
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


def test_issue_9_job_satisfaction_multiseries_and_no_strip_error():
    q = "Compare the average job satisfaction across departments for each age."
    c = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_ibm_hr", "q9_1")

    # Semantic contract assertions
    assert "Age" in c.group_by
    assert "Department" in c.group_by
    assert c.series_dimension == "Department"
    assert c.x_axis == "Age"
    assert c.measure_column == "JobSatisfaction"
    assert c.aggregation == "AVG"
    assert c.suggested_chart_type == "multi_line"

    # SQL generation and validation
    sql = sql_agent._build_query_from_contract(c, IBM_HR_SCHEMA, "dataset_631ad5c504")
    assert '"Age"' in sql
    assert '"Department"' in sql
    assert 'AVG(' in sql.upper()
    assert 'GROUP BY' in sql.upper()
    is_valid, _ = validate_sql(sql)
    assert is_valid

    # Run SQL agent and visualization agent to ensure NO .strip() or TypeError unhashable list
    state = {
        "question": q,
        "table_name": "dataset_631ad5c504",
        "schema": IBM_HR_SCHEMA,
        "query_contract": c.to_dict(),
    }
    sql_res = sql_agent.run(state)
    assert sql_res.get("sql_result") is not None
    assert sql_res["sql_result"]["row_count"] > 0

    state["sql_result"] = sql_res.get("sql_result")
    vis_res = visualization_agent.run(state)
    assert vis_res.get("visualization_result") is not None
    assert bool(vis_res["visualization_result"]["visualization_available"]) is True
    assert vis_res["visualization_result"]["chart_paths"]
