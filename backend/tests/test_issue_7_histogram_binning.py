import os
import sys
import sqlite3
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.visualization.spec_validator import SpecValidator
from app.agents.visualization.plotly_renderer import PlotlyRenderer
from app.agents.sql.agent import sql_agent
from app.agents.supervisor.semantic_planner import semantic_planner
from app.schemas.query_contract import QueryContract

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


def test_histogram_spec_structure_auto_binning():
    """
    Verify visualization specification supports:
    {
      "chart_type": "HISTOGRAM",
      "dimension": "Age",
      "binning": {
        "enabled": true,
        "method": "AUTO"
      }
    }
    """
    contract = QueryContract(
        query_id="q_hist_auto",
        dataset_id="dataset_631ad5c504",
        original_query="Show the distribution of employee ages.",
        intent="DISTRIBUTION",
        answer_type="DISTRIBUTION",
        measure="Age",
        measure_column="Age",
        x_axis="Age",
        suggested_chart_type="histogram",
        should_visualize=True,
    )

    spec = SpecValidator.build_deterministic_spec(contract, ["Age", "MonthlyIncome"])
    assert spec["chart_type"] == "HISTOGRAM"
    assert spec["dimension"] == "Age"
    assert isinstance(spec["binning"], dict)
    assert spec["binning"]["enabled"] is True
    assert spec["binning"]["method"] == "AUTO"
    assert spec["chart"]["type"] == "histogram"

    # Also test normalize_spec directly on canonical input
    canonical_input = {
        "chart_type": "HISTOGRAM",
        "dimension": "Age",
        "binning": {
            "enabled": True,
            "method": "AUTO",
        },
    }
    norm = SpecValidator.normalize_spec(canonical_input)
    assert norm["chart_type"] == "HISTOGRAM"
    assert norm["dimension"] == "Age"
    assert norm["fields"]["x"] == "Age"
    assert norm["chart"]["type"] == "histogram"
    assert norm["binning"]["enabled"] is True
    assert norm["binning"]["method"] == "AUTO"


def test_histogram_spec_structure_fixed_binning():
    """
    Verify explicit bin width creates fixed binning spec.
    """
    contract = QueryContract(
        query_id="q_hist_fixed",
        dataset_id="dataset_631ad5c504",
        original_query="Show the distribution of employee ages in bins of 5.",
        intent="DISTRIBUTION",
        answer_type="DISTRIBUTION",
        measure="Age",
        measure_column="Age",
        x_axis="Age",
        bin_width=5.0,
        suggested_chart_type="histogram",
        should_visualize=True,
    )

    spec = SpecValidator.build_deterministic_spec(contract, ["Age", "MonthlyIncome"])
    assert spec["chart_type"] == "HISTOGRAM"
    assert spec["dimension"] == "Age"
    assert spec["binning"]["enabled"] is True
    assert spec["binning"]["method"] == "FIXED"
    assert spec["binning"]["size"] == 5.0


def test_histogram_rendering_raw_numeric_observations():
    """
    Verify PlotlyRenderer preserves actual continuous data, detects min/max,
    chooses reasonable bin behavior, and does not aggregate using SUM/AVG.
    """
    np.random.seed(42)
    ages = np.random.randint(18, 65, size=200).tolist()
    df = pd.DataFrame({"Age": ages})

    spec = {
        "chart_type": "HISTOGRAM",
        "dimension": "Age",
        "fields": {"x": "Age", "y": None},
        "chart": {"type": "histogram"},
        "binning": {"enabled": True, "method": "AUTO"},
        "title": "Age Distribution",
    }

    fig, title, chart_path, chart_json_path = PlotlyRenderer.render(df, spec, "Age Distribution")
    assert fig is not None
    # Verify trace is histogram
    assert len(fig.data) >= 1
    trace = fig.data[0]
    assert trace.type == "histogram"
    # Verify raw observations are preserved (length matches raw data)
    assert len(trace.x) == 200
    # Verify xbins are configured
    xbins = trace.xbins
    assert xbins is not None
    assert xbins.size is not None and xbins.size > 0


def test_live_dataset_age_and_income_histograms():
    """
    End-to-end integration: Run Age and MonthlyIncome distribution on live IBM HR dataset.
    Verify:
    1. SQL query retrieves raw observations without aggregation.
    2. SQL result has 1470 raw observations.
    3. Plotly figure renders cleanly with automatic binning.
    """
    table = get_live_table()

    for col in ("Age", "MonthlyIncome"):
        q = f"Show the distribution of {col}."
        contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_631ad5c504", f"q_{col.lower()}")

        assert contract.answer_type == "DISTRIBUTION"
        assert contract.aggregation is None
        assert contract.group_by == []

        # Execute SQL
        state = {
            "question": q,
            "query_contract": contract.model_dump(),
            "schema": IBM_HR_SCHEMA,
            "dataset_id": "dataset_631ad5c504",
            "table_name": table,
        }
        res = sql_agent.run(state)
        assert res["sql_result"] is not None
        sql = res["sql_query"]
        assert "SUM(" not in sql.upper()
        assert "AVG(" not in sql.upper()
        assert "GROUP BY" not in sql.upper()

        assert res["sql_result"]["row_count"] == 1470, f"Expected 1470 observations, got {res['sql_result']['row_count']}"
        rows = res["sql_result"]["rows"]
        assert len(rows) >= 1000

        # Render with PlotlyRenderer
        df = pd.DataFrame(rows)
        spec = SpecValidator.build_deterministic_spec(contract, [col])
        fig, chart_title, _, _ = PlotlyRenderer.render(df, spec, q)
        assert fig is not None
        assert len(fig.data) >= 1
        assert fig.data[0].type == "histogram"
        assert len(fig.data[0].x) == len(rows)
