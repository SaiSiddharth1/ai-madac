"""
Tests for Scatter Plot and Explicit Chart Type Locking.
Verifies Phase 1 through Phase 10 fixes:
1. Explicit chart type detection and locking (scatter, pie, histogram, bar, line)
2. Scatter parameter locking (x=Age, y=MonthlyIncome) for "relationship between Age and MonthlyIncome in a scatter plot"
3. Elimination of hardcoded "Quantity" and "UnitPrice" leakage
4. Numeric validation enforcement for scatter plots
5. Raw observation SQL generation without GROUP BY
6. Quality gate evaluation without "Quantity" errors
7. Preservation of automatic chart recommendation when no chart type is explicitly requested
"""

import pandas as pd
import pytest

from app.agents.supervisor.semantic_planner import SemanticPlanner
from app.agents.sql.agent import SQLAgent
from app.agents.visualization.contract import extract_query_contract
from app.agents.visualization.spec_validator import SpecValidator
from app.agents.visualization.quality_gate import VisualizationQualityGate
from app.schemas.query_contract import QueryContract
from app.schemas.analysis_spec_validator import AnalysisSpecValidator


@pytest.fixture
def hr_schema():
    return {
        "columns": [
            {"name": "Age", "dtype": "int64", "sample_values": [41, 49, 37]},
            {"name": "Attrition", "dtype": "object", "sample_values": ["Yes", "No"]},
            {"name": "Department", "dtype": "object", "sample_values": ["Sales", "Research & Development", "Human Resources"]},
            {"name": "MonthlyIncome", "dtype": "int64", "sample_values": [5993, 5130, 2090]},
            {"name": "JobRole", "dtype": "object", "sample_values": ["Sales Executive", "Research Scientist"]},
        ],
        "row_count": 1470,
    }


def test_scatter_type_lock_vs(hr_schema):
    """'Show Age vs MonthlyIncome in a scatter plot' locks chart_type to scatter with correct axes."""
    planner = SemanticPlanner()
    contract = planner._heuristic_contract_fallback(
        question="Show Age vs MonthlyIncome in a scatter plot",
        schema=hr_schema,
        dataset_id="test_ds",
        query_id="q1",
    )
    assert contract.suggested_chart_type == "scatter"
    assert contract.intent == "RELATIONSHIP"
    assert contract.x_axis == "Age"
    assert contract.y_axis == "MonthlyIncome"
    assert contract.group_by == []
    assert "Quantity" not in (contract.x_axis, contract.y_axis)


def test_scatter_parameter_lock_relationship_between(hr_schema):
    """'Show the relationship between Age and MonthlyIncome in a scatter plot.' locks x=Age, y=MonthlyIncome."""
    planner = SemanticPlanner()
    contract = planner._heuristic_contract_fallback(
        question="Show the relationship between Age and MonthlyIncome in a scatter plot.",
        schema=hr_schema,
        dataset_id="test_ds",
        query_id="q2",
    )
    assert contract.suggested_chart_type == "scatter"
    assert contract.intent == "RELATIONSHIP"
    assert contract.answer_type == "RELATIONSHIP"
    assert contract.x_axis == "Age"
    assert contract.y_axis == "MonthlyIncome"
    assert contract.group_by == []
    assert "Quantity" not in (contract.x_axis, contract.y_axis)


def test_scatter_of_syntax(hr_schema):
    """'Scatter plot of Age and MonthlyIncome' correctly locks x=Age, y=MonthlyIncome."""
    planner = SemanticPlanner()
    contract = planner._heuristic_contract_fallback(
        question="Scatter plot of Age and MonthlyIncome",
        schema=hr_schema,
        dataset_id="test_ds",
        query_id="q3",
    )
    assert contract.suggested_chart_type == "scatter"
    assert contract.x_axis == "Age"
    assert contract.y_axis == "MonthlyIncome"


def test_no_quantity_leakage_across_sequential_queries(hr_schema):
    """Sequential query execution does not leak parameters or default to Quantity."""
    planner = SemanticPlanner()
    # Query 1: Grouped bar query
    contract_1 = planner._heuristic_contract_fallback(
        question="Average MonthlyIncome by Department",
        schema=hr_schema,
        dataset_id="test_ds",
        query_id="q4_1",
    )
    assert contract_1.suggested_chart_type == "bar"
    assert contract_1.group_by == ["Department"]

    # Query 2: Scatter plot query
    contract_2 = planner._heuristic_contract_fallback(
        question="Show the relationship between Age and MonthlyIncome in a scatter plot.",
        schema=hr_schema,
        dataset_id="test_ds",
        query_id="q4_2",
    )
    assert contract_2.suggested_chart_type == "scatter"
    assert contract_2.x_axis == "Age"
    assert contract_2.y_axis == "MonthlyIncome"
    assert contract_2.group_by == []
    assert contract_2.x_axis != "Quantity"
    assert contract_2.y_axis != "UnitPrice"


def test_reject_categorical_columns_in_scatter(hr_schema):
    """AnalysisSpecValidator rejects non-numeric columns in a scatter plot."""
    spec = QueryContract(
        query_id="q_invalid_scatter",
        dataset_id="test_ds",
        original_query="Show Department vs MonthlyIncome in a scatter plot",
        intent="RELATIONSHIP",
        answer_type="RELATIONSHIP",
        suggested_chart_type="scatter",
        x_axis="Department",
        y_axis="MonthlyIncome",
    )
    is_valid, errors = AnalysisSpecValidator.validate(spec, hr_schema)
    assert is_valid is False
    assert any("must be numeric" in err for err in errors)


def test_accept_numeric_columns_in_scatter(hr_schema):
    """AnalysisSpecValidator passes valid numeric columns for scatter."""
    spec = QueryContract(
        query_id="q_valid_scatter",
        dataset_id="test_ds",
        original_query="Show Age vs MonthlyIncome in a scatter plot",
        intent="RELATIONSHIP",
        answer_type="RELATIONSHIP",
        suggested_chart_type="scatter",
        x_axis="Age",
        y_axis="MonthlyIncome",
    )
    is_valid, errors = AnalysisSpecValidator.validate(spec, hr_schema)
    assert is_valid is True
    assert len(errors) == 0


def test_explicit_pie_lock(hr_schema):
    """Explicit request for pie chart locks suggested_chart_type to pie."""
    planner = SemanticPlanner()
    contract = planner._heuristic_contract_fallback(
        question="Show employee count by department in a pie chart",
        schema=hr_schema,
        dataset_id="test_ds",
        query_id="q5",
    )
    assert contract.suggested_chart_type == "pie"


def test_explicit_histogram_lock(hr_schema):
    """Explicit request for histogram locks suggested_chart_type to histogram."""
    planner = SemanticPlanner()
    contract = planner._heuristic_contract_fallback(
        question="Show distribution of MonthlyIncome in a histogram",
        schema=hr_schema,
        dataset_id="test_ds",
        query_id="q6",
    )
    assert contract.suggested_chart_type == "histogram"


def test_auto_chart_selection_preserved(hr_schema):
    """Questions without explicit chart requests preserve automatic selection."""
    planner = SemanticPlanner()
    contract = planner._heuristic_contract_fallback(
        question="What is the average MonthlyIncome by Department?",
        schema=hr_schema,
        dataset_id="test_ds",
        query_id="q7",
    )
    assert contract.suggested_chart_type == "bar"
    assert contract.group_by == ["Department"]


def test_sql_agent_generates_raw_observations_for_scatter(hr_schema):
    """SQLAgent generates raw row SELECT without GROUP BY and without Quantity."""
    agent = SQLAgent()
    spec = QueryContract(
        query_id="q_sql_scatter",
        dataset_id="test_ds",
        original_query="Show the relationship between Age and MonthlyIncome in a scatter plot.",
        intent="RELATIONSHIP",
        answer_type="RELATIONSHIP",
        suggested_chart_type="scatter",
        x_axis="Age",
        y_axis="MonthlyIncome",
    )
    sql = agent._build_query_from_contract(contract=spec, schema=hr_schema, safe_table="dataset_631ad5c504")
    assert 'GROUP BY' not in sql.upper()
    assert '"Age"' in sql
    assert '"MonthlyIncome"' in sql
    assert 'Quantity' not in sql
    assert 'UnitPrice' not in sql
    assert 'LIMIT 5000' in sql


def test_deterministic_spec_and_quality_gate(hr_schema):
    """Deterministic spec generation and quality gate pass for Age and MonthlyIncome."""
    spec = QueryContract(
        query_id="q_spec_scatter",
        dataset_id="test_ds",
        original_query="Show the relationship between Age and MonthlyIncome in a scatter plot.",
        intent="RELATIONSHIP",
        answer_type="RELATIONSHIP",
        suggested_chart_type="scatter",
        x_axis="Age",
        y_axis="MonthlyIncome",
    )
    avail_cols = [c["name"] for c in hr_schema["columns"]]
    vis_spec = SpecValidator.build_deterministic_spec(spec, avail_cols)
    assert vis_spec["chart"]["type"] == "scatter"
    assert vis_spec["fields"]["x"] == "Age"
    assert vis_spec["fields"]["y"] == "MonthlyIncome"

    # Evaluate quality gate with valid numeric data
    df = pd.DataFrame({
        "Age": [25, 30, 45, 50, 38],
        "MonthlyIncome": [3000, 4500, 8000, 9500, 6000],
    })
    report = VisualizationQualityGate.evaluate(vis_spec, df, spec)
    assert report.passed is True
    assert len(report.critical_errors) == 0
    assert not any("Quantity" in err for err in report.critical_errors)
