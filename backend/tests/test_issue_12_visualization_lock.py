"""
Tests for Issue 12: Visualization Must Not Reinterpret the Question.
Verifies that Visualization Agent and SpecValidator strictly lock presentation
to the canonical AnalysisSpec (QueryContract):
- Cannot alter measure (e.g. MonthlyIncome -> Age)
- Cannot alter aggregation (e.g. AVG -> MAX)
- Cannot alter dimension / X-axis (e.g. Age -> Department)
- Cannot alter series dimension (e.g. Department -> Gender)
- Fallback builds an exact specification locked to the AnalysisSpec
"""

import pytest

from app.schemas.query_contract import QueryContract, AnalysisSpec
from app.agents.visualization.spec_validator import SpecValidator


@pytest.fixture
def available_columns():
    return ["Age", "Attrition", "Department", "Gender", "JobRole", "MonthlyIncome", "OverTime"]


def test_reject_altering_measure(available_columns):
    """If AnalysisSpec requests MonthlyIncome, candidate spec cannot switch to Age."""
    contract = QueryContract(
        query_id="q1",
        dataset_id="d1",
        original_query="Show average monthly income by age",
        requested_fields=["Age", "MonthlyIncome"],
        dimensions=["Age"],
        measure_column="MonthlyIncome",
        aggregation="AVG",
        suggested_chart_type="line",
    )
    contract.x_axis = "Age"
    contract.y_axis = "MonthlyIncome"

    # Gemini incorrectly tries to plot MAX(Age)
    bad_spec = {
        "chart": {"type": "line"},
        "fields": {"x": "Age", "y": "Age"},
        "measure": {"type": "max", "column": "Age"},
    }
    is_valid, reason = SpecValidator.validate(bad_spec, contract, available_columns)
    assert is_valid is False
    assert "MonthlyIncome" in reason


def test_reject_altering_aggregation(available_columns):
    """If AnalysisSpec requests AVG, candidate spec cannot switch to MAX or SUM."""
    contract = QueryContract(
        query_id="q2",
        dataset_id="d1",
        original_query="Show average monthly income by age",
        requested_fields=["Age", "MonthlyIncome"],
        dimensions=["Age"],
        measure_column="MonthlyIncome",
        aggregation="AVG",
        suggested_chart_type="line",
    )
    contract.x_axis = "Age"
    contract.y_axis = "MonthlyIncome"

    bad_spec = {
        "chart": {"type": "line"},
        "fields": {"x": "Age", "y": "MonthlyIncome"},
        "measure": {"type": "max", "column": "MonthlyIncome"},
    }
    is_valid, reason = SpecValidator.validate(bad_spec, contract, available_columns)
    assert is_valid is False
    assert "altered canonical aggregation 'AVG' to 'max'" in reason


def test_reject_altering_series_dimension(available_columns):
    """If AnalysisSpec requests series_dimension='Department', candidate spec cannot switch to 'Gender'."""
    contract = QueryContract(
        query_id="q3",
        dataset_id="d1",
        original_query="Compare average income across departments for each age",
        requested_fields=["Age", "Department", "MonthlyIncome"],
        dimensions=["Age", "Department"],
        measure_column="MonthlyIncome",
        aggregation="AVG",
        series_dimension="Department",
        suggested_chart_type="multi_line",
    )
    contract.x_axis = "Age"

    bad_spec = {
        "chart": {"type": "line"},
        "fields": {"x": "Age", "y": "MonthlyIncome", "color": "Gender"},
        "measure": {"type": "avg", "column": "MonthlyIncome"},
    }
    is_valid, reason = SpecValidator.validate(bad_spec, contract, available_columns)
    assert is_valid is False
    assert "Department" in reason


def test_pass_valid_conforming_spec(available_columns):
    """Spec conforming to AnalysisSpec passes validation."""
    contract = QueryContract(
        query_id="q4",
        dataset_id="d1",
        original_query="Compare average income across departments for each age",
        requested_fields=["Age", "Department", "MonthlyIncome"],
        dimensions=["Age", "Department"],
        measure_column="MonthlyIncome",
        aggregation="AVG",
        series_dimension="Department",
        suggested_chart_type="multi_line",
    )
    contract.x_axis = "Age"

    good_spec = {
        "chart": {"type": "line"},
        "fields": {"x": "Age", "y": "MonthlyIncome", "color": "Department"},
        "measure": {"type": "avg", "column": "MonthlyIncome"},
    }
    is_valid, reason = SpecValidator.validate(good_spec, contract, available_columns)
    assert is_valid is True
    assert reason is None


def test_deterministic_spec_locks_to_contract(available_columns):
    """Deterministic fallback spec faithfully preserves all AnalysisSpec parameters."""
    contract = QueryContract(
        query_id="q5",
        dataset_id="d1",
        original_query="Compare average income across departments for each age",
        requested_fields=["Age", "Department", "MonthlyIncome"],
        dimensions=["Age", "Department"],
        measure_column="MonthlyIncome",
        aggregation="AVG",
        series_dimension="Department",
        suggested_chart_type="multi_line",
    )
    contract.x_axis = "Age"

    built = SpecValidator.build_deterministic_spec(contract, available_columns)
    assert built["fields"]["x"] == "Age"
    assert built["fields"]["color"] == "Department"
    assert built["fields"]["y"] == "MonthlyIncome"
    assert built["measure"]["type"] == "avg"
    assert built["measure"]["column"] == "MonthlyIncome"
