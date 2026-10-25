"""
Tests for Issue 10: AnalysisSpec Validator.
Verifies strict deterministic validation of AnalysisSpec (QueryContract) against dataset schema:
1. Field existence (dimensions, measures, filters, series)
2. Aggregation validity against datatypes (COUNT(*), SUM/AVG numeric, MIN/MAX real, reject SUM(*)/MAX(*))
3. Categorical filter validation and value alignment
4. Grouping and series dimension integrity
5. Chart compatibility (Histogram numeric only, Bar/Pie for categorical, Multi-line structure)
"""

import pytest
from app.schemas.query_contract import QueryContract, FilterSpec, CalculationSpec, OperandSpec
from app.schemas.analysis_spec_validator import AnalysisSpecValidator


@pytest.fixture
def hr_schema():
    return {
        "columns": [
            {"name": "Age", "dtype": "int64", "sample_values": [41, 49, 37]},
            {"name": "Attrition", "dtype": "object", "sample_values": ["Yes", "No"]},
            {"name": "Department", "dtype": "object", "sample_values": ["Sales", "Research & Development", "Human Resources"]},
            {"name": "Gender", "dtype": "object", "sample_values": ["Female", "Male"]},
            {"name": "JobRole", "dtype": "object", "sample_values": ["Sales Executive", "Research Scientist", "Laboratory Technician"]},
            {"name": "MonthlyIncome", "dtype": "int64", "sample_values": [5993, 5130, 2090]},
            {"name": "OverTime", "dtype": "object", "sample_values": ["Yes", "No"]},
            {"name": "YearsAtCompany", "dtype": "int64", "sample_values": [6, 10, 0]},
        ]
    }


def test_field_validation_nonexistent_column(hr_schema):
    """Fields not in schema must be rejected."""
    spec = QueryContract(
        query_id="q1",
        dataset_id="d1",
        original_query="What is the average profit by region?",
        measure_column="profit",
        aggregation="AVG",
        dimensions=["region"],
        group_by=["region"],
    )
    is_valid, errors = AnalysisSpecValidator.validate(spec, hr_schema)
    assert is_valid is False
    assert any("profit" in e.lower() for e in errors)
    assert any("region" in e.lower() for e in errors)


def test_field_validation_casing_alignment(hr_schema):
    """Casing variations are deterministically repaired to exact schema casing."""
    spec = QueryContract(
        query_id="q2",
        dataset_id="d1",
        original_query="Show average monthlyincome by department",
        measure_column="monthlyincome",
        aggregation="AVG",
        dimensions=["department"],
        group_by=["department"],
    )
    repaired, repairs = AnalysisSpecValidator.validate_and_repair(spec, hr_schema)
    assert repaired.measure_column == "MonthlyIncome"
    assert repaired.group_by == ["Department"]
    assert repaired.dimensions == ["Department"]
    is_valid, errors = AnalysisSpecValidator.validate(repaired, hr_schema)
    assert is_valid is True


def test_aggregation_rules_count_star_allowed(hr_schema):
    """COUNT(*) is valid and permitted."""
    spec = QueryContract(
        query_id="q3",
        dataset_id="d1",
        original_query="What is the total number of employees?",
        aggregation="COUNT",
        measure="total_records",
        measure_column=None,
    )
    is_valid, errors = AnalysisSpecValidator.validate(spec, hr_schema)
    assert is_valid is True
    assert len(errors) == 0


def test_aggregation_rules_reject_sum_star_and_repair(hr_schema):
    """SUM(*) and MAX(*) are invalid and must be repaired to COUNT(*)."""
    spec = QueryContract(
        query_id="q4",
        dataset_id="d1",
        original_query="What is the total number of employees?",
        aggregation="SUM",
        measure_column="*",
    )
    is_valid, errors = AnalysisSpecValidator.validate(spec, hr_schema)
    assert is_valid is False
    assert any("sum" in e.lower() and "*" in e for e in errors)

    repaired, repairs = AnalysisSpecValidator.validate_and_repair(spec, hr_schema)
    assert repaired.aggregation == "COUNT"
    assert repaired.measure == "total_records"
    assert repaired.measure_column is None


def test_aggregation_rules_reject_sum_on_non_numeric(hr_schema):
    """SUM/AVG on a categorical column must be rejected and repaired."""
    spec = QueryContract(
        query_id="q5",
        dataset_id="d1",
        original_query="Sum of department",
        aggregation="SUM",
        measure_column="Department",
    )
    is_valid, errors = AnalysisSpecValidator.validate(spec, hr_schema)
    assert is_valid is False
    assert any("numeric" in e.lower() for e in errors)

    repaired, repairs = AnalysisSpecValidator.validate_and_repair(spec, hr_schema)
    assert repaired.aggregation == "COUNT"


def test_filter_categorical_value_alignment(hr_schema):
    """Categorical filter values with different casing are aligned to schema sample values."""
    spec = QueryContract(
        query_id="q6",
        dataset_id="d1",
        original_query="How many employees have left?",
        aggregation="COUNT",
        measure="total_records",
        filters=[FilterSpec(column="attrition", operator="=", value="yes")],
    )
    repaired, repairs = AnalysisSpecValidator.validate_and_repair(spec, hr_schema)
    assert repaired.filters[0].column == "Attrition"
    assert repaired.filters[0].value == "Yes"


def test_grouping_integrity_prevents_scalar(hr_schema):
    """A spec with group_by cannot have answer_type SCALAR."""
    spec = QueryContract(
        query_id="q7",
        dataset_id="d1",
        original_query="Employees by department",
        aggregation="COUNT",
        measure="total_records",
        group_by=["Department"],
        answer_type="SCALAR",
    )
    is_valid, errors = AnalysisSpecValidator.validate(spec, hr_schema)
    assert is_valid is False
    assert any("scalar" in e.lower() for e in errors)

    repaired, repairs = AnalysisSpecValidator.validate_and_repair(spec, hr_schema)
    assert repaired.answer_type == "GROUPED"


def test_series_dimension_integrity(hr_schema):
    """Series dimension must be included in group_by."""
    spec = QueryContract(
        query_id="q8",
        dataset_id="d1",
        original_query="Compare income by age across gender",
        aggregation="AVG",
        measure_column="MonthlyIncome",
        group_by=["Age"],
        series_dimension="Gender",
        suggested_chart_type="multi_line",
    )
    is_valid, errors = AnalysisSpecValidator.validate(spec, hr_schema)
    assert is_valid is False
    assert any("series" in e.lower() for e in errors)

    repaired, repairs = AnalysisSpecValidator.validate_and_repair(spec, hr_schema)
    assert "Gender" in repaired.group_by
    assert repaired.series_dimension == "Gender"


def test_chart_compatibility_categorical_histogram_repaired(hr_schema):
    """Histogram requested on a categorical column (Gender) is repaired to Bar/Pie."""
    spec = QueryContract(
        query_id="q9",
        dataset_id="d1",
        original_query="Distribution of gender",
        suggested_chart_type="histogram",
        dimensions=["Gender"],
        x_axis="Gender",
    )
    is_valid, errors = AnalysisSpecValidator.validate(spec, hr_schema)
    assert is_valid is False
    assert any("histogram" in e.lower() for e in errors)

    repaired, repairs = AnalysisSpecValidator.validate_and_repair(spec, hr_schema)
    assert repaired.suggested_chart_type in ("pie", "bar")
    assert repaired.aggregation == "COUNT"
    assert repaired.group_by == ["Gender"]


def test_chart_compatibility_numeric_histogram_valid(hr_schema):
    """Histogram requested on a numeric column (Age) is valid."""
    spec = QueryContract(
        query_id="q10",
        dataset_id="d1",
        original_query="Distribution of employee ages",
        suggested_chart_type="histogram",
        dimensions=["Age"],
        x_axis="Age",
        measure_column="Age",
        intent="DISTRIBUTION",
        answer_type="DISTRIBUTION",
    )
    is_valid, errors = AnalysisSpecValidator.validate(spec, hr_schema)
    assert is_valid is True
    assert len(errors) == 0
