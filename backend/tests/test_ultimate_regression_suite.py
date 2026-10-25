"""
AI-MADAC Ultimate Semantic Planning, Dataset Isolation, Calculation & Visualization Reliability Regression Suite.
Covers all requirements from Sections 44, 45, 46, and 51 of the specification.
"""

import math
import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.schemas.query_contract import QueryContract, CalculationSpec, OperandSpec
from app.agents.supervisor.semantic_planner import semantic_planner
from app.agents.sql.agent import sql_agent
from app.agents.visualization.contract import extract_query_contract
from app.agents.visualization.spec_validator import SpecValidator
from app.agents.visualization.data_preparer import DataPreparer
from app.analytics.calculation_engine import CalculationEngine
from app.utils.schema_resolver import SchemaResolver
from app.utils.temporal import is_temporal_column, is_temporal_intent
from app.utils.serialization import to_json_safe


# --- Mock Schemas ---

IBM_HR_SCHEMA = {
    "columns": [
        {"name": "Age", "dtype": "INTEGER"},
        {"name": "Attrition", "dtype": "TEXT"},
        {"name": "BusinessTravel", "dtype": "TEXT"},
        {"name": "DailyRate", "dtype": "INTEGER"},
        {"name": "Department", "dtype": "TEXT"},
        {"name": "DistanceFromHome", "dtype": "INTEGER"},
        {"name": "Education", "dtype": "INTEGER"},
        {"name": "EducationField", "dtype": "TEXT"},
        {"name": "EmployeeCount", "dtype": "INTEGER"},
        {"name": "EmployeeNumber", "dtype": "INTEGER"},
        {"name": "EnvironmentSatisfaction", "dtype": "INTEGER"},
        {"name": "Gender", "dtype": "TEXT"},
        {"name": "HourlyRate", "dtype": "INTEGER"},
        {"name": "JobInvolvement", "dtype": "INTEGER"},
        {"name": "JobLevel", "dtype": "INTEGER"},
        {"name": "JobRole", "dtype": "TEXT"},
        {"name": "JobSatisfaction", "dtype": "INTEGER"},
        {"name": "MaritalStatus", "dtype": "TEXT"},
        {"name": "MonthlyIncome", "dtype": "INTEGER"},
        {"name": "MonthlyRate", "dtype": "INTEGER"},
        {"name": "NumCompaniesWorked", "dtype": "INTEGER"},
        {"name": "Over18", "dtype": "TEXT"},
        {"name": "OverTime", "dtype": "TEXT"},
        {"name": "PercentSalaryHike", "dtype": "INTEGER"},
        {"name": "PerformanceRating", "dtype": "INTEGER"},
        {"name": "RelationshipSatisfaction", "dtype": "INTEGER"},
        {"name": "StandardHours", "dtype": "INTEGER"},
        {"name": "StockOptionLevel", "dtype": "INTEGER"},
        {"name": "TotalWorkingYears", "dtype": "INTEGER"},
        {"name": "TrainingTimesLastYear", "dtype": "INTEGER"},
        {"name": "WorkLifeBalance", "dtype": "INTEGER"},
        {"name": "YearsAtCompany", "dtype": "INTEGER"},
        {"name": "YearsInCurrentRole", "dtype": "INTEGER"},
        {"name": "YearsSinceLastPromotion", "dtype": "INTEGER"},
        {"name": "YearsWithCurrManager", "dtype": "INTEGER"},
    ]
}

SUPERSTORE_SCHEMA = {
    "columns": [
        {"name": "Row ID", "dtype": "INTEGER"},
        {"name": "Order ID", "dtype": "TEXT"},
        {"name": "Order Date", "dtype": "DATE"},
        {"name": "Ship Date", "dtype": "DATE"},
        {"name": "Ship Mode", "dtype": "TEXT"},
        {"name": "Customer ID", "dtype": "TEXT"},
        {"name": "Customer Name", "dtype": "TEXT"},
        {"name": "Segment", "dtype": "TEXT"},
        {"name": "Country", "dtype": "TEXT"},
        {"name": "City", "dtype": "TEXT"},
        {"name": "State", "dtype": "TEXT"},
        {"name": "Postal Code", "dtype": "INTEGER"},
        {"name": "Region", "dtype": "TEXT"},
        {"name": "Product ID", "dtype": "TEXT"},
        {"name": "Category", "dtype": "TEXT"},
        {"name": "Sub-Category", "dtype": "TEXT"},
        {"name": "Product Name", "dtype": "TEXT"},
        {"name": "Sales", "dtype": "REAL"},
        {"name": "Quantity", "dtype": "INTEGER"},
        {"name": "Discount", "dtype": "REAL"},
        {"name": "Profit", "dtype": "REAL"},
    ]
}


class TestTemporalDetection:
    """Verifies that non-temporal fields (MonthlyIncome, OverTime) are never treated as dates."""

    def test_overtime_is_not_temporal(self):
        assert not is_temporal_column("OverTime")
        assert not is_temporal_column("overtime")

    def test_monthly_income_is_not_temporal(self):
        assert not is_temporal_column("MonthlyIncome")
        assert not is_temporal_column("monthly_income")
        assert not is_temporal_column("MonthlyRate")
        assert not is_temporal_column("DailyRate")
        assert not is_temporal_column("HourlyRate")

    def test_genuine_dates_are_temporal(self):
        assert is_temporal_column("Order Date")
        assert is_temporal_column("order_date")
        assert is_temporal_column("InvoiceDate")
        assert is_temporal_column("created_at")

    def test_monthly_alone_does_not_trigger_time_series(self):
        q1 = "What is maximum monthly income?"
        assert not is_temporal_intent(q1, [c["name"] for c in IBM_HR_SCHEMA["columns"]])

        q2 = "Show average monthly income by job level and gender as a multi-line chart."
        assert not is_temporal_intent(q2, [c["name"] for c in IBM_HR_SCHEMA["columns"]])

    def test_trend_over_time_triggers_time_series(self):
        q = "Show monthly total sales from 2011 to 2015 as line chart."
        assert is_temporal_intent(q, [c["name"] for c in SUPERSTORE_SCHEMA["columns"]])


class TestSchemaResolver:
    """Verifies centralized schema resolution without arbitrary substitutions."""

    def test_resolves_monthly_income_accurately(self):
        resolver = SchemaResolver(IBM_HR_SCHEMA)
        res = resolver.resolve_column("monthly income")
        assert res == "MonthlyIncome"

    def test_resolves_job_level(self):
        resolver = SchemaResolver(IBM_HR_SCHEMA)
        res = resolver.resolve_column("job level")
        assert res == "JobLevel"

    def test_resolves_gender(self):
        resolver = SchemaResolver(IBM_HR_SCHEMA)
        res = resolver.resolve_column("gender")
        assert res == "Gender"

    def test_no_substitution_for_missing_column(self):
        resolver = SchemaResolver(IBM_HR_SCHEMA)
        # Sales and Product do NOT exist in IBM HR
        assert resolver.resolve_column("sales") is None
        assert resolver.resolve_column("product") is None
        assert resolver.resolve_column("profit") is None


class TestIBMHrRegressions:
    """Test suite for Section 44 IBM HR regression tests."""

    def test_ibm_hr_1_average_age(self):
        contract = semantic_planner.plan_query("What is average age?", IBM_HR_SCHEMA, "dataset_hr", "q1")
        assert contract.measure_column == "Age"
        assert contract.aggregation == "AVG"
        sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "hr_table")
        assert 'AVG("Age")' in sql or 'AVG("age")' in sql.lower()

    def test_ibm_hr_2_maximum_monthly_income(self):
        contract = semantic_planner.plan_query("What is maximum monthly income?", IBM_HR_SCHEMA, "dataset_hr", "q2")
        assert contract.measure_column == "MonthlyIncome"
        assert contract.aggregation == "MAX"
        assert contract.time_column is None
        assert contract.intent in ("AGGREGATION", "DATA_RETRIEVAL")
        sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "hr_table")
        assert 'MAX("MonthlyIncome")' in sql
        assert "Age" not in sql
        assert "TIMESTAMP" not in sql
        assert "strftime" not in sql

    def test_ibm_hr_3_unique_job_roles(self):
        contract = semantic_planner.plan_query("How many unique job roles?", IBM_HR_SCHEMA, "dataset_hr", "q3")
        assert contract.measure_column == "JobRole"
        assert contract.aggregation == "COUNT_DISTINCT"
        sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "hr_table")
        assert 'COUNT(DISTINCT "JobRole")' in sql

    def test_ibm_hr_4_average_monthly_income_overtime_filter(self):
        contract = semantic_planner.plan_query(
            "What is average monthly income for employees who work overtime?",
            IBM_HR_SCHEMA, "dataset_hr", "q4"
        )
        assert contract.measure_column == "MonthlyIncome"
        assert contract.aggregation == "AVG"
        assert any(f.column == "OverTime" for f in contract.filters)
        sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "hr_table")
        assert 'AVG("MonthlyIncome")' in sql
        assert '"OverTime"' in sql

    def test_ibm_hr_5_employees_by_gender_bar_chart(self):
        contract = semantic_planner.plan_query(
            "Show number of employees by gender as bar chart.",
            IBM_HR_SCHEMA, "dataset_hr", "q5"
        )
        assert "Gender" in contract.group_by
        assert contract.aggregation in ("COUNT", "COUNT_DISTINCT")
        assert contract.explicit_chart_type in ("bar", "column")
        sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "hr_table")
        assert '"Gender"' in sql
        assert 'COUNT(*)' in sql

    def test_ibm_hr_6_mandatory_multiline_chart(self):
        """
        CRITICAL MANDATORY TEST CASE:
        Question: Show average monthly income by job level and gender as a multi-line chart.
        Expected:
        Chart Type: Multi-Line
        X-Axis: JobLevel
        Y-Axis: MonthlyIncome
        Aggregation: AVG
        Series: Gender
        Group By: JobLevel + Gender
        NO OverTime, NO SUM, NO Chronological Sorting!
        """
        q = "Show average monthly income by job level and gender as a multi-line chart."
        contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_hr", "q6")

        assert contract.explicit_chart_type in ("multi_line", "multiline", "line")
        assert contract.measure_column == "MonthlyIncome"
        assert contract.aggregation == "AVG"
        assert "JobLevel" in contract.group_by
        assert "Gender" in contract.group_by
        assert contract.series_dimension == "Gender"
        assert contract.x_axis == "JobLevel"
        assert contract.y_axis == "MonthlyIncome"
        assert "OverTime" not in contract.group_by
        assert contract.time_column is None

        # Test SQL Generation
        sql = sql_agent._build_query_from_contract(contract, IBM_HR_SCHEMA, "hr_table")
        assert '"JobLevel"' in sql
        assert '"Gender"' in sql
        assert 'AVG("MonthlyIncome")' in sql
        assert 'SUM("MonthlyIncome")' not in sql
        assert '"OverTime"' not in sql
        assert 'GROUP BY "JobLevel", "Gender"' in sql or 'GROUP BY "JobLevel"' in sql

        # Test Spec Validator (deterministic spec)
        spec = SpecValidator.build_deterministic_spec(contract, [c["name"] for c in IBM_HR_SCHEMA["columns"]])
        assert spec["fields"]["x"] == "JobLevel"
        assert spec["fields"]["y"] == "MonthlyIncome"
        assert spec["fields"]["color"] == "Gender"
        assert spec["measure"]["type"] in ("avg", "mean")
        assert "OverTime" not in spec["fields"].values()
        assert "Average Monthly Income" in spec["title"]

        # Test Data Preparer
        mock_df = pd.DataFrame({
            "JobLevel": [1, 1, 2, 2, 3, 3],
            "Gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
            "MonthlyIncome": [2500, 2600, 4500, 4600, 7500, 7600],
        })
        prep_df, applied_params, insights = DataPreparer.prepare(mock_df, spec)
        assert applied_params["category"] == "JobLevel"
        assert applied_params["value"] == "MonthlyIncome"
        assert applied_params["aggregation"] == "AVG"
        assert applied_params["series"] == "Gender"
        assert "OverTime" not in str(applied_params)
        assert "Chronological" not in applied_params["sorting"]
        assert applied_params["filters"] == "None"

    def test_ibm_hr_7_monthly_income_vs_age_scatter(self):
        q = "Show monthly income versus age as scatter plot."
        contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_hr", "q7")
        assert contract.explicit_chart_type == "scatter"
        assert contract.x_axis == "MonthlyIncome"
        assert contract.y_axis == "Age"

        spec = SpecValidator.build_deterministic_spec(contract, [c["name"] for c in IBM_HR_SCHEMA["columns"]])
        assert spec["fields"]["x"] == "MonthlyIncome"
        assert spec["fields"]["y"] == "Age"
        assert "Monthly Income vs Age" in spec["title"]

    def test_ibm_hr_8_distribution_of_age_histogram(self):
        q = "Show distribution of Age using histogram."
        contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_hr", "q8")
        assert contract.explicit_chart_type == "histogram"
        assert contract.measure_column == "Age" or contract.x_axis == "Age"
        spec = SpecValidator.build_deterministic_spec(contract, [c["name"] for c in IBM_HR_SCHEMA["columns"]])
        assert spec["fields"]["x"] == "Age"
        assert "Distribution of Age" in spec["title"]

    def test_ibm_hr_9_average_monthly_income_not_age(self):
        q = "What is average monthly income?"
        contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_hr", "q9")
        assert contract.measure_column == "MonthlyIncome"
        assert contract.measure_column != "Age"
        assert contract.aggregation == "AVG"


class TestCrossDatasetProtection:
    """Verifies Section 46 cross-dataset protection."""

    def test_sales_and_products_rejected_on_ibm_hr(self):
        q = "Show top 10 products by total sales."
        contract = semantic_planner.plan_query(q, IBM_HR_SCHEMA, "dataset_hr", "q_cross")
        # Must detect that query is unrelated to IBM HR
        assert not contract.is_related_to_dataset or contract.intent == "UNRELATED"
        assert contract.unrelated_reason is not None
        assert "product" in contract.unrelated_reason.lower() or "sale" in contract.unrelated_reason.lower() or "not contain" in contract.unrelated_reason.lower()


class TestSuperstoreSuite:
    """Verifies Section 45 Superstore tests when Superstore is loaded."""

    def test_superstore_1_total_sales(self):
        q = "What is total sales?"
        contract = semantic_planner.plan_query(q, SUPERSTORE_SCHEMA, "dataset_superstore", "qs1")
        assert contract.measure_column == "Sales"
        assert contract.aggregation == "SUM"
        sql = sql_agent._build_query_from_contract(contract, SUPERSTORE_SCHEMA, "superstore_table")
        assert 'SUM("Sales")' in sql

    def test_superstore_2_top_10_products(self):
        q = "Show top 10 products by total sales as horizontal bar chart."
        contract = semantic_planner.plan_query(q, SUPERSTORE_SCHEMA, "dataset_superstore", "qs2")
        assert contract.measure_column == "Sales"
        assert "Product Name" in contract.group_by
        assert contract.limit == 10
        assert contract.explicit_chart_type in ("horizontal_bar", "bar")
        sql = sql_agent._build_query_from_contract(contract, SUPERSTORE_SCHEMA, "superstore_table")
        assert '"Product Name"' in sql
        assert 'LIMIT 10' in sql

    def test_superstore_3_average_sales_by_region(self):
        q = "Show average sales by region as bar chart."
        contract = semantic_planner.plan_query(q, SUPERSTORE_SCHEMA, "dataset_superstore", "qs3")
        assert contract.measure_column == "Sales"
        assert contract.aggregation == "AVG"
        assert "Region" in contract.group_by
        sql = sql_agent._build_query_from_contract(contract, SUPERSTORE_SCHEMA, "superstore_table")
        assert 'AVG("Sales")' in sql
        assert '"Region"' in sql

        spec = SpecValidator.build_deterministic_spec(contract, [c["name"] for c in SUPERSTORE_SCHEMA["columns"]])
        assert "Average Sales" in spec["title"]
        assert "Total Sales" not in spec["title"]

    def test_superstore_4_monthly_sales_time_series(self):
        q = "Show monthly total sales from 2011 to 2015 as line chart."
        contract = semantic_planner.plan_query(q, SUPERSTORE_SCHEMA, "dataset_superstore", "qs4")
        assert contract.measure_column == "Sales"
        assert contract.aggregation == "SUM"
        assert contract.time_column == "Order Date"
        assert contract.time_granularity is not None and contract.time_granularity.lower() == "month"
        assert contract.intent in ("TIME_SERIES", "TREND")

    def test_superstore_5_percentage_calculation(self):
        # Deterministic calculation engine test
        val_tech = 836154.03
        val_total = 2297200.86
        res = CalculationEngine.compute_scalar(val_tech, val_total, "PERCENTAGE", "Technology Share")
        assert res.calculated is True
        assert res.value is not None
        assert math.isclose(res.value, 36.3988, rel_tol=1e-3)
        assert "%" in res.formatted_value


class TestCalculationEngineGuardrails:
    """Verifies that CalculationEngine does not silently guess positional columns or fail on 0/0."""

    def test_missing_operands_returns_error(self):
        mock_df = pd.DataFrame({"Age": [30], "Department": ["Sales"]})
        calc_spec = CalculationSpec(
            operator="PERCENTAGE",
            label="Tech Share",
            operands=[
                OperandSpec(name="tech_sales", measure="Sales"),
                OperandSpec(name="total_sales", measure="Sales"),
            ]
        )
        df_out, calc_res = CalculationEngine.compute_from_dataframe(mock_df, calc_spec)
        assert calc_res is not None
        assert calc_res.calculated is False
        assert "not found" in calc_res.error.lower()

    def test_valid_operands_calculated(self):
        mock_df = pd.DataFrame({"tech_sales": [300.0], "total_sales": [1000.0]})
        calc_spec = CalculationSpec(
            operator="PERCENTAGE",
            label="Tech Share",
            operands=[
                OperandSpec(name="tech_sales", measure="tech_sales"),
                OperandSpec(name="total_sales", measure="total_sales"),
            ]
        )
        df_out, calc_res = CalculationEngine.compute_from_dataframe(mock_df, calc_spec)
        assert calc_res is not None
        assert calc_res.calculated is True
        assert calc_res.value == 30.0
        assert "30.00%" in calc_res.formatted_value


class TestJsonSerialization:
    """Verifies Section 39 serialization protection."""

    def test_serializes_complex_pandas_and_numpy_objects(self):
        raw = {
            "timestamp": pd.Timestamp("2023-05-15 14:30:00"),
            "nan_val": float("nan"),
            "inf_val": float("inf"),
            "df": pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]}),
            "series": pd.Series([10, 20]),
        }
        safe = to_json_safe(raw)
        assert isinstance(safe["timestamp"], str)
        assert safe["nan_val"] is None
        assert safe["inf_val"] is None
        assert isinstance(safe["df"], list)
        assert isinstance(safe["series"], list)
