"""
Comprehensive Visualization Reliability Test Suite for AI-MADAC.
Tests all requirements and test matrices specified in the Ultimate Visualization Reliability Specification:
- Universal 1-D normalizer regression tests (2-D DataFrame, (N, 1) numpy array, multi-column rejection)
- Line chart numeric X-axis sorting (2, 10, 20) and aggregation
- Multi-line chart series dimension grouping (1 line per category, sorted chronological/numeric)
- Scatter plot 1-D array validation and color dimension
- Histogram raw observations and theme styling
- Bar, Horizontal Bar, Grouped Bar, Pie charts
- State isolation across sequential queries
- Fallback schema awareness (no hardcoded Sales/Quantity)
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from app.agents.visualization.data_preparer import DataPreparer, ensure_1d
from app.agents.visualization.plotly_renderer import PlotlyRenderer
from app.agents.visualization.contract import extract_query_contract
from app.agents.visualization.spec_validator import SpecValidator
from app.agents.supervisor.semantic_planner import semantic_planner
from app.agents.sql.agent import sql_agent
from app.agents.report.agent import report_agent
from app.schemas.query_contract import QueryContract


@pytest.fixture
def investor_df():
    """Mock dataset matching AI-MADAC investor data."""
    return pd.DataFrame({
        "Age": [21, 25, 30, 22, 21, 35, 40, 22, 30, 25],
        "Gender": ["Male", "Female", "Male", "Female", "Female", "Male", "Female", "Male", "Female", "Male"],
        "Investment_Avenues": ["Mutual Funds", "Equity", "Mutual Funds", "Fixed Deposits", "Equity", "Gold", "Mutual Funds", "Debentures", "Gold", "Equity"],
        "Expect": ["10-20%", "20-30%", "10-20%", "20-30%", "30-40%", "10-20%", "20-30%", "10-20%", "20-30%", "30-40%"],
        "Avenue": ["Mutual Funds", "Equity", "Mutual Funds", "Fixed Deposits", "Equity", "Gold", "Mutual Funds", "Debentures", "Gold", "Equity"],
    })


@pytest.fixture
def investor_schema():
    return {
        "columns": [
            {"name": "Age", "dtype": "int64"},
            {"name": "Gender", "dtype": "object"},
            {"name": "Investment_Avenues", "dtype": "object"},
            {"name": "Expect", "dtype": "object"},
            {"name": "Avenue", "dtype": "object"},
        ]
    }


# =====================================================================
# 1. UNIVERSAL 1-D NORMALIZER REGRESSION TESTS (Sections 14, 18, 47, 48, 49)
# =====================================================================

class TestUniversal1DNormalizer:
    """Tests guaranteeing that every Plotly axis receives strictly 1-D structures."""

    def test_single_column_dataframe_normalized_to_1d_series(self):
        """Passing df[['Age']] (2-D DataFrame) must normalize safely to 1-D Series."""
        df = pd.DataFrame({"Age": [21, 22, 23, 24, 25]})
        val_2d = df[["Age"]]
        assert isinstance(val_2d, pd.DataFrame)
        assert val_2d.ndim == 2

        norm = ensure_1d(val_2d, "Age")
        assert isinstance(norm, pd.Series)
        assert norm.ndim == 1
        assert len(norm) == 5
        assert list(norm) == [21, 22, 23, 24, 25]

    def test_2d_numpy_array_nx1_squeezed_to_1d_series(self):
        """Passing numpy array with shape (N, 1) must normalize to 1-D Series."""
        arr_2d = np.array([10.5, 20.2, 30.8]).reshape(-1, 1)
        assert arr_2d.shape == (3, 1)

        norm = ensure_1d(arr_2d, "Values")
        assert isinstance(norm, pd.Series)
        assert norm.ndim == 1
        assert len(norm) == 3

    def test_multi_column_dataframe_raises_error(self):
        """Passing multi-column DataFrame must be rejected with ValueError."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        with pytest.raises(ValueError, match=r"Expected 1-D series|Expected one column"):
            ensure_1d(df, "Multi")

    def test_multi_column_numpy_array_raises_error(self):
        """Passing (N, 2) array must be rejected."""
        arr_multi = np.zeros((5, 2))
        with pytest.raises(ValueError, match="Expected 1-D array"):
            ensure_1d(arr_multi, "Multi")

    def test_nested_list_flattened_to_1d(self):
        """Passing [[1], [2], [3]] flattens to 1-D Series."""
        nested = [[1], [2], [3]]
        norm = ensure_1d(nested, "Nested")
        assert isinstance(norm, pd.Series)
        assert norm.ndim == 1
        assert list(norm) == [1, 2, 3]


# =====================================================================
# 2. LINE CHART TESTS (Sections 6, 7, 8, 31, 48, 51)
# =====================================================================

class TestLineChartReliability:
    """Validates line chart semantic contract, numeric sorting, and Plotly rendering."""

    def test_average_expected_return_by_age_line_chart(self, investor_df, investor_schema):
        question = "Show the average expected return by age as a line chart."
        contract = extract_query_contract(question, investor_schema)

        assert contract.explicit_chart_type in ("line", "trend")
        assert contract.x_axis == "Age" or "Age" in contract.dimensions
        assert contract.sorting.get("direction") == "asc"

        spec = SpecValidator.build_deterministic_spec(contract, list(investor_df.columns))
        assert spec["chart"]["type"] == "line"
        assert spec["fields"]["x"] == "Age"

        prepared_df, applied_params, insights = DataPreparer.prepare(investor_df, spec)
        assert not prepared_df.empty
        # Verify X-axis is sorted in numeric ascending order (e.g., 21 < 22 < 25 < 30 < 35 < 40)
        x_vals = list(prepared_df["Age"])
        assert x_vals == sorted(x_vals)

        # Plotly rendering must succeed without error
        fig, title, chart_path, json_path = PlotlyRenderer.render(prepared_df, spec, question)
        assert fig is not None
        assert len(fig.data) >= 1
        assert fig.data[0].type == "scatter"


# =====================================================================
# 3. MULTI-LINE CHART TESTS (Sections 9, 10, 11, 12, 13, 49, 51)
# =====================================================================

class TestMultiLineChartReliability:
    """Validates multi-line chart grouping, categorical series dimension, and sorting."""

    def test_average_expected_return_by_age_for_male_and_female(self, investor_df, investor_schema):
        question = "Show average expected return by age for male and female investors as a multi-line chart."
        contract = extract_query_contract(question, investor_schema)

        assert contract.explicit_chart_type in ("multi_line", "line")
        assert contract.series_dimension == "Gender" or "Gender" in contract.dimensions
        assert "Age" in contract.dimensions

        spec = SpecValidator.build_deterministic_spec(contract, list(investor_df.columns))
        assert spec["chart"]["type"] == "line"
        assert spec["fields"]["color"] == "Gender"

        prepared_df, applied_params, insights = DataPreparer.prepare(investor_df, spec)
        assert not prepared_df.empty
        # Verify Gender has multiple series (Male and Female)
        assert set(prepared_df["Gender"].unique()) == {"Male", "Female"}

        # Verify rendering produces multiple lines (one per Gender)
        fig, title, chart_path, json_path = PlotlyRenderer.render(prepared_df, spec, question)
        assert fig is not None
        # Must have at least 2 traces (one for Male, one for Female)
        assert len(fig.data) >= 2


# =====================================================================
# 4. SCATTER PLOT TESTS (Sections 14, 15, 16, 17, 47, 51)
# =====================================================================

class TestScatterPlotReliability:
    """Validates scatter plots receive strictly 1-D structures and do not throw dimension errors."""

    def test_relationship_between_age_and_expected_return(self, investor_df, investor_schema):
        question = "Show the relationship between age and expected return as a scatter plot."
        contract = extract_query_contract(question, investor_schema)

        assert contract.explicit_chart_type == "scatter"
        assert "Age" in contract.requested_fields

        spec = SpecValidator.build_deterministic_spec(contract, list(investor_df.columns))
        assert spec["chart"]["type"] == "scatter"

        prepared_df, applied_params, insights = DataPreparer.prepare(investor_df, spec)
        assert not prepared_df.empty
        assert len(prepared_df) == len(investor_df)

        fig, title, chart_path, json_path = PlotlyRenderer.render(prepared_df, spec, question)
        assert fig is not None
        assert fig.data[0].type == "scatter"
        assert fig.data[0].mode == "markers"

    def test_scatter_with_color_dimension(self, investor_df, investor_schema):
        question = "Show age vs expected return by gender as a scatter plot."
        contract = extract_query_contract(question, investor_schema)

        assert contract.explicit_chart_type == "scatter"
        assert contract.color_by == "Gender" or contract.series_dimension == "Gender"

        spec = SpecValidator.build_deterministic_spec(contract, list(investor_df.columns))
        assert spec["fields"]["color"] == "Gender"

        prepared_df, applied_params, insights = DataPreparer.prepare(investor_df, spec)
        fig, title, chart_path, json_path = PlotlyRenderer.render(prepared_df, spec, question)
        assert fig is not None
        assert len(fig.data) >= 2  # Male and Female marker groups


# =====================================================================
# 5. HISTOGRAM TESTS (Sections 19, 20, 21, 22, 23, 24, 50, 51)
# =====================================================================

class TestHistogramReliability:
    """Validates histogram styling, bin logic, and raw observations."""

    def test_distribution_of_age_histogram(self, investor_df, investor_schema):
        question = "Show the distribution of age as a histogram."
        contract = extract_query_contract(question, investor_schema)

        assert contract.explicit_chart_type == "histogram"
        assert "Age" in contract.requested_fields

        spec = SpecValidator.build_deterministic_spec(contract, list(investor_df.columns))
        assert spec["chart"]["type"] == "histogram"
        assert spec["fields"]["x"] == "Age"

        prepared_df, applied_params, insights = DataPreparer.prepare(investor_df, spec)
        # Raw observations: row count must equal valid input records
        assert len(prepared_df) == len(investor_df)

        fig, title, chart_path, json_path = PlotlyRenderer.render(prepared_df, spec, question)
        assert fig is not None
        assert fig.data[0].type == "histogram"
        # Check professional styling
        assert fig.data[0].marker.color is not None
        assert fig.data[0].marker.line.color is not None
        assert fig.layout.plot_bgcolor == "#101426"

    def test_distribution_of_expected_return_histogram(self, investor_df, investor_schema):
        question = "Show the distribution of expected return as a histogram."
        contract = extract_query_contract(question, investor_schema)

        assert contract.explicit_chart_type == "histogram"
        spec = SpecValidator.build_deterministic_spec(contract, list(investor_df.columns))

        prepared_df, applied_params, insights = DataPreparer.prepare(investor_df, spec)
        assert not prepared_df.empty
        fig, title, chart_path, json_path = PlotlyRenderer.render(prepared_df, spec, question)
        assert fig is not None
        assert fig.data[0].type == "histogram"


# =====================================================================
# 6. REGRESSION TESTS: BAR, HORIZONTAL BAR, GROUPED BAR, PIE (Sections 25, 26, 27)
# =====================================================================

class TestOtherChartTypesRegression:
    """Verifies that fixing line/multi-line/scatter does not regress bar, grouped bar, pie."""

    def test_bar_chart_investors_by_avenue(self, investor_df, investor_schema):
        question = "Show the number of investors by investment avenue as a bar chart."
        contract = extract_query_contract(question, investor_schema)
        spec = SpecValidator.build_deterministic_spec(contract, list(investor_df.columns))
        prepared_df, _, _ = DataPreparer.prepare(investor_df, spec)

        fig, _, _, _ = PlotlyRenderer.render(prepared_df, spec, question)
        assert fig is not None
        assert fig.data[0].type == "bar"

    def test_horizontal_bar_top_3(self, investor_df, investor_schema):
        question = "Show the top 3 investment avenues by number of investors as a horizontal bar chart."
        contract = extract_query_contract(question, investor_schema)
        contract.limit = 3
        spec = SpecValidator.build_deterministic_spec(contract, list(investor_df.columns))
        spec["chart"]["orientation"] = "horizontal"
        prepared_df, _, _ = DataPreparer.prepare(investor_df, spec)

        fig, _, _, _ = PlotlyRenderer.render(prepared_df, spec, question)
        assert fig is not None
        assert fig.data[0].type == "bar"
        assert fig.data[0].orientation == "h"
        assert len(prepared_df) <= 3

    def test_grouped_bar_multi_measure(self, investor_df, investor_schema):
        question = "Show average age and average expected return for each investment avenue as a grouped bar chart."
        contract = extract_query_contract(question, investor_schema)
        contract.secondary_measure_column = "Expect"
        spec = SpecValidator.build_deterministic_spec(contract, list(investor_df.columns))

        prepared_df, _, _ = DataPreparer.prepare(investor_df, spec)
        fig, _, _, _ = PlotlyRenderer.render(prepared_df, spec, question)
        assert fig is not None
        assert len(fig.data) >= 2  # 2 measures per avenue

    def test_pie_chart_percentage_by_avenue(self, investor_df, investor_schema):
        question = "Show the percentage of investors by investment avenue as a pie chart."
        contract = extract_query_contract(question, investor_schema)
        spec = SpecValidator.build_deterministic_spec(contract, list(investor_df.columns))

        prepared_df, _, _ = DataPreparer.prepare(investor_df, spec)
        fig, _, _, _ = PlotlyRenderer.render(prepared_df, spec, question)
        assert fig is not None
        assert fig.data[0].type == "pie"


# =====================================================================
# 7. STATE ISOLATION & INDEPENDENCE (Sections 28, 45)
# =====================================================================

class TestStateIsolation:
    """Verifies that sequential queries do not contaminate each other's visualization state."""

    def test_sequential_queries_have_isolated_contracts(self, investor_schema):
        queries = [
            ("Show number of investors by investment avenue as a bar chart.", "bar"),
            ("Show average expected return by age as a line chart.", "line"),
            ("Show average expected return by age for male and female investors as a multi-line chart.", "multi_line"),
            ("Show age vs expected return as a scatter plot.", "scatter"),
            ("Show distribution of expected return as a histogram.", "histogram"),
            ("Show percentage of investors by investment avenue as a pie chart.", "pie"),
        ]

        previous_contract = None
        for q_text, expected_chart in queries:
            contract = extract_query_contract(q_text, investor_schema, previous_result=None)
            assert not contract.is_follow_up
            assert contract.explicit_chart_type == expected_chart or (expected_chart == "multi_line" and contract.explicit_chart_type in ("multi_line", "line"))
            if previous_contract:
                # Ensure fields from previous query did not leak
                if expected_chart == "histogram":
                    assert "Gender" not in contract.requested_fields
                if expected_chart == "pie":
                    assert "Age" not in contract.requested_fields
            previous_contract = contract


# =====================================================================
# 8. SCHEMA-AWARE FALLBACK PLANNER (Sections 38, 39, 52)
# =====================================================================

class TestSchemaAwareFallbackPlanner:
    """Ensures fallback planner never uses hardcoded retail columns (Quantity/UnitPrice/Sales)."""

    def test_fallback_planner_resolves_investor_columns(self, investor_schema):
        # Fallback when Gemini is unavailable
        q1 = "Show the average expected return by age as a line chart."
        c1 = semantic_planner._heuristic_contract_fallback(q1, investor_schema, "ds-1", "q-1")
        assert c1.x_axis == "Age"
        assert c1.measure == "Expect" or "return" in str(c1.measure).lower()
        assert c1.aggregation == "AVG"
        assert c1.suggested_chart_type in ("line", "multi_line")

        q2 = "Show the relationship between age and expected return as a scatter plot."
        c2 = semantic_planner._heuristic_contract_fallback(q2, investor_schema, "ds-2", "q-2")
        assert c2.suggested_chart_type == "scatter"
        assert c2.x_axis in ("Age", "Expect")
        assert c2.y_axis in ("Age", "Expect")
        assert c2.aggregation is None  # Raw observations!

        q3 = "Show the distribution of age as a histogram."
        c3 = semantic_planner._heuristic_contract_fallback(q3, investor_schema, "ds-3", "q-3")
        assert c3.suggested_chart_type == "histogram"
        assert c3.x_axis == "Age"
        assert c3.aggregation is None  # Raw observations!
