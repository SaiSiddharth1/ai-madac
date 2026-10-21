"""Unit tests for VisualizationAgent adhering to AI-MADAC visualization specifications."""

import os
import unittest
import pandas as pd

from app.agents.visualization.agent import VisualizationAgent


class VisualizationAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = VisualizationAgent()

    def test_empty_data_returns_unavailable_payload(self):
        state = {
            "question": "Can you show in pie chart how many genders are there?",
            "sql_result": {"rows": [], "columns": ["gender", "investor_count"]},
            "analysis_result": None,
        }
        output = self.agent.run(state)
        viz = output["visualization_result"]

        self.assertIsNotNone(viz)
        self.assertFalse(viz["visualization_available"])
        self.assertIsNone(viz["chart"])
        self.assertIn("not enough data", viz["explanation"])
        self.assertEqual(viz["key_insights"], [])

    def test_single_value_kpi_skips_chart(self):
        state = {
            "question": "Show count of genders",
            "sql_result": {"rows": [{"gender_count": 2}], "columns": ["gender_count"]},
            "analysis_result": None,
        }
        output = self.agent.run(state)
        viz = output["visualization_result"]

        self.assertIsNotNone(viz)
        self.assertFalse(viz["visualization_available"])
        self.assertIsNone(viz["chart"])
        self.assertEqual(viz["key_insights"], [])

    def test_gender_pie_chart_generation(self):
        state = {
            "question": "can u show in pie chart how many genders are there",
            "sql_result": {
                "rows": [
                    {"gender": "Male", "investor_count": 23},
                    {"gender": "Female", "investor_count": 17},
                ],
                "columns": ["gender", "investor_count"],
                "query": 'SELECT "gender", COUNT(*) AS investor_count FROM "finance" GROUP BY "gender"',
                "filters": {},
            },
            "analysis_result": None,
            "structured_intent": {"target_column": "gender", "aggregation": "COUNT"},
        }
        output = self.agent.run(state)
        viz = output["visualization_result"]

        self.assertIsNotNone(viz)
        self.assertTrue(viz["visualization_available"])
        self.assertIsNotNone(viz["chart"])
        self.assertEqual(viz["chart"]["chart_type"], "pie")
        self.assertEqual(viz["chart"]["title"], "Distribution: gender")

        # Verify parameters adhere to spec
        params = viz["chart"]["parameters"]
        self.assertEqual(params["chart_type"], "Pie chart")
        self.assertEqual(params["category"], "gender")
        self.assertEqual(params["value"], "investor_count")
        self.assertEqual(params["aggregation"], "Already aggregated by source agent")
        # Ensure no unused filters are claimed
        self.assertNotIn("filters", params)

        # Verify explanation
        self.assertIn("Parameters used:", viz["explanation"])
        self.assertIn("Category: gender", viz["explanation"])
        self.assertIn("Value: investor_count", viz["explanation"])
        self.assertIn("Male", viz["explanation"])

        # Verify insights
        self.assertGreater(len(viz["key_insights"]), 0)
        self.assertTrue(any("Male" in s for s in viz["key_insights"]))

        # Verify PNG file was generated on disk
        self.assertGreater(len(viz["chart_paths"]), 0)
        png_rel_path = viz["chart_paths"][0].replace("/static/charts/", "")
        charts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "charts")
        png_full_path = os.path.join(charts_dir, png_rel_path)
        self.assertTrue(os.path.exists(png_full_path))

    def test_bar_chart_generation_with_filter(self):
        state = {
            "question": "Show investment avenues for female investors",
            "sql_result": {
                "rows": [
                    {"avenue": "Mutual Funds", "investor_count": 5},
                    {"avenue": "Fixed Deposits", "investor_count": 4},
                ],
                "columns": ["avenue", "investor_count"],
                "filters": {"gender": "Female"},
            },
            "analysis_result": None,
            "structured_intent": {"target_column": "avenue", "aggregation": "COUNT"},
        }
        output = self.agent.run(state)
        viz = output["visualization_result"]

        self.assertTrue(viz["visualization_available"])
        self.assertEqual(viz["chart"]["chart_type"], "bar")
        params = viz["chart"]["parameters"]
        self.assertEqual(params["chart_type"], "Bar chart")
        self.assertEqual(params["x_axis"], "avenue")
        self.assertEqual(params["y_axis"], "investor_count")
        # Filter is properly included when applied
        self.assertIn("gender = 'Female'", params.get("filters", ""))


if __name__ == "__main__":
    unittest.main()
