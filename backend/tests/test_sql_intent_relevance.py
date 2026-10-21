import unittest
from unittest.mock import patch
import pandas as pd

from app.agents.sql.agent import SQLAgent
from app.agents.supervisor.agent import SupervisorAgent

INVESTMENT_SCHEMA = {
    "columns": [
        {"name": "age", "dtype": "int64"},
        {"name": "gender", "dtype": "object"},
        {"name": "investment_avenues", "dtype": "object"},
        {"name": "objective", "dtype": "object"},
        {"name": "duration", "dtype": "object"},
        {"name": "expect", "dtype": "object"},
        {"name": "avenue", "dtype": "object"},
    ]
}


class SQLIntentRelevanceTests(unittest.TestCase):
    def test_age_distribution_single_column_satisfies_intent(self):
        """Single column 'age' must satisfy FREQUENCY intent for histogram binning."""
        intent = {
            "operation": "FREQUENCY",
            "target_column": "age",
            "group_by_needed": True,
            "aggregation": "COUNT",
        }
        self.assertTrue(SQLAgent._result_answers_intent(["age"], intent))
        self.assertTrue(SQLAgent._result_answers_intent(["Age"], intent))

    def test_case_insensitive_matching(self):
        """Columns in different casing must satisfy target_column."""
        intent = {
            "operation": "FREQUENCY",
            "target_column": "Avenue",
            "group_by_needed": True,
        }
        self.assertTrue(SQLAgent._result_answers_intent(["avenue", "investor_count"], intent))

    def test_cross_tabulation_satisfies_intent(self):
        """Multi-dimension result with counts satisfies visualization intent."""
        intent = {
            "operation": "FREQUENCY",
            "target_column": "objective",
            "group_by_needed": True,
        }
        self.assertTrue(
            SQLAgent._result_answers_intent(["gender", "objective", "investor_count"], intent)
        )

    def test_heuristic_query_does_not_match_age_inside_average(self):
        """'Show average expected return by investment avenue' must not match 'age' from 'average'."""
        sql = SQLAgent._build_heuristic_query(
            "Show average expected return by investment avenue.",
            INVESTMENT_SCHEMA,
            "finance_data",
        )
        self.assertNotIn('"age"', sql)
        self.assertIn('"avenue"', sql)

    @patch("app.agents.sql.agent.pd.read_sql")
    def test_non_empty_df_does_not_fail_with_fatal_error(self, mock_read_sql):
        """Non-empty dataframe proceeds even if heuristic check is cautious."""
        mock_read_sql.return_value = pd.DataFrame({
            "gender": ["Male", "Female"],
            "objective": ["Wealth Creation", "Retirement"],
            "investor_count": [15, 6],
        })
        state = {
            "table_name": "finance_data",
            "question": "Show gender versus investment objective visually",
            "schema": INVESTMENT_SCHEMA,
            "structured_intent": {
                "operation": "FREQUENCY",
                "target_column": "objective",
            },
        }
        result = SQLAgent().run(state)
        self.assertIsNotNone(result["sql_result"])
        self.assertEqual(len(result["sql_result"]["rows"]), 2)
        self.assertNotIn(
            "SQL result columns do not satisfy the requested analytical intent.",
            result.get("errors", []),
        )


if __name__ == "__main__":
    unittest.main()
