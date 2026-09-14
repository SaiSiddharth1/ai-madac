import unittest

from app.agents.report.agent import ReportAgent


class ReportAgentFallbackTests(unittest.TestCase):
    def test_mixed_blank_and_numeric_metrics_do_not_break_ranking(self):
        answer, _ = ReportAgent()._deterministic_fallback_report(
            "Show project values", None,
            {"columns": ["project_code", "commitment_in_u_a"], "rows": [
                {"project_code": "missing", "commitment_in_u_a": ""},
                {"project_code": "top", "commitment_in_u_a": 42.5},
                {"project_code": "lower", "commitment_in_u_a": 9.0},
            ], "row_count": 3},
        )
        self.assertIn("42.5", answer)
        self.assertIn("top", answer)


if __name__ == "__main__":
    unittest.main()
