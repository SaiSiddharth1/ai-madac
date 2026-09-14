import unittest

from app.agents.sql.agent import SQLAgent


SCHEMA = {
    "columns": [
        {"name": "country", "dtype": "object"},
        {"name": "project_name", "dtype": "object"},
        {"name": "source_of_financing", "dtype": "object"},
        {"name": "commitment_in_u_a", "dtype": "float64"},
    ]
}


class SQLMetricGuardrailTests(unittest.TestCase):
    def test_top_countries_by_total_commitment_uses_sum_not_count(self):
        sql = SQLAgent._build_heuristic_query(
            "Create a bar chart of the top 10 countries by total financial commitment (commitment_in_u_a)",
            SCHEMA,
            "finance1",
        )

        self.assertIn('SUM("commitment_in_u_a")', sql)
        self.assertIn('GROUP BY "country"', sql)
        self.assertIn('ORDER BY "total_commitment_in_u_a" DESC', sql)
        self.assertIn('LIMIT 10', sql)
        self.assertNotIn('COUNT(*)', sql)

    def test_top_projects_by_commitment_sorts_project_values(self):
        sql = SQLAgent._build_heuristic_query(
            "Top 10 projects by commitment",
            SCHEMA,
            "finance1",
        )

        self.assertIn('SELECT "project_name", "commitment_in_u_a"', sql)
        self.assertIn('ORDER BY "commitment_in_u_a" DESC LIMIT 10', sql)


if __name__ == "__main__":
    unittest.main()
