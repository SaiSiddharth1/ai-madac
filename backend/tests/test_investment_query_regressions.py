"""Dataset-backed regressions for investment analysis questions.

These assertions independently calculate the expected values from the shipped
CSV, while separately checking that the SQL builder selects the matching
operation and dimension.
"""

from pathlib import Path
import unittest

import pandas as pd

from app.agents.sql.agent import SQLAgent
from app.agents.supervisor.agent import SupervisorAgent


DATASET = Path(__file__).parents[2] / "data" / "sample" / "finance_investment_data.csv"
DF = pd.read_csv(DATASET)
SCHEMA = {"columns": [
    {"name": column, "dtype": str(dtype)} for column, dtype in DF.dtypes.items()
]}


class InvestmentQuestionRegressions(unittest.TestCase):
    table_name = "investment_data"

    def build_sql(self, question: str) -> str:
        intent = SupervisorAgent._parse_structured_intent(question, SCHEMA)
        return SQLAgent._build_heuristic_query(question, SCHEMA, self.table_name, intent)

    def test_total_people(self):
        sql = self.build_sql("How many people are in the dataset?")
        self.assertEqual(len(DF), 40)
        self.assertIn("COUNT(*) AS total_rows", sql)

    def test_wealth_creation_count(self):
        sql = self.build_sql("How many investors have Wealth Creation as their objective?")
        self.assertEqual(int((DF["Objective"] == "Wealth Creation").sum()), 19)
        self.assertIn('"Objective" = \'Wealth Creation\'', sql)
        self.assertIn("COUNT(*) AS investor_count", sql)

    def test_most_popular_avenue(self):
        sql = self.build_sql("What is the most popular investment avenue based on the Avenue column?")
        result = DF["Avenue"].value_counts()
        self.assertEqual((result.index[0], int(result.iloc[0])), ("Mutual Funds", 14))
        self.assertIn('SELECT "Avenue", COUNT(*) AS investor_count', sql)
        self.assertIn('GROUP BY "Avenue"', sql)
        self.assertIn("ORDER BY investor_count DESC", sql)
        self.assertNotIn("SELECT *", sql)

    def test_counts_by_source(self):
        sql = self.build_sql("Count investors for each source.")
        self.assertEqual(DF["Source"].value_counts().to_dict(), {
            "Internet": 19, "Financial Consultants": 14, "Newspapers": 7,
        })
        self.assertIn('GROUP BY "Source"', sql)

    def test_percentages_by_avenue(self):
        sql = self.build_sql("What percentage of investors choose each investment avenue?")
        percentages = (DF["Avenue"].value_counts(normalize=True) * 100).round(2)
        self.assertEqual(float(percentages["Mutual Funds"]), 35.0)
        self.assertIn("SUM(COUNT(*)) OVER ()", sql)
        self.assertIn("AS percentage", sql)

    def test_most_popular_avenue_for_female_investors(self):
        sql = self.build_sql("Which investment avenue is most popular among female investors?")
        result = DF.loc[DF["Gender"] == "Female", "Avenue"].value_counts()
        self.assertEqual((result.index[0], int(result.iloc[0])), ("Mutual Funds", 5))
        self.assertIn('"Gender" = \'Female\'', sql)
        self.assertIn('GROUP BY "Avenue"', sql)


if __name__ == "__main__":
    unittest.main()
