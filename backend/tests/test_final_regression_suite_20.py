"""
Final 20-Question Regression Suite for AI-MADAC.
Validates all 20 canonical questions covering:
1. Basic aggregations (no SUM(*), MAX(*), scalar answers)
2. Semantic filters (Attrition='Yes', OverTime='Yes', Department='Sales')
3. Grouping (Department, Age, JobSatisfaction)
4. Ratios (Attrition rate, Overtime attrition percentage)
5. Distributions (Categorical Bar/Pie vs Numeric Histogram)
6. Multi-Series (Age across OverTime, Gender, Department)
7. Python .strip() exception immunity
8. State isolation across sequential multi-turn questions
"""

import pytest
from app.agents.state import create_fresh_question_state
from app.agents.supervisor.agent import supervisor_agent
from app.agents.sql.agent import sql_agent
from app.agents.visualization.agent import visualization_agent
from app.db.database import sync_engine
import sqlalchemy as sa


@pytest.fixture(scope="module")
def setup_dataset():
    with sync_engine.connect() as conn:
        res = conn.execute(sa.text("SELECT id, table_name, schema_info FROM datasets ORDER BY created_at DESC LIMIT 1")).fetchone()
        if not res:
            pytest.skip("No dataset found in database")
        dataset_id, table_name, schema_json = res[0], res[1], res[2]
        import json
        schema = json.loads(schema_json) if isinstance(schema_json, str) else schema_json
        return dataset_id, table_name, schema


def run_pipeline(question: str, dataset_id: str, table_name: str, schema: dict, prior_state=None):
    """Executes supervisor -> sql -> visualization pipeline with guaranteed fresh question state."""
    state = create_fresh_question_state(
        user_id="test_user",
        dataset_id=dataset_id,
        query_id=f"q_{hash(question)}",
        question=question,
        table_name=table_name,
        schema=schema,
        previous_validated_result=prior_state.get("validated_result") if prior_state else None,
    )

    sup_res = supervisor_agent.run(state)
    state.update(sup_res)

    sql_res = sql_agent.run(state)
    state.update(sql_res)

    vis_res = None
    if state.get("query_contract", {}).get("should_visualize") or "visualization_agent" in state.get("execution_plan", []):
        vis_res = visualization_agent.run(state)
        state.update(vis_res)

    return state


# =========================================================================
# 1. BASIC AGGREGATIONS
# =========================================================================

def test_q01_total_number_of_employees(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("What is the total number of employees?", dataset_id, table_name, schema)

    sql = state["sql_query"].upper()
    assert "SUM(*)" not in sql
    assert "COUNT(*)" in sql
    assert state["sql_result"]["response_type"] == "SCALAR"
    assert int(state["sql_result"]["scalar_value"]) == 1470


def test_q02_average_age_of_employees(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("What is the average age of employees?", dataset_id, table_name, schema)

    sql = state["sql_query"].upper()
    assert "AVG(" in sql and "AGE" in sql
    assert state["sql_result"]["response_type"] == "SCALAR"
    assert 36 <= float(state["sql_result"]["scalar_value"]) <= 38


def test_q03_highest_monthly_income(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("What is the highest monthly income?", dataset_id, table_name, schema)

    sql = state["sql_query"].upper()
    assert "MAX(" in sql and "MONTHLYINCOME" in sql
    assert "MAX(*)" not in sql
    assert state["sql_result"]["response_type"] == "SCALAR"
    assert int(float(state["sql_result"]["scalar_value"])) == 19999


# =========================================================================
# 2. FILTERS
# =========================================================================

def test_q04_how_many_employees_have_left_company(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("How many employees have left the company?", dataset_id, table_name, schema)

    sql = state["sql_query"]
    assert "Attrition" in sql and "Yes" in sql
    assert int(state["sql_result"]["scalar_value"]) == 237


def test_q05_how_many_employees_work_overtime(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("How many employees work overtime?", dataset_id, table_name, schema)

    sql = state["sql_query"]
    assert "OverTime" in sql and "Yes" in sql
    assert int(state["sql_result"]["scalar_value"]) == 416


def test_q06_sales_department_employees_overtime(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("How many Sales department employees work overtime?", dataset_id, table_name, schema)

    sql = state["sql_query"]
    assert "Sales" in sql
    assert "OverTime" in sql and "Yes" in sql
    assert state["sql_result"]["response_type"] != "UNRELATED"
    assert int(state["sql_result"]["scalar_value"]) > 0


# =========================================================================
# 3. GROUPING
# =========================================================================

def test_q07_employees_in_each_department(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("How many employees are there in each department?", dataset_id, table_name, schema)

    sql = state["sql_query"].upper()
    assert "GROUP BY" in sql and "DEPARTMENT" in sql
    assert len(state["sql_result"]["rows"]) == 3
    assert state["sql_result"]["response_type"] == "GROUPED"


def test_q08_attrition_count_by_department(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("What is the attrition count by department?", dataset_id, table_name, schema)

    sql = state["sql_query"]
    assert "GROUP BY" in sql.upper() and "Department" in sql
    assert "Attrition" in sql and "Yes" in sql
    rows = state["sql_result"]["rows"]
    assert len(rows) == 3
    total_attrition = sum(r[list(r.keys())[1]] for r in rows)
    assert total_attrition == 237


def test_q09_average_monthly_income_by_age(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("Show the average monthly income by age.", dataset_id, table_name, schema)

    sql = state["sql_query"].upper()
    assert "GROUP BY" in sql and "AGE" in sql
    assert len(state["sql_result"]["rows"]) > 10


def test_q10_average_job_satisfaction_by_department(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("Show the average job satisfaction by department.", dataset_id, table_name, schema)

    sql = state["sql_query"].upper()
    assert "GROUP BY" in sql and "DEPARTMENT" in sql
    assert "JOBSATISFACTION" in sql
    assert len(state["sql_result"]["rows"]) == 3


# =========================================================================
# 4. RATIOS
# =========================================================================

def test_q11_overall_attrition_rate(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("What is the overall attrition rate?", dataset_id, table_name, schema)

    val = float(state["sql_result"]["scalar_value"])
    assert 15.0 <= val <= 17.5  # 237 / 1470 * 100 = 16.12%


def test_q12_percentage_overtime_employees_left(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("What percentage of employees who work overtime have left the company?", dataset_id, table_name, schema)

    val = float(state["sql_result"]["scalar_value"])
    assert 28.0 <= val <= 33.0  # 127 / 416 * 100 = 30.53%


# =========================================================================
# 5. DISTRIBUTIONS
# =========================================================================

def test_q13_distribution_by_gender_is_categorical(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("Show the distribution of employees by gender.", dataset_id, table_name, schema)

    sql = state["sql_query"].upper()
    assert "GROUP BY" in sql and "GENDER" in sql
    chart_t = state["query_contract"]["suggested_chart_type"]
    assert chart_t in ("bar", "pie")
    assert chart_t != "histogram"


def test_q14_distribution_of_ages_is_histogram(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("Show the distribution of employee ages.", dataset_id, table_name, schema)

    chart_t = state["query_contract"]["suggested_chart_type"]
    assert chart_t == "histogram"
    assert "SUM(" not in state["sql_query"].upper()


def test_q15_distribution_of_monthly_income_is_histogram(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("Show the distribution of monthly income.", dataset_id, table_name, schema)

    chart_t = state["query_contract"]["suggested_chart_type"]
    assert chart_t == "histogram"
    assert "SUM(" not in state["sql_query"].upper()


def test_q16_distribution_of_job_roles_is_categorical(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("Show the distribution of employees by job role.", dataset_id, table_name, schema)

    sql = state["sql_query"].upper()
    assert "GROUP BY" in sql and "JOBROLE" in sql
    chart_t = state["query_contract"]["suggested_chart_type"]
    assert chart_t in ("bar", "horizontal_bar", "pie")
    assert chart_t != "histogram"


# =========================================================================
# 6. MULTI-SERIES
# =========================================================================

def test_q17_compare_income_overtime_across_age(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("Compare average monthly income for employees who work overtime and those who do not across age.", dataset_id, table_name, schema)

    group_by = state["query_contract"]["group_by"]
    assert "Age" in group_by and "OverTime" in group_by
    assert state["query_contract"]["series_dimension"] == "OverTime"
    assert state["query_contract"]["suggested_chart_type"] == "multi_line"


def test_q18_compare_attrition_rates_gender_across_age(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("Compare attrition rates for male and female employees across age.", dataset_id, table_name, schema)

    group_by = state["query_contract"]["group_by"]
    assert "Age" in group_by and "Gender" in group_by
    assert len(state["sql_result"]["rows"]) > 10


def test_q19_compare_income_across_departments_for_each_age(setup_dataset):
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("Compare average monthly income across departments for each age.", dataset_id, table_name, schema)

    group_by = state["query_contract"]["group_by"]
    assert "Age" in group_by and "Department" in group_by
    assert state["query_contract"]["series_dimension"] == "Department"


def test_q20_compare_job_satisfaction_across_departments_for_each_age(setup_dataset):
    """Test Question 20 passes without Python .strip() exception."""
    dataset_id, table_name, schema = setup_dataset
    state = run_pipeline("Compare the average job satisfaction across departments for each age.", dataset_id, table_name, schema)

    assert state.get("sql_result") is not None
    assert len(state["sql_result"]["rows"]) > 10
    assert not any("strip" in str(err).lower() for err in state.get("errors", []))
    if state.get("visualization_result"):
        assert bool(state["visualization_result"]["visualization_available"]) is True


# =========================================================================
# 7. SEQUENTIAL SAME-SESSION EXECUTION (ZERO STATE CONTAMINATION)
# =========================================================================

def test_sequential_multi_turn_same_session(setup_dataset):
    """
    Run 4 questions sequentially in the exact same session, feeding prior state.
    Verifies that transient state (filters, measures, group_by) does NOT leak into subsequent queries.
    """
    dataset_id, table_name, schema = setup_dataset

    # Turn 1: Filter query (OverTime = Yes)
    s1 = run_pipeline("How many employees work overtime?", dataset_id, table_name, schema)
    assert int(s1["sql_result"]["scalar_value"]) == 416

    # Turn 2: Unrelated scalar average (Age) - must NOT inherit OverTime filter
    s2 = run_pipeline("What is the average age of employees?", dataset_id, table_name, schema, prior_state=s1)
    assert "OverTime" not in s2["sql_query"]
    assert 36 <= float(s2["sql_result"]["scalar_value"]) <= 38

    # Turn 3: Grouped query (Department) - must NOT inherit OverTime or Age
    s3 = run_pipeline("How many employees are there in each department?", dataset_id, table_name, schema, prior_state=s2)
    assert "OverTime" not in s3["sql_query"]
    assert "Age" not in s3["sql_query"]
    assert len(s3["sql_result"]["rows"]) == 3

    # Turn 4: Scalar max (MonthlyIncome) - must NOT inherit Department grouping
    s4 = run_pipeline("What is the highest monthly income?", dataset_id, table_name, schema, prior_state=s3)
    assert "GROUP BY" not in s4["sql_query"].upper()
    assert int(float(s4["sql_result"]["scalar_value"])) == 19999
