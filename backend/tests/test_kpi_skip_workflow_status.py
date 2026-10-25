"""
Tests for Surgical Fix: KPI Skip Must Not Be Reported As Failure.
Verifies Phases 1 through 11:
1. Intentional KPI skip for single-row results emits 'skipped' status (NOT 'failed').
2. Emitted message is informative: 'Chart skipped — result is better represented as a KPI'.
3. Real rendering errors and quality gate errors still emit 'failed'.
4. Explicit chart requests (e.g. 'Show in a bar chart') are preserved and not skipped.
5. Composite visualization_node path also correctly emits 'skipped'.
"""

import asyncio
import pandas as pd
import pytest

from app.agents.state import AgentState
from app.agents.visualization.agent import visualization_agent
from app.agents.visualization.contract import extract_query_contract
from app.core.workflow_events import workflow_event_manager
from app.graphs.workflow import chart_renderer_node, visualization_node


@pytest.fixture
def hr_schema():
    return {
        "columns": [
            {"name": "JobRole", "dtype": "object", "sample_values": ["Sales Executive", "Research Scientist"]},
            {"name": "count", "dtype": "int64", "sample_values": [326, 292]},
        ],
        "row_count": 1470,
    }


def test_kpi_skip_emits_skipped_status_in_chart_renderer_node(hr_schema):
    """
    For 'Which job role has the highest number of employees?', a 1-row result
    should intentionally skip chart rendering with status 'skipped', NOT 'failed'.
    """
    async def _test():
        query_id = "test_q_kpi_skip_1"
        queue = workflow_event_manager.register(query_id)

        state: AgentState = {
            "query_id": query_id,
            "question": "Which job role has the highest number of employees?",
            "schema": hr_schema,
            "execution_plan": ["sql_agent", "visualization_agent", "visualization_validator", "chart_renderer", "report_agent"],
            "agents_used": ["supervisor_agent", "sql_agent", "visualization_agent", "visualization_validator"],
            "sql_result": {
                "columns": ["JobRole", "count"],
                "rows": [{"JobRole": "Sales Executive", "count": 326}],
                "row_count": 1,
                "query": "SELECT JobRole, COUNT(*) AS count FROM dataset GROUP BY JobRole ORDER BY count DESC LIMIT 1",
            },
            "_query_df": pd.DataFrame([{"JobRole": "Sales Executive", "count": 326}]),
            "errors": [],
        }

        try:
            res = await chart_renderer_node(state)
            assert "chart_renderer" in res.get("agents_used", [])
            viz_res = res.get("visualization_result", {})
            assert viz_res.get("visualization_available") is False
            assert "KPI" in viz_res.get("reason", "")

            # Collect emitted events
            events = []
            while not queue.empty():
                events.append(await queue.get())

            renderer_events = [e for e in events if e.get("agent") == "chart_renderer"]
            assert len(renderer_events) >= 2  # 'running' then final status

            final_event = renderer_events[-1]
            assert final_event["status"] == "skipped", f"Expected 'skipped' but got '{final_event['status']}': {final_event.get('message')}"
            assert "KPI" in final_event["message"]
            assert "failed" not in [e["status"] for e in renderer_events]
        finally:
            workflow_event_manager.unregister(query_id)

    asyncio.run(_test())


def test_composite_visualization_node_emits_skipped_for_kpi(hr_schema):
    """
    When visualization_node runs as single composite step, it must emit 'skipped'
    for KPI skip rather than 'failed'.
    """
    async def _test():
        query_id = "test_q_kpi_skip_comp"
        queue = workflow_event_manager.register(query_id)

        state: AgentState = {
            "query_id": query_id,
            "question": "Which job role has the highest number of employees?",
            "schema": hr_schema,
            "execution_plan": ["sql_agent", "visualization_agent", "report_agent"],  # No separate validator/renderer
            "agents_used": ["supervisor_agent", "sql_agent"],
            "sql_result": {
                "columns": ["JobRole", "count"],
                "rows": [{"JobRole": "Sales Executive", "count": 326}],
                "row_count": 1,
                "query": "SELECT JobRole, COUNT(*) AS count FROM dataset GROUP BY JobRole ORDER BY count DESC LIMIT 1",
            },
            "_query_df": pd.DataFrame([{"JobRole": "Sales Executive", "count": 326}]),
            "errors": [],
        }

        try:
            res = await visualization_node(state)
            events = []
            while not queue.empty():
                events.append(await queue.get())

            renderer_events = [e for e in events if e.get("agent") == "chart_renderer"]
            assert len(renderer_events) >= 1
            final_renderer_event = renderer_events[-1]
            assert final_renderer_event["status"] == "skipped"
            assert "KPI" in final_renderer_event["message"]
        finally:
            workflow_event_manager.unregister(query_id)

    asyncio.run(_test())


def test_real_rendering_error_still_emits_failed(hr_schema, monkeypatch):
    """
    If there is a genuine error or failure in chart rendering, it MUST still emit 'failed'.
    """
    async def _test():
        query_id = "test_q_real_fail_1"
        queue = workflow_event_manager.register(query_id)

        # Simulate genuine rendering failure (e.g. visual quality validation failure)
        monkeypatch.setattr(
            visualization_agent,
            "run_rendering",
            lambda state: {
                "visualization_result": {
                    "visualization_available": False,
                    "reason": "Chart failed visual quality validation: Scatter plot X-axis must be numeric.",
                },
                "agents_used": ["chart_renderer"],
            },
        )

        state: AgentState = {
            "query_id": query_id,
            "question": "Show relationship between Department and MonthlyIncome in a scatter plot",
            "schema": hr_schema,
            "execution_plan": ["chart_renderer"],
            "agents_used": [],
        }

        try:
            res = await chart_renderer_node(state)
            events = []
            while not queue.empty():
                events.append(await queue.get())

            renderer_events = [e for e in events if e.get("agent") == "chart_renderer"]
            final_event = renderer_events[-1]
            assert final_event["status"] == "failed"
            assert "Chart failed visual quality validation" in final_event["message"]
        finally:
            workflow_event_manager.unregister(query_id)

    asyncio.run(_test())


def test_explicit_chart_request_with_single_row_does_not_skip(hr_schema):
    """
    When the user explicitly asks for a chart (e.g. 'Show in a bar chart'),
    the system must NOT skip for KPI even if DataFrame has 1 row.
    """
    question = "Which job role has the highest number of employees? Show in a bar chart"
    contract = extract_query_contract(question, hr_schema)
    assert contract.requested_chart == "bar"

    df = pd.DataFrame([{"JobRole": "Sales Executive", "count": 326}])
    spec = {
        "chart": {"type": "bar"},
        "fields": {"x": "JobRole", "y": "count"},
    }

    state: AgentState = {
        "question": question,
        "schema": hr_schema,
        "_query_contract": contract,
        "_validated_spec": spec,
        "_query_df": df,
        "sql_result": {"rows": df.to_dict(orient="records"), "columns": list(df.columns)},
    }

    res = visualization_agent.run_rendering(state)
    viz_res = res.get("visualization_result", {})
    # Should NOT have the KPI skip reason
    reason = viz_res.get("reason") or ""
    assert "Single value results are better communicated directly as a KPI" not in reason
