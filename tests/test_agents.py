"""
tests/test_agents.py

Unit + integration tests for all agents.
Uses pytest-asyncio for async test support.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from datetime import datetime

from core.state import AgentState, AgentTask, AgentResult
from agents.planner import PlannerAgent
from agents.sql_agent import SQLAgent
from agents.python_agent import PythonAgent
from agents.response_agent import ResponseAgent


# ─── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def base_state():
    return AgentState(
        session_id="test-session",
        question="Why did revenue drop last quarter?",
        trace_id="test-trace-001",
    )


@pytest.fixture
def state_with_sql_result(base_state):
    base_state.tasks = [
        AgentTask(task_id="sql_1", agent_type="sql", description="Query revenue by region")
    ]
    base_state.results = [
        AgentResult(
            task_id="sql_1",
            agent_type="sql",
            output={
                "sql": "SELECT region, SUM(revenue) FROM sales GROUP BY region",
                "columns": ["region", "revenue"],
                "rows": [
                    {"region": "APAC", "revenue": 1200000},
                    {"region": "AMER", "revenue": 4500000},
                ],
                "row_count": 2,
            },
            sources=["database:analytics"],
        )
    ]
    return base_state


# ─── Planner tests ────────────────────────────────────────────────────────

class TestPlannerAgent:
    @patch("agents.planner.ChatOpenAI")
    @pytest.mark.asyncio
    async def test_planner_produces_tasks(self, mock_llm_cls, base_state):
        """Planner should return at least one task."""
        mock_response = AsyncMock()
        mock_response.content = '{"tasks": [{"task_id": "sql_1", "agent_type": "sql", "description": "Query revenue", "dependencies": []}], "rationale": "Need data"}'
        mock_llm_cls.return_value.__or__ = lambda s, o: AsyncMock(ainvoke=AsyncMock(return_value=mock_response))

        agent = PlannerAgent()
        # Inject mocked chain
        agent.chain = AsyncMock(ainvoke=AsyncMock(return_value=mock_response))

        result = await agent.run(base_state)
        assert len(result.tasks) >= 1
        assert result.tasks[0].agent_type in ("sql", "rag", "python")

    @patch("agents.planner.ChatOpenAI")
    @pytest.mark.asyncio
    async def test_planner_fallback_on_error(self, mock_llm_cls, base_state):
        """Planner should fall back to sql+rag if LLM fails."""
        agent = PlannerAgent()
        agent.chain = AsyncMock(ainvoke=AsyncMock(side_effect=Exception("LLM timeout")))

        result = await agent.run(base_state)
        agent_types = {t.agent_type for t in result.tasks}
        assert "sql" in agent_types or "rag" in agent_types


# ─── Python Agent tests ───────────────────────────────────────────────────

class TestPythonAgent:
    @pytest.mark.asyncio
    async def test_safety_check_blocks_os_import(self):
        agent = PythonAgent()
        with pytest.raises(ValueError, match="Blocked import"):
            agent._check_code_safety("import os\nprint(os.getcwd())")

    @pytest.mark.asyncio
    async def test_safety_check_blocks_subprocess(self):
        agent = PythonAgent()
        with pytest.raises(ValueError, match="Blocked import"):
            agent._check_code_safety("import subprocess\nsubprocess.run(['ls'])")

    @pytest.mark.asyncio
    async def test_safety_check_allows_pandas(self):
        agent = PythonAgent()
        # Should not raise
        agent._check_code_safety("import pandas as pd\ndf = pd.DataFrame({'a': [1,2]})\nprint(df)")

    @pytest.mark.asyncio
    async def test_code_execution(self):
        agent = PythonAgent()
        code = "print('hello from agent')"
        output = await agent._execute_code(code, {}, timeout=10)
        assert "hello from agent" in output

    @pytest.mark.asyncio
    async def test_syntax_error_caught(self):
        agent = PythonAgent()
        with pytest.raises(ValueError, match="Syntax error"):
            agent._check_code_safety("def broken(\nprint('oops')")


# ─── Response Agent tests ─────────────────────────────────────────────────

class TestResponseAgent:
    def test_confidence_parsing(self):
        agent = ResponseAgent()
        text = "Revenue dropped in APAC.\nConfidence: 0.85"
        assert agent._parse_confidence(text) == pytest.approx(0.85)

    def test_confidence_default(self):
        agent = ResponseAgent()
        assert agent._parse_confidence("No confidence mentioned") == 0.5

    def test_confidence_clamped(self):
        agent = ResponseAgent()
        assert agent._parse_confidence("Confidence: 1.5") == 1.0

    def test_collect_sources(self, state_with_sql_result):
        agent = ResponseAgent()
        sources = agent._collect_sources(state_with_sql_result)
        assert "database:analytics" in sources

    def test_format_findings_handles_error(self, base_state):
        base_state.results = [
            AgentResult(task_id="sql_1", agent_type="sql", output=None, error="DB timeout")
        ]
        agent = ResponseAgent()
        findings = agent._format_findings(base_state)
        assert "ERROR" in findings
        assert "DB timeout" in findings


# ─── State tests ──────────────────────────────────────────────────────────

class TestAgentState:
    def test_results_accumulate(self):
        """operator.add annotation means results from parallel agents stack up."""
        state = AgentState(session_id="s", question="q", trace_id="t")
        r1 = AgentResult(task_id="sql_1", agent_type="sql", output="data1")
        r2 = AgentResult(task_id="rag_1", agent_type="rag", output="data2")
        state.results.extend([r1, r2])
        assert len(state.results) == 2

    def test_state_summary(self):
        from core.state import state_summary
        state = AgentState(session_id="s", question="q", trace_id="t")
        summary = state_summary(state)
        assert "session_id" in summary
        assert "total_tokens" in summary
