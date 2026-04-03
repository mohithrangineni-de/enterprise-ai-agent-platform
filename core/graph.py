"""
core/graph.py

LangGraph state machine that wires all agents together.
Defines the nodes, edges, and conditional routing logic.

Flow:
  START → planner → [sql | rag | python] (parallel) → response → END
"""

from __future__ import annotations
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from core.state import AgentState
from agents.planner import PlannerAgent
from agents.sql_agent import SQLAgent
from agents.rag_agent import RAGAgent
from agents.python_agent import PythonAgent
from agents.response_agent import ResponseAgent
from observability.logger import get_logger

log = get_logger(__name__)


# ─── Instantiate agents (one per type — stateless) ────────────────────────

planner = PlannerAgent()
sql_agent = SQLAgent()
rag_agent = RAGAgent()
python_agent = PythonAgent()
response_agent = ResponseAgent()


# ─── Node functions (each wraps an agent's .run()) ────────────────────────

async def run_planner(state: AgentState) -> dict:
    log.info("graph.node", node="planner", session=state.session_id)
    updated = await planner.run(state)
    return {"tasks": updated.tasks, "routing_rationale": updated.routing_rationale}


async def run_sql(state: AgentState) -> dict:
    log.info("graph.node", node="sql_agent", session=state.session_id)
    updated = await sql_agent.run(state)
    return {"results": updated.results, "spans": updated.spans}


async def run_rag(state: AgentState) -> dict:
    log.info("graph.node", node="rag_agent", session=state.session_id)
    updated = await rag_agent.run(state)
    return {"results": updated.results, "spans": updated.spans}


async def run_python(state: AgentState) -> dict:
    log.info("graph.node", node="python_agent", session=state.session_id)
    updated = await python_agent.run(state)
    return {"results": updated.results, "spans": updated.spans}


async def run_response(state: AgentState) -> dict:
    log.info("graph.node", node="response_agent", session=state.session_id)
    updated = await response_agent.run(state)
    return {
        "final_answer": updated.final_answer,
        "answer_confidence": updated.answer_confidence,
        "cited_sources": updated.cited_sources,
        "total_tokens": updated.total_tokens,
        "total_latency_ms": updated.total_latency_ms,
        "completed": True,
    }


# ─── Conditional routing: which specialist agents to invoke? ──────────────

def route_to_specialists(state: AgentState) -> list[str]:
    """
    After the planner, fan out to only the agents that are needed.
    Returns a list of node names (LangGraph parallel branching).
    """
    needed = {task.agent_type for task in state.tasks if task.agent_type != "response"}
    nodes = []
    if "sql" in needed:
        nodes.append("sql_agent")
    if "rag" in needed:
        nodes.append("rag_agent")
    if "python" in needed:
        nodes.append("python_agent")

    if not nodes:
        log.warning("graph.route", msg="No specialist agents needed — going direct to response")
        nodes = ["response_agent"]

    log.info("graph.route", nodes=nodes, session=state.session_id)
    return nodes


# ─── Build the graph ──────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Register nodes
    g.add_node("planner", run_planner)
    g.add_node("sql_agent", run_sql)
    g.add_node("rag_agent", run_rag)
    g.add_node("python_agent", run_python)
    g.add_node("response_agent", run_response)

    # Entry point
    g.add_edge(START, "planner")

    # After planner: fan out to whichever specialists are needed
    g.add_conditional_edges(
        "planner",
        route_to_specialists,
        {
            "sql_agent": "sql_agent",
            "rag_agent": "rag_agent",
            "python_agent": "python_agent",
            "response_agent": "response_agent",  # direct path if no specialists needed
        },
    )

    # All specialist agents converge on response_agent
    g.add_edge("sql_agent", "response_agent")
    g.add_edge("rag_agent", "response_agent")
    g.add_edge("python_agent", "response_agent")

    # Exit
    g.add_edge("response_agent", END)

    return g.compile()


# ─── Singleton compiled graph ──────────────────────────────────────────────

agent_graph = build_graph()
