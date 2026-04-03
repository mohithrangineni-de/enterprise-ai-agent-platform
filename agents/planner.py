"""
agents/planner.py

Planner Agent — the entry point of every query.
Analyzes the user's question and produces a task plan:
which agents to invoke, in what order, with what instructions.
"""

from __future__ import annotations
import json
import uuid
import time
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from core.state import AgentState, AgentTask, TraceSpan
from core.config import settings
from observability.logger import get_logger
from observability.tracer import trace_span

log = get_logger(__name__)

PLANNER_SYSTEM = """You are a planning agent for an enterprise analytics AI system.

Given a user question, produce a JSON task plan. Decide which of these agents to invoke:
- sql    : queries a relational database (for metrics, sales, revenue, counts, trends)
- rag    : searches documents (for policies, reports, strategy docs, meeting notes)
- python : runs statistical analysis or generates charts (for anomaly detection, forecasting)

Output JSON only — no prose:
{
  "tasks": [
    {
      "task_id": "<short_id>",
      "agent_type": "sql|rag|python",
      "description": "<what this agent should do>",
      "dependencies": []
    }
  ],
  "rationale": "<one sentence explaining the plan>"
}

Rules:
- Include only agents that are actually needed.
- If the question is purely factual from documents, use only rag.
- If the question needs data and context, use both sql and rag.
- Use python only when computation or comparison is required.
- task_ids must be short slugs like "sql_1", "rag_1", "py_1".
"""

PLANNER_HUMAN = "Question: {question}\n\nConversation history (last 3 turns):\n{history}"


class PlannerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNER_SYSTEM),
            ("human", PLANNER_HUMAN),
        ])
        self.chain = self.prompt | self.llm

    async def run(self, state: AgentState) -> AgentState:
        started = datetime.utcnow()
        t0 = time.monotonic()

        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in state.conversation_history[-6:]  # last 3 turns
        ) or "(none)"

        log.info("planner.start", session=state.session_id, question=state.question[:80])

        try:
            response = await self.chain.ainvoke({
                "question": state.question,
                "history": history_text,
            })

            raw = response.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            parsed = json.loads(raw)
            tasks = [AgentTask(**t) for t in parsed["tasks"]]
            rationale = parsed.get("rationale", "")

            latency = (time.monotonic() - t0) * 1000
            log.info("planner.done", tasks=[t.agent_type for t in tasks], latency_ms=latency)

        except Exception as e:
            log.error("planner.error", error=str(e))
            # Fallback: use both sql and rag
            tasks = [
                AgentTask(task_id="sql_1", agent_type="sql", description="Query relevant database tables"),
                AgentTask(task_id="rag_1", agent_type="rag", description="Search relevant documents"),
            ]
            rationale = f"Fallback plan due to planner error: {e}"
            latency = (time.monotonic() - t0) * 1000

        span = TraceSpan(
            span_id=str(uuid.uuid4())[:8],
            agent="planner",
            action="decompose_question",
            input_summary=state.question[:100],
            output_summary=f"{len(tasks)} tasks: {[t.agent_type for t in tasks]}",
            started_at=started,
            ended_at=datetime.utcnow(),
        )

        state.tasks = tasks
        state.routing_rationale = rationale
        state.spans.append(span)
        return state
