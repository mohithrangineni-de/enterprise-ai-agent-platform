"""
agents/sql_agent.py

SQL Agent — translates natural language into SQL, executes it,
and returns structured results with the query for auditability.

Features:
- Schema-aware prompt (injected at runtime)
- SQL generation with retry on syntax error
- Read-only query guard (blocks INSERT/UPDATE/DELETE)
- Result summarization for the response agent
"""

from __future__ import annotations
import uuid
import time
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from core.state import AgentState, AgentResult, TraceSpan
from core.config import settings
from tools.database import DatabaseTool
from observability.logger import get_logger

log = get_logger(__name__)

SQL_SYSTEM = """You are a SQL expert for an enterprise analytics database.

Available schema:
{schema}

Rules:
- Generate valid PostgreSQL only.
- SELECT queries only — no INSERT, UPDATE, DELETE, DROP.
- Use explicit column names, not SELECT *.
- Add LIMIT 1000 as a safety net.
- If the question is ambiguous, write the most reasonable interpretation.

Respond with ONLY the SQL query — no prose, no markdown fences.
"""

SQL_HUMAN = "Task: {task_description}\n\nUser question context: {question}"


class SQLAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
        self.db = DatabaseTool()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SQL_SYSTEM),
            ("human", SQL_HUMAN),
        ])
        self.chain = self.prompt | self.llm

    async def run(self, state: AgentState) -> AgentState:
        started = datetime.utcnow()
        t0 = time.monotonic()

        sql_tasks = [t for t in state.tasks if t.agent_type == "sql"]
        if not sql_tasks:
            return state

        schema = await self.db.get_schema_description()
        results = []

        for task in sql_tasks:
            log.info("sql_agent.start", task_id=task.task_id, description=task.description[:60])
            try:
                # Generate SQL
                response = await self.chain.ainvoke({
                    "schema": schema,
                    "task_description": task.description,
                    "question": state.question,
                })
                sql = response.content.strip().strip("```sql").strip("```").strip()

                # Safety check
                sql_upper = sql.upper()
                for forbidden in ("INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER"):
                    if forbidden in sql_upper:
                        raise ValueError(f"SQL agent attempted a write operation: {forbidden}")

                # Execute
                rows, columns = await self.db.execute(sql)
                latency = (time.monotonic() - t0) * 1000

                result = AgentResult(
                    task_id=task.task_id,
                    agent_type="sql",
                    output={
                        "sql": sql,
                        "columns": columns,
                        "rows": rows[:100],          # cap payload
                        "row_count": len(rows),
                    },
                    sources=[f"database:{settings.db_name}"],
                    latency_ms=latency,
                    token_usage={"prompt": 0, "completion": 0},  # fill from response.usage
                )
                log.info("sql_agent.done", rows=len(rows), latency_ms=latency)

            except Exception as e:
                log.error("sql_agent.error", error=str(e), task_id=task.task_id)
                result = AgentResult(
                    task_id=task.task_id,
                    agent_type="sql",
                    output=None,
                    error=str(e),
                    latency_ms=(time.monotonic() - t0) * 1000,
                )

            results.append(result)

        span = TraceSpan(
            span_id=str(uuid.uuid4())[:8],
            agent="sql_agent",
            action="query_database",
            input_summary=f"{len(sql_tasks)} task(s)",
            output_summary=f"{sum(1 for r in results if not r.error)} succeeded, "
                           f"{sum(1 for r in results if r.error)} failed",
            started_at=started,
            ended_at=datetime.utcnow(),
        )

        state.results.extend(results)
        state.spans.append(span)
        return state
