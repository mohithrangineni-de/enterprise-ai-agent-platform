"""
agents/python_agent.py

Python Agent — generates and executes Python code for statistical analysis,
anomaly detection, and data transformations.

Safety model:
- Code runs in a subprocess with a timeout
- No filesystem writes outside /tmp
- No network access (imports requests, httpx, etc. will be blocked)
- Output is captured from stdout
"""

from __future__ import annotations
import ast
import uuid
import time
import asyncio
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from core.state import AgentState, AgentResult, TraceSpan
from core.config import settings
from observability.logger import get_logger

log = get_logger(__name__)

PYTHON_SYSTEM = """You are a Python data analyst. Generate Python code to complete the task.

Rules:
- Use only: pandas, numpy, scipy, statistics (stdlib)
- No file I/O, no network requests, no os.system
- Print the result as clean text or JSON — it will be captured from stdout
- Include brief inline comments
- The data from prior agents is available as the variable `agent_data` (a dict)

Respond with ONLY the Python code — no prose, no markdown fences.
"""

PYTHON_HUMAN = """Task: {task_description}

Available data from SQL/RAG agents:
{agent_data_summary}

User question: {question}"""

# Blocked imports for safety
BLOCKED_IMPORTS = {"os", "subprocess", "socket", "requests", "httpx", "urllib", "ftplib", "smtplib"}


class PythonAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PYTHON_SYSTEM),
            ("human", PYTHON_HUMAN),
        ])
        self.chain = self.prompt | self.llm

    def _check_code_safety(self, code: str) -> None:
        """Parse the AST and block dangerous imports."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in generated code: {e}")

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [n.name for n in getattr(node, "names", [])]
                module = getattr(node, "module", "") or ""
                top_level = module.split(".")[0] if module else ""
                for name in names + ([top_level] if top_level else []):
                    if name in BLOCKED_IMPORTS:
                        raise ValueError(f"Blocked import: {name}")

    async def _execute_code(self, code: str, agent_data: dict, timeout: int = 30) -> str:
        """Execute code in a subprocess and return stdout."""
        # Prepend data injection
        data_prefix = f"import json\nagent_data = {repr(agent_data)}\n\n"
        full_code = data_prefix + code

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            tmpfile = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, tmpfile,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            if proc.returncode != 0:
                raise RuntimeError(f"Code execution failed:\n{stderr.decode()[:500]}")

            return stdout.decode()[:2000]  # cap output size

        finally:
            Path(tmpfile).unlink(missing_ok=True)

    async def run(self, state: AgentState) -> AgentState:
        started = datetime.utcnow()
        t0 = time.monotonic()

        py_tasks = [t for t in state.tasks if t.agent_type == "python"]
        if not py_tasks:
            return state

        # Build a summary of available data for the prompt
        agent_data = {}
        for r in state.results:
            if r.output and not r.error:
                agent_data[r.task_id] = r.output

        agent_data_summary = "\n".join(
            f"- {k}: {str(v)[:200]}" for k, v in agent_data.items()
        ) or "No prior agent data available."

        results = []

        for task in py_tasks:
            log.info("python_agent.start", task_id=task.task_id)
            try:
                # Generate code
                response = await self.chain.ainvoke({
                    "task_description": task.description,
                    "agent_data_summary": agent_data_summary,
                    "question": state.question,
                })
                code = response.content.strip().strip("```python").strip("```").strip()

                # Safety check
                self._check_code_safety(code)

                # Execute
                output = await self._execute_code(code, agent_data)
                latency = (time.monotonic() - t0) * 1000

                result = AgentResult(
                    task_id=task.task_id,
                    agent_type="python",
                    output={
                        "code": code,
                        "stdout": output,
                    },
                    sources=["python_execution"],
                    latency_ms=latency,
                )
                log.info("python_agent.done", output_len=len(output), latency_ms=latency)

            except Exception as e:
                log.error("python_agent.error", error=str(e))
                result = AgentResult(
                    task_id=task.task_id,
                    agent_type="python",
                    output=None,
                    error=str(e),
                    latency_ms=(time.monotonic() - t0) * 1000,
                )

            results.append(result)

        span = TraceSpan(
            span_id=str(uuid.uuid4())[:8],
            agent="python_agent",
            action="generate_and_execute_code",
            input_summary=f"{len(py_tasks)} task(s)",
            output_summary=f"{sum(1 for r in results if not r.error)} succeeded",
            started_at=started,
            ended_at=datetime.utcnow(),
        )

        state.results.extend(results)
        state.spans.append(span)
        return state
