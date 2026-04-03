"""
agents/response_agent.py

Response Agent — the final step in the pipeline.
Takes all agent results and synthesizes a coherent, cited answer
with a confidence score.
"""

from __future__ import annotations
import uuid
import time
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from core.state import AgentState, TraceSpan
from core.config import settings
from observability.logger import get_logger

log = get_logger(__name__)

RESPONSE_SYSTEM = """You are a senior enterprise analytics advisor.
Synthesize the findings from multiple AI agents into a clear, executive-ready answer.

Guidelines:
- Lead with the direct answer to the question
- Support claims with data from the SQL agent results
- Reference relevant documents from the RAG agent
- Include any analysis from the Python agent
- Use bullet points for structured findings
- End with a "Confidence:" line (0.0–1.0) based on completeness of available data
- Be concise — target 150–250 words

If agents produced errors, acknowledge the gaps and work with what's available.
"""

RESPONSE_HUMAN = """User question: {question}

Agent findings:
{findings}

Generate the final synthesized answer."""


class ResponseAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.2,  # slight temperature for more natural synthesis
            api_key=settings.openai_api_key,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RESPONSE_SYSTEM),
            ("human", RESPONSE_HUMAN),
        ])
        self.chain = self.prompt | self.llm

    def _format_findings(self, state: AgentState) -> str:
        """Format all agent results for the synthesis prompt."""
        sections = []

        for result in state.results:
            if result.error:
                sections.append(f"[{result.agent_type.upper()} AGENT — ERROR]\n{result.error}")
                continue

            if result.agent_type == "sql":
                data = result.output
                rows_preview = "\n".join(
                    str(row) for row in (data.get("rows") or [])[:10]
                )
                sections.append(
                    f"[SQL AGENT]\nQuery: {data.get('sql', '')[:200]}\n"
                    f"Columns: {data.get('columns', [])}\n"
                    f"Sample rows ({data.get('row_count', 0)} total):\n{rows_preview}"
                )

            elif result.agent_type == "rag":
                sections.append(
                    f"[RAG AGENT]\n{result.output.get('answer', '')}"
                )

            elif result.agent_type == "python":
                sections.append(
                    f"[PYTHON AGENT]\nOutput:\n{result.output.get('stdout', '')}"
                )

        return "\n\n---\n\n".join(sections) if sections else "No agent results available."

    def _parse_confidence(self, text: str) -> float:
        """Extract the confidence score from the response text."""
        import re
        match = re.search(r"confidence[:\s]+([0-9.]+)", text, re.IGNORECASE)
        if match:
            try:
                return min(1.0, max(0.0, float(match.group(1))))
            except ValueError:
                pass
        return 0.5  # default

    def _collect_sources(self, state: AgentState) -> list[str]:
        sources = []
        for result in state.results:
            sources.extend(result.sources)
        return list(dict.fromkeys(sources))  # deduplicate, preserve order

    async def run(self, state: AgentState) -> AgentState:
        started = datetime.utcnow()
        t0 = time.monotonic()

        log.info("response_agent.start", session=state.session_id, results=len(state.results))

        findings = self._format_findings(state)

        try:
            response = await self.chain.ainvoke({
                "question": state.question,
                "findings": findings,
            })

            answer = response.content
            confidence = self._parse_confidence(answer)
            latency = (time.monotonic() - t0) * 1000
            sources = self._collect_sources(state)

            # Count total tokens across all results
            total_tokens = sum(
                sum(r.token_usage.values()) for r in state.results
            ) + (response.usage_metadata.get("total_tokens", 0) if hasattr(response, "usage_metadata") else 0)

            # Sum all latencies
            total_latency = sum(r.latency_ms for r in state.results) + latency

            log.info("response_agent.done",
                     confidence=confidence,
                     sources=len(sources),
                     latency_ms=latency)

        except Exception as e:
            log.error("response_agent.error", error=str(e))
            answer = f"I encountered an error synthesizing the results: {e}"
            confidence = 0.0
            sources = []
            total_tokens = 0
            total_latency = (time.monotonic() - t0) * 1000

        span = TraceSpan(
            span_id=str(uuid.uuid4())[:8],
            agent="response_agent",
            action="synthesize_answer",
            input_summary=f"{len(state.results)} agent results",
            output_summary=f"answer_len={len(answer)}, confidence={confidence:.2f}",
            started_at=started,
            ended_at=datetime.utcnow(),
        )

        state.final_answer = answer
        state.answer_confidence = confidence
        state.cited_sources = sources
        state.total_tokens = total_tokens
        state.total_latency_ms = total_latency
        state.spans.append(span)
        return state
