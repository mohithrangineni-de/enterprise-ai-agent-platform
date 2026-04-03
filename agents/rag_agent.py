"""
agents/rag_agent.py

RAG Agent — retrieves relevant document chunks using hybrid search
(vector similarity + BM25 keyword), then synthesizes a focused answer.

This agent re-uses the same FAISS vector store from the prior RAG project,
extended with metadata filtering (doc_type, date range).
"""

from __future__ import annotations
import uuid
import time
from datetime import datetime

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

from core.state import AgentState, AgentResult, TraceSpan
from core.config import settings
from tools.vector_store import VectorStoreTool
from observability.logger import get_logger

log = get_logger(__name__)

RAG_SYSTEM = """You are a document analyst. Given retrieved document chunks,
answer the task concisely and cite the source for each key claim.

Format your answer as:
- Key finding 1 [source: <filename>#<page>]
- Key finding 2 [source: <filename>#<page>]
...
Summary: <1-2 sentence synthesis>

If no relevant information is found, say "No relevant documents found."
"""

RAG_HUMAN = """Task: {task_description}

Retrieved chunks:
{chunks}

User question: {question}"""


class RAGAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
        self.embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
        self.vector_store = VectorStoreTool(embeddings=self.embeddings)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM),
            ("human", RAG_HUMAN),
        ])
        self.chain = self.prompt | self.llm

    async def run(self, state: AgentState) -> AgentState:
        started = datetime.utcnow()
        t0 = time.monotonic()

        rag_tasks = [t for t in state.tasks if t.agent_type == "rag"]
        if not rag_tasks:
            return state

        results = []

        for task in rag_tasks:
            log.info("rag_agent.start", task_id=task.task_id)
            try:
                # Retrieve top-k chunks (hybrid: vector + keyword)
                chunks = await self.vector_store.search(
                    query=f"{task.description} — {state.question}",
                    k=settings.rag_top_k,
                )

                if not chunks:
                    log.warning("rag_agent.no_results", task_id=task.task_id)

                # Format chunks for the prompt
                chunk_text = "\n\n".join(
                    f"[{c['source']}]\n{c['content']}" for c in chunks
                )

                response = await self.chain.ainvoke({
                    "task_description": task.description,
                    "chunks": chunk_text or "No documents available.",
                    "question": state.question,
                })

                latency = (time.monotonic() - t0) * 1000
                sources = [c["source"] for c in chunks]

                result = AgentResult(
                    task_id=task.task_id,
                    agent_type="rag",
                    output={
                        "answer": response.content,
                        "chunks_retrieved": len(chunks),
                        "chunks": [{"source": c["source"], "snippet": c["content"][:200]} for c in chunks],
                    },
                    sources=sources,
                    latency_ms=latency,
                )
                log.info("rag_agent.done", chunks=len(chunks), latency_ms=latency)

            except Exception as e:
                log.error("rag_agent.error", error=str(e), task_id=task.task_id)
                result = AgentResult(
                    task_id=task.task_id,
                    agent_type="rag",
                    output=None,
                    error=str(e),
                    latency_ms=(time.monotonic() - t0) * 1000,
                )

            results.append(result)

        span = TraceSpan(
            span_id=str(uuid.uuid4())[:8],
            agent="rag_agent",
            action="retrieve_and_synthesize",
            input_summary=f"{len(rag_tasks)} task(s)",
            output_summary=f"{sum(1 for r in results if not r.error)} succeeded",
            started_at=started,
            ended_at=datetime.utcnow(),
        )

        state.results.extend(results)
        state.spans.append(span)
        return state
