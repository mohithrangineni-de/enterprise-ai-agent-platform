"""
observability/dashboard.py

Streamlit monitoring dashboard.
Run with: streamlit run observability/dashboard.py

Shows:
- Live query log
- Agent trace waterfall
- Token usage per session
- Error rate by agent
"""

import streamlit as st
import httpx
import json
from datetime import datetime

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AI Agent Platform — Observability",
    page_icon="🔭",
    layout="wide",
)

st.title("🔭 Enterprise AI Agent Platform")
st.caption("Observability Dashboard — traces, latency, token usage")

# ─── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.header("Query Tester")
question = st.sidebar.text_area("Ask a question", value="Why did revenue drop last quarter?")
run_btn = st.sidebar.button("Run Query", type="primary")

# ─── Main layout ──────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

if run_btn and question:
    with st.spinner("Running agent pipeline..."):
        try:
            resp = httpx.post(f"{API_BASE}/query", json={"question": question}, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            st.session_state["last_result"] = data

        except Exception as e:
            st.error(f"API Error: {e}")

if "last_result" in st.session_state:
    data = st.session_state["last_result"]

    with col1:
        st.subheader("Answer")
        st.markdown(data["answer"])

        st.subheader("Trace Details")
        trace_id = data["trace_id"]

        try:
            trace_resp = httpx.get(f"{API_BASE}/traces/{trace_id}", timeout=10)
            trace = trace_resp.json()

            for span in trace.get("spans", []):
                color = "🟢" if span["status"] == "ok" else "🔴"
                st.markdown(
                    f"{color} **{span['name']}** — {span['duration_ms']:.0f}ms"
                    + (f" ❌ {span['error']}" if span.get('error') else "")
                )
        except Exception:
            st.caption(f"Trace ID: {trace_id}")

    with col2:
        st.subheader("Metrics")
        st.metric("Confidence", f"{data['confidence']:.0%}")
        st.metric("Latency", f"{data['total_latency_ms']:.0f}ms")
        st.metric("Tokens Used", data["total_tokens"])

        st.subheader("Agents Used")
        for agent in data["agents_used"]:
            st.badge(agent)

        st.subheader("Sources")
        for s in data["sources"]:
            st.caption(f"📄 {s}")

        st.subheader("Routing")
        st.caption(data["routing_rationale"])

# ─── Footer ───────────────────────────────────────────────────────────────
st.divider()
st.caption("Enterprise AI Agent Platform · github.com/yourname/enterprise-ai-agent-platform")
