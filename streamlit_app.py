import os, json
import pandas as pd
import streamlit as st

from sn_langgraph_llm import (
    IncidentStore, FlowState, build_graph, load_store, compute_kpis,
    AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, AZURE_DEPLOYMENT,
)

st.set_page_config(page_title="ServiceNow Agentic POC", layout="wide")

# Basic env checks
missing = [k for k,v in {
    "AZURE_OPENAI_ENDPOINT": AZURE_ENDPOINT,
    "AZURE_OPENAI_API_KEY": AZURE_API_KEY,
    "AZURE_OPENAI_API_VERSION": AZURE_API_VERSION,
    "AZURE_OPENAI_DEPLOYMENT": AZURE_DEPLOYMENT,
}.items() if not v]
if missing:
    st.error(f"Missing Azure OpenAI env vars: {', '.join(missing)}")
    st.stop()

# Session state: store + graph
if "store" not in st.session_state:
    st.session_state.store = None
if "graph" not in st.session_state:
    st.session_state.graph = None
if "history" not in st.session_state:
    st.session_state.history = []  # [(user, reply, actions)]

st.title("ServiceNow-style Agentic Chat (LangGraph + Azure OpenAI)")

# Data loader sidebar
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload incidents CSV/Excel", type=["csv","xlsx","xls"])
    if st.button("Load Data", type="primary"):
        if uploaded is None:
            st.warning("Please upload a CSV/Excel first.")
        else:
            try:
                if uploaded.name.lower().endswith((".xlsx",".xls")):
                    df = pd.read_excel(uploaded)
                else:
                    df = pd.read_csv(uploaded)
                st.session_state.store = IncidentStore(df.fillna(""))
                st.session_state.graph = build_graph().compile()
                st.success(f"Loaded {len(df)} rows")
            except Exception as e:
                st.error(f"Failed to load: {e}")

    if st.button("Load Default Mock"):
        try:
            from sn_langgraph_llm import DEFAULT_DATA_PATH, DEFAULT_SHEET_NAME
            st.session_state.store = load_store(DEFAULT_DATA_PATH, DEFAULT_SHEET_NAME)
            st.session_state.graph = build_graph().compile()
            st.success(f"Loaded default: {DEFAULT_DATA_PATH}")
        except Exception as e:
            st.error(f"Failed to load default: {e}")

# Main chat column
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Chat")
    if st.session_state.store is None or st.session_state.graph is None:
        st.info("Load data from the sidebar to start chatting.")
    else:
        user_text = st.text_input("Say something like: 'I'm locked out; need password reset' or 'Where are we on my ticket INC100234'")
        if st.button("Send", type="secondary"):
            from sn_langgraph_llm import store as global_store
            # Point module-global store to session store
            import sn_langgraph_llm as mod
            mod.store = st.session_state.store

            st_state = FlowState(user_text=user_text)
            out = st.session_state.graph.invoke(st_state)
            st.session_state.history.append((user_text, out.user_message, list(out.actions)))

        # Render history
        for i, (u, r, a) in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**You:** {u}")
            st.markdown(f"**Agent:** {r}")
            with st.expander("Actions trace"):
                st.write(", ".join(a))

with col2:
    st.subheader("Queue & KPIs")
    if st.session_state.store is not None:
        kpis = compute_kpis(st.session_state.store)
        st.json(kpis)
        st.markdown("**Open incidents (sample):**")
        sample = st.session_state.store.list_open().head(10)
        st.dataframe(sample)
    else:
        st.info("KPIs will appear after loading data.")

st.markdown("---")
st.caption("This POC performs LLM-backed intent & triage, deterministic routing & guardrails, and writes updates to the in-memory store. Replace IncidentStore with ServiceNow Table API for production.")