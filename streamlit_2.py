import os, json
import pandas as pd
import streamlit as st

from sn_langgraph_llm import (
    IncidentStore, FlowState, build_graph, load_store, compute_kpis,
    AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, AZURE_DEPLOYMENT
)

st.set_page_config(page_title="ServiceNow Agentic POC", layout="wide")

# ---- Env sanity ----
missing = [k for k,v in {
    "AZURE_OPENAI_ENDPOINT": AZURE_ENDPOINT,
    "AZURE_OPENAI_API_KEY": AZURE_API_KEY,
    "AZURE_OPENAI_API_VERSION": AZURE_API_VERSION,
    "AZURE_OPENAI_DEPLOYMENT": AZURE_DEPLOYMENT,
}.items() if not v]
if missing:
    st.error(f"Missing Azure OpenAI env vars: {', '.join(missing)} (check your .env in this folder)")
    st.stop()

# ---- Session
if "store" not in st.session_state:
    st.session_state.store = None
if "graph" not in st.session_state:
    st.session_state.graph = None
if "history" not in st.session_state:
    st.session_state.history = []  # [(user, reply, actions)]
if "pending_state" not in st.session_state:
    st.session_state.pending_state = None

# ---- Page
st.title("ServiceNow-style Agentic Chat (LangGraph + Azure OpenAI)")

# ---------- Diagram helpers ----------
def static_flow_dot() -> str:
    return r"""
digraph G {
  rankdir=LR;
  graph [pad="0.2"];
  node [shape=box, style="rounded,filled", color="#bbbbbb", fillcolor="#f9f9f9", fontsize=11];
  edge [color="#888888"];

  intent  [label="intent_agent (LLM)\n{intent, fields}"];
  status  [label="status"];
  dedupe  [label="dedupe_agent (TF-IDF)"];
  ensure  [label="ensure_min_fields (LLM Q)\n(asks 1 question)"];
  triage  [label="triage_agent (LLM)\n{category, priority, urgency}"];
  create  [label="create (store)\nINC number"];
  first   [label="user_comms_agent (LLM)\nfirst-touch"];
  routekb [label="routing + kb (rules)"];
  resolve [label="resolver_agent (LLM+rule)\n+ guardrail"];
  other   [label="other"];
  end     [shape=doublecircle, label="END", fillcolor="#efefef"];

  intent -> status [label="intent=status"];
  intent -> dedupe [label="intent=create"];
  intent -> other  [label="else"];

  dedupe -> ensure;
  ensure -> triage;
  triage -> create -> first -> routekb -> resolve -> end;
  status -> end;
  other -> end;
}
"""

NODE_MAP = {
    "triage": "triage",
    "created": "create",
    "user_update:first_touch": "first",
    "routed": "routekb",
    "kb": "routekb",
    "auto_resolved": "resolve",
    "worknote:drafted": "resolve",
    "status": "status",
    "dedup": "dedupe",
    "clarify:request": "ensure",
    "closed": "end",
    "other_intent": "other",
}

def trace_to_nodes(actions):
    visited = {"intent"}  # entry always
    for a in actions or []:
        key = a.split(":")[0]
        visited.add(NODE_MAP.get(key, key))
    return visited

def dynamic_trace_dot(actions) -> str:
    visited = trace_to_nodes(actions)
    # Build DOT with highlighted nodes
    nodes = {
        "intent":  'intent  [label="intent_agent (LLM)\\n{intent, fields}"]',
        "status":  'status  [label="status"]',
        "dedupe":  'dedupe  [label="dedupe_agent (TF-IDF)"]',
        "ensure":  'ensure  [label="ensure_min_fields (LLM Q)\\n(asks 1 question)"]',
        "triage":  'triage  [label="triage_agent (LLM)\\n{category, priority, urgency}"]',
        "create":  'create  [label="create (store)\\nINC number"]',
        "first":   'first   [label="user_comms_agent (LLM)\\nfirst-touch"]',
        "routekb": 'routekb [label="routing + kb (rules)"]',
        "resolve": 'resolve [label="resolver_agent (LLM+rule)\\n+ guardrail"]',
        "other":   'other   [label="other"]',
        "end":     'end     [shape=doublecircle, label="END"]',
    }
    def style(n):
        if n in visited:
            return 'color="#2563eb", fillcolor="#dbeafe"'
        return 'color="#bbbbbb", fillcolor="#f9f9f9"'

    dot = ['digraph G {',
           '  rankdir=LR;',
           '  graph [pad="0.2"];',
           '  node [shape=box, style="rounded,filled", fontsize=11];',
           '  edge [color="#888888"];']
    for name, base in nodes.items():
        dot.append(f'  {name} [{style(name)} {base[base.find("[")+1:]}];')

    dot += [
      '  intent -> status [label="intent=status"];',
      '  intent -> dedupe [label="intent=create"];',
      '  intent -> other  [label="else"];',
      '  dedupe -> ensure;',
      '  ensure -> triage;',
      '  triage -> create -> first -> routekb -> resolve -> end;',
      '  status -> end;',
      '  other -> end;',
      '}'
    ]
    return "\n".join(dot)

# ---------- Sidebar: data controls ----------
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
                st.session_state.history.clear()
                st.session_state.pending_state = None
                st.success(f"Loaded {len(df)} rows")
            except Exception as e:
                st.error(f"Failed to load: {e}")

    if st.button("Load Default Mock"):
        try:
            from sn_langgraph_llm import DEFAULT_DATA_PATH, DEFAULT_SHEET_NAME
            st.session_state.store = load_store(DEFAULT_DATA_PATH, DEFAULT_SHEET_NAME)
            st.session_state.graph = build_graph().compile()
            st.session_state.history.clear()
            st.session_state.pending_state = None
            st.success(f"Loaded default: {DEFAULT_DATA_PATH}")
        except Exception as e:
            st.error(f"Failed to load default: {e}")

# ---------- Layout
top1, top2 = st.columns([2,1])

with top1:
    st.subheader("Chat")
    if st.session_state.store is None or st.session_state.graph is None:
        st.info("Load data from the sidebar to start chatting.")
    else:
        user_text = st.text_input("Say something (e.g., 'I'm locked out', 'Status on INC123456')")
        if st.button("Send", type="secondary"):
            import sn_langgraph_llm as mod
            mod.store = st.session_state.store

            # If we’re waiting for a clarification, merge user input into fields
            if st.session_state.pending_state:
                ps = st.session_state.pending_state
                if "short_description" in (ps.get("missing_fields") or []):
                    ps["intent_fields"]["short_description"] = user_text
                    ps["short"] = user_text
                st_state = FlowState(**{**ps, "user_text": user_text, "await_input": False, "missing_fields": []})
                st.session_state.pending_state = None
            else:
                st_state = FlowState(user_text=user_text)

            out = st.session_state.graph.invoke(st_state)
            from sn_langgraph_llm import _to_state_dict
            state = _to_state_dict(out)

            st.session_state.history.append((user_text, state.get("user_message"), list(state.get("actions") or [])))

            if state.get("await_input"):
                st.session_state.pending_state = state

        # Render history
        for i, (u, r, a) in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**You:** {u}")
            st.markdown(f"**Agent:** {r}")
            with st.expander("Actions trace (diagram)"):
                st.graphviz_chart(dynamic_trace_dot(a), use_container_width=True)
            st.markdown("---")

with top2:
    st.subheader("Agent Flow (static)")
    st.graphviz_chart(static_flow_dot(), use_container_width=True)

    st.subheader("KPIs")
    if st.session_state.store is not None:
        kpis = compute_kpis(st.session_state.store)
        st.json(kpis)
        st.markdown("**Open incidents (sample):**")
        sample = st.session_state.store.list_open().head(10)
        st.dataframe(sample, use_container_width=True)
    else:
        st.info("Load data to see KPIs and queue.")

st.markdown("---")
st.caption("Only user_comms_agent talks to the user. All other nodes are back-office. Replace IncidentStore with ServiceNow Table API for production.")