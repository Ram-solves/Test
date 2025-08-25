#!/usr/bin/env python3
"""
Streamlit adapter UI that calls the existing backend agent in sn_langgraph_llm (or sn_langgraph_tool).

It expects your backend module to expose:
  - FlowState (Pydantic/BaseModel used for graph state)
  - build_graph() -> StateGraph (which we compile here), OR compile_graph() -> compiled graph
  - store with .get(number) to fetch the incident snapshot
  - TOOL_GATEWAY_URL / TOOL_API_KEY (optional; just for display)

Run:
  streamlit run streamlit_sn_adapter.py

Tip:
  Put this file next to your backend script so import works, e.g.:
    sn_langgraph_llm.py
    tool_gateway.py
    streamlit_sn_adapter.py
"""

import os
import time
import json
import requests
import streamlit as st

# ---------- import your backend ----------
backend = None
mod_name = None
try:
    import sn_langgraph_llm as backend  # preferred name you used earlier
    mod_name = "sn_langgraph_llm"
except Exception:
    try:
        import sn_langgraph_tool as backend  # if your file is named this
        mod_name = "sn_langgraph_tool"
    except Exception as e:
        st.stop()

# ---------- get compiled graph ----------
GRAPH = None
if hasattr(backend, "compile_graph"):
    GRAPH = backend.compile_graph()
elif hasattr(backend, "build_graph"):
    GRAPH = backend.build_graph().compile()
else:
    st.error("Backend does not expose build_graph() or compile_graph(). Add one of them.")
    st.stop()

# ---------- figure out FlowState & store ----------
FlowState = getattr(backend, "FlowState", None)
store = getattr(backend, "store", None)
if FlowState is None or store is None:
    st.error("Backend must export FlowState and store.")
    st.stop()

# ---------- get tool gateway info (optional) ----------
TOOL_GATEWAY_URL = getattr(backend, "TOOL_GATEWAY_URL", os.getenv("TOOL_GATEWAY_URL", "http://127.0.0.1:8088")).rstrip("/")
TOOL_API_KEY = getattr(backend, "TOOL_API_KEY", os.getenv("TOOL_API_KEY", "dev-secret"))

# ---------- helpers ----------
def ping_gateway(url: str) -> dict:
    t0 = time.perf_counter()
    out = {"ok": False, "elapsed_ms": None}
    try:
        r = requests.get(url.rstrip("/") + "/health", timeout=2.5)
        out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if r.status_code == 200:
            body = r.json()
            out["ok"] = bool(body.get("ok"))
            out["body"] = body
        else:
            out["status_code"] = r.status_code
            out["text"] = r.text[:300]
    except Exception as e:
        out["error"] = str(e)
    return out

def run_backend_once(user_text: str):
    """
    Invoke the backend graph with a fresh FlowState(user_text=...).
    Returns the state the backend produces.
    """
    st.session_state.setdefault("last_tool", {})
    state_in = FlowState(user_text=user_text)
    state_out = GRAPH.invoke(state_in)

    # Try to extract tool call info if your backend exposes it on state
    tool_info = {}
    for attr in ("tool_name", "tool_args", "tool_result"):
        if hasattr(state_out, attr):
            tool_info[attr] = getattr(state_out, attr)
    st.session_state["last_tool"] = tool_info

    return state_out

# ---------- UI ----------
st.set_page_config(page_title="ServiceNow Agent POC (backend adapter)", layout="wide")
st.title("ServiceNow Agent POC — Backend Adapter")
st.caption(f"Using backend module: {mod_name}")

with st.sidebar:
    st.header("Tool Gateway")
    st.text_input("Gateway URL", value=TOOL_GATEWAY_URL, key="gw_url")
    st.text_input("API key", value=TOOL_API_KEY, key="gw_key", type="password")

    status = ping_gateway(st.session_state["gw_url"])
    if status.get("ok"):
        st.write(f"Status: UP, {status.get('elapsed_ms')} ms")
        with st.expander("Health JSON", expanded=False):
            st.json(status.get("body", {}))
    else:
        st.write("Status: DOWN")
        err = status.get("error") or f"HTTP {status.get('status_code')}"
        st.caption(err)

    if st.button("Refresh"):
        st.rerun()

st.subheader("Chat")
default_msg = "I'm locked out and need a password reset. I am ram."
user_msg = st.text_area("User message", value=default_msg, height=120)
run_btn = st.button("Run Agent")

col1, col2 = st.columns([3, 2])

if run_btn:
    state = run_backend_once(user_msg)

    with col1:
        st.subheader("Agent Reply")
        st.write(getattr(state, "user_message", "") or "(no reply)")

        st.subheader("Action Trace")
        actions = getattr(state, "actions", []) or []
        st.code(" -> ".join(actions) if actions else "(no actions)")

        st.subheader("Incident Snapshot")
        number = getattr(state, "number", "") or ""
        if number and store is not None:
            try:
                snap = store.get(number)
            except Exception:
                snap = None
            if snap:
                st.json(snap)
            else:
                st.write("(no snapshot from store)")
        else:
            st.write("(no incident number)")

    with col2:
        st.subheader("Last Tool Call (if any)")
        tool_info = st.session_state.get("last_tool", {})
        if tool_info:
            st.json(tool_info)
        else:
            st.write("No tool call recorded on state. This is normal if the flow did not need a tool.")

        st.subheader("Backend Details")
        st.write("FlowState fields on output state:")
        # dump a small view of state
        try:
            as_dict = state.model_dump()  # Pydantic v2
        except Exception:
            # Pydantic v1 fallback
            try:
                as_dict = state.dict()
            except Exception:
                as_dict = {k: getattr(state, k) for k in dir(state) if not k.startswith("_")}
        st.json({k: as_dict.get(k) for k in sorted(as_dict.keys()) if k in ("user_text","intent","number","tool_name","tool_args","tool_result","error")})

else:
    st.info("Enter a message and press Run Agent. This UI will call your backend module directly and display the outputs.")
