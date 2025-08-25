#!/usr/bin/env python3
"""
Streamlit — Access Lockout Agent (LLM + Tools), simplified

What you see:
- Agent Acknowledgement (first reply)
- Agent Resolution (final reply; contains the temporary password on success)
- Action Trace
- Tool Calls (request/response JSONs)
- Incident Snapshot
- Flow Diagram
- KPIs (updated correctly even if not auto-resolved)

No IdP toggle. No username override.
Username is parsed from the free-text message.
"""

import os, time, json, re, requests, streamlit as st
from datetime import datetime
from typing import Dict, Any, List, Tuple

# ----------------------------
# Config (secrets override env)
# ----------------------------
def cfg(name: str, default: str = "") -> str:
    return str(st.secrets.get(name, os.getenv(name, default)))

AZ_ENDPOINT   = cfg("AZURE_OPENAI_ENDPOINT", "")
AZ_KEY        = cfg("AZURE_OPENAI_API_KEY", "")
AZ_VERSION    = cfg("AZURE_OPENAI_API_VERSION", "")
AZ_DEPLOYMENT = cfg("AZURE_OPENAI_DEPLOYMENT", "")

DEFAULT_TOOL_URL = cfg("TOOL_GATEWAY_URL", "http://127.0.0.1:8088").rstrip("/")
DEFAULT_TOOL_KEY = cfg("TOOL_API_KEY", "dev-secret")

# ----------------------------
# Azure OpenAI client
# ----------------------------
from openai import AzureOpenAI
_client = None
def get_client():
    global _client
    if _client is None:
        if not all([AZ_ENDPOINT, AZ_KEY, AZ_VERSION, AZ_DEPLOYMENT]):
            st.error("Azure OpenAI credentials missing. Set env or .streamlit/secrets.toml")
            st.stop()
        _client = AzureOpenAI(
            api_key=AZ_KEY,
            api_version=AZ_VERSION,
            azure_endpoint=AZ_ENDPOINT,
        )
    return _client

# ----------------------------
# LLM helpers
# ----------------------------
def llm_json(system_prompt: str, user_prompt: str, tag: str, max_tokens: int = 300) -> Dict[str, Any]:
    sys_full = (
        "You are a JSON-only API. Always reply with a single valid JSON object and nothing else. "
        "The word JSON is present here to satisfy API constraints. Use only double quotes.\n\n" + system_prompt
    )
    usr_full = f"{user_prompt}\n\nRespond with JSON only."
    c = get_client()
    try:
        resp = c.chat.completions.create(
            model=AZ_DEPLOYMENT,
            messages=[{"role":"system","content":sys_full},
                      {"role":"user","content":usr_full}],
            temperature=0.2,
            max_tokens=max_tokens,
            response_format={"type":"json_object"},
        )
        txt = (resp.choices[0].message.content or "").strip()
        return json.loads(txt)
    except Exception:
        # retry without response_format
        try:
            resp = c.chat.completions.create(
                model=AZ_DEPLOYMENT,
                messages=[{"role":"system","content":sys_full},
                          {"role":"user","content":usr_full}],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            txt = (resp.choices[0].message.content or "").strip()
            try:
                return json.loads(txt)
            except Exception:
                import re as _re
                m = _re.search(r"\{.*\}", txt, flags=_re.DOTALL)
                return json.loads(m.group(0)) if m else {}
        except Exception:
            return {}

def llm_text(system_prompt: str, user_prompt: str, max_tokens: int = 180) -> str:
    c = get_client()
    try:
        resp = c.chat.completions.create(
            model=AZ_DEPLOYMENT,
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(LLM error: {e})"

# ----------------------------
# Username extractor (email > labeled > bare token)
# ----------------------------
def extract_username(text: str) -> str:
    t = (text or "").strip()
    m = re.search(r'[\w\.\-]+@[\w\.\-]+\.\w+', t)
    if m: return m.group(0)
    m = re.search(r'\b(?:i am|iam|this is|user(?:name)?(?: is)?|login)\s*[:\-]?\s*([A-Za-z][A-Za-z0-9\.\-_]{1,})\b', t, re.IGNORECASE)
    if m: return m.group(1)
    m = re.search(r'\b([A-Za-z][A-Za-z0-9\.\-_]{2,})\b', t)
    return m.group(1) if m else ""

# ----------------------------
# Session stores
# ----------------------------
if "incidents" not in st.session_state:
    st.session_state["incidents"] = []
if "kpis" not in st.session_state:
    st.session_state["kpis"] = {"total": 0, "closed": 0, "auto_resolved": 0}
if "tool_logs" not in st.session_state:
    st.session_state["tool_logs"] = []

def _now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def gen_inc_number(): return f"INC{100000 + len(st.session_state['incidents'])}"

# ----------------------------
# Tool calls
# ----------------------------
def ping_gateway(base_url: str) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/health"
    t0 = time.perf_counter(); out = {"ok": False, "elapsed_ms": None}
    try:
        r = requests.get(url, timeout=2.5)
        out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if r.status_code == 200:
            body = r.json(); out["ok"] = bool(body.get("ok")); out["body"] = body
        else:
            out["status_code"] = r.status_code; out["text"] = r.text[:300]
    except Exception as e:
        out["error"] = str(e)
    return out

def call_tool(base_url: str, api_key: str, path: str, payload: Dict[str, Any], timeout: float = 6.0) -> Tuple[bool, Dict[str, Any], int]:
    url = base_url.rstrip("/") + path
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    t0 = time.perf_counter()
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text[:400]}
        st.session_state["tool_logs"].append({
            "when": _now(), "path": path, "request": payload,
            "status_code": r.status_code, "response": body, "elapsed_ms": elapsed_ms
        })
        ok = (200 <= r.status_code < 300)
        return ok, body, r.status_code
    except Exception as e:
        st.session_state["tool_logs"].append({
            "when": _now(), "path": path, "request": payload,
            "status_code": 0, "response": {"error": str(e)}, "elapsed_ms": None
        })
        return False, {"error": str(e)}, 0

# ----------------------------
# LLM agents
# ----------------------------
def llm_intent_and_user(text: str) -> Dict[str, Any]:
    sys_p = (
        "You read end-user IT messages. "
        "Classify intent and extract username/email if available. "
        "For this POC, intent must be 'access_reset' when the message is about password/lockout/login. "
        "Return strict JSON with keys: intent, username (string or null)."
    )
    usr_p = f"Message: {text}"
    out = llm_json(sys_p, usr_p, tag="intent")
    if not out:
        out = {"intent": "access_reset", "username": None}
    if not out.get("username"):
        out["username"] = extract_username(text)
    return out

def llm_plan_action(context: Dict[str, Any]) -> Dict[str, Any]:
    sys_p = (
        "You are a cautious IT agent planner. Decide the next step for an access reset. "
        "Rules: If idp_ok is false → route_identity. "
        "If username is missing or lookup.exists is false → ask_username. "
        "Else → reset_password. "
        "Return strict JSON with keys: action, reason."
    )
    usr_p = json.dumps(context, ensure_ascii=False)
    out = llm_json(sys_p, usr_p, tag="plan")
    if not out:
        idp_ok = context.get("idp_ok", True)
        exists = context.get("lookup", {}).get("exists", True)
        action = "route_identity" if not idp_ok else ("ask_username" if not exists or not context.get("username") else "reset_password")
        out = {"action": action, "reason": "fallback decision"}
    return out

# ----------------------------
# One run of the agent (returns ack + final)
# ----------------------------
def run_agent_once(user_text: str, base_url: str, api_key: str) -> Dict[str, Any]:
    actions: List[str] = []
    tool_logs_start = len(st.session_state["tool_logs"])

    # 1) Understand
    iu = llm_intent_and_user(user_text)
    intent = iu.get("intent", "access_reset")
    username = iu.get("username") or extract_username(user_text)
    actions.append("intent:access_reset")

    # 2) Create incident (+KPIs total)
    inc_number = gen_inc_number()
    incident = {
        "number": inc_number,
        "opened_at": _now(),
        "short_description": "User reports account lockout / password reset",
        "description": user_text,
        "priority": "High",
        "category": "Access",
        "urgency": "High",
        "state": "Open",
        "assignment_group": "Service Desk",
        "knowledge_match": "KB-1001",
        "resolved_at": "",
        "closed_at": "",
        "sla_due": "",
    }
    st.session_state["incidents"].append(incident)
    st.session_state["kpis"]["total"] += 1  # count every new ticket
    actions.append(f"created:{inc_number}")

    # 3) First touch (LLM)
    ack = llm_text(
        "You write concise IT support acknowledgements. 2 sentences max.",
        f"Ticket: {inc_number}. Scenario: Access lockout/password reset. Write a short acknowledgement."
    )
    actions.append("user_update:first_touch")

    # 4) IdP status (no UI toggle; just ask the tool)
    ok, idp_body, _ = call_tool(base_url, api_key, "/tools/v1/idp/status", {})
    idp_ok = ok and bool(idp_body.get("ok", True))
    actions.append(f"idp_ok:{'True' if idp_ok else 'False'}")

    # 5) Plan (and lookup if IdP ok)
    context = {"idp_ok": idp_ok, "username": username, "number": inc_number, "lookup": {}}
    if not idp_ok:
        plan = {"action": "route_identity", "reason": "idp outage"}
    else:
        lok, lookup_body, _ = call_tool(base_url, api_key, "/tools/v1/itsm/dir/user_lookup", {"user": username})
        actions.append("lookup:user")
        context["lookup"] = lookup_body if lok else {"exists": False}
        plan = llm_plan_action(context)

    # 6) Branches
    if plan["action"] == "route_identity":
        incident["state"] = "In Progress"
        incident["assignment_group"] = "Identity"
        final = llm_text(
            "You write concise IT status notes. 2 sentences max.",
            f"Explain to the user that identity provider reports an outage; ticket {inc_number} is routed to Identity."
        )
        actions.extend(["routed:Identity", "worknote:idp_outage"])
        return {
            "ack": ack,
            "final": final,
            "actions": actions,
            "incident": incident,
            "tool_logs": st.session_state["tool_logs"][tool_logs_start:]
        }

    if plan["action"] == "ask_username" or not username:
        final = "Please provide your username or email to proceed with the reset."
        actions.append("clarify:username")
        return {
            "ack": ack,
            "final": final,
            "actions": actions,
            "incident": incident,
            "tool_logs": st.session_state["tool_logs"][tool_logs_start:]
        }

    # 7) Reset + close
    rok, reset_body, _ = call_tool(base_url, api_key, "/tools/v1/itsm/dir/reset_password", {"user": username, "delivery": "chat"})
    if not rok:
        final = f"Could not issue a temporary password at this time. Your ticket {inc_number} remains in progress."
        actions.append("tool_error:reset_password")
        return {
            "ack": ack,
            "final": final,
            "actions": actions,
            "incident": incident,
            "tool_logs": st.session_state["tool_logs"][tool_logs_start:]
        }

    temp_password = reset_body.get("temp_password", "")
    actions.extend(["tool_exec:reset_password", "worknote:reset_issued"])
    incident["state"] = "Resolved"; incident["resolved_at"] = _now()
    incident["state"] = "Closed";   incident["closed_at"] = _now()
    actions.extend(["auto_resolved", "closed"])

    # KPIs on closure
    st.session_state["kpis"]["closed"] += 1
    st.session_state["kpis"]["auto_resolved"] += 1

    final = llm_text(
        "You write concise IT resolutions. 2 sentences max.",
        f"Tell the user the temporary password and that ticket {inc_number} is closed. Password: {temp_password}. TTL: 15 minutes."
    )
    if not final or "password" not in final.lower():
        final = (
            f"A temporary password has been issued for {username}. "
            f"Ticket {inc_number} is now closed. "
            f"Your temporary password is: {temp_password}. "
            f"Please change it within 15 minutes."
        )

    return {
        "ack": ack,
        "final": final,
        "actions": actions,
        "incident": incident,
        "tool_logs": st.session_state["tool_logs"][tool_logs_start:]
    }

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Access Lockout Agent (LLM + Tools)", layout="wide")
st.title("Access Lockout Agent — LLM + Tools POC (Simplified)")

# Sidebar: gateway and status
with st.sidebar:
    st.header("Tool Gateway")
    gw_url = st.text_input("Gateway URL", value=DEFAULT_TOOL_URL, key="gw_url")
    gw_key = st.text_input("API key", value=DEFAULT_TOOL_KEY, key="gw_key")
    status = ping_gateway(gw_url)
    if status.get("ok"):
        st.write(f"Status: UP, {status.get('elapsed_ms')} ms")
        with st.expander("Health JSON", expanded=False):
            st.json(status.get("body", {}))
    else:
        st.write("Status: DOWN")
    st.divider()
    if not all([AZ_ENDPOINT, AZ_KEY, AZ_VERSION, AZ_DEPLOYMENT]):
        st.warning("Azure OpenAI credentials are missing.")

# Chat
default_prompt = "I'm locked out and need a password reset. My username is john.smith"
user_text = st.text_area("User message", value=default_prompt, height=120)
run_btn = st.button("Run Agent")

col_left, col_right = st.columns([3, 2])

if run_btn:
    result = run_agent_once(user_text=user_text, base_url=gw_url, api_key=gw_key)

    with col_left:
        st.subheader("Agent Acknowledgement")
        st.write(result["ack"])

        st.subheader("Agent Resolution")
        st.write(result["final"])  # this is where the password shows up on success

        st.subheader("Action Trace")
        st.code(" -> ".join(result["actions"]))

        st.subheader("Flow Diagram")
        taken = set(a.split(":")[0] for a in result["actions"])
        dot = [
            'digraph G {',
            'rankdir=LR;',
            'node [shape=box, style="rounded,filled", fillcolor=white];',
            f'"understand" [label="understand" {"fillcolor=lightgreen" if "intent" in taken else ""}];',
            f'"create" [label="create" {"fillcolor=lightgreen" if any(x.startswith("created:") for x in result["actions"]) else ""}];',
            f'"first_touch" [label="first_touch" {"fillcolor=lightgreen" if "user_update" in taken else ""}];',
            f'"check_idp" [label="check_idp" {"fillcolor=lightgreen" if any(x.startswith("idp_ok:") for x in result["actions"]) else ""}];',
            f'"lookup" [label="lookup_user" {"fillcolor=lightgreen" if "lookup" in taken else ""}];',
            f'"reset" [label="reset_password" {"fillcolor=lightgreen" if "tool_exec" in taken else ""}];',
            f'"close" [label="close" {"fillcolor=lightgreen" if "closed" in result["actions"] else ""}];',
            '"understand" -> "create";',
            '"create" -> "first_touch";',
            '"first_touch" -> "check_idp";',
            '"check_idp" -> "lookup";',
            '"lookup" -> "reset";',
            '"reset" -> "close";',
            '}'
        ]
        st.graphviz_chart("\n".join(dot))

        st.subheader("Incident Snapshot")
        st.json(result["incident"])

    with col_right:
        st.subheader("Tool Calls")
        for i, rec in enumerate(result["tool_logs"], 1):
            with st.expander(f"{i}. {rec['path']}  [{rec['status_code']}]  {rec['elapsed_ms']} ms", expanded=False):
                st.write("Request");  st.json(rec["request"])
                st.write("Response"); st.json(rec["response"])

        st.subheader("Session KPIs")
        k = st.session_state["kpis"]
        total = max(k["total"], 1)
        auto_pct = round((k["auto_resolved"] / total) * 100, 2)
        st.write(f"Total incidents: {k['total']}")
        st.write(f"Closed incidents: {k['closed']}")
        st.write(f"Auto-resolved percent: {auto_pct}")
else:
    st.info("Type any lockout message (with a name) and click Run Agent.")
