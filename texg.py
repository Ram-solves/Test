# ---------- Session stores ----------
st.set_page_config(page_title="Access Lockout Agent — Chat", layout="wide")

if "chat" not in st.session_state:
    st.session_state["chat"] = []  # list of {"role": "user"/"assistant", "content": str}
if "incidents" not in st.session_state:
    st.session_state["incidents"] = []  # each dict snapshot
if "kpis" not in st.session_state:
    st.session_state["kpis"] = {"total": 0, "closed": 0, "auto_resolved": 0}
if "tool_logs" not in st.session_state:
    st.session_state["tool_logs"] = []  # all runs
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None

def _now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def gen_inc_number(): return f"INC{100000 + len(st.session_state['incidents'])}"

# ---------- Tools ----------
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

# ---------- Planner ----------
def llm_intent_and_user(text: str) -> Dict[str, Any]:
    sys_p = (
        "You read end-user IT messages. "
        "Classify intent and extract username/email if available. "
        "For this POC, intent must be 'access_reset' when the message is about password/lockout/login. "
        "Return strict JSON with keys: intent, username (string or null)."
    )
    usr_p = f"Message: {text}"
    out = llm_json(sys_p, usr_p, tag="intent")
    if not out: out = {"intent": "access_reset", "username": None}
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

# ---------- One agent run ----------
def run_agent_once(user_text: str, base_url: str, api_key: str) -> Dict[str, Any]:
    actions: List[str] = []
    tool_logs_start = len(st.session_state["tool_logs"])

    # Understand
    iu = llm_intent_and_user(user_text)
    username = iu.get("username") or extract_username(user_text)
    parsed_username = username
    actions.append("intent:access_reset")

    # Create incident
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
    st.session_state["kpis"]["total"] += 1
    actions.append(f"created:{inc_number}")

    # Acknowledgement
    ack = llm_text(
        "You write concise IT support acknowledgements. 2 sentences max.",
        f"Ticket: {inc_number}. Scenario: Access lockout/password reset. Write a short acknowledgement."
    )
    actions.append("user_update:first_touch")

    # Tools: idp status
    ok, idp_body, _ = call_tool(base_url, api_key, "/tools/v1/idp/status", {})
    idp_ok = ok and bool(idp_body.get("ok", True))
    actions.append(f"idp_ok:{'True' if idp_ok else 'False'}")

    # Planner + lookup
    context = {"idp_ok": idp_ok, "username": username, "number": inc_number, "lookup": {}}
    if not idp_ok:
        decision = "route_identity"
    else:
        lok, lookup_body, _ = call_tool(base_url, api_key, "/tools/v1/itsm/dir/user_lookup", {"user": username})
        actions.append("lookup:user")
        context["lookup"] = lookup_body if lok else {"exists": False}
        plan = llm_plan_action(context)
        decision = plan.get("action", "ask_username")

    # Branches
    if decision == "route_identity":
        incident["state"] = "In Progress"
        incident["assignment_group"] = "Identity"
        final = llm_text(
            "You write concise IT status notes. 2 sentences max.",
            f"Explain to the user that identity provider reports an outage; ticket {inc_number} is routed to Identity."
        )
        actions.extend(["routed:Identity", "worknote:idp_outage"])
        return {
            "ack": ack, "final": final, "actions": actions, "incident": incident,
            "tool_logs": st.session_state["tool_logs"][tool_logs_start:],
            "parsed_username": parsed_username, "decision": decision
        }

    if decision == "ask_username" or not username:
        final = "Please provide your username or email to proceed with the reset."
        actions.append("clarify:username")
        return {
            "ack": ack, "final": final, "actions": actions, "incident": incident,
            "tool_logs": st.session_state["tool_logs"][tool_logs_start:],
            "parsed_username": parsed_username, "decision": decision
        }

    # reset + close
    rok, reset_body, _ = call_tool(base_url, api_key, "/tools/v1/itsm/dir/reset_password", {"user": username, "delivery": "chat"})
    if not rok:
        final = f"Could not issue a temporary password at this time. Your ticket {inc_number} remains in progress."
        actions.append("tool_error:reset_password")
        return {
            "ack": ack, "final": final, "actions": actions, "incident": incident,
            "tool_logs": st.session_state["tool_logs"][tool_logs_start:],
            "parsed_username": parsed_username, "decision": "reset_password_failed"
        }

    temp_password = reset_body.get("temp_password", "")
    actions.extend(["tool_exec:reset_password", "worknote:reset_issued"])
    incident["state"] = "Resolved"; incident["resolved_at"] = _now()
    incident["state"] = "Closed";   incident["closed_at"] = _now()
    actions.extend(["auto_resolved", "closed"])

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
        "ack": ack, "final": final, "actions": actions, "incident": incident,
        "tool_logs": st.session_state["tool_logs"][tool_logs_start:],
        "parsed_username": parsed_username, "decision": "reset_password"
    }

# ---------- UI ----------

st.title("Access Lockout Agent — Chat (LLM + Tools)")

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
    k = st.session_state["kpis"]
    total = max(k["total"], 1)
    auto_pct = round((k["auto_resolved"] / total) * 100, 2)
    st.subheader("Session KPIs")
    st.write(f"Total incidents: {k['total']}")
    st.write(f"Closed incidents: {k['closed']}")
    st.write(f"Auto-resolved percent: {auto_pct}")

    if st.button("Reset KPIs & history"):
        st.session_state["chat"].clear()
        st.session_state["incidents"].clear()
        st.session_state["tool_logs"].clear()
        st.session_state["kpis"] = {"total": 0, "closed": 0, "auto_resolved": 0}
        st.session_state["last_run"] = None
        st.experimental_rerun()

# Render chat history
for msg in st.session_state["chat"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
prompt = st.chat_input("Describe your issue (e.g., 'I'm locked out. I am ram')")
if prompt:
    # user message
    st.session_state["chat"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # agent run
    result = run_agent_once(user_text=prompt, base_url=gw_url, api_key=gw_key)
    st.session_state["last_run"] = result

    # assistant messages: ACK then FINAL (password in final on success)
    with st.chat_message("assistant"):
        st.write(result["ack"])
    st.session_state["chat"].append({"role": "assistant", "content": result["ack"]})

    with st.chat_message("assistant"):
        st.write(result["final"])
    st.session_state["chat"].append({"role": "assistant", "content": result["final"]})

    # optional mini-debug under last run
    with st.expander("Last run details", expanded=False):
        st.caption(f"Parsed username: **{result.get('parsed_username','')}**")
        st.caption(f"Planner decision: **{result.get('decision','(n/a)')}**")
        st.write("Action Trace:")
        st.code(" -> ".join(result["actions"]))
        st.write("Incident Snapshot:")
        st.json(result["incident"])
        st.write("Tool Calls:")
        for i, rec in enumerate(result["tool_logs"], 1):
            st.write(f"{i}. {rec['path']} [{rec['status_code']}] {rec['elapsed_ms']} ms")
            st.json({"request": rec["request"], "response": rec["response"]})