#!/usr/bin/env python3
"""
Agentic IT Helpdesk — Conductor Mode Only (LLM-first with Supervisor gates)

What this does (short):
- A single LLM ("Conductor") plans ONE next step at a time:
  Planner → (Supervisor PRE for high-impact) → Run Step → Reflect → (Supervisor POST) → Reply.
- Understands messy asks across: password, vpn, network, info (LLM-only), status.
- Deterministic toolbox for tickets (CSV-backed) and mock tools:
    POST /tools/v1/password/reset
    POST /tools/v1/vpn/diagnose
    POST /tools/v1/network/diagnose
- Friendly CLI output: "You / Agent" and a compact "(background)" line with key actions.

How it’s different from a pipeline:
- No fixed DAG of actions. The LLM decides the next single step (ask/create/tool/link/close/timing/reply)
  based on the current thread state and the most recent outcomes.
"""

import os, json, time, uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd
import httpx
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# ============================
# Azure OpenAI (fill these)
# ============================
AZURE_ENDPOINT    = ""  # e.g., "https://<your-endpoint>.openai.azure.com/"
AZURE_API_KEY     = ""  # e.g., "xxxxxxxxxxxxxxxx"
AZURE_API_VERSION = ""  # e.g., "2024-08-01-preview"
AZURE_DEPLOYMENT  = ""  # your chat model deployment name

try:
    from openai import AzureOpenAI
    _client = AzureOpenAI(
        api_key=AZURE_API_KEY or "set-me",
        api_version=AZURE_API_VERSION or "set-me",
        azure_endpoint=AZURE_ENDPOINT or "https://example.invalid",
    )
except Exception:
    _client = None  # we'll raise a clear error on first use

# ============================
# Tool Gateway (Bearer auth)
# ============================
TOOL_GATEWAY_URL = os.getenv("TOOL_GATEWAY_URL", "http://localhost:8088").rstrip("/")
TOOL_API_KEY     = os.getenv("TOOL_API_KEY", "dev-secret")

# ============================
# Output mode
# ============================
# "conversation" → You / Agent / (background)
# "agent_only"   → only the final chat text
OUTPUT_VIEW = (os.getenv("OUTPUT_VIEW", "conversation") or "conversation").lower()

# ============================
# Logging & defaults
# ============================
DEFAULT_TICKETS_CSV = "incidents_mock.csv"
PRINT_WIDTH = 200

def clock() -> str:
    """Short clock for trace lines."""
    return time.strftime("%H:%M:%S")

def shorten(x, width: int = PRINT_WIDTH) -> str:
    """Pretty-printer: safely stringify and truncate long JSON/text for console logs."""
    try:
        s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    except Exception:
        s = str(x)
    return s if len(s) <= width else s[:width] + " …"

def trace(msg: str) -> None:
    """Unified trace line."""
    print(f"[{clock()}] {msg}", flush=True)

# ============================
# HTTP tool caller
# ============================
def call_tool(endpoint: str, payload: Dict[str, Any], timeout: float = 8.0) -> Dict[str, Any]:
    """
    Post JSON to the local tool gateway with Bearer auth.
    Returns parsed JSON; raises on non-2xx responses.
    """
    url = f"{TOOL_GATEWAY_URL}{endpoint}"
    headers = {"Authorization": f"Bearer {TOOL_API_KEY}", "Content-Type": "application/json"}
    with httpx.Client() as c:
        resp = c.post(url, headers=headers, json=payload, timeout=timeout)
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"Tool call HTTP {resp.status_code}: {resp.text}")
    try:
        return resp.json()
    except Exception:
        return {"ok": False, "error": "invalid_json", "raw": resp.text}

# ============================
# LLM helpers (friendly names)
# ============================
def _ensure_client():
    """Fail fast if Azure settings are not filled."""
    if _client is None or not (AZURE_ENDPOINT and AZURE_API_KEY and AZURE_API_VERSION and AZURE_DEPLOYMENT):
        raise RuntimeError("Azure OpenAI not configured. Fill AZURE_* constants at the top of the file.")

def ask_text(system_prompt: str, user_prompt: str, *, temperature: float = 0.2, max_tokens: int = 400, tag: str = "llm") -> str:
    """LLM call that returns plain text."""
    _ensure_client()
    resp = _client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
        temperature=temperature, max_tokens=max_tokens,
    )
    out = (resp.choices[0].message.content or "").strip()
    trace(f"[{tag}] {shorten(out)}")
    return out

def ask_json(system_prompt: str, user_prompt: str, *, json_schema: Optional[Dict[str,Any]] = None,
             temperature: float = 0.1, max_tokens: int = 600, tag: str = "llm.json") -> Dict[str,Any]:
    """LLM call that must return JSON. Optionally includes a function schema to guide structure."""
    _ensure_client()
    resp = _client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
        temperature=temperature, max_tokens=max_tokens,
        response_format={"type":"json_object"},
        tools=([{"type":"function","function":{"name":"emit","parameters":json_schema}}] if json_schema else None),
        tool_choice=({"type":"function","function":{"name":"emit"}} if json_schema else None),
    )
    txt = (resp.choices[0].message.content or "{}").strip()
    trace(f"[{tag}] {shorten(txt)}")
    try:
        return json.loads(txt or "{}")
    except Exception:
        # very defensive fallback
        return {}

# ============================
# Ticket store (CSV-backed)
# ============================
class TicketStore:
    """
    Tiny ticket table for the POC. CSV-backed for write-through persistence.
    Columns are human-friendly (number, state, priority, category, etc.).
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        required = [
            "number","opened_at","opened_by","short_description","description",
            "priority","category","urgency","state","assignment_group",
            "knowledge_match","resolved_at","closed_at","parent_number"
        ]
        for c in required:
            if c not in self.df.columns:
                self.df[c] = ""
        self._existing_numbers = set(self.df["number"].dropna().astype(str).tolist())

    def _gen_number(self) -> str:
        """Generate a pseudo ServiceNow-like incident number."""
        while True:
            n = f"INC{100000 + int(uuid.uuid4().int % 900000)}"
            if n not in self._existing_numbers:
                self._existing_numbers.add(n)
                return n

    def open_ticket(self, record: Dict[str,Any]) -> Dict[str,Any]:
        """Create a new ticket row and return the created record."""
        row = {
            "number": self._gen_number(),
            "opened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "opened_by": record.get("opened_by","chat_user"),
            "short_description": record.get("short_description","").strip(),
            "description": record.get("description","").strip(),
            "priority": record.get("priority","Moderate"),
            "category": record.get("category","Service Desk"),
            "urgency": record.get("urgency","Medium"),
            "state": "Open",
            "assignment_group": record.get("assignment_group",""),
            "knowledge_match": record.get("knowledge_match",""),
            "resolved_at": "",
            "closed_at": "",
            "parent_number": record.get("parent_number",""),
        }
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        trace(f"[Ticket] opened {row['number']} ({row['category']})")
        return row

    def update_ticket(self, number: str, updates: Dict[str,Any]) -> Dict[str,Any]:
        """Update one ticket by number; returns the new row as dict."""
        idx = self.df.index[self.df["number"] == number]
        if not len(idx):
            raise ValueError(f"Ticket {number} not found")
        i = idx[0]
        for k,v in updates.items():
            if k in self.df.columns:
                self.df.at[i, k] = v
        trace(f"[Ticket] updated {number} {shorten(updates)}")
        return self.df.loc[i].to_dict()

    def close_ticket(self, number: str) -> Dict[str,Any]:
        """Mark a ticket Closed with a timestamp."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.update_ticket(number, {"state":"Closed","closed_at":now})

    def to_csv(self, path: str) -> None:
        """Persist the DataFrame to CSV."""
        self.df.to_csv(path, index=False)

# Global store set in main()
store: Optional[TicketStore] = None

# ============================
# Mappings & timing line
# ============================
def issue_to_category(issue: str) -> str:
    """Map issue keyword → category name."""
    return {
        "password":"Access",
        "vpn":"Network",
        "network":"Network",
        "info":"Info",
        "status":"Info"
    }.get((issue or "").lower(), "Service Desk")

def group_for_category(category: str) -> str:
    """Map category → assignment group."""
    return {
        "Access":"Service Desk",
        "Network":"Network",
        "Info":"Service Desk",
        "Service Desk":"Service Desk"
    }.get(category or "Service Desk", "Service Desk")

def make_timing_line(timing: Dict[str,Any]) -> str:
    """User-friendly timing sentence for replies."""
    by = timing.get("next_update_by","--:--")
    pct = int(timing.get("risk_pct", 20))
    return f"Next update by {by}. Risk of delay: {pct}%."

def compute_timing_info(context: Dict[str,Any]) -> Dict[str,Any]:
    """Basic timing heuristic; you can replace with SLA rules later."""
    nxt = (datetime.now() + timedelta(minutes=25)).strftime("%H:%M")
    risk = 30 if (context.get("priority","").lower() in ("high","critical")) else 20
    return {"next_update_by": nxt, "risk_pct": risk}

# ============================
# Toolbox wrappers (deterministic)
# ============================
def create_ticket(category: str, summary: str, description: str) -> Dict[str,Any]:
    """Create a ticket in the CSV store."""
    rec = store.open_ticket({
        "opened_by": "chat_user",
        "short_description": summary,
        "description": description,
        "category": category,
        "assignment_group": group_for_category(category),
    })
    return {"number": rec["number"], "category": rec["category"]}

def update_ticket(number: str, **fields) -> Dict[str,Any]:
    """Update any fields of a ticket."""
    store.update_ticket(number, fields)
    return {"ok": True}

def link_child_issue(child_issue: str, parent_number: str) -> Dict[str,Any]:
    """Log a link under a parent (CSV demo keeps it as a trace only)."""
    trace(f"[Link] {child_issue} linked to {parent_number}")
    return {"ok": True}

def close_ticket(number: str) -> Dict[str,Any]:
    """Close a ticket by number."""
    store.close_ticket(number)
    return {"ok": True}

def list_open_tickets(category: str) -> List[Dict[str,Any]]:
    """Optional finder: list open tickets by category."""
    if store is None: return []
    df = store.df[(store.df["category"]==category) & (store.df["state"].isin(["Open","In Progress","On Hold"]))]
    out = []
    for _, r in df.iterrows():
        out.append({"number": r["number"], "category": r["category"], "state": r["state"], "group": r["assignment_group"], "opened_at": r["opened_at"]})
    return out

def tool_password_reset(user: str, incident_ref: Optional[str] = None) -> Dict[str,Any]:
    """Call mock password reset tool."""
    return call_tool("/tools/v1/password/reset", {"user": user, "delivery": "chat", "incident_ref": incident_ref})

def tool_vpn_diagnose(user: str, gateway: str, client: str, version: str, incident_ref: Optional[str] = None) -> Dict[str,Any]:
    """Call mock VPN diagnose tool."""
    return call_tool("/tools/v1/vpn/diagnose", {"user": user, "gateway": gateway, "client": client, "version": version, "incident_ref": incident_ref})

def tool_network_diagnose(user: str, adapter: str, os_name: str, symptoms: List[str], incident_ref: Optional[str] = None) -> Dict[str,Any]:
    """Call mock Network diagnose tool."""
    return call_tool("/tools/v1/network/diagnose", {"user": user, "adapter": adapter, "os": os_name, "symptoms": symptoms, "incident_ref": incident_ref})

def tool_kb_quick_answer(topic: str) -> str:
    """
    LLM-only helper for 'info' asks or light guidance.
    Keep it short (5–7 lines). This is intentional since we lack an API for 'info'.
    """
    system_prompt = "Answer briefly (5–7 lines max). If topic is vague, give a short helpful overview."
    user_prompt = f"Topic: {topic}"
    return ask_text(system_prompt, user_prompt, max_tokens=180, tag="kb.quick")

# ============================
# LLM helpers for the loop
# ============================
def plan_next_step_llm(user_text: str, thread_state: Dict[str,Any], step_log: List[Dict[str,Any]]) -> Dict[str,Any]:
    """
    Decide ONE next step.
    Returns: {action, args, high_impact}
    """
    system_prompt = (
        "You are the Conductor. Choose ONE next step from:\n"
        "ask_user, call_tool, create_ticket, update_ticket, link_ticket, close_ticket, summarize_status, compute_timing, reply, stop.\n"
        "Prefer ask_user when a single missing detail blocks progress (e.g., username, gateway).\n"
        "If user mentions both network and vpn, prefer one Network ticket and link VPN under it.\n"
        "Use clear args. Mark high_impact=true only for close_ticket or priority changes."
    )
    user_prompt = json.dumps({"user_text": user_text, "thread": thread_state, "recent": step_log[-3:]}, ensure_ascii=False)
    schema = {
        "type":"object","properties":{
            "action":{"type":"string","enum":["ask_user","call_tool","create_ticket","update_ticket","link_ticket","close_ticket","summarize_status","compute_timing","reply","stop"]},
            "args":{"type":"object","additionalProperties":True},
            "high_impact":{"type":"boolean"}
        },"required":["action","args","high_impact"]
    }
    return ask_json(system_prompt, user_prompt, json_schema=schema, tag="conductor.plan")

def reflect_next_step_llm(user_text: str, thread_state: Dict[str,Any], step_log: List[Dict[str,Any]]) -> Dict[str,Any]:
    """
    Decide what to do AFTER running a step.
    Returns: {action, args, high_impact}
    """
    system_prompt = (
        "Reflect on the latest steps (step_log). Next, choose ONE: ask_user, call_tool, create_ticket, "
        "update_ticket, link_ticket, close_ticket, summarize_status, compute_timing, reply, stop.\n"
        "If enough info was gathered, prefer reply (include ticket numbers and helpful summary)."
    )
    user_prompt = json.dumps({"user_text": user_text, "thread": thread_state, "recent": step_log[-4:]}, ensure_ascii=False)
    schema = {
        "type":"object","properties":{
            "action":{"type":"string","enum":["ask_user","call_tool","create_ticket","update_ticket","link_ticket","close_ticket","summarize_status","compute_timing","reply","stop"]},
            "args":{"type":"object","additionalProperties":True},
            "high_impact":{"type":"boolean"}
        },"required":["action","args","high_impact"]
    }
    return ask_json(system_prompt, user_prompt, json_schema=schema, tag="conductor.reflect")

def supervisor_precheck_llm(step: Dict[str,Any]) -> str:
    """Gatekeeper for high-impact changes. Returns 'APPROVE' or 'DENY:<reason>'."""
    system_prompt = "You are the Supervisor. Approve or deny this action for safety/compliance. Reply 'APPROVE' or 'DENY:<short reason>'."
    user_prompt = json.dumps(step, ensure_ascii=False)
    out = ask_text(system_prompt, user_prompt, max_tokens=20, tag="supervisor.pre").strip().upper()
    return out or "APPROVE"

def supervisor_qacheck_llm(final_text: str) -> str:
    """
    Final QA on the outbound message.
    APPROVED or RETRY:<hint>. Only one retry is attempted by the node.
    """
    system_prompt = "You are the Supervisor. If clear and helpful, reply ONLY 'APPROVED'. Otherwise 'RETRY:<short hint>'."
    user_prompt = final_text
    out = ask_text(system_prompt, user_prompt, max_tokens=40, tag="supervisor.post").strip().upper()
    return out or "APPROVED"

def write_fix_steps_llm(issue: str, tool_envelope: Dict[str,Any], user_text: str) -> str:
    """3–5 short actionable steps (end with ONE confirmation question)."""
    system_prompt = "Write clear steps (3–5 bullets) for the user to try now. Use tool signals if available. End with ONE confirmation question."
    user_prompt = f"Issue: {issue}\nUser text: {user_text}\nTool: {json.dumps(tool_envelope, ensure_ascii=False)}"
    return ask_text(system_prompt, user_prompt, max_tokens=250, tag=f"fix_steps.{issue}")

def write_status_blurb_llm(tickets: List[Dict[str,Any]]) -> str:
    """Brief friendly status update (3–5 lines)."""
    system_prompt = "Write a brief, friendly status update for the user. 3–5 lines max."
    user_prompt = json.dumps({"tickets": tickets}, ensure_ascii=False)
    return ask_text(system_prompt, user_prompt, max_tokens=200, tag="status.writer")

def write_user_response_llm(state: "FlowState") -> str:
    """
    Compose the user-facing message.
    - If no tickets/steps, give a helpful short answer and end with ONE question.
    - Otherwise summarize ticket numbers (opened/updated/closed), include Fix Steps blocks (if present),
      and add the timing sentence exactly once.
    """
    # Summarize ticket numbers from step_log
    opened, updated, closed = [], [], []
    for a in state.step_log:
        s = (a.get("summary") or "")
        if s.startswith("ticket:opened "):   opened.append(s.split()[-1])
        if s.startswith("ticket:updated "):  updated.append(s.split()[-1])
        if s.startswith("ticket:closed "):   closed.append(s.split()[-1])

    made = {
        "original": state.user_text,
        "tickets": {"opened": opened, "updated": updated, "closed": closed},
        "steps": state.fix_steps_by_issue,
        "tools": state.tool_results,
        "timing_line": make_timing_line(state.timing_info) if state.timing_info else "",
    }
    system_prompt = (
        "Compose a helpful support message.\n"
        "If steps is empty and no tickets changed, give a short helpful answer (5–8 lines max) and end with ONE question.\n"
        "Otherwise, summarize ticket numbers (opened/updated/closed), include Fix Steps blocks (only for listed issues), "
        "and include the provided timing line exactly once."
    )
    user_prompt = json.dumps(made, ensure_ascii=False)
    return ask_text(system_prompt, user_prompt, max_tokens=700, tag="response_writer")

# ============================
# Flow state (Conductor)
# ============================
class FlowState(BaseModel):
    """State carried through the Conductor loop in a single turn."""
    user_text: str = ""

    # Conductor choices & memory
    thread_state: Dict[str,Any] = Field(default_factory=lambda: {
        "pending_question": {},        # {"field": "...", "question": "..."}
        "active_ticket_ids": [],       # ["INC123456", ...]
        "user_profile": {}             # reserved for later
    })
    step_log: List[Dict[str,Any]] = Field(default_factory=list)  # [{step_id, action, args, summary, ts}]
    next_action: str = ""
    next_args: Dict[str,Any] = Field(default_factory=dict)
    next_is_high_impact: bool = False
    supervisor_pre_verdict: str = ""

    # Artifacts for composing replies
    tool_results: Dict[str,Dict[str,Any]] = Field(default_factory=dict)
    fix_steps_by_issue: Dict[str,str] = Field(default_factory=dict)
    timing_info: Dict[str,Any] = Field(default_factory=dict)

    # Output
    final_text: str = ""
    supervisor_verdict: str = ""
    retry_count: int = 0
    await_input: bool = False
    escalate_to_human: bool = False  # optional safety switch
    user_message: str = ""  # mirror for CLI printing

# ============================
# Conductor nodes
# ============================
def node_plan_next_step(state: FlowState) -> FlowState:
    """Call the Planner LLM; store next_action/args/high_impact."""
    step = plan_next_step_llm(state.user_text, state.thread_state, state.step_log)
    state.next_action = step.get("action","")
    state.next_args = step.get("args",{}) or {}
    state.next_is_high_impact = bool(step.get("high_impact", False))
    trace(f"[Planner] action={state.next_action} args={shorten(state.next_args)} hi={state.next_is_high_impact}")
    return state

def node_supervisor_precheck(state: FlowState) -> FlowState:
    """If action is high-impact, ask Supervisor to approve/deny it."""
    verdict = supervisor_precheck_llm({"action": state.next_action, "args": state.next_args})
    state.supervisor_pre_verdict = verdict
    trace(f"[Supervisor:pre] {verdict}")
    return state

def _append_step_log(state: FlowState, action: str, args: Dict[str,Any], summary: str):
    """Add a compact record of what we just did for downstream LLMs and the CLI background line."""
    state.step_log.append({
        "step_id": f"{int(time.time()*1000)}",
        "action": action,
        "args": args,
        "summary": summary,
        "ts": datetime.utcnow().isoformat(timespec="seconds")+"Z"
    })

# ---- Action handlers (dispatcher) ----
def _handle_ask_user(state: FlowState, args: Dict[str,Any]) -> str:
    q = args.get("question") or "What username or more details should I use?"
    field = args.get("field","")
    state.thread_state["pending_question"] = {"field": field, "question": q}
    state.final_text = q
    state.user_message = q
    state.await_input = True
    return f"asked:{field or 'unknown'}"

def _handle_call_tool(state: FlowState, args: Dict[str,Any]) -> str:
    tool = (args.get("tool") or "").lower()
    num  = args.get("incident_ref")
    # sub-dispatch for tools
    TOOL_DISPATCH: Dict[str, Callable[[], Tuple[str, Dict[str,Any]]]] = {
        "password_reset": lambda: ("password", tool_password_reset(args.get("user","demo@corp.com"), num)),
        "password":       lambda: ("password", tool_password_reset(args.get("user","demo@corp.com"), num)),
        "vpn_diag":       lambda: ("vpn",      tool_vpn_diagnose(args.get("user","demo@corp.com"), args.get("gateway","vpn.corp.com"), args.get("client","GlobalProtect"), args.get("version","6.2.2"), num)),
        "vpn":            lambda: ("vpn",      tool_vpn_diagnose(args.get("user","demo@corp.com"), args.get("gateway","vpn.corp.com"), args.get("client","GlobalProtect"), args.get("version","6.2.2"), num)),
        "network_diag":   lambda: ("network",  tool_network_diagnose(args.get("user","demo@corp.com"), args.get("adapter","wifi"), args.get("os","Windows 11"), args.get("symptoms",[]), num)),
        "network":        lambda: ("network",  tool_network_diagnose(args.get("user","demo@corp.com"), args.get("adapter","wifi"), args.get("os","Windows 11"), args.get("symptoms",[]), num)),
        "kb":             lambda: ("info",     {"ok": True, "status":"kb_answered", "text": tool_kb_quick_answer(args.get("topic","IT help"))}),
    }
    if tool not in TOOL_DISPATCH:
        return f"tool:unknown({tool})"
    try:
        issue_key, env = TOOL_DISPATCH[tool]()
        state.tool_results[issue_key] = env
        # produce fix steps immediately for UX on actionable issues
        if issue_key in ("password","vpn","network"):
            state.fix_steps_by_issue[issue_key] = write_fix_steps_llm(issue_key, env, state.user_text)
        return f"tool:{tool} -> {env.get('status','ok')}"
    except Exception as e:
        return f"error:{type(e).__name__}"

def _handle_create_ticket(state: FlowState, args: Dict[str,Any]) -> str:
    cat = args.get("category") or issue_to_category(args.get("issue","info"))
    mk = create_ticket(cat, args.get("summary", f"{cat} issue from chat"), args.get("description", state.user_text))
    state.thread_state["active_ticket_ids"].append(mk["number"])
    # stash number so subsequent steps can reference it
    state.next_args["number"] = mk["number"]
    return f"ticket:opened {mk['number']}"

def _handle_update_ticket(state: FlowState, args: Dict[str,Any]) -> str:
    num = args.get("number") or (state.thread_state["active_ticket_ids"][-1] if state.thread_state["active_ticket_ids"] else None)
    if not num:
        return "ticket:update_skipped (no number)"
    update_ticket(num, **{k:v for k,v in args.items() if k!="number"})
    return f"ticket:updated {num}"

def _handle_link_ticket(state: FlowState, args: Dict[str,Any]) -> str:
    parent = args.get("parent_number") or (state.thread_state["active_ticket_ids"][-1] if state.thread_state["active_ticket_ids"] else None)
    child_issue = (args.get("child_issue") or "vpn")
    if not parent:
        return "ticket:link_skipped (no parent)"
    link_child_issue(child_issue, parent)
    return f"ticket:linked {child_issue}->{parent}"

def _handle_close_ticket(state: FlowState, args: Dict[str,Any]) -> str:
    num = args.get("number") or (state.thread_state["active_ticket_ids"][-1] if state.thread_state["active_ticket_ids"] else None)
    if not num:
        return "ticket:close_skipped (no number)"
    close_ticket(num)
    return f"ticket:closed {num}"

def _handle_summarize_status(state: FlowState, args: Dict[str,Any]) -> str:
    if store is None or store.df.empty:
        state.fix_steps_by_issue["status"] = "No incidents yet."
        return "status:none"
    last3 = store.df.tail(3)[["number","short_description","state","assignment_group","opened_at"]]
    items = []
    for _, r in last3.iterrows():
        items.append({"number": r["number"], "title": r["short_description"], "state": r["state"], "group": r["assignment_group"], "opened_at": r["opened_at"]})
    state.fix_steps_by_issue["status"] = write_status_blurb_llm(items)
    return "status:written"

def _handle_compute_timing(state: FlowState, args: Dict[str,Any]) -> str:
    state.timing_info = compute_timing_info(args)
    return f"timing:{make_timing_line(state.timing_info)}"

def _handle_reply_or_stop(state: FlowState, args: Dict[str,Any]) -> str:
    # explicit reply text (optional)
    if args.get("text"):
        state.final_text = args["text"]
        state.user_message = args["text"]
    return f"end:{state.next_action}"

def _handle_unknown(state: FlowState, args: Dict[str,Any]) -> str:
    return f"unknown_action:{state.next_action}"

# Action dispatcher table
ACTION_DISPATCH: Dict[str, Callable[[FlowState, Dict[str,Any]], str]] = {
    "ask_user":         _handle_ask_user,
    "call_tool":        _handle_call_tool,
    "create_ticket":    _handle_create_ticket,
    "update_ticket":    _handle_update_ticket,
    "link_ticket":      _handle_link_ticket,
    "close_ticket":     _handle_close_ticket,
    "summarize_status": _handle_summarize_status,
    "compute_timing":   _handle_compute_timing,
    "reply":            _handle_reply_or_stop,
    "stop":             _handle_reply_or_stop,
}

def node_run_step(state: FlowState) -> FlowState:
    """
    Deterministic executor. The LLM chose the action+args already.
    We simply route to the right handler via a dict dispatcher, run it,
    and append a compact summary to step_log for downstream LLMs / CLI.
    """
    action = (state.next_action or "").lower()
    args = state.next_args or {}
    handler = ACTION_DISPATCH.get(action, _handle_unknown)
    summary = "ok"
    try:
        summary = handler(state, args)
    except Exception as e:
        summary = f"error:{type(e).__name__}"
    _append_step_log(state, action, args, summary)
    trace(f"[RunStep] {summary}")
    return state

def node_reflect_next_step(state: FlowState) -> FlowState:
    """Call the Reflect LLM; choose next action/args/high_impact."""
    step = reflect_next_step_llm(state.user_text, state.thread_state, state.step_log)
    state.next_action = step.get("action","")
    state.next_args = step.get("args",{}) or {}
    state.next_is_high_impact = bool(step.get("high_impact", False))
    trace(f"[Reflect] next={state.next_action} args={shorten(state.next_args)} hi={state.next_is_high_impact}")
    return state

def node_compose_reply(state: FlowState) -> FlowState:
    """
    Compose the final text for this turn:
    - If we just asked a question, send it directly.
    - Else if explicit reply text is provided, use it.
    - Else build from current artifacts (tickets, fix steps, timing).
    """
    if state.next_action == "ask_user" and state.thread_state.get("pending_question"):
        q = state.thread_state["pending_question"].get("question","Could you share more details?")
        state.final_text = q
        state.user_message = q
        state.await_input = True
        return state
    if state.next_action == "reply" and state.next_args.get("text"):
        msg = state.next_args["text"]
        state.final_text = msg
        state.user_message = msg
        return state
    msg = write_user_response_llm(state)
    state.final_text = msg
    state.user_message = msg
    return state

def node_supervisor_qacheck(state: FlowState) -> FlowState:
    """Ask Supervisor to approve the final message; allow one retry if asked."""
    v = supervisor_qacheck_llm(state.final_text)
    if v.startswith("RETRY") and state.retry_count == 0:
        state.retry_count = 1
        msg = write_user_response_llm(state)
        state.final_text = msg
        state.user_message = msg
        v = supervisor_qacheck_llm(state.final_text)
    state.supervisor_verdict = v
    trace(f"[Supervisor:post] {v}")
    return state

def node_reply(state: FlowState) -> FlowState:
    """Ensure a non-empty message goes out."""
    if not (state.final_text or "").strip():
        state.final_text = "I can help with password, VPN, or network issues — what’s the username or device details?"
    state.user_message = state.final_text
    return state

# ============================
# Routers
# ============================
def route_after_planner(state: FlowState) -> str:
    a = (state.next_action or "").lower()
    if a in ("reply","stop","ask_user"):
        return "reply_like"
    return "highimpact" if state.next_is_high_impact else "act"

def route_after_supervisor_pre(state: FlowState) -> str:
    v = state.supervisor_pre_verdict.upper() if state.supervisor_pre_verdict else "APPROVE"
    return "act" if v.startswith("APPROVE") else "reflect"

def route_after_reflect(state: FlowState) -> str:
    a = (state.next_action or "").lower()
    return "loop" if a not in ("reply","stop","ask_user") else "reply_like"

# ============================
# Graph builder (Conductor-only)
# ============================
def build_graph_conductor() -> StateGraph:
    g = StateGraph(FlowState)

    g.add_node("plan",        node_plan_next_step)
    g.add_node("precheck",    node_supervisor_precheck)
    g.add_node("run",         node_run_step)
    g.add_node("reflect",     node_reflect_next_step)
    g.add_node("compose",     node_compose_reply)
    g.add_node("qa",          node_supervisor_qacheck)
    g.add_node("reply",       node_reply)

    g.set_entry_point("plan")

    g.add_conditional_edges("plan", route_after_planner, {
        "reply_like": "compose",
        "highimpact": "precheck",
        "act":        "run",
    })
    g.add_conditional_edges("precheck", route_after_supervisor_pre, {
        "act":     "run",
        "reflect": "reflect",
    })
    g.add_edge("run", "reflect")
    g.add_conditional_edges("reflect", route_after_reflect, {
        "loop":       "plan",
        "reply_like": "compose",
    })
    g.add_edge("compose", "qa")
    g.add_edge("qa", "reply")
    g.add_edge("reply", END)
    return g

# ============================
# Output helpers
# ============================
def _reply_text(state: dict) -> str:
    return (state.get("final_text") or state.get("user_message") or "").strip() or "(no reply)"

def print_conversation_block(user_text: str, state: dict) -> None:
    print(f"\nYou: {user_text}")
    print(f"Agent: {_reply_text(state)}")

    bg = []
    acts = state.get("step_log", [])[-3:]
    if acts:
        bg.append("steps: " + " | ".join([f"{a.get('action')}→{a.get('summary')}" for a in acts]))
    tbi = state.get("tool_results") or {}
    if tbi:
        tb = []
        for k, v in tbi.items():
            st = (v or {}).get("status") or ("error" if not (v or {}).get("ok", True) else "ok")
            tb.append(f"{k}:{st}")
        if tb: bg.append("tools: " + "; ".join(tb))
    tm = state.get("timing_info") or {}
    if tm:
        bg.append(f"timing: {make_timing_line(tm)}")
    vd = (state.get("supervisor_verdict") or "").strip()
    if vd:
        bg.append(f"supervisor: {vd}")
    if bg:
        print("(background) " + " | ".join(bg))

def print_result(user_text: str, state: dict) -> None:
    if OUTPUT_VIEW == "agent_only":
        print(_reply_text(state))
    else:
        print_conversation_block(user_text, state)

# ============================
# Store & runner
# ============================
def load_ticket_store(path: str) -> TicketStore:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    return TicketStore(df.fillna(""))

def process_message(graph, text: str, thread_state: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
    st = FlowState(user_text=text)
    if thread_state:
        st.thread_state = thread_state  # persist clarifications between turns
    out = graph.invoke(st)
    # normalize to dict
    if hasattr(out, "model_dump"): state = out.model_dump()
    elif hasattr(out, "dict"):     state = out.dict()
    else:
        try: state = vars(out)
        except: state = {"user_message": None}
    return state

# ============================
# Main — demos + REPL
# ============================
def main():
    """
    - Loads CSV-backed ticket store
    - Builds & compiles Conductor graph
    - Runs demo scenarios
    - Starts REPL with persistent thread state (for follow-ups)
    - Writes a snapshot CSV on exit
    """
    global store
    store = load_ticket_store(DEFAULT_TICKETS_CSV)

    graph = build_graph_conductor().compile()
    trace("[INFO] Mode: conductor; LangGraph compiled.")

    # Demos (set [] to mute)
    scenarios = [
        "change pass bro",
        "VPN drops every 10 minutes from Chennai office. gateway: vpn.corp.com",
        "Where are we on my issue?",
        "Wi-Fi connected but no internet after sleep (Windows 11).",
        "Tell me about the company.",
        "Close my password ticket.",
    ]

    session_thread = None  # persisted across turns

    for s in scenarios:
        try:
            state = process_message(graph, s, thread_state=session_thread)
            print_result(s, state)
            session_thread = state.get("thread_state", session_thread)
        except Exception as e:
            print(f"\nYou: {s}")
            print(f"Agent: (error) {e}")

    print("\nType your own messages (q to quit):")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Bye.")
            break
        if user_text.lower() in {"q","quit","exit"}:
            break
        try:
            state = process_message(graph, user_text, thread_state=session_thread)
            print_result(user_text, state)
            session_thread = state.get("thread_state", session_thread)
        except Exception as e:
            print(f"Agent: (error) {e}")

    out_path = "incidents_updated_demo.csv"
    store.to_csv(out_path)
    print(f"[INFO] Wrote updated dataset -> {out_path}")

if __name__ == "__main__":
    main()
