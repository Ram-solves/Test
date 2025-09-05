#!/usr/bin/env python3
"""

- Asks, extracts issues (password / vpn / network / info).
- Supervisor (LLM) plans: update existing ticket vs create new (multi-issue OK).
- Deterministic parts write to a tiny CSV table and call plain mock tools.
- Fix Steps (LLM) writes the instructions per issue.
- Response Writer (LLM) composes the entire final chat message (user-facing).
- Supervisor (LLM) gives the final APPROVED / RETRY verdict (one retry max).
- Friendly traces: What You Said, Supervisor, Ticket Update, Tools, Steps, Timing, Learning.

Mock tools(from `tool_gateway.py`):
  POST /tools/v1/password/reset
  POST /tools/v1/vpn/diagnose
  POST /tools/v1/network/diagnose


Outputs:
- Console traces (friendly labels).
- incidents_updated_demo.csv snapshot on exit.
"""

import os, sys, json, argparse, uuid, time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import pandas as pd
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
import httpx

# ----------------------------
# Azure OpenAI config (FILL THESE MANUALLY)
# ----------------------------
AZURE_ENDPOINT    = ""  # e.g., "https://<your-endpoint>.openai.azure.com/"
AZURE_API_KEY     = ""  # e.g., "xxxxxxxxxxxxxxxxxxxxxxxx"
AZURE_API_VERSION = ""  # e.g., "2024-08-01-preview"
AZURE_DEPLOYMENT  = ""  # your chat model deployment name
try:
    from openai import AzureOpenAI
    _client = AzureOpenAI(
        api_key=AZURE_API_KEY or "set-me",
        api_version=AZURE_API_VERSION or "set-me",
        azure_endpoint=AZURE_ENDPOINT or "https://example.invalid",
    )
except Exception as _e:
    _client = None  # Will error later when ask_text/ask_json are used.

# ----------------------------
# Tool Gateway config (Bearer auth)
# ----------------------------
TOOL_GATEWAY_URL = os.getenv("TOOL_GATEWAY_URL", "http://localhost:8088").rstrip("/")
TOOL_API_KEY     = os.getenv("TOOL_API_KEY", "dev-secret")

# ----------------------------
# ogging & defaults
# ----------------------------
DEFAULT_TICKETS_CSV = "incidents_mock.csv"
PRINT_WIDTH = 200

def clock() -> str:
    """HH:MM:SS timestamp for tidy logs."""
    return time.strftime("%H:%M:%S")

def shorten(x, width: int = PRINT_WIDTH) -> str:
    """Shorten big dicts/strings for logs."""
    try:
        s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    except Exception:
        s = str(x)
    return s if len(s) <= width else s[:width] + " …"

def trace(msg: str) -> None:
    """Friendly console trace with time."""
    print(f"[{clock()}] {msg}")

# ----------------------------
# LLM wrappers (text / JSON)
# ----------------------------
def _ensure_client():
    if _client is None or not (AZURE_ENDPOINT and AZURE_API_KEY and AZURE_API_VERSION and AZURE_DEPLOYMENT):
        raise RuntimeError("Azure OpenAI not configured. Fill AZURE_* at top of file.")

def ask_text(system_prompt: str, user_prompt: str, *, temperature: float = 0.2, max_tokens: int = 400, tag: str = "llm") -> str:
    """Ask the model for short prose; returns text."""
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
             temperature: float = 0.1, max_tokens: int = 400, tag: str = "llm.json") -> Dict[str,Any]:
    """Ask the model for compact JSON; returns dict."""
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
    return json.loads(txt)

# ----------------------------
# Tool calls
# ----------------------------
def call_tool(endpoint: str, payload: Dict[str,Any], timeout: float = 6.0) -> Dict[str,Any]:
    """POST to mock tool gateway with Bearer auth; returns parsed JSON."""
    url = f"{TOOL_GATEWAY_URL}{endpoint}"
    headers = {"Authorization": f"Bearer {TOOL_API_KEY}", "Content-Type":"application/json"}
    with httpx.Client() as c:
        resp = c.post(url, headers=headers, json=payload, timeout=timeout)
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"Tool call HTTP {resp.status_code}: {resp.text}")
    try:
        return resp.json()
    except Exception:
        return {"ok": False, "error": "invalid_json", "raw": resp.text}

# ----------------------------
# CSV ticket store
# ----------------------------
class TicketStore:
    """(pandas DataFrame) CSV columns."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        required = [
            "number","opened_at","opened_by","short_description","description",
            "priority","category","urgency","state","assignment_group",
            "knowledge_match","resolved_at","closed_at"
        ]
        for c in required:
            if c not in self.df.columns:
                self.df[c] = ""
        self._existing_numbers = set(self.df["number"].dropna().astype(str).tolist())

    def generate_number(self) -> str:
        while True:
            n = f"INC{100000 + int(uuid.uuid4().int % 900000)}"
            if n not in self._existing_numbers:
                self._existing_numbers.add(n)
                return n

    def find_open_in(self, category: str) -> Optional[Dict[str,Any]]:
        df = self.df[(self.df["category"] == category) & (self.df["state"].isin(["Open","In Progress","On Hold"]))]
        if df.empty: return None
        return df.iloc[-1].to_dict()

    def open_ticket(self, record: Dict[str,Any]) -> Dict[str,Any]:
        row = {
            "number": self.generate_number(),
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
        }
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        trace(f"[Ticket Update] opened {row['number']} ({row['category']})")
        return row

    def update_ticket(self, number: str, updates: Dict[str,Any]) -> Dict[str,Any]:
        idx_series = self.df.index[self.df["number"] == number]
        if not len(idx_series):
            raise ValueError(f"Ticket {number} not found")
        row_idx = idx_series[0]  # friendlier than 'idx'
        for k,v in updates.items():
            if k in self.df.columns:
                self.df.at[row_idx, k] = v
        trace(f"[Ticket Update] updated {number} {shorten(updates)}")
        return self.df.loc[row_idx].to_dict()

    def to_csv(self, path: str) -> None:
        self.df.to_csv(path, index=False)

# ----------------------------
# Flow state ()
# ----------------------------
class FlowState(BaseModel):
    # Input
    user_text: str = ""

    # Issue Reader
    parsed_issues: List[Dict[str,Any]] = Field(default_factory=list)
    overall_confidence: float = 0.0

    # Supervisor (plan)
    plan: List[Dict[str,Any]] = Field(default_factory=list)
    clarify_question: str = ""
    await_input: bool = False

    # Work by issue
    tools_by_issue: Dict[str,Dict[str,Any]] = Field(default_factory=dict)
    steps_by_issue: Dict[str,str] = Field(default_factory=dict)

    # Response Writer & Supervisor (approve)
    final_message: str = ""
    verdict: str = ""
    iteration_count: int = 0

    # Timing & Learning
    timing: Dict[str,Any] = Field(default_factory=dict)
    learning_hints: Dict[str,Any] = Field(default_factory=dict)

    # Traces & user reply
    actions: List[str] = Field(default_factory=list)
    user_message: str = ""

# ----------------------------
# Helpers — mapping & timing line
# ----------------------------
def issue_to_category(issue: str) -> str:
    return {"password":"Access","vpn":"Network","network":"Network","info":"Info"}.get(issue, "Service Desk")

def issue_to_group(issue: str) -> str:
    return {"password":"Service Desk","vpn":"Network","network":"Network","info":"Service Desk"}.get(issue, "Service Desk")


TIMING_RISK_STYLE = "percent"  # "percent" | "words"
TIMING_PHRASE     = "next_update"  # "next_update" | "check_back"

def risk_words(pct: int) -> str:
    if pct <= 25: return "Low"
    if pct <= 60: return "Medium"
    return "High"

def make_timing_line(timing: Dict[str,Any]) -> str:
    by = timing.get("next_update_by") or timing.get("due_first_reply") or "--:--"
    pct = int(timing.get("risk_pct", 0))
    if TIMING_RISK_STYLE == "words":
        risk = risk_words(pct)
        if TIMING_PHRASE == "check_back":
            return f"I’ll check back by {by}. Chance: {risk}."
        return f"Target next update: {by}. Risk: {risk}."
    # percent
    if TIMING_PHRASE == "check_back":
        return f"I’ll check back by {by}. Chance of delay: {pct}%."
    return f"Next update by {by}. Risk of delay: {pct}%."

# ----------------------------
# LLM agents
# ----------------------------
def issue_reader_llm(text: str) -> Dict[str,Any]:
    """Extract issues with confidence + fields from messy user text."""
    system_prompt = (
        "You extract IT support issues from text. "
        "Allowed issue types: password, vpn, network, info. "
        "Return compact JSON: {overall_confidence, issues:[{type,confidence,fields}...]}. "
        "Fields may include: user, gateway, client, version, adapter, os, symptoms, topic."
    )
    user_prompt = f"User said:\n{text}\nExtract issues."
    json_schema = {
        "type":"object",
        "properties":{
            "overall_confidence":{"type":"number"},
            "issues":{"type":"array","items":{
                "type":"object",
                "properties":{
                    "type":{"type":"string","enum":["password","vpn","network","info"]},
                    "confidence":{"type":"number"},
                    "fields":{"type":"object","additionalProperties":True}
                },
                "required":["type","confidence"]
            }}
        },
        "required":["overall_confidence","issues"]
    }
    return ask_json(system_prompt, user_prompt, json_schema=json_schema, tag="issue_reader")

def supervisor_plan_llm(issues: List[Dict[str,Any]], overall: float, open_like: List[Dict[str,Any]]) -> Dict[str,Any]:
    """Supervisor (plan): choose update_existing/create_new/respond_only; ask one clarify only if low confidence."""
    system_prompt = (
        "You are the Supervisor. Make a short, safe plan. "
        "If overall_confidence < 0.35, ask ONE short clarifying question and return an empty plan. "
        "Otherwise for the top 1-2 issues: "
        "  - password/vpn/network -> 'update_existing' (with number from open_like) or 'create_new' "
        "  - info -> 'respond_only' "
        "Return JSON: {plan:[{issue,action,number|null}], clarify:{question}|null}"
    )
    user_prompt = json.dumps({"overall_confidence": overall, "issues": issues, "open_like": open_like}, ensure_ascii=False)
    json_schema = {
        "type":"object",
        "properties":{
            "plan":{"type":"array","items":{
                "type":"object","properties":{
                    "issue":{"type":"string","enum":["password","vpn","network","info"]},
                    "action":{"type":"string","enum":["update_existing","create_new","respond_only"]},
                    "number":{"type":["string","null"]}
                },"required":["issue","action"]
            }},
            "clarify":{"type":["object","null"],"properties":{"question":{"type":"string"}}}
        },
        "required":["plan","clarify"]
    }
    return ask_json(system_prompt, user_prompt, json_schema=json_schema, tag="supervisor.plan")

def fix_steps_llm(issue: str, tool_envelope: Dict[str,Any], user_text: str) -> str:
    """Write 3–8 clear steps using tool signals; finish with ONE confirmation question."""
    system_prompt = (
        "Write clear, short steps (3–8 bullets) for the user to try now. "
        "Use the provided signals and note if a temporary password was issued, etc. "
        "End with exactly ONE confirmation question."
    )
    user_prompt = f"Issue: {issue}\nUser text: {user_text}\nTool: {json.dumps(tool_envelope, ensure_ascii=False)}"
    return ask_text(system_prompt, user_prompt, max_tokens=250, tag=f"fix_steps.{issue}")

def response_writer_llm(state: FlowState) -> str:
    """Compose the entire chat message (tickets summary, steps, timing, one confirmation)."""
    system_prompt = (
        "Compose a friendly support message. "
        "Include: 1) what ticket actions happened (updated/opened with numbers), "
        "2) Fix Steps blocks per issue (already drafted: keep them as-is except minimal trims), "
        "3) a one-line timing sentence exactly as provided in 'timing_line', "
        "4) end with a single clarification/confirmation question if appropriate. "
        "Keep it concise and skimmable."
    )
    # Build a compact context
    made = {
        "plan": state.plan,
        "tools": {k: {"status": v.get("status"), "signals": v.get("signals", {})} for k,v in (state.tools_by_issue or {}).items()},
        "steps": state.steps_by_issue,
        "timing_line": make_timing_line(state.timing or {}),
        "original": state.user_text
    }
    user_prompt = json.dumps(made, ensure_ascii=False)
    return ask_text(system_prompt, user_prompt, max_tokens=600, tag="response_writer")

def supervisor_approve_llm(final_message: str, steps_by_issue: Dict[str,str], tools_by_issue: Dict[str,Any]) -> str:
    """Supervisor (approve): return 'APPROVED' or 'RETRY:<issue|message>' (one retry max)."""
    system_prompt = (
        "You are the Supervisor. Approve if the message is safe, clear, and helpful. "
        "If something is unsafe/unclear/missing, respond with 'RETRY:<short reason or issue>'. "
        "Otherwise respond ONLY 'APPROVED'."
    )
    user_prompt = json.dumps({"message": final_message, "steps": steps_by_issue, "tools": tools_by_issue}, ensure_ascii=False)
    out = ask_text(system_prompt, user_prompt, max_tokens=40, tag="supervisor.approve").strip().upper()
    if not out:
        out = "APPROVED"
    if "RETRY" in out and ":" not in out:
        out = "RETRY:clarify"
    return out

def status_writer_llm(tickets: List[Dict[str,Any]]) -> str:
    """Write a short, friendly status summary given a small list of tickets."""
    system_prompt = "Write a brief, friendly status update for the user. 3–5 lines max."
    user_prompt = json.dumps({"tickets": tickets}, ensure_ascii=False)
    return ask_text(system_prompt, user_prompt, max_tokens=200, tag="status.writer")

# ----------------------------
# Nodes (prefixed with node_)
# ----------------------------
store: Optional[TicketStore] = None

def node_issue_reader(state: FlowState) -> FlowState:
    out = issue_reader_llm(state.user_text)
    state.parsed_issues = out.get("issues", []) or []
    state.overall_confidence = float(out.get("overall_confidence", 0))
    parts = [f"{i['type']}:{i.get('confidence',0):.2f}" for i in state.parsed_issues]
    trace(f"[What You Said] {', '.join(parts)} overall={state.overall_confidence:.2f}")
    return state

def node_records_digest(state: FlowState) -> FlowState:
    """Build a tiny list of open tickets that match extracted categories."""
    open_like: List[Dict[str,Any]] = []
    if store:
        for i in state.parsed_issues[:3]:
            cat = issue_to_category(i["type"])
            ex = store.find_open_in(cat)
            if ex:
                open_like.append({"number": ex["number"], "type": i["type"], "category": cat})
    # stash for next step via transient field on state (not persisted; just recompute below if needed)
    state.actions.append(f"records_digest:{len(open_like)}")
    return state

def node_supervisor_plan(state: FlowState) -> FlowState:
    # Rebuild open_like for plan (same as digest above; keeps state simple)
    open_like: List[Dict[str,Any]] = []
    if store:
        for i in state.parsed_issues[:3]:
            cat = issue_to_category(i["type"])
            ex = store.find_open_in(cat)
            if ex:
                open_like.append({"number": ex["number"], "type": i["type"], "category": cat})
    plan_out = supervisor_plan_llm(state.parsed_issues, state.overall_confidence, open_like)
    state.plan = plan_out.get("plan", []) or []
    clar = plan_out.get("clarify")
    state.clarify_question = (clar or {}).get("question","") if clar else ""
    if state.clarify_question:
        state.await_input = True
        trace(f"[Supervisor (plan)] clarify: {state.clarify_question}")
    else:
        nice = []
        for p in state.plan:
            if p["action"] == "update_existing":
                nice.append(f"update existing **{p['issue']}** ({p.get('number') or '-'})")
            elif p["action"] == "create_new":
                nice.append(f"open new **{p['issue']}**")
            else:
                nice.append("answer **info**")
        trace(f"[Supervisor (plan)] " + "; ".join(nice))
    return state

def node_ticket_update(state: FlowState) -> FlowState:
    """Create/update tickets per plan; set group/category; add short notes."""
    if not store or state.clarify_question:
        return state
    notes = []
    for p in state.plan:
        issue = p["issue"]; action = p["action"]
        cat = issue_to_category(issue)
        group = issue_to_group(issue)
        if action == "update_existing" and p.get("number"):
            store.update_ticket(p["number"], {"state":"In Progress","assignment_group":group, "category":cat})
            notes.append(f"updated {p['number']}")
        elif action == "create_new":
            rec = store.open_ticket({
                "opened_by":"chat_user",
                "short_description": f"{issue.title()} issue from chat",
                "description": state.user_text,
                "category": cat,
                "assignment_group": group
            })
            p["number"] = rec["number"]
            notes.append(f"opened {rec['number']}")
        # respond_only (info) → no ticket
    if notes:
        trace(f"[Ticket Update] " + "; ".join(notes))
    return state

def node_tool_calls(state: FlowState) -> FlowState:
    """Call plain-language tools per issue and stash envelopes."""
    if state.clarify_question:
        return state
    for p in state.plan:
        issue = p["issue"]; action = p["action"]
        if action == "respond_only" or issue == "info":
            continue
        number = p.get("number")
        # Pull fields from parsed_issues
        fields = {}
        for it in state.parsed_issues:
            if it["type"] == issue:
                fields = it.get("fields", {}) or {}
                break
        user = fields.get("user") or "demo@corp.com"
        try:
            if issue == "password":
                env = call_tool("/tools/v1/password/reset", {"user": user, "delivery": "chat", "incident_ref": number})
            elif issue == "vpn":
                env = call_tool("/tools/v1/vpn/diagnose", {
                    "user": user,
                    "gateway": fields.get("gateway","vpn.corp.com"),
                    "client": fields.get("client","GlobalProtect"),
                    "version": fields.get("version","6.2.2"),
                    "incident_ref": number
                })
            elif issue == "network":
                env = call_tool("/tools/v1/network/diagnose", {
                    "user": user,
                    "adapter": fields.get("adapter","wifi"),
                    "os": fields.get("os","Windows 11"),
                    "symptoms": fields.get("symptoms",[]),
                    "incident_ref": number
                })
            else:
                env = {"ok": True, "status":"info_answered", "signals":{}}
            state.tools_by_issue[issue] = env
            trace(f"[Tools] {issue} -> {env.get('status')}")
        except Exception as e:
            state.tools_by_issue[issue] = {"ok": False, "error": str(e)}
            trace(f"[Tools] {issue} error: {e}")
    return state

def node_timing(state: FlowState) -> FlowState:
    """Compute a 'next update by' time and risk % ( for trace + message)."""
   
    next_update_by = (datetime.now() + timedelta(minutes=25)).strftime("%H:%M")
    risk = min(90, 20 + 10*len([p for p in state.plan if p['action'] != 'respond_only']))
    state.timing = {"next_update_by": next_update_by, "risk_pct": risk}
    trace(f"[Timing] {make_timing_line(state.timing)}")
    return state

def node_fix_steps(state: FlowState) -> FlowState:
    """LLM writes Fix Steps per issue using tool signals."""
    if state.clarify_question:
        return state
    for p in state.plan:
        issue = p["issue"]; action = p["action"]
        if action == "respond_only" or issue == "info":
            continue
        env = state.tools_by_issue.get(issue, {})
        steps = fix_steps_llm(issue, env, state.user_text)
        state.steps_by_issue[issue] = steps
        trace(f"[Steps ({issue})] {shorten(steps)}")
    return state

def node_response_writer(state: FlowState) -> FlowState:
    """LLM composes the entire final message (tickets summary + steps + timing)."""
    if state.clarify_question:
        state.final_message = state.clarify_question
        return state
    msg = response_writer_llm(state)
    state.final_message = msg
    return state

def node_supervisor_approve(state: FlowState) -> FlowState:
    """LLM approves the final message; single retry path if needed."""
    verdict = supervisor_approve_llm(state.final_message, state.steps_by_issue, state.tools_by_issue)
    if verdict.startswith("RETRY") and state.iteration_count == 0 and not state.clarify_question:
        state.iteration_count = 1
        # Ask the response writer to produce a clearer/safer variant (hint included)
        hint = "\n\nMake it shorter and clearer. Avoid any risky advice."
        state.final_message = response_writer_llm(state) + hint  # gentle nudge
        verdict = supervisor_approve_llm(state.final_message, state.steps_by_issue, state.tools_by_issue)
    state.verdict = verdict
    trace(f"[Supervisor] {verdict}")
    return state

def node_learning(state: FlowState) -> FlowState:
    """learning note (in-memory)."""
    rec = {"plan": state.plan, "verdict": state.verdict, "ts": datetime.utcnow().isoformat(timespec="seconds")+"Z"}
    state.learning_hints = {"last": rec}
    trace(f"[Learning] {shorten(rec)}")
    return state

def node_status(state: FlowState) -> FlowState:
    """Status path: LLM writes a short status based on last few tickets."""
    if store is None or store.df.empty:
        state.user_message = "No incidents yet."
        return state
    last = store.df.tail(3)[["number","short_description","state","assignment_group","opened_at"]]
    items = []
    for _, r in last.iterrows():
        items.append({
            "number": r["number"],
            "title": r["short_description"],
            "state": r["state"],
            "group": r["assignment_group"],
            "opened_at": r["opened_at"],
        })
    state.final_message = status_writer_llm(items)
    return state

def node_reply(state: FlowState) -> FlowState:
    """Set final user_message from the LLM-written final_message (or clarify question)."""
    state.user_message = state.final_message or "(no reply)"
    return state

# ----------------------------
# Router & graph
# ----------------------------
def looks_like_status(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["status", "where are we", "update on my issue", "progress"])

def build_graph() -> StateGraph:
    g = StateGraph(FlowState)
    # Nodes
    g.add_node("issue_reader",        node_issue_reader)
    g.add_node("records_digest",      node_records_digest)
    g.add_node("supervisor_plan",     node_supervisor_plan)
    g.add_node("ticket_update",       node_ticket_update)
    g.add_node("tool_calls",          node_tool_calls)
    g.add_node("timing",              node_timing)           
    g.add_node("fix_steps",           node_fix_steps)
    g.add_node("response_writer",     node_response_writer)  #all user text comes from here
    g.add_node("supervisor_approve",  node_supervisor_approve)
    g.add_node("learning",            node_learning)
    g.add_node("status",              node_status)
    g.add_node("reply",               node_reply)

    # Main path
    g.set_entry_point("issue_reader")
    g.add_edge("issue_reader", "records_digest")
    g.add_edge("records_digest", "supervisor_plan")
    g.add_edge("supervisor_plan", "ticket_update")
    g.add_edge("ticket_update", "tool_calls")
    g.add_edge("tool_calls", "timing")
    g.add_edge("timing", "fix_steps")
    g.add_edge("fix_steps", "response_writer")
    g.add_edge("response_writer", "supervisor_approve")
    g.add_edge("supervisor_approve", "learning")
    g.add_edge("learning", "reply")

    # Status shortcut
    g.add_edge("status", "reply")
    g.add_edge("reply", END)
    return g

# ----------------------------
# CLI helpers
# ----------------------------
def load_ticket_store(path: str) -> TicketStore:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    return TicketStore(df.fillna(""))

def process_message(graph, text: str) -> Dict[str, Any]:
    st = FlowState(user_text=text)
    if looks_like_status(text):
        out = graph.invoke(FlowState(user_text=text), start_at="status")
    else:
        out = graph.invoke(st)
    if hasattr(out, "model_dump"):
        return out.model_dump()
    if hasattr(out, "dict"):
        return out.dict()
    try:
        return vars(out)
    except Exception:
        return {"user_message": None, "actions": []}

# ----------------------------
# Main
# ----------------------------
def main():

    global store
    store = load_ticket_store(DEFAULT_TICKETS_CSV)

    graph = build_graph().compile()
    trace("[INFO] LangGraph compiled.")

    # Demo scenarios
    scenarios = [
        "I'm locked out and need a password reset ASAP. user: sam@corp.com",
        "VPN drops every 10 minutes from Chennai office. gateway: vpn.corp.com",
        "Where are we on my issue?",
        "Wi-Fi connected but no internet after sleep (Windows 11).",
        "Tell me about the company."
    ]
    for s in scenarios:
        state = process_message(graph, s)
        print("\n=== USER ==="); print(s)
        print("--- REPLY ---"); print(state.get("user_message") or "(no direct reply)")

    # Interactive loop
    print("\nType your own messages (q to quit):")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Bye.")
            break
        if user_text.lower() in {"q", "quit", "exit"}:
            break
        state = process_message(graph, user_text)
        print("Agent:", state.get("user_message") or "(no reply)")

    # snapshot
    out_path = "incidents_updated_demo.csv"
    store.to_csv(out_path)
    print(f"[INFO] Wrote updated dataset -> {out_path}")


if __name__ == "__main__":
    main()
