#!/usr/bin/env python3
"""
ServiceNow-style Agentic POC (LangGraph + Azure OpenAI)
Now with agent-decided tool-calling to a Flask gateway (dns_lookup, url_check, business_hours).
"""

import os, sys, json, argparse, uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- .env loading ---
from dotenv import load_dotenv
from pathlib import Path as _Path
load_dotenv(override=False)
_script_dir = _Path(__file__).resolve().parent
for _name in [".env", ".env.local"]:
    _p = _script_dir / _name
    if _p.exists():
        load_dotenv(dotenv_path=_p, override=False)

# ---------- Defaults ----------
DEFAULT_DATA_PATH = "incidents_mock.csv"
DEFAULT_SHEET_NAME = None
TOOL_GATEWAY_URL=http://localhost:8088
TOOL_API_KEY=dev-secret
# ---------- Azure OpenAI ----------
AZURE_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
AZURE_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
if not (AZURE_ENDPOINT and AZURE_API_KEY and AZURE_API_VERSION and AZURE_DEPLOYMENT):
    sys.exit("[ERROR] Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT.")

# --- Optional CA/proxy ---
import httpx
http_client = None
ca_path = os.getenv("CA_CERT_PATH", "").strip() or None
proxies = {}
if os.getenv("HTTPS_PROXY"): proxies["https://"] = os.getenv("HTTPS_PROXY")
if os.getenv("HTTP_PROXY"):  proxies["http://"] = os.getenv("HTTP_PROXY")
try:
    if ca_path or proxies:
        http_client = httpx.Client(verify=ca_path if ca_path else True, proxies=proxies or None, timeout=30.0)
except Exception as e:
    print(f"[WARN] httpx client (CA/proxy) init failed: {e}")

# --- Azure client ---
from openai import AzureOpenAI
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    http_client=http_client
)

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------- LLM helpers ----------
def chat(system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 400) -> str:
    resp = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
        temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

def _extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    import re
    m = re.search(r'\{[\s\S]*\}', text)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def chat_json(system_prompt: str, user_prompt: str, temperature: float = 0.1, max_tokens: int = 300,
              retries: int = 2, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # 1) JSON mode
    try:
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
            temperature=temperature, max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content.strip())
    except Exception:
        pass
    # 2) Tool calling
    if schema is not None:
        try:
            resp = client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
                temperature=temperature, max_tokens=max_tokens,
                tools=[{"type":"function","function":{"name":"emit","parameters":schema}}],
                tool_choice={"type":"function","function":{"name":"emit"}},
            )
            msg = resp.choices[0].message
            if getattr(msg, "tool_calls", None):
                return json.loads(msg.tool_calls[0].function.arguments)
        except Exception:
            pass
    # 3) Strict retries
    base_user = user_prompt + "\n\nReturn ONLY compact valid JSON. No prose."
    last_txt = ""
    for attempt in range(retries + 1):
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role":"system","content":system_prompt + " Respond using ONLY strict JSON."},
                {"role":"user","content": base_user if attempt==0 else "Previous reply not JSON. Return ONLY JSON."}
            ],
            temperature=temperature, max_tokens=max_tokens
        )
        text = resp.choices[0].message.content.strip()
        last_txt = text
        try:
            return json.loads(text)
        except Exception:
            maybe = _extract_first_json_obj(text)
            if maybe is not None:
                return maybe
            if attempt == retries:
                raise ValueError(f"Model did not return valid JSON. Last output:\n{text}")

# ---------- Store ----------
class IncidentStore:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        required = ["number","opened_at","short_description","description","priority","category","urgency","state",
                    "assignment_group","assigned_to","sla_due","knowledge_match","resolved_at","closed_at"]
        for c in required:
            if c not in self.df.columns: self.df[c] = ""
        self.existing_numbers = set(self.df["number"].dropna().astype(str).tolist())
    def gen_number(self) -> str:
        while True:
            n = f"INC{100000 + int(uuid.uuid4().int % 900000)}"
            if n not in self.existing_numbers:
                self.existing_numbers.add(n); return n
    def create_incident(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        record = {
            "number": rec.get("number") or self.gen_number(),
            "opened_at": rec.get("opened_at") or now_str(),
            "short_description": rec.get("short_description","").strip(),
            "description": rec.get("description","").strip(),
            "priority": rec.get("priority","").strip(),
            "category": rec.get("category","").strip(),
            "urgency": rec.get("urgency","").strip(),
            "state": rec.get("state","Open"),
            "assignment_group": rec.get("assignment_group",""),
            "assigned_to": rec.get("assigned_to",""),
            "sla_due": rec.get("sla_due",""),
            "knowledge_match": rec.get("knowledge_match",""),
            "resolved_at": rec.get("resolved_at",""),
            "closed_at": rec.get("closed_at",""),
        }
        self.df = pd.concat([self.df, pd.DataFrame([record])], ignore_index=True)
        return record
    def update_incident(self, number: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        idx = self.df.index[self.df["number"] == number]
        if not len(idx): raise ValueError(f"Incident {number} not found")
        i = idx[0]
        for k,v in updates.items():
            if k in self.df.columns: self.df.at[i, k] = v
        return self.df.loc[i].to_dict()
    def get_incident(self, number: str) -> Optional[Dict[str, Any]]:
        r = self.df[self.df["number"] == number]
        return None if r.empty else r.iloc[0].to_dict()
    def list_open(self) -> pd.DataFrame:
        return self.df[self.df["state"].isin(["Open","In Progress","On Hold"])].copy()
    def to_csv(self, path: str): self.df.to_csv(path, index=False)

# ---------- Agents ----------
def intent_agent(user_text: str) -> Dict[str, Any]:
    sys_p = ("You are an intent classifier for ITSM chat. "
             "Keys: intent in {create,status,close,update}, number (nullable), fields (dict).")
    usr_p = (f"User message:\n{user_text}\n"
             "Infer intent and any fields (short_description, description, category, urgency, priority, number if mentioned).")
    schema = {
        "type":"object",
        "properties":{
            "intent":{"type":"string","enum":["create","status","close","update"]},
            "number":{"type":["string","null"]},
            "fields":{"type":"object","additionalProperties":True}
        },
        "required":["intent"]
    }
    out = chat_json(sys_p, usr_p, schema=schema)
    return out

def triage_agent(short_desc: str, description: str) -> Dict[str, Any]:
    sys_p = ("You are a triage assistant. Return JSON keys: category, "
             "priority in [Critical,High,Moderate,Low], urgency in [High,Medium,Low].")
    usr_p = f"Short: {short_desc}\nDetails: {description}\n"
    schema = {
        "type":"object",
        "properties":{
            "category":{"type":"string"},
            "priority":{"type":"string","enum":["Critical","High","Moderate","Low"]},
            "urgency":{"type":"string","enum":["High","Medium","Low"]},
            "confidence":{"type":["number","string"]}
        },
        "required":["category","priority","urgency"]
    }
    data = chat_json(sys_p, usr_p, schema=schema)
    data.setdefault("category","Software")
    data.setdefault("priority","Moderate")
    data.setdefault("urgency","Medium")
    try: data["confidence"] = float(data.get("confidence", 0.6))
    except: data["confidence"] = 0.6
    return data

def routing_agent(category: str) -> str:
    route = {"Access":"Service Desk","Network":"Network","Hardware":"Systems","Security":"Security","Software":"Applications"}
    return route.get(category, "Service Desk")

def kb_match_agent(category: str, text: str) -> str:
    cat2kb = {"Access":"KB-1001","Network":"KB-2044","Hardware":"KB-3102","Security":"KB-4520","Software":"KB-2044"}
    return cat2kb.get(category, "")

def dedupe_agent(store: 'IncidentStore', new_text: str, threshold: float = 0.72) -> Optional[Tuple[str, float]]:
    open_df = store.list_open()
    if open_df.empty: return None
    corpus = (open_df["short_description"].fillna("") + " " + open_df["description"].fillna("")).tolist()
    vect = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    X = vect.fit_transform(corpus + [new_text])
    sims = cosine_similarity(X[-1], X[:-1]).ravel()
    best_idx = sims.argmax()
    best_score = sims[best_idx]
    if best_score >= threshold:
        parent_number = open_df.iloc[best_idx]["number"]
        return parent_number, float(best_score)
    return None

def resolver_agent(category: str, text: str) -> Dict[str, Any]:
    lower = text.lower()
    if category=="Access" and any(k in lower for k in ["password","reset","locked","lockout"]):
        return {"resolved": True, "note": "Auto-resolved via password reset playbook (SSPR). User confirmed access restored."}
    steps = chat("Provide concise resolution steps as bullet points. Title: resolution steps", text)
    return {"resolved": False, "note": steps}

def user_comms_agent(purpose: str, payload: Dict[str, Any]) -> str:
    sys_p = "You craft concise, friendly ITSM user updates. 2-3 sentences max."
    usr_p = f"Purpose: {purpose}\nData: {json.dumps(payload)}\nWrite update:"
    return chat(sys_p, usr_p, max_tokens=120)

def guardrail_agent(old: Dict[str,Any], new: Dict[str,Any]) -> List[str]:
    issues = []
    if new.get("state") == "Closed" and not new.get("resolved_at"):
        issues.append("Closed without resolved_at.")
    if new.get("state") == "Resolved" and not new.get("knowledge_match"):
        issues.append("Resolved without knowledge article reference.")
    return issues

# ---------- Tool gateway config ----------
TOOL_GATEWAY_URL = os.getenv("TOOL_GATEWAY_URL","").rstrip("/")
TOOL_API_KEY = os.getenv("TOOL_API_KEY","")

def call_tool(tool: str, payload: dict, timeout=5.0) -> Tuple[bool, Dict[str,Any], str]:
    if not (TOOL_GATEWAY_URL and TOOL_API_KEY):
        return False, {}, "tool gateway not configured"
    MAP = {
        "dns_lookup": "/tools/v1/net/dns_lookup",
        "url_check": "/tools/v1/net/url_check",
        "business_hours": "/tools/v1/util/business_hours",
    }
    path = MAP.get(tool)
    if not path: return False, {}, f"unknown tool: {tool}"
    url = TOOL_GATEWAY_URL + path
    try:
        headers = {"Authorization": f"Bearer {TOOL_API_KEY}", "Content-Type":"application/json"}
        resp = (http_client or httpx.Client()).post(url, headers=headers, json=payload, timeout=timeout)
        ok = 200 <= resp.status_code < 300
        data = {}
        try: data = resp.json()
        except: pass
        return ok, data, "" if ok else f"http {resp.status_code}"
    except Exception as e:
        return False, {}, str(e)

# ---------- Conversational ----------
def clarifier_agent(missing: List[str], context: Dict[str, Any]) -> str:
    sys_p = "You ask the MINIMUM clarifying question needed to proceed. One short sentence."
    usr_p = f"Missing fields: {missing}\nContext: {json.dumps(context)}\nAsk exactly ONE short question:"
    return chat(sys_p, usr_p, max_tokens=60)

REQUIRED_FOR_CREATE = ["short_description"]

# ---------- State ----------
class FlowState(BaseModel):
    user_text: str = ""
    intent: str = ""
    intent_fields: Dict[str, Any] = Field(default_factory=dict)

    number: str = ""
    short: str = ""
    desc: str = ""
    category: str = ""
    priority: str = ""
    urgency: str = ""
    parent_number: Optional[str] = None

    user_message: str = ""
    actions: List[str] = Field(default_factory=list)
    error: str = ""

    # Conversational
    await_input: bool = False
    missing_fields: List[str] = Field(default_factory=list)

    # Tool planning/execution
    tool_name: str = ""
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    tool_result: Dict[str, Any] = Field(default_factory=dict)

# ---------- Nodes ----------
def n_intent(state: FlowState) -> FlowState:
    data = intent_agent(state.user_text)
    state.intent = data.get("intent","")
    fields = data.get("fields",{}) or {}
    num_from_top = data.get("number")
    if "number" in fields and not num_from_top:
        num_from_top = fields.get("number")
    state.intent_fields = fields
    state.intent_fields["number"] = num_from_top or fields.get("number")
    state.short = fields.get("short_description","").strip() or state.user_text.strip()
    state.desc  = fields.get("description","").strip() or state.short
    print(f"[intent] intent={state.intent} fields={json.dumps(state.intent_fields)}")
    return state

def route_from_intent(state: FlowState) -> str:
    return state.intent if state.intent in {"create","status"} else "other"

def n_status(state: FlowState) -> FlowState:
    num = state.intent_fields.get("number")
    if not num and store is not None and not store.df.empty:
        num = store.df.iloc[-1]["number"]
    if not num:
        state.user_message = "No incidents yet to report status on."
        state.actions.append("status:no_ticket")
        print("[status] no_ticket")
        return state
    inc = store.get_incident(str(num))
    if not inc:
        state.user_message = f"Could not find incident {num}."
        state.actions.append("status:not_found")
        print(f"[status] not_found number={num}")
        return state
    msg = user_comms_agent("status", {"number": inc["number"], "state": inc["state"], "assignment_group": inc["assignment_group"], "kb": inc["knowledge_match"]})
    state.user_message = msg
    state.actions.append(f"status:{inc['number']}")
    print(f"[status] number={inc['number']} state={inc['state']}")
    return state

def n_dedupe(state: FlowState) -> FlowState:
    link = dedupe_agent(store, f"{state.short} {state.desc}") if store else None
    if link:
        state.parent_number, score = link
        state.actions.append(f"dedup:{state.parent_number}")
        print(f"[dedupe] linked_to={state.parent_number} score={score:.2f}")
    else:
        print("[dedupe] none")
    return state

def n_ensure_min_fields(state: FlowState) -> FlowState:
    fields = state.intent_fields or {}
    ext2state = {"short_description":"short"}
    missing: List[str] = []
    for f in REQUIRED_FOR_CREATE:
        present = bool(fields.get(f)) or bool(getattr(state, ext2state.get(f, f), ""))
        if not present: missing.append(f)
    if missing:
        q = clarifier_agent(missing, {"user_text": state.user_text})
        state.user_message = q
        state.await_input = True
        state.missing_fields = missing
        state.actions.append("clarify:request")
        print(f"[clarify] ask={q!r}")
    else:
        print("[clarify] ok")
    return state

def n_triage(state: FlowState) -> FlowState:
    t = triage_agent(state.short, state.desc)
    state.category, state.priority, state.urgency = t["category"], t["priority"], t["urgency"]
    state.actions.append("triage")
    print(f"[triage] category={state.category} priority={state.priority} urgency={state.urgency}")
    return state

def n_create(state: FlowState) -> FlowState:
    rec = {
        "short_description": state.short, "description": state.desc,
        "priority": state.priority, "category": state.category, "urgency": state.urgency,
        "state": "Open", "opened_at": now_str()
    }
    inc = store.create_incident(rec)
    state.number = inc["number"]
    state.actions.append(f"created:{state.number}")
    print(f"[create] number={state.number}")
    return state

def n_first_touch(state: FlowState) -> FlowState:
    msg = user_comms_agent("first_touch", {"number": state.number, "summary": state.short, "category": state.category, "priority": state.priority})
    state.user_message = msg
    state.actions.append("user_update:first_touch")
    print("[comms] first_touch_sent")
    return state

def n_route_kb(state: FlowState) -> FlowState:
    ag = routing_agent(state.category)
    _ = store.update_incident(state.number, {"assignment_group": ag, "state": "In Progress"})
    kb = kb_match_agent(state.category, f"{state.short} {state.desc}")
    if kb:
        _ = store.update_incident(state.number, {"knowledge_match": kb})
        state.actions.append(f"kb:{kb}")
    state.actions.append(f"routed:{ag}")
    print(f"[route] group={ag} kb={kb or '-'}")
    return state

# ---- Tool planner (LLM) ----
def tool_planner_agent(intent: str, category: str, text: str, fields: Dict[str,Any]) -> Dict[str,Any]:
    sys_p = (
        "Decide if a diagnostic tool should be called now. "
        "Call ONLY if args are clearly present. "
        "Tools: dns_lookup(hostname), url_check(url), business_hours(site|tz). "
        "Return JSON: {call: bool, tool: string|null, args: object}."
    )
    usr_p = f"intent={intent}\ncategory={category}\ntext={text}\nfields={json.dumps(fields)}"
    schema = {
        "type":"object",
        "properties":{
            "call":{"type":"boolean"},
            "tool":{"type":["string","null"],"enum":["dns_lookup","url_check","business_hours", None]},
            "args":{"type":"object","additionalProperties":True}
        },
        "required":["call","tool","args"]
    }
    out = chat_json(sys_p, usr_p, schema=schema)
    return out if out.get("call") else {"call": False, "tool": None, "args": {}}

def n_tool_plan(state: FlowState) -> FlowState:
    plan = tool_planner_agent(state.intent, state.category, f"{state.short} {state.desc}", state.intent_fields)
    if plan.get("call"):
        state.tool_name = plan.get("tool","")
        state.tool_args = plan.get("args") or {}
        state.actions.append(f"tool_plan:{state.tool_name}")
        print(f"[tool_plan] call=True tool={state.tool_name} args={state.tool_args}")
    else:
        state.actions.append("tool_plan:none")
        print("[tool_plan] call=False")
    return state

def n_tool_exec(state: FlowState) -> FlowState:
    if not state.tool_name:
        state.actions.append("tool_exec:skipped")
        return state
    ok, data, err = call_tool(state.tool_name, state.tool_args)
    if ok:
        state.tool_result = data or {}
        state.actions.append(f"tool_exec:{state.tool_name}")
        print(f"[tool_exec] ok=True tool={state.tool_name} result={json.dumps(state.tool_result)[:200]}")
        # Add a brief internal note summary
        snippet = {k: state.tool_result.get(k) for k in list(state.tool_result)[:3]}
        state.actions.append("worknote:tool_result")
        print(f"[worknote] {snippet}")
    else:
        state.actions.append(f"tool_error:{state.tool_name}")
        print(f"[tool_exec] ok=False tool={state.tool_name} err={err}")
    return state

def n_resolve_close(state: FlowState) -> FlowState:
    dec = resolver_agent(state.category, f"{state.short} {state.desc}")
    if dec["resolved"]:
        inc = store.update_incident(state.number, {"state":"Resolved","resolved_at": now_str()})
        state.actions.append("auto_resolved")
        _ = user_comms_agent("resolution", {"number": state.number, "note": dec["note"]})
        inc_new = inc.copy(); inc_new["state"] = "Closed"
        if not guardrail_agent(inc, inc_new):
            _ = store.update_incident(state.number, {"state":"Closed","closed_at": now_str()})
            state.actions.append("closed")
        print("[resolve] auto_resolved=True closed=True")
    else:
        state.actions.append("worknote:drafted")
        print("[resolve] auto_resolved=False; worknote_drafted=True")
    return state

def n_other(state: FlowState) -> FlowState:
    state.user_message = "Intent not supported in this demo. Try reporting an issue or asking for status."
    state.actions.append("other_intent")
    print("[other] unsupported_intent")
    return state

# ---------- Utils ----------
def load_store(path: str, sheet: Optional[str] = None) -> 'IncidentStore':
    ext = os.path.splitext(path)[1].lower()
    if os.path.exists(path):
        if ext in [".xlsx",".xls"]:
            df = pd.read_excel(path, sheet_name=sheet or 0)
        else:
            df = pd.read_csv(path)
        return IncidentStore(df.fillna(""))
    cols = ["number","opened_at","short_description","description","priority","category","urgency","state",
            "assignment_group","assigned_to","sla_due","knowledge_match","resolved_at","closed_at"]
    print(f"[WARN] Data file not found: {path}. Initializing empty store.")
    return IncidentStore(pd.DataFrame(columns=cols))

def compute_kpis(s: 'IncidentStore') -> Dict[str, Any]:
    df = s.df
    total = len(df)
    auto_closed = len(df[(df["state"]=="Closed") & (df["resolved_at"]!="")])
    return {
        "total": total,
        "open": int((df["state"].isin(["Open","In Progress","On Hold"])).sum()),
        "closed": int((df["state"]=="Closed").sum()),
        "auto_resolution_rate_pct": round((auto_closed/total)*100,2) if total else 0.0
    }

def build_graph() -> StateGraph:
    g = StateGraph(FlowState)
    g.add_node("intent", n_intent)
    g.add_node("status", n_status)
    g.add_node("dedupe", n_dedupe)
    g.add_node("ensure_min_fields", n_ensure_min_fields)
    g.add_node("triage", n_triage)
    g.add_node("create", n_create)
    g.add_node("first_touch", n_first_touch)
    g.add_node("route_kb", n_route_kb)
    g.add_node("tool_plan", n_tool_plan)     # NEW
    g.add_node("tool_exec", n_tool_exec)     # NEW
    g.add_node("resolve_close", n_resolve_close)
    g.add_node("other", n_other)

    g.set_entry_point("intent")
    g.add_conditional_edges("intent", route_from_intent, {
        "create": "dedupe",
        "status": "status",
        "other": "other",
    })
    g.add_edge("dedupe", "ensure_min_fields")
    g.add_edge("ensure_min_fields", "triage")
    g.add_edge("triage", "create")
    g.add_edge("create", "first_touch")
    g.add_edge("first_touch", "route_kb")
    g.add_edge("route_kb", "tool_plan")      # NEW
    g.add_edge("tool_plan", "tool_exec")     # NEW
    g.add_edge("tool_exec", "resolve_close") # NEW

    g.add_edge("status", END)
    g.add_edge("resolve_close", END)
    g.add_edge("other", END)
    return g

def _to_state_dict(out) -> dict:
    if isinstance(out, dict): return out
    if hasattr(out, "model_dump"): return out.model_dump()
    if hasattr(out, "dict"): return out.dict()
    try: return vars(out)
    except: return {"user_message": None, "actions": []}

# ---------- Main ----------
def main():
    global store
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=False, help="CSV or Excel path; if omitted uses DEFAULT_DATA_PATH.")
    ap.add_argument("--sheet", default=None, help="Excel sheet name if applicable")
    ap.add_argument("--no-demo", action="store_true", help="Skip demo scenarios")
    ap.add_argument("--no-interactive", action="store_true", help="Skip interactive loop")
    args = ap.parse_args()

    data_path = args.data or DEFAULT_DATA_PATH
    sheet = args.sheet or DEFAULT_SHEET_NAME

    store = load_store(data_path, sheet)
    graph = build_graph().compile()

    scenarios = [
        "I'm locked out and need a password reset ASAP.",
        "VPN drops every 10 minutes from Chennai office. Check https://example.com/vpn too.",
        "Where are we on my issue?",
        "My VPN keeps disconnecting from the Chennai office every few minutes.",
        "Laptop WIN-7420 screen flickers after sleep.",
    ]

    if not args.no_demo:
        for s in scenarios:
            st_in = FlowState(user_text=s)
            out = graph.invoke(st_in)
            state = _to_state_dict(out)
            print("\n=== USER ===")
            print(s)
            print("--- SYSTEM REPLY ---")
            print(state.get("user_message") or "(no direct reply)")
            print("--- ACTION TRACE ---")
            print(", ".join(state.get("actions") or []))

        print("\n=== KPIs ===")
        print(json.dumps(compute_kpis(store), indent=2))

    if not args.no_interactive:
        print("\nType your own messages (q to quit):")
        pending_state: Optional[dict] = None
        while True:
            try:
                user_text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[INFO] Bye."); break
            if user_text.lower() in {"q","quit","exit"}: break

            if pending_state and ("short_description" in (pending_state.get("missing_fields") or [])):
                pending_state["intent_fields"]["short_description"] = user_text
                pending_state["short"] = user_text
                st_in = FlowState(**{**pending_state, "user_text": user_text, "await_input": False, "missing_fields": []})
                pending_state = None
            else:
                st_in = FlowState(user_text=user_text)

            out = graph.invoke(st_in)
            state = _to_state_dict(out)
            print("Agent:", state.get("user_message") or "(no reply)")
            print("Actions:", ", ".join(state.get("actions") or []))

            if state.get("await_input"):
                pending_state = state

    out_path = "incidents_updated_demo.csv"
    store.to_csv(out_path)
    print(f"[INFO] Wrote updated dataset -> {out_path}")

if __name__ == "__main__":
    main()
