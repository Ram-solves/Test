#!/usr/bin/env python3
"""
ServiceNow-style Agentic POC (LangGraph + Azure OpenAI, LLM-only, Conversational)

What you get
- Agents (LLM): intent, triage, user comms, resolver (steps)
- Agents (deterministic): routing, KB map, TF-IDF dedupe, guardrails
- Orchestrator: LangGraph (nodes: intent -> (status | dedupe -> ensure_min_fields -> triage -> create -> first_touch -> route_kb -> resolve_close))
- Conversational: clarify node asks one crisp question if required info is missing
- Environment: .env auto-loaded (CWD + script dir); optional CA/proxies for corp networks
- Resilient JSON handling (json mode -> function-calling -> retries -> extract)

Quickstart
  pip install -r requirements_streamlit.txt   # (or see minimal deps below)
  # .env in the same folder:
  # AZURE_OPENAI_ENDPOINT=https://<your-endpoint>.openai.azure.com/
  # AZURE_OPENAI_API_KEY=<your-key>
  # AZURE_OPENAI_API_VERSION=2024-08-01-preview
  # AZURE_OPENAI_DEPLOYMENT=<your-chat-deployment>
  # (optional) CA_CERT_PATH=/path/to/corp-root.pem  HTTPS_PROXY=...  HTTP_PROXY=...

  python sn_langgraph_llm.py
  # or
  python sn_langgraph_llm.py --data incidents_mock.csv
  python sn_langgraph_llm.py --data incidents.xlsx --sheet Sheet1
"""

import os, sys, json, argparse, uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- .env loading from CWD and script directory ---
from dotenv import load_dotenv
from pathlib import Path as _Path
load_dotenv(override=False)  # CWD
_script_dir = _Path(__file__).resolve().parent
for _name in [".env", ".env.local"]:
    _p = _script_dir / _name
    if _p.exists():
        load_dotenv(dotenv_path=_p, override=False)

# ---------- Defaults: hardcode your Excel/CSV here if you want ----------
DEFAULT_DATA_PATH = "incidents_mock.csv"   # or "incidents.xlsx"
DEFAULT_SHEET_NAME = None                  # e.g., "Sheet1"

# ---------- Azure OpenAI (required) ----------
AZURE_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
AZURE_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

if not (AZURE_ENDPOINT and AZURE_API_KEY and AZURE_API_VERSION and AZURE_DEPLOYMENT):
    sys.exit("[ERROR] Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT.")

# --- Corporate network support: optional custom CA + proxies ---
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
    print(f"[WARN] Could not initialize custom httpx client (CA/proxy): {e}")

# --- Azure client ---
try:
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        http_client=http_client  # may be None; SDK will use defaults
    )
except Exception as e:
    sys.exit(f"[ERROR] Failed to init AzureOpenAI client: {e}")

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
    if not m:
        return None
    snippet = m.group(0)
    try:
        return json.loads(snippet)
    except Exception:
        return None

def chat_json(system_prompt: str,
              user_prompt: str,
              temperature: float = 0.1,
              max_tokens: int = 300,
              retries: int = 2,
              schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Force strict-JSON replies using (in order):
      1) response_format=json_object (if supported)
      2) tool/function-calling with JSON Schema (if provided)
      3) strict JSON retries
      4) fallback: extract first {...} from text
    """
    # 1) JSON mode (some Azure models support this)
    try:
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        txt = resp.choices[0].message.content.strip()
        return json.loads(txt)
    except Exception:
        pass  # not supported or blocked

    # 2) Function/tool calling with JSON Schema (reliable on chat models)
    if schema is not None:
        try:
            resp = client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
                temperature=temperature, max_tokens=max_tokens,
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "emit",
                        "description": "Return the structured JSON as arguments",
                        "parameters": schema
                    }
                }],
                tool_choice={"type": "function", "function": {"name": "emit"}},
            )
            msg = resp.choices[0].message
            if getattr(msg, "tool_calls", None):
                args = msg.tool_calls[0].function.arguments
                return json.loads(args)
        except Exception:
            pass

    # 3) Strict JSON retries
    base_user = user_prompt + "\n\nReturn ONLY compact valid JSON. No prose."
    last_txt = ""
    for attempt in range(retries + 1):
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role":"system","content":system_prompt + " Respond using ONLY strict JSON."},
                {"role":"user","content": base_user if attempt == 0 else
                    "Your previous reply was not valid JSON. Return ONLY strict JSON for the same request."}
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
                raise ValueError(
                    f"Model did not return valid JSON after {retries+1} attempts. Last output:\n{last_txt}"
                )

# ---------- Data Store ----------
class IncidentStore:
    """In-memory store sourced from CSV/Excel; writes back to CSV when finished."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        required = ["number","opened_at","short_description","description","priority","category","urgency","state",
                    "assignment_group","assigned_to","sla_due","knowledge_match","resolved_at","closed_at"]
        for c in required:
            if c not in self.df.columns:
                self.df[c] = ""
        self.existing_numbers = set(self.df["number"].dropna().astype(str).tolist())

    def gen_number(self) -> str:
        while True:
            n = f"INC{100000 + int(uuid.uuid4().int % 900000)}"
            if n not in self.existing_numbers:
                self.existing_numbers.add(n)
                return n

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
        if not len(idx):
            raise ValueError(f"Incident {number} not found")
        i = idx[0]
        for k,v in updates.items():
            if k in self.df.columns:
                self.df.at[i, k] = v
        return self.df.loc[i].to_dict()

    def get_incident(self, number: str) -> Optional[Dict[str, Any]]:
        r = self.df[self.df["number"] == number]
        return None if r.empty else r.iloc[0].to_dict()

    def list_open(self) -> pd.DataFrame:
        return self.df[self.df["state"].isin(["Open","In Progress","On Hold"])].copy()

    def to_csv(self, path: str):
        self.df.to_csv(path, index=False)

# ---------- Agents ----------
def intent_agent(user_text: str) -> Dict[str, Any]:
    sys_p = ("You are an intent classifier for ITSM chat. "
             "Keys: intent in {create,status,close,update}, number (nullable), fields (dict).")
    usr_p = (f"User message:\n{user_text}\n"
             "Infer intent and any fields (short_description, description, category, urgency, priority, number if mentioned).")
    schema = {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": ["create","status","close","update"]},
            "number": {"type": ["string","null"]},
            "fields": {
                "type": "object",
                "properties": {
                    "short_description": {"type": "string"},
                    "description": {"type": "string"},
                    "category": {"type": "string"},
                    "urgency": {"type": "string"},
                    "priority": {"type": "string"},
                    "number": {"type": ["string","null"]}
                },
                "additionalProperties": True
            }
        },
        "required": ["intent"],
        "additionalProperties": False
    }
    return chat_json(sys_p, usr_p, schema=schema)

def triage_agent(short_desc: str, description: str) -> Dict[str, Any]:
    sys_p = ("You are a triage assistant. Output JSON keys: "
             "category, priority in [Critical,High,Moderate,Low], "
             "urgency in [High,Medium,Low], confidence (0-1).")
    usr_p = f"Short: {short_desc}\nDetails: {description}\nClassify and return JSON."
    schema = {
        "type": "object",
        "properties": {
            "category": {"type": "string"},
            "priority": {"type": "string", "enum": ["Critical","High","Moderate","Low"]},
            "urgency": {"type": "string", "enum": ["High","Medium","Low"]},
            "confidence": {"type": ["number","string"]}
        },
        "required": ["category","priority","urgency"],
        "additionalProperties": True
    }
    data = chat_json(sys_p, usr_p, schema=schema)
    data.setdefault("category","Software")
    data.setdefault("priority","Moderate")
    data.setdefault("urgency","Medium")
    try:
        data["confidence"] = float(data.get("confidence", 0.6))
    except Exception:
        data["confidence"] = 0.6
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

# ---------- Conversational: clarifier ----------
def clarifier_agent(missing: List[str], context: Dict[str, Any]) -> str:
    sys_p = "You ask the MINIMUM clarifying question needed to proceed. One short sentence."
    usr_p = f"Missing fields: {missing}\nContext: {json.dumps(context)}\nAsk exactly ONE short question:"
    return chat(sys_p, usr_p, max_tokens=60)

REQUIRED_FOR_CREATE = ["short_description"]  # expand as needed

# ---------- LangGraph State ----------
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

    # Conversational flags
    await_input: bool = False
    missing_fields: List[str] = Field(default_factory=list)

# ---------- Global store (set in main or Streamlit) ----------
store: Optional[IncidentStore] = None

# ---------- Graph Nodes ----------
def n_intent(state: FlowState) -> FlowState:
    data = intent_agent(state.user_text)
    state.intent = data.get("intent","")
    # accept "number" at top-level or in fields
    num_from_top = data.get("number")
    fields = data.get("fields",{}) or {}
    if "number" in fields and not num_from_top:
        num_from_top = fields.get("number")
    state.intent_fields = fields
    state.intent_fields["number"] = num_from_top or fields.get("number")
    state.short = fields.get("short_description","").strip() or state.user_text.strip()
    state.desc  = fields.get("description","").strip() or state.short
    return state

def route_from_intent(state: FlowState) -> str:
    if state.intent in {"create","status"}:
        return state.intent
    return "other"

def n_status(state: FlowState) -> FlowState:
    num = state.intent_fields.get("number")
    if not num and store is not None and not store.df.empty:
        num = store.df.iloc[-1]["number"]
    if not num:
        state.user_message = "No incidents yet to report status on."
        state.actions.append("status:no_ticket")
        return state
    inc = store.get_incident(str(num))
    if not inc:
        state.user_message = f"Could not find incident {num}."
        state.actions.append("status:not_found")
        return state
    msg = user_comms_agent("status", {"number": inc["number"], "state": inc["state"], "assignment_group": inc["assignment_group"], "kb": inc["knowledge_match"]})
    state.user_message = msg
    state.actions.append(f"status:{inc['number']}")
    return state

def n_dedupe(state: FlowState) -> FlowState:
    link = dedupe_agent(store, f"{state.short} {state.desc}") if store else None
    if link:
        state.parent_number, score = link
        state.actions.append(f"dedup:{state.parent_number}")
        _ = user_comms_agent("dedup_notice", {"number": "(pending)", "parent": state.parent_number})
    return state

def n_ensure_min_fields(state: FlowState) -> FlowState:
    # Determine which required fields are missing (for create flow)
    fields = state.intent_fields or {}
    ext2state = {"short_description": "short"}
    missing: List[str] = []
    for f in REQUIRED_FOR_CREATE:
        present = bool(fields.get(f)) or bool(getattr(state, ext2state.get(f, f), ""))
        if not present:
            missing.append(f)
    if missing:
        q = clarifier_agent(missing, {"user_text": state.user_text})
        state.user_message = q
        state.await_input = True
        state.missing_fields = missing
        state.actions.append("clarify:request")
    return state

def n_triage(state: FlowState) -> FlowState:
    t = triage_agent(state.short, state.desc)
    state.category, state.priority, state.urgency = t["category"], t["priority"], t["urgency"]
    state.actions.append("triage")
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
    return state

def n_first_touch(state: FlowState) -> FlowState:
    msg = user_comms_agent("first_touch", {"number": state.number, "summary": state.short, "category": state.category, "priority": state.priority})
    state.user_message = msg
    state.actions.append("user_update:first_touch")
    return state

def n_route_kb(state: FlowState) -> FlowState:
    ag = routing_agent(state.category)
    _ = store.update_incident(state.number, {"assignment_group": ag, "state": "In Progress"})
    kb = kb_match_agent(state.category, f"{state.short} {state.desc}")
    if kb:
        _ = store.update_incident(state.number, {"knowledge_match": kb})
        state.actions.append(f"kb:{kb}")
    state.actions.append(f"routed:{ag}")
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
    else:
        state.actions.append("worknote:drafted")
    return state

def n_other(state: FlowState) -> FlowState:
    state.user_message = "Intent not supported in this demo. Try reporting an issue or asking for status."
    state.actions.append("other_intent")
    return state

# ---------- Utilities ----------
def load_store(path: str, sheet: Optional[str] = None) -> 'IncidentStore':
    ext = os.path.splitext(path)[1].lower()
    if os.path.exists(path):
        if ext in [".xlsx",".xls"]:
            df = pd.read_excel(path, sheet_name=sheet or 0)
        else:
            df = pd.read_csv(path)
        return IncidentStore(df.fillna(""))
    # If file doesn't exist, boot with an empty store (so demo still runs)
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
    g.add_edge("route_kb", "resolve_close")
    g.add_edge("status", END)
    g.add_edge("resolve_close", END)
    g.add_edge("other", END)
    return g

def _to_state_dict(out) -> dict:
    if isinstance(out, dict):
        return out
    if hasattr(out, "model_dump"):
        return out.model_dump()
    if hasattr(out, "dict"):
        return out.dict()
    try:
        return vars(out)
    except Exception:
        return {"user_message": None, "actions": []}

# ---------- Main ----------
def main():
    global store
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=False, help="CSV or Excel path; if omitted uses DEFAULT_DATA_PATH.")
    ap.add_argument("--sheet", default=None, help="Excel sheet name if applicable")
    ap.add_argument("--no-demo", action="store_true", help="Skip built-in demo scenarios and go straight to interactive loop")
    ap.add_argument("--no-interactive", action="store_true", help="Skip interactive loop")
    args = ap.parse_args()

    data_path = args.data or DEFAULT_DATA_PATH
    sheet = args.sheet or DEFAULT_SHEET_NAME

    store = load_store(data_path, sheet)
    graph = build_graph().compile()

    # ----- Demo scenarios -----
    scenarios = [
        "I'm locked out and need a password reset ASAP.",
        "VPN drops every 10 minutes from Chennai office.",
        "Where are we on my issue?",
        "My VPN keeps disconnecting from the Chennai office every few minutes.",
        # You can paste a real INC number printed above to demonstrate explicit status:
        # "Status on INC123457?"
    ]

    if not args.no_demo:
        for s in scenarios:
            st_in = FlowState(user_text=s)
            out = graph.invoke(st_in)
            state = _to_state_dict(out)
            msg = state.get("user_message") or "(no direct reply)"
            actions = state.get("actions") or []

            print("\n=== USER ===")
            print(s)
            print("--- SYSTEM REPLY ---")
            print(msg)
            print("--- ACTION TRACE ---")
            print(", ".join(actions) if actions else "(none)")

        print("\n=== KPIs ===")
        print(json.dumps(compute_kpis(store), indent=2))

    # ----- Interactive loop (optional) -----
    if not args.no_interactive:
        print("\nType your own messages (q to quit):")
        pending_state: Optional[dict] = None
        while True:
            try:
                user_text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[INFO] Bye.")
                break
            if user_text.lower() in {"q","quit","exit"}:
                break

            # If we're waiting for an answer, merge it
            if pending_state and ("short_description" in (pending_state.get("missing_fields") or [])):
                pending_state["intent_fields"]["short_description"] = user_text
                pending_state["short"] = user_text
                st_in = FlowState(**{**pending_state, "user_text": user_text, "await_input": False, "missing_fields": []})
                pending_state = None
            else:
                st_in = FlowState(user_text=user_text)

            out = graph.invoke(st_in)
            state = _to_state_dict(out)
            msg = state.get("user_message") or "(no direct reply)"
            actions = state.get("actions") or []
            print("Agent:", msg)
            print("Actions:", ", ".join(actions) if actions else "(none)")

            if state.get("await_input"):
                pending_state = state

    # Snapshot
    out_path = "incidents_updated_demo.csv"
    store.to_csv(out_path)
    print(f"[INFO] Wrote updated dataset -> {out_path}")

if __name__ == "__main__":
    main()
