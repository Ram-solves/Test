#!/usr/bin/env python3
"""
ServiceNow-style Agentic POC (LangGraph + Azure OpenAI, LLM-only)

Run (CLI):
  export AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com/"
  export AZURE_OPENAI_API_KEY="<your-key>"
  export AZURE_OPENAI_API_VERSION="2024-08-01-preview"
  export AZURE_OPENAI_DEPLOYMENT="<your-chat-deployment-name>"

  pip install pandas scikit-learn openai langgraph pydantic
  python sn_langgraph_llm.py                # uses DEFAULT_DATA_PATH below if present
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

# ---------- Defaults: hardcode your Excel/CSV here if you want ----------
# If you set DEFAULT_DATA_PATH to a valid file, the script will use it when --data is not provided.
DEFAULT_DATA_PATH = "incidents_mock.csv"   # or "incidents.xlsx"
DEFAULT_SHEET_NAME = None                  # e.g., "Sheet1" for Excel

# ---------- Azure OpenAI (required) ----------
AZURE_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
AZURE_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

if not (AZURE_ENDPOINT and AZURE_API_KEY and AZURE_API_VERSION and AZURE_DEPLOYMENT):
    sys.exit("[ERROR] Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT.")

try:
    from openai import AzureOpenAI
    client = AzureOpenAI(api_key=AZURE_API_KEY, api_version=AZURE_API_VERSION, azure_endpoint=AZURE_ENDPOINT)
except Exception as e:
    sys.exit(f"[ERROR] Failed to init AzureOpenAI client: {e}")

def chat(system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 400) -> str:
    resp = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
        temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

def chat_json(system_prompt: str, user_prompt: str, temperature: float = 0.1, max_tokens: int = 300, retries: int = 2) -> Dict[str, Any]:
    """Force strict-JSON replies with minimal retries."""
    base_user = user_prompt + "\n\nReturn ONLY compact valid JSON. No prose."
    for attempt in range(retries + 1):
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role":"system","content":system_prompt + " Respond using ONLY strict JSON."},
                {"role":"user","content":base_user if attempt == 0 else "Your previous reply was not valid JSON. Return ONLY strict JSON for the same request."}
            ],
            temperature=temperature, max_tokens=max_tokens
        )
        text = resp.choices[0].message.content.strip()
        try:
            return json.loads(text)
        except Exception:
            if attempt == retries:
                raise ValueError(f"Model did not return valid JSON after {retries+1} attempts.\nLast output:\n{text}")
    raise RuntimeError("JSON retries exhausted")

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
    sys_p = "You are an intent classifier for ITSM chat. Keys: intent in {create,status,close,update}, number (nullable), fields (dict)."
    usr_p = f"User message:\n{user_text}\nInfer intent and any fields (short_description, description, category, urgency, priority, number if mentioned)."
    return chat_json(sys_p, usr_p)

def triage_agent(short_desc: str, description: str) -> Dict[str, Any]:
    sys_p = "You are a triage assistant. Output JSON keys: category, priority in [Critical,High,Moderate,Low], urgency in [High,Medium,Low], confidence (0-1)."
    usr_p = f"Short: {short_desc}\nDetails: {description}\nClassify and return JSON."
    data = chat_json(sys_p, usr_p)
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

def dedupe_agent(store: IncidentStore, new_text: str, threshold: float = 0.72) -> Optional[Tuple[str, float]]:
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

# ---------- Global store (set in main or Streamlit) ----------
store: Optional[IncidentStore] = None

# ---------- Graph Nodes ----------
def n_intent(state: FlowState) -> FlowState:
    data = intent_agent(state.user_text)
    state.intent = data.get("intent","")
    state.intent_fields = data.get("fields",{}) or {}
    state.short = state.intent_fields.get("short_description","").strip() or state.user_text.strip()
    state.desc  = state.intent_fields.get("description","").strip() or state.short
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

# ---------- Loader / KPIs ----------
def load_store(path: str, sheet: Optional[str] = None) -> IncidentStore:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx",".xls"]:
        df = pd.read_excel(path, sheet_name=sheet or 0)
    else:
        df = pd.read_csv(path)
    return IncidentStore(df.fillna(""))

def compute_kpis(s: IncidentStore) -> Dict[str, Any]:
    df = s.df
    total = len(df)
    auto_closed = len(df[(df["state"]=="Closed") & (df["resolved_at"]!="")])
    return {
        "total": total,
        "open": int((df["state"].isin(["Open","In Progress","On Hold"])).sum()),
        "closed": int((df["state"]=="Closed").sum()),
        "auto_resolution_rate_pct": round((auto_closed/total)*100,2) if total else 0.0
    }

# ---------- Graph Builder ----------
def build_graph() -> StateGraph:
    g = StateGraph(FlowState)
    g.add_node("intent", n_intent)
    g.add_node("status", n_status)
    g.add_node("dedupe", n_dedupe)
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
    g.add_edge("dedupe", "triage")
    g.add_edge("triage", "create")
    g.add_edge("create", "first_touch")
    g.add_edge("first_touch", "route_kb")
    g.add_edge("route_kb", "resolve_close")
    g.add_edge("status", END)
    g.add_edge("resolve_close", END)
    g.add_edge("other", END)
    return g

# ---------- Main ----------
def main():
    global store
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=False, help="CSV or Excel path; if omitted uses DEFAULT_DATA_PATH.")
    ap.add_argument("--sheet", default=None, help="Excel sheet name if applicable")
    args = ap.parse_args()

    data_path = args.data or DEFAULT_DATA_PATH
    sheet = args.sheet or DEFAULT_SHEET_NAME
    if not data_path or not os.path.exists(data_path):
        sys.exit(f"[ERROR] Data file not found: {data_path}. Set DEFAULT_DATA_PATH or pass --data.")

    store = load_store(data_path, sheet)
    graph = build_graph().compile()

    scenarios = [
        "I'm locked out and need a password reset ASAP.",
        "VPN drops every 10 minutes from Chennai office.",
        "Where are we on my issue?"
    ]

    for s in scenarios:
        st = FlowState(user_text=s)
        out = graph.invoke(st)
        print("\n=== USER ===")
        print(s)
        print("--- SYSTEM REPLY ---")
        print(out.user_message or "(no direct reply)")
        print("--- ACTION TRACE ---")
        print(", ".join(out.actions) if out.actions else "(none)")

    print("\n=== KPIs ===")
    print(json.dumps(compute_kpis(store), indent=2))

    out_path = "incidents_updated_demo.csv"
    store.to_csv(out_path)
    print(f"[INFO] Wrote updated dataset -> {out_path}")

if __name__ == "__main__":
    main()
