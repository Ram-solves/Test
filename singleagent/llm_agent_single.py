#!/usr/bin/env python3
"""
Access Reset Agent (POC) — LangGraph + Azure OpenAI + Flask tool

WHAT THIS SHOWS (end-to-end, simple)
1) Agent reads: "I'm locked out… I am ram."
2) Agent decides: This is an Access Reset.
3) Agent creates a ticket (in-memory store), sends first-touch.
4) Agent calls your Flask tool: /tools/v1/idp/status  (checks IdP is healthy)
5) Agent calls your Flask tool: /tools/v1/itsm/dir/user_lookup (confirms user + lock state)
6) Agent calls your Flask tool: /tools/v1/itsm/dir/reset_password (gets NORMAL temp password)
7) Agent posts a resolution message to the user (includes password in clear for demo).
8) Agent closes the ticket and logs worknotes.

REQUIRED FLASK TOOLS (all mocked/safe; no real directory)
- POST /tools/v1/idp/status
- POST /tools/v1/itsm/dir/user_lookup      { "user": "ram" }
- POST /tools/v1/itsm/dir/reset_password   { "user": "ram", "delivery": "chat" }

ENV (.env in the same folder)
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT=<your-chat-deployment>
TOOL_GATEWAY_URL=http://127.0.0.1:8088
TOOL_API_KEY=dev-secret

RUN
python access_reset_agent.py
"""

import os, sys, json, time, re, uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# ----- env / azure client -----
from dotenv import load_dotenv
load_dotenv()
from openai import AzureOpenAI
import httpx

AZURE_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT","")
AZURE_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY","")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION","")
AZURE_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT","")
if not (AZURE_ENDPOINT and AZURE_API_KEY and AZURE_API_VERSION and AZURE_DEPLOYMENT):
    sys.exit("[ERROR] Set AZURE_OPENAI_* env vars in .env")

TOOL_GATEWAY_URL = os.getenv("TOOL_GATEWAY_URL","").rstrip("/")
TOOL_API_KEY     = os.getenv("TOOL_API_KEY","")
if not (TOOL_GATEWAY_URL and TOOL_API_KEY):
    sys.exit("[ERROR] Set TOOL_GATEWAY_URL and TOOL_API_KEY in .env")

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)

def _now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def _t(): return time.strftime("%H:%M:%S")
def log(msg): print(f"[{_t()}] {msg}")
def trunc(x, n=160): 
    s = x if isinstance(x,str) else json.dumps(x, ensure_ascii=False)
    return s if len(s)<=n else s[:n]+" …"

# ----- tiny in-memory incident store (CSV snapshot at exit) -----
class IncidentStore:
    def __init__(self):
        self.df = pd.DataFrame(columns=[
            "number","opened_at","short_description","description",
            "priority","category","urgency","state",
            "assignment_group","knowledge_match",
            "resolved_at","closed_at","sla_due"
        ])
        self._numbers = set()

    def _gen_num(self) -> str:
        while True:
            n = f"INC{100000 + int(uuid.uuid4().int % 900000)}"
            if n not in self._numbers:
                self._numbers.add(n); return n

    def create(self, short, desc, category="Access", priority="High", urgency="High") -> Dict[str,Any]:
        rec = dict(
            number=self._gen_num(),
            opened_at=_now(),
            short_description=short, description=desc,
            priority=priority, category=category, urgency=urgency,
            state="Open", assignment_group="Service Desk",
            knowledge_match="KB-1001", resolved_at="", closed_at="", sla_due=""
        )
        self.df = pd.concat([self.df, pd.DataFrame([rec])], ignore_index=True)
        log(f"[STORE] created {rec['number']} Access/High → Service Desk, KB-1001")
        return rec

    def update(self, number: str, updates: Dict[str,Any]):
        idx = self.df.index[self.df["number"]==number]
        if not len(idx): raise ValueError(f"incident {number} not found")
        i = idx[0]
        for k,v in updates.items():
            if k in self.df.columns:
                self.df.at[i,k] = v
        log(f"[STORE] update {number} {trunc(updates)}")

    def get(self, number: str) -> Optional[Dict[str,Any]]:
        r = self.df[self.df["number"]==number]
        return None if r.empty else r.iloc[0].to_dict()

    def snapshot(self, path="incidents_access_demo.csv"):
        self.df.to_csv(path, index=False)
        log(f"[STORE] wrote snapshot → {path}")

store = IncidentStore()

# ----- LLM helpers -----
def chat_json(system_prompt: str, user_prompt: str, tag="llm.json") -> Dict[str,Any]:
    """Ask the model for strict JSON (uses response_format=json_object)."""
    log(f"[LLM:{tag}] ask")
    resp = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_prompt}],
        temperature=0.1, max_tokens=300,
        response_format={"type":"json_object"},
    )
    try:
        u = resp.usage
        log(f"[LLM:{tag}] tokens prompt={u.prompt_tokens} completion={u.completion_tokens} total={u.total_tokens}")
    except: pass
    txt = resp.choices[0].message.content.strip()
    log(f"[LLM:{tag}] out {trunc(txt)}")
    return json.loads(txt)

def user_message(purpose: str, payload: Dict[str,Any]) -> str:
    """Generate short user-facing text."""
    sys_p = "You write concise, friendly IT support messages (2 sentences max)."
    usr_p = f"Purpose: {purpose}\nData: {json.dumps(payload)}\nWrite message:"
    resp = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role":"system","content":sys_p},{"role":"user","content":usr_p}],
        temperature=0.2, max_tokens=120
    )
    msg = resp.choices[0].message.content.strip()
    log(f"[LLM:comms] {trunc(msg)}")
    return msg

# ----- tool client -----
def call_tool(path: str, payload: Dict[str,Any], timeout=6.0) -> Dict[str,Any]:
    """POST to Flask tool with auth; raise on error."""
    url = TOOL_GATEWAY_URL + path
    headers = {"Authorization": f"Bearer {TOOL_API_KEY}", "Content-Type": "application/json"}
    log(f"[TOOL→] POST {url} body={trunc(payload)}")
    with httpx.Client(timeout=timeout) as s:
        r = s.post(url, headers=headers, json=payload)
        body = r.json() if r.headers.get("content-type","").startswith("application/json") else {"raw": r.text}
        log(f"[TOOL←] {r.status_code} body={trunc(body)}")
        if r.status_code>=400:
            raise RuntimeError(f"tool http {r.status_code}: {body}")
        return body

# ----- State for the graph -----
class AccessState(BaseModel):
    user_text: str = ""
    intent: str = ""                # "access_reset"
    username: str = ""              # "ram" or email
    number: str = ""                # incident number after create
    idp_ok: Optional[bool] = None   # None until checked
    lookup: Dict[str,Any] = Field(default_factory=dict)
    temp_password: str = ""         # from reset_password tool
    user_message: str = ""          # last message to user
    actions: List[str] = Field(default_factory=list)
    error: str = ""

# ----- Agents / Nodes -----
def node_understand_request(st: AccessState) -> AccessState:
    """
    Understand the user's request and extract username if present.
    We expect 'access_reset' intent from lockout text.
    """
    sys_p = ("Classify the intent and extract username/email if present.\n"
             "Return keys: intent in ['access_reset'], username (string or null).")
    usr_p = f"User: {st.user_text}\n"
    out = chat_json(sys_p, usr_p, tag="intent")
    st.intent = out.get("intent") or "access_reset"
    # Try regex if model missed it
    user = out.get("username") or ""
    if not user:
        m = re.search(r'[\w\.\-]+@[\w\.\-]+\.\w+', st.user_text)
        user = m.group(0) if m else ""
        if not user:
            m2 = re.search(r'\b(?:i am|iam|this is|user(?:name)?:?)\s*([A-Za-z][A-Za-z0-9\.\-_]{2,})\b', st.user_text, re.I)
            user = m2.group(1) if m2 else ""
    st.username = user
    log(f"[DECIDE] intent={st.intent} username={st.username or '(missing)'}")
    st.actions.append("intent:access_reset")
    return st

def node_create_ticket(st: AccessState) -> AccessState:
    """Create the incident as Access/High and stash the number."""
    rec = store.create(
        short="User reports account lockout / password reset",
        desc=st.user_text,
        category="Access", priority="High", urgency="High"
    )
    st.number = rec["number"]
    st.actions.append(f"created:{st.number}")
    return st

def node_first_touch(st: AccessState) -> AccessState:
    """Send a quick acknowledgement to the user."""
    st.user_message = user_message("first_touch", {"number": st.number, "summary": "Access lockout / password reset"})
    st.actions.append("user_update:first_touch")
    return st

def node_check_idp(st: AccessState) -> AccessState:
    """Check IdP health (Okta/AzureAD mock). If down, we do not reset/unlock."""
    body = call_tool("/tools/v1/idp/status", {})
    st.idp_ok = bool(body.get("ok", True))
    st.actions.append(f"idp_ok:{st.idp_ok}")
    return st

def node_outage_response(st: AccessState) -> AccessState:
    """If IdP is down, route to Identity and inform the user."""
    store.update(st.number, {"state":"In Progress", "assignment_group":"Identity"})
    st.user_message = user_message("status", {
        "number": st.number,
        "note": "Authentication provider indicates an outage. Identity team engaged."
    })
    st.actions += ["routed:Identity","worknote:idp_outage"]
    return st

def node_lookup_user(st: AccessState) -> AccessState:
    """Lookup user in directory: exists/locked/sspr_enabled/etc."""
    if not st.username:
        # ask for username (in a real chat you'd await input; here we just error)
        st.error = "username_missing"
        st.user_message = "Please provide your username or email to proceed with the reset."
        st.actions.append("clarify:username")
        return st
    body = call_tool("/tools/v1/itsm/dir/user_lookup", {"user": st.username})
    st.lookup = body
    st.actions.append("lookup:user")
    return st

def node_plan_and_reset(st: AccessState) -> AccessState:
    """
    Decide and execute the reset action:
      - If user exists and is locked → call reset_password (delivery=chat)
      - If exists but not locked → still issue reset to be safe (POC simplification)
      - If not exists → ask for correct username
    """
    exists = bool(st.lookup.get("exists", True))
    locked = bool(st.lookup.get("locked", True))
    if not exists:
        st.user_message = f"User '{st.username}' not found. Please recheck the username or provide your email."
        st.actions.append("clarify:username_not_found")
        return st

    # Execute reset (demo)
    body = call_tool("/tools/v1/itsm/dir/reset_password", {"user": st.username, "delivery": "chat"})
    st.temp_password = body.get("temp_password","")
    st.actions += ["tool_exec:reset_password","worknote:reset_issued"]
    log(f"[RESET] temp password for {st.username}: {st.temp_password}")

    # Communicate and close
    msg = user_message("resolution", {
        "number": st.number,
        "info": f"Temporary password issued for {st.username}. Please sign in and change it immediately.",
        "temp_password": st.temp_password,
        "ttl_minutes": body.get("ttl_minutes", 15)
    })
    st.user_message = msg
    store.update(st.number, {"state":"Resolved","resolved_at": _now()})
    store.update(st.number, {"state":"Closed","closed_at": _now()})
    st.actions += ["auto_resolved","closed"]
    return st

# ----- Graph wiring -----
def build_graph() -> StateGraph:
    g = StateGraph(AccessState)
    g.add_node("understand_request", node_understand_request)
    g.add_node("create_ticket",      node_create_ticket)
    g.add_node("first_touch",        node_first_touch)
    g.add_node("check_idp",          node_check_idp)
    g.add_node("outage_response",    node_outage_response)
    g.add_node("lookup_user",        node_lookup_user)
    g.add_node("plan_and_reset",     node_plan_and_reset)

    g.set_entry_point("understand_request")
    g.add_edge("understand_request", "create_ticket")
    g.add_edge("create_ticket", "first_touch")
    g.add_edge("first_touch", "check_idp")
    # If IdP down → outage_response → END; else → lookup_user
    def after_idp(st: AccessState) -> str:
        return "outage_response" if st.idp_ok is False else "lookup_user"
    g.add_conditional_edges("check_idp", after_idp, {
        "outage_response":"outage_response",
        "lookup_user":"lookup_user"
    })
    g.add_edge("outage_response", END)
    g.add_edge("lookup_user", "plan_and_reset")
    g.add_edge("plan_and_reset", END)
    return g

# ----- Demo main -----
def main():
    log("[BOOT] Access Reset Agent POC starting")

    # Build graph
    graph = build_graph().compile()
    log("[GRAPH] compiled")

    # Demo input (the exact scenario you asked)
    demo_text = "I'm locked out and need a password reset. I am ram."
    log(f"\n=== USER ===\n{demo_text}")

    # Run
    st_in = AccessState(user_text=demo_text)
    st_out = graph.invoke(st_in)

    # Show results
    print("\n--- AGENT REPLY ---")
    print(st_out.user_message or "(no reply)")
    print("\n--- ACTION TRACE ---")
    print(", ".join(st_out.actions))
    print("\n--- INCIDENT SNAPSHOT ---")
    if st_out.number:
        print(json.dumps(store.get(st_out.number), indent=2))
    else:
        print("(no incident created)")

    # Persist CSV
    store.snapshot()

if __name__ == "__main__":
    main()
