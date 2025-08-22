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

# Map action tags to node ids. We support both full tokens and roots (before ':').
NODE_MAP = {
    "intent": "intent",
    "status": "status",
    "dedup": "dedupe",
    "clarify": "ensure",
    "triage": "triage",
    "created": "create",
    "create": "create",
    "user_update": "first",
    "first_touch": "first",
    "routed": "routekb",
    "kb": "routekb",
    "auto_resolved": "resolve",
    "worknote": "resolve",
    "closed": "end",
    "other_intent": "other",
    "other": "other",
}

def trace_to_nodes(actions):
    visited = {"intent"}  # entry always
    for token in (actions or []):
        token = (token or "").strip()
        if not token:
            continue
        # Try full token (e.g., "clarify:request")
        node = NODE_MAP.get(token)
        if not node:
            # Fallback to root before colon (e.g., "clarify")
            root = token.split(":", 1)[0]
            node = NODE_MAP.get(root, root)
        visited.add(node)
    return visited

def dynamic_trace_dot(actions) -> str:
    visited = trace_to_nodes(actions)

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

    def base_attrs(s: str) -> str:
        return s[s.find("[")+1 : s.rfind("]")]

    def style(name: str) -> str:
        return 'color="#2563eb", fillcolor="#dbeafe"' if name in visited else 'color="#bbbbbb", fillcolor="#f9f9f9"'

    dot = [
        "digraph G {",
        '  rankdir=LR;',
        '  graph [pad="0.2"];',
        '  node [shape=box, style="rounded,filled", fontsize=11];',
        '  edge [color="#888888"];'
    ]
    for name, base in nodes.items():
        dot.append(f'  {name} [{style(name)}, {base_attrs(base)}];')  # <-- note the comma after style()

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
