#!/usr/bin/env python3
"""
Tool Gateway (POC) — Access/Password Reset helpers for the agent

Endpoints (all JSON, POST, Bearer-auth):
  - /tools/v1/idp/status                 {} → { ok, incident|null, latency_ms }
  - /tools/v1/itsm/dir/user_lookup       { user } → { exists, locked, sspr_enabled, failed_24h, last_login_utc }
  - /tools/v1/itsm/dir/reset_password    { user, delivery? } → { ok, user, temp_password, policy, ttl_minutes, delivery, issued_utc }

Environment:
  TOOL_API_KEY=dev-secret          # shared secret (agent uses the same)
  PORT=8088                        # optional
  IDP_STATUS_SCENARIO=ok|outage    # optional default simulation

Run:
  pip install flask
  python tool_gateway.py
"""

import os, time, secrets, string, random
from datetime import datetime, timedelta, timezone
from functools import wraps
from flask import Flask, request, jsonify

app = Flask(__name__)

# --------------------------
# Auth
# --------------------------
def auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        expected = os.getenv("TOOL_API_KEY", "dev-secret")
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify({"error": "missing bearer auth"}), 401
        token = header.split(" ", 1)[1]
        if token != expected:
            return jsonify({"error": "bad token"}), 401
        return f(*args, **kwargs)
    return wrapper

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# --------------------------
# Health
# --------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "tool-gateway", "time": datetime.utcnow().isoformat() + "Z"})

# --------------------------
# 1) IdP status (mock)
# --------------------------
@app.post("/tools/v1/idp/status")
@auth
def idp_status():
    """
    Simulate identity provider status (Okta/AAD).
    Body: {} or {"simulate": "ok"|"outage"}
    Env default: IDP_STATUS_SCENARIO=ok|outage
    """
    body = request.json or {}
    simulate = (body.get("simulate") or os.getenv("IDP_STATUS_SCENARIO", "ok")).lower()
    ok = simulate != "outage"
    payload = {
        "ok": ok,
        "incident": None if ok else "auth-outage",
        "latency_ms": random.randint(90, 220)
    }
    log(f"[idp.status] simulate={simulate} -> {payload}")
    return jsonify(payload)

# --------------------------
# 2) User directory lookup (mock)
# --------------------------
_USERS = {
    # username -> profile
    "ram":   {"exists": True, "locked": True,  "sspr_enabled": True,  "failed_24h": 3},
    "jdoe":  {"exists": True, "locked": True,  "sspr_enabled": False, "failed_24h": 7},
    "alice": {"exists": True, "locked": False, "sspr_enabled": True,  "failed_24h": 0},
}

def _to_username(s: str) -> str:
    s = (s or "").strip()
    if "@" in s:  # email → username before '@'
        s = s.split("@", 1)[0]
    return s.lower()

@app.post("/tools/v1/itsm/dir/user_lookup")
@auth
def user_lookup():
    """
    Body: { "user": "ram" | "ram@corp.com" }
    """
    data = request.json or {}
    user_raw = (data.get("user") or "").strip()
    if not user_raw:
        return jsonify({"error": "user required"}), 400
    u = _to_username(user_raw)

    profile = _USERS.get(u, {"exists": False, "locked": False, "sspr_enabled": False, "failed_24h": 0})
    # add last_login timestamp
    profile = dict(profile)  # copy
    profile["user"] = user_raw
    profile["last_login_utc"] = (datetime.utcnow() - timedelta(hours=random.randint(1, 24))).isoformat() + "Z"

    log(f"[dir.lookup] user={user_raw} -> {profile}")
    return jsonify(profile)

# --------------------------
# 3) Reset password (mock)
# --------------------------
def _gen_password(n: int = 14) -> str:
    alphabet = string.ascii_lowercase + string.ascii_uppercase + string.digits + "!@#$%^&*-_"
    while True:
        p = "".join(secrets.choice(alphabet) for _ in range(n))
        if (any(c.islower() for c in p)
            and any(c.isupper() for c in p)
            and any(c.isdigit() for c in p)
            and any(c in "!@#$%^&*-_" for c in p)):
            return p

@app.post("/tools/v1/itsm/dir/reset_password")
@auth
def reset_password():
    """
    Body: { "user": "ram" | "ram@corp.com", "delivery": "chat|sms|email" }
    Returns FULL temp_password (POC ONLY — do not do this in prod).
    """
    data = request.json or {}
    user = (data.get("user") or "").strip()
    delivery = (data.get("delivery") or "chat").lower()
    if not user:
        return jsonify({"ok": False, "error": "user required"}), 400

    pwd = _gen_password(14)  # Full password returned (requested for POC)
    resp = {
        "ok": True,
        "user": user,
        "temp_password": pwd,                 # ← clear text, by request
        "policy": "min12 + upper/lower/number/symbol",
        "ttl_minutes": 15,
        "delivery": delivery,
        "issued_utc": datetime.utcnow().isoformat() + "Z",
    }
    log(f"[dir.reset] user={user} -> temp_password={pwd}")
    return jsonify(resp)

# --------------------------
# Boot
# --------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8088"))
    log(f"Starting Tool Gateway on 0.0.0.0:{port} (debug=False, reloader=False)")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
