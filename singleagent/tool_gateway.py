#!/usr/bin/env python3
"""
Flask Tool Gateway — relaxed directory (any username works)

Endpoints
- GET /health
- POST /tools/v1/idp/status
- POST /tools/v1/itsm/dir/user_lookup
- POST /tools/v1/itsm/dir/reset_password

Auth
- Send: Authorization: Bearer <TOOL_API_KEY>

Environment (optional)
- TOOL_API_KEY           (default: 'dev-secret')
- IDP_STATUS_SCENARIO    ('ok' | 'outage'; default: 'ok')
- STRICT_DIRECTORY       ('0' relaxed [default], '1' strict)
"""

import os
from datetime import datetime, timedelta
import random
import string
from functools import wraps

from flask import Flask, request, jsonify

# -----------------------
# Config
# -----------------------
TOOL_API_KEY = os.getenv("TOOL_API_KEY", "dev-secret")
IDP_STATUS_SCENARIO = os.getenv("IDP_STATUS_SCENARIO", "ok").lower()
STRICT_DIRECTORY = os.getenv("STRICT_DIRECTORY", "0") == "1"   # we default to RELAXED

app = Flask(__name__)

# -----------------------
# Helpers
# -----------------------
def log(msg: str):
    print(msg, flush=True)

def _now_utc() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def auth(f):
    @wraps(f)
    def inner(*args, **kwargs):
        hdr = request.headers.get("Authorization", "")
        ok = hdr.startswith("Bearer ") and hdr.split(" ", 1)[1] == TOOL_API_KEY
        if not ok:
            return jsonify({"error": "unauthorized"}), 401
        return f(*args, **kwargs)
    return inner

def _to_username(s: str) -> str:
    """Normalize emails/usernames for dictionary lookup."""
    s = (s or "").strip()
    if "@" in s:
        s = s.split("@", 1)[0]
    return s.lower()

def _gen_password(n: int = 14) -> str:
    """Strong temp password (contains upper/lower/digit/symbol)."""
    rng = random.SystemRandom()
    upper = rng.choice(string.ascii_uppercase)
    lower = rng.choice(string.ascii_lowercase)
    digit = rng.choice(string.digits)
    symbol = rng.choice("!@#$%^&*-_+=?")
    pool = string.ascii_letters + string.digits + "!@#$%^&*-_+=?"
    rest = "".join(rng.choice(pool) for _ in range(max(0, n - 4)))
    pwd = upper + lower + digit + symbol + rest
    # shuffle
    pwd_list = list(pwd)
    rng.shuffle(pwd_list)
    return "".join(pwd_list)

# Demo seed users (used only if STRICT_DIRECTORY=1)
_USERS = {
    "ram":   {"exists": True, "locked": True,  "sspr_enabled": True,  "failed_24h": 3},
    "jdoe":  {"exists": True, "locked": True,  "sspr_enabled": False, "failed_24h": 2},
    "alice": {"exists": True, "locked": False, "sspr_enabled": True,  "failed_24h": 0},
}

# -----------------------
# Endpoints
# -----------------------
@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "tool-gateway",
        "directory_mode": "strict" if STRICT_DIRECTORY else "relaxed",
        "idp_status_scenario": IDP_STATUS_SCENARIO,
        "time_utc": _now_utc(),
    })

@app.post("/tools/v1/idp/status")
@auth
def idp_status():
    """
    Body (optional): {"simulate": "ok" | "outage"}
    If omitted, uses IDP_STATUS_SCENARIO env (default 'ok').
    """
    data = request.json or {}
    simulate = (data.get("simulate") or IDP_STATUS_SCENARIO or "ok").lower()
    ok = simulate != "outage"
    payload = {
        "ok": ok,
        "incident": None if ok else "auth-outage",
        "latency_ms": random.randint(90, 230)
    }
    log(f"[idp.status] simulate={simulate} -> {payload}")
    return jsonify(payload), 200

@app.post("/tools/v1/itsm/dir/user_lookup")
@auth
def user_lookup():
    """
    Body: {"user": "ram" | "ram@corp.com" | "anyname"}
    RELAXED (default): unknown users -> exists:true, locked:true, sspr_enabled:true
    STRICT_DIRECTORY=1: unknown users -> exists:false
    """
    data = request.json or {}
    user_raw = (data.get("user") or "").strip()
    if not user_raw:
        return jsonify({"error": "user required"}), 400

    ukey = _to_username(user_raw)
    base = _USERS.get(ukey)

    if base is None:
        if STRICT_DIRECTORY:
            profile = {"exists": False, "locked": False, "sspr_enabled": False, "failed_24h": 0}
        else:
            # RELAXED: any name is treated as a valid, currently locked user with SSPR enabled
            profile = {"exists": True, "locked": True, "sspr_enabled": True, "failed_24h": 1}
    else:
        profile = dict(base)

    profile.update({
        "user": user_raw,
        "last_login_utc": (datetime.utcnow() - timedelta(hours=random.randint(1, 24))).isoformat(timespec="seconds") + "Z"
    })

    log(f"[dir.lookup] {user_raw} -> {profile}")
    return jsonify(profile), 200

@app.post("/tools/v1/itsm/dir/reset_password")
@auth
def reset_password():
    """
    Body: {"user": "<name or email>", "delivery": "chat" | "sms" | "email"}
    Always issues a new temporary password (demo).
    """
    data = request.json or {}
    user_raw = (data.get("user") or "").strip()
    if not user_raw:
        return jsonify({"error": "user required"}), 400

    delivery = (data.get("delivery") or "chat").lower()
    temp = _gen_password(14)
    ttl = 15  # minutes
    payload = {
        "ok": True,
        "user": user_raw,
        "temp_password": temp,
        "policy": "min12 + upper/lower/number/symbol",
        "ttl_minutes": ttl,
        "delivery": delivery,
        "issued_utc": _now_utc(),
    }
    log(f"[dir.reset] user={user_raw} delivery={delivery} -> temp={temp}")
    return jsonify(payload), 200

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    log("Serving Flask app 'tool_gateway' (relaxed directory by default)")
    app.run(host="0.0.0.0", port=8088, debug=False)
