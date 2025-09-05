#!/usr/bin/env python3
"""
Flask Tool Gateway —Mock APIs

Brief:
ANY username/email is accepted (no directory modes). Use the optional
`simulate` field to force specific scenarios. All tool endpoints require Bearer auth.

Endpoints:
- GET  /health
- POST /tools/v1/password/reset
- POST /tools/v1/vpn/diagnose
- POST /tools/v1/network/diagnose

Auth header:
  Authorization: Bearer <TOOL_API_KEY>

Environment:
  TOOL_API_KEY  (default: 'dev-secret')

Run:
  python tool_gateway.py
"""

import os
import random
import string
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Dict, Any, List

from flask import Flask, request, jsonify

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
TOOL_API_KEY = os.getenv("TOOL_API_KEY", "dev-secret")

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)

def now_utc() -> str:
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

def to_username(s: str) -> str:
    """Normalize emails/usernames into a simple account name (lowercase, domain stripped)."""
    s = (s or "").strip()
    if "@" in s:
        s = s.split("@", 1)[0]
    return s.lower()

def gen_password(n: int = 14) -> str:
    """Generate a strong temporary password (upper/lower/digit/symbol)."""
    rng = random.SystemRandom()
    upper = rng.choice(string.ascii_uppercase)
    lower = rng.choice(string.ascii_lowercase)
    digit = rng.choice(string.digits)
    symbol = rng.choice("!@#$%^&*-_+=?")
    pool = string.ascii_letters + string.digits + "!@#$%^&*-_+=?"
    rest = "".join(rng.choice(pool) for _ in range(max(0, n - 4)))
    pwd_list = list(upper + lower + digit + symbol + rest)
    rng.shuffle(pwd_list)
    return "".join(pwd_list)

def envelope(
    ok: bool = True,
    status: Optional[str] = None,
    incident_ref: Optional[str] = None,
    worknote: str = "",
    signals: Optional[Dict[str, Any]] = None,
    recommendations: Optional[List[str]] = None,
):
    """Build the consistent response envelope and headers if present."""
    out: Dict[str, Any] = {
        "ok": bool(ok),
        "status": status,
        "incident_ref": incident_ref,
        "worknote": worknote,
        "signals": signals or {},
    }
    if recommendations:
        out["recommendations"] = recommendations
    req_id = request.headers.get("X-Request-Id")
    idem = request.headers.get("Idempotency-Key")
    if req_id or idem:
        out["meta"] = {"request_id": req_id, "idempotency_key": idem}
    return out

def demo_last_login() -> str:
    return (datetime.utcnow() - timedelta(hours=random.randint(1, 24))).isoformat(timespec="seconds") + "Z"

@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "tool-gateway", "time_utc": now_utc()}), 200

# -----------------------------------------------------------------------------
# Password Reset 
# -----------------------------------------------------------------------------
@app.post("/tools/v1/password/reset")
@auth
def password_reset_simple():
    """
    Body (example):
    {
      "user": "ram@corp.com",
      "delivery": "chat" | "sms" | "email",     // optional; default "chat"
      "incident_ref": "INC123456",              // optional
      "simulate": "ok" | "system_down" | "unlocked" | "user_not_found"   // optional; default "ok"
    }

    Status:
      - "temp_password_issued"   (issued a temp password; 15-min TTL)
      - "guide_self_reset"       (account not locked; guide via portal)
      - "system_down"            (service is unavailable; try later)
      - "user_not_found"         (only if simulate forces it)
    """
    data = request.json or {}
    user_raw = (data.get("user") or "").strip()
    if not user_raw:
        return jsonify({"error": "user required"}), 400

    delivery = (data.get("delivery") or "chat").lower()
    incident_ref = data.get("incident_ref")
    simulate = (data.get("simulate") or "ok").lower()

    # ANY username works by default; only simulate not-found when asked
    if simulate == "user_not_found":
        out = envelope(
            ok=True,
            status="user_not_found",
            incident_ref=incident_ref,
            worknote="The username was not found. Please verify the account.",
            signals={"user": user_raw, "exists": False},
        )
        log(f"[password.reset] {user_raw} -> user_not_found")
        return jsonify(out), 200

    # Simulate system availability
    if simulate == "system_down":
        out = envelope(
            ok=True,
            status="system_down",
            incident_ref=incident_ref,
            worknote="Password reset system is currently unavailable. Advise user to try later.",
            signals={"system_available": False},
        )
        log(f"[password.reset] {user_raw} -> system_down")
        return jsonify(out), 200

    # Decide locked/unlocked:
    locked = False if simulate == "unlocked" else True

    if locked:
        temp = gen_password(14)
        ttl = int(data.get("ttl_minutes") or 15)
        out = envelope(
            ok=True,
            status="temp_password_issued",
            incident_ref=incident_ref,
            worknote=f"Temporary password issued via {delivery}. Valid for {ttl} minutes.",
            signals={
                "user": user_raw,
                "account_locked": True,
                "system_available": True,
                "temp_password": temp,
                "ttl_minutes": ttl,
                "delivery": delivery,
                "last_login_utc": demo_last_login(),
            },
            recommendations=[
                "Ask the user to sign in with the temporary password",
                "Remind them to change it immediately after login",
            ],
        )
        log(f"[password.reset] {user_raw} -> temp_password_issued")
        return jsonify(out), 200

    # Unlocked path → guide to portal
    out = envelope(
        ok=True,
        status="guide_self_reset",
        incident_ref=incident_ref,
        worknote="Account is not locked. Guide the user to reset password using the company portal.",
        signals={
            "user": user_raw,
            "account_locked": False,
            "system_available": True,
            "last_login_utc": demo_last_login(),
        },
    )
    log(f"[password.reset] {user_raw} -> guide_self_reset")
    return jsonify(out), 200

# -----------------------------------------------------------------------------
# VPN Diagnose (plain language)
# -----------------------------------------------------------------------------
@app.post("/tools/v1/vpn/diagnose")
@auth
def vpn_diagnose():
    """
    Body (example):
    {
      "user": "ram@corp.com",
      "client": "GlobalProtect",        // optional
      "version": "6.2.2",               // optional
      "gateway": "vpn.corp.com",        // optional
      "check_website": "https://intranet.corp.com", // optional (hint only)
      "incident_ref": "INC123456",      // optional
      "simulate": "ok" | "degraded" | "down"         // optional; default "ok"
    }

    Status:
      - "vpn_working"
      - "vpn_unstable"
      - "vpn_blocked"
    """
    data = request.json or {}
    user_raw = (data.get("user") or "").strip()
    if not user_raw:
        return jsonify({"error": "user required"}), 400

    incident_ref = data.get("incident_ref")
    client = data.get("client") or "GlobalProtect"
    version = data.get("version") or "unknown"
    gateway = data.get("gateway") or "vpn.corp.com"
    simulate = (data.get("simulate") or "ok").lower()

    rng = random.Random(to_username(user_raw) + version + gateway)

    if simulate == "down":
        gateway_reachable = False
        internet_ok = False
        client_supported = True
        needs_mfa_retry = False
        status = "vpn_blocked"
        worknote = "VPN gateway unreachable; basic checks failing."
    elif simulate == "degraded":
        gateway_reachable = True
        internet_ok = rng.choice([True, False])
        client_supported = True if client.lower() in ["globalprotect", "anyconnect"] else rng.choice([True, False])
        needs_mfa_retry = not internet_ok
        status = "vpn_unstable"
        worknote = "Gateway reachable but connection is unstable."
    else:
        gateway_reachable = True
        internet_ok = True
        client_supported = True
        needs_mfa_retry = False
        status = "vpn_working"
        worknote = "Gateway reachable; basic checks passed. Client version looks fine."

    signals = {
        "gateway_reachable": gateway_reachable,
        "internet_ok": internet_ok,
        "client_supported": client_supported,
        "needs_mfa_retry": needs_mfa_retry,
        "client": client,
        "version": version,
        "gateway": gateway,
        "probe_latency_ms": rng.randint(80, 180) if gateway_reachable else None,
    }

    if status == "vpn_working":
        recs = ["If drops continue, reinstall the VPN profile", "Retry login and approve MFA when prompted"]
    elif status == "vpn_unstable":
        recs = [
            "Reset the VPN adapter and re-import the profile",
            "Clear cached credentials and retry sign-in",
            "If possible, test on a different network",
        ]
    else:
        recs = [
            "Check local network connection (Wi-Fi/Ethernet)",
            "Try a mobile hotspot or different network",
            "If still blocked, contact the network team",
        ]

    out = envelope(
        ok=True,
        status=status,
        incident_ref=incident_ref,
        worknote=worknote,
        signals=signals,
        recommendations=recs,
    )
    log(f"[vpn.diagnose] {user_raw} -> {status} signals={signals}")
    return jsonify(out), 200

# -----------------------------------------------------------------------------
# Network/Laptop Diagnose (plain language)
# -----------------------------------------------------------------------------
@app.post("/tools/v1/network/diagnose")
@auth
def network_diagnose():
    """
    Body (example):
    {
      "user": "ram@corp.com",
      "os": "Windows 11",               // optional
      "adapter": "wifi" | "ethernet",   // optional; default "wifi"
      "symptoms": ["connected_no_internet","dns_issue"],  // optional list
      "incident_ref": "INC123456",      // optional
      "simulate": "ok" | "degraded" | "down"             // optional; default "ok"
    }

    Status:
      - "network_ok"
      - "network_degraded"
      - "network_down"
    """
    data = request.json or {}
    user_raw = (data.get("user") or "").strip()
    if not user_raw:
        return jsonify({"error": "user required"}), 400

    incident_ref = data.get("incident_ref")
    os_name = data.get("os") or "unknown"
    adapter = (data.get("adapter") or "wifi").lower()
    symptoms = data.get("symptoms") or []
    simulate = (data.get("simulate") or "ok").lower()

    rng = random.Random(to_username(user_raw) + os_name + adapter + str(symptoms))

    if simulate == "down":
        ip_assigned = False
        dns_resolves = False
        web_reachable = False
        captive_portal = False
        driver_ok = True
        status = "network_down"
        worknote = "No IP/DNS/web connectivity detected."
    elif simulate == "degraded":
        ip_assigned = True
        dns_resolves = rng.choice([True, False])
        web_reachable = dns_resolves and rng.choice([True, False])
        captive_portal = not web_reachable and rng.choice([True, False])
        driver_ok = rng.choice([True, True, False])  # mostly OK
        status = "network_degraded"
        worknote = "Adapter active but some checks are failing."
    else:
        ip_assigned = True
        dns_resolves = True
        web_reachable = True
        captive_portal = False
        driver_ok = True
        status = "network_ok"
        worknote = "Adapter active; IP and DNS look healthy. Web access is working."

    default_gateway = "192.168.1.1" if ip_assigned else None
    signals = {
        "ip_assigned": ip_assigned,
        "default_gateway": default_gateway,
        "dns_resolves": dns_resolves,
        "web_reachable": web_reachable,
        "captive_portal": captive_portal,
        "driver_ok": driver_ok,
        "adapter": adapter,
        "os": os_name,
        "probe_latency_ms": rng.randint(50, 140) if web_reachable else None,
    }

    if status == "network_ok":
        recs = ["If the issue recurs, flush DNS and renew IP", "Temporarily disable third-party firewall and retry"]
    elif status == "network_degraded":
        recs = [
            "Flush DNS and renew DHCP lease",
            "Forget and rejoin the Wi-Fi (prefer 5 GHz)" if adapter == "wifi" else "Test with a known-good cable/port",
            "Update network adapter driver",
            "Temporarily disable third-party firewall/proxy and retry",
        ]
    else:
        recs = [
            "Check physical connection / Wi-Fi signal",
            "Try a different network (mobile hotspot/ethernet)",
            "If still down, contact the network team",
        ]

    out = envelope(
        ok=True,
        status=status,
        incident_ref=incident_ref,
        worknote=worknote,
        signals=signals,
        recommendations=recs,
    )
    log(f"[network.diagnose] {user_raw} -> {status} signals={signals}")
    return jsonify(out), 200

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    log("Serving Flask app 'tool_gateway' (plain-language mocks; ANY username accepted)")
    app.run(host="0.0.0.0", port=8088, debug=False)
