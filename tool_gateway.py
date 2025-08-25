# tool_gateway.py
from flask import Flask, request, jsonify
from functools import wraps
import time, socket
from zoneinfo import ZoneInfo
from datetime import datetime
import requests  # pip install flask requests

API_KEY = "dev-secret"  # change for real use
app = Flask(__name__)

def auth(f):
    @wraps(f)
    def _(*a, **k):
        if request.headers.get("Authorization") != f"Bearer {API_KEY}":
            return jsonify({"error":"unauthorized"}), 401
        return f(*a, **k)
    return _

@app.post("/tools/v1/net/dns_lookup")
@auth
def dns_lookup():
    hostname = (request.json or {}).get("hostname","").strip()
    if not hostname:
        return jsonify({"error":"hostname required"}), 400
    try:
        infos = socket.getaddrinfo(hostname, None)
        addrs = sorted({i[4][0] for i in infos})
        return jsonify({"hostname": hostname, "addresses": addrs})
    except Exception as e:
        return jsonify({"hostname": hostname, "error": str(e)}), 502

@app.post("/tools/v1/net/url_check")
@auth
def url_check():
    data = request.json or {}
    url = data.get("url","").strip()
    timeout = float(data.get("timeout", 4.0))
    if not (url.startswith("http://") or url.startswith("https://")):
        return jsonify({"error":"url must start with http(s)://"}), 400
    t0 = time.perf_counter()
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        return jsonify({"url": url, "ok": 200 <= r.status_code < 400,
                        "status_code": r.status_code, "elapsed_ms": ms})
    except Exception as e:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        return jsonify({"url": url, "ok": False, "error": str(e), "elapsed_ms": ms}), 502

SITE_TZ = {
    "chennai": "Asia/Kolkata",
    "new york": "America/New_York",
    "london": "Europe/London",
    "sydney": "Australia/Sydney",
}
@app.post("/tools/v1/util/business_hours")
@auth
def business_hours():
    data = request.json or {}
    site = (data.get("site","") or "").lower().strip()
    tz = data.get("tz") or SITE_TZ.get(site) or "UTC"
    now_iso = data.get("now")  # optional ISO timestamp
    now = datetime.fromisoformat(now_iso) if now_iso else datetime.now(ZoneInfo(tz))
    start_h, end_h = data.get("start_hour", 9), data.get("end_hour", 18)
    is_open = start_h <= now.hour < end_h and now.weekday() < 5
    return jsonify({"tz": tz, "now": now.isoformat(), "is_open": is_open,
                    "window": f"{start_h:02d}:00-{end_h:02d}:00", "weekday": now.weekday()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8088, debug=True)
