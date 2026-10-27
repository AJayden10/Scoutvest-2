"""Check a deployed Free Agent stack from the outside, the way the browser sees it.

    python scripts/smoke_test.py --api https://your-api.example.com/api --origin https://your-app.vercel.app

Checks, in order (exits 1 on the first failure, so CI goes red):
  1. /health/ answers 200, the database is reachable, and a trained model has written predictions
  2. /players/?predicted=true returns players in the shape the frontend expects
  3. CORS: the API allows requests from the frontend's origin (otherwise the site silently shows demo data)
  4. the frontend itself loads (if --origin is given)
Standard library only, so it runs anywhere without installing the backend.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

REQUIRED_PLAYER_FIELDS = {"id", "name", "currentValue", "predictedValue", "upside", "riskScore", "signal"}


def fetch(url: str, origin: str | None = None, timeout: int = 30):
    req = urllib.request.Request(url, headers={"Accept": "application/json", **({"Origin": origin} if origin else {})})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return r.status, dict(r.headers), body, (time.perf_counter() - start) * 1000
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read(), (time.perf_counter() - start) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True, help="API base URL ending in /api")
    ap.add_argument("--origin", help="frontend origin, e.g. https://scoutvest-frontend.vercel.app")
    a = ap.parse_args()
    api = a.api.rstrip("/")
    failures = 0

    def check(ok: bool, msg: str):
        nonlocal failures
        print(("PASS " if ok else "FAIL ") + msg)
        failures += 0 if ok else 1
        return ok

    status, _, body, ms = fetch(f"{api}/health/")
    if not check(status == 200, f"GET /health/ -> {status} in {ms:.0f}ms"):
        print(body[:300].decode(errors="replace"))
        return 1
    health = json.loads(body)
    check(health.get("database") == "ok", f"database: {health.get('database')}")
    check(health.get("model", {}).get("trained") is True, f"trained model: {health.get('model', {}).get('version')}")
    check(health.get("playersWithPrediction", 0) > 0, f"players with a prediction: {health.get('playersWithPrediction')}")

    status, headers, body, ms = fetch(f"{api}/players/?predicted=true&limit=5&ordering=-upside", a.origin)
    if check(status == 200, f"GET /players/ -> {status} in {ms:.0f}ms"):
        results = json.loads(body).get("results", [])
        check(bool(results), f"{len(results)} players returned")
        if results:
            missing = REQUIRED_PLAYER_FIELDS - set(results[0])
            check(not missing, "player JSON has the fields the frontend reads" + (f" (missing {missing})" if missing else ""))

    if a.origin:
        allowed = headers.get("Access-Control-Allow-Origin") or headers.get("access-control-allow-origin")
        check(allowed in (a.origin.rstrip("/"), "*"),
              f"CORS allows {a.origin} (got {allowed!r}; set CORS_ALLOWED_ORIGINS on the API)")
        status, _, body, ms = fetch(a.origin)
        check(status == 200 and b"<div id=\"root\"" in body, f"frontend loads -> {status} in {ms:.0f}ms")

    print(f"\n{'All checks passed' if not failures else f'{failures} check(s) failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
