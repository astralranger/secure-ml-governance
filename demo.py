"""Interactive demo of the Secure ML Governance Platform.

Start the API first, then run this:

    python -m src.api.app        # terminal 1
    python demo.py               # terminal 2 (after startup)

Walks through: JWT login, a valid prediction, attacking the API with
adversarial payloads, a canary traffic burst, and live routing metrics.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:5000"
from src.models.dataset import EXAMPLE
VALID = EXAMPLE.copy()

TOKEN: str = ""


def call(path: str, payload: dict | None = None, auth: bool = True) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if auth and TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode())


def section(title: str) -> None:
    print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")


def main() -> None:
    section("1. Login -> JWT token")
    status, body = call("/auth/login", {"username": "admin", "password": "password"}, auth=False)
    global TOKEN
    TOKEN = body["access_token"]
    print(f"  {status}  token: {TOKEN[:32]}...")

    section("2. Valid prediction (routed through canary layer)")
    status, body = call("/predict", VALID)
    print(f"  {status}  -> {json.dumps(body, indent=2)}")

    section("3. Attacking the API (should all be rejected)")
    attacks = [
        ("NULL feature (schema)", {**VALID, "age": None}),
        ("Unknown feature (injection probe)", {**VALID, "credit_score": 720}),
        ("Out-of-bounds magnitude (DoS)", {**VALID, "age": 1e6}),
        ("Invalid categories", {k: 7.0 for k in VALID}),
        ("No JWT (auth bypass)", VALID),
    ]
    for name, payload in attacks:
        status, body = call("/predict", payload, auth=not (name.startswith("No JWT")))
        print(f"  {status}  {name:36s} {body.get('message', body.get('error', ''))[:55]}")

    section("4. Canary burst: 30 requests split v1=90% / v2=10%")
    counts = {"v1": 0, "v2": 0}
    for _ in range(30):
        code, body = call("/predict", VALID)
        if code == 200:
            counts[body["model_version"]] += 1
    print(f"  v1: {counts['v1']:2d}   v2: {counts['v2']:2d}")

    section("5. Live routing metrics")
    status, body = call("/metrics")
    print(f"  {json.dumps(body, indent=2)}")

    section("6. Audit trail (SQLite last 3 rows)")
    con = sqlite3.connect("adult_audit.db")
    for row in con.execute(
        "SELECT id, model_version, error, latency_ms FROM prediction_logs ORDER BY id DESC LIMIT 3"
    ):
        print(f"  id={row[0]:>3}  model={row[1]:<10}  error={row[2]}  latency={round(row[3], 2)}ms")
    con.close()


if __name__ == "__main__":
    main()
