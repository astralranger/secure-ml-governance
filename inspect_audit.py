"""Quick inspection of the audit trail DB."""
import json
import sqlite3

con = sqlite3.connect("adult_audit.db")
cur = con.cursor()

total, errors, shadowed = cur.execute(
    "SELECT COUNT(*), COALESCE(SUM(error),0), COALESCE(SUM(shadow),0) FROM prediction_logs"
).fetchone()
print(f"audit rows: {total} | errors: {errors} | shadowed: {shadowed}")

rejections = cur.execute(
    "SELECT COUNT(*) FROM prediction_logs WHERE model_version = 'rejected'"
).fetchone()[0]
print(f"rejected-request audit rows: {rejections}")

print("--- first 5 audit records ---")
rows = cur.execute(
    "SELECT id, model_version, shadow, error, latency_ms, features, prediction "
    "FROM prediction_logs ORDER BY id LIMIT 5"
).fetchall()
for r in rows:
    print(
        {
            "id": r[0],
            "model": r[1],
            "shadow": r[2],
            "error": r[3],
            "latency_ms": round(r[4], 2),
            "features": json.loads(r[5]),
            "prediction": json.loads(r[6]),
        }
    )
