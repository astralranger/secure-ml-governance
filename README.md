# Secure ML Model Governance & Zero-Downtime Deployment Platform

Post-training MLOps platform covering **ML security**, **model risk evaluation**,
**canary/shadow deployment**, **drift monitoring**, and **audit logging**.

## Architecture

```
Raw Payload --> Input Validator --> Security Filter --> API Router (Flask)
                                              |
                                      90% / 10% Canary Split
                                              |
                              Model v1 (Prod)   Model v2 (Canary)
                                              |
                              Audit Logs <--- Drift Monitor (Evidently)
```

## Repository Structure

```
secure-ml-governance-platform/
├── .github/workflows/           # CI/CD (QA + canary deploy)
├── config/model_config.json     # Traffic splitting & rollback rules
├── src/
│   ├── api/
│   │   ├── app.py               # Flask API + security middleware
│   │   ├── router.py            # Canary / shadow traffic router
│   │   └── security.py          # JWT, rate limiting, input validation
│   ├── evaluation/
│   │   ├── risk_evaluator.py    # QA / fairness / reproducibility gates
│   │   └── drift_monitor.py     # Statistical drift detection
│   └── models/                  # ML artifacts (v1.pkl, v2.pkl)
├── tests/                       # Penetration & model-quality tests
├── Dockerfile
└── requirements.txt
```

## Quick Start

```bash
cd C:\The Sketchbook\MLOPS\secure-ml-governance-platform

# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # PowerShell on Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train / generate model artifacts
python -m src.models.train_models

# 4. Start the API
python -m src.api.app

# 5. Get a JWT token
curl -X POST http://localhost:5000/auth/login `
  -H "Content-Type: application/json" `
  -d '{\"username\":\"admin\",\"password\":\"password\"}'

# 6. Run a prediction (use the access_token from step 5)
curl http://localhost:5000/predict `
  -H "Authorization: Bearer <ACCESS_TOKEN>" `
  -H "Content-Type: application/json" `
  -d '{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}'

# 7. View routing metrics
curl http://localhost:5000/metrics -H "Authorization: Bearer <ACCESS_TOKEN>"
```

### Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/auth/login` | — | Obtain a JWT access token |
| POST | `/predict` | JWT | Validated inference (canary/shadow routed) |
| GET | `/health` | — | Service health + loaded models |
| GET | `/metrics` | JWT | Traffic, error rate, latency, rollback state |
| GET | `/config` | JWT | Active routing configuration |
| GET | `/drift` | JWT | Drift snapshot computed from successful audit events |
| POST | `/admin/rollback/reset` | JWT | Clear an active automated rollback |

### Demos

```bash
# Interactive menu: login, predict, attack the API, watch routing & audit logs
python demo.py

# Inspect the SQLite audit trail
python inspect_audit.py
```

## Key Components

| Component | File | Responsibility |
|---|---|---|
| ML Security Gate | `src/api/security.py` | JWT auth, rate limiting, schema validation, adversarial checks |
| Traffic Router | `src/api/router.py` | Canary split, shadow (dark) traffic, automated rollback |
| Flask API | `src/api/app.py` | REST endpoints, audit logging to SQLite |
| Risk Evaluator | `src/evaluation/risk_evaluator.py` | Pre-deploy QA, fairness, reproducibility gates |
| Drift Monitor | `src/evaluation/drift_monitor.py` | KS-test drift vs. training baseline |

## MLOps Capabilities

This is a post-training MLOps project: it focuses on safely operating a model
after it has been trained rather than on data collection or model research.

* **Model versioning** — independent `v1` production and `v2` candidate
  artifacts make releases traceable and reversible.
* **Pre-deployment governance** — a runnable gate checks artifact integrity,
  accuracy, reproducibility, stress handling, and a configurable fairness
  proxy before a candidate can be promoted.
* **Secure model serving** — JWT or environment-provided API-key access,
  rate limits, strict request schemas, and adversarial-input rejection protect
  the inference boundary.
* **Zero-downtime releases** — canary routing gradually sends traffic to a
  candidate; shadow routing evaluates it without changing the user response.
* **Automated rollback** — candidate error rate and latency are measured
  independently; threshold breaches immediately send traffic back to `v1`.
* **Observability and auditability** — health, routing metrics, request IDs,
  model version, latency, outcome, and validated inputs are persisted locally.
* **Drift monitoring** — `/drift` compares recent successful requests with the
  saved training baseline, with a minimum-sample safeguard.
* **CI/CD enforcement** — GitHub Actions runs training, tests, and the
  pre-deployment admission gate before deployment configuration changes.

The bundled Iris data is deliberately small so the platform is easy to run.
For a more realistic demonstration, replace it with a domain dataset and its
own schema, preprocessing pipeline, baseline distribution, and model
artifacts.

## Configuration (`config/model_config.json`)

* `models.*.weight` — traffic share for canary routing (e.g., v1=90, v2=10)
* `routing.shadow_mode` / `shadow_percentage` — duplicate a controlled share of live traffic to the candidate model (dark traffic)
* `rollback.*` — auto-revert to v1 when v2 error rate or latency exceeds thresholds
* `security.*` — JWT secret, rate limits, feature bounds; set `API_KEY` to also authorize requests with `X-API-Key`

`/drift` uses the most recent successful audit records and the saved training
baseline. It reports `insufficient_data` until the configured minimum sample
count is available, rather than claiming a result from an unreliable sample.

## Testing

```bash
pytest tests/ -v
```

## CI/CD

* `.github/workflows/test_and_evaluate.yml` — runs QA, risk, and security tests on every push
* `.github/workflows/canary_deploy.yml` — CI/CD traffic-routing script for canary releases

The candidate deployment gate enforces artifact loading, deterministic output,
stress handling, a minimum accuracy, and (when a reviewed sensitive attribute
is supplied) a maximum demographic-parity gap. The bundled Iris slice is only
a technical demonstration; it is not a valid proxy for a real protected trait.
