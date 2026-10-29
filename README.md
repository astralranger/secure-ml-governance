# Secure ML Governance & Zero-Downtime Deployment

An MLOps project for **UCI Adult Census Income**: predict whether annual income
exceeds $50,000, then govern how production and candidate models are served.

## Run locally (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.models.train_models
python -m src.evaluation.pre_deploy_gate
python -m pytest tests -q
python -m src.api.app
```

In a second terminal, activate the same environment and run `python demo.py`.
The demo logs in, predicts income, sends invalid records, exercises canary
routing, and reads the audit trail. Local API: http://localhost:5000.
Demo credentials: `admin` / `password`. Override `ADMIN_USERNAME`,
`ADMIN_PASSWORD`, `JWT_SECRET_KEY`, and optionally `API_KEY` through environment
variables. All protected endpoints accept JWT or `X-API-Key`.

## Dataset and reproducibility

Source: [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), Becker & Kohavi
(1996), DOI: https://doi.org/10.24432/C5XW20, CC BY 4.0.
Official source files are included in `data/raw` so the demonstration can train
offline. Missing files are downloaded from UCI; `ADULT_DATA_DIR` overrides the
cache location. No synthetic or Iris records are used.

* Official training file: 32,561 rows. A fixed stratified split yields 26,048
  fitting rows and 6,513 validation rows for admission decisions.
* Official test file: 16,281 rows, used only for the final reported test score.
* All 14 original fields are accepted, using snake_case names. Numeric fields
  have individual bounds; categorical fields have allowlists. `?` or JSON null
  is accepted only for workclass, occupation, and native_country.
* Each artifact contains its preprocessing pipeline: numeric median imputation
  and scaling, categorical mode imputation and one-hot encoding. Preprocessing
  is fitted exclusively on fitting rows. Test labels have their trailing period
  removed before binary encoding: `0 = <=50K`, `1 = >50K`.
* v1 is logistic regression; v2 is a random forest. Race and sex are retained for
  auditing but excluded from predictors. Other variables can still act as proxies.
* `src/models/model_summary.json` records source-file hashes, baseline hash,
  artifact hashes, schema, split sizes, library versions, quality results,
  official test accuracy, and MLflow run/registry IDs when tracking is enabled.

Measured results (seed 42, scikit-learn 1.7.2):

| Model | Validation accuracy | Official test accuracy | Validation sex parity gap |
|---|---|---|---|
| v1 Logistic Regression | 85.57% | 85.11% | 0.1715 |
| v2 Random Forest | 86.57% | 86.32% | 0.1762 |

Admission requires accuracy >= 0.80 and demographic-parity gap <= 0.20,
plus artifact/data integrity, deterministic inference, and stress handling.
The gap compares positive prediction rates for the dataset's recorded Female
and Male groups. This is a configurable academic demonstration threshold,
not a claim that the model is unbiased or suitable for consequential decisions.
Run `python -m src.evaluation.pre_deploy_gate --help` to inspect gate options.

## Prediction example

```powershell
$login = Invoke-RestMethod http://localhost:5000/auth/login -Method Post -ContentType 'application/json' -Body '{"username":"admin","password":"password"}'
$headers = @{ Authorization = "Bearer $($login.access_token)" }
$record = Get-Content examples/adult_request.json -Raw
Invoke-RestMethod http://localhost:5000/predict -Method Post -Headers $headers -ContentType 'application/json' -Body $record
```

The response includes the binary prediction, `income_label`, probabilities
ordered `[<=50K, >50K]`, model version, request ID, and latency.

## Governance and releases

The default weighted split is v1=90%, v2=10%. To change traffic without restarting:

```powershell
Invoke-RestMethod http://localhost:5000/admin/routing -Method Post -Headers $headers -ContentType 'application/json' -Body '{"v2_weight":25,"shadow_mode":false}'
# Shadow evaluates v2 in a bounded background queue while serving v1.
Invoke-RestMethod http://localhost:5000/admin/routing -Method Post -Headers $headers -ContentType 'application/json' -Body '{"v2_weight":0,"shadow_mode":true,"shadow_percentage":100}'
```

The live routing endpoint admits only a loaded candidate with a passing training
gate. Changes affect this running process; edit `config/model_config.json` for
restart defaults. Rollback overrides routing until explicitly reset. It stops
further shadow submissions and returns traffic to v1 when v2 error rate or
latency breaches configured thresholds after the minimum request count.
Already queued shadow tasks may finish. Shadow results have separate audit rows
sharing the served request's ID. Prediction disagreements are measured; live
accuracy requires ground-truth labels and is not inferred from agreement.

The API verifies schema and model hashes before unpickling. The admission CLI
also verifies source and baseline hashes. These hashes detect accidental changes;
only load artifacts/manifests from trusted sources.

Audit events are stored in `adult_audit.db`; `python inspect_audit.py` inspects
them. Drift uses successful, served requests, excludes shadow duplicates, and
waits for 30 samples. Numerical features use KS tests; categorical features use
chi-square tests. Reports compare distributions, not prediction accuracy.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | /auth/login | JWT login |
| POST | /predict | Validated income inference |
| GET | /health | Loaded model status (public) |
| GET | /metrics | Model traffic, errors, latency, shadow comparisons and rollback |
| GET | /metrics/prometheus | Prometheus exposition |
| GET | /models/metadata | Dataset/model lineage and evaluation |
| GET | /config | Current routing configuration |
| GET | /drift | Numeric and categorical distribution drift |
| POST | /admin/routing | Live canary/shadow changes |
| POST | /admin/rollback/reset | Clear rollback and candidate observation window |

## MLflow and dashboards

```powershell
# API at :8000, MLflow UI at :5001; training and admission run before serving.
docker compose up --build
# Also start Prometheus at :9090 and Grafana at :3000:
docker compose --profile monitoring up --build
```

Grafana has a provisioned **Adult Income - Secure ML Governance** dashboard for
traffic, errors, latency, rollback and weights. Local Grafana login: admin/admin;
set `GRAFANA_PASSWORD` to override. Prometheus configuration is generated from
the runtime `API_KEY`. Set that variable before starting Compose to override the
local demo key. Monitoring ports bind to localhost.

To track Python training directly:

```powershell
$env:MLFLOW_ENABLED = 'true'
$env:MLFLOW_TRACKING_URI = 'sqlite:///mlflow-local.db'
$env:MLFLOW_REGISTER_MODEL = 'true'
python -m src.models.train_models
mlflow ui --backend-store-uri sqlite:///mlflow-local.db --port 5001
```

MLflow logs both pipelines, signatures, accuracy, dataset SHA-256 and gate
results. Registration uses `adult-governance-model`; override with
`MLFLOW_REGISTERED_MODEL_NAME`. Explicitly enabled tracking failures stop
training instead of silently skipping lineage.

The demonstration uses one API worker with threads because routing, rollback
state and counters are process-local. Horizontal replicas need shared state.

## CI and project report

GitHub Actions trains and logs models to MLflow, runs security/model/integration tests, and enforces
the candidate admission CLI. The manual canary workflow validates weights and
uploads a release bundle containing artifacts, manifest, baseline and config.
It prepares a release; an actual remote rollout requires a configured hosting
target. Local live weight changes above demonstrate switching without restart.

The two-column IEEE-style [project report](Secure_ML_Governance_IEEE_Report.pdf)
covers the architecture, methodology, results and limitations.
