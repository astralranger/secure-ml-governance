"""Component 1 & 2 & 3: Flask API with security middleware, traffic routing,
and audit logging.

Endpoints:
  POST   /auth/login                 - obtain a JWT access token
  POST   /predict                    - authenticated, validated inference
  GET    /health                     - service health
  GET    /metrics                    - routing & rollback metrics (auth)
  GET    /config                     - active routing config (auth)
  POST   /admin/rollback/reset       - clear an active rollback (auth)
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from flask import Flask, jsonify, request
import pandas as pd
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .router import TrafficRouter
from .prometheus import render_router_metrics
from .security import (
    SecurityError,
    init_security,
    parse_json_body,
    require_auth,
    security_error_handler,
    validate_prediction_payload,
)
from ..evaluation.drift_monitor import compute_drift
from ..models.metadata import read_manifest

# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # repo root
CONFIG_PATH = BASE_DIR / "config" / "model_config.json"
MODEL_DIR = BASE_DIR / "src" / "models"


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Audit database (SQLite via SQLAlchemy)
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    client_ip = Column(String(64))
    model_version = Column(String(32))
    features = Column(Text)        # JSON string
    prediction = Column(Text)      # JSON string
    latency_ms = Column(Float)
    shadow = Column(Integer, default=0)   # 0/1 bool
    error = Column(Integer, default=0)    # 0/1 bool
    request_id = Column(String(36), index=True)
    http_status = Column(Integer, default=200)


_engine = None
_Session = None


def init_db(db_path: str):
    global _engine, _Session
    _engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
    Base.metadata.create_all(_engine)
    # ``create_all`` does not migrate existing SQLite tables.  These additive
    # columns keep audit databases created by earlier project versions usable.
    with _engine.begin() as connection:
        existing = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(prediction_logs)")}
        for column, definition in {
            "request_id": "VARCHAR(36)",
            "http_status": "INTEGER DEFAULT 200",
        }.items():
            if column not in existing:
                connection.exec_driver_sql(
                    f"ALTER TABLE prediction_logs ADD COLUMN {column} {definition}"
                )
    _Session = sessionmaker(bind=_engine, future=True)


def audit_log(entry: Dict[str, Any]) -> None:
    if _Session is None:
        return
    session = _Session()
    try:
        row = PredictionLog(
            client_ip=entry.get("client_ip"),
            model_version=entry.get("model_version"),
            features=json.dumps(entry.get("features", {})),
            prediction=json.dumps(entry.get("prediction", {})),
            latency_ms=entry.get("latency_ms"),
            shadow=int(entry.get("shadow", False)),
            error=int(entry.get("error", False)),
            request_id=entry.get("request_id"),
            http_status=int(entry.get("http_status", 200)),
        )
        session.add(row)
        session.commit()
    except Exception:  # audit failures must never break inference
        session.rollback()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app(config_path: str = str(CONFIG_PATH)) -> Flask:
    global CONFIG_PATH
    CONFIG_PATH = Path(config_path)
    config = load_config()

    app = Flask(__name__)

    # ---- security middleware ------------------------------------------- #
    sec_cfg = config.get("security", {})
    app.config["JWT_SECRET_KEY"] = sec_cfg.get(
        "jwt_secret_key", os.getenv("JWT_SECRET_KEY", "dev-secret-change-me")
    )
    api_key_env = sec_cfg.get("api_key_env_var", "API_KEY")
    app.config["INFERENCE_API_KEY"] = os.getenv(api_key_env)
    rate_limit = int(sec_cfg.get("rate_limit_per_minute", 60))
    _, limiter = init_security(app, rate_limit_per_minute=rate_limit)

    @app.errorhandler(SecurityError)
    def handle_security_error(e: SecurityError):
        return security_error_handler(e)

    # ---- load models ---------------------------------------------------- #
    models_cfg = config.get("models", {})
    models: Dict[str, Any] = {}
    for name, info in models_cfg.items():
        # Artifacts are stored flat inside src/models/ - take the filename only,
        # so config paths like "src/models/v1.pkl" or "v1.pkl" both work.
        artifact = MODEL_DIR / Path(info.get("artifact", f"{name}.pkl")).name
        if artifact.exists():
            models[name] = joblib.load(artifact)
        else:
            print(f"[warn] artifact not found: {artifact} (model {name} unavailable)")

    router = TrafficRouter(config, models)

    # ---- audit database ------------------------------------------------- #
    audit_cfg = config.get("audit", {})
    if audit_cfg.get("enabled", True):
        configured_db_path = audit_cfg.get("db_path", "audit.db")
        db_path = os.getenv("AUDIT_DB_PATH", str(BASE_DIR / configured_db_path))
        init_db(db_path)

    baseline_path = BASE_DIR / audit_cfg.get("baseline_path", "src/models/iris_baseline.csv")
    drift_window_size = int(audit_cfg.get("drift_window_size", 100))
    drift_min_samples = int(audit_cfg.get("drift_min_samples", 30))
    metadata_path = MODEL_DIR / "model_summary.json"

    # ---- routes --------------------------------------------------------- #
    allowed_features = sec_cfg.get(
        "allowed_feature_names",
        ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )
    max_abs_value = float(sec_cfg.get("max_abs_feature_value", 1000.0))

    # Demo credentials (override via env in production)
    ADMIN_USER = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "password")

    from flask_jwt_extended import create_access_token

    @app.post("/auth/login")
    @limiter.limit("10 per minute")
    def login():
        body = request.get_json(silent=True) or {}
        username = body.get("username", "")
        password = body.get("password", "")
        if username == ADMIN_USER and password == ADMIN_PASS:
            token = create_access_token(identity=username)
            return jsonify({"access_token": token}), 200
        return jsonify({"error": "invalid_credentials"}), 401

    @app.get("/health")
    def health():
        loaded = list(models.keys())
        production_ready = "v1" in models
        return jsonify(
            {
                "status": "ok" if production_ready else "degraded",
                "models_loaded": loaded,
                "rollback_active": router.get_metrics()["rollback_active"],
                "model_metadata_available": metadata_path.exists(),
            }
        ), 200

    @app.post("/predict")
    @require_auth
    @limiter.limit("60 per minute")
    def predict():
        request_id = str(uuid.uuid4())
        try:
            raw = parse_json_body(request)
        except SecurityError as exc:
            return security_error_handler(exc)

        try:
            features_dict = validate_prediction_payload(
                raw,
                allowed_features=allowed_features,
                max_abs_value=max_abs_value,
            )
        except SecurityError as exc:
            # audit the rejected request
            audit_log(
                {
                    "client_ip": request.remote_addr,
                    "model_version": "rejected",
                    "features": raw,
                    "prediction": {"error": exc.message},
                    "latency_ms": 0.0,
                    "shadow": False,
                    "error": True,
                    "request_id": request_id,
                    "http_status": exc.status_code,
                }
            )
            return security_error_handler(exc)

        features = np.array(
            [features_dict[f] for f in allowed_features], dtype=float
        )
        result, model_used = router.predict(features)
        inference_failed = "error" in result.get("prediction", {})
        status_code = 503 if inference_failed else 200

        audit_log(
            {
                "client_ip": request.remote_addr,
                "model_version": model_used,
                "features": features_dict,
                "prediction": result.get("prediction", {}),
                "latency_ms": result.get("latency_ms"),
                "shadow": result.get("shadow", False),
                "error": inference_failed,
                "request_id": request_id,
                "http_status": status_code,
            }
        )
        result["request_id"] = request_id
        if inference_failed:
            result["error"] = "inference_failed"
        return jsonify(result), status_code

    @app.get("/metrics")
    @require_auth
    def metrics():
        return jsonify(router.get_metrics()), 200

    @app.get("/metrics/prometheus")
    @require_auth
    def prometheus_metrics():
        """Prometheus-compatible metrics for dashboards and alert rules."""
        return app.response_class(
            render_router_metrics(router.get_metrics()),
            mimetype="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/models/metadata")
    @require_auth
    def model_metadata():
        """Expose reproducibility and lineage metadata without model binaries."""
        manifest = read_manifest(metadata_path)
        if not manifest:
            return jsonify({"error": "metadata_not_found"}), 404
        return jsonify(manifest), 200

    @app.get("/config")
    @require_auth
    def config_view():
        return jsonify(config), 200

    @app.get("/drift")
    @require_auth
    def drift():
        """Compute a current data-drift snapshot from successful audit events."""
        if _Session is None:
            return jsonify({"error": "audit_disabled"}), 503
        if not baseline_path.exists():
            return jsonify({"error": "baseline_not_found"}), 503

        requested_window = request.args.get("window", type=int)
        window = min(max(requested_window or drift_window_size, 1), 1000)
        session = _Session()
        try:
            rows = session.execute(
                select(PredictionLog.features)
                .where(PredictionLog.error == 0)
                .order_by(PredictionLog.id.desc())
                .limit(window)
            ).scalars().all()
        finally:
            session.close()
        live_rows = [json.loads(row) for row in reversed(rows)]
        if len(live_rows) < drift_min_samples:
            return jsonify({
                "status": "insufficient_data",
                "samples_available": len(live_rows),
                "minimum_samples": drift_min_samples,
            }), 200
        report = compute_drift(
            pd.read_csv(baseline_path), pd.DataFrame(live_rows), allowed_features
        ).to_dict()
        report.update({"status": "ok", "window_size": len(live_rows)})
        return jsonify(report), 200

    @app.post("/admin/rollback/reset")
    @require_auth
    def reset_rollback():
        router.reset_rollback()
        return jsonify({"status": "rollback_reset"}), 200

    return app


# ---------------------------------------------------------------------------
# Allow running directly: `python -m src.api.app`
# ---------------------------------------------------------------------------
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
