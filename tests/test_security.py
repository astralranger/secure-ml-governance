"""Component 1: ML Security & Input Validation Gate - penetration tests.

Covers schema validation, adversarial input checks, and auth requirements.
"""

from __future__ import annotations

import math

import pytest

from src.api.security import SecurityError, validate_prediction_payload

from src.models.dataset import EXAMPLE
VALID = EXAMPLE.copy()

def test_valid_payload_returns_clean_features():
    out = validate_prediction_payload(VALID)
    assert out == VALID
    assert isinstance(out["age"], int)
    assert isinstance(out["workclass"], str)


def test_missing_feature_rejected():
    payload = {k: v for k, v in VALID.items() if k != "workclass"}
    with pytest.raises(SecurityError):
        validate_prediction_payload(payload)


def test_unknown_feature_rejected():
    payload = dict(VALID, extra_feature=1.0)
    with pytest.raises(SecurityError):
        validate_prediction_payload(payload)


def test_null_feature_rejected():
    payload = dict(VALID, age=None)
    with pytest.raises(SecurityError):
        validate_prediction_payload(payload)


def test_wrong_type_rejected():
    payload = dict(VALID, age="not-a-number")
    with pytest.raises(SecurityError):
        validate_prediction_payload(payload)


def test_nan_rejected():
    payload = dict(VALID, age=float("nan"))
    with pytest.raises(SecurityError):
        validate_prediction_payload(payload)


def test_inf_rejected():
    payload = dict(VALID, age=float("inf"))
    with pytest.raises(SecurityError):
        validate_prediction_payload(payload)


def test_negative_inf_rejected():
    payload = dict(VALID, capital_loss=float("-inf"))
    with pytest.raises(SecurityError):
        validate_prediction_payload(payload)


def test_out_of_bounds_rejected():
    payload = dict(VALID, age=1e6)
    with pytest.raises(SecurityError):
        validate_prediction_payload(payload)


def test_numeric_values_in_categorical_fields_rejected():
    payload = {k: 7.0 for k in VALID}
    with pytest.raises(SecurityError):
        validate_prediction_payload(payload)


def test_non_dict_body_rejected():
    with pytest.raises(SecurityError):
        validate_prediction_payload([1, 2, 3])  # type: ignore[arg-type]


def test_boundary_values_accepted():
    """Values at the edge of the allowed range should pass."""
    payload = dict(VALID, age=100, capital_loss=99999)
    out = validate_prediction_payload(payload)
    assert out["age"] == 100


def test_predict_endpoint_requires_auth(client):
    resp = client.post(
        "/predict",
        json=VALID,
        content_type="application/json",
    )
    assert resp.status_code == 401


def test_predict_rejects_adversarial_payload(client, auth_headers):
    payload = dict(VALID, age=float("nan"))
    resp = client.post(
        "/predict",
        json=payload,
        headers=auth_headers,
        content_type="application/json",
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["error"] == "security_validation_failed"


def test_predict_rejects_unknown_feature(client, auth_headers):
    payload = dict(VALID, credit_score=720)
    resp = client.post(
        "/predict",
        json=payload,
        headers=auth_headers,
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_predict_valid_flow(client, auth_headers):
    resp = client.post(
        "/predict",
        json=VALID,
        headers=auth_headers,
        content_type="application/json",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert "prediction" in body
    assert "model_version" in body
    assert "latency_ms" in body
    assert "request_id" in body


def test_api_key_can_authorize_inference(client, app):
    app.config["INFERENCE_API_KEY"] = "test-key"
    resp = client.post("/predict", json=VALID, headers={"X-API-Key": "test-key"})
    assert resp.status_code == 200


def test_drift_endpoint_reports_insufficient_data(client, auth_headers):
    resp = client.get("/drift", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "insufficient_data"


def test_prometheus_metrics_are_exposed_for_authenticated_monitoring(client, auth_headers):
    resp = client.get("/metrics/prometheus", headers=auth_headers)
    assert resp.status_code == 200
    assert "text/plain" in resp.content_type
    assert "ml_governance_model_requests_total" in resp.get_data(as_text=True)


def test_model_metadata_endpoint_returns_training_lineage(client, auth_headers):
    resp = client.get("/models/metadata", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["dataset"]["feature_schema_version"] == "adult-income-v1"
    assert "v1" in body["models"]


def test_config_does_not_expose_signing_secret(client, auth_headers):
    response = client.get("/config", headers=auth_headers)
    assert response.status_code == 200
    assert "jwt_secret_key" not in response.json["security"]
