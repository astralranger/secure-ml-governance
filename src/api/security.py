"""Component 1: ML Security & Input Validation Gate.

Responsibilities:
  * JWT authentication for protected inference routes
  - API-key / IP based rate limiting (Flask-Limiter)
  * Strict schema validation of inference payloads (Pydantic)
  * Adversarial input checks: out-of-bounds values, NaN/Inf, extreme magnitudes,
    constant probe vectors, and oversized payloads
"""

from __future__ import annotations

import os
from functools import wraps
from typing import Any, Dict

import numpy as np
from flask import Request, current_app, jsonify, request
from flask_jwt_extended import JWTManager, verify_jwt_in_request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pydantic import BaseModel, Field, ValidationError, field_validator


# ---------------------------------------------------------------------------
# Schema: strict validation of the Iris inference payload
# ---------------------------------------------------------------------------
class PredictionInput(BaseModel):
    sepal_length: float = Field(..., description="sepal length (cm)")
    sepal_width: float = Field(..., description="sepal width (cm)")
    petal_length: float = Field(..., description="petal length (cm)")
    petal_width: float = Field(..., description="petal width (cm)")

    # Reject any undeclared fields (strict schema)
    model_config = {"extra": "forbid"}

    @field_validator("*")
    @classmethod
    def _must_be_finite(cls, v: Any) -> float:
        if v is None or not np.isfinite(float(v)):
            raise ValueError("must be a finite number (no NaN / Inf / null)")
        return float(v)


class SecurityError(Exception):
    """Raised when a request fails security validation."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Auth & rate-limiting initialization
# ---------------------------------------------------------------------------
def init_security(app, rate_limit_per_minute: int = 60):
    """Attach JWTManager and Flask-Limiter to the Flask app.

    Returns (jwt, limiter) for optional use elsewhere.
    """
    app.config.setdefault(
        "JWT_SECRET_KEY",
        os.getenv("JWT_SECRET_KEY", "dev-secret-change-me-min-32-bytes-for-hs256"),
    )
    jwt = JWTManager(app)

    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=[f"{rate_limit_per_minute} per minute"],
        storage_uri="memory://",
    )
    return jwt, limiter


def require_auth(fn):
    """Require a valid JWT or the configured ``X-API-Key``.

    API keys are intentionally supplied only through ``API_KEY`` at runtime;
    they are never committed to the routing configuration or audit database.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        configured_key = current_app.config.get("INFERENCE_API_KEY")
        supplied_key = request.headers.get("X-API-Key")
        if configured_key and supplied_key and supplied_key == configured_key:
            return fn(*args, **kwargs)
        verify_jwt_in_request()
        return fn(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Payload validation & adversarial scanning
# ---------------------------------------------------------------------------
def validate_prediction_payload(
    payload: Dict[str, Any],
    allowed_features=None,
    max_abs_value: float = 1000.0,
) -> Dict[str, float]:
    """Validate and sanitize an inference payload.

    Returns a clean feature dict. Raises SecurityError on any validation or
    adversarial failure.
    """
    allowed_features = allowed_features or [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    if not isinstance(payload, dict):
        raise SecurityError("Request body must be a JSON object", 400)

    # Reject unknown / extra fields before structural validation
    unknown = set(payload.keys()) - set(allowed_features)
    if unknown:
        raise SecurityError(
            f"Unexpected feature(s) in payload: {sorted(unknown)}", 400
        )
    missing = set(allowed_features) - set(payload.keys())
    if missing:
        raise SecurityError(
            f"Missing required feature(s): {sorted(missing)}", 400
        )

    # Pydantic structural + type validation (finite, float, bounds)
    try:
        validated = PredictionInput(**payload)
    except ValidationError as exc:
        raise SecurityError(f"Schema validation failed: {exc}", 400)

    features = validated.model_dump()

    # Adversarial heuristics
    _adversarial_scan(features, max_abs_value=max_abs_value)
    return features


def _adversarial_scan(features: Dict[str, float], max_abs_value: float = 1000.0) -> None:
    """Deep adversarial checks that run after schema validation passes.

    Checks:
      * extreme magnitudes (numerical instability / potential DoS)
      * constant feature vectors (model-boundary probing)
    """
    values = list(features.values())
    if not values:
        return

    max_abs = max(abs(v) for v in values)
    if max_abs > max_abs_value:
        raise SecurityError(
            f"Feature magnitude {max_abs:.2f} exceeds allowed bound "
            f"{max_abs_value}", 400
        )

    if len(values) > 1 and len(set(values)) == 1:
        raise SecurityError(
            "Constant feature vector rejected (possible adversarial probe)", 400
        )


def parse_json_body(request: Request) -> Dict[str, Any]:
    """Safely parse the request body as JSON, raising SecurityError on failure."""
    if request.content_length is not None and request.content_length > 1_048_576:
        raise SecurityError("Request body too large (>1 MB)", 413)
    data = request.get_json(silent=True)
    if data is None:
        raise SecurityError("Invalid or missing JSON body", 400)
    return data


def security_error_handler(e: SecurityError):
    """Flask error handler -> JSON error response."""
    return jsonify(
        {
            "error": "security_validation_failed",
            "message": e.message,
        }
    ), e.status_code
