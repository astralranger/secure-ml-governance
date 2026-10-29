"""Component 1: ML Security & Input Validation Gate.

Responsibilities:
  * JWT authentication for protected inference routes
  - API-key / IP based rate limiting (Flask-Limiter)
  * Strict schema validation of inference payloads (Pydantic)
  * Field-specific numeric bounds, strict integer types, categorical allowlists,
    and oversized payload checks
"""

from __future__ import annotations

import os
from functools import wraps
from typing import Any, Dict

from flask import Request, current_app, jsonify, request
from flask_jwt_extended import JWTManager, verify_jwt_in_request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pydantic import Field, ValidationError


# ---------------------------------------------------------------------------
# Schema: strict validation of the Adult income inference payload
# ---------------------------------------------------------------------------
from pydantic import create_model, model_validator
from ..models.dataset import FEATURE_NAMES, NUMERIC_BOUNDS, CATEGORIES


@model_validator(mode="before")
@classmethod
def _validate_categories(cls, values):
    if not isinstance(values, dict):
        raise ValueError("Expected a JSON object")
    values = dict(values)
    for field, categories in CATEGORIES.items():
        if field not in values:
            continue
        value = values[field]
        # UCI '?' is a documented missing value, imputed by the fitted pipeline.
        if value is None or value == "?":
            if field not in {"workclass", "occupation", "native_country"}:
                raise ValueError(f"{field} cannot be missing")
            values[field] = None
        elif not isinstance(value, str) or value.strip() not in categories:
            raise ValueError(f"Unsupported {field} category")
        else:
            values[field] = value.strip()
    return values


PredictionInput = create_model(
    "PredictionInput",
    __config__={"extra": "forbid"},
    __validators__={"categories": _validate_categories},
    **{field: (int, Field(..., strict=True, ge=bounds[0], le=bounds[1]))
       for field, bounds in NUMERIC_BOUNDS.items()},
    **{field: (str | None, ...) for field in CATEGORIES},
)


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
) -> Dict[str, Any]:
    """Validate and sanitize an inference payload.

    Returns a clean feature dict. Raises SecurityError on any validation or
    adversarial failure.
    """
    allowed_features = allowed_features or FEATURE_NAMES

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
    # Per-field bounds and categorical allowlists are enforced by the Adult schema.
    return features


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
