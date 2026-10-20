"""Pytest fixtures: Flask test client, temp audit DB, and auth headers."""

from __future__ import annotations

import os
import tempfile

import pytest

from src.api.app import create_app, init_db


@pytest.fixture
def app():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    application = create_app()
    application.config["TESTING"] = True
    init_db(db_path)
    yield application
    # Dispose the SQLAlchemy engine so Windows can release the SQLite file
    import src.api.app as app_module

    if app_module._engine is not None:
        try:
            app_module._engine.dispose()
        except Exception:
            pass
    try:
        os.unlink(db_path)
    except PermissionError:
        pass


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth_headers(client):
    """Log in as the demo admin and return Bearer auth headers."""
    resp = client.post(
        "/auth/login",
        json={"username": "admin", "password": "password"},
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    token = resp.get_json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
