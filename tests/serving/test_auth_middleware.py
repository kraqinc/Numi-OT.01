"""Auth middleware must return 401/503 as JSON, never raise into 500."""

from __future__ import annotations

import os

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from serving.middleware.auth import APIKeyMiddleware


def _app():
    async def ok(request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/v1/chat", ok, methods=["GET"])])
    app.add_middleware(APIKeyMiddleware)
    return app


def test_missing_key_returns_401(monkeypatch):
    monkeypatch.setenv("NUMI_ENV", "production")
    monkeypatch.setenv("NUMI_API_KEY", "unit-test-secret-key-32chars!!")
    client = TestClient(_app())
    r = client.get("/v1/chat")
    assert r.status_code == 401
    assert "detail" in r.json()


def test_valid_key_passes(monkeypatch):
    monkeypatch.setenv("NUMI_ENV", "production")
    monkeypatch.setenv("NUMI_API_KEY", "unit-test-secret-key-32chars!!")
    client = TestClient(_app())
    r = client.get(
        "/v1/chat",
        headers={"Authorization": "Bearer unit-test-secret-key-32chars!!"},
    )
    assert r.status_code == 200
    assert r.text == "ok"


def test_live_is_public(monkeypatch):
    monkeypatch.setenv("NUMI_ENV", "production")
    monkeypatch.setenv("NUMI_API_KEY", "unit-test-secret-key-32chars!!")

    async def live(request):
        return PlainTextResponse("alive")

    app = Starlette(routes=[Route("/live", live)])
    app.add_middleware(APIKeyMiddleware)
    r = TestClient(app).get("/live")
    assert r.status_code == 200
