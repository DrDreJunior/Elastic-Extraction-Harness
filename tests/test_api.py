"""HTTP boundary tests. The serving LLM client is never constructed."""
from __future__ import annotations

import json

import pytest
from app import main
from app.fakes import AsyncFakeLLMClient
from app.harness import AsyncExtractionHarness
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    main._async_harness = None
    main._probe_client = None
    main._probe_state["checked_at"] = 0.0
    main._probe_state["ok"] = False
    with TestClient(main.app) as test_client:
        yield test_client
    main._async_harness = None


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_missing_key(client):
    if main.settings.llm_api_key:
        pytest.skip("OPENAI_API_KEY is set in this environment")
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["reason"] == "missing_api_key"


def test_blank_transcript_is_422(client):
    response = client.post("/extract", json={"transcript": "   "})
    assert response.status_code == 422


def test_extract_strips_raw_output_and_returns_request_id(client):
    model_output = json.dumps({
        "issue_category": "OTHER",
        "urgency": "LOW",
        "summary": "ok",
    })
    async_fake = AsyncFakeLLMClient([model_output])
    main._async_harness = AsyncExtractionHarness(async_fake, main.settings)
    response = client.post(
        "/extract",
        json={"transcript": "hello", "transcript_id": "demo-1"},
        headers={"x-request-id": "req-123"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCESS"
    assert body["request_id"] == "req-123"
    assert body["raw_llm_outputs"] == []
    assert response.headers["x-request-id"] == "req-123"


def test_ready_probe_failure_returns_503(client):
    """When an API key appears set but the probe fails, /ready reports not ready."""
    pytest.importorskip("openai")
    # Simulate key present and a failing probe (a bad/loopback base URL is
    # unreachable within the 5s probe timeout).
    key = main.settings.llm_api_key
    base_url = main.settings.llm_base_url
    object.__setattr__(main.settings, "llm_api_key", "test-key")
    object.__setattr__(main.settings, "llm_base_url", "http://127.0.0.1:9")
    main._probe_client = None
    try:
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["reason"] == "provider_unreachable"
    finally:
        object.__setattr__(main.settings, "llm_api_key", key)
        object.__setattr__(main.settings, "llm_base_url", base_url)
