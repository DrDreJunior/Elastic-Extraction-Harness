"""Real integration tests against Ollama.

These run only when an Ollama instance is reachable at OLLAMA_HOST
(default http://192.168.2.7:11434); otherwise the whole module is skipped so the
offline suite stays green. Assertions are written against the *contract*
(status, enums, flags), not against exact model wording, so they survive model
non-determinism.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest
from app.injection import has_critical_business_evidence
from app.ollama import _base_url, _host, _model, ollama_async_harness
from app.sample_data import ALL_CASES
from app.schemas import ExtractionStatus as Status
from app.schemas import IssueCategory, Urgency

CASES = dict(ALL_CASES)


def _native_base() -> str:
    return _base_url(_host()).removesuffix("/v1")


def _pulled_models() -> list[str] | None:
    try:
        resp = httpx.get(_native_base() + "/api/tags", timeout=5.0)
        resp.raise_for_status()
    except Exception:  # noqa: BLE001 - any failure means "not reachable"
        return None
    return [m["name"] for m in resp.json().get("models", [])]


PULLED = _pulled_models()
pytestmark = pytest.mark.skipif(PULLED is None, reason=f"Ollama not reachable at {_base_url(_host())}")


@pytest.fixture(scope="module")
def harness():
    if PULLED and _model() not in PULLED:
        pytest.skip(f"model '{_model()}' not pulled on {_base_url(_host())}")
    return ollama_async_harness(json_mode=True)


def _run(harness, transcript_id: str):
    return asyncio.run(harness.extract(CASES[transcript_id], transcript_id=transcript_id))


def test_tx001_billing(harness):
    result = _run(harness, "TX-001")
    assert result.status in (Status.SUCCESS, Status.REPAIRED)
    assert result.ticket is not None
    assert result.ticket.issue_category == IssueCategory.BILLING


def test_tx002_pii(harness):
    result = _run(harness, "TX-002")
    assert result.status in (Status.SUCCESS, Status.REPAIRED)
    assert result.ticket is not None
    assert result.ticket.detected_pii is True


def test_tx003_injection_not_hijacked(harness):
    result = _run(harness, "TX-003")
    if result.ticket is not None:
        # Model resisted the injection and produced a real ticket.
        assert result.status in (Status.SUCCESS, Status.REPAIRED)
        if result.ticket.urgency == Urgency.CRITICAL:
            # CRITICAL is only allowed with independent business evidence.
            assert has_critical_business_evidence(CASES["TX-003"])
    else:
        # Model complied: the output guard rejected it without retrying.
        assert result.status == Status.FALLBACK


def test_tx004_ambiguous(harness):
    result = _run(harness, "TX-004")
    assert result.status in (Status.SUCCESS, Status.REPAIRED, Status.FALLBACK)
    if result.ticket is not None:
        assert result.ticket.issue_category in list(IssueCategory)
