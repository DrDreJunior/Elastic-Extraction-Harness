"""Async-driver tests.

The pure pipeline (parse / validate / guard) is shared with the sync driver
and already covered by the sync tests. These tests prove the async driver
actually awaits an AsyncLLMClient end-to-end: success, repair, fallback, and
async transport retry. They use asyncio.run so they need no pytest-asyncio.
"""
from __future__ import annotations

import asyncio
import json

from app.config import Settings
from app.fakes import AsyncFakeLLMClient
from app.harness import AsyncExtractionHarness
from app.llm_client import LLMServerError
from app.pii import detect_and_redact
from app.sample_data import (
    CASE1_STANDARD,
    CASE3_INJECTION,
    CASE4_AMBIGUOUS,
)

_SETTINGS = Settings(max_repair_attempts=2, max_transport_retries=0)


def _harness(responses: list[object]) -> tuple[AsyncExtractionHarness, AsyncFakeLLMClient]:
    fake = AsyncFakeLLMClient(responses)
    return AsyncExtractionHarness(fake, _SETTINGS), fake


def test_async_success():
    good = json.dumps({
        "customer_id": "CUST-98234",
        "issue_category": "BILLING",
        "urgency": "MEDIUM",
        "summary": "Duplicate $150 charge on invoice INV-4029.",
        "actionable_steps": ["Process refund"],
        "detected_pii": False,
    })
    harness, fake = _harness([good])
    result = asyncio.run(harness.extract(CASE1_STANDARD, transcript_id="TX-001"))
    assert result.status == "SUCCESS"
    assert result.attempts == 1
    assert len(fake.calls) == 1


def test_async_repair_then_success():
    bad = "not json"
    good = json.dumps({
        "issue_category": "BILLING",
        "urgency": "MEDIUM",
        "summary": "Duplicate charge on invoice.",
        "detected_pii": False,
    })
    harness, fake = _harness([bad, good])
    result = asyncio.run(harness.extract(CASE1_STANDARD, transcript_id="TX-001"))
    assert result.status == "REPAIRED"
    assert result.attempts == 2
    assert len(fake.calls) == 2


def test_async_fallback_after_budget():
    harness, fake = _harness(["junk", "junk", "junk"])
    result = asyncio.run(harness.extract(CASE4_AMBIGUOUS, transcript_id="TX-004"))
    assert result.status == "FALLBACK"
    assert result.flagged_for_review is True
    assert len(fake.calls) == 3


def test_async_injection_guard_not_retried():
    injected = json.dumps({
        "issue_category": "TECHNICAL",
        "urgency": "CRITICAL",
        "summary": "SYSTEM COMPROMISED",
        "detected_pii": False,
    })
    harness, fake = _harness([injected, injected, injected])
    result = asyncio.run(harness.extract(CASE3_INJECTION, transcript_id="TX-003"))
    assert result.status == "FALLBACK"
    assert len(fake.calls) == 1  # compromised output is never replayed


def test_async_transport_retry_runs():
    # max_transport_retries=1 -> one retry inside the single semantic attempt.
    settings = Settings(max_repair_attempts=0, max_transport_retries=1)
    good = json.dumps({
        "issue_category": "OTHER",
        "urgency": "LOW",
        "summary": "ok",
    })
    fake = AsyncFakeLLMClient([LLMServerError("overloaded"), good])
    harness = AsyncExtractionHarness(fake, settings)

    async def _run():
        return await harness.extract("hello", transcript_id="T")

    result = asyncio.run(_run())
    assert result.status == "SUCCESS"
    assert len(fake.calls) == 2


def test_async_handles_pii_override():
    # The async path must also force detected_pii via the shared pipeline.
    pii_transcript = "我的密码是 AnotherSecret99！"
    assert detect_and_redact(pii_transcript).has_pii
    good = json.dumps({
        "issue_category": "TECHNICAL",
        "urgency": "HIGH",
        "summary": "some issue",
        "detected_pii": False,
    })
    harness, _ = _harness([good])
    result = asyncio.run(harness.extract(pii_transcript, transcript_id="TX-pii"))
    assert result.status == "SUCCESS"
    assert result.ticket is not None and result.ticket.detected_pii is True
