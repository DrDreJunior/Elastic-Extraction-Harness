"""Fallback tests: when the repair budget is exhausted, the harness must
return a structured FALLBACK result instead of raising.
"""
from __future__ import annotations

import json

from app.llm_client import LLMServerError
from app.sample_data import CASE4_AMBIGUOUS


def test_fallback_after_budget_exhausted(harness_factory):
    """Every attempt returns invalid JSON -> after max_repair_attempts+1 tries
    we should land on FALLBACK, not SUCCESS/REPAIRED."""
    junk_responses = ["not json at all", "still not json", "nope"]
    harness, fake = harness_factory(junk_responses)

    result = harness.extract(CASE4_AMBIGUOUS, transcript_id="TX-004")

    assert result.status == "FALLBACK"
    # max_repair_attempts=2 -> 1 initial + 2 repairs = 3 attempts.
    assert result.attempts == 3
    assert result.flagged_for_review is True
    assert result.ticket is None
    assert result.review_reason is not None
    assert "budget" in result.review_reason.lower() or "exhausted" in result.review_reason.lower()
    assert len(fake.calls) == 3


def test_fallback_on_persistent_transport_error(harness_factory):
    """If the LLM endpoint is down, we should still return a structured
    FALLBACK envelope -- never an unhandled exception."""
    harness, fake = harness_factory([
        LLMServerError("overloaded"),
        LLMServerError("overloaded"),
        LLMServerError("overloaded"),
    ])
    # Disable the transport retry inside the harness so we see all 3 attempts.
    result = harness.extract(CASE4_AMBIGUOUS, transcript_id="TX-004")

    assert result.status == "FALLBACK"
    assert result.flagged_for_review is True
    assert result.ticket is None
    assert any("transport_error" in e for e in result.validation_errors)


def test_fallback_result_is_serializable(harness_factory):
    """The FALLBACK envelope must survive JSON serialisation so it can be
    written to a review queue / database / Kafka topic."""
    harness, _ = harness_factory(["garbage", "garbage", "garbage"])
    result = harness.extract("x", transcript_id="T-1")
    dumped = result.model_dump_json()
    assert "FALLBACK" in dumped
