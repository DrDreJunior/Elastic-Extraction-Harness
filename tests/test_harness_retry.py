"""Self-correction loop tests.

We script the fake LLM to emit a bad response on attempt 1 and a valid one on
attempt 2, then assert:
* the harness called the LLM twice,
* the second prompt contained the validation error from the first attempt,
* the final status is REPAIRED (not SUCCESS).
"""
from __future__ import annotations

import json

from app.sample_data import CASE1_STANDARD


def test_repair_loop_recovers_after_bad_json(harness_factory):
    # Attempt 1: model emits prose instead of JSON.
    bad_output = "I think the category is billing and urgency is medium."
    good_output = json.dumps({
        "customer_id": "CUST-98234",
        "issue_category": "BILLING",
        "urgency": "MEDIUM",
        "summary": "Duplicate $150 charge on invoice INV-4029.",
        "actionable_steps": ["Process refund"],
        "detected_pii": False,
    })
    harness, fake = harness_factory([bad_output, good_output])

    result = harness.extract(CASE1_STANDARD, transcript_id="TX-001")

    assert result.status == "REPAIRED"
    assert result.attempts == 2
    assert result.flagged_for_review is False
    assert result.ticket is not None
    assert result.ticket.issue_category == "BILLING"
    # The second prompt must contain the validation error feedback.
    second_prompt = fake.calls[1]["user"]
    assert "failed validation" in second_prompt.lower() or "valid json" in second_prompt.lower()


def test_repair_loop_recovers_after_schema_violation(harness_factory):
    # Attempt 1: JSON but invalid enum.
    bad = json.dumps({
        "customer_id": "CUST-1",
        "issue_category": "REFUND",  # invalid
        "urgency": "SOON",           # invalid
        "summary": "x",
    })
    good = json.dumps({
        "customer_id": "CUST-1",
        "issue_category": "BILLING",
        "urgency": "MEDIUM",
        "summary": "Duplicate charge.",
        "actionable_steps": [],
        "detected_pii": False,
    })
    harness, fake = harness_factory([bad, good])

    result = harness.extract(CASE1_STANDARD, transcript_id="TX-001")

    assert result.status == "REPAIRED"
    assert result.attempts == 2
    # The feedback must mention the invalid fields.
    second_prompt = fake.calls[1]["user"]
    assert "issue_category" in second_prompt or "urgency" in second_prompt
