"""Happy-path tests: first-attempt extraction succeeds."""
from __future__ import annotations

import json

from app.sample_data import CASE1_STANDARD, CASE2_PII


def test_case1_standard_extraction(harness_factory):
    """用例 1: normal path should yield SUCCESS with a well-formed ticket."""
    model_output = json.dumps({
        "customer_id": "CUST-98234",
        "issue_category": "BILLING",
        "urgency": "MEDIUM",
        "summary": "客户反馈发票 #INV-4029 被重复扣款 150 美元，客服已发起退款。",
        "actionable_steps": ["核实退款状态", "3-5 个工作日到账后跟进"],
        "detected_pii": False,
    })
    harness, fake = harness_factory([model_output])

    result = harness.extract(CASE1_STANDARD, transcript_id="TX-001")

    assert result.status == "SUCCESS"
    assert result.attempts == 1
    assert result.flagged_for_review is False
    assert result.ticket is not None
    assert result.ticket.customer_id == "CUST-98234"
    assert result.ticket.issue_category == "BILLING"
    assert result.ticket.urgency == "MEDIUM"
    assert len(fake.calls) == 1


def test_case2_pii_is_detected_even_if_model_misses(harness_factory):
    """用例 2: transcript contains password + NRIC. Our scanner must flip
    detected_pii to True even if the model forgets."""
    # Model *forgets* to flag PII.
    model_output = json.dumps({
        "customer_id": None,
        "issue_category": "TECHNICAL",
        "urgency": "HIGH",
        "summary": "应用在点击支付账单时崩溃，账户明日将被锁定，用户需跑工资单。",
        "actionable_steps": ["排查支付账单崩溃问题", "避免账户锁定"],
        "detected_pii": False,  # model missed it
    })
    harness, fake = harness_factory([model_output])

    result = harness.extract(CASE2_PII, transcript_id="TX-002")

    assert result.status == "SUCCESS"
    assert result.ticket is not None
    assert result.ticket.detected_pii is True, "PII scanner should override model"


def test_fenced_json_response_is_parsed(harness_factory):
    """Model wraps output in ```json fences; harness must strip them."""
    fenced = "```json\n" + json.dumps({
        "issue_category": "OTHER",
        "urgency": "LOW",
        "summary": "ok",
    }) + "\n```"
    harness, _ = harness_factory([fenced])
    result = harness.extract("hello", transcript_id="T-99")
    assert result.status == "SUCCESS"
    assert result.ticket.issue_category == "OTHER"
