"""Schema-level tests: validation, enums, summary truncation."""
from __future__ import annotations

import pytest
from app.schemas import ExtractedTicket, IssueCategory, Urgency
from pydantic import ValidationError


def test_valid_ticket_passes():
    t = ExtractedTicket(
        customer_id="CUST-1",
        issue_category="BILLING",
        urgency="HIGH",
        summary="Duplicate charge on invoice #1001",
        actionable_steps=["Issue refund", "Email customer"],
        detected_pii=False,
    )
    assert t.issue_category == IssueCategory.BILLING
    assert t.urgency == Urgency.HIGH


def test_invalid_category_rejected():
    with pytest.raises(ValidationError):
        ExtractedTicket(
            issue_category="REFUNDS",  # not in enum
            urgency="LOW",
            summary="x",
        )


def test_invalid_urgency_rejected():
    with pytest.raises(ValidationError):
        ExtractedTicket(
            issue_category="OTHER",
            urgency="URGENT",  # not in enum
            summary="x",
        )


def test_summary_truncated_to_200():
    long = "a" * 250
    t = ExtractedTicket(issue_category="OTHER", urgency="LOW", summary=long)
    assert len(t.summary) <= 200


def test_blank_actionable_steps_dropped():
    t = ExtractedTicket(
        issue_category="OTHER",
        urgency="LOW",
        summary="ok",
        actionable_steps=["  ", "do thing", "", "  fix ", "  "],
    )
    assert t.actionable_steps == ["do thing", "fix"]


def test_customer_id_optional():
    t = ExtractedTicket(issue_category="OTHER", urgency="LOW", summary="x")
    assert t.customer_id is None
