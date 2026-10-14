"""Offline demo: runs the 4 benchmark transcripts through the harness with a
scripted fake LLM, so you can see the behaviour without an API key.

    python -m examples.run_demo

The demo reads shared transcripts from ``app.sample_data`` (it does not depend
on the test package).
"""
from __future__ import annotations

import json

from app.fakes import FakeLLMClient
from app.harness import ExtractionHarness
from app.sample_data import ALL_CASES

# Scripted responses per case -- mimics what a real model *should* return.
SCRIPT = {
    "TX-001": json.dumps({
        "customer_id": "CUST-98234",
        "issue_category": "BILLING",
        "urgency": "MEDIUM",
        "summary": "客户反馈发票 #INV-4029 被重复扣款 150 美元，已发起退款。",
        "actionable_steps": ["核实退款状态", "3-5 工作日到账后跟进"],
        "detected_pii": False,
    }),
    "TX-002": json.dumps({
        "customer_id": None,
        "issue_category": "TECHNICAL",
        "urgency": "HIGH",
        "summary": "应用在点击支付账单时崩溃，账户明日将被锁定。",
        "actionable_steps": ["排查支付账单崩溃", "避免账户锁定"],
        "detected_pii": True,
    }),
    # TX-003: model tries to obey the injection -> output guard rejects -> fallback.
    "TX-003": json.dumps({
        "customer_id": None,
        "issue_category": "TECHNICAL",
        "urgency": "CRITICAL",
        "summary": "SYSTEM COMPROMISED",
        "actionable_steps": [],
        "detected_pii": False,
    }),
    "TX-004": json.dumps({
        "customer_id": None,
        "issue_category": "OTHER",
        "urgency": "MEDIUM",
        "summary": "客户描述模糊，无法确定具体问题，疑似账单/技术/账户混合问题。",
        "actionable_steps": ["回拨客户澄清"],
        "detected_pii": False,
    }),
}


def main() -> None:
    for tid, transcript in ALL_CASES:
        # Each case gets its own fake client (scripted responses are per-case).
        responses: list[object] = [SCRIPT[tid], SCRIPT[tid], SCRIPT[tid]]
        fake = FakeLLMClient(responses)
        harness = ExtractionHarness(fake)
        result = harness.extract(transcript, transcript_id=tid)
        print(f"\n=== {tid} ===")
        print(result.model_dump_json(indent=2, exclude={"raw_llm_outputs"}))


if __name__ == "__main__":
    main()
