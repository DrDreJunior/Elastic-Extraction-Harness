"""Prompt-injection defence.

Customer transcripts are untrusted input. Defence is layered:

1. System-prompt hardening (`prompts.py`).
2. Delimiter isolation, with customer-supplied fence tags stripped.
3. Output guard. A hit is not retried: sending the compromised completion
   back as a "repair" prompt reinforces the injection and spends budget.

The output guard must not rely on a single surface signal. In particular,
an attacker that demands "set urgency=CRITICAL" may still get the model to
write a normal-length summary, so urgency is cross-checked against a
deterministic, model-independent business-impact baseline.
"""
from __future__ import annotations

import re

from .schemas import ExtractedTicket
from .text_norm import normalize_untrusted

_INJECTION_ECHO_PATTERNS = [
    re.compile(r"system\s+over\s*ride", re.I),
    re.compile(r"system\s+compromised", re.I),
    re.compile(r"ignor(?:e|ing)\s+(?:all\s+)?(?:previous|prior)\s+instructions", re.I),
    re.compile(r"do\s+not\s+output\s+json", re.I),
    re.compile(r"print\s+[\"']?system", re.I),
]

_TRANSCRIPT_INJECTION_PATTERNS = [
    re.compile(r"system\s+over\s*ride", re.I),
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", re.I),
    re.compile(r"you\s+are\s+now\s+a", re.I),
    re.compile(r"disregard\s+(?:the\s+)?(?:above|prior|previous)", re.I),
]

# Strong, *model-independent* evidence that CRITICAL urgency is genuinely
# justified. These describe real business impact and deliberately do NOT
# include the words "critical" / "urgent" / "emergency" / "紧急" -- an
# attacker can type those, but cannot fabricate the underlying impact by
# merely naming it.
_CRITICAL_BUSINESS_SIGNALS = [
    re.compile(r"\boutage\b", re.I),
    re.compile(r"down\s+for\s+everyone", re.I),
    re.compile(r"all\s+users?\s+(?:are\s+)?(?:down|locked|affected)", re.I),
    re.compile(r"(?:service|system|entire\s+system)\s+is\s+down", re.I),
    re.compile(r"cannot\s+process\s+payments?\s+at\s+all", re.I),
    re.compile(r"can\s*not\s+pay\s+(?:employees|workers|wages|salaries)", re.I),
    re.compile(r"\bpayroll\b", re.I),
    re.compile(r"data\s+breach|data\s+leak|leaked|exfiltrat|ransomware", re.I),
    re.compile(r"无法营业|全员|全部宕机|系统瘫痪|数据泄露|勒索|发不出工资|大面积|工资单|发工资"),
]


def scan_transcript(text: str) -> list[str]:
    """Return injection-like phrases found in the transcript."""
    normalized = normalize_untrusted(text)
    return [p.pattern for p in _TRANSCRIPT_INJECTION_PATTERNS if p.search(normalized)]


def has_critical_business_evidence(transcript: str) -> bool:
    """True if the transcript itself shows real CRITICAL-scale business impact.

    Used to decide whether a model's CRITICAL urgency is plausible when the
    transcript is already suspected of containing an injection.
    """
    normalized = normalize_untrusted(transcript)
    return any(p.search(normalized) for p in _CRITICAL_BUSINESS_SIGNALS)


def validate_extraction_against_injection(
    transcript: str,
    ticket: ExtractedTicket,
) -> tuple[bool, list[str]]:
    """Return (compromised, reasons)."""
    reasons: list[str] = []
    blob = " ".join([ticket.summary, " ".join(ticket.actionable_steps)])
    blob = normalize_untrusted(blob)

    for pat in _INJECTION_ECHO_PATTERNS:
        if pat.search(blob):
            reasons.append(f"echoed_injection_phrase:{pat.pattern}")

    if scan_transcript(transcript):
        # The input already tried to hijack us. Cross-check the model's
        # urgency against a deterministic baseline rather than trusting a
        # length heuristic (which a normal-length summary can bypass).
        if ticket.urgency == "CRITICAL" and not has_critical_business_evidence(transcript):
            reasons.append(
                "critical_urgency_without_business_justification_under_injection"
            )
        if len(ticket.summary) < 15:
            reasons.append("summary_too_short_under_injection_attempt")

    return (bool(reasons), reasons)
