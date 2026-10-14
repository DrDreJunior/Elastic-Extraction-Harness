"""Pydantic schemas for the extraction domain.

All LLM output must validate against `ExtractedTicket` before it is considered
a successful extraction.  Anything that fails validation is routed through the
repair loop in `harness.py`.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from .pii import detect_and_redact


class IssueCategory(str, Enum):
    BILLING = "BILLING"
    TECHNICAL = "TECHNICAL"
    ACCOUNT = "ACCOUNT"
    OTHER = "OTHER"


class Urgency(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ExtractedTicket(BaseModel):
    """Strongly-typed extraction result returned by the LLM.

    Field naming follows the challenge spec verbatim so that downstream
    consumers (CRM, ticketing system) get a stable contract.
    """

    customer_id: str | None = Field(
        default=None,
        description="Customer identifier if mentioned, else null.",
    )
    issue_category: IssueCategory = Field(
        description="One of BILLING / TECHNICAL / ACCOUNT / OTHER.",
    )
    urgency: Urgency = Field(description="One of LOW / MEDIUM / HIGH / CRITICAL.")
    summary: str = Field(
        description="Brief summary of the core issue, at most 200 characters.",
    )
    actionable_steps: list[str] = Field(
        default_factory=list,
        description="Concrete follow-up steps or resolution items.",
    )
    detected_pii: bool = Field(
        default=False,
        description="True if PII (credit card, national ID, password) appears in transcript.",
    )

    @field_validator("customer_id", mode="before")
    @classmethod
    def _blank_customer_id(cls, v: object) -> object:
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    @field_validator("issue_category", "urgency", mode="before")
    @classmethod
    def _coerce_enum(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().upper()
        return v

    @field_validator("summary")
    @classmethod
    def _summary_length(cls, v: str) -> str:
        # Hard cap at 200 chars; truncate *after* stripping so the model can
        # emit a slightly over-long summary without blowing up the pipeline.
        v = v.strip()
        if len(v) > 200:
            v = v[:197].rstrip() + "..."
        return v

    @field_validator("actionable_steps")
    @classmethod
    def _steps_non_empty(cls, v: list[str]) -> list[str]:
        # Drop blank steps and cap size so a runaway completion cannot bloat storage.
        cleaned: list[str] = []
        for step in v:
            if not step or not step.strip():
                continue
            text = step.strip()
            if len(text) > 500:
                text = text[:497].rstrip() + "..."
            cleaned.append(text)
            if len(cleaned) >= 20:
                break
        return cleaned

    @model_validator(mode="after")
    def _no_pii_leakage(self) -> ExtractedTicket:
        # The model can echo a secret even when the prompt was redacted.
        # Persist only the redacted form, and force the PII flag on.
        summary_scan = detect_and_redact(self.summary)
        self.summary = summary_scan.redacted_text
        leaked = summary_scan.has_pii
        cleaned_steps: list[str] = []
        for step in self.actionable_steps:
            scan = detect_and_redact(step)
            cleaned_steps.append(scan.redacted_text)
            leaked = leaked or scan.has_pii
        self.actionable_steps = cleaned_steps
        if self.customer_id:
            id_scan = detect_and_redact(self.customer_id)
            if id_scan.has_pii:
                self.customer_id = id_scan.redacted_text
                leaked = True
        if leaked:
            self.detected_pii = True
        if len(self.summary) > 200:
            self.summary = self.summary[:197].rstrip() + "..."
        return self


class ExtractionStatus(str, Enum):
    SUCCESS = "SUCCESS"          # First-attempt clean extraction
    REPAIRED = "REPAIRED"         # Succeeded after >=1 self-correction round
    FALLBACK = "FALLBACK"         # Retry budget exhausted -> human review queue
    REJECTED = "REJECTED"         # Invalid input (blank / oversized), not a model failure


class ExtractionResult(BaseModel):
    """Final envelope returned to the caller.

    Always serialises -- even on FALLBACK -- so the caller never has to catch
    an exception from the harness.
    """

    status: ExtractionStatus
    ticket: ExtractedTicket | None = None
    transcript_id: str | None = None
    request_id: str | None = None
    attempts: int = 0
    validation_errors: list[str] = Field(default_factory=list)
    raw_llm_outputs: list[str] = Field(default_factory=list)
    flagged_for_review: bool = False
    review_reason: str | None = None
    latency_ms: float = 0.0
