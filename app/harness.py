"""The extraction harness -- the heart of the system.

Responsibilities, in order:

1. Reject inputs that must never reach the model (blank, oversized).
2. Normalise + redact PII before the transcript is logged or sent.
3. Record injection attempts for observability. The system prompt is the
   primary defence; the output guard is the backstop.
4. Call the LLM, parse, and validate against `ExtractedTicket`.
5. Repair loop: on validation failure, feed only the latest error back,
   up to `max_repair_attempts` times.
6. Output guard: a compromised parse is not repaired. The compromised text
   is not sent back to the model. Result is FALLBACK for human review.
7. Any unexpected exception becomes a FALLBACK envelope. This module does
   not raise to callers.

Two thin drivers share the same pure (deterministic) pipeline:

* `ExtractionHarness`      drives a synchronous `LLMClient`.
* `AsyncExtractionHarness` drives an `AsyncLLMClient`; awaiting the model
  does not block a worker thread, which is what makes 10k ops/min feasible.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from . import injection, pii
from .config import Settings
from .llm_client import AsyncLLMClient, LLMClient, LLMError, LLMResponse
from .logging_config import get_logger, reset_request_id, set_request_id
from .prompts import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt
from .schemas import ExtractedTicket, ExtractionResult, ExtractionStatus
from .text_norm import normalize_untrusted

log = get_logger("harness")

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_CONTROL_CHARS = re.compile(r"[\r\n\x00]")
_MAX_STORED_OUTPUT = 8_000


@dataclass
class _Prepared:
    """Result of the deterministic pre-processing of one transcript."""

    normalized: str
    clean_transcript: str
    has_pii: bool
    inject_hits: list[str]
    user_prompt: str


@dataclass
class _StepDecision:
    """Deterministic decision about one model response."""

    outcome: str  # "ok" | "retry" | "block"
    raw: str = ""
    error: str | None = None
    ticket: ExtractedTicket | None = None


def _prepare_input(transcript: object, settings: Settings) -> tuple[_Prepared | None, str | None]:
    """Return (prepared, None) or (None, reject_reason). Never calls the model."""
    if not isinstance(transcript, str) or not transcript.strip():
        return None, "blank_transcript"

    normalized = normalize_untrusted(transcript)
    if len(normalized) > settings.max_transcript_chars:
        return None, "transcript_exceeds_max_length"

    pii_scan = pii.detect_and_redact(normalized)
    inject_hits = injection.scan_transcript(normalized)
    return (
        _Prepared(
            normalized=normalized,
            clean_transcript=pii_scan.redacted_text,
            has_pii=pii_scan.has_pii,
            inject_hits=inject_hits,
            user_prompt=build_user_prompt(pii_scan.redacted_text),
        ),
        None,
    )


def _log_prepared(prepared: _Prepared, transcript_id: str | None, model_name: str) -> None:
    if prepared.inject_hits:
        log.warning(
            "transcript_injection_attempt",
            extra={
                "transcript_id": transcript_id,
                "injection_suspected": True,
                "injection_patterns": prepared.inject_hits,
            },
        )
    log.info(
        "extraction_started",
        extra={
            "transcript_id": transcript_id,
            "model": model_name,
            "injection_suspected": bool(prepared.inject_hits),
        },
    )


def _prompt_for(
    attempt: int,
    prepared: _Prepared,
    last_raw: str,
    validation_errors: list[str],
) -> str:
    if attempt == 1:
        return prepared.user_prompt
    return build_repair_prompt(prepared.clean_transcript, last_raw, validation_errors[-1:])


def _handle_response(resp: LLMResponse, prepared: _Prepared) -> _StepDecision:
    """Parse, validate, and run the output guard. Pure apart from logging."""
    safe_text = _cap(pii.detect_and_redact(resp.text).redacted_text)
    log.info(
        "llm_response_received",
        extra={
            "latency_ms": resp.latency_ms,
            "model": resp.model,
        },
    )

    parsed = _parse_json(resp.text)
    if parsed is None:
        return _StepDecision("retry", raw=safe_text, error="response_is_not_valid_json")

    try:
        ticket = ExtractedTicket.model_validate(parsed)
    except Exception as exc:  # pydantic ValidationError
        return _StepDecision("retry", raw=safe_text, error=_format_validation_error(exc))

    compromised, reasons = injection.validate_extraction_against_injection(
        prepared.normalized, ticket
    )
    if compromised:
        log.warning(
            "output_injection_compromised",
            extra={"error": ";".join(reasons), "injection_suspected": True},
        )
        return _StepDecision(
            "block", raw=safe_text, error="injection_guard:" + ";".join(reasons)
        )
    return _StepDecision("ok", raw=safe_text, ticket=ticket)


class _BaseHarness:
    """Shared envelope + finalisation for the sync and async drivers."""

    def __init__(self, llm: object, settings: Settings | None = None) -> None:
        self._llm = llm
        self._settings = settings or Settings()

    def _envelope(
        self,
        *,
        status: ExtractionStatus,
        transcript_id: str | None,
        request_id: str,
        t0: float,
        attempts: int = 0,
        validation_errors: list[str] | None = None,
        raw_outputs: list[str] | None = None,
        ticket: ExtractedTicket | None = None,
        flagged_for_review: bool = False,
        review_reason: str | None = None,
    ) -> ExtractionResult:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = ExtractionResult(
            status=status,
            ticket=ticket,
            transcript_id=transcript_id,
            request_id=request_id,
            attempts=attempts,
            validation_errors=validation_errors or [],
            raw_llm_outputs=raw_outputs or [],
            latency_ms=latency_ms,
            flagged_for_review=flagged_for_review,
            review_reason=review_reason,
        )
        log.info(
            "extraction_finished",
            extra={
                "transcript_id": transcript_id,
                "status": result.status.value,
                "attempts": attempts,
                "latency_ms": round(latency_ms, 2),
                "flagged_for_review": result.flagged_for_review,
                "review_reason": review_reason,
                "validation_errors": (validation_errors or [])[-5:],
            },
        )
        return result

    def _finalize(
        self,
        *,
        ticket: ExtractedTicket | None,
        attempts: int,
        blocked: bool,
        validation_errors: list[str],
        raw_outputs: list[str],
        prepared: _Prepared,
        transcript_id: str | None,
        request_id: str,
        t0: float,
    ) -> ExtractionResult:
        if ticket is not None and prepared.has_pii:
            ticket.detected_pii = True

        if ticket is not None:
            status = ExtractionStatus.SUCCESS if attempts == 1 else ExtractionStatus.REPAIRED
            return self._envelope(
                status=status,
                transcript_id=transcript_id,
                request_id=request_id,
                t0=t0,
                attempts=attempts,
                validation_errors=validation_errors,
                raw_outputs=raw_outputs,
                ticket=ticket,
            )

        if blocked:
            reason = "Output rejected by injection guard; not retried. Errors: " + " | ".join(
                validation_errors[-5:]
            )
        else:
            reason = (
                "Repair budget exhausted after "
                f"{attempts} attempts. Errors: " + " | ".join(validation_errors[-5:])
            )
        return self._envelope(
            status=ExtractionStatus.FALLBACK,
            transcript_id=transcript_id,
            request_id=request_id,
            t0=t0,
            attempts=attempts,
            validation_errors=validation_errors,
            raw_outputs=raw_outputs,
            flagged_for_review=True,
            review_reason=reason,
        )


class ExtractionHarness(_BaseHarness):
    """Synchronous driver. `extract()` never raises to the caller."""

    _llm: LLMClient

    def __init__(self, llm: LLMClient, settings: Settings | None = None) -> None:
        super().__init__(llm, settings)

    def extract(
        self,
        transcript: str,
        transcript_id: str | None = None,
        request_id: str | None = None,
    ) -> ExtractionResult:
        rid = _safe_id(request_id) or str(uuid.uuid4())
        token = set_request_id(rid)
        t0 = time.perf_counter()
        try:
            return self._extract(_safe_id(transcript_id), transcript, rid, t0)
        except Exception as exc:  # noqa: BLE001 - harness contract is never-raise
            log.exception("extraction_unhandled", extra={"error": type(exc).__name__})
            return self._envelope(
                status=ExtractionStatus.FALLBACK,
                transcript_id=_safe_id(transcript_id),
                request_id=rid,
                t0=t0,
                validation_errors=[f"unhandled:{type(exc).__name__}"],
                flagged_for_review=True,
                review_reason=f"unhandled_error:{type(exc).__name__}",
            )
        finally:
            reset_request_id(token)

    def _extract(
        self,
        transcript_id: str | None,
        transcript: object,
        request_id: str,
        t0: float,
    ) -> ExtractionResult:
        prepared, reject_reason = _prepare_input(transcript, self._settings)
        if prepared is None:
            return self._envelope(
                status=ExtractionStatus.REJECTED,
                transcript_id=transcript_id,
                request_id=request_id,
                t0=t0,
                review_reason=reject_reason,
                validation_errors=[reject_reason] if reject_reason else [],
            )

        _log_prepared(prepared, transcript_id, self._llm.model_name)

        last_raw = ""
        validation_errors: list[str] = []
        raw_outputs: list[str] = []
        ticket: ExtractedTicket | None = None
        attempts = 0
        blocked = False

        for attempt in range(1, self._settings.max_repair_attempts + 2):
            attempts = attempt
            prompt = _prompt_for(attempt, prepared, last_raw, validation_errors)
            try:
                resp = self._call_with_transport_retry(prompt)
            except LLMError as exc:
                validation_errors.append(f"transport_error:{type(exc).__name__}")
                log.error(
                    "llm_transport_error",
                    extra={"attempts": attempt, "error": type(exc).__name__},
                )
                if not getattr(exc, "retryable", True):
                    break
                continue
            except Exception as exc:  # noqa: BLE001
                validation_errors.append(f"unexpected_error:{type(exc).__name__}")
                log.exception("llm_unexpected_error", extra={"attempts": attempt})
                break

            decision = _handle_response(resp, prepared)
            raw_outputs.append(decision.raw)
            last_raw = decision.raw
            if decision.outcome == "ok" and decision.ticket is not None:
                ticket = decision.ticket
                break
            if decision.error:
                validation_errors.append(decision.error)
            if decision.outcome == "block":
                blocked = True
                break

        return self._finalize(
            ticket=ticket,
            attempts=attempts,
            blocked=blocked,
            validation_errors=validation_errors,
            raw_outputs=raw_outputs,
            prepared=prepared,
            transcript_id=transcript_id,
            request_id=request_id,
            t0=t0,
        )

    def _call_with_transport_retry(self, user_prompt: str) -> LLMResponse:
        last_exc: LLMError | None = None
        attempts = self._settings.max_transport_retries + 1
        for i in range(attempts):
            try:
                return self._llm.chat(SYSTEM_PROMPT, user_prompt)
            except LLMError as exc:
                last_exc = exc
                if not getattr(exc, "retryable", True):
                    raise
                if i < attempts - 1:
                    time.sleep(min(0.5 * (i + 1), 2.0))
        assert last_exc is not None
        raise last_exc


class AsyncExtractionHarness(_BaseHarness):
    """Asynchronous driver. `extract()` never raises to the caller."""

    _llm: AsyncLLMClient

    def __init__(self, llm: AsyncLLMClient, settings: Settings | None = None) -> None:
        super().__init__(llm, settings)

    async def extract(
        self,
        transcript: str,
        transcript_id: str | None = None,
        request_id: str | None = None,
    ) -> ExtractionResult:
        rid = _safe_id(request_id) or str(uuid.uuid4())
        token = set_request_id(rid)
        t0 = time.perf_counter()
        try:
            return await self._extract_async(_safe_id(transcript_id), transcript, rid, t0)
        except Exception as exc:  # noqa: BLE001 - harness contract is never-raise
            log.exception("extraction_unhandled", extra={"error": type(exc).__name__})
            return self._envelope(
                status=ExtractionStatus.FALLBACK,
                transcript_id=_safe_id(transcript_id),
                request_id=rid,
                t0=t0,
                validation_errors=[f"unhandled:{type(exc).__name__}"],
                flagged_for_review=True,
                review_reason=f"unhandled_error:{type(exc).__name__}",
            )
        finally:
            reset_request_id(token)

    async def _extract_async(
        self,
        transcript_id: str | None,
        transcript: object,
        request_id: str,
        t0: float,
    ) -> ExtractionResult:
        prepared, reject_reason = _prepare_input(transcript, self._settings)
        if prepared is None:
            return self._envelope(
                status=ExtractionStatus.REJECTED,
                transcript_id=transcript_id,
                request_id=request_id,
                t0=t0,
                review_reason=reject_reason,
                validation_errors=[reject_reason] if reject_reason else [],
            )

        _log_prepared(prepared, transcript_id, self._llm.model_name)

        last_raw = ""
        validation_errors: list[str] = []
        raw_outputs: list[str] = []
        ticket: ExtractedTicket | None = None
        attempts = 0
        blocked = False

        for attempt in range(1, self._settings.max_repair_attempts + 2):
            attempts = attempt
            prompt = _prompt_for(attempt, prepared, last_raw, validation_errors)
            try:
                resp = await self._call_with_transport_retry_async(prompt)
            except LLMError as exc:
                validation_errors.append(f"transport_error:{type(exc).__name__}")
                log.error(
                    "llm_transport_error",
                    extra={"attempts": attempt, "error": type(exc).__name__},
                )
                if not getattr(exc, "retryable", True):
                    break
                continue
            except Exception as exc:  # noqa: BLE001
                validation_errors.append(f"unexpected_error:{type(exc).__name__}")
                log.exception("llm_unexpected_error", extra={"attempts": attempt})
                break

            decision = _handle_response(resp, prepared)
            raw_outputs.append(decision.raw)
            last_raw = decision.raw
            if decision.outcome == "ok" and decision.ticket is not None:
                ticket = decision.ticket
                break
            if decision.error:
                validation_errors.append(decision.error)
            if decision.outcome == "block":
                blocked = True
                break

        return self._finalize(
            ticket=ticket,
            attempts=attempts,
            blocked=blocked,
            validation_errors=validation_errors,
            raw_outputs=raw_outputs,
            prepared=prepared,
            transcript_id=transcript_id,
            request_id=request_id,
            t0=t0,
        )

    async def _call_with_transport_retry_async(self, user_prompt: str) -> LLMResponse:
        last_exc: LLMError | None = None
        attempts = self._settings.max_transport_retries + 1
        for i in range(attempts):
            try:
                return await self._llm.chat(SYSTEM_PROMPT, user_prompt)
            except LLMError as exc:
                last_exc = exc
                if not getattr(exc, "retryable", True):
                    raise
                if i < attempts - 1:
                    await asyncio.sleep(min(0.5 * (i + 1), 2.0))
        assert last_exc is not None
        raise last_exc


def _safe_id(value: str | None) -> str | None:
    if not value or not isinstance(value, str):
        return None
    cleaned = _CONTROL_CHARS.sub("", value).strip()
    return cleaned[:128] or None


def _cap(text: str) -> str:
    if len(text) <= _MAX_STORED_OUTPUT:
        return text
    return text[:_MAX_STORED_OUTPUT] + "...[truncated]"


def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _format_validation_error(exc: Exception) -> str:
    errs: list[dict[str, Any]] = getattr(exc, "errors", lambda: [])()
    if not errs:
        return f"validation_error:{type(exc).__name__}"
    bits = []
    for err in errs[:3]:
        loc = ".".join(str(x) for x in err.get("loc", []))
        msg = err.get("msg", "invalid")
        bits.append(f"{loc}:{msg}")
    return "validation:" + ";".join(bits)
