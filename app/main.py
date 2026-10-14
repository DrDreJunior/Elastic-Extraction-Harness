"""FastAPI entrypoint (async serving path).

    uvicorn app.main:app --port 8000

The request handler awaits an async harness, so waiting on the model does not
block a worker thread. A synchronous harness (`app.harness.ExtractionHarness`)
is still available for direct library callers and the offline demo.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .config import get_settings
from .harness import AsyncExtractionHarness
from .llm_client import AsyncOpenAICompatibleClient
from .logging_config import configure_logging, get_logger
from .schemas import ExtractionResult, ExtractionStatus

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger("api")

app = FastAPI(title=settings.service_name, version="1.0.0")

_async_harness: AsyncExtractionHarness | None = None
_harness_lock = asyncio.Lock()


async def get_harness() -> AsyncExtractionHarness:
    global _async_harness
    if _async_harness is not None:
        return _async_harness
    async with _harness_lock:
        if _async_harness is None:
            client = AsyncOpenAICompatibleClient(
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                temperature=settings.llm_temperature,
                timeout_s=settings.llm_timeout_s,
                max_tokens=settings.llm_max_tokens,
                json_mode=settings.llm_json_mode,
            )
            _async_harness = AsyncExtractionHarness(client, settings)
        return _async_harness


class ExtractRequest(BaseModel):
    transcript: str = Field(..., description="Raw support conversation transcript.")
    transcript_id: str | None = Field(default=None, max_length=128)

    @field_validator("transcript")
    @classmethod
    def _transcript_bounds(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("transcript must not be blank")
        limit = settings.max_transcript_chars
        if len(value) > limit:
            raise ValueError(f"transcript exceeds {limit} characters")
        return value

    @field_validator("transcript_id")
    @classmethod
    def _id_chars(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if any(ch in value for ch in ("\r", "\n", "\x00")):
            raise ValueError("transcript_id contains illegal characters")
        return value.strip()


@app.get("/health")
def health() -> dict:
    """Liveness: the process is up. Cheap, no external dependency."""
    return {"status": "ok", "service": settings.service_name}


# --- Readiness: cached provider probe -------------------------------------

_PROBE_TTL_S = 30.0
_probe_lock = asyncio.Lock()
_probe_state: dict[str, Any] = {"checked_at": 0.0, "ok": False}
_probe_client: AsyncOpenAICompatibleClient | None = None


def _get_probe_client() -> AsyncOpenAICompatibleClient:
    """A dedicated, cheap client for the readiness probe (1 token, 5s timeout)."""
    global _probe_client
    if _probe_client is None:
        _probe_client = AsyncOpenAICompatibleClient(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout_s=5.0,
            max_tokens=1,
            json_mode=False,
        )
    return _probe_client


async def _probe_provider() -> bool:
    """Return a cached reachability result so frequent LB probes do not hammer the model."""
    now = time.monotonic()
    if now - _probe_state["checked_at"] < _PROBE_TTL_S:
        return _probe_state["ok"]
    async with _probe_lock:
        now = time.monotonic()
        if now - _probe_state["checked_at"] < _PROBE_TTL_S:
            return _probe_state["ok"]
        try:
            await _get_probe_client().chat(
                "You are a connectivity check.", "Reply with the single character: ok"
            )
            ok = True
        except Exception as exc:  # noqa: BLE001 - readiness must not raise
            ok = False
            log.warning("readiness_probe_failed", extra={"error": type(exc).__name__})
        _probe_state["checked_at"] = now
        _probe_state["ok"] = ok
        return ok


@app.get("/ready")
async def ready() -> JSONResponse:
    """Readiness: config present and the provider reachable (cached)."""
    if not settings.llm_api_key:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "missing_api_key"},
        )
    if not await _probe_provider():
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "provider_unreachable"},
        )
    return JSONResponse(status_code=200, content={"status": "ready", "model": settings.llm_model})


@app.post("/extract", response_model=ExtractionResult)
async def extract(req: ExtractRequest, request: Request, response: Response) -> ExtractionResult:
    incoming = (request.headers.get("x-request-id") or "").strip()
    if any(ch in incoming for ch in ("\r", "\n", "\x00")):
        incoming = ""
    rid = incoming[:128] or str(uuid.uuid4())
    response.headers["x-request-id"] = rid
    try:
        harness = await get_harness()
        result = await harness.extract(req.transcript, req.transcript_id, request_id=rid)
    except Exception:  # noqa: BLE001 - last line of defence; harness already catches
        log.exception("extract_unhandled", extra={"transcript_id": req.transcript_id})
        result = ExtractionResult(
            status=ExtractionStatus.FALLBACK,
            transcript_id=req.transcript_id,
            request_id=rid,
            flagged_for_review=True,
            review_reason="unhandled_error",
        )
    if not settings.expose_raw_outputs:
        result = result.model_copy(update={"raw_llm_outputs": []})
    return result
