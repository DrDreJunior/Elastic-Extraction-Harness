"""LLM client abstraction.

The harness depends on `LLMClient`, not a vendor SDK, so tests inject a fake
and a new provider is one adapter.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_ms: float
    raw: dict | None = None


class LLMError(Exception):
    """Transport or provider failure. `retryable` controls both inner and outer loops."""

    retryable = True


class LLMRateLimit(LLMError):
    """429 / temporary overload. Safe to retry."""


class LLMServerError(LLMError):
    """5xx, timeout, connection reset, empty completion. Safe to retry."""


class LLMAuthError(LLMError):
    """401/403. Retrying burns budget and will not succeed."""

    retryable = False


class LLMBadRequest(LLMError):
    """4xx other than 429. The same prompt will fail again."""

    retryable = False


@runtime_checkable
class LLMClient(Protocol):
    model_name: str

    def chat(self, system: str, user: str) -> LLMResponse:  # pragma: no cover - protocol
        ...


@runtime_checkable
class AsyncLLMClient(Protocol):
    model_name: str

    async def chat(self, system: str, user: str) -> LLMResponse:  # pragma: no cover - protocol
        ...


class OpenAICompatibleClient:
    """Adapter for any OpenAI-compatible /chat/completions endpoint.

    The SDK is imported lazily so the offline test suite does not need it.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
        max_tokens: int = 1024,
        json_mode: bool = True,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
        self.model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._json_mode = json_mode

    def chat(self, system: str, user: str) -> LLMResponse:
        from openai import (
            APIConnectionError,
            APIError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            PermissionDeniedError,
            RateLimitError,
        )

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            raise LLMRateLimit(_short(exc)) from exc
        except (APITimeoutError, APIConnectionError) as exc:
            raise LLMServerError(_short(exc)) from exc
        except (AuthenticationError, PermissionDeniedError) as exc:
            raise LLMAuthError(_short(exc)) from exc
        except BadRequestError as exc:
            raise LLMBadRequest(_short(exc)) from exc
        except APIError as exc:
            status = getattr(exc, "status_code", None) or 0
            if 400 <= status < 500 and status != 429:
                raise LLMBadRequest(_short(exc)) from exc
            raise LLMServerError(_short(exc)) from exc
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK surface is not closed
            raise LLMServerError(f"unexpected_sdk_error:{type(exc).__name__}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = _message_text(resp)
        if text is None:
            raise LLMServerError("empty_completion")
        return LLMResponse(text=text, model=self.model_name, latency_ms=latency_ms)


def _short(exc: Exception) -> str:
    status = getattr(exc, "status_code", "")
    return f"{type(exc).__name__}:{status}"


def _message_text(resp: object) -> str | None:
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    if message is None:
        return None
    content = getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(getattr(part, "text", "") or ""))
        return "".join(parts)
    return str(content)


class AsyncOpenAICompatibleClient:
    """Async adapter for any OpenAI-compatible /chat/completions endpoint.

    This is the production path: awaiting the provider does not block a worker
    thread, so a single event loop can hold many in-flight requests. The SDK is
    imported lazily so the offline test suite does not need it.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
        max_tokens: int = 1024,
        json_mode: bool = True,
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
        self.model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._json_mode = json_mode

    async def chat(self, system: str, user: str) -> LLMResponse:
        from openai import (
            APIConnectionError,
            APIError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            PermissionDeniedError,
            RateLimitError,
        )

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            raise LLMRateLimit(_short(exc)) from exc
        except (APITimeoutError, APIConnectionError) as exc:
            raise LLMServerError(_short(exc)) from exc
        except (AuthenticationError, PermissionDeniedError) as exc:
            raise LLMAuthError(_short(exc)) from exc
        except BadRequestError as exc:
            raise LLMBadRequest(_short(exc)) from exc
        except APIError as exc:
            status = getattr(exc, "status_code", None) or 0
            if 400 <= status < 500 and status != 429:
                raise LLMBadRequest(_short(exc)) from exc
            raise LLMServerError(_short(exc)) from exc
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - SDK surface is not closed
            raise LLMServerError(f"unexpected_sdk_error:{type(exc).__name__}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = _message_text(resp)
        if text is None:
            raise LLMServerError("empty_completion")
        return LLMResponse(text=text, model=self.model_name, latency_ms=latency_ms)
