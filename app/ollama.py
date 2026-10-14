"""Ollama deployment helpers.

Ollama exposes an OpenAI-compatible API at ``<host>/v1``. These factory
functions build a client/harness pointed at a local or LAN Ollama instance with
sensible defaults, so the whole stack can run with no external provider.

Override the built-in target with environment variables:

    OLLAMA_HOST=http://192.168.2.7:11434
    OLLAMA_MODEL=deepseek-coder-v2:16b

Ollama does not authenticate, but the OpenAI SDK wants a non-empty key, so we
send the placeholder ``"ollama"``. The default model is verified end-to-end on a
local Ollama; qwen3:8b works too once pulled.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse

from .config import Settings
from .harness import AsyncExtractionHarness, ExtractionHarness
from .llm_client import AsyncOpenAICompatibleClient, OpenAICompatibleClient

DEFAULT_HOST = "http://192.168.2.7:11434"
DEFAULT_PORT = 11434
DEFAULT_MODEL = "deepseek-coder-v2:16b"

# First call after an idle period can include a model load, so allow headroom.
DEFAULT_TIMEOUT_S = 120.0
_PLACEHOLDER_KEY = "ollama"


def _base_url(host: str) -> str:
    raw = host if host.startswith("http") else f"http://{host}"
    parsed = urlparse(raw.rstrip("/"))
    name = parsed.hostname or ""
    # 0.0.0.0 is a bind spec (Ollama's OLLAMA_HOST), not a connectable address.
    if name == "0.0.0.0":
        name = "127.0.0.1"
    port = parsed.port or DEFAULT_PORT
    path = parsed.path or ""
    if not path.endswith("/v1"):
        path = path.rstrip("/") + "/v1"
    return urlunparse((parsed.scheme, f"{name}:{port}", path, "", "", ""))


def _host() -> str:
    return os.getenv("OLLAMA_HOST", DEFAULT_HOST)


def _model() -> str:
    return os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)


def ollama_client(
    *,
    host: str | None = None,
    model: str | None = None,
    json_mode: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        model=model or _model(),
        api_key=_PLACEHOLDER_KEY,
        base_url=_base_url(host or _host()),
        timeout_s=timeout_s,
        json_mode=json_mode,
    )


def ollama_async_client(
    *,
    host: str | None = None,
    model: str | None = None,
    json_mode: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> AsyncOpenAICompatibleClient:
    return AsyncOpenAICompatibleClient(
        model=model or _model(),
        api_key=_PLACEHOLDER_KEY,
        base_url=_base_url(host or _host()),
        timeout_s=timeout_s,
        json_mode=json_mode,
    )


def ollama_harness(settings: Settings | None = None, **kwargs) -> ExtractionHarness:
    return ExtractionHarness(ollama_client(**kwargs), settings)


def ollama_async_harness(settings: Settings | None = None, **kwargs) -> AsyncExtractionHarness:
    return AsyncExtractionHarness(ollama_async_client(**kwargs), settings)
