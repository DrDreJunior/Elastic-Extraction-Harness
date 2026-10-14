"""Runtime configuration.

Tunables are read from the environment at process start.  Invalid values fail
fast so a misconfigured deploy does not silently run with unsafe defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_str(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if raw is None:
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # LLM
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_api_key: str | None = field(default_factory=lambda: _env_str("OPENAI_API_KEY"))
    llm_base_url: str | None = field(default_factory=lambda: _env_str("OPENAI_BASE_URL"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 30.0))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 1024))
    # Some OpenAI-compatible servers reject response_format. Default on.
    llm_json_mode: bool = field(default_factory=lambda: _env_bool("LLM_JSON_MODE", True))

    # Harness
    max_repair_attempts: int = field(
        default_factory=lambda: _env_int("MAX_REPAIR_ATTEMPTS", 2)
    )
    # Transport retries *inside* one semantic attempt. Kept separate so a
    # blip does not consume the whole repair budget.
    max_transport_retries: int = field(
        default_factory=lambda: _env_int("MAX_TRANSPORT_RETRIES", 1)
    )
    max_transcript_chars: int = field(
        default_factory=lambda: _env_int("MAX_TRANSCRIPT_CHARS", 20_000)
    )

    # Safety
    block_on_injection_suspected: bool = field(
        default_factory=lambda: _env_bool("BLOCK_ON_INJECTION", True)
    )
    # Raw model text is for internal replay only. The HTTP response strips it
    # unless this is explicitly enabled.
    expose_raw_outputs: bool = field(
        default_factory=lambda: _env_bool("EXPOSE_RAW_LLM_OUTPUTS", False)
    )

    # Service
    service_name: str = "elastic-extraction-harness"
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        if not 0 <= self.max_repair_attempts <= 5:
            raise ValueError("MAX_REPAIR_ATTEMPTS must be between 0 and 5")
        if not 0 <= self.max_transport_retries <= 5:
            raise ValueError("MAX_TRANSPORT_RETRIES must be between 0 and 5")
        if self.max_transcript_chars < 1 or self.max_transcript_chars > 200_000:
            raise ValueError("MAX_TRANSCRIPT_CHARS must be between 1 and 200000")
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2")
        if self.llm_timeout_s <= 0 or self.llm_timeout_s > 120:
            raise ValueError("LLM_TIMEOUT_S must be between 0 and 120")
        if not 64 <= self.llm_max_tokens <= 4096:
            raise ValueError("LLM_MAX_TOKENS must be between 64 and 4096")


def get_settings() -> Settings:
    return Settings()
