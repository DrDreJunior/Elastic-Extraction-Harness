"""Scriptable LLM stub for tests and the offline demo.

Not used by the serving path.  Lives next to the harness so `examples/` does
not have to import the test package.
"""
from __future__ import annotations

from .llm_client import LLMError, LLMResponse


class FakeLLMClient:
    """Ordered stub. A str is returned as model text; an exception is raised."""

    def __init__(self, responses: list[object], model_name: str = "fake-llm") -> None:
        self._responses = list(responses)
        self.model_name = model_name
        self.calls: list[dict] = []

    def chat(self, system: str, user: str) -> LLMResponse:
        self.calls.append({"system": system, "user": user})
        if not self._responses:
            raise AssertionError("FakeLLMClient called more times than expected")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if not isinstance(nxt, str):
            raise LLMError(f"unsupported_fake_response:{type(nxt).__name__}")
        return LLMResponse(text=nxt, model=self.model_name, latency_ms=1.5)


class AsyncFakeLLMClient:
    """Async ordered stub, mirroring FakeLLMClient for the async harness."""

    def __init__(self, responses: list[object], model_name: str = "fake-async-llm") -> None:
        self._responses = list(responses)
        self.model_name = model_name
        self.calls: list[dict] = []

    async def chat(self, system: str, user: str) -> LLMResponse:
        self.calls.append({"system": system, "user": user})
        if not self._responses:
            raise AssertionError("AsyncFakeLLMClient called more times than expected")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if not isinstance(nxt, str):
            raise LLMError(f"unsupported_fake_response:{type(nxt).__name__}")
        return LLMResponse(text=nxt, model=self.model_name, latency_ms=1.5)
