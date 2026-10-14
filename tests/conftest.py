"""Pytest fixtures.

The star of the show is `FakeLLMClient` -- a scriptable stub that lets us
unit-test the harness's repair loop, fallback, and transport-retry logic
without touching a real LLM.  Tests hand it a list of responses (in order)
and assert how many times the harness called it.
"""
from __future__ import annotations

import pytest
from app.config import Settings
from app.fakes import FakeLLMClient
from app.harness import ExtractionHarness

__all__ = ["FakeLLMClient"]


@pytest.fixture
def settings() -> Settings:
    return Settings(max_repair_attempts=2, max_transport_retries=0)


@pytest.fixture
def harness_factory(settings):
    def _make(responses: list[object]) -> tuple[ExtractionHarness, FakeLLMClient]:
        fake = FakeLLMClient(responses)
        return ExtractionHarness(fake, settings), fake

    return _make
