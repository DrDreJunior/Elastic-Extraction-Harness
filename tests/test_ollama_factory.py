"""Offline unit tests for the Ollama factory helpers."""
from __future__ import annotations

from app.ollama import _base_url, ollama_async_client, ollama_client


def test_base_url_appends_scheme_and_v1():
    assert _base_url("192.168.2.7:11434") == "http://192.168.2.7:11434/v1"
    assert _base_url("http://192.168.2.7:11434") == "http://192.168.2.7:11434/v1"
    assert _base_url("http://192.168.2.7:11434/") == "http://192.168.2.7:11434/v1"


def test_base_url_keeps_existing_v1():
    assert _base_url("http://192.168.2.7:11434/v1") == "http://192.168.2.7:11434/v1"


def test_base_url_rewrites_bind_spec_and_defaults_port():
    assert _base_url("0.0.0.0") == "http://127.0.0.1:11434/v1"
    assert _base_url("http://0.0.0.0:11434") == "http://127.0.0.1:11434/v1"


def test_clients_target_correct_base_and_placeholder_key():
    sync = ollama_client(host="192.168.2.7:11434", model="qwen3:8b")
    assert sync.model_name == "qwen3:8b"
    assert "192.168.2.7:11434/v1" in str(sync._client.base_url)
    assert sync._client.api_key == "ollama"

    async_ = ollama_async_client(host="http://192.168.2.7:11434", model="qwen3:8b")
    assert async_.model_name == "qwen3:8b"
    assert "192.168.2.7:11434/v1" in str(async_._client.base_url)
    assert async_._client.api_key == "ollama"
