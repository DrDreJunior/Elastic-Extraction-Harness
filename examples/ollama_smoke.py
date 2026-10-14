"""Real end-to-end smoke test against Ollama (no mocks).

    python -m examples.ollama_smoke
    OLLAMA_HOST=http://127.0.0.1:11434 OLLAMA_MODEL=qwen3:8b python -m examples.ollama_smoke

It verifies Ollama is reachable, lists the pulled models, warns if the chosen
model is missing, and then runs all four benchmark transcripts through the
real async harness.
"""
from __future__ import annotations

import asyncio
import sys

import httpx
from app.ollama import _base_url, _host, _model, ollama_async_harness
from app.sample_data import ALL_CASES


async def _list_models(host: str) -> list[str]:
    # Native base, reusing the same normalization (scheme and 0.0.0.0 handling).
    base = _base_url(host).removesuffix("/v1")
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(base + "/api/tags")
        resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


async def main() -> int:
    host = _host()
    model = _model()
    print(f"Ollama endpoint : {_base_url(host)}")
    print(f"Model           : {model}")

    try:
        names = await _list_models(host)
    except Exception as exc:  # noqa: BLE001 - print a clear diagnostic
        print(f"\n[!] Cannot reach Ollama at {_base_url(host)}: {type(exc).__name__}: {exc}")
        print("    Check: (1) the host is on a routable network from this machine,")
        print("    (2) Ollama is started with OLLAMA_HOST=0.0.0.0 to listen on the LAN,")
        print("    (3) the firewall allows TCP 11434.")
        return 2

    print("\nPulled models:")
    for name in names:
        mark = " *" if name == model else ""
        print(f"  - {name}{mark}")

    if names and model not in names:
        print(f"\n[!] Model '{model}' is not pulled. Closest available:")
        close = [n for n in names if n.split(":")[0] == model.split(":")[0]]
        for n in close or names[:5]:
            print(f"    - {n}")
        print("    Pull it with: ollama pull " + model)

    harness = ollama_async_harness(json_mode=True)
    print("\nRunning 4 benchmark transcripts (real model, this may take a while)...")
    for tid, transcript in ALL_CASES:
        result = await harness.extract(transcript, transcript_id=tid)
        ticket = result.ticket
        extra = ""
        if ticket is not None:
            extra = f" category={ticket.issue_category.value} urgency={ticket.urgency.value}"
        print(f"  {tid}: {result.status.value}{extra}")
        if result.review_reason:
            print(f"      review: {result.review_reason[:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
