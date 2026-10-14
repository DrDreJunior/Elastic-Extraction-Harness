"""Normalisation for untrusted text.

Applied before PII scan, injection scan, and the LLM prompt so that trivial
obfuscation (fullwidth ASCII, zero-width characters) cannot bypass either
regex layer.  Kept in its own module to avoid import cycles between `pii`
and `injection`.
"""
from __future__ import annotations

import re
import unicodedata

# Zero-width and bidi controls commonly used to hide injection phrases.
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff]")


def normalize_untrusted(text: str) -> str:
    """NFKC-fold and strip invisible characters. Idempotent for our inputs."""
    if not text:
        return ""
    return _INVISIBLE.sub("", unicodedata.normalize("NFKC", text))
