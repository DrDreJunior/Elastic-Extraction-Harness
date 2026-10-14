"""PII detection + redaction.

Conservative, high-confidence patterns for the identifiers called out by the
challenge: payment cards (Luhn-checked), PRC resident identity cards (18-digit
with ISO 7064 checksum + date validation), Singapore NRIC, passwords, and
secret/token assignments.  Redaction runs before the transcript is logged,
sent to the model, or returned to callers.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

from .text_norm import normalize_untrusted

# Visa / MC / Amex / Discover, plus a generic 13-19 digit run with separators.
# A match is only treated as a card when it passes the Luhn check, so invoice
# numbers and phone-like runs are not destroyed.
_CARD_CANDIDATE = re.compile(
    r"(?<![0-9A-Z])"
    r"(?:"
    r"4[0-9]{12}(?:[0-9]{3})?"
    r"|5[1-5][0-9]{14}"
    r"|3[47][0-9]{13}"
    r"|6(?:011|5[0-9]{2})[0-9]{12}"
    r"|(?:\d[ -]?){13,19}"
    r")(?![0-9])"
)

# PRC Resident Identity Card (2nd generation): 17 digits + 1 check char
# (digit or X). The 18-digit compact form is checked before cards so a valid
# national ID is never mistaken for a Luhn card.
_PRC_ID = re.compile(r"(?<![0-9A-Z])\d{17}[\dX](?![0-9A-Z])", re.IGNORECASE)
_PRC_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_PRC_CHECK = "10X98765432"

# Singapore NRIC: S/T/F/G + 7 digits + checksum letter. Case-insensitive
# because customers type it either way.
_NRIC = re.compile(r"\b[STFG]\d{7}[A-Z]\b", re.IGNORECASE)

# Label is group 1 so the secret value can be replaced without echoing it.
_PASSWORD = re.compile(
    r"(?i)((?:password|passwd|pwd|密码)\s*(?:is|:|=|是)\s*[:=]?\s*)([^\s,.;，。；]{4,})"
)
_SECRET = re.compile(
    r"(?i)((?:secret|token|api[_-]?key|private[_-]?key)\s*(?:is|:|=|是)?\s*[:=]?\s*)"
    r"([A-Za-z0-9_\-]{8,})"
)


@dataclass
class PIIDetection:
    has_pii: bool
    matched_patterns: list[str]
    redacted_text: str


def _luhn_ok(digits: str) -> bool:
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total = 0
    double = False
    for ch in reversed(digits):
        n = ord(ch) - 48
        if double:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        double = not double
    return total % 10 == 0


def _prc_check_digit(first17: str) -> str:
    total = sum(int(d) * w for d, w in zip(first17, _PRC_WEIGHTS, strict=True))
    return _PRC_CHECK[total % 11]


def _prc_id_valid(num: str) -> bool:
    """Validate checksum, date of birth, and a sane year range."""
    if len(num) != 18 or not num[:17].isdigit():
        return False
    last = num[17].upper()
    if not (last.isdigit() or last == "X"):
        return False
    try:
        year = int(num[6:10])
        month = int(num[10:12])
        day = int(num[12:14])
    except ValueError:
        return False
    if not 1900 <= year <= date.today().year:
        return False
    if not 1 <= month <= 12 or not 1 <= day <= calendar.monthrange(year, month)[1]:
        return False
    return _prc_check_digit(num[:17]) == last


def _redact_prc_ids(text: str) -> tuple[str, bool]:
    found = False

    def _repl(match: re.Match[str]) -> str:
        nonlocal found
        num = match.group(0).upper()
        if not _prc_id_valid(num):
            return match.group(0)
        found = True
        return "[REDACTED_NATIONAL_ID]"

    return _PRC_ID.sub(_repl, text), found


def _redact_cards(text: str) -> tuple[str, bool]:
    found = False

    def _repl(match: re.Match[str]) -> str:
        nonlocal found
        digits = re.sub(r"[ -]", "", match.group(0))
        if not _luhn_ok(digits):
            return match.group(0)
        found = True
        return "[REDACTED_CARD]"

    return _CARD_CANDIDATE.sub(_repl, text), found


def detect_and_redact(text: str) -> PIIDetection:
    """Scan *text* for PII and return a redacted copy."""
    if not text:
        return PIIDetection(has_pii=False, matched_patterns=[], redacted_text=text or "")

    matched: list[str] = []
    redacted = normalize_untrusted(text)

    # National IDs first (18-digit, checksummed), then cards, so an 18-digit ID
    # that fails Luhn is not mistaken for either and a valid one is not eaten.
    redacted, id_hit = _redact_prc_ids(redacted)
    if id_hit:
        matched.append("NATIONAL_ID")

    redacted, card_hit = _redact_cards(redacted)
    if card_hit:
        matched.append("CREDIT_CARD")

    if _NRIC.search(redacted):
        matched.append("NRIC")
        redacted = _NRIC.sub("[REDACTED_NRIC]", redacted)

    redacted, pw_hits = _PASSWORD.subn(r"\1[REDACTED_PASSWORD]", redacted)
    if pw_hits:
        matched.append("PASSWORD")

    # Group 1 is the label, group 2 is the secret. Never write group 2 back.
    redacted, secret_hits = _SECRET.subn(r"\1[REDACTED_SECRET]", redacted)
    if secret_hits:
        matched.append("SECRET")

    return PIIDetection(
        has_pii=bool(matched),
        matched_patterns=matched,
        redacted_text=redacted,
    )
