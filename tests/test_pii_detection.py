"""PII detector unit tests."""
from __future__ import annotations

from app.pii import detect_and_redact
from app.sample_data import CASE2_PII


def test_nric_detected():
    r = detect_and_redact("My NRIC is S9123456A")
    assert r.has_pii
    assert "NRIC" in r.matched_patterns
    assert "S9123456A" not in r.redacted_text
    assert "[REDACTED_NRIC]" in r.redacted_text


def test_password_detected():
    r = detect_and_redact("my password is SuperSecretPass123")
    assert r.has_pii
    assert "PASSWORD" in r.matched_patterns
    assert "SuperSecretPass123" not in r.redacted_text


def test_credit_card_detected():
    r = detect_and_redact("my card is 4111111111111111")
    assert r.has_pii
    assert "CREDIT_CARD" in r.matched_patterns
    assert "4111111111111111" not in r.redacted_text


def test_clean_transcript_no_pii():
    r = detect_and_redact("hi I need help with my invoice #1001")
    assert r.has_pii is False
    assert r.redacted_text == "hi I need help with my invoice #1001"


def test_case2_fixture_has_both_password_and_nric():
    r = detect_and_redact(CASE2_PII)
    assert r.has_pii
    assert "PASSWORD" in r.matched_patterns
    assert "NRIC" in r.matched_patterns
    # Both secrets must be gone from the redacted transcript.
    assert "SuperSecretPass123" not in r.redacted_text
    assert "S9123456A" not in r.redacted_text


# --- PRC Resident Identity Card (身份证号) ---

def test_prc_national_id_with_x_detected():
    r = detect_and_redact("我的身份证号是 11010519491231002X")
    assert r.has_pii
    assert "NATIONAL_ID" in r.matched_patterns
    assert "11010519491231002X" not in r.redacted_text
    assert "[REDACTED_NATIONAL_ID]" in r.redacted_text


def test_prc_national_id_with_digit_check_detected():
    # Build a valid number dynamically so the test does not depend on a magic value.
    from app.pii import _prc_check_digit
    first17 = "11010519900307123"  # birth 1990-03-07
    valid = first17 + _prc_check_digit(first17)
    r = detect_and_redact(f"id: {valid}")
    assert r.has_pii
    assert "NATIONAL_ID" in r.matched_patterns
    assert valid not in r.redacted_text


def test_prc_id_wrong_checksum_is_not_redacted():
    # 11010519491231002 + check digit '9' (correct digit is X) -> invalid.
    r = detect_and_redact("110105194912310029")
    assert r.has_pii is False
    assert "110105194912310029" in r.redacted_text


def test_prc_id_bad_date_is_not_redacted():
    # Month 13 is impossible even if the digit count looks right.
    from app.pii import _prc_check_digit
    first17 = "11010519901307123"  # month 13
    num = first17 + _prc_check_digit(first17)
    r = detect_and_redact(num)
    assert "NATIONAL_ID" not in r.matched_patterns


def test_prc_id_lowercase_x_is_detected():
    r = detect_and_redact("my id is 11010519491231002x")
    assert "NATIONAL_ID" in r.matched_patterns
    assert "11010519491231002x" not in r.redacted_text
