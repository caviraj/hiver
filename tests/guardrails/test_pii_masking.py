"""Unit tests for PII masking with Luhn checksum validation."""

import pytest
from src.guardrails.pii_masking import mask_pii, luhn_checksum


def test_luhn_checksum_valid_cards():
    """Verify Luhn checksum returns True for standard valid test card numbers."""
    # Standard Luhn-valid test numbers
    assert luhn_checksum("4532015112830366") is True
    assert luhn_checksum("4532-0151-1283-0366") is True
    assert luhn_checksum("5425 2334 3010 9903") is True
    assert luhn_checksum("49927398716") is True


def test_luhn_checksum_invalid_cards():
    """Verify Luhn checksum returns False for invalid numbers."""
    # Altered last digit
    assert luhn_checksum("4532015112830367") is False
    assert luhn_checksum("1234567812345678") is False
    assert luhn_checksum("0000000000000001") is False
    assert luhn_checksum("abc") is False
    assert luhn_checksum("") is False


def test_mask_pii_credit_card_valid():
    """Valid credit card numbers must be replaced with [REDACTED_CREDIT_CARD]."""
    text = "My card number is 4532-0151-1283-0366, please process the refund."
    masked, pii_types = mask_pii(text)
    assert "4532-0151-1283-0366" not in masked
    assert "[REDACTED_CREDIT_CARD]" in masked
    assert "credit_card" in pii_types
    assert masked == "My card number is [REDACTED_CREDIT_CARD], please process the refund."


def test_mask_pii_credit_card_invalid_luhn_not_masked():
    """A 16-digit order or reference number failing Luhn check must NOT be masked."""
    # 1234-5678-1234-5678 fails Luhn check
    order_num = "1234-5678-1234-5678"
    assert luhn_checksum(order_num) is False

    text = f"Order reference ID is {order_num}. Do not charge it."
    masked, pii_types = mask_pii(text)
    assert order_num in masked
    assert "[REDACTED_CREDIT_CARD]" not in masked
    assert "credit_card" not in pii_types


def test_mask_pii_ssn_standard():
    """SSN in standard XXX-XX-XXXX format must be masked."""
    text = "Customer SSN: 123-45-6789 on file."
    masked, pii_types = mask_pii(text)
    assert "123-45-6789" not in masked
    assert "[REDACTED_SSN]" in masked
    assert "ssn" in pii_types
    assert masked == "Customer SSN: [REDACTED_SSN] on file."


def test_mask_pii_ssn_variants():
    """SSN with space or continuous digits must be masked."""
    text1 = "SSN is 987 65 4321."
    masked1, pii_types1 = mask_pii(text1)
    assert "[REDACTED_SSN]" in masked1
    assert "ssn" in pii_types1

    text2 = "SSN is 987654321."
    masked2, pii_types2 = mask_pii(text2)
    assert "[REDACTED_SSN]" in masked2
    assert "ssn" in pii_types2


def test_mask_pii_multiple_types():
    """Verify both SSN and Credit Card are detected and masked in a single text."""
    text = "User SSN 123-45-6789 paid with card 4532015112830366."
    masked, pii_types = mask_pii(text)
    assert "[REDACTED_SSN]" in masked
    assert "[REDACTED_CREDIT_CARD]" in masked
    assert "ssn" in pii_types
    assert "credit_card" in pii_types
    assert "123-45-6789" not in masked
    assert "4532015112830366" not in masked


def test_mask_pii_clean_text():
    """Clean text should remain unchanged with empty pii_types."""
    text = "Hello, I would like to check the status of ticket TCKT-100200."
    masked, pii_types = mask_pii(text)
    assert masked == text
    assert pii_types == []


def test_mask_pii_empty_or_none():
    """Empty or None text should return empty string and empty list."""
    assert mask_pii("") == ("", [])
    assert mask_pii(None) == ("", [])
