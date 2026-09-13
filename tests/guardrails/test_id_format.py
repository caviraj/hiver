"""Unit tests for structural ID format validation."""

import pytest
from src.guardrails.id_format import validate_id_format, validate_ticket_tags


def test_validate_id_format_valid():
    """Verify valid ticket IDs matching ^TCKT-\\d{6}$ return True."""
    assert validate_id_format("TCKT-123456") is True
    assert validate_id_format("TCKT-000000") is True
    assert validate_id_format("TCKT-999999") is True


def test_validate_id_format_malformed():
    """Verify malformed ticket IDs return False."""
    assert validate_id_format("TCKT-12345") is False  # 5 digits
    assert validate_id_format("TCKT-1234567") is False  # 7 digits
    assert validate_id_format("TCKT-ABCDEF") is False  # non-digits
    assert validate_id_format("TKT-123456") is False  # wrong prefix
    assert validate_id_format("TCKT123456") is False  # missing hyphen
    assert validate_id_format("tckt-123456") is False  # lowercase prefix
    assert validate_id_format("TCKT-12345 ") is False  # trailing space
    assert validate_id_format("") is False
    assert validate_id_format(None) is False


def test_validate_id_format_custom_pattern():
    """Verify custom regex patterns function properly."""
    custom_pattern = r"^ORD-[A-Z]{3}-\d{4}$"
    assert validate_id_format("ORD-ABC-1234", pattern=custom_pattern) is True
    assert validate_id_format("ORD-abc-1234", pattern=custom_pattern) is False
    assert validate_id_format("TCKT-123456", pattern=custom_pattern) is False


def test_validate_ticket_tags_valid():
    """Verify ticket tags with valid ticket IDs pass."""
    assert validate_ticket_tags(["TCKT-123456", "billing"]) is True
    assert validate_ticket_tags(["id:TCKT-999888", "urgent", "tier2"]) is True
    assert validate_ticket_tags(["general", "inquiry"]) is True
    assert validate_ticket_tags([]) is True


def test_validate_ticket_tags_malformed():
    """Verify ticket tags with malformed ticket IDs fail."""
    assert validate_ticket_tags(["TCKT-12345", "billing"]) is False
    assert validate_ticket_tags(["id:TCKT-XYZ", "urgent"]) is False
    assert validate_ticket_tags(["TCKT-1234567"]) is False
