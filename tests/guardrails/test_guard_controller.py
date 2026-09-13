"""Unit tests for GuardController orchestration, retry-and-correct loop, and invariants."""

import json
import pytest
from unittest.mock import patch, mock_open
from src.guardrails.guard_controller import GuardController
from src.guardrails.output_schema import AgentResponse, GuardValidationResult


@pytest.fixture
def controller(tmp_path):
    """Fixture providing a GuardController writing to a temporary log file."""
    log_file = tmp_path / "test_guardrail_validation_log.jsonl"
    return GuardController(log_file_path=str(log_file), toxicity_threshold=0.7)


def test_valid_response_passes_immediately(controller):
    """Valid JSON matching AgentResponse schema passes on attempt 1 without retries."""
    valid_payload = json.dumps({
        "reply_text": "Your return request has been submitted successfully.",
        "ticket_tags": ["TCKT-123456", "refund"],
        "requires_human_review": False,
        "confidence": 0.95
    })

    def mock_retry(instruction: str) -> str:
        pytest.fail("Retry function should not be called when initial output is valid")

    result = controller.validate_and_correct(valid_payload, generation_retry_fn=mock_retry)

    assert result.validated is True
    assert result.escalate is False
    assert result.retry_count == 0
    assert result.response is not None
    assert result.response.reply_text == "Your return request has been submitted successfully."
    assert result.response.ticket_tags == ["TCKT-123456", "refund"]
    assert result.response.confidence == 0.95


def test_unconditional_pii_masking(controller):
    """PII must be masked unconditionally before schema parsing and validator execution."""
    raw_payload = json.dumps({
        "reply_text": "We verified SSN 123-45-6789 and card 4532015112830366 for refund.",
        "ticket_tags": ["TCKT-100200"],
        "requires_human_review": False,
        "confidence": 0.90
    })

    result = controller.validate_and_correct(raw_payload, generation_retry_fn=lambda x: raw_payload)

    assert result.validated is True
    assert result.response is not None
    assert "123-45-6789" not in result.response.reply_text
    assert "4532015112830366" not in result.response.reply_text
    assert "[REDACTED_SSN]" in result.response.reply_text
    assert "[REDACTED_CREDIT_CARD]" in result.response.reply_text
    assert result.response.requires_human_review is True  # Flagged because PII was detected


def test_retry_and_correct_succeeds_on_second_attempt(controller):
    """If initial output is malformed or invalid, generation_retry_fn is called and can fix it."""
    # Attempt 1: Malformed ticket ID in tags
    bad_payload = json.dumps({
        "reply_text": "Ticket opened.",
        "ticket_tags": ["TCKT-999"],  # Invalid format (only 3 digits)
        "requires_human_review": False,
        "confidence": 0.85
    })

    # Attempt 2: Fixed payload
    good_payload = json.dumps({
        "reply_text": "Ticket opened.",
        "ticket_tags": ["TCKT-999000"],  # Fixed format
        "requires_human_review": False,
        "confidence": 0.85
    })

    call_count = 0

    def mock_retry(instruction: str) -> str:
        nonlocal call_count
        call_count += 1
        assert "ID Format Validator Failed" in instruction or "TCKT" in instruction
        return good_payload

    result = controller.validate_and_correct(bad_payload, generation_retry_fn=mock_retry, max_retries=2)

    assert call_count == 1
    assert result.validated is True
    assert result.escalate is False
    assert result.retry_count == 1
    assert result.response is not None
    assert result.response.ticket_tags == ["TCKT-999000"]


def test_retry_exhaustion_inviolable_invariant_never_return_invalid_output(controller):
    """When retries are exhausted, validated=False, escalate=True, and response is strictly None."""
    # Persistently invalid output (competitor recommendation)
    bad_payload = json.dumps({
        "reply_text": "You must switch to Zendesk right now.",
        "ticket_tags": ["TCKT-123456"],
        "requires_human_review": False,
        "confidence": 0.9
    })

    retry_calls = 0

    def persistent_bad_generator(instruction: str) -> str:
        nonlocal retry_calls
        retry_calls += 1
        return bad_payload

    result = controller.validate_and_correct(
        bad_payload,
        generation_retry_fn=persistent_bad_generator,
        max_retries=2
    )

    assert retry_calls == 2
    assert result.validated is False
    assert result.escalate is True
    assert result.retry_count == 2
    # CRITICAL INVARIANT: The invalid output is NEVER returned to the caller
    assert result.response is None
    assert "Competitor Validator Failed" in result.failure_reasons[0]


def test_empty_or_none_raw_output_escalates_immediately(controller):
    """Empty or None output escalates immediately without wasting retry calls."""
    retry_called = False

    def mock_retry(instruction: str) -> str:
        nonlocal retry_called
        retry_called = True
        return "{}"

    for empty_input in ["", "   ", None]:
        result = controller.validate_and_correct(empty_input, generation_retry_fn=mock_retry)
        assert result.validated is False
        assert result.escalate is True
        assert result.response is None
        assert result.retry_count == 0
        assert retry_called is False


def test_multiple_validators_fail_simultaneously(controller):
    """When multiple validators fail, PII is still masked unconditionally and all failures are logged."""
    # Payload has SSN, malformed ticket ID, and competitor recommendation
    multi_fail_payload = json.dumps({
        "reply_text": "Customer SSN 987-65-4321, you should really switch to Zendesk.",
        "ticket_tags": ["TCKT-BAD"],
        "requires_human_review": False,
        "confidence": 0.95
    })

    retry_instruction_captured = ""

    def mock_retry(instruction: str) -> str:
        nonlocal retry_instruction_captured
        retry_instruction_captured = instruction
        # Return valid payload on retry
        return json.dumps({
            "reply_text": "Customer SSN [REDACTED_SSN], your account is updated.",
            "ticket_tags": ["TCKT-123456"],
            "requires_human_review": True,
            "confidence": 0.95
        })

    result = controller.validate_and_correct(multi_fail_payload, generation_retry_fn=mock_retry)

    # Instruction sent to generator should mention both failed validators
    assert "Competitor Validator Failed" in retry_instruction_captured
    assert "ID Format Validator Failed" in retry_instruction_captured

    assert result.validated is True
    assert result.response is not None
    assert "[REDACTED_SSN]" in result.response.reply_text


def test_log_failure_non_crashing(controller):
    """If log file write fails (disk error or permission denied), validation must NOT crash."""
    valid_payload = json.dumps({
        "reply_text": "Normal support reply.",
        "ticket_tags": ["TCKT-123456"],
        "requires_human_review": False,
        "confidence": 0.9
    })

    with patch("builtins.open", side_effect=IOError("Disk full or permission denied")):
        # Log failure internally catches IOError and logs a warning
        controller._log_failure(
            attempt=1,
            failed_validators=["test_error"],
            failure_reasons=["reason"],
            raw_output="raw"
        )
        # Verify validate_and_correct also proceeds without crashing
        result = controller.validate_and_correct(valid_payload, generation_retry_fn=lambda x: valid_payload)
        assert result.validated is True
        assert result.response is not None
