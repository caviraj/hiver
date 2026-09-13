"""Unit tests for Faithfulness claim extraction and verification."""

from unittest.mock import MagicMock
import pytest

from src.evaluation.claim_verification import (
    extract_claims,
    verify_claims,
    compute_faithfulness,
)


def test_extract_claims_empty_or_trivial():
    """Trivial or empty responses yield zero claims."""
    assert extract_claims("") == []
    assert extract_claims("   ") == []
    assert extract_claims("Hello! Thank you. Best regards.") == []


def test_extract_claims_standard_sentences():
    """Substantive sentences are extracted as claims."""
    response = (
        "The standard refund period is 30 days from purchase. "
        "Customers must provide a valid receipt to qualify."
    )
    claims = extract_claims(response)
    assert len(claims) == 2
    assert "30 days" in claims[0]
    assert "valid receipt" in claims[1]


def test_extract_claims_custom_model_method():
    """If hhem_model provides a custom extract_claims method, it is utilized."""
    mock_model = MagicMock()
    mock_model.extract_claims.return_value = ["Custom Claim 1", "Custom Claim 2"]

    claims = extract_claims("Some text", hhem_model=mock_model)
    assert claims == ["Custom Claim 1", "Custom Claim 2"]
    mock_model.extract_claims.assert_called_once_with("Some text")


def test_compute_faithfulness_zero_claims():
    """Zero extracted claims must return (None, 0, 0), not a 0/0 division error or 0.0."""
    response = "Hi! Thank you."
    contexts = ["Context information here."]

    score, num_claims, num_supported = compute_faithfulness(response, contexts, MagicMock())
    assert score is None, "Faithfulness must be None (undefined) when no verifiable claims exist"
    assert num_claims == 0
    assert num_supported == 0


def test_compute_faithfulness_empty_contexts():
    """Empty retrieved contexts return (None, num_claims, 0) because claims cannot be evaluated."""
    response = "The refund window is 30 calendar days."

    score, num_claims, num_supported = compute_faithfulness(response, [], MagicMock())
    assert score is None, "Faithfulness must be None when no context is available to verify against"
    assert num_claims > 0
    assert num_supported == 0


def test_compute_faithfulness_all_claims_unsupported():
    """All claims unsupported yields score 0.0 (hallucination signal, not None or error)."""
    response = "The refund window is 30 days. You will also receive free coffee."
    contexts = ["Only exchange is permitted. No refunds under any condition."]

    mock_model = MagicMock()
    # Mock predict returning low consistency for everything
    mock_model.predict.return_value = [0.1]

    score, num_claims, num_supported = compute_faithfulness(response, contexts, mock_model)
    assert score == 0.0
    assert num_claims == 2
    assert num_supported == 0


def test_compute_faithfulness_full_support():
    """All claims supported yields score 1.0."""
    response = "The refund window is 30 days. A receipt is required."
    contexts = ["The refund window is 30 days and a receipt is required."]

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.95]

    score, num_claims, num_supported = compute_faithfulness(response, contexts, mock_model)
    assert pytest.approx(score, 0.001) == 1.0
    assert num_claims == 2
    assert num_supported == 2


def test_compute_faithfulness_partial_support():
    """Partial claim support (1 of 2) yields score 0.5."""
    response = "The refund window is 30 days. Customers also receive $100 bonus cash."
    contexts = ["The refund window is 30 days from purchase."]

    def mock_predict(pairs):
        premise, hypothesis = pairs[0]
        if "30 days" in hypothesis:
            return [0.9]
        return [0.1]

    mock_model = MagicMock()
    mock_model.predict.side_effect = mock_predict

    score, num_claims, num_supported = compute_faithfulness(response, contexts, mock_model)
    assert pytest.approx(score, 0.001) == 0.5
    assert num_claims == 2
    assert num_supported == 1


def test_verify_claims_callable_model():
    """Directly callable model interface should be supported by verify_claims."""
    claims = ["Claim A"]
    contexts = ["Context A"]

    def mock_callable(premise, hypothesis):
        return 0.85

    verifications = verify_claims(claims, contexts, mock_callable)
    assert verifications == [True]
