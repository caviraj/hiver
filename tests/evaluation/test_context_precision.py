"""Unit tests for Context Precision metric and relevance judge."""

from unittest.mock import MagicMock
import pytest

from src.evaluation.context_precision import compute_context_precision
from src.evaluation.relevance_judge import (
    judge_relevance,
    RelevanceJudgeError,
)


def test_order_sensitivity_proof():
    """Mathematical proof test for Context Precision rank-order sensitivity.

    A relevant document ranked at position #1 MUST yield a strictly higher
    score than the exact same relevant document ranked at position #5
    (when preceded by 4 irrelevant documents).

    Rank #1 calculation:
        Rank 1: Relevant -> P@1 = 1/1 = 1.0
        Ranks 2-5: Irrelevant
        Context Precision = (1.0 * 1) / 1 = 1.0

    Rank #5 calculation:
        Ranks 1-4: Irrelevant -> P@1..4 = 0
        Rank 5: Relevant -> P@5 = 1/5 = 0.2
        Context Precision = (0.2 * 1) / 1 = 0.2
    """
    rel_chunk = "The refund policy allows 30-day full refunds with receipt."
    irrel_chunk_1 = "Bananas are high in potassium and fiber."
    irrel_chunk_2 = "The capital of France is Paris."
    irrel_chunk_3 = "Quantum computing uses qubits instead of bits."
    irrel_chunk_4 = "Mount Everest is the highest mountain on Earth."

    reference = "Customers can receive a full refund within 30 days if they have a receipt."

    def mock_llm_judge(prompt: str, temperature: int = 0) -> str:
        assert temperature == 0, "Evaluation judge calls MUST enforce temperature=0"
        if rel_chunk in prompt:
            return "YES"
        return "NO"

    # Scenario A: Relevant chunk at Rank #1
    contexts_rank1 = [rel_chunk, irrel_chunk_1, irrel_chunk_2, irrel_chunk_3, irrel_chunk_4]
    score_rank1 = compute_context_precision(reference, contexts_rank1, mock_llm_judge)

    # Scenario B: Relevant chunk at Rank #5
    contexts_rank5 = [irrel_chunk_1, irrel_chunk_2, irrel_chunk_3, irrel_chunk_4, rel_chunk]
    score_rank5 = compute_context_precision(reference, contexts_rank5, mock_llm_judge)

    assert score_rank1 == 1.0
    assert pytest.approx(score_rank5, 0.001) == 0.2
    assert score_rank1 > score_rank5, "Rank #1 relevant item must score strictly higher than Rank #5"


def test_context_precision_zero_relevant():
    """When retrieved contexts contain 0 relevant items, score must be 0.0 (not div by zero)."""
    reference = "Expected delivery time is 2-3 business days."
    contexts = [
        "Cats are domestic animals.",
        "The sun rises in the east.",
    ]

    mock_llm = lambda prompt, temperature=0: "NO"
    score = compute_context_precision(reference, contexts, mock_llm)
    assert score == 0.0


def test_context_precision_empty_contexts():
    """Empty retrieved contexts list returns 0.0."""
    reference = "Some reference answer."
    score = compute_context_precision(reference, [], MagicMock())
    assert score == 0.0


def test_context_precision_empty_reference():
    """Empty reference answer returns 0.0."""
    score = compute_context_precision("", ["Some context"], MagicMock())
    assert score == 0.0


def test_context_precision_perfect_score():
    """All retrieved chunks are relevant -> Context Precision must be 1.0."""
    reference = "Hiver provides shared inboxes."
    contexts = ["Context 1 about Hiver", "Context 2 about shared inboxes"]

    mock_llm = lambda prompt, temperature=0: "YES"
    score = compute_context_precision(reference, contexts, mock_llm)
    assert pytest.approx(score, 0.001) == 1.0


def test_relevance_judge_retry_success():
    """Judge should retry once on initial failure and succeed if retry returns valid verdict."""
    mock_llm = MagicMock(side_effect=[Exception("Transient network timeout"), "YES"])

    verdict = judge_relevance("Target query", "Context chunk", mock_llm)
    assert verdict is True
    assert mock_llm.call_count == 2


def test_relevance_judge_retry_failure_raises_error():
    """Judge should raise RelevanceJudgeError when retry also fails."""
    mock_llm = MagicMock(side_effect=[Exception("Timeout 1"), Exception("Timeout 2")])

    with pytest.raises(RelevanceJudgeError):
        judge_relevance("Target query", "Context chunk", mock_llm)


def test_context_precision_excludes_failed_judge_chunk():
    """If one chunk fails after retries, it is excluded without crashing the entire evaluation."""
    reference = "Target info"
    contexts = ["Valid context 1", "Failing context 2", "Valid context 3"]

    def mock_llm(prompt: str, temperature: int = 0):
        if "Failing context 2" in prompt:
            raise RuntimeError("API Error")
        return "YES"

    # "Failing context 2" will be excluded; "Valid context 1" and "Valid context 3" are relevant
    # Evaluated contexts: 2 relevant out of 2 evaluated -> score 1.0
    score = compute_context_precision(reference, contexts, mock_llm)
    assert pytest.approx(score, 0.001) == 1.0
