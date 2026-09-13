"""Unit tests for Context Recall metric and reference decomposition."""

from unittest.mock import MagicMock
import pytest

from src.evaluation.context_recall import (
    decompose_reference,
    compute_context_recall,
)


def test_decompose_reference_json_format():
    """Decomposition should extract list of facts when LLM returns clean JSON."""
    reference = "Hiver is a shared inbox tool. It works directly inside Gmail."

    mock_llm = MagicMock(return_value='["Hiver is a shared inbox tool.", "It works directly inside Gmail."]')
    facts = decompose_reference(reference, mock_llm)

    assert len(facts) == 2
    assert facts[0] == "Hiver is a shared inbox tool."
    assert facts[1] == "It works directly inside Gmail."


def test_decompose_reference_markdown_codeblock():
    """Decomposition should handle markdown-fenced JSON responses."""
    reference = "Python was created by Guido van Rossum. It was released in 1991."

    mock_response = """```json
[
  "Python was created by Guido van Rossum.",
  "Python was released in 1991."
]
```"""
    mock_llm = MagicMock(return_value=mock_response)
    facts = decompose_reference(reference, mock_llm)

    assert len(facts) == 2
    assert "Guido van Rossum" in facts[0]
    assert "1991" in facts[1]


def test_decompose_reference_bullet_fallback():
    """Decomposition should fall back to parsing bullet points if JSON is invalid."""
    reference = "The refund window is 30 days. Receipts are required."

    mock_response = """* The refund window is 30 days.
* Receipts are required for all returns."""
    mock_llm = MagicMock(return_value=mock_response)
    facts = decompose_reference(reference, mock_llm)

    assert len(facts) == 2
    assert "30 days" in facts[0]
    assert "Receipts are required" in facts[1]


def test_decompose_reference_empty_input():
    """Empty or whitespace reference returns an empty list."""
    assert decompose_reference("", MagicMock()) == []
    assert decompose_reference("   ", MagicMock()) == []


def test_decompose_reference_exception_fallback():
    """If LLM call fails completely, fallback splits on sentences."""
    reference = "First fact here. Second fact here!"
    mock_llm = MagicMock(side_effect=RuntimeError("API down"))

    facts = decompose_reference(reference, mock_llm)
    assert len(facts) == 2
    assert facts[0] == "First fact here"
    assert facts[1] == "Second fact here"


def test_context_recall_full_attribution():
    """When all atomic facts are supported by retrieved contexts, Context Recall is 1.0."""
    reference = "Fact A. Fact B."
    contexts = [
        "Context containing evidence for Fact A.",
        "Context containing evidence for Fact B.",
    ]

    def mock_llm(prompt: str, temperature: int = 0) -> str:
        assert temperature == 0, "Judge calls must enforce temperature=0"
        if "Break down the following reference" in prompt:
            return '["Fact A", "Fact B"]'
        # Attribution checks
        return "YES"

    score = compute_context_recall(reference, contexts, mock_llm)
    assert pytest.approx(score, 0.001) == 1.0


def test_context_recall_partial_attribution():
    """When 1 of 2 facts is supported by contexts, Context Recall is 0.5."""
    reference = "Fact 1 is true. Fact 2 is true."
    contexts = ["Context supporting only Fact 1."]

    def mock_llm(prompt: str, temperature: int = 0) -> str:
        if "Break down the following reference" in prompt:
            return '["Fact 1", "Fact 2"]'
        if "Target:\nFact 1" in prompt:
            return "YES"
        return "NO"

    score = compute_context_recall(reference, contexts, mock_llm)
    assert pytest.approx(score, 0.001) == 0.5


def test_context_recall_zero_attribution():
    """When no facts are supported by contexts, Context Recall is 0.0."""
    reference = "Fact 1. Fact 2."
    contexts = ["Completely irrelevant context."]

    def mock_llm(prompt: str, temperature: int = 0) -> str:
        if "Break down the following reference" in prompt:
            return '["Fact 1", "Fact 2"]'
        return "NO"

    score = compute_context_recall(reference, contexts, mock_llm)
    assert score == 0.0


def test_context_recall_empty_inputs():
    """Empty contexts or reference returns 0.0."""
    mock_llm = MagicMock()
    assert compute_context_recall("", ["context"], mock_llm) == 0.0
    assert compute_context_recall("reference", [], mock_llm) == 0.0


def test_context_recall_batching_and_short_circuit():
    """Long contexts list is chunked in batches and short-circuits when fact is found."""
    reference = "Target Fact"
    # 6 context chunks; with batch_chunk_size=2, there will be 3 batches
    contexts = [
        "Chunk 1: Irrelevant",
        "Chunk 2: Irrelevant",
        "Chunk 3: Contains Target Fact evidence!",
        "Chunk 4: Irrelevant",
        "Chunk 5: Irrelevant",
        "Chunk 6: Irrelevant",
    ]

    judge_call_count = 0

    def mock_llm(prompt: str, temperature: int = 0) -> str:
        nonlocal judge_call_count
        if "Break down the following reference" in prompt:
            return '["Target Fact"]'
        judge_call_count += 1
        if "Chunk 3" in prompt:
            return "YES"
        return "NO"

    score = compute_context_recall(reference, contexts, mock_llm, batch_chunk_size=2)
    assert score == 1.0
    # Batch 1 (Chunk 1 & 2) -> NO
    # Batch 2 (Chunk 3 & 4) -> YES (should short-circuit, so Batch 3 is never called)
    assert judge_call_count == 2
