import json
import os
from unittest.mock import MagicMock, patch
import pytest

from src.evaluation.schema import RAGASInput, RAGASResult
from src.evaluation.ragas_evaluator import evaluate_ragas


class MockLLMClient:
    def __init__(self, response: str = "yes"):
        self.response = response

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        # If prompt is asking for decomposition
        if "factual statements" in prompt.lower() or "json list" in prompt.lower():
            return '["Fact 1", "Fact 2"]'
        return self.response


class MockHHEMModel:
    def __init__(self, score: float = 0.9):
        self.score = score

    def predict(self, premise_hypothesis_pairs):
        return [self.score for _ in premise_hypothesis_pairs]


def test_evaluate_ragas_full(tmp_path):
    log_file = tmp_path / "ragas_results.jsonl"
    
    llm = MockLLMClient(response="yes")
    hhem = MockHHEMModel(score=0.95)
    
    inp = RAGASInput(
        query="What is the refund policy?",
        retrieved_contexts=["Refunds are issued within 14 days of purchase."],
        generated_response="Refunds are available within 14 days.",
        reference_answer="Refunds are provided within 14 days of purchase."
    )
    
    result = evaluate_ragas(inp, llm_client=llm, hhem_model=hhem, log_path=str(log_file))
    
    assert isinstance(result, RAGASResult)
    assert result.context_precision == 1.0
    assert result.context_recall == 1.0
    assert result.faithfulness == 1.0
    assert result.num_claims_extracted > 0
    assert result.num_claims_supported == result.num_claims_extracted
    
    # Check log file written
    assert os.path.exists(log_file)
    with open(log_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 1
    log_data = json.loads(lines[0])
    assert log_data["query"] == "What is the refund policy?"
    assert log_data["context_precision"] == 1.0
    assert log_data["faithfulness"] == 1.0


def test_evaluate_ragas_production_mode_no_reference(tmp_path):
    log_file = tmp_path / "ragas_prod.jsonl"
    llm = MockLLMClient(response="yes")
    hhem = MockHHEMModel(score=0.9)
    
    inp = RAGASInput(
        query="Explain delivery charges.",
        retrieved_contexts=["Delivery is free over $50."],
        generated_response="Shipping is free for orders exceeding $50.",
        reference_answer=None
    )
    
    result = evaluate_ragas(inp, llm_client=llm, hhem_model=hhem, log_path=str(log_file))
    
    # Precision and recall must remain None (undefined, NOT 0.0)
    assert result.context_precision is None
    assert result.context_recall is None
    assert result.faithfulness == 1.0
    assert any("reference_answer is None" in note for note in result.notes)


def test_evaluate_ragas_zero_contexts(tmp_path):
    llm = MockLLMClient(response="yes")
    hhem = MockHHEMModel(score=0.9)
    
    inp = RAGASInput(
        query="What is the refund policy?",
        retrieved_contexts=[],
        generated_response="Refunds are 14 days.",
        reference_answer="Refunds are 14 days."
    )
    
    result = evaluate_ragas(inp, llm_client=llm, hhem_model=hhem, log_path=None)
    
    # Empty contexts -> precision and recall are 0.0, faithfulness is None
    assert result.context_precision == 0.0
    assert result.context_recall == 0.0
    assert result.faithfulness is None
    assert any("Empty retrieved_contexts" in note for note in result.notes)


def test_evaluate_ragas_zero_claims(tmp_path):
    llm = MockLLMClient(response="yes")
    hhem = MockHHEMModel(score=0.9)
    
    inp = RAGASInput(
        query="What is your name?",
        retrieved_contexts=["I am an AI assistant."],
        generated_response="   ",  # Blank/trivial response
        reference_answer="I am an AI assistant."
    )
    
    result = evaluate_ragas(inp, llm_client=llm, hhem_model=hhem, log_path=None)
    
    # Blank response -> zero claims -> faithfulness is None (NOT 0.0)
    assert result.faithfulness is None
    assert result.num_claims_extracted == 0
    assert result.num_claims_supported == 0
    assert any("No verifiable claims" in note for note in result.notes)


def test_evaluate_ragas_all_claims_unsupported(tmp_path):
    llm = MockLLMClient(response="yes")
    hhem = MockHHEMModel(score=0.1)  # Low score => unsupported
    
    inp = RAGASInput(
        query="What is the delivery fee?",
        retrieved_contexts=["Delivery is $5 everywhere."],
        generated_response="Delivery is always free worldwide.",
        reference_answer="Delivery is $5."
    )
    
    result = evaluate_ragas(inp, llm_client=llm, hhem_model=hhem, log_path=None)
    
    # All claims unsupported -> faithfulness is 0.0 (NOT None)
    assert result.faithfulness == 0.0
    assert result.num_claims_extracted >= 1
    assert result.num_claims_supported == 0


def test_evaluate_ragas_io_failure_does_not_crash():
    llm = MockLLMClient(response="yes")
    hhem = MockHHEMModel(score=0.9)
    
    inp = RAGASInput(
        query="Test query",
        retrieved_contexts=["Context"],
        generated_response="Response",
        reference_answer="Reference"
    )
    
    # Mock open to raise PermissionError during log writing
    with patch("builtins.open", side_effect=PermissionError("Disk write blocked")):
        # Should not raise exception
        result = evaluate_ragas(inp, llm_client=llm, hhem_model=hhem, log_path="/invalid/path/results.jsonl")
        assert result is not None
        assert isinstance(result, RAGASResult)
