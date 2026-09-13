"""Tests for GoldenExample schema, dataset loading, and stratified sampling."""

import json
import logging
from pathlib import Path
from typing import List

import pytest
from pydantic import ValidationError

from src.evaluation.golden_dataset import (
    GoldenExample,
    load_golden_set,
    stratified_sample,
)


def _generate_synthetic_examples(n: int) -> List[dict]:
    """Generate n synthetic golden example dictionaries."""
    return [
        {
            "example_id": f"ex_{i:04d}",
            "query": f"Test question {i}?",
            "retrieved_contexts": [f"Context passage for {i}"],
            "generated_response": f"Generated answer {i}",
            "human_score": (i % 5) + 1,
            "annotator_id": f"ann_{i % 3}",
            "rubric_name": "brand_tone",
            "metadata": {"intent": "billing" if i % 2 == 0 else "technical"},
        }
        for i in range(n)
    ]


class TestGoldenExampleSchema:
    """Validation tests for the GoldenExample model."""

    def test_valid_golden_example(self):
        item = GoldenExample(
            example_id="ex_001",
            query="Where do I see invoices?",
            retrieved_contexts=["Invoice section is under settings."],
            generated_response="You can view invoices under settings.",
            human_score=4,
            annotator_id="annotator_1",
            rubric_name="brand_tone",
            metadata={"severity": "low"},
        )
        assert item.example_id == "ex_001"
        assert item.human_score == 4

    def test_optional_human_score_none(self):
        item = GoldenExample(
            example_id="ex_002",
            query="How do I reset password?",
            generated_response="Click forgot password.",
            human_score=None,
            annotator_id="annotator_1",
            rubric_name="empathy",
        )
        assert item.human_score is None

    @pytest.mark.parametrize("invalid_score", [0, 6, -1, 10])
    def test_invalid_human_score_range_raises(self, invalid_score: int):
        with pytest.raises(ValidationError):
            GoldenExample(
                example_id="ex_003",
                query="Query",
                generated_response="Response",
                human_score=invalid_score,
                annotator_id="ann_1",
                rubric_name="brand_tone",
            )


class TestLoadGoldenSet:
    """Tests for load_golden_set file reading and statistical size enforcement."""

    def test_fewer_than_150_examples_raises_value_error(self, tmp_path: Path):
        """Hard architectural requirement: < 150 examples must raise a ValueError."""
        items = _generate_synthetic_examples(149)
        dataset_file = tmp_path / "golden_149.json"
        with open(dataset_file, "w", encoding="utf-8") as f:
            json.dump(items, f)

        with pytest.raises(ValueError, match="statistically insufficient.*150 examples"):
            load_golden_set(dataset_file)

    def test_exactly_150_examples_succeeds(self, tmp_path: Path):
        """Baseline boundary: exactly 150 examples succeeds without error."""
        items = _generate_synthetic_examples(150)
        dataset_file = tmp_path / "golden_150.json"
        with open(dataset_file, "w", encoding="utf-8") as f:
            json.dump(items, f)

        loaded = load_golden_set(dataset_file)
        assert len(loaded) == 150
        assert isinstance(loaded[0], GoldenExample)

    def test_over_250_examples_warns_but_succeeds(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        """Over 250 examples emits an over-provisioning warning but loads successfully."""
        items = _generate_synthetic_examples(251)
        dataset_file = tmp_path / "golden_251.jsonl"
        with open(dataset_file, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        with caplog.at_level(logging.WARNING):
            loaded = load_golden_set(dataset_file)

        assert len(loaded) == 251
        assert any("251 examples" in rec.message and "Over-provisioning" in rec.message for rec in caplog.records)

    def test_load_json_with_dict_wrapper(self, tmp_path: Path):
        """Supports JSON files structured as {'examples': [...]}."""
        items = _generate_synthetic_examples(155)
        dataset_file = tmp_path / "golden_wrapped.json"
        with open(dataset_file, "w", encoding="utf-8") as f:
            json.dump({"examples": items}, f)

        loaded = load_golden_set(dataset_file)
        assert len(loaded) == 155

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_golden_set(tmp_path / "non_existent.json")


class TestStratifiedSample:
    """Tests for stratified_sample proportion preservation and edge cases."""

    def test_stratified_proportions(self):
        """Validates that sample preserves the proportion across strata."""
        pool: List[GoldenExample] = []
        # 80 billing items, 20 technical items (80/20 split)
        for i in range(80):
            pool.append(
                GoldenExample(
                    example_id=f"b_{i}",
                    query=f"Billing Q {i}",
                    generated_response="Ans",
                    annotator_id="ann",
                    rubric_name="rubric",
                    metadata={"intent": "billing"},
                )
            )
        for i in range(20):
            pool.append(
                GoldenExample(
                    example_id=f"t_{i}",
                    query=f"Tech Q {i}",
                    generated_response="Ans",
                    annotator_id="ann",
                    rubric_name="rubric",
                    metadata={"intent": "technical"},
                )
            )

        sampled = stratified_sample(pool, target_size=20, stratify_by="intent", seed=42)
        assert len(sampled) == 20

        billing_count = sum(1 for x in sampled if x.metadata["intent"] == "billing")
        tech_count = sum(1 for x in sampled if x.metadata["intent"] == "technical")

        # 80% of 20 = 16, 20% of 20 = 4
        assert billing_count == 16
        assert tech_count == 4

    def test_stratum_with_zero_examples_logs_warning(self, caplog: pytest.LogCaptureFixture):
        """Empty stratum condition logs a warning and does not crash."""
        pool = [
            GoldenExample(
                example_id=f"ex_{i}",
                query="Query",
                generated_response="Resp",
                annotator_id="ann",
                rubric_name="rubric",
                metadata={"tier": "tier_1"},
            )
            for i in range(10)
        ]

        with caplog.at_level(logging.WARNING):
            sampled = stratified_sample(pool, target_size=5, stratify_by="tier", seed=42)

        assert len(sampled) == 5

    def test_edge_cases_empty_or_oversized(self):
        pool = [
            GoldenExample(
                example_id=f"ex_{i}",
                query="Query",
                generated_response="Resp",
                annotator_id="ann",
                rubric_name="rubric",
                metadata={"category": "general"},
            )
            for i in range(5)
        ]

        # Target size 0
        assert stratified_sample(pool, target_size=0, stratify_by="category") == []
        # Target size > pool size
        assert len(stratified_sample(pool, target_size=10, stratify_by="category")) == 5
        # Empty pool
        assert stratified_sample([], target_size=5, stratify_by="category") == []
