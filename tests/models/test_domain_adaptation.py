"""Unit tests for src.models.intent.domain_adaptation and domain_adapt_dataset.

All tests run completely hermetic without network requests or external dependencies,
using synthetic tokenizers, lightweight 1-layer DistilBERT models, and mock datasets.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List
import numpy as np
import pytest
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    PreTrainedTokenizerFast,
)

from src.models.intent.config import TaxonomyConfig
from src.models.intent.domain_adapt_dataset import (
    assemble_domain_dataset,
    calculate_cluster_purity,
    filter_by_purity,
    merge_sparse_clusters,
    stratified_three_way_split,
    validate_surviving_classes,
)
from src.models.intent.domain_adaptation import (
    DomainAdaptResult,
    evaluate_zero_shot_baseline,
    fine_tune_domain_model,
    swap_classification_head,
)


@pytest.fixture
def dummy_tokenizer() -> PreTrainedTokenizerFast:
    """Create a minimal in-memory tokenizer."""
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "hello": 2,
        "world": 3,
        "battery": 4,
        "screen": 5,
        "charge": 6,
        "account": 7,
        "update": 8,
        "broken": 9,
    }
    raw_tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    raw_tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=raw_tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
    )


@pytest.fixture
def dummy_model() -> DistilBertForSequenceClassification:
    """Create a minimal 1-layer DistilBERT model for testing."""
    cfg = DistilBertConfig(
        vocab_size=10,
        dim=16,
        n_layers=1,
        n_heads=2,
        hidden_dim=32,
        num_labels=5,
    )
    return DistilBertForSequenceClassification(cfg)


def test_calculate_cluster_purity() -> None:
    """Test calculate_cluster_purity with empty and populated lists."""
    assert calculate_cluster_purity([]) == 0.0
    assert pytest.approx(calculate_cluster_purity([0.8, 0.9, 1.0]), 1e-5) == 0.9


def test_filter_by_purity() -> None:
    """Test filter_by_purity drops noise (-1) and low purity records."""
    records = ["text_0", "text_1", "text_2", "text_3", "text_4"]
    labels = [0, -1, 1, 0, 1]
    probabilities = [0.85, 0.99, 0.40, 0.75, 0.90]

    filt_rec, filt_lbl, filt_prob = filter_by_purity(
        records, labels, probabilities, purity_threshold=0.70
    )

    # text_1 is dropped (label == -1)
    # text_2 is dropped (prob 0.40 < 0.70)
    # text_0, text_3, text_4 survive
    assert filt_rec == ["text_0", "text_3", "text_4"]
    assert filt_lbl == [0, 0, 1]
    assert filt_prob == [0.85, 0.75, 0.90]

    # Test length mismatch raises ValueError
    with pytest.raises(ValueError, match="Length mismatch"):
        filter_by_purity(["a"], [0, 1], [0.9])


def test_merge_sparse_clusters() -> None:
    """Test merge_sparse_clusters renames classes below min_cluster_size."""
    records = [f"rec_{i}" for i in range(12)]
    # class 'A' has 6 items, class 'B' has 4 items, class 'C' has 2 items
    labels = ["A"] * 6 + ["B"] * 4 + ["C"] * 2

    merged_rec, merged_lbl = merge_sparse_clusters(
        records, labels, min_cluster_size=5, other_label="Other_Uncategorized"
    )

    assert len(merged_rec) == 12
    # 'A' has 6 >= 5, so retained
    assert merged_lbl[:6] == ["A"] * 6
    # 'B' (4) and 'C' (2) are < 5, so merged into 'Other_Uncategorized'
    assert merged_lbl[6:] == ["Other_Uncategorized"] * 6


def test_validate_surviving_classes() -> None:
    """Test validate_surviving_classes passes with >= 3 classes and fails with < 3."""
    validate_surviving_classes(["ClassA", "ClassB", "ClassC"], min_classes=3)

    with pytest.raises(ValueError, match="Too few distinct classes"):
        validate_surviving_classes(["ClassA", "ClassB", "ClassA"], min_classes=3)


def test_stratified_three_way_split() -> None:
    """Test stratified_three_way_split partitions data cleanly."""
    records = [f"text_{i}" for i in range(30)]
    labels = ["A"] * 10 + ["B"] * 10 + ["C"] * 9 + ["D"] * 1  # 'D' is singleton

    train_idx, val_idx, test_idx = stratified_three_way_split(
        records, labels, val_ratio=0.2, test_ratio=0.2, seed=42
    )

    all_indices = set(train_idx) | set(val_idx) | set(test_idx)
    assert len(all_indices) == 30
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0

    # Singleton 'D' at index 29 must be assigned to train
    assert 29 in train_idx

    # Invalid ratio raises ValueError
    with pytest.raises(ValueError, match="Sum of val_ratio and test_ratio"):
        stratified_three_way_split(records, labels, val_ratio=0.6, test_ratio=0.5)


def test_assemble_domain_dataset(dummy_tokenizer: PreTrainedTokenizerFast) -> None:
    """Test assemble_domain_dataset builds tokenized DatasetDict and label maps."""
    records = ["hello world", "battery charge", "account update"] * 5
    labels = ["Intent_A", "Intent_B", "Intent_C"] * 5

    dataset_dict, label2id, id2label = assemble_domain_dataset(
        records, labels, tokenizer=dummy_tokenizer, max_length=16, val_ratio=0.2, test_ratio=0.2
    )

    assert isinstance(dataset_dict, DatasetDict)
    assert "train" in dataset_dict
    assert "validation" in dataset_dict
    assert "test" in dataset_dict

    assert "input_ids" in dataset_dict["train"].column_names
    assert "attention_mask" in dataset_dict["train"].column_names
    assert "label" in dataset_dict["train"].column_names

    assert len(label2id) == 3
    assert set(label2id.keys()) == {"Intent_A", "Intent_B", "Intent_C"}
    for lbl, idx in label2id.items():
        assert id2label[idx] == lbl


def test_swap_classification_head(dummy_model: DistilBertForSequenceClassification) -> None:
    """Test swap_classification_head updates head dim and preserves backbone weights."""
    original_backbone_state = copy.deepcopy(dummy_model.distilbert.state_dict())
    original_classifier_weight = dummy_model.classifier.weight.clone()

    adapted_model = swap_classification_head(
        dummy_model, num_new_labels=7, freeze_backbone=True
    )

    assert adapted_model.classifier.out_features == 7
    assert adapted_model.num_labels == 7

    # Verify backbone weights are identical
    new_backbone_state = adapted_model.distilbert.state_dict()
    for key in original_backbone_state:
        assert torch.equal(original_backbone_state[key], new_backbone_state[key])

    # Verify backbone parameters are frozen
    for param in adapted_model.distilbert.parameters():
        assert not param.requires_grad

    # Verify classifier parameter requires grad
    for param in adapted_model.classifier.parameters():
        assert param.requires_grad


def test_evaluate_zero_shot_baseline(
    dummy_model: DistilBertForSequenceClassification,
    dummy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Test evaluate_zero_shot_baseline calculates metrics on dummy dataset."""
    eval_data = {
        "text": ["battery broken", "screen hello", "account charge"],
        "label": [0, 1, 2],
    }
    eval_dataset = Dataset.from_dict(eval_data)
    id2label = {0: "Battery_Issue", 1: "Screen_Damage", 2: "Account_Billing"}

    metrics = evaluate_zero_shot_baseline(
        model=dummy_model,
        tokenizer=dummy_tokenizer,
        eval_dataset=eval_dataset,
        id2label_target=id2label,
        reference_descriptions={
            "Battery_Issue": "battery power charge",
            "Screen_Damage": "screen glass display",
            "Account_Billing": "account subscription money",
        },
    )

    assert "macro_f1" in metrics
    assert "accuracy" in metrics
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_fine_tune_domain_model_file_exists_error(tmp_path: Path) -> None:
    """Test fine_tune_domain_model raises FileExistsError if output dir exists and overwrite=False."""
    out_dir = tmp_path / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = TaxonomyConfig()
    config.training.output_dir = str(out_dir)
    config.training.overwrite = False

    dataset_dict = DatasetDict({
        "train": Dataset.from_dict({"text": ["a"], "label": [0]}),
        "validation": Dataset.from_dict({"text": ["b"], "label": [0]}),
        "test": Dataset.from_dict({"text": ["c"], "label": [0]}),
    })

    with pytest.raises(FileExistsError, match="Output directory already exists"):
        fine_tune_domain_model(
            config=config,
            dataset_dict=dataset_dict,
            label2id={"ClassA": 0},
            id2label={0: "ClassA"},
        )


def test_fine_tune_domain_model_synthetic(
    tmp_path: Path,
    dummy_model: DistilBertForSequenceClassification,
    dummy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Test fine_tune_domain_model executes training loop with dummy model."""
    out_dir = tmp_path / "adapted_model"
    config = TaxonomyConfig()
    config.training.output_dir = str(out_dir)
    config.training.overwrite = True
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.training.learning_rate = 1e-3
    config.training.max_length = 16
    config.training.freeze_backbone = True

    # Build 3 classes with 4 samples each
    records = ["hello battery", "world screen", "account charge"] * 4
    labels = ["ClassA", "ClassB", "ClassC"] * 4

    dataset_dict, label2id, id2label = assemble_domain_dataset(
        records=records,
        labels=labels,
        tokenizer=dummy_tokenizer,
        max_length=16,
        val_ratio=0.25,
        test_ratio=0.25,
    )

    result = fine_tune_domain_model(
        config=config,
        dataset_dict=dataset_dict,
        label2id=label2id,
        id2label=id2label,
        base_model=dummy_model,
        tokenizer=dummy_tokenizer,
        compute_baseline=True,
    )

    assert isinstance(result, DomainAdaptResult)
    assert result.model_path == str(out_dir)
    assert "macro_f1" in result.adapted_metrics
    assert "macro_f1" in result.baseline_metrics
    assert isinstance(result.improvement_delta, float)

    # Check files were saved to out_dir
    assert (out_dir / "config.json").exists()
    assert (out_dir / "taxonomy_label2id.json").exists()
    assert (out_dir / "adaptation_metrics.json").exists()
