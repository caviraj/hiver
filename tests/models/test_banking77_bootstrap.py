"""Unit tests for src.models.intent.banking77_bootstrap.

All tests execute 100% offline without downloading Hugging Face Hub weights
or datasets, using fast synthetic models, tokenizers, and datasets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    PreTrainedTokenizerFast,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.models.intent.banking77_bootstrap import (
    EarlyStoppingMonitor,
    WeightedTrainer,
    check_hardware,
    compute_detailed_metrics,
    compute_metrics,
    load_pretrained_encoder,
    save_checkpoint,
    train,
)
from src.models.intent.config import TrainConfig


@pytest.fixture
def dummy_tokenizer() -> PreTrainedTokenizerFast:
    """Create a fast, completely offline dummy tokenizer."""
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "hello": 2,
        "world": 3,
        "foo": 4,
        "bar": 5,
        "transfer": 6,
        "payment": 7,
        "fee": 8,
        "card": 9,
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
    """Create a minimal 1-layer DistilBERT model for ultra-fast offline testing."""
    cfg = DistilBertConfig(
        vocab_size=10,
        dim=16,
        n_layers=1,
        n_heads=2,
        hidden_dim=32,
        num_labels=3,
    )
    return DistilBertForSequenceClassification(cfg)


@pytest.fixture
def dummy_dataset_dict() -> DatasetDict:
    """Create a minimal synthetic DatasetDict with 3 classes."""
    train_data = {
        "text": ["hello world", "foo bar", "transfer payment", "card fee", "hello foo", "bar world"] * 3,
        "label": [0, 1, 2, 0, 1, 2] * 3,
    }
    val_data = {
        "text": ["hello world", "foo bar", "transfer payment"],
        "label": [0, 1, 2],
    }
    return DatasetDict({
        "train": Dataset.from_dict(train_data),
        "val": Dataset.from_dict(val_data),
    })


def test_check_hardware_cpu_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Verify check_hardware returns 'cpu' and logs the exact required warning when CUDA is absent."""
    with patch("torch.cuda.is_available", return_value=False):
        with caplog.at_level(logging.WARNING):
            device = check_hardware()
            assert device == "cpu"
            assert (
                "No GPU detected. Training on CPU will take approximately 30-90 minutes. "
                "For rapid iterations, consider testing on a sample or running on GPU."
            ) in caplog.text


def test_check_hardware_gpu() -> None:
    """Verify check_hardware returns 'cuda' and logs GPU name when CUDA is available."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
        device = check_hardware()
        assert device == "cuda"


def test_compute_metrics() -> None:
    """Verify compute_metrics calculates accuracy and macro F1 accurately."""
    # Class 0: 2 items, Class 1: 2 items
    # True: [0, 0, 1, 1]
    # Preds logits: argmax gives [0, 1, 1, 1] -> 3/4 correct
    logits = np.array([
        [2.0, 0.5],
        [0.2, 1.8],
        [0.1, 2.5],
        [0.3, 1.2],
    ])
    labels = np.array([0, 0, 1, 1])
    res = compute_metrics((logits, labels))

    assert "accuracy" in res
    assert "macro_f1" in res
    assert res["accuracy"] == pytest.approx(0.75)
    # Class 0: prec=1/1=1.0, rec=1/2=0.5, f1=2/3 ~ 0.6667
    # Class 1: prec=2/3 ~ 0.6667, rec=2/2=1.0, f1=4/5 = 0.8
    # Macro F1 = (2/3 + 0.8) / 2 = 0.7333
    assert res["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2, rel=1e-3)

    # Test tuple predictions format (logits, hidden_states)
    res_tuple = compute_metrics(((logits, None), labels))
    assert res_tuple["accuracy"] == pytest.approx(0.75)


def test_compute_detailed_metrics_sorting_and_tie_breaking() -> None:
    """Verify compute_detailed_metrics outputs schema and sorts top 5 worst classes correctly.

    Sorting order:
    1. Ascending f1 (worst first)
    2. Descending support (higher support first if tied on f1)
    3. Ascending class_id (lower class_id first if tied on f1 and support)
    """
    # 6 classes: 0, 1, 2, 3, 4, 5
    # Class 0: perfect f1 (support=10)
    # Class 1: f1=0.0, support=10 (should come before Class 2)
    # Class 2: f1=0.0, support=5  (should come after Class 1)
    # Class 3: f1=0.0, support=5  (tied with Class 2 on support -> sorted by class_id ascending: 2 then 3)
    # Class 4: f1=0.5, support=4
    # Class 5: f1=0.8, support=8
    labels = (
        [0] * 10
        + [1] * 10
        + [2] * 5
        + [3] * 5
        + [4] * 4
        + [5] * 8
    )
    # Make predictions:
    # Class 0: all correct (10) -> Precision=1.0, Recall=1.0, F1=1.0
    # Class 1: all predicted as 2 (f1=0, support=10)
    # Class 2: all predicted as 1 (f1=0, support=5)
    # Class 3: all predicted as 1 (f1=0, support=5)
    # Class 4: 2 correct, 2 as 1 (f1 ≈ 0.67, support=4)
    # Class 5: 7 correct, 1 as 1 (f1 ≈ 0.82, support=8)
    preds = (
        [0] * 10
        + [2] * 10
        + [1] * 5
        + [1] * 5
        + [4, 4, 1, 1]
        + [5] * 7 + [1]
    )
    label_names = ["c0", "c1", "c2", "c3", "c4", "c5"]

    detailed = compute_detailed_metrics(labels, preds, label_names=label_names)

    assert "accuracy" in detailed
    assert "macro_f1" in detailed
    assert "top_5_worst_classes" in detailed

    worst = detailed["top_5_worst_classes"]
    assert len(worst) == 5

    # Check the top worst classes order:
    # 1st: class_id 1 (f1=0.0, support=10)
    assert worst[0]["class_id"] == 1
    assert worst[0]["support"] == 10
    assert worst[0]["f1"] == 0.0

    # 2nd: class_id 2 (f1=0.0, support=5, id 2 < 3)
    assert worst[1]["class_id"] == 2
    assert worst[1]["support"] == 5
    assert worst[1]["f1"] == 0.0

    # 3rd: class_id 3 (f1=0.0, support=5, id 3 > 2)
    assert worst[2]["class_id"] == 3
    assert worst[2]["support"] == 5
    assert worst[2]["f1"] == 0.0

    # 4th: class_id 4
    assert worst[3]["class_id"] == 4

    # 5th: class_id 5
    assert worst[4]["class_id"] == 5


def test_save_and_load_pretrained_encoder_roundtrip(
    dummy_model: DistilBertForSequenceClassification,
    dummy_tokenizer: PreTrainedTokenizerFast,
    tmp_path: Path,
) -> None:
    """Verify that saving and loading an encoder preserves architecture and produces identical logits."""
    dummy_model.eval()
    save_path = tmp_path / "saved_encoder"
    save_checkpoint(dummy_model, dummy_tokenizer, save_path, overwrite=False)

    assert save_path.exists()
    assert (save_path / "config.json").exists()

    loaded_model, loaded_tokenizer = load_pretrained_encoder(save_path)
    loaded_model.eval()

    # Evaluate logits on same dummy input
    input_ids = torch.tensor([[2, 3, 4]])
    with torch.no_grad():
        orig_logits = dummy_model(input_ids).logits
        loaded_logits = loaded_model(input_ids).logits

    assert torch.allclose(orig_logits, loaded_logits, atol=1e-5)


def test_save_checkpoint_overwrite_protection(
    dummy_model: DistilBertForSequenceClassification,
    dummy_tokenizer: PreTrainedTokenizerFast,
    tmp_path: Path,
) -> None:
    """Verify that save_checkpoint protects against accidental overwrites unless overwrite=True."""
    save_path = tmp_path / "protected_ckpt"
    save_checkpoint(dummy_model, dummy_tokenizer, save_path, overwrite=False)

    # Attempt saving again without overwrite flag
    with pytest.raises(FileExistsError, match="already exists and is non-empty"):
        save_checkpoint(dummy_model, dummy_tokenizer, save_path, overwrite=False)

    # Saving with overwrite=True should succeed
    save_checkpoint(dummy_model, dummy_tokenizer, save_path, overwrite=True)


def test_load_pretrained_encoder_not_found(tmp_path: Path) -> None:
    """Verify load_pretrained_encoder raises FileNotFoundError on non-existent path."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_pretrained_encoder(tmp_path / "non_existent")


def test_early_stopping_monitor_epoch_1_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Verify EarlyStoppingMonitor logs the exact required warning when triggered on epoch 1."""
    monitor = EarlyStoppingMonitor(early_stopping_patience=1)
    args = MagicMock()
    state = TrainerState()
    state.epoch = 1.0
    control = TrainerControl()
    control.should_training_stop = True

    with caplog.at_level(logging.WARNING):
        monitor.on_evaluate(args, state, control, metrics={"macro_f1": 0.5})
        assert monitor.stopped_epoch == 1
        assert (
            "Early stopping triggered on epoch 1. Model training ended unusually early; "
            "check learning rate and data split."
        ) in caplog.text


def test_weighted_trainer_loss_computation(
    dummy_model: DistilBertForSequenceClassification,
    tmp_path: Path,
) -> None:
    """Verify WeightedTrainer applies class weights to CrossEntropyLoss."""
    weights = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32)
    args = TrainingArguments(output_dir=str(tmp_path / "trainer_out"), report_to="none")
    trainer = WeightedTrainer(
        model=dummy_model,
        args=args,
        class_weights=weights,
    )

    inputs = {
        "input_ids": torch.tensor([[2, 3], [4, 5]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }

    loss = trainer.compute_loss(dummy_model, inputs)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0


def test_train_pipeline_end_to_end_fast(
    dummy_model: DistilBertForSequenceClassification,
    dummy_tokenizer: PreTrainedTokenizerFast,
    dummy_dataset_dict: DatasetDict,
    tmp_path: Path,
) -> None:
    """Execute complete end-to-end training pipeline offline in under 2 seconds."""
    ckpt_dir = tmp_path / "test_checkpoint"
    metrics_file = tmp_path / "test_metrics.json"

    config = TrainConfig(
        model_name="distilbert-base-uncased",
        num_labels=3,
        max_length=16,
        batch_size=4,
        eval_batch_size=4,
        epochs=1,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_ratio=0.0,
        patience=1,
        val_ratio=0.2,
        seed=42,
        class_weighting=True,
        checkpoint_dir=str(ckpt_dir),
        metrics_path=str(metrics_file),
        overwrite=True,
    )

    metrics = train(
        config=config,
        dataset_dict=dummy_dataset_dict,
        model=dummy_model,
        tokenizer=dummy_tokenizer,
    )

    # 1. Assert return dictionary contents
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "stopped_epoch" in metrics
    assert "top_5_worst_classes" in metrics
    assert isinstance(metrics["stopped_epoch"], int)
    assert isinstance(metrics["top_5_worst_classes"], list)

    # 2. Assert metrics file written correctly
    assert metrics_file.exists()
    with open(metrics_file, "r", encoding="utf-8") as f:
        saved_metrics = json.load(f)
    assert saved_metrics["accuracy"] == metrics["accuracy"]
    assert saved_metrics["macro_f1"] == metrics["macro_f1"]
    assert saved_metrics["stopped_epoch"] == metrics["stopped_epoch"]

    # 3. Assert checkpoint files created
    assert ckpt_dir.exists()
    assert (ckpt_dir / "config.json").exists()

    # 4. Assert checkpoint can be reloaded via load_pretrained_encoder
    loaded_model, loaded_tokenizer = load_pretrained_encoder(ckpt_dir)
    assert loaded_model.config.num_labels == 3


def test_train_pipeline_overwrite_protection(
    dummy_model: DistilBertForSequenceClassification,
    dummy_tokenizer: PreTrainedTokenizerFast,
    dummy_dataset_dict: DatasetDict,
    tmp_path: Path,
) -> None:
    """Verify train raises FileExistsError upfront when checkpoint_dir exists and overwrite=False."""
    ckpt_dir = tmp_path / "existing_checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "some_file.txt").write_text("already here", encoding="utf-8")

    config = TrainConfig(
        checkpoint_dir=str(ckpt_dir),
        overwrite=False,
    )

    with pytest.raises(FileExistsError, match="already exists and is non-empty"):
        train(
            config=config,
            dataset_dict=dummy_dataset_dict,
            model=dummy_model,
            tokenizer=dummy_tokenizer,
        )
