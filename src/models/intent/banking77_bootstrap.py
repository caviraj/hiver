"""Banking77 Transfer-Learning Bootstrap module.

Implements transfer-learning sequence classification fine-tuning on the Banking77
dataset to generate a bootstrap encoder checkpoint for subsequent domain adaptation.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from src.models.intent.config import TrainConfig
from src.models.intent.dataset import (
    count_truncated_queries,
    load_banking77,
    stratified_split,
    tokenize_dataset,
)

logger = logging.getLogger(__name__)


def check_hardware() -> str:
    """Check available compute hardware and log training duration warning if on CPU.

    Returns:
        "cuda" if GPU is available, otherwise "cpu".
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.warning(
            "No GPU detected. Training on CPU will take approximately 30-90 minutes. "
            "For rapid iterations, consider testing on a sample or running on GPU."
        )
    return device


class EarlyStoppingMonitor(EarlyStoppingCallback):
    """Early stopping callback that tracks the stopped epoch and logs warnings."""

    def __init__(
        self,
        early_stopping_patience: int = 2,
        early_stopping_threshold: float = 0.0,
    ) -> None:
        super().__init__(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
        self.stopped_epoch: int | None = None

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        super().on_evaluate(args, state, control, metrics=metrics, **kwargs)
        if control.should_training_stop:
            current_epoch = int(round(state.epoch)) if state.epoch is not None else 1
            self.stopped_epoch = max(1, current_epoch)
            if self.stopped_epoch == 1:
                logger.warning(
                    "Early stopping triggered on epoch 1. Model training ended unusually early; "
                    "check learning rate and data split."
                )


class WeightedTrainer(Trainer):
    """Custom Trainer supporting class-weighted CrossEntropyLoss."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[1]

        if self.class_weights is not None and labels is not None and logits is not None:
            device = logits.device
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(device))
            num_labels = getattr(model.config, "num_labels", logits.shape[-1])
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Trainer-compatible metric evaluation function computing accuracy and macro F1."""
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    }


def compute_detailed_metrics(
    labels: list[int] | np.ndarray,
    preds: list[int] | np.ndarray,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute overall accuracy, macro F1, and top 5 worst performing classes by F1.

    Args:
        labels: True integer labels.
        preds: Predicted integer labels.
        label_names: Optional human-readable class names.

    Returns:
        Dict with keys: 'accuracy', 'macro_f1', 'top_5_worst_classes'.
    """
    labels_arr = np.asarray(labels)
    preds_arr = np.asarray(preds)

    acc = float(accuracy_score(labels_arr, preds_arr))
    macro_f1 = float(f1_score(labels_arr, preds_arr, average="macro", zero_division=0))

    if label_names is not None:
        classes = list(range(len(label_names)))
    else:
        classes = sorted(list(set(labels_arr) | set(preds_arr)))

    precision, recall, f1, support = precision_recall_fscore_support(
        labels_arr,
        preds_arr,
        labels=classes,
        average=None,
        zero_division=0,
    )

    class_stats = []
    for idx, c in enumerate(classes):
        c_name = label_names[c] if label_names and c < len(label_names) else str(c)
        class_stats.append({
            "class_id": int(c),
            "class_name": str(c_name),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx]),
        })

    # Sort ascending by f1 (worst first), then descending by support, then ascending by class_id
    class_stats.sort(key=lambda x: (x["f1"], -x["support"], x["class_id"]))
    top_5_worst = class_stats[:5]

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "top_5_worst_classes": top_5_worst,
    }


def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    path: str | Path,
    overwrite: bool = False,
) -> None:
    """Save model and tokenizer with overwrite protection.

    Args:
        model: Trained PreTrainedModel instance.
        tokenizer: PreTrainedTokenizerBase instance.
        path: Target checkpoint directory path.
        overwrite: If False, raises FileExistsError if path is existing and non-empty.

    Raises:
        FileExistsError: If destination directory already exists and contains files.
    """
    target_path = Path(path)
    if target_path.exists() and any(target_path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Checkpoint directory '{path}' already exists and is non-empty. Set overwrite=True to replace it."
        )

    target_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target_path)
    tokenizer.save_pretrained(target_path)
    logger.info(f"Checkpoint successfully saved to '{target_path}'.")


def load_pretrained_encoder(
    path: str | Path,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a fine-tuned sequence classification model and tokenizer from disk.

    Args:
        path: Path to checkpoint directory.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        FileNotFoundError: If path does not exist.
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path '{ckpt_path}' does not exist.")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    model.eval()
    return model, tokenizer


def train(
    config: TrainConfig,
    dataset_dict: DatasetDict | None = None,
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """Execute transfer learning bootstrap fine-tuning on Banking77.

    Args:
        config: Training configuration options.
        dataset_dict: Optional pre-loaded DatasetDict.
        model: Optional pre-instantiated model.
        tokenizer: Optional pre-instantiated tokenizer.

    Returns:
        Evaluation metrics dictionary containing accuracy, macro_f1, stopped_epoch, and top_5_worst_classes.
    """
    # 1. Upfront overwrite check
    ckpt_dir = Path(config.checkpoint_dir)
    if ckpt_dir.exists() and any(ckpt_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(
            f"Checkpoint directory '{config.checkpoint_dir}' already exists and is non-empty. "
            "Set overwrite=True to replace it."
        )

    # 2. Hardware check
    device = check_hardware()

    # 3. Load tokenizer and dataset
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if dataset_dict is None:
        dataset_dict = load_banking77()

    # 4. Prepare train / validation splits
    if "validation" in dataset_dict:
        train_raw = dataset_dict["train"]
        val_raw = dataset_dict["validation"]
    elif "val" in dataset_dict:
        train_raw = dataset_dict["train"]
        val_raw = dataset_dict["val"]
    elif "train" in dataset_dict:
        split_dict = stratified_split(
            dataset_dict["train"],
            val_ratio=config.val_ratio,
            seed=config.seed,
        )
        train_raw, val_raw = split_dict["train"], split_dict["val"]
    else:
        raise ValueError("DatasetDict must contain at least a 'train' split.")

    # 5. Tokenize
    train_dataset, _ = tokenize_dataset(train_raw, tokenizer, max_length=config.max_length)
    val_dataset, _ = tokenize_dataset(val_raw, tokenizer, max_length=config.max_length)

    # Truncation check
    trunc_count = count_truncated_queries(train_raw, tokenizer, max_length=config.max_length)
    if trunc_count > 0:
        logger.warning(
            f"Truncation detected: {trunc_count}/{len(train_raw)} "
            f"({(trunc_count / len(train_raw)) * 100.0:.2f}%) queries exceed max_length={config.max_length}"
        )

    # 6. Extract label metadata
    label_names = None
    if "label" in train_raw.features and hasattr(train_raw.features["label"], "names"):
        label_names = train_raw.features["label"].names
        num_labels = len(label_names)
    else:
        num_labels = config.num_labels
        label_names = [f"intent_{i}" for i in range(num_labels)]

    # 7. Model instantiation
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=num_labels,
        )

    # 8. Compute class weights if requested
    class_weights_tensor = None
    if config.class_weighting:
        train_labels = np.array(train_dataset["label"])
        unique_classes = np.unique(train_labels)
        cw = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=train_labels,
        )
        full_weights = np.ones(num_labels, dtype=np.float32)
        for cls_idx, w in zip(unique_classes, cw):
            full_weights[cls_idx] = w
        class_weights_tensor = torch.tensor(full_weights, dtype=torch.float32)

    # 9. Setup TrainingArguments & EarlyStopping
    early_stopping = EarlyStoppingMonitor(
        early_stopping_patience=config.patience,
        early_stopping_threshold=0.0,
    )

    with tempfile.TemporaryDirectory() as temp_output_dir:
        training_args = TrainingArguments(
            output_dir=temp_output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.eval_batch_size,
            num_train_epochs=config.epochs,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_strategy="epoch",
            save_total_limit=1,
            report_to="none",
            use_cpu=(device == "cpu"),
            seed=config.seed,
        )

        trainer_cls = WeightedTrainer if class_weights_tensor is not None else Trainer
        extra_kwargs = {"class_weights": class_weights_tensor} if class_weights_tensor is not None else {}

        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping],
            **extra_kwargs,
        )

        # 10. Execute Training
        trainer.train()

        # 11. Determine stopped epoch
        if early_stopping.stopped_epoch is not None:
            stopped_epoch = early_stopping.stopped_epoch
        else:
            stopped_epoch = int(round(trainer.state.epoch)) if trainer.state.epoch is not None else config.epochs

        # 12. Evaluate best model on validation set
        eval_output = trainer.predict(val_dataset)
        preds = np.argmax(eval_output.predictions, axis=1)
        detailed_metrics = compute_detailed_metrics(
            labels=eval_output.label_ids,
            preds=preds,
            label_names=label_names,
        )
        detailed_metrics["stopped_epoch"] = int(stopped_epoch)

        # 13. Persist checkpoint and metrics
        save_checkpoint(
            model=trainer.model,
            tokenizer=tokenizer,
            path=config.checkpoint_dir,
            overwrite=config.overwrite,
        )

        metrics_file = Path(config.metrics_path)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(detailed_metrics, f, indent=2)
        logger.info(f"Metrics saved to '{config.metrics_path}'.")

    return detailed_metrics
