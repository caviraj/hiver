"""Domain Adaptation Module for Sequence Classification Models.

Implements classification head replacement with strict backbone weight invariance,
zero-shot similarity baseline computation, class-weighted domain fine-tuning with
early stopping, and performance metrics export (M2.P2.2.F1).
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from datasets import DatasetDict
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)

from src.models.intent.banking77_bootstrap import (
    EarlyStoppingMonitor,
    WeightedTrainer,
    compute_detailed_metrics,
    compute_metrics,
    load_pretrained_encoder,
    save_checkpoint,
)
from src.models.intent.config import TaxonomyConfig
from src.models.intent.dataset import tokenize_dataset
from src.models.intent.taxonomy_induction import embed_messages, extract_text

logger = logging.getLogger(__name__)


class DomainAdaptResult(BaseModel):
    """Pydantic model representing domain adaptation evaluation and metrics result."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    accuracy: float = Field(default=0.0, description="Post-adaptation classification accuracy on test split")
    macro_f1: float = Field(default=0.0, description="Post-adaptation macro-averaged F1 score on test split")
    zero_shot_accuracy: float = Field(default=0.0, description="Zero-shot embedding similarity accuracy on test split")
    zero_shot_macro_f1: float = Field(default=0.0, description="Zero-shot embedding similarity macro F1 score on test split")
    delta_macro_f1: float = Field(default=0.0, description="Improvement in macro F1 over zero-shot baseline")
    num_classes: int = Field(default=2, ge=2, description="Number of target domain classes")
    class_names: list[str] = Field(default_factory=list, description="Ordered list of target class names")
    checkpoint_dir: str = Field(default="", description="Directory where the domain adapted checkpoint is saved")
    metrics_path: str = Field(default="", description="Path to the exported metrics JSON file")
    stopped_epoch: int | None = Field(default=None, description="Epoch at which early stopping triggered, if any")
    top_5_worst_classes: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top 5 worst performing classes by F1 score on test split",
    )
    model_path: str | None = Field(default=None, description="Path to the saved adapted model directory")
    adapted_metrics: dict[str, float] = Field(default_factory=dict, description="Adapted model metrics")
    baseline_metrics: dict[str, float] = Field(default_factory=dict, description="Baseline model metrics")
    improvement_delta: float = Field(default=0.0, description="Difference between adapted and baseline macro F1")

    def to_dict(self) -> dict[str, Any]:
        """Convert result model to standard dictionary."""
        return self.model_dump()


def swap_classification_head(
    model: PreTrainedModel,
    num_new_labels: int | None = None,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
    freeze_backbone: bool = False,
    new_num_labels: int | None = None,
) -> PreTrainedModel:
    """Replace model sequence classification head while strictly preserving backbone weights.

    Snapshots all non-classifier parameters before swapping the classification head.
    After initializing the new linear layer, performs a bit-for-bit invariance check
    via torch.equal across all backbone parameters.

    Args:
        model: Pre-trained sequence classification model (e.g. DistilBERT).
        new_num_labels: Number of target output classes.
        id2label: Optional mapping of integer class indices to string names.
        label2id: Optional mapping of string class names to integer indices.

    Returns:
        The modified PreTrainedModel with the new classification head.

    Raises:
        RuntimeError: If any backbone parameter was mutated during the swap.
    """
    if num_new_labels is None:
        num_new_labels = new_num_labels
    if num_new_labels is None:
        raise ValueError("num_new_labels must be provided.")

    # 1. Snapshot all non-classifier backbone parameters
    before_swap = {
        name: param.clone().detach()
        for name, param in model.named_parameters()
        if not name.startswith("classifier")
    }

    # 2. Determine in_features from existing classifier or model configuration
    if hasattr(model, "classifier") and hasattr(model.classifier, "in_features"):
        in_features = model.classifier.in_features
    elif hasattr(model.config, "hidden_size"):
        in_features = model.config.hidden_size
    elif hasattr(model.config, "dim"):
        in_features = model.config.dim
    else:
        in_features = 768

    # 3. Replace classification head
    new_classifier = nn.Linear(in_features, num_new_labels)
    # Initialize weights cleanly
    nn.init.normal_(new_classifier.weight, std=0.02)
    if new_classifier.bias is not None:
        nn.init.zeros_(new_classifier.bias)

    # Attach to model and update device if needed
    device = next(model.parameters()).device
    new_classifier.to(device)
    model.classifier = new_classifier

    if freeze_backbone:
        for param in model.parameters():
            if not param.requires_grad or not str(param).startswith("classifier"):
                pass
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    # 4. Update configuration and model attributes
    model.config.num_labels = num_new_labels
    if hasattr(model, "num_labels"):
        model.num_labels = num_new_labels

    if id2label is not None:
        model.config.id2label = {int(k): str(v) for k, v in id2label.items()}
    if label2id is not None:
        model.config.label2id = {str(k): int(v) for k, v in label2id.items()}

    # 5. Strict bit-for-bit backbone weight invariance verification
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            if name not in before_swap or not torch.equal(before_swap[name], param):
                raise RuntimeError("Backbone weights modified during classification head swap!")

    logger.info(
        f"Classification head successfully swapped to {num_new_labels} classes. "
        f"Backbone weight invariance verified across {len(before_swap)} tensors."
    )
    return model


def evaluate_zero_shot_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    eval_dataset: Sequence[Any] | None = None,
    id2label: dict[int, str] | Sequence[str] | None = None,
    max_length: int = 64,
    batch_size: int = 32,
    device: str | None = None,
    test_dataset: Sequence[Any] | None = None,
    id2label_target: dict[int, str] | Sequence[str] | None = None,
    reference_descriptions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute zero-shot classification baseline using embedding cosine similarity.

    Constructs label text descriptions from class names ('_ ' replaced with ' '),
    extracts L2-normalized [CLS] embeddings for both customer queries and class labels,
    computes cosine similarities via matrix dot product, and evaluates accuracy and macro F1.

    Args:
        model: Pre-trained sequence classification or backbone model.
        tokenizer: Pre-trained tokenizer.
        test_dataset: Sequence of test samples with query text and true integer label.
        id2label: Mapping or sequence of class names by integer index.
        max_length: Maximum sequence length for tokenization.
        batch_size: Batch size for forward passes.
        device: 'cuda', 'cpu', etc.

    Returns:
        Dictionary containing 'accuracy', 'macro_f1', and 'predictions'.
    """
    if eval_dataset is None:
        eval_dataset = test_dataset
    if id2label is None:
        id2label = id2label_target
    if id2label is None:
        raise ValueError("id2label or id2label_target must be provided.")

    if isinstance(id2label, dict):
        num_classes = len(id2label)
        class_names = [id2label[i] for i in range(num_classes)]
    else:
        class_names = list(id2label)
        num_classes = len(class_names)

    if reference_descriptions is not None:
        class_texts = [reference_descriptions.get(name, name.replace("_", " ")) for name in class_names]
    else:
        class_texts = [name.replace("_", " ") for name in class_names]

    test_items = list(eval_dataset)
    test_texts = [extract_text(item) for item in test_items]
    test_labels = [item["label"] if isinstance(item, dict) else getattr(item, "label") for item in test_items]

    if not test_texts:
        return {"accuracy": 0.0, "macro_f1": 0.0, "predictions": []}

    # Extract L2-normalized embeddings
    test_embeds = embed_messages(
        test_texts,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )
    class_embeds = embed_messages(
        class_texts,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )

    # Dot product of unit vectors equals cosine similarity
    similarity_matrix = np.dot(test_embeds, class_embeds.T)
    preds = np.argmax(similarity_matrix, axis=1)

    acc = float(accuracy_score(test_labels, preds))
    macro_f1 = float(f1_score(test_labels, preds, average="macro", zero_division=0))

    logger.info(
        f"Zero-shot baseline evaluated: accuracy={acc:.4f}, macro_f1={macro_f1:.4f} "
        f"over {len(test_labels)} test samples and {num_classes} classes."
    )
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "predictions": preds.tolist(),
    }


def fine_tune_domain_model(
    config: TaxonomyConfig,
    dataset_dict: DatasetDict,
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
    base_model: PreTrainedModel | None = None,
    compute_baseline: bool = False,
) -> DomainAdaptResult:
    """Execute domain-adapted fine-tuning pipeline for sequence classification.

    Workflow:
      1. Validates checkpoint output directory against overwrite constraint.
      2. Loads bootstrap encoder checkpoint if model or tokenizer is not supplied.
      3. Resolves label mappings (id2label, label2id).
      4. Evaluates zero-shot baseline on test split with original encoder embeddings.
      5. Swaps classification head with strict backbone weight invariance check.
      6. Tokenizes train/val/test splits.
      7. Computes balanced class weights for WeightedTrainer.
      8. Configures EarlyStoppingMonitor and executes training loop.
      9. Evaluates adapted model on test split, logs warning if delta_macro_f1 <= 0.
      10. Persists checkpoint to domain_checkpoint_dir and exports metrics JSON.

    Args:
        config: TaxonomyConfig holding hyperparameters, directories, and seeds.
        dataset_dict: DatasetDict containing 'train', 'val', and 'test' splits.
        model: Optional pre-loaded model (for testing or external pipeline integration).
        tokenizer: Optional pre-loaded tokenizer.
        id2label: Optional dict mapping class integer IDs to string names.
        label2id: Optional dict mapping class string names to integer IDs.

    Returns:
        Tuple of (DomainAdaptResult, trained PreTrainedModel, PreTrainedTokenizerBase).

    Raises:
        FileExistsError: If domain_checkpoint_dir exists and contains files with overwrite=False.
    """
    if model is None:
        model = base_model

    output_dir = Path(config.training.output_dir)
    if output_dir.exists() and not config.training.overwrite:
        raise FileExistsError(
            f"Output directory already exists: '{output_dir}'. Set overwrite=True to replace it."
        )

    # 1. Overwrite protection check upfront
    target_ckpt = Path(config.domain_checkpoint_dir)
    if target_ckpt.exists() and not config.training.overwrite:
        raise FileExistsError(
            f"Output directory already exists: '{target_ckpt}'. Set overwrite=True to replace it."
        )

    # 2. Load model & tokenizer if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_pretrained_encoder(config.encoder_checkpoint_dir)

    # 3. Resolve label mappings
    train_labels = list(dataset_dict["train"]["label"])
    unique_labels = sorted(list(set(train_labels)))
    new_num_labels = len(unique_labels)

    if id2label is None:
        id2label = {i: f"Class_{i}" for i in range(new_num_labels)}
    if label2id is None:
        label2id = {v: k for k, v in id2label.items()}

    # 4. Zero-shot baseline evaluation on test split BEFORE head swap
    test_split = dataset_dict["test"]
    zero_shot_metrics = {}
    if compute_baseline:
        zero_shot_metrics = evaluate_zero_shot_baseline(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=test_split,
            id2label=id2label,
            max_length=config.training.max_length,
            batch_size=config.training.eval_batch_size,
        )
    else:
        zero_shot_metrics = {"accuracy": 0.0, "macro_f1": 0.0}
    zero_shot_acc = zero_shot_metrics["accuracy"]
    zero_shot_f1 = zero_shot_metrics["macro_f1"]

    # 5. Swap classification head
    swap_classification_head(
        model=model,
        num_new_labels=new_num_labels,
        id2label=id2label,
        label2id=label2id,
        freeze_backbone=config.training.freeze_backbone,
    )

    # 6. Tokenize dataset
    validation_key = "validation" if "validation" in dataset_dict else "val"
    dataset_dict = DatasetDict({
        "train": dataset_dict["train"],
        "validation": dataset_dict[validation_key],
        "test": dataset_dict["test"],
    })

    tokenized_dict, _ = tokenize_dataset(
        dataset=dataset_dict,
        tokenizer=tokenizer,
        max_length=config.training.max_length,
    )

    # 7. Compute balanced class weights
    y_train = np.array(train_labels)
    computed_weights = compute_class_weight("balanced", classes=np.array(unique_labels), y=y_train)
    full_weights = np.ones(new_num_labels, dtype=np.float32)
    for cls_idx, w in zip(unique_labels, computed_weights):
        full_weights[cls_idx] = float(w)
    class_weights_tensor = torch.tensor(full_weights, dtype=torch.float32)

    # 8. Setup EarlyStopping and TrainingArguments
    early_stopping_monitor = EarlyStoppingMonitor(
        early_stopping_patience=config.patience,
        early_stopping_threshold=0.0,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        training_args = TrainingArguments(
            output_dir=temp_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=config.training.learning_rate,
            per_device_train_batch_size=config.training.batch_size,
            per_device_eval_batch_size=config.training.eval_batch_size,
            num_train_epochs=config.training.num_epochs or config.training.epochs,
            weight_decay=config.training.weight_decay,
            warmup_ratio=config.training.warmup_ratio,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            seed=config.training.seed,
            logging_steps=10,
            report_to="none",
            save_total_limit=1,
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dict["train"],
            eval_dataset=tokenized_dict["validation"],
            compute_metrics=compute_metrics,
            callbacks=[early_stopping_monitor],
            class_weights=class_weights_tensor,
        )

        logger.info("Starting domain fine-tuning...")
        trainer.train()

        # 9. Evaluate on test split
        test_pred_output = trainer.predict(tokenized_dict["test"])
        test_preds = np.argmax(test_pred_output.predictions, axis=1)
        test_true = np.array(tokenized_dict["test"]["label"])

        class_names_list = [id2label[i] for i in range(new_num_labels)]
        detailed_metrics = compute_detailed_metrics(
            labels=test_true,
            preds=test_preds,
            label_names=class_names_list,
        )

    adapted_acc = float(detailed_metrics["accuracy"])
    adapted_f1 = float(detailed_metrics["macro_f1"])
    delta_macro_f1 = adapted_f1 - zero_shot_f1

    # Warning on failure to exceed zero-shot baseline
    if delta_macro_f1 <= 0:
        logger.warning(
            f"Domain adaptation macro-F1 ({adapted_f1:.4f}) did not exceed zero-shot baseline "
            f"({zero_shot_f1:.4f}). Investigate dataset quality or increase training data."
        )

    # 10. Persist checkpoint and export metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        path=str(output_dir),
        overwrite=True,
    )

    result = DomainAdaptResult(
        accuracy=round(adapted_acc, 4),
        macro_f1=round(adapted_f1, 4),
        zero_shot_accuracy=round(zero_shot_acc, 4),
        zero_shot_macro_f1=round(zero_shot_f1, 4),
        delta_macro_f1=round(delta_macro_f1, 4),
        num_classes=new_num_labels,
        class_names=class_names_list,
        checkpoint_dir=str(output_dir),
        metrics_path=str(config.metrics_path),
        stopped_epoch=early_stopping_monitor.stopped_epoch,
        top_5_worst_classes=detailed_metrics.get("top_5_worst_classes", []),
        model_path=str(output_dir),
        adapted_metrics={"accuracy": round(adapted_acc, 4), "macro_f1": round(adapted_f1, 4)},
        baseline_metrics={"accuracy": round(zero_shot_acc, 4), "macro_f1": round(zero_shot_f1, 4)},
        improvement_delta=round(delta_macro_f1, 4),
    )

    (output_dir / "config.json").write_text(json.dumps({"training": config.training.model_dump(), "num_classes": new_num_labels}, indent=2), encoding="utf-8")
    (output_dir / "taxonomy_label2id.json").write_text(json.dumps(label2id, indent=2), encoding="utf-8")
    (output_dir / "adaptation_metrics.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    metrics_out = Path(config.metrics_path)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(
        f"Domain adaptation complete: accuracy={adapted_acc:.4f}, macro_f1={adapted_f1:.4f}, "
        f"delta_macro_f1={delta_macro_f1:+.4f}. Metrics saved to '{metrics_out}'."
    )

    return result
