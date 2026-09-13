"""
Domain adaptation dataset preparation, purity filtering, sparse cluster merging,
and stratified train/val/test splitting for AppleSupport intent classification.
"""

from collections import Counter
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _extract_text(record: Any) -> str:
    """Extract text content from either a dict, an object with attributes, or a string."""
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        return str(
            record.get("clean_text")
            or record.get("text")
            or record.get("first_turn_text")
            or ""
        )
    return str(
        getattr(record, "clean_text", None)
        or getattr(record, "text", None)
        or getattr(record, "first_turn_text", None)
        or ""
    )


def calculate_cluster_purity(probabilities: Sequence[float]) -> float:
    """Calculate the mean cluster purity/confidence from membership probabilities.

    Parameters
    ----------
    probabilities : Sequence[float]
        Sequence of membership probabilities or confidence scores (in [0, 1]).

    Returns
    -------
    float
        Mean purity score, or 0.0 if sequence is empty.
    """
    if len(probabilities) == 0:
        return 0.0
    return float(np.mean(probabilities))


def filter_by_purity(
    records: Sequence[Any],
    labels: Sequence[str],
    probabilities: Sequence[float],
    purity_threshold: float = 0.7,
    unclassified_label: str = "Unclassified",
) -> Tuple[List[Any], List[str], List[float]]:
    """Filter out records whose cluster membership probability is below threshold,

    and discard unclassified outlier points (-1 cluster).
    """
    if not (len(records) == len(labels) == len(probabilities)):
        raise ValueError(
            f"Length mismatch: records={len(records)}, labels={len(labels)}, probabilities={len(probabilities)}"
        )

    filtered_records = []
    filtered_labels = []
    filtered_probs = []

    dropped_unclassified = 0
    dropped_low_purity = 0

    for rec, lbl, prob in zip(records, labels, probabilities):
        if lbl == -1 or lbl == unclassified_label or str(lbl) == str(unclassified_label):
            dropped_unclassified += 1
            continue
        if prob < purity_threshold:
            dropped_low_purity += 1
            continue
        filtered_records.append(rec)
        filtered_labels.append(lbl)
        filtered_probs.append(float(prob))

    logger.info(
        "Purity filtering (threshold=%.2f): retained %d / %d records. "
        "(Dropped %d unclassified, %d low-purity < %.2f)",
        purity_threshold,
        len(filtered_records),
        len(records),
        dropped_unclassified,
        dropped_low_purity,
        purity_threshold,
    )
    return filtered_records, filtered_labels, filtered_probs


def merge_sparse_clusters(
    records: Sequence[Any],
    labels: Sequence[str],
    min_cluster_size: int = 1,
    other_label: str = "Other_Uncategorized",
) -> Tuple[List[Any], List[str]]:
    """Merge clusters with fewer than `min_cluster_size` examples into a fallback class.

    Parameters
    ----------
    records : Sequence[Any]
        Retained records.
    labels : Sequence[str]
        Cluster/intent labels.
    min_cluster_size : int, default=20
        Minimum count required for a cluster to remain standalone.
    other_label : str, default="Other_Uncategorized"
        Fallback label for merged sparse clusters.

    Returns
    -------
    Tuple[List[Any], List[str]]
        (records, updated_labels)
    """
    counts = Counter(labels)
    sparse_labels = {lbl for lbl, cnt in counts.items() if cnt < min_cluster_size}

    if sparse_labels:
        logger.info(
            "Merging %d sparse clusters (< %d samples) into '%s': %s",
            len(sparse_labels),
            min_cluster_size,
            other_label,
            sorted(sparse_labels),
        )

    updated_labels = [
        other_label if lbl in sparse_labels else lbl for lbl in labels
    ]

    new_counts = Counter(updated_labels)
    logger.info(
        "Distribution after sparse merge: %d distinct classes. %s",
        len(new_counts),
        dict(new_counts.most_common(10)),
    )
    return list(records), updated_labels


def validate_surviving_classes(
    labels: Sequence[str], min_classes: int = 3
) -> List[str]:
    """Validate that at least `min_classes` distinct labels remain after filtering."""
    unique_classes = sorted(set(labels))
    if len(unique_classes) < min_classes:
        raise ValueError(
            f"Too few distinct classes ({len(unique_classes)} < {min_classes}). "
            f"Found: {unique_classes}. Purity threshold or min_cluster_size may be too strict."
        )
    return unique_classes


def stratified_three_way_split(
    records: Sequence[Any],
    labels: Sequence[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Generate indices for a 3-way stratified split (train / val / test)."""
    if not (0.0 < val_ratio < 1.0 and 0.0 < test_ratio < 1.0 and (val_ratio + test_ratio < 1.0)):
        raise ValueError(
            f"Sum of val_ratio and test_ratio must be < 1.0, got {val_ratio + test_ratio}."
        )

    rng = np.random.RandomState(seed)
    class_indices: Dict[str, List[int]] = {}
    for idx, lbl in enumerate(labels):
        class_indices.setdefault(lbl, []).append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for lbl, indices in class_indices.items():
        indices = list(indices)
        rng.shuffle(indices)
        n = len(indices)

        if n == 1:
            # Singleton: assign to train to prevent zero-sample crash in val/test
            train_indices.extend(indices)
        elif n == 2:
            # 2 samples: 1 in train, 1 in val
            train_indices.append(indices[0])
            val_indices.append(indices[1])
        else:
            n_test = max(1, int(round(n * test_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            # Ensure at least 1 in train
            if n_test + n_val >= n:
                n_test = max(1, (n - 1) // 2)
                n_val = max(1, n - 1 - n_test)

            test_part = indices[:n_test]
            val_part = indices[n_test : n_test + n_val]
            train_part = indices[n_test + n_val :]

            test_indices.extend(test_part)
            val_indices.extend(val_part)
            train_indices.extend(train_part)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return train_indices, val_indices, test_indices


def assemble_domain_dataset(
    records: Sequence[Any],
    labels: Optional[Sequence[str]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_length: int = 64,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    probabilities: Optional[Sequence[float]] = None,
    seed: int = 42,
    purity_threshold: float = 0.7,
    min_cluster_size: int = 1,
    other_label: str = "Other_Uncategorized",
    unclassified_label: str = "Unclassified",
    min_classes: int = 3,
    cluster_labels: Optional[Sequence[str]] = None,
) -> Tuple[DatasetDict, Dict[str, int], Dict[int, str]]:
    """Assemble, filter, merge, and split domain adaptation dataset for fine-tuning.

    Parameters
    ----------
    records : Sequence[Any]
        Input context window records or texts.
    cluster_labels : Sequence[str]
        Cluster labels assigned during taxonomy induction.
    probabilities : Sequence[float]
        HDBSCAN assignment probabilities.
    tokenizer : Optional[PreTrainedTokenizerBase], default=None
        If provided, dataset splits are tokenized with padding and truncation.
    max_length : int, default=64
        Sequence length limit for tokenization.
    val_ratio : float, default=0.15
        Validation set ratio.
    test_ratio : float, default=0.15
        Test set ratio.
    seed : int, default=42
        Random seed.
    purity_threshold : float, default=0.7
        Minimum HDBSCAN probability threshold.
    min_cluster_size : int, default=20
        Sparse cluster threshold.
    other_label : str, default="Other_Uncategorized"
        Fallback label for merged sparse clusters.
    unclassified_label : str, default="Unclassified"
        Outlier label to discard.
    min_classes : int, default=3
        Minimum number of surviving classes.

    Returns
    -------
    Tuple[DatasetDict, Dict[str, int], Dict[int, str]]
        (dataset_dict, label2id, id2label)
    """
    if labels is None:
        labels = cluster_labels
    if labels is None:
        raise ValueError("Either 'labels' or 'cluster_labels' must be provided.")

    if probabilities is None:
        probabilities = [1.0] * len(records)

    # 1. Purity filtering
    filt_records, filt_labels, filt_probs = filter_by_purity(
        records=records,
        labels=labels,
        probabilities=probabilities,
        purity_threshold=purity_threshold,
        unclassified_label=unclassified_label,
    )

    # 2. Sparse cluster merging
    merged_records, final_labels = merge_sparse_clusters(
        records=filt_records,
        labels=filt_labels,
        min_cluster_size=min_cluster_size,
        other_label=other_label,
    )

    # 3. Validation of surviving classes
    unique_classes = validate_surviving_classes(
        labels=final_labels, min_classes=min_classes
    )

    label2id = {lbl: idx for idx, lbl in enumerate(unique_classes)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}

    # 4. Stratified 3-way split
    train_idx, val_idx, test_idx = stratified_three_way_split(
        records=merged_records,
        labels=final_labels,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    def _build_split_dict(indices: List[int]) -> Dict[str, List[Any]]:
        texts = [_extract_text(merged_records[i]) for i in indices]
        lbls = [final_labels[i] for i in indices]
        ids = [label2id[lbl] for lbl in lbls]
        probs = [filt_probs[i] for i in indices]
        return {
            "text": texts,
            "label": ids,
            "label_text": lbls,
            "cluster_probability": probs,
        }

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_dict(_build_split_dict(train_idx)),
            "validation": Dataset.from_dict(_build_split_dict(val_idx)),
            "test": Dataset.from_dict(_build_split_dict(test_idx)),
        }
    )

    logger.info(
        "Assembled domain dataset: train=%d, validation=%d, test=%d across %d classes.",
        len(dataset_dict["train"]),
        len(dataset_dict["validation"]),
        len(dataset_dict["test"]),
        len(label2id),
    )

    # 5. Tokenization if tokenizer provided
    if tokenizer is not None:
        def _tokenize_batch(batch):
            return tokenizer(
                batch["text"],
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )

        tokenized_dict = DatasetDict()
        for split_name, split_ds in dataset_dict.items():
            tokenized_dict[split_name] = split_ds.map(
                _tokenize_batch,
                batched=True,
                desc=f"Tokenizing domain {split_name}",
            )
        return tokenized_dict, label2id, id2label

    return dataset_dict, label2id, id2label
