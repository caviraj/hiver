"""
Dataset handling, tokenization, and stratified splitting for Banking77 transfer learning.
"""

from collections import Counter
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def load_banking77(
    dataset_name: str = "PolyAI/banking77",
    local_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """Load the Banking77 dataset from HuggingFace Hub or local cache/filesystem.

    Parameters
    ----------
    dataset_name : str, default="PolyAI/banking77"
        HuggingFace Hub repository identifier.
    local_path : Optional[str], default=None
        Path to local dataset directory saved via `save_to_disk` or raw files.
    cache_dir : Optional[str], default=None
        Custom directory for caching downloaded files.

    Returns
    -------
    DatasetDict
        Dataset dictionary containing the splits (typically 'train' and 'test').

    Raises
    ------
    RuntimeError
        If dataset cannot be downloaded due to network failure and no local path is found.
    """
    if local_path:
        local_dir = Path(local_path)
        if local_dir.is_dir():
            logger.info("Loading Banking77 dataset from local disk: %s", local_dir)
            try:
                loaded = load_from_disk(str(local_dir))
                if isinstance(loaded, Dataset):
                    return DatasetDict({"train": loaded})
                return loaded
            except Exception as e:
                logger.warning(
                    "Failed to load from disk as Arrow dataset (%s). Attempting load_dataset on local path.",
                    e,
                )
                try:
                    return load_dataset(str(local_dir), cache_dir=cache_dir)
                except Exception as inner_e:
                    raise RuntimeError(
                        f"Failed to load Banking77 from local path '{local_path}': {inner_e}"
                    ) from inner_e
        else:
            raise FileNotFoundError(f"Specified local_dataset_path does not exist: {local_path}")

    logger.info("Loading Banking77 dataset from HuggingFace Hub: %s", dataset_name)
    try:
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        if isinstance(dataset, Dataset):
            dataset = DatasetDict({"train": dataset})
        return dataset
    except Exception as e:
        logger.error("Failed to load dataset '%s' from Hub: %s", dataset_name, e)
        raise RuntimeError(
            f"Failed to download Banking77 dataset ('{dataset_name}'). "
            "If running offline, pre-download or specify 'local_dataset_path' in config."
        ) from e


def count_truncated_queries(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 64,
    text_column: str = "text",
) -> int:
    """Count queries in a dataset split that exceed max_length before truncation.

    Parameters
    ----------
    dataset : Dataset
        Hugging Face Dataset split.
    tokenizer : PreTrainedTokenizerBase
        Hugging Face tokenizer.
    max_length : int, default=64
        Maximum allowed token length.
    text_column : str, default="text"
        Name of text column.

    Returns
    -------
    int
        Number of queries that exceed max_length.
    """
    truncated_count = 0
    for item in dataset:
        text = item.get(text_column, "")
        token_ids = tokenizer.encode(text, truncation=False, add_special_tokens=True)
        if len(token_ids) > max_length:
            truncated_count += 1
    return truncated_count


def tokenize_dataset(
    dataset: Union[Dataset, DatasetDict],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 64,
    text_column: str = "text",
) -> Tuple[Union[Dataset, DatasetDict], int]:
    """Tokenize dataset queries with padding and truncation, tracking truncated query count.

    Banking77 service queries have a median length of 12-15 words. A max_length of 64
    covers >99% of queries without truncation while drastically saving computation on CPU.

    Parameters
    ----------
    dataset : Union[Dataset, DatasetDict]
        Single Dataset or DatasetDict.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer to use.
    max_length : int, default=64
        Maximum sequence length.
    text_column : str, default="text"
        Column containing input text queries.

    Returns
    -------
    Tuple[Union[Dataset, DatasetDict], int]
        (tokenized_dataset, total_truncated_queries_count)
    """
    total_truncated = 0

    def _tokenize_fn(batch):
        return tokenizer(
            batch[text_column],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    if isinstance(dataset, DatasetDict):
        tokenized_dict = DatasetDict()
        total_samples = 0
        for split_name, split_ds in dataset.items():
            split_truncated = count_truncated_queries(
                split_ds, tokenizer, max_length=max_length, text_column=text_column
            )
            total_truncated += split_truncated
            total_samples += len(split_ds)
            tokenized_dict[split_name] = split_ds.map(
                _tokenize_fn,
                batched=True,
                desc=f"Tokenizing {split_name}",
            )
            logger.info(
                "Split '%s': %d / %d queries (%.2f%%) exceeded max_length=%d and were truncated.",
                split_name,
                split_truncated,
                len(split_ds),
                (split_truncated / len(split_ds) * 100.0) if len(split_ds) > 0 else 0.0,
                max_length,
            )
        return tokenized_dict, total_truncated
    else:
        total_truncated = count_truncated_queries(
            dataset, tokenizer, max_length=max_length, text_column=text_column
        )
        tokenized_ds = dataset.map(
            _tokenize_fn,
            batched=True,
            desc="Tokenizing dataset",
        )
        logger.info(
            "Dataset: %d / %d queries (%.2f%%) exceeded max_length=%d and were truncated.",
            total_truncated,
            len(dataset),
            (total_truncated / len(dataset) * 100.0) if len(dataset) > 0 else 0.0,
            max_length,
        )
        return tokenized_ds, total_truncated


def stratified_split(
    dataset: Union[Dataset, DatasetDict],
    val_ratio: float = 0.15,
    seed: int = 42,
    label_column: str = "label",
) -> DatasetDict:
    """Split dataset into stratified train and validation splits ensuring full class representation.

    Every class with 2 or more instances is guaranteed to have at least 1 instance in
    both the train and validation sets. Singleton classes (N=1) are placed in the train
    set with an explicit warning log rather than throwing a ValueError.

    Parameters
    ----------
    dataset : Union[Dataset, DatasetDict]
        Source dataset (if DatasetDict, 'train' split is extracted and split).
    val_ratio : float, default=0.15
        Proportion of instances per class allocated to validation.
    seed : int, default=42
        Random seed for shuffling indices.
    label_column : str, default="label"
        Name of class label column.

    Returns
    -------
    DatasetDict
        DatasetDict with 'train' and 'val' splits (and existing non-train splits preserved).

    Raises
    ------
    ValueError
        If dataset is empty or val_ratio is not between 0 and 1.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be between 0.0 and 1.0, got {val_ratio}")

    other_splits = {}
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError("DatasetDict must contain a 'train' split to apply stratified_split.")
        target_dataset = dataset["train"]
        for k, v in dataset.items():
            if k != "train":
                other_splits[k] = v
    else:
        target_dataset = dataset

    if len(target_dataset) == 0:
        raise ValueError("Cannot split an empty dataset.")

    labels = target_dataset[label_column]
    label_counts = Counter(labels)

    logger.info(
        "Stratified split on %d samples across %d distinct classes. Min support: %d, Max support: %d, Median support: %.1f",
        len(target_dataset),
        len(label_counts),
        min(label_counts.values()),
        max(label_counts.values()),
        float(np.median(list(label_counts.values()))),
    )

    rng = np.random.RandomState(seed)

    # Group sample indices by label
    class_indices: Dict[Union[int, str], List[int]] = {}
    for idx, lbl in enumerate(labels):
        class_indices.setdefault(lbl, []).append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []

    for lbl, indices in class_indices.items():
        n_samples = len(indices)
        indices = list(indices)
        rng.shuffle(indices)

        if n_samples == 1:
            logger.warning(
                "Class '%s' has only 1 sample. Placing in train split; cannot be represented in validation split.",
                lbl,
            )
            train_indices.extend(indices)
        else:
            # Guarantee at least 1 in val and at least 1 in train
            n_val = int(round(n_samples * val_ratio))
            n_val = max(1, min(n_samples - 1, n_val))
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

    # Shuffle combined indices deterministically
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    train_split = target_dataset.select(train_indices)
    val_split = target_dataset.select(val_indices)

    result_splits = {
        "train": train_split,
        "val": val_split,
    }
    # Preserve other splits (e.g. test)
    result_splits.update(other_splits)

    logger.info(
        "Stratified split completed: %d train samples, %d val samples (%.2f%% val).",
        len(train_split),
        len(val_split),
        (len(val_split) / (len(train_split) + len(val_split)) * 100.0),
    )

    return DatasetDict(result_splits)
