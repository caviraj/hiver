"""Unit tests for dataset loading, tokenization, and stratified splitting."""

from unittest.mock import patch
import pytest
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from src.models.intent.dataset import (
    count_truncated_queries,
    load_banking77,
    stratified_split,
    tokenize_dataset,
)


@pytest.fixture
def dummy_tokenizer() -> PreTrainedTokenizerFast:
    """Create a lightweight fast tokenizer for offline testing."""
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "hello": 2,
        "world": 3,
        "foo": 4,
        "bar": 5,
        "baz": 6,
        "qux": 7,
        "a": 8,
        "b": 9,
        "c": 10,
    }
    raw = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="[PAD]",
        unk_token="[UNK]",
    )


def test_load_banking77_offline_failure():
    """Verify that when load_dataset fails (e.g. offline with no cache), a RuntimeError is raised."""
    with patch("src.models.intent.dataset.load_dataset", side_effect=Exception("Connection refused")):
        with pytest.raises(RuntimeError, match="Failed to download Banking77 dataset"):
            load_banking77()


def test_load_banking77_local_path(tmp_path):
    """Verify loading dataset from a local path saved with save_to_disk."""
    sample_data = DatasetDict({
        "train": Dataset.from_dict({"text": ["how to transfer money"], "label": [0]}),
        "test": Dataset.from_dict({"text": ["card lost"], "label": [1]}),
    })
    save_dir = tmp_path / "local_banking77"
    sample_data.save_to_disk(str(save_dir))

    loaded = load_banking77(local_path=str(save_dir))
    assert isinstance(loaded, DatasetDict)
    assert "train" in loaded
    assert "test" in loaded
    assert len(loaded["train"]) == 1
    assert loaded["train"][0]["text"] == "how to transfer money"


def test_load_banking77_local_path_not_found(tmp_path):
    """Verify FileNotFoundError is raised when local_path does not exist."""
    non_existent = tmp_path / "non_existent_dataset_folder"
    with pytest.raises(FileNotFoundError, match="Specified local_dataset_path does not exist"):
        load_banking77(local_path=str(non_existent))


def test_count_truncated_queries_and_tokenize(dummy_tokenizer):
    """Verify truncation counting and dataset tokenization."""
    # "hello world" is 2 tokens; "hello world foo bar baz qux" is 6 tokens
    ds = Dataset.from_dict({
        "text": ["hello world", "hello world foo bar baz qux"],
        "label": [0, 1],
    })

    # max_length=4: "hello world" (len 2 <= 4) not truncated, 6 tokens > 4 truncated
    trunc_count = count_truncated_queries(ds, dummy_tokenizer, max_length=4)
    assert trunc_count == 1

    tokenized_ds, total_trunc = tokenize_dataset(ds, dummy_tokenizer, max_length=4)
    assert total_trunc == 1
    assert "input_ids" in tokenized_ds.column_names
    assert "attention_mask" in tokenized_ds.column_names
    assert len(tokenized_ds[0]["input_ids"]) == 4
    assert len(tokenized_ds[1]["input_ids"]) == 4

    # Test DatasetDict support
    ds_dict = DatasetDict({"train": ds})
    tokenized_dict, total_trunc_dict = tokenize_dataset(ds_dict, dummy_tokenizer, max_length=4)
    assert total_trunc_dict == 1
    assert isinstance(tokenized_dict, DatasetDict)
    assert len(tokenized_dict["train"][0]["input_ids"]) == 4


def test_stratified_split_proportions_and_coverage():
    """Verify stratified split maintains class balance and covers all classes with N >= 2."""
    data = {
        "text": [f"query_{i}" for i in range(30)],
        "label": [0] * 10 + [1] * 10 + [2] * 10,
    }
    ds = Dataset.from_dict(data)

    split = stratified_split(ds, val_ratio=0.2, seed=42)
    assert isinstance(split, DatasetDict)
    assert "train" in split
    assert "val" in split

    train_labels = split["train"]["label"]
    val_labels = split["val"]["label"]

    assert len(split["train"]) == 24
    assert len(split["val"]) == 6

    # Verify coverage: each class has 2 in val and 8 in train
    for cls in [0, 1, 2]:
        assert val_labels.count(cls) == 2
        assert train_labels.count(cls) == 8


def test_stratified_split_singleton_handling(caplog):
    """Verify singleton classes (N=1) are placed in train with a warning log."""
    data = {
        "text": ["q0_a", "q0_b", "q0_c", "q0_d", "q1_a", "q1_b", "q2_singleton"],
        "label": [0, 0, 0, 0, 1, 1, 2],
    }
    ds = Dataset.from_dict(data)

    with caplog.at_level("WARNING"):
        split = stratified_split(ds, val_ratio=0.25, seed=42)

    assert "Class '2' has only 1 sample. Placing in train split" in caplog.text

    train_labels = split["train"]["label"]
    val_labels = split["val"]["label"]

    # Singleton class 2 is in train, not in val
    assert 2 in train_labels
    assert 2 not in val_labels

    # Classes 0 and 1 have representation in both
    assert 0 in train_labels and 0 in val_labels
    assert 1 in train_labels and 1 in val_labels


def test_stratified_split_invalid_inputs():
    """Verify ValueError on empty dataset, invalid ratio, or missing train split."""
    ds = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})

    with pytest.raises(ValueError, match="val_ratio must be between 0.0 and 1.0"):
        stratified_split(ds, val_ratio=0.0)

    with pytest.raises(ValueError, match="val_ratio must be between 0.0 and 1.0"):
        stratified_split(ds, val_ratio=1.0)

    with pytest.raises(ValueError, match="val_ratio must be between 0.0 and 1.0"):
        stratified_split(ds, val_ratio=-0.1)

    empty_ds = Dataset.from_dict({"text": [], "label": []})
    with pytest.raises(ValueError, match="Cannot split an empty dataset"):
        stratified_split(empty_ds, val_ratio=0.2)

    no_train_dict = DatasetDict({"test": ds})
    with pytest.raises(ValueError, match="DatasetDict must contain a 'train' split"):
        stratified_split(no_train_dict, val_ratio=0.2)
