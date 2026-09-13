"""Unit tests for src.models.intent.taxonomy_induction.

All tests run completely hermetic without network requests or external dependencies,
using synthetic embeddings, mock LLMs, and minimal local fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    PreTrainedTokenizerFast,
)

from src.models.intent.severity_rules import SeverityAuditor
from src.models.intent.taxonomy_induction import (
    _extract_text,
    cluster_embeddings,
    embed_messages,
    get_representative_indices,
    induce_taxonomy,
    label_cluster,
    load_taxonomy,
    persist_taxonomy,
    resolve_naming_collisions,
    sanitize_label,
)


@pytest.fixture
def dummy_tokenizer() -> PreTrainedTokenizerFast:
    """Create a minimal in-memory tokenizer."""
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "hello": 2,
        "battery": 3,
        "update": 4,
        "lawyer": 5,
        "screen": 6,
        "broken": 7,
        "apple": 8,
        "support": 9,
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
    """Create a minimal 1-layer DistilBERT model for ultra-fast embedding extraction."""
    cfg = DistilBertConfig(
        vocab_size=10,
        dim=16,
        n_layers=1,
        n_heads=2,
        hidden_dim=32,
        num_labels=3,
    )
    return DistilBertForSequenceClassification(cfg)


def test_extract_text() -> None:
    """Test _extract_text across strings, dicts, and objects."""
    assert _extract_text("hello world") == "hello world"
    assert _extract_text({"text": "sample text"}) == "sample text"
    assert _extract_text({"content": "content text"}) == "content text"
    assert _extract_text({"message": "message text"}) == "message text"

    class MockRecord:
        def __init__(self, text: str) -> None:
            self.text = text

    assert _extract_text(MockRecord("obj text")) == "obj text"


def test_embed_messages_dummy_model(
    dummy_model: DistilBertForSequenceClassification,
    dummy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Test embed_messages extracts unit L2-normalized embeddings of shape (N, D)."""
    texts = ["hello battery", "update apple support", "broken screen"]
    embeddings = embed_messages(
        records=texts,
        model=dummy_model,
        tokenizer=dummy_tokenizer,
        batch_size=2,
        max_length=16,
    )

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 16)
    # Check L2 normalization (norms should be approx 1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), rtol=1e-5)


def test_embed_messages_empty() -> None:
    """Test embed_messages handles empty inputs cleanly."""
    embeddings = embed_messages(records=[])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 0
    assert embeddings.shape[1] == 768


def test_cluster_embeddings_synthetic() -> None:
    """Test cluster_embeddings partitions dense points and calculates centroids."""
    rng = np.random.RandomState(42)
    # Generate 3 distinct clusters of 25 points each in 4 dimensions
    c1 = rng.normal(loc=0.0, scale=0.05, size=(25, 4))
    c2 = rng.normal(loc=5.0, scale=0.05, size=(25, 4))
    c3 = rng.normal(loc=10.0, scale=0.05, size=(25, 4))
    embeddings = np.vstack([c1, c2, c3])

    labels, probs, centroids = cluster_embeddings(
        embeddings,
        min_cluster_size=10,
        min_samples=5,
    )

    assert len(labels) == 75
    assert len(probs) == 75
    unique_labels = set(labels)
    # HDBSCAN should discover at least 2 or 3 non-noise clusters
    valid_clusters = [lbl for lbl in unique_labels if lbl != -1]
    assert len(valid_clusters) >= 2
    for cid in valid_clusters:
        assert cid in centroids
        assert centroids[cid].shape == (4,)


def test_get_representative_indices() -> None:
    """Test get_representative_indices returns indices sorted by distance to centroid."""
    # 4 points: index 2 is closest to [0, 0], then 0, 1, 3
    embeddings = np.array([
        [0.2, 0.2],
        [0.5, 0.5],
        [0.01, 0.01],
        [0.9, 0.9],
    ])
    labels = np.array([0, 0, 0, 0])
    centroid = np.array([0.0, 0.0])

    top_idx = get_representative_indices(embeddings, labels, cluster_id=0, centroid=centroid, max_examples=2)
    assert top_idx == [2, 0]


def test_sanitize_label() -> None:
    """Test sanitize_label removes numbering, formats PascalCase, and falls back correctly."""
    assert sanitize_label("1. Device Performance Degradation", 0) == "Device_Performance_Degradation"
    assert sanitize_label("Software_Update_Failure", 1) == "Software_Update_Failure"
    assert sanitize_label("Label: Account Recovery Issues", 2) == "Account_Recovery_Issues"
    assert sanitize_label("", 3) == "Cluster_3"
    assert sanitize_label("???!!!", 4) == "Cluster_4"


def test_label_cluster_heuristics_and_mock_llm() -> None:
    """Test label_cluster with heuristic fallback and mock LLM."""
    # Test heuristic match
    battery_texts = ["My battery is draining so fast", "Battery dies within 1 hour"]
    label = label_cluster(battery_texts, cluster_id=0, llm_client=None)
    assert label == "Device_Performance_Degradation"

    # Test mock LLM client
    mock_llm = MagicMock(return_value="Subscription_Billing_Issue")
    other_texts = ["Why was I charged twice for iCloud?", "Refund my subscription"]
    label_llm = label_cluster(other_texts, cluster_id=1, llm_client=mock_llm)
    assert label_llm == "Subscription_Billing_Issue"
    mock_llm.assert_called_once()

    # Test LLM failure falls back
    failing_llm = MagicMock(side_effect=RuntimeError("API error"))
    fallback_label = label_cluster(["unusual text without keywords"], cluster_id=5, llm_client=failing_llm)
    assert fallback_label == "Cluster_5"


def test_resolve_naming_collisions() -> None:
    """Test resolve_naming_collisions deduplicates cluster names."""
    input_labels = {
        -1: "Unclassified",
        0: "Account_Recovery",
        1: "Device_Performance_Degradation",
        2: "Account_Recovery",
        3: "Account_Recovery",
    }
    resolved = resolve_naming_collisions(input_labels)
    assert resolved[-1] == "Unclassified"
    assert resolved[0] == "Account_Recovery"
    assert resolved[1] == "Device_Performance_Degradation"
    assert resolved[2] == "Account_Recovery_2"
    assert resolved[3] == "Account_Recovery_3"


def test_induce_taxonomy_synthetic() -> None:
    """Test end-to-end taxonomy induction with synthetic embeddings and records."""
    rng = np.random.RandomState(42)
    c1 = rng.normal(loc=0.0, scale=0.05, size=(25, 4))
    c2 = rng.normal(loc=10.0, scale=0.05, size=(25, 4))
    embeddings = np.vstack([c1, c2])

    records = (
        ["My battery dies rapidly and screen freezes" for _ in range(25)]
        + ["I want to sue you, contact my lawyer immediately" for _ in range(25)]
    )

    auditor = SeverityAuditor()
    taxonomy = induce_taxonomy(
        records=records,
        embeddings=embeddings,
        min_cluster_size=10,
        min_samples=5,
        purity_threshold=0.5,
        auditor=auditor,
    )

    assert isinstance(taxonomy, dict)
    assert len(taxonomy) >= 2
    for cid_str, meta in taxonomy.items():
        assert "label" in meta
        assert "size" in meta
        assert "purity_score" in meta
        assert "representative_examples" in meta
        assert "is_severity_escalation" in meta

    # Check auditor tracked records
    audit_summary = auditor.summary()
    assert audit_summary["total_evaluated"] > 0


def test_persist_and_load_taxonomy(tmp_path: Path) -> None:
    """Test persist_taxonomy and load_taxonomy roundtrip."""
    taxonomy = {
        "0": {
            "cluster_id": 0,
            "label": "Account_Recovery",
            "size": 42,
            "purity_score": 0.91,
            "representative_examples": ["Can't reset password"],
            "is_severity_escalation": False,
            "keyword_candidate": False,
            "llm_agreed": False,
        },
        "-1": {
            "cluster_id": -1,
            "label": "Unclassified",
            "size": 5,
            "purity_score": 0.0,
            "representative_examples": ["xyz"],
            "is_severity_escalation": False,
            "keyword_candidate": False,
            "llm_agreed": False,
        },
    }

    out_file = tmp_path / "apple_taxonomy.json"
    persist_taxonomy(taxonomy, out_file)
    assert out_file.exists()

    loaded = load_taxonomy(out_file)
    assert loaded["0"]["label"] == "Account_Recovery"
    assert loaded["0"]["size"] == 42
    assert loaded["-1"]["label"] == "Unclassified"
