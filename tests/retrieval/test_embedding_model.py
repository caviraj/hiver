"""Unit tests for embedding_model.py (M3.P3.2.F1).

Verifies encoder loading, offline failure handling with actionable diagnostics,
explicit L2 normalization, empty batch handling, and batching consistency.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.retrieval.embedding_model import (
    DEFAULT_MODEL_NAME,
    embed_texts,
    load_encoder,
)


@pytest.fixture(scope="module")
def shared_encoder():
    """Load the default encoder once for test module to save initialization time."""
    return load_encoder(DEFAULT_MODEL_NAME)


class TestLoadEncoder:
    """Tests for load_encoder function."""

    def test_load_encoder_default(self, shared_encoder):
        """Confirm default encoder loads successfully and has expected attributes."""
        assert shared_encoder is not None
        assert hasattr(shared_encoder, "encode")
        assert hasattr(shared_encoder, "max_seq_length")
        assert shared_encoder.max_seq_length > 0

    def test_load_encoder_failure_actionable_error(self):
        """Confirm loading an invalid/unreachable model raises an actionable RuntimeError."""
        with pytest.raises(RuntimeError) as exc_info:
            load_encoder("nonexistent-model-org/completely-fake-model-xyz-99999")

        err_msg = str(exc_info.value)
        assert "Failed to load sentence encoder model" in err_msg
        assert "pre-downloaded" in err_msg or "offline" in err_msg

    def test_load_encoder_network_failure_mocked(self):
        """Verify explicit network/download exception wrapping with mock."""
        with patch("sentence_transformers.SentenceTransformer", side_effect=ConnectionError("Offline")):
            with pytest.raises(RuntimeError) as exc_info:
                load_encoder("test-model")
            assert "offline environment" in str(exc_info.value).lower() or "local" in str(exc_info.value).lower()


class TestEmbedTexts:
    """Tests for embed_texts function."""

    def test_embed_texts_empty_list(self, shared_encoder):
        """Confirm empty input returns an empty 2D numpy array without error."""
        result = embed_texts([], shared_encoder)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 0
        assert len(result.shape) == 2

    def test_embed_texts_l2_normalization(self, shared_encoder):
        """Verify every returned embedding has an L2 norm of exactly 1.0."""
        texts = [
            "My screen is completely frozen and black.",
            "Can I update my billing information for next month?",
            "App crash on startup with error code 500.",
        ]
        embeddings = embed_texts(texts, shared_encoder, batch_size=2)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.ndim == 2
        assert embeddings.dtype == np.float32

        # Verify unit norm
        norms = np.linalg.norm(embeddings, axis=1)
        for i, norm_val in enumerate(norms):
            assert norm_val == pytest.approx(1.0, abs=1e-5), f"Text {i} norm is {norm_val}, expected 1.0"

    def test_embed_texts_batching_consistency(self, shared_encoder):
        """Verify embeddings computed with batch_size=1 match those with batch_size=4."""
        texts = [
            "Payment failed with card decline.",
            "Where do I find my account settings?",
            "WiFi disconnects intermittently on SM-T280.",
            "How to reset password?",
        ]
        emb_b1 = embed_texts(texts, shared_encoder, batch_size=1)
        emb_b4 = embed_texts(texts, shared_encoder, batch_size=4)

        np.testing.assert_allclose(emb_b1, emb_b4, rtol=1e-5, atol=1e-6)

    def test_embed_texts_zeros_vector_handled(self):
        """Verify zero vector is handled safely without division by zero NaN."""
        mock_encoder = MagicMock()
        # Mock returning an all-zero vector
        mock_encoder.encode.return_value = np.zeros((1, 16), dtype=np.float32)
        embeddings = embed_texts(["dummy text"], mock_encoder)
        assert not np.isnan(embeddings).any()
        assert not np.isinf(embeddings).any()
