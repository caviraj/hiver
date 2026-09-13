"""Sentence embedding model loader and text encoder (M3.P3.2.F1).

Provides loading of lightweight transformer encoders (defaulting to all-MiniLM-L6-v2)
and batch text embedding with explicit L2 normalization for cosine similarity search.
"""

import logging
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_encoder(model_name: str = DEFAULT_MODEL_NAME):
    """Load a SentenceTransformer encoder model.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model repository ID or local filesystem path,
        by default "sentence-transformers/all-MiniLM-L6-v2".

    Returns
    -------
    SentenceTransformer
        Instantiated sentence-transformers encoder.

    Raises
    ------
    RuntimeError
        If model loading fails due to network/cache/filesystem errors, providing
        clear actionable instructions for offline usage.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "The 'sentence-transformers' package is required for dense retrieval. "
            "Install it via 'pip install sentence-transformers'."
        ) from exc

    try:
        logger.info("Loading sentence encoder model: %s", model_name)
        encoder = SentenceTransformer(model_name)
        return encoder
    except Exception as exc:
        msg = (
            f"Failed to load sentence encoder model '{model_name}'. "
            "If running in an offline or sandboxed environment without internet access, "
            "please ensure the model is pre-downloaded to the Hugging Face cache or pass "
            "a local directory path to the model weights.\n"
            f"Underlying error: {type(exc).__name__}: {exc}"
        )
        logger.error(msg)
        raise RuntimeError(msg) from exc


def embed_texts(
    texts: List[str],
    encoder,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Encode a list of text strings into L2-normalized dense embedding vectors.

    Explicitly performs L2-normalization so that downstream cosine similarity
    reduces to an inner (dot) product.

    Parameters
    ----------
    texts : List[str]
        List of text strings to embed.
    encoder : SentenceTransformer
        The loaded sentence encoder instance.
    batch_size : int, optional
        Batch size for inference, by default 32.
    show_progress_bar : bool, optional
        Whether to display embedding progress bar, by default False.

    Returns
    -------
    np.ndarray
        2D float32 array of shape (len(texts), embedding_dim) with unit L2 norm.
    """
    if not texts:
        if hasattr(encoder, "get_embedding_dimension"):
            dim = encoder.get_embedding_dimension()
        elif hasattr(encoder, "get_sentence_embedding_dimension"):
            dim = encoder.get_sentence_embedding_dimension()
        else:
            dim = 384
        return np.empty((0, dim), dtype=np.float32)

    # Encode texts using SentenceTransformer (or compatible mock/callable)
    if hasattr(encoder, "encode"):
        raw_embeddings = encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=False,  # We normalize explicitly below
        )
    elif callable(encoder):
        raw_embeddings = encoder(texts)
    else:
        raise TypeError(f"Unsupported encoder type: {type(encoder)}")

    embeddings = np.asarray(raw_embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    # Explicit L2 normalization: v / max(||v||_2, 1e-12)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normalized_embeddings = embeddings / norms

    return normalized_embeddings.astype(np.float32)
