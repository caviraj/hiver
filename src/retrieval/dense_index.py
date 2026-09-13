"""Dense Semantic Retrieval Index implementation (M3.P3.2.F1).

Provides building, querying, and persistent serialization with corpus hash
staleness detection for dense vector similarity search over conversation resolutions.
Supports FAISS IndexFlatIP with automatic fallback to NumPy brute-force cosine search.
"""

import logging
from pathlib import Path
import pickle
from typing import Any, List, Optional, Set, Tuple, Union
import numpy as np

from src.retrieval.embedding_model import DEFAULT_MODEL_NAME, embed_texts
from src.retrieval.hashing import compute_corpus_hash
from src.retrieval.schema import RetrievalDocument, RetrievalResult

logger = logging.getLogger(__name__)

_FAISS_WARNED = False


class DenseIndex:
    """Container for dense embeddings, FAISS or fallback search index, and metadata.

    Parameters
    ----------
    documents : List[RetrievalDocument]
        List of indexed documents.
    embeddings : np.ndarray
        2D float32 array of shape (N, dim) with unit L2 norm.
    corpus_hash : str
        SHA-256 digest of the indexed corpus.
    model_name : str
        Encoder model name or path used to produce embeddings.
    use_faiss : bool
        Whether FAISS index was attempted/enabled.
    faiss_index : Optional[Any]
        Instantiated FAISS index (IndexFlatIP) if available, otherwise None.
    near_duplicate_count : int
        Number of document pairs having pairwise cosine similarity > 0.98.
    """

    def __init__(
        self,
        documents: List[RetrievalDocument],
        embeddings: np.ndarray,
        corpus_hash: str,
        model_name: str = DEFAULT_MODEL_NAME,
        use_faiss: bool = True,
        faiss_index: Optional[Any] = None,
        near_duplicate_count: int = 0,
    ) -> None:
        self.documents = documents
        self.corpus = documents  # Alias for consistency with BM25Index
        self.embeddings = embeddings
        self.corpus_hash = corpus_hash
        self.model_name = model_name
        self.use_faiss = use_faiss
        self.faiss_index = faiss_index
        self.near_duplicate_count = near_duplicate_count

    def get_corpus_doc_ids(self) -> Set[str]:
        """Return the set of document IDs present in this index."""
        return {doc.doc_id for doc in self.documents}

    @property
    def doc_ids(self) -> List[str]:
        """Return the list of document IDs present in this index."""
        return [doc.doc_id for doc in self.documents]

    def __len__(self) -> int:
        return len(self.documents)


def get_corpus_doc_ids(index: Any) -> Set[str]:
    """Retrieve the set of document IDs indexed by a retrieval index.

    Exposed for P3.3 Reciprocal Rank Fusion to verify that sparse and dense
    tracks share an identical corpus before fusing rankings.

    Parameters
    ----------
    index : Any
        A DenseIndex, BM25Index, or any object with a `documents` or `corpus` attribute.

    Returns
    -------
    Set[str]
        Set of all document IDs in the index.
    """
    if hasattr(index, "get_corpus_doc_ids") and callable(index.get_corpus_doc_ids):
        return index.get_corpus_doc_ids()
    docs = getattr(index, "documents", None) or getattr(index, "corpus", None)
    if docs is None:
        raise AttributeError(
            f"Index object of type {type(index).__name__} does not expose 'documents' or 'corpus'."
        )
    return {doc.doc_id for doc in docs}


def _create_faiss_index(embeddings: np.ndarray) -> Optional[Any]:
    """Attempt to instantiate and populate a FAISS IndexFlatIP index.

    Returns None if faiss is unavailable.
    """
    global _FAISS_WARNED
    try:
        import faiss  # type: ignore

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        return index
    except (ImportError, Exception) as exc:
        if not _FAISS_WARNED:
            logger.warning(
                "FAISS is not available or failed to load (%s); falling back to "
                "brute-force NumPy cosine similarity search. Ranked results are functionally "
                "equivalent, but search may not scale to very large corpora.",
                exc,
            )
            _FAISS_WARNED = True
        return None


def _count_near_duplicates(embeddings: np.ndarray, threshold: float = 0.98) -> int:
    """Count pairs of documents with cosine similarity exceeding threshold."""
    n = len(embeddings)
    if n <= 1:
        return 0
    sim_matrix = np.dot(embeddings, embeddings.T)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    return int(np.sum(upper_tri > threshold))


def build_dense_index(
    documents: List[RetrievalDocument],
    encoder: Any,
    use_faiss: bool = True,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
) -> DenseIndex:
    """Build a DenseIndex over a list of RetrievalDocument objects.

    Uses the IDENTICAL RetrievalDocument list as BM25 sparse retrieval.
    Embeds each document's query_text with explicit L2 normalization.

    Parameters
    ----------
    documents : List[RetrievalDocument]
        List of documents to index.
    encoder : Any
        SentenceTransformer encoder model or embedding callable.
    use_faiss : bool, optional
        Whether to attempt FAISS indexing, by default True.
    model_name : str, optional
        Name of the model being used, by default DEFAULT_MODEL_NAME.
    batch_size : int, optional
        Batch size for text encoding, by default 32.

    Returns
    -------
    DenseIndex
        The fitted dense index.

    Raises
    ------
    ValueError
        If the documents list is empty.
    """
    if not documents:
        raise ValueError("Cannot build a dense index on an empty document corpus.")

    logger.info("Building dense index over %d documents using '%s'", len(documents), model_name)

    # Embed all document query_texts
    query_texts = [doc.query_text for doc in documents]
    embeddings = embed_texts(query_texts, encoder, batch_size=batch_size)

    corpus_hash = compute_corpus_hash(documents)

    # Near-duplicate tracking
    near_dup_count = _count_near_duplicates(embeddings, threshold=0.98)
    if near_dup_count > 0:
        logger.warning(
            "Found %d near-duplicate query embeddings detected (cosine similarity > 0.98). "
            "Retaining all entries per corpus consistency constraint.",
            near_dup_count,
        )

    # Instantiate FAISS or fallback
    faiss_index = _create_faiss_index(embeddings) if use_faiss else None

    return DenseIndex(
        documents=documents,
        embeddings=embeddings,
        corpus_hash=corpus_hash,
        model_name=model_name,
        use_faiss=(faiss_index is not None),
        faiss_index=faiss_index,
        near_duplicate_count=near_dup_count,
    )


def query_dense_index(
    index: DenseIndex,
    query_text: str,
    encoder: Any,
    top_k: int = 10,
) -> List[RetrievalResult]:
    """Query a DenseIndex with natural language text and return ranked results.

    Parameters
    ----------
    index : DenseIndex
        The populated DenseIndex.
    query_text : str
        Incoming user query string.
    encoder : Any
        The sentence encoder matching the index's embedding dimension and space.
    top_k : int, optional
        Maximum number of ranked results to return, by default 10.

    Returns
    -------
    List[RetrievalResult]
        Ranked list of RetrievalResult objects ordered by cosine similarity descending.
    """
    if not query_text or not query_text.strip():
        logger.debug("Received empty query text; returning empty results list.")
        return []

    cleaned_query = query_text.strip()
    corpus_size = len(index.documents)
    if corpus_size == 0:
        return []

    # Check max sequence length truncation
    max_len = getattr(encoder, "max_seq_length", None)
    tokenizer = getattr(encoder, "tokenizer", None)

    if max_len is not None:
        if tokenizer is not None:
            try:
                tokens = tokenizer.encode(cleaned_query, add_special_tokens=False)
                if len(tokens) > max_len:
                    logger.warning(
                        "Query text token length (%d) exceeds encoder max sequence length (%d); "
                        "truncating query to %d tokens.",
                        len(tokens),
                        max_len,
                        max_len,
                    )
                    truncated_tokens = tokens[:max_len]
                    cleaned_query = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            except Exception as exc:
                logger.debug("Tokenizer truncation check failed (%s); checking char length", exc)
        elif len(cleaned_query) > max_len:
            logger.warning(
                "Query text character length (%d) exceeds encoder max sequence length (%d); "
                "truncating query.",
                len(cleaned_query),
                max_len,
            )
            cleaned_query = cleaned_query[:max_len]

    k = min(top_k, corpus_size)
    if k <= 0:
        return []

    query_vec = embed_texts([cleaned_query], encoder)  # Shape (1, dim), L2-normalized

    if index.faiss_index is not None:
        scores, indices = index.faiss_index.search(
            np.ascontiguousarray(query_vec, dtype=np.float32), k
        )
        retrieved_scores = scores[0]
        retrieved_indices = indices[0]
    else:
        # Fallback: Dot product of normalized vectors = Cosine similarity
        sims = np.dot(index.embeddings, query_vec.T).squeeze(axis=1)  # Shape (N,)
        top_indices = np.argsort(-sims)[:k]
        retrieved_indices = top_indices
        retrieved_scores = sims[top_indices]

    results: List[RetrievalResult] = []
    for rank, (idx, score) in enumerate(zip(retrieved_indices, retrieved_scores), start=1):
        if idx < 0 or idx >= corpus_size:
            continue
        doc = index.documents[idx]
        results.append(
            RetrievalResult(
                doc_id=doc.doc_id,
                score=float(score),
                rank=rank,
            )
        )

    return results




def save_dense_index(
    index: DenseIndex,
    corpus: List[RetrievalDocument],
    path: Union[str, Path],
) -> None:
    """Persist a DenseIndex and associated corpus to disk.

    Verifies that the index's corpus hash matches the current corpus hash
    before serializing to prevent saving mismatched or stale states.

    Parameters
    ----------
    index : DenseIndex
        DenseIndex instance to persist.
    corpus : List[RetrievalDocument]
        The ground truth documents corresponding to this index.
    path : Union[str, Path]
        Target directory or file path for the .pkl artifact.

    Raises
    ------
    ValueError
        If index corpus hash does not match provided corpus hash.
    """
    p = Path(path)
    if p.is_dir() or not p.suffix:
        p.mkdir(parents=True, exist_ok=True)
        file_path = p / "dense_index.pkl"
    else:
        file_path = p
        file_path.parent.mkdir(parents=True, exist_ok=True)

    current_hash = compute_corpus_hash(corpus)
    if index.corpus_hash != current_hash:
        raise ValueError(
            f"Corpus hash mismatch during save: Index corpus hash '{index.corpus_hash[:12]}' "
            f"does not match provided corpus hash '{current_hash[:12]}'. Cannot save inconsistent index."
        )

    payload = {
        "embeddings": index.embeddings,
        "corpus": [doc.model_dump() for doc in corpus],
        "corpus_hash": current_hash,
        "model_name": index.model_name,
        "near_duplicate_count": index.near_duplicate_count,
    }

    with open(file_path, "wb") as f:
        pickle.dump(payload, f)

    logger.info("Saved Dense index and corpus to %s (hash=%s)", file_path, current_hash[:12])


def load_dense_index(
    path: Union[str, Path],
    expected_corpus: Optional[List[RetrievalDocument]] = None,
    use_faiss: bool = True,
) -> Tuple[DenseIndex, List[RetrievalDocument]]:
    """Load a persisted DenseIndex and corpus from disk with staleness detection.

    Parameters
    ----------
    path : Union[str, Path]
        Directory path or file path where the dense index was saved.
    expected_corpus : Optional[List[RetrievalDocument]], default=None
        If provided, its content hash is verified against the saved index hash.
        If they differ, a ValueError is raised to prevent serving stale knowledge.
    use_faiss : bool, default=True
        Whether to attempt initializing a FAISS index from the loaded embeddings.

    Returns
    -------
    Tuple[DenseIndex, List[RetrievalDocument]]
        Loaded (DenseIndex, List[RetrievalDocument]).

    Raises
    ------
    FileNotFoundError
        If the index file does not exist.
    ValueError
        If expected_corpus is provided and its hash differs from saved index hash.
    """
    p = Path(path)
    if p.is_dir():
        file_path = p / "dense_index.pkl"
    else:
        file_path = p

    if not file_path.exists():
        raise FileNotFoundError(f"Dense index file not found at: {file_path}")

    with open(file_path, "rb") as f:
        payload = pickle.load(f)

    saved_corpus_hash = payload["corpus_hash"]
    raw_corpus = payload["corpus"]
    corpus = [
        RetrievalDocument.model_validate(doc_dict) if isinstance(doc_dict, dict) else doc_dict
        for doc_dict in raw_corpus
    ]

    if expected_corpus is not None:
        expected_hash = compute_corpus_hash(expected_corpus)
        if expected_hash != saved_corpus_hash:
            raise ValueError(
                f"Corpus hash mismatch. Dense index is stale: Saved corpus hash '{saved_corpus_hash[:12]}' "
                f"does not match expected corpus hash '{expected_hash[:12]}'. "
                "The underlying corpus has changed since the index was built."
            )

    embeddings = payload["embeddings"]
    faiss_index = _create_faiss_index(embeddings) if use_faiss else None

    index = DenseIndex(
        documents=corpus,
        embeddings=embeddings,
        corpus_hash=saved_corpus_hash,
        model_name=payload.get("model_name", DEFAULT_MODEL_NAME),
        use_faiss=(faiss_index is not None),
        faiss_index=faiss_index,
        near_duplicate_count=payload.get("near_duplicate_count", 0),
    )

    logger.info("Successfully loaded Dense index from %s", file_path)
    return index, corpus


# Provide aliases matching bm25_index naming convention
save_index = save_dense_index
load_index = load_dense_index
