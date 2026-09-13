"""BM25 Sparse Retrieval Index implementation (M3.P3.1.F1).

Provides building, querying, and persistent serialization with corpus hash
staleness detection for BM25Okapi index over conversation resolutions.
"""

import hashlib
import logging
from pathlib import Path
import pickle
from typing import List, Optional, Tuple, Union

from rank_bm25 import BM25Okapi

from src.retrieval.schema import RetrievalDocument, RetrievalResult
from src.retrieval.tokenizer import tokenize

logger = logging.getLogger(__name__)


def compute_corpus_hash(documents: List[RetrievalDocument]) -> str:
    """Compute a deterministic SHA-256 hash representing the corpus content.

    Sorts documents by doc_id to ensure order-invariant hashing across rebuilds.

    Parameters
    ----------
    documents : List[RetrievalDocument]
        List of documents forming the corpus.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest of sorted doc_ids, query_texts, and resolution_texts.
    """
    hasher = hashlib.sha256()
    sorted_docs = sorted(documents, key=lambda d: d.doc_id)
    for doc in sorted_docs:
        hasher.update(doc.doc_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(doc.query_text.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(doc.resolution_text.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


class BM25Index:
    """Container for the BM25Okapi model, corpus, and metadata.

    Parameters
    ----------
    bm25 : BM25Okapi
        Fitted BM25Okapi scoring model.
    corpus : List[RetrievalDocument]
        The indexed ground truth documents.
    corpus_hash : str
        SHA-256 digest of the indexed corpus.
    k1 : float
        Term frequency saturation parameter.
    b : float
        Document length normalization parameter.
    """

    def __init__(
        self,
        bm25: BM25Okapi,
        corpus: List[RetrievalDocument],
        corpus_hash: str,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.bm25 = bm25
        self.corpus = corpus
        self.corpus_hash = corpus_hash
        self.k1 = k1
        self.b = b
        self.doc_ids = [doc.doc_id for doc in corpus]


def build_index(
    documents: List[RetrievalDocument],
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Index:
    """Build a BM25Okapi index from a list of RetrievalDocument objects.

    Parameters
    ----------
    documents : List[RetrievalDocument]
        Non-empty list of grounded retrieval documents to index.
    k1 : float, default=1.5
        BM25 term frequency saturation parameter. Standard default is 1.5:
        controls how quickly term frequency contribution saturates. Higher values
        allow repeated query terms to increase document score further, while lower
        values (e.g. 1.2) cause score to saturate rapidly approaching binary matching.
    b : float, default=0.75
        BM25 document length normalization parameter. Standard default is 0.75:
        penalizes longer documents relative to average document length in the corpus,
        preventing verbose documents from dominating short, precise inquiries.

    Returns
    -------
    BM25Index
        Initialized and fitted BM25Index instance.

    Raises
    ------
    ValueError
        If `documents` is empty.
    """
    if not documents:
        raise ValueError("Cannot build BM25 index on an empty document corpus.")

    # Tokenize every document's query_text with entity-aware tokenizer
    tokenized_corpus: List[List[str]] = [tokenize(doc.query_text) for doc in documents]

    # Initialize BM25Okapi model
    bm25_model = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    corpus_hash = compute_corpus_hash(documents)
    logger.info(
        "Built BM25 index for %d documents (k1=%.2f, b=%.2f, corpus_hash=%s)",
        len(documents),
        k1,
        b,
        corpus_hash[:12],
    )

    return BM25Index(
        bm25=bm25_model,
        corpus=documents,
        corpus_hash=corpus_hash,
        k1=k1,
        b=b,
    )


def query_index(
    index: BM25Index,
    query_text: str,
    top_k: int = 10,
) -> List[RetrievalResult]:
    """Retrieve ranked RetrievalResult candidates for a customer query string.

    Parameters
    ----------
    index : BM25Index
        Fitted BM25Index instance.
    query_text : str
        Incoming customer query string to score against the corpus.
    top_k : int, default=10
        Maximum number of ranked candidates to return.

    Returns
    -------
    List[RetrievalResult]
        Rank-ordered list of scored candidates, with 1-indexed `rank` (1..top_k).
        Returns empty list if query_text is empty or contains no tokens.
    """
    if not query_text or not query_text.strip():
        return []

    tokens = tokenize(query_text)
    if not tokens:
        return []

    # Get BM25 scores across all corpus documents
    scores = index.bm25.get_scores(tokens)

    k = min(top_k, len(index.corpus))
    if k <= 0:
        return []

    # Sort indices descending by score
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    top_matches = indexed_scores[:k]

    results: List[RetrievalResult] = []
    for rank, (doc_idx, score) in enumerate(top_matches, start=1):
        results.append(
            RetrievalResult(
                doc_id=index.doc_ids[doc_idx],
                score=float(score),
                rank=rank,
            )
        )

    return results


def save_index(
    index: BM25Index,
    corpus: List[RetrievalDocument],
    path: Union[str, Path],
) -> None:
    """Serialize the BM25Index, corpus, and metadata to disk.

    Parameters
    ----------
    index : BM25Index
        The fitted index to save.
    corpus : List[RetrievalDocument]
        The ground truth documents associated with this index.
    path : Union[str, Path]
        Directory path or target file path to save the index bundle.
    """
    p = Path(path)
    if p.suffix == "":
        p.mkdir(parents=True, exist_ok=True)
        file_path = p / "bm25_index.pkl"
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        file_path = p

    corpus_hash = compute_corpus_hash(corpus)

    payload = {
        "bm25": index.bm25,
        "corpus": [doc.model_dump() for doc in corpus],
        "corpus_hash": corpus_hash,
        "k1": index.k1,
        "b": index.b,
    }

    with open(file_path, "wb") as f:
        pickle.dump(payload, f)

    logger.info("Saved BM25 index and corpus to %s (hash=%s)", file_path, corpus_hash[:12])


def load_index(
    path: Union[str, Path],
    expected_corpus: Optional[List[RetrievalDocument]] = None,
) -> Tuple[BM25Index, List[RetrievalDocument]]:
    """Load a persisted BM25Index and corpus from disk, with optional staleness detection.

    Parameters
    ----------
    path : Union[str, Path]
        Directory path or file path where the index was saved.
    expected_corpus : Optional[List[RetrievalDocument]], default=None
        If provided, its content hash is verified against the saved index hash.
        If they differ, a ValueError is raised to prevent serving stale knowledge.

    Returns
    -------
    Tuple[BM25Index, List[RetrievalDocument]]
        Loaded (BM25Index, List[RetrievalDocument]).

    Raises
    ------
    FileNotFoundError
        If the index file does not exist.
    ValueError
        If `expected_corpus` is provided and its hash does not match the index hash.
    """
    p = Path(path)
    if p.is_dir():
        file_path = p / "bm25_index.pkl"
    else:
        file_path = p

    if not file_path.exists():
        raise FileNotFoundError(f"Index file not found at: {file_path}")

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
                f"Stale index detected: Saved corpus hash '{saved_corpus_hash[:12]}' "
                f"does not match expected corpus hash '{expected_hash[:12]}'. "
                "The underlying corpus has changed since the index was built."
            )

    index = BM25Index(
        bm25=payload["bm25"],
        corpus=corpus,
        corpus_hash=saved_corpus_hash,
        k1=payload["k1"],
        b=payload["b"],
    )

    logger.info("Successfully loaded BM25 index from %s", file_path)
    return index, corpus
