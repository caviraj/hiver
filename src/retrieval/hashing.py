"""Corpus hashing utilities for retrieval indices (M3.P3.1 / M3.P3.2).

Provides deterministic corpus-level SHA-256 hashing to detect index staleness
across BM25 sparse and dense semantic retrieval tracks.
"""

import hashlib
from typing import List

from src.retrieval.schema import RetrievalDocument


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
