"""Retrieval module for Hybrid RAG Architecture (Milestone 3).

Provides schemas, corpus building, entity-aware tokenization, BM25 sparse indexing,
and transformer-based dense semantic retrieval.
"""

from src.retrieval.schema import RetrievalDocument, RetrievalResult
from src.retrieval.corpus_builder import build_corpus
from src.retrieval.tokenizer import tokenize
from src.retrieval.hashing import compute_corpus_hash
from src.retrieval.bm25_index import (
    BM25Index,
    build_index,
    query_index,
    save_index,
    load_index,
)
from src.retrieval.embedding_model import (
    DEFAULT_MODEL_NAME,
    embed_texts,
    load_encoder,
)
from src.retrieval.dense_index import (
    DenseIndex,
    build_dense_index,
    get_corpus_doc_ids,
    load_dense_index,
    query_dense_index,
    save_dense_index,
)

from src.retrieval.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion
from src.retrieval.hybrid_retriever import CorpusMismatchError, HybridRetriever

__all__ = [
    "RetrievalDocument",
    "RetrievalResult",
    "build_corpus",
    "tokenize",
    "compute_corpus_hash",
    "BM25Index",
    "build_index",
    "query_index",
    "save_index",
    "load_index",
    "DEFAULT_MODEL_NAME",
    "load_encoder",
    "embed_texts",
    "DenseIndex",
    "build_dense_index",
    "query_dense_index",
    "save_dense_index",
    "load_dense_index",
    "get_corpus_doc_ids",
    "DEFAULT_RRF_K",
    "reciprocal_rank_fusion",
    "CorpusMismatchError",
    "HybridRetriever",
]
