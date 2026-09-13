"""Retrieval module for Hybrid RAG Architecture (Milestone 3).

Provides schemas, corpus building, entity-aware tokenization, and BM25 indexing.
"""

from src.retrieval.schema import RetrievalDocument, RetrievalResult
from src.retrieval.corpus_builder import build_corpus
from src.retrieval.tokenizer import tokenize
from src.retrieval.bm25_index import BM25Index, build_index, query_index, save_index, load_index

__all__ = [
    "RetrievalDocument",
    "RetrievalResult",
    "build_corpus",
    "tokenize",
    "BM25Index",
    "build_index",
    "query_index",
    "save_index",
    "load_index",
]
