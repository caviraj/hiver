"""Schema definitions for retrieval documents and retrieval results (M3.P3.1.F1).

Defines Pydantic models for grounding documents in the retrieval corpus and
rank-ordered retrieval result objects consumed by Reciprocal Rank Fusion (P3.3).
"""

from pydantic import BaseModel, ConfigDict, Field


class RetrievalDocument(BaseModel):
    """Pydantic model representing a grounded document in the retrieval corpus.

    Attributes
    ----------
    doc_id : str
        Unique document identifier, corresponds to the root thread_id.
    query_text : str
        The customer's initial message or inquiry (indexed for sparse retrieval).
    resolution_text : str
        The brand agent's response that provided the resolution.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    doc_id: str = Field(description="Unique document identifier (corresponds to thread_id)")
    query_text: str = Field(description="Customer's initial message / inquiry")
    resolution_text: str = Field(description="Brand's resolution reply")


class RetrievalResult(BaseModel):
    """Pydantic model representing a scored and ranked retrieval output.

    Attributes
    ----------
    doc_id : str
        Unique document identifier matching RetrievalDocument.doc_id.
    score : float
        Retrieval relevance score (e.g. BM25 score).
    rank : int
        1-indexed rank position among retrieved candidates (required for RRF fusion).
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    doc_id: str = Field(description="Unique document identifier matching RetrievalDocument.doc_id")
    score: float = Field(description="Relevance score (e.g. BM25 score)")
    rank: int = Field(ge=1, description="1-indexed rank position among retrieved candidates")
