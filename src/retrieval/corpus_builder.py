"""Corpus builder for sparse and dense retrieval (M3.P3.1.F1).

Extracts grounded RetrievalDocument instances from reconstructed conversation threads
where a customer-initiated query is resolved by a brand response.
Logs exact-duplicate resolution_text occurrences without deduplicating documents.
"""

from collections import Counter
import logging
from typing import List, Optional

from src.data.thread_schema import Thread
from src.retrieval.schema import RetrievalDocument

logger = logging.getLogger(__name__)


def build_corpus(threads: List[Thread]) -> List[RetrievalDocument]:
    """Build a retrievable document corpus from a list of conversation threads.

    Filters for threads where a customer-initiated message (inbound == True) is
    followed by at least one brand-authored reply (inbound == False).
    The customer message forms `query_text` and the brand's reply forms `resolution_text`.
    Threads lacking brand resolution are excluded.

    Duplicate resolution texts (e.g. canned agent responses) are retained in full
    to preserve distinct customer query contexts, but their occurrence counts
    are logged.

    Parameters
    ----------
    threads : List[Thread]
        List of reconstructed conversation Thread objects.

    Returns
    -------
    List[RetrievalDocument]
        List of valid, groundable RetrievalDocument objects.
    """
    documents: List[RetrievalDocument] = []

    for thread in threads:
        if not thread.tweets:
            continue

        # Must be customer-initiated
        root_tweet = thread.tweets[0]
        if not root_tweet.inbound:
            continue

        # Find the first subsequent brand-authored response
        brand_reply: Optional[str] = None
        for tweet in thread.tweets[1:]:
            if not tweet.inbound:
                brand_reply = tweet.text
                break

        if brand_reply is None or not brand_reply.strip():
            # Exclude threads with no brand resolution
            continue

        documents.append(
            RetrievalDocument(
                doc_id=thread.thread_id,
                query_text=root_tweet.text,
                resolution_text=brand_reply,
            )
        )

    # Detect and log exact-duplicate resolution_text occurrences
    resolution_counts = Counter(doc.resolution_text for doc in documents)
    duplicate_resolutions = {text: count for text, count in resolution_counts.items() if count > 1}

    if duplicate_resolutions:
        total_dup_docs = sum(duplicate_resolutions.values())
        logger.info(
            "Found %d distinct resolution_text strings duplicated across %d documents (canned responses).",
            len(duplicate_resolutions),
            total_dup_docs,
        )
        for text, count in duplicate_resolutions.items():
            preview = (text[:60] + "...") if len(text) > 60 else text
            logger.debug("Duplicate resolution (%d occurrences): %s", count, preview)

    return documents
