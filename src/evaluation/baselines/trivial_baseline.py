"""Trivial baseline responder using regex word-boundary keyword matching.

Represents the absolute floor-level capability for customer support:
matches specific keywords and directs to a static support URL, or
honestly declines to respond (returns None) when no trigger keyword is found.
"""

from __future__ import annotations

import re
from typing import List, Optional

DEFAULT_KEYWORDS: List[str] = ["frozen", "broken", "help"]
DEFAULT_SUPPORT_URL: str = "https://support.apple.com"


def trivial_baseline_response(
    tweet_text: str,
    trigger_keywords: Optional[List[str]] = None,
    support_url: str = DEFAULT_SUPPORT_URL,
) -> Optional[str]:
    """Generate a trivial static response if trigger keywords are matched.

    Uses word-boundary regex matching (case-insensitive) so that keywords
    do not trigger on substrings (e.g., 'help' will not match inside 'helpful').

    Args:
        tweet_text: Inbound customer tweet or message text.
        trigger_keywords: List of keyword triggers. Defaults to ['frozen', 'broken', 'help'].
        support_url: Static support URL to embed in the response.

    Returns:
        A static templated response string if any keyword matches, or None
        if no trigger keywords match (representing an honest decline).
    """
    if not tweet_text or not tweet_text.strip():
        return None

    keywords = trigger_keywords if trigger_keywords is not None else DEFAULT_KEYWORDS

    for kw in keywords:
        clean_kw = kw.strip()
        if not clean_kw:
            continue
        pattern = rf"\b{re.escape(clean_kw)}\b"
        if re.search(pattern, tweet_text, flags=re.IGNORECASE):
            return (
                f"I am an automated assistant. Please visit our support center at "
                f"{support_url} for assistance with your device."
            )

    return None
