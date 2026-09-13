"""Entity-aware tokenizer for BM25 sparse retrieval (M3.P3.1.F1).

Protects alphanumeric device models, operating system versions, and PII tokens
as single atomic tokens, preventing BM25 from splitting on hyphens, spaces, or periods.
Reuses entity patterns defined in src/nlp/entity_patterns.py.
"""

import re
from typing import List, Tuple
from src.nlp.entity_patterns import ALL_PROTECTED_PATTERNS


def _find_entity_spans(text: str) -> List[Tuple[int, int]]:
    """Identify all non-overlapping protected entity spans within the input text.

    If multiple patterns match overlapping spans, preference is given to the
    longest match starting at the earliest position.

    Parameters
    ----------
    text : str
        Input string to scan for entity patterns.

    Returns
    -------
    List[Tuple[int, int]]
        Sorted list of non-overlapping (start, end) character offsets.
    """
    candidates: List[Tuple[int, int]] = []
    for pattern in ALL_PROTECTED_PATTERNS:
        for match in pattern.finditer(text):
            candidates.append((match.start(), match.end()))

    if not candidates:
        return []

    # Sort candidates by start ascending, then by span length descending
    candidates.sort(key=lambda span: (span[0], -(span[1] - span[0])))

    # Greedily select non-overlapping spans
    selected_spans: List[Tuple[int, int]] = []
    for start, end in candidates:
        # Check for overlap with any already selected span
        overlap = False
        for s_start, s_end in selected_spans:
            if not (end <= s_start or start >= s_end):
                overlap = True
                break
        if not overlap:
            selected_spans.append((start, end))

    # Return sorted by starting index
    selected_spans.sort(key=lambda span: span[0])
    return selected_spans


def tokenize(text: str) -> List[str]:
    """Tokenize input text while preserving protected device and OS entities as atomic tokens.

    Protected entity spans (e.g. "SM-T280", "iOS 11.1", "iPhone 11 Pro Max") are
    kept intact as single tokens (lowercased with normalized internal spaces).
    Punctuation immediately adjacent to entity boundaries (such as "SM-T280," or
    "(iOS 11.1)") is stripped and does not contaminate the protected entity token.
    All non-entity spans undergo standard lowercased alphanumeric word tokenization.

    Parameters
    ----------
    text : str
        Raw input string (customer query or document text).

    Returns
    -------
    List[str]
        List of atomic tokens for indexing or retrieval.
    """
    if not text:
        return []

    entity_spans = _find_entity_spans(text)
    tokens: List[str] = []
    curr_pos = 0

    for start, end in entity_spans:
        # Process preceding non-entity text
        if start > curr_pos:
            non_entity_text = text[curr_pos:start]
            tokens.extend(re.findall(r"[a-z0-9]+", non_entity_text.lower()))

        # Process the protected entity span as a single atomic token
        entity_token = re.sub(r"\s+", " ", text[start:end].strip()).lower()
        if entity_token:
            tokens.append(entity_token)

        curr_pos = end

    # Process any remaining non-entity text
    if curr_pos < len(text):
        non_entity_text = text[curr_pos:]
        tokens.extend(re.findall(r"[a-z0-9]+", non_entity_text.lower()))

    return tokens
