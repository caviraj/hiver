"""
Identifier Structural Validation
================================

Validates generated customer and ticket identifiers against expected structural regexes
to catch hallucinated, truncated, or malformed IDs before forwarding to backend ticketing APIs.
"""

import re
from typing import List, Optional


DEFAULT_TICKET_ID_PATTERN = r"^TCKT-\d{6}$"


def validate_id_format(id_str: str, pattern: str = DEFAULT_TICKET_ID_PATTERN) -> bool:
    """
    Validate whether an identifier string matches the expected regex pattern.

    Args:
        id_str: Identifier string to test (e.g. 'TCKT-123456').
        pattern: Regular expression pattern string (default: r"^TCKT-\\d{6}$").

    Returns:
        bool: True if id_str matches pattern exactly, False otherwise.
    """
    if not isinstance(id_str, str) or not id_str.strip():
        return False

    compiled = re.compile(pattern)
    return bool(compiled.match(id_str.strip()))


def validate_ticket_tags(
    tags: List[str],
    id_pattern: str = DEFAULT_TICKET_ID_PATTERN,
    id_tag_prefix: str = "id:",
) -> bool:
    """
    Validate any embedded ID tags inside ticket_tags list.
    Checks tags beginning with `id:` or starting with the expected ticket prefix (e.g. 'TCKT-').

    Args:
        tags: List of string tags.
        id_pattern: Expected pattern for the ID value.
        id_tag_prefix: Prefix denoting an identifier tag (e.g. 'id:').

    Returns:
        bool: True if all ID tags are well-formed (or none present), False otherwise.
    """
    for tag in tags:
        if tag.startswith(id_tag_prefix):
            raw_id = tag[len(id_tag_prefix) :].strip()
            if not validate_id_format(raw_id, id_pattern):
                return False
        elif tag.startswith("TCKT-") or re.match(r"^TCKT-", tag, re.IGNORECASE):
            if not validate_id_format(tag, id_pattern):
                return False
    return True
