"""
PII Detection & Masking
=======================

Implements unconditional PII detection and redaction for customer service LLM responses.
Enforces Luhn algorithm checksum validation on candidate credit card numbers to
prevent false positives on order IDs, tracking codes, and long ticket numbers.
"""

import re
from typing import List, Tuple


def luhn_checksum(digits: str) -> bool:
    """
    Validate whether a numeric string satisfies the Luhn algorithm (mod 10).

    Args:
        digits: String containing numeric digits.

    Returns:
        bool: True if the checksum matches the Luhn formula, False otherwise.
    """
    clean_digits = re.sub(r"\D", "", str(digits or ""))
    if not clean_digits or len(clean_digits) < 2:
        return False

    total = 0
    reverse_digits = clean_digits[::-1]
    for idx, char in enumerate(reverse_digits):
        d = int(char)
        if idx % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d

    return total % 10 == 0


# SSN regex pattern with prefix (SSN, Social Security, Tax ID)
SSN_PREFIX_PATTERN = re.compile(
    r"\b(?P<prefix>(?:SSN|Social Security(?: Number)?|Tax ID)(?:\s+is|\s*[:#=])?\s*)(?P<number>\d{3}[-\s]?\d{2}[-\s]?\d{4}|\d{9})\b",
    re.IGNORECASE,
)

# Standard SSN pattern (XXX-XX-XXXX or XXX XX XXXX) without requiring prefix
SSN_STANDARD_PATTERN = re.compile(
    r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"
)

# Potential Credit Card candidate pattern (13 to 19 digits formatted or unformatted)
# Matches 4x4 groups, 4-6-5 (Amex), or continuous blocks of 13-19 digits.
CC_CANDIDATE_PATTERN = re.compile(
    r"\b(?:\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}"
    r"|\d{4}[-\s]\d{6}[-\s]\d{5}"
    r"|\d{13,19})\b"
)


def mask_pii(text: str) -> Tuple[str, List[str]]:
    """
    Detect and mask personally identifiable information (PII) in text.
    Must run unconditionally and first before any downstream schema or validator checks.

    Supported PII Types:
      - ssn: Social Security Numbers (masked as [REDACTED_SSN])
      - credit_card: Luhn-validated credit/debit card numbers (masked as [REDACTED_CREDIT_CARD])

    Args:
        text: Input string potentially containing sensitive personal data.

    Returns:
        Tuple[str, List[str]]: Masked text and a list of detected PII types (e.g. ['ssn', 'credit_card']).
    """
    if not text:
        return "", []

    pii_types_found: List[str] = []
    masked_text = text

    # 1. Mask SSN with prefix first
    def _ssn_prefix_replacer(match: re.Match) -> str:
        if "ssn" not in pii_types_found:
            pii_types_found.append("ssn")
        prefix = match.group("prefix") or ""
        return f"{prefix}[REDACTED_SSN]"

    masked_text = SSN_PREFIX_PATTERN.sub(_ssn_prefix_replacer, masked_text)

    # Standard SSN pattern without prefix
    def _ssn_std_replacer(match: re.Match) -> str:
        if "ssn" not in pii_types_found:
            pii_types_found.append("ssn")
        return "[REDACTED_SSN]"

    masked_text = SSN_STANDARD_PATTERN.sub(_ssn_std_replacer, masked_text)

    # 2. Mask Credit Card numbers (with Luhn checksum validation)
    candidates = list(CC_CANDIDATE_PATTERN.finditer(masked_text))
    for match in reversed(candidates):
        matched_str = match.group(0)
        digits_only = re.sub(r"\D", "", matched_str)
        if luhn_checksum(digits_only):
            if "credit_card" not in pii_types_found:
                pii_types_found.append("credit_card")
            start, end = match.span()
            masked_text = (
                masked_text[:start] + "[REDACTED_CREDIT_CARD]" + masked_text[end:]
            )

    return masked_text, pii_types_found
