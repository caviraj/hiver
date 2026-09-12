"""Lexical de-contraction module for Phase 1.2 (NLP Normalization Pipeline).

Performs word-boundary-aware, case-preserving expansion of English contractions
and social-media shorthand in conversational tweet texts prior to entity masking.
"""

import argparse
import logging
from pathlib import Path
import re
import sys
from typing import Union

from src.data.thread_schema import Thread
from src.nlp.contraction_map import CONTRACTIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_contraction_pattern(key: str) -> str:
    """Build a regex pattern for a contraction key allowing standard and curly apostrophes.

    Replaces any apostrophe in key with ['’‘] to match straight or smart quotes.
    """
    parts = key.split("'")
    escaped_parts = [re.escape(p) for p in parts]
    return r"['’‘]".join(escaped_parts)


# Build regular expression:
# 1. Sort keys descending by length so compound contractions like "couldn't've"
#    match before single-layer contractions like "couldn't".
_sorted_keys = sorted(CONTRACTIONS.keys(), key=len, reverse=True)
_contraction_alts = [_build_contraction_pattern(k) for k in _sorted_keys]
_contraction_group = "|".join(_contraction_alts)

# 2. Word boundary lookaround:
#    Contractions must not be immediately preceded or followed by alphanumeric chars or apostrophes.
#    This protects possessive apostrophes (e.g., "Apple's policy" has 's which is not in CONTRACTIONS,
#    and "Spirit's" will not match "it's" because 'r' precedes it).
#    Also handles emoji and non-ASCII punctuation boundaries cleanly.
_CONTRACTION_PATTERN = (
    rf"(?<![a-zA-Z0-9'’‘])(?:{_contraction_group})(?![a-zA-Z0-9'’‘])"
)

# 3. Defensive URL pattern:
#    Matches URLs starting with http://, https://, or www. to protect embedded slugs/tokens.
_URL_PATTERN = r"(?:https?://|www\.)\S+"

# Master compiled regex combining URL protection and contraction expansion
_MASTER_REGEX = re.compile(
    rf"(?P<url>{_URL_PATTERN})|(?P<contraction>{_CONTRACTION_PATTERN})",
    flags=re.IGNORECASE,
)


def _preserve_casing(matched_text: str, replacement: str) -> str:
    """Preserve leading-capital or all-caps casing pattern from matched text onto replacement.

    Parameters
    ----------
    matched_text : str
        The raw contraction matched from the original text (e.g., "Can't", "CAN'T", "'Bout").
    replacement : str
        The target expanded string in lowercase (e.g., "cannot", "about").

    Returns
    -------
    str
        Replacement string adapted to match original casing style.
    """
    if not replacement:
        return replacement

    alpha_chars = [c for c in matched_text if c.isalpha()]
    if not alpha_chars:
        return replacement

    # If entire match is uppercase (e.g., "CAN'T", "I'LL", "WON'T")
    if matched_text.isupper():
        return replacement.upper()

    # If leading alphabetical character is uppercase (e.g., "Can't", "Won't", "'Bout")
    if alpha_chars[0].isupper():
        for i, ch in enumerate(replacement):
            if ch.isalpha():
                return replacement[:i] + ch.upper() + replacement[i + 1 :]
        return replacement

    return replacement.lower()


def expand_contractions(text: str) -> str:
    """Expand contractions and social-media shorthand in a string.

    Preserves case-style (leading capital vs lowercase vs all-caps) and protects
    possessive apostrophes and URL slugs. Idempotent on repeated calls.

    Parameters
    ----------
    text : str
        Input raw text string.

    Returns
    -------
    str
        Expanded text string.
    """
    if not text or not text.strip():
        return text

    def _replace_match(match: re.Match) -> str:
        # If matched a URL, return it untouched
        if match.group("url"):
            return match.group(0)

        raw_match = match.group("contraction")
        # Normalize any smart/curly apostrophes to standard single quote for dictionary lookup
        normalized_key = re.sub(r"[’‘]", "'", raw_match).lower()
        replacement = CONTRACTIONS.get(normalized_key)

        if replacement is None:
            return match.group(0)

        return _preserve_casing(raw_match, replacement)

    return _MASTER_REGEX.sub(_replace_match, text)


def expand_thread(thread: Thread) -> Thread:
    """Expand contractions in all tweets of a thread without in-place mutation.

    Parameters
    ----------
    thread : Thread
        Original Thread instance.

    Returns
    -------
    Thread
        New Thread instance with de-contracted tweet texts.
    """
    new_tweets = [
        tweet.model_copy(update={"text": expand_contractions(tweet.text)})
        for tweet in thread.tweets
    ]
    return thread.model_copy(update={"tweets": new_tweets})


def run(
    input_path: Union[str, Path] = "data/processed/apple_support_threads.jsonl",
    output_path: Union[str, Path] = "data/processed/apple_support_threads_decontracted.jsonl",
) -> int:
    """Read conversation threads JSONL, de-contract all tweet texts, and write to output JSONL.

    Streams line-by-line to minimize peak memory consumption.

    Parameters
    ----------
    input_path : Union[str, Path]
        Path to input threads JSONL.
    output_path : Union[str, Path]
        Path to output de-contracted threads JSONL.

    Returns
    -------
    int
        Number of processed threads.
    """
    src = Path(input_path)
    dest = Path(output_path)

    if not src.exists():
        raise FileNotFoundError(f"Input thread file not found: {src}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    logger.info("De-contracting threads from %s -> %s", src, dest)
    with open(src, "r", encoding="utf-8") as in_f, open(
        dest, "w", encoding="utf-8"
    ) as out_f:
        for line_num, line in enumerate(in_f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                thread = Thread.model_validate_json(line_str)
                expanded = expand_thread(thread)
                out_f.write(expanded.model_dump_json() + "\n")
                count += 1
            except Exception as err:
                logger.error(
                    "Failed processing thread at line %d: %s", line_num, err
                )
                raise

    logger.info("Successfully de-contracted %d threads to %s", count, dest)
    return count


def main() -> None:
    """CLI entrypoint for de-contraction feature."""
    parser = argparse.ArgumentParser(
        description="M1.P1.2.F1: Lexical De-contraction of conversation threads"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/apple_support_threads.jsonl",
        help="Input JSONL threads file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/apple_support_threads_decontracted.jsonl",
        help="Output de-contracted JSONL threads file path",
    )
    args = parser.parse_args()

    try:
        run(input_path=args.input, output_path=args.output)
    except Exception as e:
        logger.error("De-contraction pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
