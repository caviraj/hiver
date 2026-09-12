"""Entity masking and URL neutralization module for Phase 1.2 (NLP Normalization Pipeline).

Neutralizes URLs to generic `<url>` tokens while preserving device model identifiers,
operating system versions, and pre-masked PII tokens for downstream BM25 sparse
indexing (M3.P3.1) and dense embedding indexing (M3.P3.2).
"""

import argparse
import logging
from pathlib import Path
import re
import sys
from typing import Callable, List, Optional, Tuple, Union

from src.data.thread_schema import Thread
from src.nlp.entity_patterns import (
    ALL_PROTECTED_PATTERNS,
    URL_PATTERN,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Pattern to restore Unicode Private Use Area placeholders
_RESTORE_PATTERN = re.compile(r"\ue000ENT_([0-9]+)\ue000")


def mask_urls(text: str) -> str:
    """Replace all URL occurrences in text with a generic `<url>` token.

    Parameters
    ----------
    text : str
        Input raw or preprocessed text string.

    Returns
    -------
    str
        Text with all matched URLs replaced by `<url>`.
    """
    if not text:
        return text
    return URL_PATTERN.sub("<url>", text)


def _protect_process_restore(
    text: str, process_fn: Callable[[str], str]
) -> str:
    """Protect designated entities, apply a transformation function, and restore them.

    Follows a strict 'protect -> process -> restore' workflow:
    1. Finds all URL spans to prevent protecting entity-like substrings inside URLs.
    2. Collects valid candidate entity spans that do not overlap with any URL.
    3. Resolves overlapping entity matches greedily by longest span first.
    4. Replaces protected spans with collision-proof private-use placeholders.
    5. Applies `process_fn` (e.g. `mask_urls`) to the placeholder-substituted text.
    6. Restores original protected entity strings in a single pass.

    Parameters
    ----------
    text : str
        Input text.
    process_fn : Callable[[str], str]
        Transformation function applied to the text while entities are protected.

    Returns
    -------
    str
        Transformed text with protected entities restored intact.
    """
    if not text:
        return text

    # Step 1: Detect all URL spans
    url_spans: List[Tuple[int, int]] = [
        m.span() for m in URL_PATTERN.finditer(text)
    ]

    # Step 2: Collect candidate protected entity matches outside of URLs
    candidates: List[Tuple[int, int, str]] = []
    for pattern in ALL_PROTECTED_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            # Discard any candidate that intersects with a detected URL
            in_url = any(
                max(start, u_start) < min(end, u_end)
                for u_start, u_end in url_spans
            )
            if not in_url:
                candidates.append((start, end, match.group(0)))

    if not candidates:
        return process_fn(text)

    # Step 3: Resolve overlaps greedily (longest match first)
    candidates.sort(key=lambda c: (c[1] - c[0]), reverse=True)
    reserved_indices: set[int] = set()
    selected_entities: List[Tuple[int, int, str]] = []

    for start, end, matched_str in candidates:
        span_set = set(range(start, end))
        if not (span_set & reserved_indices):
            reserved_indices.update(span_set)
            selected_entities.append((start, end, matched_str))

    # Sort selected entities by start offset ascending
    selected_entities.sort(key=lambda c: c[0])

    # Step 4: Substitute placeholders from right to left to keep offsets stable
    placeholder_map: dict[int, str] = {}
    temp_text = text

    for i, (start, end, matched_str) in reversed(list(enumerate(selected_entities))):
        placeholder_map[i] = matched_str
        placeholder = f"\ue000ENT_{i}\ue000"
        temp_text = temp_text[:start] + placeholder + temp_text[end:]

    # Step 5: Apply processing function (e.g. mask_urls)
    processed_text = process_fn(temp_text)

    # Step 6: Restore protected entity text
    restored_text = _RESTORE_PATTERN.sub(
        lambda m: placeholder_map[int(m.group(1))], processed_text
    )

    return restored_text


def preserve_device_entities(
    text: str, process_fn: Optional[Callable[[str], str]] = None
) -> str:
    """Protect device entities and pre-masked PII before applying text transformations.

    Extracts protected spans first, applies the transformation function to the
    remainder, and restores protected spans intact. If no process_fn is specified,
    defaults to `mask_urls`.

    Parameters
    ----------
    text : str
        Input text.
    process_fn : Optional[Callable[[str], str]], optional
        Transformation function to run between protection and restoration.
        Defaults to `mask_urls`.

    Returns
    -------
    str
        Text with device entities preserved and transformations applied.
    """
    fn = process_fn if process_fn is not None else mask_urls
    return _protect_process_restore(text, fn)


def mask_entities(text: str) -> str:
    """Combined entry point: protect device/PII entities -> mask URLs -> restore.

    Parameters
    ----------
    text : str
        Input raw or de-contracted tweet text.

    Returns
    -------
    str
        Text with URLs neutralized to `<url>` and device models, OS versions,
        and pre-masked PII tokens preserved intact.
    """
    return preserve_device_entities(text, mask_urls)


def mask_thread(thread: Thread) -> Thread:
    """Neutralize URLs and preserve entities in all tweets of a thread immutably.

    Parameters
    ----------
    thread : Thread
        Original Thread instance.

    Returns
    -------
    Thread
        New Thread instance with masked tweet texts.
    """
    new_tweets = [
        tweet.model_copy(update={"text": mask_entities(tweet.text)})
        for tweet in thread.tweets
    ]
    return thread.model_copy(update={"tweets": new_tweets})


def run(
    input_path: Union[
        str, Path
    ] = "data/processed/apple_support_threads_decontracted.jsonl",
    output_path: Union[
        str, Path
    ] = "data/processed/apple_support_threads_masked.jsonl",
) -> int:
    """Read de-contracted threads JSONL, apply entity masking/URL neutralization, and write to output JSONL.

    Streams line-by-line to minimize peak memory consumption.

    Parameters
    ----------
    input_path : Union[str, Path]
        Path to input de-contracted threads JSONL.
    output_path : Union[str, Path]
        Path to output masked threads JSONL.

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

    logger.info("Masking entities and URLs from %s -> %s", src, dest)
    with open(src, "r", encoding="utf-8") as in_f, open(
        dest, "w", encoding="utf-8"
    ) as out_f:
        for line_num, line in enumerate(in_f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                thread = Thread.model_validate_json(line_str)
                masked = mask_thread(thread)
                out_f.write(masked.model_dump_json() + "\n")
                count += 1
            except Exception as err:
                logger.error(
                    "Failed processing thread at line %d: %s", line_num, err
                )
                raise

    logger.info("Successfully masked entities for %d threads to %s", count, dest)
    return count


def main() -> None:
    """CLI entrypoint for entity masking and URL handling feature."""
    parser = argparse.ArgumentParser(
        description="M1.P1.2.F2: Entity Masking & URL Handling for conversation threads"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/apple_support_threads_decontracted.jsonl",
        help="Input JSONL threads file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/apple_support_threads_masked.jsonl",
        help="Output masked JSONL threads file path",
    )
    args = parser.parse_args()

    try:
        run(input_path=args.input, output_path=args.output)
    except Exception as e:
        logger.error("Entity masking pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
