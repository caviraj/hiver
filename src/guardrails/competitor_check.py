"""Competitor mention validator for Guardrails AI schema validation.

Reuses the centralized competitor list defined in config/guardrails/restricted_topics.yaml.
Distinguishes between recommending/disparaging mentions (blocked) and factual/compatibility mentions
(logged as borderline for review, not blocked).
"""

import re
import logging
from pathlib import Path
from typing import List, Optional, Union
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "guardrails" / "restricted_topics.yaml"

# Factual / compatibility indicators that permit neutral technical statements
NEUTRAL_COMPATIBILITY_PATTERNS = [
    r"\bcompatible\s+with\b",
    r"\bcompatibility\s+with\b",
    r"\bworks\s+with\b",
    r"\bsupport(?:ed|s)?\s+(?:on|for|with)\b",
    r"\bsupports?\b",
    r"\bconnect(?:s|ed|ing)?\s+to\b",
    r"\badapter\s+for\b",
    r"\bswitch(?:ing)?\s+from\b",
    r"\btransfer(?:ring)?\s+from\b",
    r"\bmigrat(?:e|ing|ion)\s+from\b",
    r"\bmigration\b",
    r"\balso\s+works\s+on\b",
    r"\bintegrat(?:e|es|ed|ing|ion)?\s*(?:with|into|to)?\b",
    r"\bsimilar\s+to\b",
    r"\bdata\s+export\b",
    r"\bexport(?:ing)?\s+to\b",
]

# Recommending or disparaging indicators that warrant hard-blocking
RECOMMENDING_DISPARAGING_PATTERNS = [
    r"\brecommend(?:ed|ing|s)?\b",
    r"\bsuggest(?:ed|ing|s)?\b",
    r"\bprefer(?:red|ring|s)?\b",
    r"\bbetter\s+than\b",
    r"\bbetter\b",
    r"\bworse\s+than\b",
    r"\bworse\b",
    r"\bworst\b",
    r"\bsuperior\s+to\b",
    r"\bsuperior\b",
    r"\binferior\s+to\b",
    r"\binferior\b",
    r"\bchoose\b",
    r"\bopt\s+for\b",
    r"\bgo\s+with\b",
    r"\bbuy\b",
    r"\bpurchase\b",
    r"\btry\b",
    r"\bconsider\b",
    r"\bswitch\s+to\b",
    r"\balternative\s+to\b",
    r"\balternative\b",
    r"\bcompetitor\b",
    r"\bterrible\b",
    r"\bhorrible\b",
    r"\bgarbage\b",
    r"\bjunk\b",
    r"\btrash\b",
    r"\bavoid\b",
    r"\bflawed\b",
    r"\bfaulty\b",
    r"\bsucks?\b",
    r"\bsubpar\b",
    r"\bpoor\s+quality\b",
    r"\boverpriced\b",
    r"\buseless\b",
    r"\bbroken\b",
]


def load_competitors(config_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Load competitor names from the centralized restricted_topics.yaml config file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        logger.warning(f"Competitor config file not found at {path}, returning empty list.")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data.get("competitor_names", [])
    except Exception as exc:
        logger.error(f"Failed to load competitor config from {path}: {exc}")
        return []


def check_competitor_mention(
    text: str,
    competitor_names: Optional[List[str]] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Check if text mentions any competitor in a recommending or disparaging context.

    Args:
        text: The LLM output text to inspect.
        competitor_names: Optional list of competitor names. If None, loaded from config.
        config_path: Optional custom path to restricted_topics.yaml.

    Returns:
        The matched competitor name if found in a recommending/disparaging context,
        or None if not found or if the mention is strictly neutral/factual.
    """
    if not text or not isinstance(text, str):
        return None

    if competitor_names is None:
        competitor_names = load_competitors(config_path)

    if not competitor_names:
        return None

    # Split text into sentences/clauses to assess localized context
    sentences = re.split(r"[.!?\n;]+", text)

    for comp in competitor_names:
        escaped_comp = re.escape(comp.strip())
        # Support case-insensitive matching with optional possessive/plural ('s or s)
        comp_regex = re.compile(rf"\b{escaped_comp}(?:'s|s)?\b", re.IGNORECASE)

        for sentence in sentences:
            if not comp_regex.search(sentence):
                continue

            clean_sentence = sentence.strip()

            # Check if sentence has a neutral/factual compatibility indicator
            is_neutral = any(
                re.search(pat, clean_sentence, re.IGNORECASE)
                for pat in NEUTRAL_COMPATIBILITY_PATTERNS
            )

            # Check if sentence has recommending or disparaging framing
            is_recommending_or_disparaging = any(
                re.search(pat, clean_sentence, re.IGNORECASE)
                for pat in RECOMMENDING_DISPARAGING_PATTERNS
            )

            if is_neutral and not is_recommending_or_disparaging:
                logger.info(
                    "Borderline competitor mention in neutral/factual context: competitor='%s', context='%s'",
                    comp,
                    clean_sentence,
                )
                # Factual compatibility mention is not blocked
                continue

            if is_recommending_or_disparaging:
                logger.warning(
                    "Competitor mention in recommending/disparaging context: competitor='%s', context='%s'",
                    comp,
                    clean_sentence,
                )
                return comp

            # If competitor is mentioned without explicit neutral indicator or recommending indicator,
            # check the whole text for recommendation/disparagement
            whole_text_recommending = any(
                re.search(pat, text, re.IGNORECASE)
                for pat in RECOMMENDING_DISPARAGING_PATTERNS
            )
            if whole_text_recommending:
                logger.warning(
                    "Competitor mention in recommending/disparaging document context: competitor='%s'",
                    comp,
                )
                return comp

            # If it has neither neutral nor recommending framing (e.g. plain mention),
            # log borderline and do not hard-block unless recommending/disparaging
            logger.info(
                "Competitor mention with neutral/unclassified framing: competitor='%s', context='%s'",
                comp,
                clean_sentence,
            )

    return None
