"""Toxic language validator for Guardrails AI schema validation.

Enforces professional-tone requirements regardless of customer provocation.
Uses a lightweight toxicity model (e.g. Detoxify) when installed in the environment,
with a graceful, weighted lexicon fallback that logs a clear one-time warning.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import detoxify
    HAS_DETOXIFY = True
except ImportError:
    HAS_DETOXIFY = False

# Global flag to ensure one-time warning for fallback mode
_FALLBACK_WARNED = False
_WARNED_FALLBACK = False

# Global cache for detoxify model instance
_DETOXIFY_MODEL = None
_DETOXIFY_INITIALIZED = False


# Weighted toxicity lexicon for fallback scoring
# Categories:
# 1.0: Severe slurs, threats of harm, explicit profanity abuse
# 0.8: Direct insults, severe demeaning language
# 0.6: Moderately rude / aggressive language
# 0.4: Mildly dismissive / unprofessional language
TOXIC_PATTERNS = [
    # Severe threats / violence
    (r"\b(?:kill\s+yourself|go\s+die|hope\s+you\s+die)\b", 1.0),
    (r"\b(?:beat\s+you\s+up|i\s+will\s+hurt\s+you|threaten)\b", 0.9),
    # Profanity / vulgarities directed at user
    (r"\b(?:fuck\s+you|fucking\s+(?:idiot|moron|fool|loser|stupid))\b", 1.0),
    (r"\b(?:fuck\s*off|fuck\s+you|bitch|bastard|asshole|dipshit|dickhead)\b", 0.9),
    (r"\b(?:bullshit|crap|dammit|shit)\b", 0.5),
    # Direct demeaning insults
    (r"\b(?:you\s+are\s+(?:an?\s+)?(?:idiot|moron|fool|imbecile|loser|incompetent|clown|trash))\b", 0.85),
    (r"\b(?:shut\s+up|piss\s+off)\b", 0.8),
    (r"\b(?:stupid|pathetic|moron|idiot|incompetent|dumbass)\b", 0.75),
    (r"\b(?:useless|lazy|brainless|worthless)\b", 0.7),
    # Passive-aggressive / condescending
    (r"\b(?:are\s+you\s+(?:blind|deaf|dumb))\b", 0.75),
    (r"\b(?:waste\s+of\s+time|waste\s+of\s+space)\b", 0.6),
]


def _init_detoxify():
    """Attempt to initialize the Detoxify model once."""
    global _DETOXIFY_MODEL, _DETOXIFY_INITIALIZED, _FALLBACK_WARNED, _WARNED_FALLBACK
    if _DETOXIFY_INITIALIZED:
        if (not HAS_DETOXIFY or _DETOXIFY_MODEL is None) and not _FALLBACK_WARNED:
            logger.warning(
                "Detoxify library not available. "
                "Falling back to lexicon-based toxicity scorer. Note: This is a reduced-accuracy fallback, "
                "not a silent equivalent."
            )
            _FALLBACK_WARNED = True
            _WARNED_FALLBACK = True
        return _DETOXIFY_MODEL

    if not HAS_DETOXIFY:
        _DETOXIFY_MODEL = None
        if not _FALLBACK_WARNED:
            logger.warning(
                "Detoxify library not available. "
                "Falling back to lexicon-based toxicity scorer. Note: This is a reduced-accuracy fallback, "
                "not a silent equivalent."
            )
            _FALLBACK_WARNED = True
            _WARNED_FALLBACK = True
        _DETOXIFY_INITIALIZED = True
        return None

    try:
        from detoxify import Detoxify
        _DETOXIFY_MODEL = Detoxify("original")
        logger.info("Detoxify model successfully loaded for toxicity validation.")
    except Exception as e:
        _DETOXIFY_MODEL = None
        if not _FALLBACK_WARNED:
            logger.warning(
                f"Detoxify library not available ({e}). "
                "Falling back to lexicon-based toxicity scorer. Note: This is a reduced-accuracy fallback, "
                "not a silent equivalent."
            )
            _FALLBACK_WARNED = True
            _WARNED_FALLBACK = True

    _DETOXIFY_INITIALIZED = True
    return _DETOXIFY_MODEL



def _score_lexicon_fallback(text: str) -> float:
    """Calculate toxicity score using the weighted lexicon fallback.

    Guaranteed to return a float in [0.0, 1.0].
    """
    if not text or not text.strip():
        return 0.0

    lower_text = text.lower()
    max_score = 0.0
    accumulated_score = 0.0

    for pattern, weight in TOXIC_PATTERNS:
        matches = re.findall(pattern, lower_text)
        if matches:
            count = len(matches)
            # Take the highest single pattern weight
            if weight > max_score:
                max_score = weight
            # Add diminishing weight for additional matches
            accumulated_score += weight * count * 0.1

    final_score = min(1.0, max_score + accumulated_score)
    return round(final_score, 4)


def score_toxicity(text: str, force_fallback: bool = False) -> float:
    """Score the toxicity of input text on a scale from 0.0 to 1.0.

    Args:
        text: Input string to evaluate.
        force_fallback: If True, forces use of the lexicon fallback (useful for testing).

    Returns:
        float: Toxicity score between 0.0 (benign) and 1.0 (highly toxic).
    """
    if not text or not isinstance(text, str) or not text.strip():
        return 0.0

    if not force_fallback:
        model = _init_detoxify()
        if model is not None:
            try:
                results = model.predict(text)
                # Primary toxicity score
                score = float(results.get("toxicity", 0.0))
                return max(0.0, min(1.0, score))
            except Exception as exc:
                logger.warning(f"Detoxify prediction error: {exc}. Using lexicon fallback.")

    global _WARNED_FALLBACK, _FALLBACK_WARNED
    if not _FALLBACK_WARNED:
        logger.warning(
            "Detoxify library not available. "
            "Falling back to lexicon-based toxicity scorer. Note: This is a reduced-accuracy fallback, "
            "not a silent equivalent."
        )
        _WARNED_FALLBACK = True
        _FALLBACK_WARNED = True

    return _score_lexicon_fallback(text)



def check_toxicity(text: str, threshold: float = 0.7, force_fallback: bool = False) -> bool:
    """Check if the toxicity of the text exceeds the given threshold.

    Args:
        text: Input string to evaluate.
        threshold: Rejection threshold (default 0.7).
        force_fallback: If True, forces use of the lexicon fallback.

    Returns:
        bool: True if toxicity score >= threshold (toxic), False otherwise (acceptable).
    """
    score = score_toxicity(text, force_fallback=force_fallback)
    return score >= threshold
