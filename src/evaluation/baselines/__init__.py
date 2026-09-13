"""Evaluation baselines package."""

from src.evaluation.baselines.trivial_baseline import (
    DEFAULT_KEYWORDS,
    DEFAULT_SUPPORT_URL,
    trivial_baseline_response,
)
from src.evaluation.baselines.zeroshot_baseline import (
    BASELINE_ERROR_PREFIX,
    is_baseline_error,
    zero_shot_baseline_response,
)

__all__ = [
    "trivial_baseline_response",
    "DEFAULT_KEYWORDS",
    "DEFAULT_SUPPORT_URL",
    "zero_shot_baseline_response",
    "BASELINE_ERROR_PREFIX",
    "is_baseline_error",
]
