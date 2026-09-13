"""Inter-annotator agreement computation using Krippendorff's Alpha.

Used during golden dataset creation to validate that multiple human annotators
agree sufficiently before their ratings are accepted as ground truth.

Krippendorff's Alpha was deliberately chosen over Fleiss' Kappa because:
1. It natively supports missing data without imputation: Fleiss' Kappa requires
   every subject to be rated by a fixed number of raters, whereas human annotation
   campaigns inevitably have sparse coverage where annotators rate overlapping
   subsets of examples.
2. It natively supports ordinal (and interval/ratio) measurement levels: Fleiss'
   Kappa treats all categories as nominal, failing to penalize a 1-vs-5 disagreement
   more heavily than a 3-vs-4 disagreement.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import krippendorff
import numpy as np

logger = logging.getLogger(__name__)


def compute_krippendorffs_alpha(
    annotations: Dict[str, Dict[str, float]],
    level_of_measurement: str = "ordinal",
) -> float:
    """Compute Krippendorff's Alpha for multi-annotator agreement with sparse missing cells.

    Krippendorff's Alpha natively accommodates missing data (e.g. where only a subset
    of examples receives redundant annotations across different annotators). Fleiss'
    Kappa is unsuitable for this workflow because it demands complete ratings and
    ignores ordinal distance.

    Args:
        annotations: Dictionary mapping example_id -> annotator_id -> score.
            Example:
            {
                "item_1": {"ann_A": 4.0, "ann_B": 4.0},
                "item_2": {"ann_B": 3.0, "ann_C": 2.0},
                "item_3": {"ann_A": 5.0}  # Single annotation item
            }
        level_of_measurement: Scale of measurement. Default is "ordinal" (1-5 ratings).
            Supported by krippendorff: 'nominal', 'ordinal', 'interval', 'ratio'.

    Returns:
        Krippendorff's Alpha agreement coefficient (-1.0 to 1.0).

    Raises:
        ValueError: If fewer than 2 examples have overlapping (>= 2) annotations,
            or if the annotation structure is empty/invalid.
    """
    if not annotations:
        raise ValueError("Annotations dictionary is empty; cannot compute Krippendorff's Alpha.")

    # Count how many examples have 2 or more annotators (redundant overlap)
    overlapping_examples = sum(1 for raters in annotations.values() if len(raters) >= 2)
    if overlapping_examples < 2:
        raise ValueError(
            f"Krippendorff's Alpha requires at least 2 examples with overlapping ratings (>= 2 annotators), "
            f"but found only {overlapping_examples}. Alpha is mathematically undefined without redundant ratings."
        )

    # Collect all unique annotators and examples
    example_ids = list(annotations.keys())
    annotator_ids = sorted(
        {ann_id for raters in annotations.values() for ann_id in raters.keys()}
    )

    if len(annotator_ids) < 2:
        raise ValueError(
            f"At least 2 distinct annotators are required to compute agreement, got {len(annotator_ids)}."
        )

    # Construct the reliability data matrix: shape (M annotators, N units/examples)
    # Missing annotations are populated with np.nan
    matrix = np.full((len(annotator_ids), len(example_ids)), np.nan, dtype=float)
    annotator_idx = {ann_id: i for i, ann_id in enumerate(annotator_ids)}

    for col_idx, ex_id in enumerate(example_ids):
        for ann_id, score in annotations[ex_id].items():
            row_idx = annotator_idx[ann_id]
            matrix[row_idx, col_idx] = float(score)

    try:
        alpha = float(
            krippendorff.alpha(
                reliability_data=matrix,
                level_of_measurement=level_of_measurement,
            )
        )
    except Exception as exc:
        logger.error("Underlying krippendorff computation failed: %s", exc)
        raise ValueError(f"Failed to compute Krippendorff's Alpha: {exc}") from exc

    return alpha
