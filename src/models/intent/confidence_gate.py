"""Classifier confidence threshold gating and calibration for intent routing.

Phase: M2.P2.3.F1
Provides threshold filtering to enforce classifier confidence before routing to
retrieval/RAG, and empirical calibration using validation recall targets.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

logger = logging.getLogger(__name__)


def apply_confidence_gate(
    predictions: Sequence[tuple[str, float]],
    threshold: float = 0.65,
) -> list[bool]:
    """Apply confidence threshold mask to a batch of predictions.

    Args:
        predictions: Sequence of (predicted_label, confidence_score) tuples.
        threshold: Minimum confidence score to pass the gate.

    Returns:
        List of boolean masks where True means passed gate (confidence >= threshold,
        proceed to RAG) and False means failed gate (route to human escalation).
    """
    mask: list[bool] = []
    for label, conf in predictions:
        # Boundary behavior: confidence exactly equal to threshold (e.g. 0.65 == 0.65)
        # is treated as PASSED (>= threshold means pass).
        passed = float(conf) >= float(threshold)
        mask.append(passed)
    return mask


def calibrate_threshold(
    val_predictions: Sequence[tuple[str, float]],
    val_labels: Sequence[str],
    target_recall: float = 0.95,
    all_classes: Sequence[str] | None = None,
) -> float:
    """Find lowest confidence threshold at which classifier achieves target_recall on val set.

    For each candidate threshold, calculates macro-average recall over ground-truth classes
    among samples that pass the threshold (predicted confidence >= threshold and predicted == actual).
    Missing classes in the validation set or classes with zero positive support are handled
    gracefully to avoid division-by-zero, with warning logs.

    Args:
        val_predictions: Sequence of (predicted_label, confidence_score) tuples.
        val_labels: Ground truth labels corresponding to val_predictions.
        target_recall: Desired macro-recall target (default 0.95).
        all_classes: Optional list of expected classes to check for representation.

    Returns:
        The lowest threshold (float) achieving >= target_recall. If target cannot be
        met, returns threshold yielding the maximum macro-recall.
    """
    if len(val_predictions) != len(val_labels):
        raise ValueError(
            f"Length mismatch: {len(val_predictions)} predictions vs {len(val_labels)} labels."
        )

    if not val_predictions:
        logger.warning("Empty validation predictions provided to calibrate_threshold. Defaulting to 0.0.")
        return 0.0

    # Count ground-truth support per class
    gt_support: dict[str, int] = defaultdict(int)
    for lbl in val_labels:
        gt_support[lbl] += 1

    # Check for missing classes if all_classes provided
    if all_classes is not None:
        missing_classes = [c for c in all_classes if gt_support[c] == 0]
        if missing_classes:
            logger.warning(
                "Classes missing from validation set ground-truth (support=0): %s. "
                "These classes will be skipped in macro-recall calculation to avoid division-by-zero.",
                missing_classes,
            )

    valid_classes = [cls_name for cls_name, count in gt_support.items() if count > 0]
    if not valid_classes:
        logger.warning("No classes with positive support found. Defaulting threshold to 0.0.")
        return 0.0

    # Collect candidate thresholds from unique confidence scores, including 0.0
    candidate_thresholds = sorted({0.0} | {float(conf) for _, conf in val_predictions})

    candidates_meeting_target: list[float] = []
    best_threshold = candidate_thresholds[0]
    best_macro_recall = -1.0

    for thresh in candidate_thresholds:
        # Count true positives per class among samples passing confidence threshold
        tp_counts: dict[str, int] = defaultdict(int)
        for (pred_label, conf), true_label in zip(val_predictions, val_labels):
            if float(conf) >= thresh and pred_label == true_label:
                tp_counts[true_label] += 1

        # Per-class recall: TP(c) / Support(c)
        recalls = [tp_counts[c] / gt_support[c] for c in valid_classes]
        macro_recall = sum(recalls) / len(recalls)

        if macro_recall >= target_recall:
            candidates_meeting_target.append(thresh)

        if macro_recall > best_macro_recall:
            best_macro_recall = macro_recall
            best_threshold = thresh

    if candidates_meeting_target:
        # Return the LOWEST confidence threshold at which target recall is achieved
        chosen_threshold = min(candidates_meeting_target)
        logger.info(
            "Calibrated lowest threshold %.4f achieves macro-recall >= %.4f (target %.4f).",
            chosen_threshold,
            target_recall,
            target_recall,
        )
        return chosen_threshold

    logger.warning(
        "Target recall %.4f could not be reached. Returning threshold %.4f with max macro-recall %.4f.",
        target_recall,
        best_threshold,
        best_macro_recall,
    )
    return best_threshold
