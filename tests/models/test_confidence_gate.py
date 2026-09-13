"""Tests for classifier confidence threshold enforcement and empirical calibration.

Phase: M2.P2.3.F1
"""

from __future__ import annotations

import pytest

from src.models.intent.confidence_gate import (
    apply_confidence_gate,
    calibrate_threshold,
)


def test_batch_gate_mask_correctness() -> None:
    """Test standard pass/fail mask for a batch of predictions."""
    predictions = [
        ("Account_Closure", 0.92),
        ("Refund_Request", 0.60),
        ("Payment_Failed", 0.75),
        ("Greeting", 0.40),
    ]
    mask = apply_confidence_gate(predictions, threshold=0.65)
    assert mask == [True, False, True, False]


def test_boundary_equality_passes() -> None:
    """Confidence exactly equal to threshold (0.65 == 0.65) must pass the gate (>=)."""
    predictions = [
        ("Card_Issue", 0.65),
        ("Account_Issue", 0.64999),
        ("Login_Issue", 0.65001),
    ]
    mask = apply_confidence_gate(predictions, threshold=0.65)
    assert mask == [True, False, True]


def test_all_fail_batch_handled_gracefully() -> None:
    """Batch where all predictions fail threshold should return all-False mask without error."""
    predictions = [
        ("Out_Of_Domain", 0.10),
        ("Random_Noise", 0.25),
        ("Unintelligible", 0.30),
    ]
    mask = apply_confidence_gate(predictions, threshold=0.70)
    assert mask == [False, False, False]


def test_empty_predictions_batch() -> None:
    """Empty batch should return empty list."""
    mask = apply_confidence_gate([], threshold=0.65)
    assert mask == []


def test_calibrate_threshold_synthetic_known_labels() -> None:
    """Verify calibrate_threshold finds lowest threshold achieving target_recall."""
    # Class A has 4 samples:
    #   - pred="A", conf=0.90, true="A" (TP at <= 0.90)
    #   - pred="A", conf=0.80, true="A" (TP at <= 0.80)
    #   - pred="A", conf=0.70, true="A" (TP at <= 0.70)
    #   - pred="A", conf=0.50, true="A" (TP at <= 0.50)
    # Class B has 4 samples:
    #   - pred="B", conf=0.85, true="B" (TP at <= 0.85)
    #   - pred="B", conf=0.75, true="B" (TP at <= 0.75)
    #   - pred="B", conf=0.60, true="B" (TP at <= 0.60)
    #   - pred="B", conf=0.40, true="B" (TP at <= 0.40)
    # Total recall per class at threshold 0.0 is 1.0 (4/4 for A, 4/4 for B).
    # At threshold 0.70:
    #   - A has 3 TP (0.90, 0.80, 0.70) -> recall 3/4 = 0.75
    #   - B has 2 TP (0.85, 0.75) -> recall 2/4 = 0.50
    # At threshold 0.0: macro recall = 1.0 >= 0.95 -> lowest threshold should be 0.0
    val_preds = [
        ("A", 0.90),
        ("A", 0.80),
        ("A", 0.70),
        ("A", 0.50),
        ("B", 0.85),
        ("B", 0.75),
        ("B", 0.60),
        ("B", 0.40),
    ]
    val_labels = ["A", "A", "A", "A", "B", "B", "B", "B"]

    threshold_95 = calibrate_threshold(val_preds, val_labels, target_recall=0.95)
    assert threshold_95 == 0.0

    # If target recall is 0.50, candidate thresholds meeting >= 0.50 are 0.0, 0.40, 0.50, 0.60, 0.70
    # The lowest threshold among those achieving >= 0.50 is 0.0
    thresh_50 = calibrate_threshold(val_preds, val_labels, target_recall=0.50)
    assert thresh_50 == 0.0

    # Let's test a case where lower thresholds have lower recall due to misclassification
    # Val set where low confidence predictions are wrong:
    # A true:
    #  pred="A", conf=0.80
    #  pred="B", conf=0.20 (wrong!)
    # B true:
    #  pred="B", conf=0.85
    #  pred="A", conf=0.30 (wrong!)
    preds_noisy = [
        ("A", 0.80),
        ("B", 0.20),
        ("B", 0.85),
        ("A", 0.30),
    ]
    labels_noisy = ["A", "A", "B", "B"]
    # At thresh 0.0: A recall = 1/2 = 0.5, B recall = 1/2 = 0.5 -> macro recall 0.5
    # At thresh 0.80: A recall = 1/2 = 0.5, B recall = 1/2 = 0.5 -> macro recall 0.5
    calibrated = calibrate_threshold(preds_noisy, labels_noisy, target_recall=0.50)
    assert calibrated == 0.0


def test_calibrate_threshold_missing_class_no_div_zero(caplog: pytest.LogCaptureFixture) -> None:
    """If all_classes contains classes absent from val_labels, ensure no ZeroDivisionError and log warning."""
    val_preds = [
        ("A", 0.90),
        ("A", 0.70),
    ]
    val_labels = ["A", "A"]
    all_classes = ["A", "B_Missing", "C_Missing"]

    with caplog.at_level("WARNING"):
        thresh = calibrate_threshold(val_preds, val_labels, target_recall=0.90, all_classes=all_classes)

    assert thresh == 0.0
    assert "Classes missing from validation set ground-truth" in caplog.text
    assert "B_Missing" in caplog.text


def test_calibrate_threshold_length_mismatch() -> None:
    """Mismatched prediction and label lengths should raise ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        calibrate_threshold([("A", 0.9)], ["A", "B"])
