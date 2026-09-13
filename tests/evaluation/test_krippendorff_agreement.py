"""Tests for Krippendorff's Alpha inter-annotator agreement."""

import pytest

from src.evaluation.krippendorff_agreement import compute_krippendorffs_alpha


class TestKrippendorffAgreement:
    """Test suite for compute_krippendorffs_alpha."""

    def test_perfect_agreement(self):
        """When multiple annotators agree identically, Alpha should be 1.0."""
        annotations = {
            "ex_1": {"ann_1": 5.0, "ann_2": 5.0, "ann_3": 5.0},
            "ex_2": {"ann_1": 1.0, "ann_2": 1.0, "ann_3": 1.0},
            "ex_3": {"ann_1": 3.0, "ann_2": 3.0, "ann_3": 3.0},
            "ex_4": {"ann_1": 4.0, "ann_2": 4.0, "ann_3": 4.0},
        }
        alpha = compute_krippendorffs_alpha(annotations, level_of_measurement="ordinal")
        assert pytest.approx(alpha, abs=1e-4) == 1.0

    def test_missing_cells_and_single_annotation_items(self):
        """Krippendorff's Alpha must handle missing ratings and single-annotated items without failure.

        This test explicitly validates the core architectural reason Krippendorff was chosen
        over Fleiss' Kappa (which cannot process missing cells without dropping data or imputing).
        """
        annotations = {
            # ex_1 rated by ann_1 and ann_2
            "ex_1": {"ann_1": 4.0, "ann_2": 4.0},
            # ex_2 rated by ann_2 and ann_3
            "ex_2": {"ann_2": 2.0, "ann_3": 2.0},
            # ex_3 has only a single annotation (ann_1) - missing for ann_2 & ann_3
            "ex_3": {"ann_1": 5.0},
            # ex_4 rated by ann_1 and ann_3
            "ex_4": {"ann_1": 1.0, "ann_3": 1.0},
            # ex_5 rated by all three
            "ex_5": {"ann_1": 3.0, "ann_2": 3.0, "ann_3": 4.0},
        }
        alpha = compute_krippendorffs_alpha(annotations, level_of_measurement="ordinal")
        # High agreement across items; alpha should be positive and high (> 0.8)
        assert 0.80 <= alpha <= 1.0

    def test_insufficient_overlapping_annotations_raises_value_error(self):
        """Fewer than 2 overlapping examples must raise an informative ValueError."""
        annotations = {
            # Only 1 item has overlapping ratings
            "ex_1": {"ann_1": 4.0, "ann_2": 3.0},
            "ex_2": {"ann_1": 2.0},
            "ex_3": {"ann_2": 5.0},
            "ex_4": {"ann_3": 1.0},
        }
        with pytest.raises(ValueError, match="at least 2 examples with overlapping ratings"):
            compute_krippendorffs_alpha(annotations)

    def test_zero_overlap_raises_value_error(self):
        """Completely disjoint annotations raise ValueError."""
        annotations = {
            "ex_1": {"ann_1": 4.0},
            "ex_2": {"ann_2": 3.0},
            "ex_3": {"ann_3": 2.0},
        }
        with pytest.raises(ValueError, match="at least 2 examples with overlapping ratings"):
            compute_krippendorffs_alpha(annotations)

    def test_empty_annotations_raises_value_error(self):
        with pytest.raises(ValueError, match="Annotations dictionary is empty"):
            compute_krippendorffs_alpha({})

    def test_single_annotator_raises_value_error(self):
        """If only one annotator exists across all items, inter-annotator agreement is impossible."""
        annotations = {
            "ex_1": {"ann_1": 4.0},
            "ex_2": {"ann_1": 3.0},
        }
        with pytest.raises(ValueError):
            compute_krippendorffs_alpha(annotations)
