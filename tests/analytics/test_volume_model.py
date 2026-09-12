"""
Unit tests for Negative Binomial Volume Modeling.

Tests overdispersion detection, Negative Binomial regression fitting, SLA breach
risk prediction, and model artifact persistence.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.analytics.schema import DispersionResult, NBModelResult, VolumeFeatureRow
from src.analytics.volume_model import (
    check_overdispersion,
    fit_negative_binomial,
    load_model_artifacts,
    predict_sla_breach_risk,
    save_model_artifacts,
)


@pytest.fixture
def synthetic_volume_data() -> pd.DataFrame:
    """Generate reproducible synthetic dataset for Negative Binomial regression testing."""
    np.random.seed(42)
    n = 600

    sentiments = np.random.choice(["neutral", "positive", "negative"], size=n, p=[0.5, 0.25, 0.25])
    text_lengths = np.random.randint(5, 50, size=n)
    times_of_day = np.random.choice(
        ["business_hours", "evening", "overnight"], size=n, p=[0.6, 0.25, 0.15]
    )
    days_of_week = np.random.choice(["weekday", "weekend"], size=n, p=[0.75, 0.25])

    # Generate follow-up count with overdispersion
    # Negative sentiment and overnight/weekend increase expected follow-up count
    log_mu = (
        0.2
        + (sentiments == "negative") * 0.4
        - (sentiments == "positive") * 0.2
        + (times_of_day == "overnight") * 0.3
        + (days_of_week == "weekend") * 0.2
        + (text_lengths / 100.0)
    )
    mu = np.exp(log_mu)
    alpha = 0.5
    size = 1.0 / alpha
    prob = size / (size + mu)
    follow_up_counts = np.random.negative_binomial(size, prob)

    return pd.DataFrame(
        {
            "thread_id": [f"t_{i}" for i in range(n)],
            "follow_up_count": follow_up_counts,
            "sentiment_category": sentiments,
            "text_length": text_lengths,
            "time_of_day": times_of_day,
            "day_of_week": days_of_week,
        }
    )


class TestCheckOverdispersion:
    """Tests for check_overdispersion diagnostic utility."""

    def test_empty_input(self):
        result = check_overdispersion([])
        assert isinstance(result, DispersionResult)
        assert result.mean == 0.0
        assert result.variance == 0.0
        assert result.dispersion_ratio == 0.0
        assert result.is_overdispersed is False

    def test_equidispersed_data(self, caplog):
        """Data where variance approx equals mean (Poisson-like)."""
        np.random.seed(42)
        poisson_data = np.random.poisson(lam=3.0, size=1000)

        with caplog.at_level(logging.WARNING):
            result = check_overdispersion(poisson_data)

        assert isinstance(result, DispersionResult)
        assert 0.8 < result.dispersion_ratio < 1.3
        assert result.is_overdispersed is False
        assert any("Overdispersion ratio is" in msg for msg in caplog.messages)

    def test_overdispersed_data(self):
        """Data where variance significantly exceeds mean (Negative Binomial)."""
        overdispersed_data = [0] * 50 + [1] * 20 + [5] * 10 + [15] * 5 + [30] * 2
        result = check_overdispersion(overdispersed_data)

        assert isinstance(result, DispersionResult)
        assert result.dispersion_ratio > 1.5
        assert result.is_overdispersed is True

    def test_pandas_series_input(self):
        series = pd.Series([1, 2, 3, 10, 20])
        result = check_overdispersion(series)
        assert isinstance(result, DispersionResult)
        assert result.mean > 0


class TestFitNegativeBinomial:
    """Tests for fit_negative_binomial regression fitting."""

    def test_missing_columns_raises_value_error(self):
        df_incomplete = pd.DataFrame(
            {
                "follow_up_count": [1, 2],
                "sentiment_category": ["positive", "neutral"],
            }
        )
        with pytest.raises(ValueError, match="DataFrame is missing required columns"):
            fit_negative_binomial(df_incomplete)

    def test_small_sample_warning(self, caplog):
        df_small = pd.DataFrame(
            {
                "follow_up_count": [1] * 20,
                "sentiment_category": ["neutral"] * 20,
                "text_length": [10] * 20,
                "time_of_day": ["business_hours"] * 20,
                "day_of_week": ["weekday"] * 20,
            }
        )
        with caplog.at_level(logging.WARNING):
            try:
                fit_negative_binomial(df_small)
            except Exception:
                pass  # Regression fit might struggle on constant data, but warning should be logged

        assert any("Sample size (20) is below recommended minimum" in msg for msg in caplog.messages)

    def test_successful_fit(self, synthetic_volume_data):
        fitted_model, result_schema = fit_negative_binomial(synthetic_volume_data)

        assert fitted_model is not None
        assert isinstance(result_schema, NBModelResult)
        assert result_schema.converged is True
        assert result_schema.sample_size == len(synthetic_volume_data)
        assert result_schema.alpha > 0
        assert result_schema.aic > 0
        assert result_schema.bic > 0

        # Check expected coefficient names from formula
        coefs = result_schema.coefficients
        assert "Intercept" in coefs
        assert "text_length" in coefs
        assert any("sentiment_category" in k for k in coefs)
        assert any("time_of_day" in k for k in coefs)
        assert any("day_of_week" in k for k in coefs)

        # Check pvalues match coefficients
        for k in coefs:
            assert k in result_schema.pvalues
            assert 0.0 <= result_schema.pvalues[k] <= 1.0


class TestPredictSLABreachRisk:
    """Tests for predict_sla_breach_risk probability calculation."""

    @pytest.fixture
    def fitted_model_bundle(self, synthetic_volume_data):
        model, schema = fit_negative_binomial(synthetic_volume_data)
        return model, schema

    def test_predict_with_volume_feature_row(self, fitted_model_bundle):
        model, _ = fitted_model_bundle
        row = VolumeFeatureRow(
            thread_id="test_row_1",
            follow_up_count=0,
            sentiment_category="negative",
            text_length=45,
            time_of_day="overnight",
            day_of_week="weekend",
        )
        risk = predict_sla_breach_risk(model, row, threshold=2)
        assert isinstance(risk, float)
        assert 0.0 <= risk <= 1.0

    def test_predict_with_dict(self, fitted_model_bundle):
        model, _ = fitted_model_bundle
        row_dict = {
            "sentiment_category": "positive",
            "text_length": 10,
            "time_of_day": "business_hours",
            "day_of_week": "weekday",
        }
        risk = predict_sla_breach_risk(model, row_dict, threshold=2)
        assert 0.0 <= risk <= 1.0

    def test_predict_with_dataframe(self, fitted_model_bundle):
        model, _ = fitted_model_bundle
        df = pd.DataFrame(
            [
                {
                    "sentiment_category": "neutral",
                    "text_length": 15,
                    "time_of_day": "business_hours",
                    "day_of_week": "weekday",
                }
            ]
        )
        risk = predict_sla_breach_risk(model, df, threshold=2)
        assert 0.0 <= risk <= 1.0

    def test_predict_with_series(self, fitted_model_bundle, synthetic_volume_data):
        model, _ = fitted_model_bundle
        series = synthetic_volume_data.iloc[0]
        risk = predict_sla_breach_risk(model, series, threshold=2)
        assert 0.0 <= risk <= 1.0

    def test_invalid_input_type_raises(self, fitted_model_bundle):
        model, _ = fitted_model_bundle
        with pytest.raises(TypeError, match="Expected VolumeFeatureRow, dict, Series, or DataFrame"):
            predict_sla_breach_risk(model, ["invalid", "list"])

    def test_threshold_monotonicity(self, fitted_model_bundle):
        """P(Y > k1) >= P(Y > k2) when k1 < k2."""
        model, _ = fitted_model_bundle
        row = {
            "sentiment_category": "negative",
            "text_length": 40,
            "time_of_day": "overnight",
            "day_of_week": "weekend",
        }
        risk_1 = predict_sla_breach_risk(model, row, threshold=1)
        risk_2 = predict_sla_breach_risk(model, row, threshold=2)
        risk_5 = predict_sla_breach_risk(model, row, threshold=5)

        assert risk_1 >= risk_2 >= risk_5
        assert risk_5 >= 0.0


class TestModelArtifactsPersistence:
    """Tests for save_model_artifacts and load_model_artifacts."""

    def test_save_and_load_roundtrip(self, synthetic_volume_data, tmp_path):
        model, result_schema = fit_negative_binomial(synthetic_volume_data)

        json_path, pkl_path = save_model_artifacts(model, result_schema, output_dir=tmp_path)
        assert json_path.exists()
        assert pkl_path.exists()

        loaded_model, loaded_schema = load_model_artifacts(model_dir=tmp_path)
        assert loaded_schema == result_schema

        # Verify predictions match between original and loaded model
        test_row = {
            "sentiment_category": "negative",
            "text_length": 30,
            "time_of_day": "evening",
            "day_of_week": "weekday",
        }
        pred_orig = predict_sla_breach_risk(model, test_row, threshold=2)
        pred_loaded = predict_sla_breach_risk(loaded_model, test_row, threshold=2)

        assert abs(pred_orig - pred_loaded) < 1e-6

    def test_load_nonexistent_directory_raises(self, tmp_path):
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Model JSON artifact not found"):
            load_model_artifacts(model_dir=empty_dir)
