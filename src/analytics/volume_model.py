"""
Negative Binomial Volume Modeling for Customer Follow-Up Turns.

Provides overdispersion diagnostics, Negative Binomial regression fitting via
statsmodels, SLA breach risk prediction via SciPy survival function, and
model artifact serialization.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.schema import DispersionResult, NBModelResult, VolumeFeatureRow

logger = logging.getLogger(__name__)


def check_overdispersion(
    counts: Union[pd.Series, List[int], np.ndarray]
) -> DispersionResult:
    """
    Check for overdispersion in customer turn count data.

    Computes mean, variance, and the dispersion ratio (variance / mean).
    Flags as overdispersed if dispersion ratio > 1.5. If <= 1.5, logs a warning
    advising that Negative Binomial may reduce to Poisson-like estimates,
    but does not fail.

    Args:
        counts: Sequence or Series of turn counts.

    Returns:
        DispersionResult containing mean, variance, dispersion_ratio, and
        is_overdispersed flag.
    """
    arr = np.asarray(counts, dtype=np.float64)
    if len(arr) == 0:
        return DispersionResult(
            mean=0.0,
            variance=0.0,
            dispersion_ratio=0.0,
            is_overdispersed=False,
            recommended_model="poisson",
        )

    mean_val = float(np.mean(arr))
    var_val = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0

    if mean_val > 0:
        ratio = var_val / mean_val
    else:
        ratio = 0.0

    is_overdispersed = bool(ratio > 1.5)

    if not is_overdispersed:
        logger.warning(
            "Overdispersion ratio is %.3f (<= 1.5). Negative Binomial model "
            "may yield estimates close to standard Poisson.",
            ratio,
        )

    recommended_model = "negative_binomial" if is_overdispersed else "poisson"

    return DispersionResult(
        mean=round(mean_val, 4),
        variance=round(var_val, 4),
        dispersion_ratio=round(ratio, 4),
        is_overdispersed=is_overdispersed,
        recommended_model=recommended_model,
    )


def fit_negative_binomial(
    df: pd.DataFrame,
) -> Tuple[Any, NBModelResult]:
    """
    Fit a Negative Binomial regression model to follow-up turn counts.

    Formula:
        follow_up_count ~ C(sentiment_category, Treatment('neutral'))
                        + text_length
                        + C(time_of_day, Treatment('business_hours'))
                        + C(day_of_week, Treatment('weekday'))

    Logs a warning if the sample size is under 500 rows.

    Args:
        df: DataFrame containing the required feature columns:
            'follow_up_count', 'sentiment_category', 'text_length',
            'time_of_day', 'day_of_week'.

    Returns:
        Tuple of:
            - statsmodels fitted regression result object
            - NBModelResult Pydantic schema with model summary metrics
    """
    import statsmodels.formula.api as smf

    required_cols = {
        "follow_up_count",
        "sentiment_category",
        "text_length",
        "time_of_day",
        "day_of_week",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"DataFrame is missing required columns for NB regression: {missing_cols}"
        )

    sample_size = len(df)
    if sample_size < 500:
        logger.warning(
            "Sample size (%d) is below recommended minimum of 500 for "
            "reliable Negative Binomial parameter estimation.",
            sample_size,
        )

    formula = (
        "follow_up_count ~ C(sentiment_category, Treatment('neutral')) "
        "+ text_length "
        "+ C(time_of_day, Treatment('business_hours')) "
        "+ C(day_of_week, Treatment('weekday'))"
    )

    model_family = smf.negativebinomial(formula=formula, data=df)
    fitted_model = model_family.fit(disp=False, maxiter=200)

    # Extract parameters and metrics
    params_dict = dict(fitted_model.params)
    pvalues_dict = dict(fitted_model.pvalues)

    # In statsmodels discrete NegativeBinomial, 'alpha' is estimated as a parameter
    alpha = float(params_dict.pop("alpha", 0.0))
    pvalues_dict.pop("alpha", None)

    # Convert remaining coefficients and pvalues to native floats
    coefficients = {k: round(float(v), 6) for k, v in params_dict.items()}
    pvalues = {k: round(float(v), 6) for k, v in pvalues_dict.items()}

    converged = bool(getattr(fitted_model, "converged", True))
    aic = round(float(fitted_model.aic), 4)
    bic = round(float(fitted_model.bic), 4)

    result_schema = NBModelResult(
        alpha=round(alpha, 6),
        aic=aic,
        bic=bic,
        coefficients=coefficients,
        pvalues=pvalues,
        sample_size=sample_size,
        converged=converged,
    )

    return fitted_model, result_schema


def predict_sla_breach_risk(
    model: Any,
    row: Union[VolumeFeatureRow, Dict[str, Any], pd.DataFrame],
    threshold: int = 2,
) -> float:
    """
    Predict probability of follow-up count exceeding a threshold: P(Y > threshold).

    Uses the fitted Negative Binomial model to predict expected count mu,
    then evaluates the survival function of the Negative Binomial distribution
    parameterized by (n, p) where:
        n = 1 / alpha
        p = 1 / (1 + alpha * mu) = n / (n + mu)

    If alpha <= 0 or negligible, falls back to Poisson survival function.

    Args:
        model: Fitted statsmodels model.
        row: Single feature row as VolumeFeatureRow, dictionary, or DataFrame.
        threshold: Threshold of follow-up turns (default: 2).

    Returns:
        Risk probability strictly bounded in [0.0, 1.0].
    """
    if isinstance(row, VolumeFeatureRow):
        input_df = pd.DataFrame([row.model_dump()])
    elif isinstance(row, dict):
        input_df = pd.DataFrame([row])
    elif isinstance(row, pd.Series):
        input_df = pd.DataFrame([row.to_dict()])
    elif isinstance(row, pd.DataFrame):
        input_df = row
    else:
        raise TypeError(
            f"Expected VolumeFeatureRow, dict, Series, or DataFrame, got {type(row)}"
        )

    # Predict expected mean mu
    pred_mu = model.predict(input_df)
    mu = float(pred_mu.iloc[0] if hasattr(pred_mu, "iloc") else pred_mu[0])

    if mu <= 0:
        return 0.0

    # Extract dispersion parameter alpha
    alpha = 0.0
    if hasattr(model, "params"):
        if "alpha" in model.params:
            alpha = float(model.params["alpha"])
        elif hasattr(model, "model") and hasattr(model.model, "alpha"):
            alpha = float(model.model.alpha)

    # Calculate P(Y > threshold) = Survival Function sf(threshold)
    if alpha > 1e-6:
        n = 1.0 / alpha
        p = n / (n + mu)
        risk = stats.nbinom.sf(threshold, n, p)
    else:
        risk = stats.poisson.sf(threshold, mu)

    # Bound in [0.0, 1.0]
    return float(np.clip(risk, 0.0, 1.0))


def save_model_artifacts(
    model: Any,
    result_schema: NBModelResult,
    output_dir: Union[Path, str] = "data/models",
) -> Tuple[Path, Path]:
    """
    Persist model summary JSON and pickled statsmodels object.

    Args:
        model: Fitted statsmodels model.
        result_schema: NBModelResult schema instance.
        output_dir: Destination directory.

    Returns:
        Tuple of (json_path, pkl_path).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "volume_nb_model.json"
    pkl_path = out_path / "volume_nb_model.pkl"

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(result_schema.model_dump_json(indent=2))

    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    logger.info("Saved model artifacts to %s and %s", json_path, pkl_path)
    return json_path, pkl_path


def load_model_artifacts(
    model_dir: Union[Path, str] = "data/models"
) -> Tuple[Any, NBModelResult]:
    """
    Load pickled statsmodels model and summary JSON from directory.

    Args:
        model_dir: Directory containing 'volume_nb_model.json' and 'volume_nb_model.pkl'.

    Returns:
        Tuple of (fitted_model, result_schema).
    """
    dir_path = Path(model_dir)
    json_path = dir_path / "volume_nb_model.json"
    pkl_path = dir_path / "volume_nb_model.pkl"

    if not json_path.exists():
        raise FileNotFoundError(f"Model JSON artifact not found: {json_path}")
    if not pkl_path.exists():
        raise FileNotFoundError(f"Model pickle artifact not found: {pkl_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        result_schema = NBModelResult.model_validate_json(f.read())

    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    logger.info("Loaded model artifacts from %s", dir_path)
    return model, result_schema
