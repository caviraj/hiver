"""Pydantic schema definitions for volume modeling features and results."""

from typing import Dict, Literal
from pydantic import BaseModel, ConfigDict, Field


class VolumeFeatureRow(BaseModel):
    """Pydantic model representing an extracted feature row for a thread.

    Attributes
    ----------
    thread_id : str
        The unique ID of the root tweet identifying the thread.
    follow_up_count : int
        Response variable: number of customer follow-up turns after the first brand reply.
    sentiment_category : Literal["positive", "neutral", "negative"]
        Sentiment category of the root tweet.
    text_length : int
        Word count (split on whitespace) of the root tweet text.
    time_of_day : Literal["business_hours", "evening", "overnight"]
        Time of day bucket when the root tweet was created.
    day_of_week : Literal["weekday", "weekend"]
        Day of week bucket when the root tweet was created.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    thread_id: str = Field(description="Root tweet_id of the thread")
    follow_up_count: int = Field(
        ge=0,
        description="Response variable: count of consumer follow-up turns",
    )
    sentiment_category: Literal["positive", "neutral", "negative"] = Field(
        description="Sentiment category of the root tweet (positive, neutral, negative)"
    )
    text_length: int = Field(
        ge=0,
        description="Word count of the root tweet text",
    )
    time_of_day: Literal["business_hours", "evening", "overnight"] = Field(
        description="Time of day bucket (business_hours [9-17), evening [17-23), overnight [23-9))"
    )
    day_of_week: Literal["weekday", "weekend"] = Field(
        description="Day of week bucket (weekday Mon-Fri, weekend Sat-Sun)"
    )


class DispersionResult(BaseModel):
    """Result of Poisson overdispersion diagnostics.

    Attributes
    ----------
    mean : float
        Empirical sample mean of follow-up counts.
    variance : float
        Empirical sample variance of follow-up counts.
    dispersion_ratio : float
        Variance-to-mean ratio (or Pearson chi2 / residual df from baseline Poisson GLM).
    is_overdispersed : bool
        True if dispersion_ratio exceeds the threshold (1.5).
    recommended_model : Literal["poisson", "negative_binomial"]
        Recommended model family based on overdispersion test.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    mean: float = Field(ge=0.0, description="Sample mean of follow-up counts")
    variance: float = Field(ge=0.0, description="Sample variance of follow-up counts")
    dispersion_ratio: float = Field(
        ge=0.0,
        description="Dispersion ratio (Pearson chi2 / df_resid or variance / mean)",
    )
    is_overdispersed: bool = Field(
        description="True if dispersion_ratio > 1.5, indicating significant overdispersion"
    )
    recommended_model: Literal["poisson", "negative_binomial"] = Field(
        description="Recommended model family"
    )


class NBModelResult(BaseModel):
    """Fitted Negative Binomial regression model summary and diagnostic metrics.

    Attributes
    ----------
    converged : bool
        Whether the optimization algorithm successfully converged.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    alpha : float
        Estimated dispersion parameter (alpha) of the Negative Binomial distribution.
    coefficients : Dict[str, float]
        Estimated regression coefficients for each predictor and intercept.
    pvalues : Dict[str, float]
        p-values associated with each estimated coefficient.
    sample_size : int
        Number of valid thread observations used in the model fit.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    converged: bool = Field(description="Optimization convergence status")
    aic: float = Field(description="Akaike Information Criterion")
    bic: float = Field(description="Bayesian Information Criterion")
    alpha: float = Field(
        ge=0.0,
        description="Estimated Negative Binomial dispersion parameter alpha",
    )
    coefficients: Dict[str, float] = Field(
        description="Dictionary mapping predictor names to estimated coefficients"
    )
    pvalues: Dict[str, float] = Field(
        description="Dictionary mapping predictor names to p-values"
    )
    sample_size: int = Field(
        ge=0,
        description="Number of observations used in model fitting",
    )
