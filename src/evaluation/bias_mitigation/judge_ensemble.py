"""Self-enhancement bias mitigation via multi-provider judge ensemble with recusal."""

import logging
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


class ConfigurationError(ValueError):
    """Raised when the judge ensemble configuration is invalid."""

    pass


@dataclass
class EnsembleVerdict:
    """Consolidated verdict from a multi-provider judge ensemble.

    Attributes:
        scores_by_provider: Mapping of provider/model family to the assigned score.
        final_score: Aggregated ensemble score (median of non-recused judges).
        recused_provider: Name of the provider recused due to self-enhancement bias, if any.
        high_disagreement: True if the spread between highest and lowest score exceeds threshold.
        recusal_unverified: True if response_provider was omitted/unknown, preventing recusal check.
    """

    scores_by_provider: Dict[str, float]
    final_score: float
    recused_provider: Optional[str]
    high_disagreement: bool
    recusal_unverified: bool = False


class MinorityVetoEnsemble:
    """Multi-provider judge ensemble mitigating Self-Enhancement Bias.

    Self-Enhancement Bias:
    LLM judges systematically rate outputs from their own model family higher.
    To neutralize this:
    1. The ensemble requires judges from at least 2 distinct model families.
    2. When evaluating a response, if the response's generating provider matches
       a judge's provider, that judge is recused from voting.
    3. Scores are aggregated using the median to provide robustness against single-judge outliers.
    4. If judge score spread exceeds `disagreement_threshold`, a `high_disagreement`
       flag is raised for human audit or downstream gating.
    """

    def __init__(
        self,
        judges: Sequence[Tuple[str, Any]],
        disagreement_threshold: float = 2.0,
    ) -> None:
        """Initialize the ensemble.

        Args:
            judges: Sequence of (provider_name, judge_client_or_callable).
                    provider_name represents model family (e.g. "openai", "anthropic", "google").
            disagreement_threshold: Maximum allowed spread (max - min) before flagging high_disagreement.

        Raises:
            ConfigurationError: If judges span fewer than 2 distinct model providers.
        """
        if not judges:
            raise ConfigurationError(
                "MinorityVetoEnsemble requires at least 2 judges from distinct model providers."
            )

        self.judges: List[Tuple[str, Any]] = [
            (str(provider).strip().lower(), client) for provider, client in judges
        ]
        self.disagreement_threshold = float(disagreement_threshold)

        distinct_providers = {provider for provider, _ in self.judges}
        if len(distinct_providers) < 2:
            raise ConfigurationError(
                f"MinorityVetoEnsemble requires judges from at least 2 distinct model families/providers. "
                f"Found only: {distinct_providers}"
            )

    @property
    def distinct_providers(self) -> set[str]:
        """Set of unique model families/providers in the ensemble."""
        return {provider for provider, _ in self.judges}

    def _invoke_judge(self, judge_client: Any, response_text: str, **kwargs: Any) -> float:
        """Invoke a single judge and extract its numerical score."""
        raw_result: Any
        if callable(judge_client):
            raw_result = judge_client(response_text, **kwargs)
        elif hasattr(judge_client, "evaluate") and callable(judge_client.evaluate):
            raw_result = judge_client.evaluate(response_text, **kwargs)
        elif hasattr(judge_client, "judge") and callable(judge_client.judge):
            raw_result = judge_client.judge(response_text, **kwargs)
        elif hasattr(judge_client, "run_geval") and callable(judge_client.run_geval):
            raw_result = judge_client.run_geval(response=response_text, **kwargs)
        else:
            raise TypeError(
                f"Judge client {type(judge_client)} is neither callable nor implements "
                "evaluate(), judge(), or run_geval()."
            )

        # Extract numerical score from result
        if isinstance(raw_result, (int, float)):
            return float(raw_result)
        elif hasattr(raw_result, "continuous_score"):
            return float(raw_result.continuous_score)
        elif hasattr(raw_result, "score"):
            return float(raw_result.score)
        elif isinstance(raw_result, dict) and "score" in raw_result:
            return float(raw_result["score"])
        elif isinstance(raw_result, dict) and "continuous_score" in raw_result:
            return float(raw_result["continuous_score"])
        else:
            try:
                return float(raw_result)
            except (ValueError, TypeError) as err:
                raise ValueError(
                    f"Could not extract float score from judge result: {raw_result!r}"
                ) from err

    def evaluate(
        self,
        response_text: str,
        response_provider: Optional[str] = None,
        **kwargs: Any,
    ) -> EnsembleVerdict:
        """Evaluate a response using the ensemble with self-enhancement recusal.

        Args:
            response_text: The generated text to evaluate.
            response_provider: Model family/provider of the generator (e.g. "openai", "anthropic").
                               If provided and matches a judge, that judge is recused.
                               If None or empty, recusal cannot be verified and recusal_unverified=True.
            **kwargs: Extra arguments passed to judge invocations.

        Returns:
            EnsembleVerdict containing scores by provider, final score, recusal info, and disagreement flag.
        """
        norm_resp_provider = (
            str(response_provider).strip().lower() if response_provider else None
        )

        recused_provider: Optional[str] = None
        recusal_unverified: bool = False

        if not norm_resp_provider:
            logger.warning(
                "Response provider unknown or unspecified. Cannot verify self-enhancement bias recusal. "
                "Setting recusal_unverified=True."
            )
            recusal_unverified = True
        else:
            # Check if any judge matches the generator's provider
            matching_judges = [p for p, _ in self.judges if p == norm_resp_provider]
            if matching_judges:
                recused_provider = norm_resp_provider
                logger.info(
                    "Self-enhancement bias mitigation: Recusing judge(s) from provider '%s'.",
                    recused_provider,
                )

        # Collect scores from active (non-recused) judges
        scores_by_provider: Dict[str, float] = {}
        active_judges = [
            (p, client) for p, client in self.judges if p != recused_provider
        ]

        if not active_judges:
            raise RuntimeError(
                f"All judges in the ensemble were recused for provider '{recused_provider}'. "
                "Cannot compute ensemble score."
            )

        for provider, client in active_judges:
            score = self._invoke_judge(client, response_text, **kwargs)
            # If multiple judges belong to same provider, record average
            if provider in scores_by_provider:
                scores_by_provider[provider] = (scores_by_provider[provider] + score) / 2.0
            else:
                scores_by_provider[provider] = score

        score_values = list(scores_by_provider.values())
        final_score = float(statistics.median(score_values))

        # Check spread / disagreement
        spread = max(score_values) - min(score_values)
        high_disagreement = spread > self.disagreement_threshold

        if high_disagreement:
            logger.warning(
                "High judge disagreement detected! Spread: %.2f > threshold %.2f. "
                "Scores: %s",
                spread,
                self.disagreement_threshold,
                scores_by_provider,
            )

        return EnsembleVerdict(
            scores_by_provider=scores_by_provider,
            final_score=final_score,
            recused_provider=recused_provider,
            high_disagreement=high_disagreement,
            recusal_unverified=recusal_unverified,
        )
