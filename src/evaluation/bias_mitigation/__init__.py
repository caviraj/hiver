"""Judge Bias Mitigation package.

Countermeasures for the three critical LLM judge biases:
1. Position Bias: ab_swap (run_ab_swap_consistency)
2. Verbosity Bias: length_control (bucket_by_token_count, apply_length_controlled_regression)
3. Self-Enhancement Bias: judge_ensemble (MinorityVetoEnsemble)
"""

from src.evaluation.bias_mitigation.ab_swap import (
    SwapConsistencyResult,
    run_ab_swap_consistency,
)
from src.evaluation.bias_mitigation.judge_ensemble import (
    ConfigurationError,
    EnsembleVerdict,
    MinorityVetoEnsemble,
)
from src.evaluation.bias_mitigation.length_control import (
    EvalItem,
    ResidualizedScores,
    apply_length_controlled_regression,
    bucket_by_token_count,
    estimate_or_count_tokens,
    residualize_eval_items,
)

__all__ = [
    "SwapConsistencyResult",
    "run_ab_swap_consistency",
    "EvalItem",
    "ResidualizedScores",
    "apply_length_controlled_regression",
    "bucket_by_token_count",
    "estimate_or_count_tokens",
    "residualize_eval_items",
    "ConfigurationError",
    "EnsembleVerdict",
    "MinorityVetoEnsemble",
]
