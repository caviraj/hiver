"""
Dashboard Data Aggregator for HITL Value-per-Cost Telemetry
Milestone 7: Business Telemetry & HITL Dynamics - Phase 7.1
"""

from collections import defaultdict
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from src.telemetry.hitl_metrics import compute_hitl_ratio
from src.telemetry.schema import (
    ConfigurationError,
    HITLDashboardSummary,
    HITLReviewEvent,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/hitl_config.yaml")


def load_hitl_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and validate HITL telemetry configuration from YAML.

    Args:
        config_path: Optional file path to configuration YAML. Defaults to 'config/hitl_config.yaml'.

    Returns:
        Dict[str, Any]: Parsed and validated configuration mapping.

    Raises:
        ConfigurationError: If configuration file is missing required fields or has invalid values.
        FileNotFoundError: If the specified config file does not exist.
    """
    target_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"HITL configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(f"Failed to parse YAML from {target_path}: {e}") from e

    required_keys = ["minute_cost_rate", "sla_penalty_per_minute", "min_sample_size", "severity_weights"]
    for key in required_keys:
        if key not in config:
            raise ConfigurationError(f"Missing required configuration key: '{key}' in {target_path}")

    if not isinstance(config["minute_cost_rate"], (int, float)) or config["minute_cost_rate"] <= 0:
        raise ConfigurationError(
            f"minute_cost_rate must be a positive number (> 0), got: {config.get('minute_cost_rate')}"
        )

    if not isinstance(config["sla_penalty_per_minute"], (int, float)) or config["sla_penalty_per_minute"] <= 0:
        raise ConfigurationError(
            f"sla_penalty_per_minute must be a positive number (> 0), got: {config.get('sla_penalty_per_minute')}"
        )

    if not isinstance(config["min_sample_size"], int) or config["min_sample_size"] < 1:
        raise ConfigurationError(
            f"min_sample_size must be a positive integer (>= 1), got: {config.get('min_sample_size')}"
        )

    if not isinstance(config["severity_weights"], dict):
        raise ConfigurationError(
            f"severity_weights must be a dictionary, got: {type(config.get('severity_weights'))}"
        )

    return config


def aggregate_hitl_ratio(
    events: List[HITLReviewEvent],
    minute_cost_rate: float,
    sla_penalty_per_minute: float,
    severity_weights: Dict[str, float],
    window_label: str,
    min_sample_size: int = 10,
    include_tier_breakdown: bool = True,
) -> HITLDashboardSummary:
    """
    Aggregate review events across a given time window into a structured dashboard summary.

    Computes total operational cost, total caught error value, mean value-per-cost ratio,
    sample size confidence indicator, and per-tier breakdown.

    Args:
        events: List of review events falling within the time window.
        minute_cost_rate: Dollar cost per minute of reviewer effort (> 0.0).
        sla_penalty_per_minute: Dollar penalty per minute for SLA breaches (> 0.0).
        severity_weights: Mapping of error severities to dollar values.
        window_label: Descriptive label for the window (e.g., '24h', '7d', '30d').
        min_sample_size: Minimum number of events required for statistical confidence (default: 10).
        include_tier_breakdown: Whether to recursively aggregate metrics per escalation tier.

    Returns:
        HITLDashboardSummary: The aggregated dashboard summary metrics.

    Raises:
        ConfigurationError: If cost rate or penalty rate is non-positive.
    """
    if minute_cost_rate <= 0:
        raise ConfigurationError(f"minute_cost_rate must be > 0, got: {minute_cost_rate}")
    if sla_penalty_per_minute <= 0:
        raise ConfigurationError(f"sla_penalty_per_minute must be > 0, got: {sla_penalty_per_minute}")
    if min_sample_size < 1:
        raise ConfigurationError(f"min_sample_size must be >= 1, got: {min_sample_size}")

    n_events = len(events)
    low_confidence = n_events < min_sample_size

    if n_events == 0:
        return HITLDashboardSummary(
            window=window_label,
            mean_ratio=None,
            total_cost=0.0,
            total_value=0.0,
            n_events=0,
            low_confidence=True,
            breakdown_by_tier={},
        )

    results = [
        compute_hitl_ratio(
            event=ev,
            minute_cost_rate=minute_cost_rate,
            sla_penalty_per_minute=sla_penalty_per_minute,
            severity_weights=severity_weights,
        )
        for ev in events
    ]

    total_cost = round(sum(r.operational_cost for r in results), 4)
    total_value = round(sum(r.error_value for r in results), 4)

    # Calculate mean ratio strictly across non-zero cost events
    valid_ratios = [r.ratio for r in results if r.ratio is not None]
    if valid_ratios:
        mean_ratio = round(sum(valid_ratios) / len(valid_ratios), 4)
    else:
        logger.warning(
            "All %d events in window '%s' had zero operational cost. Mean ratio is undefined (None).",
            n_events,
            window_label,
        )
        mean_ratio = None

    # Tier breakdown
    breakdown_by_tier: Dict[str, HITLDashboardSummary] = {}
    if include_tier_breakdown:
        tier_groups: Dict[str, List[HITLReviewEvent]] = defaultdict(list)
        for ev in events:
            tier_groups[ev.escalation_tier].append(ev)

        for tier_name, tier_events in sorted(tier_groups.items()):
            breakdown_by_tier[tier_name] = aggregate_hitl_ratio(
                events=tier_events,
                minute_cost_rate=minute_cost_rate,
                sla_penalty_per_minute=sla_penalty_per_minute,
                severity_weights=severity_weights,
                window_label=f"{window_label}:{tier_name}",
                min_sample_size=min_sample_size,
                include_tier_breakdown=False,  # Single level breakdown
            )

    return HITLDashboardSummary(
        window=window_label,
        mean_ratio=mean_ratio,
        total_cost=total_cost,
        total_value=total_value,
        n_events=n_events,
        low_confidence=low_confidence,
        breakdown_by_tier=breakdown_by_tier,
    )
