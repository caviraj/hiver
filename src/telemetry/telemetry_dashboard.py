"""
Unified Telemetry Dashboard Orchestrator.
Milestone 7: Business Telemetry & HITL Dynamics - Phase 7.2 (M7.P7.2.F1)

Unifies Containment & Resolution Rate telemetry with HITL Value-per-Cost ratio
into a single consolidated executive dashboard.

Per the PRD mandate, 'Resolution Rate WITHIN Containment' is established as the
primary headline metric, while 'Raw Containment Rate' is demoted to a secondary
contextual metric.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import ValidationError

from src.telemetry.containment_metrics import (
    compute_containment_rate,
    compute_resolution_rate_within_containment,
)
from src.telemetry.dashboard_data import aggregate_hitl_ratio, load_hitl_config
from src.telemetry.schema import (
    ConfigurationError,
    ContainmentEvent,
    HITLDashboardSummary,
    HITLReviewEvent,
    TelemetryConfig,
    TelemetryDashboard,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/telemetry_config.yaml")


def load_telemetry_config(config_path: Optional[str] = None) -> TelemetryConfig:
    """Load and validate Telemetry configuration from YAML.
    
    Args:
        config_path: Optional path to YAML configuration file.
            Defaults to 'config/telemetry_config.yaml'.
            
    Returns:
        TelemetryConfig instance with validated parameters.
        
    Raises:
        FileNotFoundError: If an explicitly provided config_path does not exist.
        ConfigurationError: If the YAML is malformed or invalid against schema.
    """
    target_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    
    if not target_path.exists():
        if config_path:
            raise FileNotFoundError(f"Telemetry configuration file not found at: {target_path}")
        logger.warning("Default telemetry config not found at %s, using defaults.", target_path)
        return TelemetryConfig()

    with open(target_path, "r", encoding="utf-8") as f:
        try:
            raw_config = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(f"Failed to parse YAML from {target_path}: {e}") from e

    try:
        return TelemetryConfig(**raw_config)
    except (ValidationError, TypeError) as e:
        raise ConfigurationError(f"Invalid telemetry configuration in {target_path}: {e}") from e


def build_telemetry_dashboard(
    containment_events: List[ContainmentEvent],
    hitl_events: List[HITLReviewEvent],
    telemetry_config: Optional[TelemetryConfig] = None,
    hitl_config: Optional[Dict[str, Any]] = None,
    window_label: str = "Last 7 Days",
) -> TelemetryDashboard:
    """Build the final unified telemetry dashboard.
    
    Combines:
    1. Primary headline metric: Resolution Rate WITHIN Containment.
    2. Secondary contextual metric: Raw Containment Rate.
    3. Verification counts: verified_count and pending_verification_count.
    4. HITL summary: Value-per-cost ratios and tier breakdowns from P7.1.
    
    Args:
        containment_events: List of customer conversation containment events.
        hitl_events: List of human reviewer events from HITL operations.
        telemetry_config: Optional TelemetryConfig instance. Defaults to loading from config file.
        hitl_config: Optional dictionary with HITL parameters (minute_cost_rate, etc.).
        window_label: Description of the reporting time window.
        
    Returns:
        TelemetryDashboard: Immutable validated dashboard model.
    """
    if telemetry_config is None:
        telemetry_config = load_telemetry_config()

    if hitl_config is None:
        try:
            hitl_config = load_hitl_config()
        except Exception as e:
            logger.warning("Could not load default HITL config (%s); using fallback values.", e)
            hitl_config = {
                "minute_cost_rate": 0.50,
                "sla_penalty_per_minute": 2.00,
                "severity_weights": {
                    "low": 10.0,
                    "medium": 25.0,
                    "high": 50.0,
                    "critical": 100.0,
                },
                "min_sample_size": 10,
            }

    # 1. Compute containment and resolution metrics
    raw_containment = compute_containment_rate(containment_events) if containment_events else None
    res_rate, verified_count, pending_count = compute_resolution_rate_within_containment(
        containment_events,
        csat_threshold=telemetry_config.csat_threshold,
    )

    # 2. Compute HITL value-per-cost summary
    hitl_summary = aggregate_hitl_ratio(
        events=hitl_events,
        minute_cost_rate=hitl_config["minute_cost_rate"],
        sla_penalty_per_minute=hitl_config["sla_penalty_per_minute"],
        severity_weights=hitl_config["severity_weights"],
        window_label=window_label,
        min_sample_size=hitl_config.get("min_sample_size", 10),
        include_tier_breakdown=True,
    )

    # 3. Assemble unified dashboard (primary headline metric positioned first)
    dashboard = TelemetryDashboard(
        generated_at=datetime.now(timezone.utc),
        window_label=window_label,
        resolution_rate_within_containment=res_rate,
        raw_containment_rate=raw_containment,
        verified_count=verified_count,
        pending_verification_count=pending_count,
        hitl_summary=hitl_summary,
    )

    raw_containment_str = f"{raw_containment * 100:.2f}%" if raw_containment is not None else "None"
    hitl_ratio_str = f"{hitl_summary.overall_ratio:.2f}x" if hitl_summary.overall_ratio is not None else "Insufficient Data"
    logger.info(
        "Built telemetry dashboard [%s]: Resolution Within Containment=%s (verified=%d, pending=%d), "
        "Raw Containment=%s, HITL Ratio=%s",
        window_label,
        f"{res_rate * 100:.1f}%" if res_rate is not None else "None (Pending)",
        verified_count,
        pending_count,
        raw_containment_str,
        hitl_ratio_str,
    )

    return dashboard
