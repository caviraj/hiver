"""Intent classification models and training pipelines."""

from src.models.intent.banking77_bootstrap import (
    EarlyStoppingMonitor,
    check_hardware,
    compute_detailed_metrics,
    compute_metrics,
    load_pretrained_encoder,
    save_checkpoint,
    train,
)
from src.models.intent.config import TaxonomyConfig, TrainConfig
from src.models.intent.dataset import (
    count_truncated_queries,
    load_banking77,
    stratified_split,
    tokenize_dataset,
)
from src.models.intent.domain_adapt_dataset import (
    assemble_domain_dataset,
    calculate_cluster_purity,
    filter_by_purity,
    merge_sparse_clusters,
    stratified_three_way_split,
    validate_surviving_classes,
)
from src.models.intent.domain_adaptation import (
    DomainAdaptResult,
    evaluate_zero_shot_baseline,
    fine_tune_domain_model,
    swap_classification_head,
)
from src.models.intent.confidence_gate import (
    apply_confidence_gate,
    calibrate_threshold,
)
from src.models.intent.escalation_router import (
    DEFAULT_CRITICAL_LABELS,
    EscalationDecision,
    append_escalation_log,
    evaluate_severity_bypass,
    route_message,
)
from src.models.intent.severity_rules import (
    DEFAULT_SEVERITY_KEYWORDS,
    SeverityAuditor,
    evaluate_severity_consensus,
    find_matching_keywords,
    flag_severity_candidates,
)
from src.models.intent.taxonomy_induction import (
    ClusterMetadata,
    TaxonomyResult,
    cluster_embeddings,
    embed_messages,
    extract_text,
    induce_taxonomy,
    load_taxonomy,
    save_taxonomy,
)

__all__ = [
    "TrainConfig",
    "TaxonomyConfig",
    "train",
    "save_checkpoint",
    "load_pretrained_encoder",
    "load_banking77",
    "stratified_split",
    "tokenize_dataset",
    "count_truncated_queries",
    "compute_metrics",
    "compute_detailed_metrics",
    "EarlyStoppingMonitor",
    "check_hardware",
    # Severity rules & consensus
    "DEFAULT_SEVERITY_KEYWORDS",
    "flag_severity_candidates",
    "find_matching_keywords",
    "evaluate_severity_consensus",
    "SeverityAuditor",
    # Taxonomy induction
    "ClusterMetadata",
    "TaxonomyResult",
    "embed_messages",
    "extract_text",
    "cluster_embeddings",
    "induce_taxonomy",
    "save_taxonomy",
    "load_taxonomy",
    # Domain adapt dataset
    "calculate_cluster_purity",
    "filter_by_purity",
    "merge_sparse_clusters",
    "validate_surviving_classes",
    "stratified_three_way_split",
    "assemble_domain_dataset",
    # Domain adaptation
    "DomainAdaptResult",
    "swap_classification_head",
    "evaluate_zero_shot_baseline",
    "fine_tune_domain_model",
    # Confidence gate & escalation router
    "apply_confidence_gate",
    "calibrate_threshold",
    "DEFAULT_CRITICAL_LABELS",
    "EscalationDecision",
    "evaluate_severity_bypass",
    "route_message",
    "append_escalation_log",
]


