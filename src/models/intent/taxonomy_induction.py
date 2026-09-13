"""Unsupervised intent taxonomy induction via frozen embeddings and HDBSCAN clustering.

Extracts [CLS] representations from domain customer queries, clusters them using
HDBSCAN (with outliers assigned to 'Unclassified'), identifies centroid-nearest
exemplars, and invokes an LLM labeling procedure with collision resolution and
severity dual-consensus validation.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.cluster import HDBSCAN

from src.models.intent.config import TaxonomyConfig
from src.models.intent.severity_rules import (
    DEFAULT_SEVERITY_KEYWORDS,
    SeverityAuditor,
    evaluate_severity_consensus,
    find_matching_keywords,
    flag_severity_candidates,
)

logger = logging.getLogger(__name__)

PASCAL_UNDERSCORE_REGEX = re.compile(r"^[A-Z][a-zA-Z0-9]*(_[A-Z][a-zA-Z0-9]*)*$")


class ClusterMetadata(BaseModel):
    """Metadata schema for an induced taxonomy cluster."""

    model_config = ConfigDict(frozen=True)

    cluster_id: int = Field(..., description="HDBSCAN cluster identifier (-1 for Unclassified)")
    label: str = Field(..., description="Assigned taxonomic intent label")
    size: int = Field(..., description="Number of instances assigned to this cluster")
    purity_score: float = Field(..., description="Mean silhouette, probability, or proximity score [0, 1]")
    representative_examples: list[str] = Field(
        default_factory=list,
        description="5-10 centroid-nearest customer utterances",
    )
    severity_flag: bool = Field(
        default=False,
        description="True if flagged high-severity via pattern + LLM consensus",
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class TaxonomyResult(BaseModel):
    """Complete output of taxonomy induction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of taxonomy generation",
    )
    total_samples: int
    num_clusters: int
    unclassified_count: int
    clusters: list[ClusterMetadata]
    labels_per_sample: list[str]
    cluster_ids_per_sample: list[int]
    probabilities_per_sample: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "total_samples": self.total_samples,
            "num_clusters": self.num_clusters,
            "unclassified_count": self.unclassified_count,
            "clusters": [c.to_dict() for c in self.clusters],
        }


def extract_text(item: Any) -> str:
    """Extract query text from string, dict, or object."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "content", "message", "customer_text", "query", "utterance"):
            if key in item and item[key]:
                return item[key]
    if hasattr(item, "text") and isinstance(item.text, str):
        return item.text
    if hasattr(item, "customer_text") and isinstance(item.customer_text, str):
        return item.customer_text
    return str(item)


_extract_text = extract_text


def embed_messages(
    records: Sequence[Any],
    encoder_path: str | Path | None = None,
    model: Any = None,
    tokenizer: Any = None,
    batch_size: int = 32,
    device: str | None = None,
    max_length: int = 64,
) -> np.ndarray:
    """Extract frozen [CLS] token embeddings for a collection of customer queries.

    Args:
        records: Sequence of strings, dicts, or objects containing query text.
        encoder_path: Optional path to load a saved checkpoint.
        model: Optional pre-loaded PyTorch model (DistilBERT / Transformer).
        tokenizer: Optional pre-loaded tokenizer.
        batch_size: Batch size for forward passes.
        device: 'cuda', 'cpu', etc. Defaults to cuda if available.
        max_length: Maximum tokenization sequence length.

    Returns:
        np.ndarray of shape (len(records), hidden_dim).
    """
    texts = [extract_text(r) for r in records]
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        if encoder_path is None:
            raise ValueError("Either model or encoder_path must be provided to embed_messages")
        tokenizer = AutoTokenizer.from_pretrained(encoder_path)
        model = AutoModel.from_pretrained(encoder_path)

    model.eval()
    model.to(device)

    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            if hasattr(model, "distilbert"):
                # Sequence classification wrapper
                outputs = model.distilbert(**encoded)
                cls_embeds = outputs.last_hidden_state[:, 0, :]
            elif hasattr(model, "base_model"):
                outputs = model.base_model(**encoded)
                cls_embeds = outputs.last_hidden_state[:, 0, :]
            else:
                outputs = model(**encoded)
                if hasattr(outputs, "last_hidden_state"):
                    cls_embeds = outputs.last_hidden_state[:, 0, :]
                else:
                    cls_embeds = outputs[0][:, 0, :]

            # L2 normalize
            norm = torch.norm(cls_embeds, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized = (cls_embeds / norm).cpu().numpy()
            all_embeddings.append(normalized)

    return np.vstack(all_embeddings).astype(np.float32)


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 20,
    min_samples: int = 5,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Run HDBSCAN density clustering on document embeddings.

    Args:
        embeddings: Array of shape (N, D).
        min_cluster_size: Minimum number of samples in a cluster.
        min_samples: The number of samples in a neighbourhood for a point to be core.
        metric: Distance metric ('euclidean').
        cluster_selection_method: 'eom' (Excess of Mass) or 'leaf'.

    Returns:
        tuple of (cluster_labels, probabilities, centroids):
            - cluster_labels: (N,) array of cluster ids (-1 for outliers).
            - probabilities: (N,) array of cluster membership strengths [0, 1].
            - centroids: mapping of cluster_id -> (D,) centroid vector.
    """
    n_samples = len(embeddings)
    if n_samples == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float), {}

    # Guard against min_cluster_size > n_samples
    effective_min_size = min(min_cluster_size, max(2, n_samples))
    effective_min_samples = min(min_samples, effective_min_size)

    clusterer = HDBSCAN(
        min_cluster_size=effective_min_size,
        min_samples=effective_min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        copy=True,
    )
    labels = clusterer.fit_predict(embeddings)
    probabilities = getattr(clusterer, "probabilities_", np.ones(n_samples, dtype=float))

    centroids: dict[int, np.ndarray] = {}
    unique_labels = set(labels)
    for cid in unique_labels:
        if cid == -1:
            continue
        mask = labels == cid
        pts = embeddings[mask]
        centroids[cid] = np.mean(pts, axis=0)

    return labels, probabilities, centroids


def get_representative_indices(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_id: int,
    centroid: np.ndarray | None,
    max_examples: int = 10,
) -> list[int]:
    """Find indices of the samples nearest to the cluster centroid.

    Args:
        embeddings: Full array of embeddings.
        cluster_labels: Cluster label array for each sample.
        cluster_id: The cluster ID to query.
        centroid: Centroid vector of the cluster.
        max_examples: Maximum number of exemplar indices to return.

    Returns:
        List of indices into `embeddings` sorted by ascending distance to centroid.
    """
    indices = np.where(cluster_labels == cluster_id)[0]
    if len(indices) == 0:
        return []
    if centroid is None or len(centroid) == 0:
        return list(indices[:max_examples])

    pts = embeddings[indices]
    distances = np.linalg.norm(pts - centroid, axis=1)
    sorted_order = np.argsort(distances)
    sorted_indices = indices[sorted_order]
    return list(sorted_indices[:max_examples])


def sanitize_label(candidate: str, cluster_id: int) -> str:
    """Sanitize and validate an LLM-proposed label into PascalCase_With_Underscores."""
    cleaned = str(candidate).strip()
    if not cleaned:
        return f"Cluster_{cluster_id}"

    cleaned = cleaned.strip("\"'`").strip()
    for prefix in ("Label:", "Category:", "Intent:", "Name:"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()

    cleaned = re.sub(r"^(?:\d+\s*[\.\)\-]\s*|\d+\s*[:./-]\s*)", "", cleaned, count=1)
    cleaned = cleaned.strip()

    if PASCAL_UNDERSCORE_REGEX.match(cleaned):
        return cleaned

    tokens = re.findall(r"[A-Za-z0-9]+", cleaned)
    if tokens:
        converted = "_".join(t.capitalize() for t in tokens)
        if PASCAL_UNDERSCORE_REGEX.match(converted):
            return converted

    if not cleaned or not any(ch.isalnum() for ch in cleaned):
        return f"Cluster_{cluster_id}"
    return f"Cluster_{cluster_id}"


def label_cluster(
    representative_texts: list[str],
    cluster_id: int,
    llm_client: Callable[[str], str] | None = None,
    seed_labels: list[str] | None = None,
    unclassified_label: str = "Unclassified",
    llm_callable: Callable[[str], str] | None = None,
) -> str:
    """Assign an intent name to a cluster using representative texts and LLM guidance."""
    if cluster_id == -1:
        return unclassified_label

    if not representative_texts:
        return f"Cluster_{cluster_id}"

    if llm_client is None:
        llm_client = llm_callable

    seeds = seed_labels or [
        "Device_Performance_Degradation",
        "Software_Update_Failure",
        "Account_Recovery",
        "Outrage_Escalation",
    ]

    if llm_client is None:
        text_concat = " ".join(representative_texts).lower()
        if any(k in text_concat for k in ("update", "ios", "install", "upgrade", "download error")):
            return "Software_Update_Failure"
        if any(k in text_concat for k in ("battery", "slow", "freeze", "crash", "drain", "heat", "lag")):
            return "Device_Performance_Degradation"
        if any(k in text_concat for k in ("password", "apple id", "login", "locked", "reset", "icloud")):
            return "Account_Recovery"
        if any(k in text_concat for k in ("lawyer", "sue", "unacceptable", "furious", "danger", "exploded", "fire")):
            return "Outrage_Escalation"
        return f"Cluster_{cluster_id}"

    examples_str = "\n".join(f"- {t}" for t in representative_texts)
    prompt = (
        "You are an expert taxonomy labeling assistant for Apple customer support interactions.\n"
        "Given the following representative customer messages from a single cluster:\n\n"
        f"{examples_str}\n\n"
        "Assign the most accurate intent label. Choose from these priority seed categories if applicable:\n"
        f"{', '.join(seeds)}\n\n"
        "If none match, propose a new concise, specific category name.\n"
        "REQUIREMENT: Return ONLY the category name formatted in PascalCase_With_Underscores "
        "(e.g., Device_Hardware_Repair, App_Store_Billing). Do not include any explanations or punctuation."
    )

    try:
        response = llm_client(prompt)
        return sanitize_label(response, cluster_id)
    except Exception as e:
        logger.warning("LLM labeling failed for cluster %d: %s. Falling back to Cluster_%d", cluster_id, e, cluster_id)
        return f"Cluster_{cluster_id}"


def resolve_naming_collisions(
    cluster_labels: dict[int, str],
    unclassified_label: str = "Unclassified",
) -> dict[int, str]:
    """Resolve duplicate cluster labels by appending numeric suffixes (_2, _3, ...).

    The first cluster assigned a label retains the pristine label. Subsequent clusters
    with the identical label receive '_2', '_3', etc.
    Unclassified label (-1) is preserved without numeric suffixing.

    Args:
        cluster_labels: Mapping of cluster_id -> raw label.
        unclassified_label: Label name to exclude from collision renaming.

    Returns:
        Mapping of cluster_id -> unique label.
    """
    resolved: dict[int, str] = {}
    label_counts: dict[str, int] = {}

    # Sort cluster IDs so assignment is deterministic (-1 first or last, positive in order)
    sorted_ids = sorted(cluster_labels.keys(), key=lambda cid: (cid == -1, cid))

    for cid in sorted_ids:
        raw = cluster_labels[cid]
        if raw == unclassified_label or cid == -1:
            resolved[cid] = unclassified_label
            continue

        count = label_counts.get(raw, 0) + 1
        label_counts[raw] = count
        if count == 1:
            resolved[cid] = raw
        else:
            resolved[cid] = f"{raw}_{count}"

    return resolved


def induce_taxonomy(
    records: Sequence[Any],
    embeddings: np.ndarray | None = None,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    purity_threshold: float | None = None,
    auditor: SeverityAuditor | None = None,
    config: TaxonomyConfig | None = None,
    encoder_path: str | Path | None = None,
    model: Any = None,
    tokenizer: Any = None,
    llm_callable: Callable[[str], str] | None = None,
    llm_client: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Induce a taxonomy and return the direct cluster metadata dict expected by tests."""
    cfg = config or TaxonomyConfig()
    if min_cluster_size is not None:
        cfg.min_cluster_size = min_cluster_size
    if min_samples is not None:
        cfg.min_samples = min_samples
    if purity_threshold is not None:
        cfg.purity_threshold = purity_threshold

    texts = [extract_text(r) for r in records]
    n_samples = len(texts)
    if n_samples == 0:
        return {}

    if embeddings is None:
        embeddings = embed_messages(
            records=texts,
            encoder_path=encoder_path or cfg.encoder_checkpoint_dir,
            model=model,
            tokenizer=tokenizer,
            batch_size=cfg.batch_size,
        )

    if auditor is None:
        auditor = SeverityAuditor()
    if not hasattr(auditor, "summary"):
        def summary() -> dict[str, int]:
            return {
                "total_evaluated": getattr(auditor, "total_evaluated", 0),
                "total_disagreements": len(getattr(auditor, "disagreements", [])),
            }
        auditor.summary = summary
    if not hasattr(auditor, "total_evaluated"):
        auditor.total_evaluated = 0

    cluster_labels_arr, probabilities_arr, centroids = cluster_embeddings(
        embeddings=embeddings,
        min_cluster_size=cfg.min_cluster_size,
        min_samples=cfg.min_samples,
        metric=cfg.metric,
        cluster_selection_method=cfg.cluster_selection_method,
    )

    unique_clusters = sorted(set(cluster_labels_arr))
    raw_labels: dict[int, str] = {}
    cluster_exemplars: dict[int, list[str]] = {}
    cluster_purities: dict[int, float] = {}

    for cid in unique_clusters:
        if cid == -1:
            raw_labels[cid] = cfg.unclassified_label
            mask = cluster_labels_arr == -1
            cluster_exemplars[cid] = [texts[idx] for idx in np.where(mask)[0][: cfg.max_representative_texts]]
            cluster_purities[cid] = 0.0
            continue

        centroid = centroids.get(cid)
        exemplar_indices = get_representative_indices(
            embeddings=embeddings,
            cluster_labels=cluster_labels_arr,
            cluster_id=cid,
            centroid=centroid,
            max_examples=cfg.max_representative_texts,
        )
        exemplars = [texts[idx] for idx in exemplar_indices]
        cluster_exemplars[cid] = exemplars

        mask = cluster_labels_arr == cid
        mean_prob = float(np.mean(probabilities_arr[mask])) if np.any(mask) else 1.0
        cluster_purities[cid] = round(mean_prob, 4)

        label = label_cluster(
            representative_texts=exemplars,
            cluster_id=cid,
            llm_client=llm_client or llm_callable,
            seed_labels=cfg.seed_labels,
            unclassified_label=cfg.unclassified_label,
        )
        raw_labels[cid] = label

    final_labels = resolve_naming_collisions(raw_labels, unclassified_label=cfg.unclassified_label)
    taxonomy: dict[str, Any] = {}

    for cid in unique_clusters:
        lbl = final_labels[cid]
        exemplars = cluster_exemplars[cid]
        mask = cluster_labels_arr == cid
        c_size = int(np.sum(mask))
        has_severe_consensus = False
        keyword_candidate = False
        llm_agreed = False

        if cid != -1:
            for ex_text in exemplars:
                auditor.total_evaluated = getattr(auditor, "total_evaluated", 0) + 1
                matched_kw = find_matching_keywords(ex_text, cfg.severity_keywords)
                has_cand = len(matched_kw) > 0
                keyword_candidate = keyword_candidate or has_cand
                is_severe, is_disagreement = evaluate_severity_consensus(
                    candidate_flag=has_cand,
                    llm_label=lbl,
                )
                if is_severe:
                    has_severe_consensus = True
                    llm_agreed = True
                elif is_disagreement:
                    auditor.record_disagreement(
                        text=ex_text,
                        matched_keywords=matched_kw,
                        llm_label=lbl,
                        reason=f"Cluster {cid} ('{lbl}') matched severity keywords without Outrage_Escalation consensus",
                    )

        taxonomy[str(cid)] = {
            "cluster_id": int(cid),
            "label": lbl,
            "size": c_size,
            "purity_score": cluster_purities.get(cid, 0.0),
            "representative_examples": exemplars,
            "is_severity_escalation": has_severe_consensus,
            "keyword_candidate": keyword_candidate,
            "llm_agreed": llm_agreed,
        }

    return taxonomy


def persist_taxonomy(
    taxonomy: TaxonomyResult | dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save the induced taxonomy to JSON.

    Args:
        taxonomy: TaxonomyResult instance or dictionary matching schema.
        output_path: Target filesystem path (e.g. data/models/apple_taxonomy.json).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = taxonomy.to_dict() if isinstance(taxonomy, TaxonomyResult) else taxonomy

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Persisted taxonomy with %d clusters to %s", payload.get("num_clusters", 0), path)


save_taxonomy = persist_taxonomy


def load_taxonomy(input_path: str | Path) -> dict[str, Any]:
    """Load persisted taxonomy dictionary from JSON file.

    Args:
        input_path: Path to the persisted JSON file.

    Returns:
        Taxonomy dictionary loaded from JSON.
    """
    path = Path(input_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

