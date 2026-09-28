"""Stage 3: visual similarity filter against the reference concept bank."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

from molf_interp.concepts.schemas import ConceptBank, Tier
from molf_interp.concepts.store import read_concept_bank
from molf_interp.curation.config import CurationConfig, VisualFilterConfig
from molf_interp.curation.sapbert import SapBert

if TYPE_CHECKING:
    from molf_interp.curation.calibration import CalibratedThresholds
    from molf_interp.curation.negative_classes import NegativeClassCentroids

_TIER_MAP: dict[str, Tier] = {
    "H1_morphology": Tier.H1_MORPHOLOGY,
    "H2_cell_type": Tier.H2_CELL_TYPE,
    "H2_niche": Tier.H2_NICHE,
    "H2_gene_program": Tier.H2_GENE_PROGRAM,
    "H2_pathway": Tier.H2_PATHWAY,
}

_DecisionLiteral = Literal[
    "keep",
    "ambiguous",
    "drop_low_similarity",
    "drop_negative_proximity",
    "drop_low_tier_agreement",
]


@dataclass(frozen=True)
class VisualFilterDecision:
    """Per-candidate visual filter outcome."""

    decision: _DecisionLiteral
    reference_tier_similarity: float
    max_negative_similarity: float
    nearest_negative_class: str
    knn_tier_agreement: float
    knn_top_tier: str


# ---------------------------------------------------------------------------
# Helpers kept for backward-compat tests
# ---------------------------------------------------------------------------


def _decide(sim: float, keep_t: float, drop_t: float) -> str:
    """Simple threshold decision (kept for unit tests of the old path).

    Args:
        sim: Cosine similarity to reference tier centroid.
        keep_t: Keep threshold.
        drop_t: Drop threshold.

    Returns:
        "keep", "ambiguous", or "drop".
    """
    if sim >= keep_t:
        return "keep"
    if sim <= drop_t:
        return "drop"
    return "ambiguous"


def _get_tier_thresholds(
    tier: str,
    config: VisualFilterConfig,
) -> tuple[float, float]:
    """Return (keep_threshold, drop_threshold) for the given tier.

    Args:
        tier: Tier string, e.g. "H1_morphology".
        config: VisualFilterConfig with optional per_tier_thresholds override.

    Returns:
        (keep_t, drop_t) tuple.
    """
    if tier in config.per_tier_thresholds:
        return config.per_tier_thresholds[tier]
    return config.similarity_keep_threshold, config.similarity_drop_threshold


# ---------------------------------------------------------------------------
# Centroid builder
# ---------------------------------------------------------------------------


def _compute_centroid(
    ref_names: list[str],
    sapbert: SapBert,
    method: Literal["mean", "topk_max"],
    topk: int,
) -> npt.NDArray[Any]:
    if not ref_names:
        result: npt.NDArray[Any] = np.zeros(sapbert.embedding_dim, dtype=np.float32)
        return result
    embs: npt.NDArray[Any] = sapbert.encode(ref_names)
    if method == "mean":
        centroid: npt.NDArray[Any] = embs.mean(axis=0)
        norm: float = float(np.linalg.norm(centroid))
        return (centroid / norm).astype(np.float32) if norm > 0 else centroid
    sims: npt.NDArray[Any] = embs @ embs.T
    avg_sims: npt.NDArray[Any] = sims.mean(axis=1)
    top_indices: npt.NDArray[Any] = np.argsort(avg_sims)[-topk:]
    centroid2: npt.NDArray[Any] = embs[top_indices].mean(axis=0)
    norm2: float = float(np.linalg.norm(centroid2))
    return (centroid2 / norm2).astype(np.float32) if norm2 > 0 else centroid2


# ---------------------------------------------------------------------------
# KNN agreement computation
# ---------------------------------------------------------------------------


def _compute_knn_agreement(
    candidate_embs: npt.NDArray[Any],
    candidate_tiers: list[str],
    ref_embs: npt.NDArray[Any],
    ref_tiers: list[str],
    k: int,
    batch_size: int,
) -> tuple[list[float], list[str]]:
    """Compute KNN tier agreement for each candidate.

    For each candidate, find the k nearest reference concepts (across all tiers) and
    compute the fraction whose tier matches the candidate's tier.

    Args:
        candidate_embs: [N, D] candidate embeddings.
        candidate_tiers: Tier string for each candidate (length N).
        ref_embs: [M, D] reference concept embeddings.
        ref_tiers: Tier string for each reference concept (length M).
        k: Number of nearest neighbours.
        batch_size: Candidates per batch to avoid OOM.

    Returns:
        Tuple of (agreements, top_tiers) each of length N.
    """
    n = candidate_embs.shape[0]
    m = ref_embs.shape[0]

    if m == 0:
        return [0.0] * n, [""] * n

    k_actual = min(k, m)
    agreements: list[float] = []
    top_tiers: list[str] = []

    for batch_start in range(0, n, batch_size):
        batch = candidate_embs[batch_start : batch_start + batch_size]
        sims: npt.NDArray[Any] = batch @ ref_embs.T  # [B, M]
        # Get top-k indices per candidate
        if k_actual < m:
            # Use partial sort for efficiency
            topk_indices = np.argpartition(sims, -k_actual, axis=1)[:, -k_actual:]
        else:
            topk_indices = np.tile(np.arange(m), (sims.shape[0], 1))

        for i in range(sims.shape[0]):
            cand_tier = candidate_tiers[batch_start + i]
            top_idx = topk_indices[i]
            neighbor_tiers = [ref_tiers[j] for j in top_idx]
            agreement = sum(1 for t in neighbor_tiers if t == cand_tier) / len(neighbor_tiers)
            top_tier = Counter(neighbor_tiers).most_common(1)[0][0] if neighbor_tiers else ""
            agreements.append(agreement)
            top_tiers.append(top_tier)

    return agreements, top_tiers


# ---------------------------------------------------------------------------
# Main stage function
# ---------------------------------------------------------------------------


def run_stage_3(
    df: pd.DataFrame,
    config: CurationConfig,
    sapbert: SapBert,
    reference_bank: ConceptBank | None = None,
    negative_centroids: NegativeClassCentroids | None = None,
    thresholds: CalibratedThresholds | None = None,
) -> pd.DataFrame:
    """Visual filter: KNN agreement + negative-class proximity check.

    For each candidate:
      1. Compute cosine similarity to the reference same-tier centroid.
      2. Compute max cosine similarity across all 4 negative-class centroids.
      3. Find top-K nearest reference concepts (all tiers) and compute tier agreement.
      4. Apply per-tier thresholds to assign one of five decisions.

    Args:
        df: DataFrame with 'embedding' column from Stage 1/2.
        config: CurationConfig instance.
        sapbert: SapBert instance for encoding reference concept names.
        reference_bank: Optional pre-loaded ConceptBank; loaded from disk if None.
        negative_centroids: Optional NegativeClassCentroids; if None, skips
            negative-proximity check.
        thresholds: Optional CalibratedThresholds; if None, falls back to
            VisualFilterConfig global thresholds.

    Returns:
        DataFrame filtered to keep + ambiguous rows, with new columns:
        reference_tier_similarity, max_negative_similarity, nearest_negative_class,
        knn_tier_agreement, knn_top_tier, visual_filter_decision.
    """
    t0 = time.monotonic()
    vf = config.visual_filter

    if reference_bank is None:
        reference_bank = read_concept_bank(vf.reference_bank_dir)

    logger.info("Stage 3: visual filter on {} rows…", len(df))

    # Build per-tier centroids and all-tier reference embedding matrix
    tier_centroids: dict[str, npt.NDArray[Any]] = {}
    all_ref_embs_list: list[npt.NDArray[Any]] = []
    all_ref_tiers: list[str] = []

    for tier_str, tier_enum in _TIER_MAP.items():
        ref_concepts = reference_bank.filter_concepts(tier=tier_enum)
        names = [c.concept_name for c in ref_concepts]
        if not names:
            logger.warning("No reference concepts for tier {}; using zero centroid", tier_str)
        centroid = _compute_centroid(names, sapbert, vf.reference_centroid_method, vf.topk)
        tier_centroids[tier_str] = centroid
        if names:
            t_embs = sapbert.encode(names)
            all_ref_embs_list.append(t_embs)
            all_ref_tiers.extend([tier_str] * len(names))
        logger.info("  Reference tier {}: {} concepts", tier_str, len(names))

    su_embs_all: npt.NDArray[Any] = (
        np.vstack(all_ref_embs_list)
        if all_ref_embs_list
        else np.zeros((0, sapbert.embedding_dim), dtype=np.float32)
    )

    # Stack all candidate embeddings for vectorised KNN
    candidate_emb_list = df["embedding"].tolist()
    if candidate_emb_list:
        cand_embs: npt.NDArray[Any] = np.stack(candidate_emb_list)
    else:
        cand_embs = np.zeros((0, sapbert.embedding_dim), dtype=np.float32)

    cand_tiers_list = df["candidate_tier"].astype(str).tolist()

    # KNN agreement
    knn_agreements, knn_top_tiers = _compute_knn_agreement(
        cand_embs,
        cand_tiers_list,
        su_embs_all,
        all_ref_tiers,
        k=vf.knn_k,
        batch_size=vf.knn_batch_size,
    )

    # Per-candidate decisions
    df = df.copy()
    ref_tier_sims: list[float] = []
    max_neg_sims: list[float] = []
    nearest_neg_classes: list[str] = []
    decisions: list[str] = []

    for i, (_, row) in enumerate(df.iterrows()):
        tier = str(row["candidate_tier"])
        emb: npt.NDArray[Any] = cand_embs[i]

        # Tier centroid similarity
        ref_sim = 0.0 if tier not in tier_centroids else float(np.dot(emb, tier_centroids[tier]))

        # Negative-class proximity
        if negative_centroids is not None:
            nearest_neg, max_neg_sim = negative_centroids.max_similarity(emb)
        else:
            nearest_neg, max_neg_sim = "", -2.0

        # Thresholds (priority: explicit per_tier_thresholds > calibrated > global)
        if tier in vf.per_tier_thresholds:
            keep_t, drop_t = vf.per_tier_thresholds[tier]
            knn_agree_t = 0.5
        elif thresholds is not None:
            try:
                tt = thresholds.for_tier(tier)
                keep_t = tt.keep_threshold
                drop_t = tt.drop_threshold
                knn_agree_t = tt.knn_tier_agreement_threshold
            except KeyError:
                keep_t = vf.similarity_keep_threshold
                drop_t = vf.similarity_drop_threshold
                knn_agree_t = 0.5
        else:
            keep_t = vf.similarity_keep_threshold
            drop_t = vf.similarity_drop_threshold
            knn_agree_t = 0.5

        knn_agree = knn_agreements[i]

        # Decision logic (order matters)
        if tier not in tier_centroids:
            decision: _DecisionLiteral = "drop_low_similarity"
        elif negative_centroids is not None and max_neg_sim >= ref_sim:
            decision = "drop_negative_proximity"
        elif ref_sim < drop_t:
            decision = "drop_low_similarity"
        elif knn_agree < knn_agree_t:
            decision = "drop_low_tier_agreement"
        elif ref_sim >= keep_t:
            decision = "keep"
        else:
            decision = "ambiguous"

        ref_tier_sims.append(ref_sim)
        max_neg_sims.append(max_neg_sim)
        nearest_neg_classes.append(nearest_neg)
        decisions.append(decision)

    df["reference_tier_similarity"] = ref_tier_sims
    df["su_similarity"] = ref_tier_sims  # backward-compat alias
    df["max_negative_similarity"] = max_neg_sims
    df["nearest_negative_class"] = nearest_neg_classes
    df["knn_tier_agreement"] = knn_agreements
    df["knn_top_tier"] = knn_top_tiers
    df["visual_filter_decision"] = decisions

    # Log per-tier breakdown
    decision_series = pd.Series(decisions)
    tier_series = pd.Series(cand_tiers_list)
    for tier_str in sorted(set(cand_tiers_list)):
        mask = tier_series == tier_str
        dcounts = decision_series[mask.values].value_counts()
        logger.info(
            "  {}: keep={}, ambiguous={}, drop_low_sim={}, drop_neg_prox={}, drop_knn={}",
            tier_str,
            dcounts.get("keep", 0),
            dcounts.get("ambiguous", 0),
            dcounts.get("drop_low_similarity", 0),
            dcounts.get("drop_negative_proximity", 0),
            dcounts.get("drop_low_tier_agreement", 0),
        )

    drop_mask = pd.Series(decisions).str.startswith("drop")
    result = df[~drop_mask.values].copy()
    elapsed = time.monotonic() - t0
    logger.info(
        "Stage 3 done: {} → {} rows ({} dropped), {:.1f}s",
        len(df),
        len(result),
        drop_mask.sum(),
        elapsed,
    )
    return result


# Alias so pipeline.py can call either name
run_stage_3_v2 = run_stage_3
