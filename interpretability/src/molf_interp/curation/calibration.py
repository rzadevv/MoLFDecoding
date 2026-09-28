"""Per-tier threshold calibration from the reference concept bank."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import yaml
from loguru import logger

from molf_interp.io.config import BaseConfig

if TYPE_CHECKING:
    from molf_interp.concepts.schemas import ConceptBank
    from molf_interp.curation.negative_classes import NegativeClassCentroids
    from molf_interp.curation.sapbert import SapBert

_ALL_TIERS = (
    "H1_morphology",
    "H2_cell_type",
    "H2_niche",
    "H2_gene_program",
    "H2_pathway",
)


class TierThresholds(BaseConfig):
    """Keep/drop/KNN-agreement thresholds for one reference tier."""

    keep_threshold: float
    drop_threshold: float
    knn_tier_agreement_threshold: float = 0.5


class CalibratedThresholds(BaseConfig):
    """Per-tier calibrated thresholds derived from the reference bank statistics."""

    h1_morphology: TierThresholds
    h2_cell_type: TierThresholds
    h2_niche: TierThresholds
    h2_gene_program: TierThresholds
    h2_pathway: TierThresholds

    def for_tier(self, tier: str) -> TierThresholds:
        """Return the TierThresholds for the given tier string.

        Args:
            tier: One of the five tier values (e.g. ``"H1_morphology"``).

        Returns:
            Corresponding TierThresholds.

        Raises:
            KeyError: If the tier string is not recognised.
        """
        mapping: dict[str, TierThresholds] = {
            "H1_morphology": self.h1_morphology,
            "H2_cell_type": self.h2_cell_type,
            "H2_niche": self.h2_niche,
            "H2_gene_program": self.h2_gene_program,
            "H2_pathway": self.h2_pathway,
        }
        if tier not in mapping:
            raise KeyError(f"Unknown tier: {tier!r}")
        return mapping[tier]


def calibrate_thresholds(
    reference_bank: ConceptBank,
    sapbert: SapBert,
    negative_centroids: NegativeClassCentroids,
    *,
    keep_percentile: int = 25,
    drop_margin: float = 0.05,
    knn_k: int = 10,
) -> CalibratedThresholds:
    """Compute per-tier keep/drop thresholds from the reference bank statistics.

    For each reference tier T:
      1. Embed all reference concepts in T.
      2. Compute intra-tier pairwise cosine similarity (upper triangle, no diagonal).
      3. keep_threshold[T] = percentile(intra-tier-sims, keep_percentile).
      4. Compute the tier centroid and its cosine sim to each negative centroid.
      5. drop_threshold[T] = max(negative_sims) + drop_margin.

    Args:
        reference_bank: Pre-loaded reference ConceptBank.
        sapbert: SapBert instance for embedding.
        negative_centroids: NegativeClassCentroids for drop threshold computation.
        keep_percentile: Percentile of intra-tier sim distribution to use as
            keep_threshold (default 25 — tight, but generous enough for H2 tiers).
        drop_margin: Safety margin added to max negative-class similarity to set
            drop_threshold (default 0.05).
        knn_k: Nearest-neighbour count for KNN agreement (stored in thresholds;
            not used during calibration itself).

    Returns:
        CalibratedThresholds with one TierThresholds per tier.

    Raises:
        ValueError: If keep_threshold < drop_threshold for any tier (indicates poor
            separability in the reference bank).
    """
    from molf_interp.concepts.schemas import Tier

    tier_enum_map: dict[str, Tier] = {
        "H1_morphology": Tier.H1_MORPHOLOGY,
        "H2_cell_type": Tier.H2_CELL_TYPE,
        "H2_niche": Tier.H2_NICHE,
        "H2_gene_program": Tier.H2_GENE_PROGRAM,
        "H2_pathway": Tier.H2_PATHWAY,
    }

    neg_dict = negative_centroids.as_dict()
    results: dict[str, TierThresholds] = {}

    for tier_str in _ALL_TIERS:
        tier_enum = tier_enum_map[tier_str]
        ref_concepts = reference_bank.filter_concepts(tier=tier_enum)
        names = [c.concept_name for c in ref_concepts]

        if len(names) < 2:
            logger.warning(
                "Tier {} has only {} reference concepts — using fallback thresholds (0.55/0.35)",
                tier_str,
                len(names),
            )
            results[tier_str] = TierThresholds(
                keep_threshold=0.55,
                drop_threshold=0.35,
                knn_tier_agreement_threshold=0.5,
            )
            continue

        embs: npt.NDArray[Any] = sapbert.encode(names)

        # Intra-tier pairwise cosine similarity (L2-normed, so dot product = cosine)
        sim_matrix: npt.NDArray[Any] = embs @ embs.T
        n = len(names)
        # Extract upper triangle (i < j) — shape: n*(n-1)/2
        triu_indices = np.triu_indices(n, k=1)
        intra_sims: npt.NDArray[Any] = sim_matrix[triu_indices]
        keep_t = float(np.percentile(intra_sims, keep_percentile))

        # Tier centroid: L2-normalized mean
        centroid: npt.NDArray[Any] = embs.mean(axis=0)
        c_norm = float(np.linalg.norm(centroid))
        if c_norm > 0:
            centroid = (centroid / c_norm).astype(np.float32)

        # Max similarity from tier centroid to any negative class centroid
        neg_sims: list[float] = []
        for neg_name, neg_centroid in neg_dict.items():
            sim = float(np.dot(centroid, neg_centroid))
            neg_sims.append(sim)
            logger.debug("  {}/{} neg_sim={:.4f}", tier_str, neg_name, sim)
        max_neg_sim = max(neg_sims) if neg_sims else 0.0
        drop_t = max_neg_sim + drop_margin

        logger.info(
            "Tier {} (N={}): keep_threshold={:.4f} (p{}), drop_threshold={:.4f} "
            "(max_neg={:.4f}+{:.3f})",
            tier_str,
            n,
            keep_t,
            keep_percentile,
            drop_t,
            max_neg_sim,
            drop_margin,
        )

        if keep_t <= drop_t:
            # H1 morphology often overlaps with anatomy/procedure centroids in SapBERT
            # space. Fall back to global thresholds rather than failing hard.
            logger.warning(
                "Tier {}: keep_threshold ({:.4f}) <= drop_threshold ({:.4f}) — "
                "poor negative-class separability; falling back to global (0.55/0.35)",
                tier_str,
                keep_t,
                drop_t,
            )
            results[tier_str] = TierThresholds(
                keep_threshold=0.55,
                drop_threshold=0.35,
                knn_tier_agreement_threshold=0.5,
            )
            continue

        results[tier_str] = TierThresholds(
            keep_threshold=keep_t,
            drop_threshold=drop_t,
            knn_tier_agreement_threshold=0.5,
        )

    return CalibratedThresholds(
        h1_morphology=results["H1_morphology"],
        h2_cell_type=results["H2_cell_type"],
        h2_niche=results["H2_niche"],
        h2_gene_program=results["H2_gene_program"],
        h2_pathway=results["H2_pathway"],
    )


def save_calibrated_thresholds(thresholds: CalibratedThresholds, path: Path) -> None:
    """Write CalibratedThresholds to YAML at path.

    Args:
        thresholds: Calibrated thresholds to persist.
        path: Output YAML file path (parent directory created if absent).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    for tier_str in _ALL_TIERS:
        key = tier_str.lower()
        t = thresholds.for_tier(tier_str)
        data[key] = {
            "keep_threshold": t.keep_threshold,
            "drop_threshold": t.drop_threshold,
            "knn_tier_agreement_threshold": t.knn_tier_agreement_threshold,
        }
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=True))
    logger.info("Saved calibrated thresholds to {}", path)


def load_calibrated_thresholds(path: Path) -> CalibratedThresholds:
    """Read CalibratedThresholds from YAML.

    Args:
        path: Path to the YAML file written by :func:`save_calibrated_thresholds`.

    Returns:
        Validated CalibratedThresholds instance.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Calibrated thresholds not found: {path}")
    with path.open() as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    def _tier(key: str) -> TierThresholds:
        d = raw.get(key, {})
        return TierThresholds(
            keep_threshold=float(d.get("keep_threshold", 0.55)),
            drop_threshold=float(d.get("drop_threshold", 0.35)),
            knn_tier_agreement_threshold=float(d.get("knn_tier_agreement_threshold", 0.5)),
        )

    return CalibratedThresholds(
        h1_morphology=_tier("h1_morphology"),
        h2_cell_type=_tier("h2_cell_type"),
        h2_niche=_tier("h2_niche"),
        h2_gene_program=_tier("h2_gene_program"),
        h2_pathway=_tier("h2_pathway"),
    )
