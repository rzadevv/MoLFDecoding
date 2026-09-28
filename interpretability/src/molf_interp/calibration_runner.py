"""Async helper to build negative centroids and run threshold calibration."""

from __future__ import annotations

from molf_interp.curation.calibration import CalibratedThresholds, calibrate_thresholds
from molf_interp.curation.config import CurationConfig
from molf_interp.curation.negative_classes import (
    NegativeClassCentroids,
    compute_negative_centroids,
)
from molf_interp.curation.sapbert import SapBert


async def run_calibration(config: CurationConfig) -> CalibratedThresholds:
    """Build negative centroids (if needed) and calibrate thresholds.

    Loads the reference bank and existing negative-centroids cache if present;
    otherwise pulls from BioPortal and recomputes.

    Args:
        config: CurationConfig instance.

    Returns:
        Calibrated CalibratedThresholds.
    """
    from loguru import logger

    from molf_interp.concepts.store import read_concept_bank

    sapbert = SapBert(config.sapbert)
    reference_bank = read_concept_bank(config.visual_filter.reference_bank_dir)

    neg_centroids_dir = config.negative_classes.cache_dir / "centroids"
    try:
        neg_centroids = NegativeClassCentroids.load(neg_centroids_dir)
        logger.info("Loaded cached negative centroids from {}", neg_centroids_dir)
    except FileNotFoundError:
        logger.info("Building negative centroids from scratch…")
        from molf_interp.curation.negative_classes import build_negative_class_terms
        from molf_interp.sources.bioportal.client import BioPortalClient
        from molf_interp.sources.bioportal.config import BioPortalConfig

        bp_config = BioPortalConfig()
        async with BioPortalClient.from_config(bp_config) as bp_client:
            terms_by_class = await build_negative_class_terms(config.negative_classes, bp_client)
        neg_centroids = compute_negative_centroids(terms_by_class, sapbert)
        neg_centroids.save(neg_centroids_dir)

    return calibrate_thresholds(reference_bank, sapbert, neg_centroids)
