"""NCIt morphology harvester — consumes BioPortalClient, emits RawConcept records."""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from molf_interp.io.config import BaseConfig
from molf_interp.sources.bioportal.client import BioPortalClient
from molf_interp.sources.harvesters.base import (
    is_obviously_non_morphological,
    normalize_synonyms,
    truncate_definition,
)
from molf_interp.sources.raw_concept import CandidateTier, RawConcept, write_raw_concepts


class NCItHarvestConfig(BaseConfig):
    """Configuration for the NCIt harvester."""

    subtree_roots: tuple[str, ...] = (
        "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C7057",
        "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C12219",
    )
    candidate_tier: CandidateTier = CandidateTier.H1_MORPHOLOGY
    include_obsolete: bool = False
    apply_obvious_reject_filter: bool = True
    output_path: Path = Path("data/cache/harvest/ncit.parquet")


async def harvest_ncit(config: NCItHarvestConfig, client: BioPortalClient) -> int:
    """Harvest NCIt morphology terms via BioPortal subtree traversal.

    Args:
        config: Harvest configuration.
        client: Authenticated BioPortal client.

    Returns:
        Number of RawConcept records written to config.output_path.
    """
    records: list[RawConcept] = []
    started = time.perf_counter()

    for root_iri in config.subtree_roots:
        visited = 0
        emitted = 0
        async for cls in client.iter_descendants(
            "NCIT", root_iri, include_obsolete=config.include_obsolete
        ):
            visited += 1
            if config.apply_obvious_reject_filter and is_obviously_non_morphological(
                cls.pref_label
            ):
                continue
            records.append(
                RawConcept(
                    source_name="ncit",
                    source_id=cls.iri,
                    preferred_label=cls.pref_label,
                    synonyms=normalize_synonyms(cls.synonyms, drop_label=cls.pref_label),
                    definition=truncate_definition(cls.definition),
                    parent_ids=cls.parent_iris,
                    candidate_tier=config.candidate_tier,
                    extra={
                        "cui": ",".join(cls.cui),
                        "semantic_types": ",".join(cls.semantic_types),
                    },
                )
            )
            emitted += 1
        logger.info("NCIt subtree {}: {} visited, {} emitted", root_iri, visited, emitted)

    write_raw_concepts(records, config.output_path)
    elapsed = time.perf_counter() - started
    logger.info("NCIt harvest complete: {} records in {:.1f}s", len(records), elapsed)
    return len(records)
