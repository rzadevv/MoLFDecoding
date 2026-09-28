"""UBERON harvester for organ-level anatomy curation-stage normalization."""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from molf_interp.io.config import BaseConfig
from molf_interp.sources.bioportal.client import BioPortalClient
from molf_interp.sources.harvesters.base import normalize_synonyms, truncate_definition
from molf_interp.sources.raw_concept import CandidateTier, RawConcept, write_raw_concepts


class UberonHarvestConfig(BaseConfig):
    """Configuration for the UBERON organ-table harvester.

    Note: UBERON output is a side table for organ-name curation-stage normalization.
    It is NOT inserted into the final ConceptBank.
    """

    subtree_roots: tuple[str, ...] = ("http://purl.obolibrary.org/obo/UBERON_0000062",)
    candidate_tier: CandidateTier = CandidateTier.ORGAN_VOCAB
    include_obsolete: bool = False
    output_path: Path = Path("data/cache/harvest/uberon_organs.parquet")


async def harvest_uberon(config: UberonHarvestConfig, client: BioPortalClient) -> int:
    """Harvest UBERON organ-level anatomical terms via BioPortal subtree traversal.

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
            "UBERON", root_iri, include_obsolete=config.include_obsolete
        ):
            visited += 1
            records.append(
                RawConcept(
                    source_name="uberon",
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
        logger.info("UBERON subtree {}: {} visited, {} emitted", root_iri, visited, emitted)

    write_raw_concepts(records, config.output_path)
    elapsed = time.perf_counter() - started
    logger.info("UBERON harvest complete: {} records in {:.1f}s", len(records), elapsed)
    return len(records)
