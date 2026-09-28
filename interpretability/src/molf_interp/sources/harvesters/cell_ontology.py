"""Cell Ontology harvester — CL cell-type subtree via BioPortalClient."""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from molf_interp.io.config import BaseConfig
from molf_interp.sources.bioportal.client import BioPortalClient
from molf_interp.sources.harvesters.base import normalize_synonyms, truncate_definition
from molf_interp.sources.raw_concept import CandidateTier, RawConcept, write_raw_concepts


class CellOntologyHarvestConfig(BaseConfig):
    """Configuration for the Cell Ontology harvester."""

    subtree_roots: tuple[str, ...] = ("http://purl.obolibrary.org/obo/CL_0000000",)
    candidate_tier: CandidateTier = CandidateTier.H2_CELL_TYPE
    include_obsolete: bool = False
    output_path: Path = Path("data/cache/harvest/cell_ontology.parquet")


async def harvest_cell_ontology(config: CellOntologyHarvestConfig, client: BioPortalClient) -> int:
    """Harvest Cell Ontology cell-type terms via BioPortal subtree traversal.

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
            "CL", root_iri, include_obsolete=config.include_obsolete
        ):
            visited += 1
            records.append(
                RawConcept(
                    source_name="cell_ontology",
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
        logger.info("CL subtree {}: {} visited, {} emitted", root_iri, visited, emitted)

    write_raw_concepts(records, config.output_path)
    elapsed = time.perf_counter() - started
    logger.info("Cell Ontology harvest complete: {} records in {:.1f}s", len(records), elapsed)
    return len(records)
