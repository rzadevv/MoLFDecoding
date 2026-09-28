"""SNOMED CT morphology harvester — gracefully degrades when not licensed."""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from molf_interp.io.config import BaseConfig
from molf_interp.sources.bioportal.client import BioPortalClient
from molf_interp.sources.bioportal.exceptions import NotFoundError
from molf_interp.sources.harvesters.base import normalize_synonyms, truncate_definition
from molf_interp.sources.raw_concept import CandidateTier, RawConcept, write_raw_concepts


class SnomedHarvestConfig(BaseConfig):
    """Configuration for the SNOMED CT morphology harvester."""

    subtree_roots: tuple[str, ...] = ("http://purl.bioontology.org/ontology/SNOMEDCT/49755003",)
    candidate_tier: CandidateTier = CandidateTier.H1_MORPHOLOGY
    include_obsolete: bool = False
    output_path: Path = Path("data/cache/harvest/snomed.parquet")
    ontology_acronym: str = "SNOMEDCT"


async def harvest_snomed(config: SnomedHarvestConfig, client: BioPortalClient) -> int:
    """Harvest SNOMED CT morphology terms via BioPortal subtree traversal.

    Graceful degradation: if the ontology is not accessible (license required),
    logs a warning, writes an empty parquet file, and returns 0 without raising.

    Args:
        config: Harvest configuration.
        client: Authenticated BioPortal client.

    Returns:
        Number of RawConcept records written to config.output_path. 0 if inaccessible.

    Raises:
        NotFoundError: If SNOMED is accessible but the subtree IRI is not found.
    """
    if not await client.can_access(config.ontology_acronym):
        logger.warning(
            "SNOMED CT ({}) is not accessible with the current API key — "
            "a license agreement may be required. Writing empty parquet and skipping.",
            config.ontology_acronym,
        )
        write_raw_concepts([], config.output_path)
        return 0

    records: list[RawConcept] = []
    started = time.perf_counter()

    for root_iri in config.subtree_roots:
        visited = 0
        emitted = 0
        try:
            async for cls in client.iter_descendants(
                config.ontology_acronym,
                root_iri,
                include_obsolete=config.include_obsolete,
            ):
                visited += 1
                records.append(
                    RawConcept(
                        source_name="snomed",
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
        except NotFoundError as exc:
            logger.error(
                "SNOMED CT subtree IRI not found: {}. "
                "Check config.ontology_acronym ('{}') and the IRI. Error: {}",
                root_iri,
                config.ontology_acronym,
                exc,
            )
            raise
        logger.info(
            "SNOMEDCT subtree {}: {} visited, {} emitted",
            root_iri,
            visited,
            emitted,
        )

    write_raw_concepts(records, config.output_path)
    elapsed = time.perf_counter() - started
    logger.info("SNOMED harvest complete: {} records in {:.1f}s", len(records), elapsed)
    return len(records)
