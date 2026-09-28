"""PubMed harvester: NCBI E-utilities fetch + scispaCy NER + aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger

from molf_interp.io.config import BaseConfig
from molf_interp.sources.harvesters._scispacy import ScispacyEntity, ScispacyPipeline
from molf_interp.sources.ncbi.client import NCBIClient
from molf_interp.sources.ncbi.models import PubMedAbstract
from molf_interp.sources.raw_concept import CandidateTier, RawConcept, write_raw_concepts

# BioNLP13CG label → CandidateTier mapping; labels absent from this dict are dropped
_LABEL_TO_TIER: dict[str, CandidateTier] = {
    "CANCER": CandidateTier.H1_MORPHOLOGY,
    "TISSUE": CandidateTier.H1_MORPHOLOGY,
    "PATHOLOGICAL_FORMATION": CandidateTier.H1_MORPHOLOGY,
    "CELL": CandidateTier.H2_CELL_TYPE,
    "CELL_TYPE": CandidateTier.H2_CELL_TYPE,
    "ORGAN": CandidateTier.ORGAN_VOCAB,
    "ANATOMICAL_SYSTEM": CandidateTier.ORGAN_VOCAB,
    "DEVELOPING_ANATOMICAL_STRUCTURE": CandidateTier.ORGAN_VOCAB,
    "MULTI-TISSUE_STRUCTURE": CandidateTier.ORGAN_VOCAB,
}


class PubMedQuery(BaseConfig):
    """One named PubMed query configuration."""

    name: str
    query: str
    date_range: tuple[str, str] | None = None
    max_results: int = 1000


class PubMedHarvestConfig(BaseConfig):
    """PubMed harvester configuration."""

    queries: tuple[PubMedQuery, ...]
    min_entity_occurrences: int = 3
    max_synonyms_per_entity: int = 10
    max_pmids_recorded_per_entity: int = 20
    scispacy_model: str = "en_ner_bionlp13cg_md"
    scispacy_batch_size: int = 32
    output_path: Path = Path("data/cache/harvest/pubmed.parquet")


def _best_preferred_label(surface_forms: list[str]) -> str:
    """Select preferred label: most common → shortest → first seen.

    Args:
        surface_forms: All observed surface forms (may repeat).

    Returns:
        The selected preferred label.
    """
    counts = Counter(surface_forms)
    max_count = counts.most_common(1)[0][1]
    # Preserve first-seen order among candidates at max_count
    seen: set[str] = set()
    candidates: list[str] = []
    for form in surface_forms:
        if counts[form] == max_count and form not in seen:
            seen.add(form)
            candidates.append(form)
    # Shortest first; stability gives "first seen" among same-length ties
    return min(candidates, key=len)


def _aggregate_entities(
    entities: list[ScispacyEntity],
    pmid_to_queries: dict[str, set[str]],
    config: PubMedHarvestConfig,
) -> list[RawConcept]:
    """Aggregate raw entity mentions into RawConcept records.

    Groups by (normalized_text, candidate_tier), applies frequency filter,
    selects preferred label, collects synonyms and PMIDs.

    Args:
        entities: All extracted entity mentions.
        pmid_to_queries: Maps each PMID to the query names that retrieved it.
        config: Harvest configuration for thresholds.

    Returns:
        List of RawConcept records surviving the frequency filter.
    """
    # Group by (normalized, tier)
    groups: dict[tuple[str, CandidateTier], list[ScispacyEntity]] = defaultdict(list)
    for entity in entities:
        tier = _LABEL_TO_TIER.get(entity.label)
        if tier is None:
            continue
        groups[(entity.normalized, tier)].append(entity)

    records: list[RawConcept] = []
    for (normalized, tier), group in groups.items():
        occurrence_count = len(group)
        if occurrence_count < config.min_entity_occurrences:
            continue

        surface_forms = [e.text for e in group]
        preferred = _best_preferred_label(surface_forms)

        # Unique non-preferred surface forms for synonyms (preserve first-seen order)
        seen_syns: set[str] = {preferred}
        syns: list[str] = []
        for form in surface_forms:
            if form not in seen_syns:
                seen_syns.add(form)
                syns.append(form)
                if len(syns) >= config.max_synonyms_per_entity:
                    break

        # First N unique PMIDs
        seen_pmids: set[str] = set()
        pmid_list: list[str] = []
        for e in group:
            if e.pmid not in seen_pmids:
                seen_pmids.add(e.pmid)
                pmid_list.append(e.pmid)
                if len(pmid_list) >= config.max_pmids_recorded_per_entity:
                    break

        # Union of query names for all contributing PMIDs
        query_names: set[str] = set()
        for e in group:
            query_names.update(pmid_to_queries.get(e.pmid, set()))

        records.append(
            RawConcept(
                source_name="pubmed",
                source_id=f"pubmed:{normalized}:{tier.value}",
                preferred_label=preferred,
                synonyms=tuple(sorted(syns)),
                definition=None,
                parent_ids=(),
                candidate_tier=tier,
                extra={
                    "occurrence_count": str(occurrence_count),
                    "pmids": ",".join(pmid_list),
                    "queries": ",".join(sorted(query_names)),
                },
            )
        )

    return records


async def harvest_pubmed(
    config: PubMedHarvestConfig,
    client: NCBIClient,
    pipeline: ScispacyPipeline | None = None,
) -> int:
    """Run all queries → fetch abstracts → NER → aggregate → write parquet.

    Args:
        config: Harvest configuration.
        client: Async NCBI client.
        pipeline: scispaCy pipeline (constructed from config if None).

    Returns:
        Number of RawConcept records written.
    """
    if pipeline is None:
        pipeline = ScispacyPipeline(
            model_name=config.scispacy_model,
            batch_size=config.scispacy_batch_size,
        )

    all_abstracts: list[PubMedAbstract] = []
    pmid_to_queries: dict[str, set[str]] = defaultdict(set)

    for query_cfg in config.queries:
        logger.info(
            "Query {!r}: {} (max_results={})",
            query_cfg.name,
            query_cfg.query,
            query_cfg.max_results,
        )
        result = await client.esearch_pubmed(
            query_cfg.query,
            max_results=query_cfg.max_results,
            date_range=query_cfg.date_range,
            use_history=True,
        )
        logger.info(
            "Query '{}': {} total PubMed records, fetching up to {}",
            query_cfg.name,
            result.count,
            query_cfg.max_results,
        )

        if result.count == 0:
            logger.warning("Query '{}' returned 0 results", query_cfg.name)
            continue

        async for abstract in client.efetch_pubmed_abstracts(
            webenv=result.webenv,
            query_key=result.query_key,
            total=min(result.count, query_cfg.max_results),
        ):
            pmid_to_queries[abstract.pmid].add(query_cfg.name)
            all_abstracts.append(abstract)

    logger.info("Total abstracts fetched: {}", len(all_abstracts))

    if not all_abstracts:
        logger.warning("No abstracts fetched — writing empty parquet")
        write_raw_concepts([], config.output_path)
        return 0

    logger.info("Running scispaCy NER on {} abstracts…", len(all_abstracts))
    all_entities = list(pipeline.extract_entities(all_abstracts))
    logger.info("Raw entity mentions: {}", len(all_entities))

    records = _aggregate_entities(all_entities, dict(pmid_to_queries), config)
    logger.info(
        "Unique entities after normalization: {}, surviving frequency filter (>= {}): {}",
        len(
            {
                (e.normalized, _LABEL_TO_TIER.get(e.label))
                for e in all_entities
                if _LABEL_TO_TIER.get(e.label) is not None
            }
        ),
        config.min_entity_occurrences,
        len(records),
    )

    write_raw_concepts(records, config.output_path)
    logger.info(
        "PubMed harvest complete: {} RawConcepts written to {}", len(records), config.output_path
    )
    return len(records)
