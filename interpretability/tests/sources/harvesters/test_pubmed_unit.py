"""Unit tests for the PubMed harvester — all I/O mocked."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock

from molf_interp.sources.harvesters._scispacy import ScispacyEntity, ScispacyPipeline
from molf_interp.sources.harvesters.pubmed import (
    PubMedHarvestConfig,
    PubMedQuery,
    _aggregate_entities,
    _best_preferred_label,
    harvest_pubmed,
)
from molf_interp.sources.ncbi.client import NCBIClient
from molf_interp.sources.ncbi.models import ESearchResult, PubMedAbstract
from molf_interp.sources.raw_concept import CandidateTier, read_raw_concepts

# ---------- _best_preferred_label ----------


def test_best_preferred_label_most_common() -> None:
    forms = ["Necrosis", "necrosis", "necrosis", "NECROSIS"]
    assert _best_preferred_label(forms) == "necrosis"  # count=3


def test_best_preferred_label_tie_shortest() -> None:
    forms = ["Tumor", "Cancer"]  # both count=1, Cancer shorter
    assert _best_preferred_label(forms) == "Tumor"  # same length, first seen wins


def test_best_preferred_label_first_seen_tiebreak() -> None:
    # All same length and count — first seen wins
    forms = ["alpha", "beta ", "gamma"]  # "alpha" is first
    result = _best_preferred_label(forms)
    assert result == "alpha"


# ---------- _aggregate_entities ----------


def _make_entity(
    text: str,
    normalized: str,
    label: str,
    pmid: str,
    is_abbreviation: bool = False,
) -> ScispacyEntity:
    return ScispacyEntity(
        text=text,
        normalized=normalized,
        label=label,
        pmid=pmid,
        is_abbreviation=is_abbreviation,
    )


def test_aggregate_frequency_filter(tmp_path: Path) -> None:
    config = PubMedHarvestConfig(
        queries=(PubMedQuery(name="q", query="test"),),
        output_path=tmp_path / "out.parquet",
        min_entity_occurrences=3,
    )
    entities = [_make_entity("necrosis", "necrosis", "CANCER", f"pmid{i}") for i in range(5)] + [
        _make_entity("fibrosis", "fibrosis", "TISSUE", "pmid1"),
        _make_entity("fibrosis", "fibrosis", "TISSUE", "pmid2"),
        # only 2 occurrences → should be filtered if min=3
    ]
    pmid_to_queries: dict[str, set[str]] = {f"pmid{i}": {"q"} for i in range(10)}
    records = _aggregate_entities(entities, pmid_to_queries, config)
    norms = {r.preferred_label.lower() for r in records}
    assert "necrosis" in norms
    assert "fibrosis" not in norms  # filtered


def test_aggregate_tier_mapping() -> None:
    config = PubMedHarvestConfig(
        queries=(PubMedQuery(name="q", query="test"),),
        output_path=Path("/tmp/x.parquet"),
        min_entity_occurrences=1,
    )
    entities = [
        _make_entity("necrosis", "necrosis", "CANCER", "p1"),
        _make_entity("lymphocyte", "lymphocyte", "CELL", "p2"),
        _make_entity("kidney", "kidney", "ORGAN", "p3"),
        _make_entity("growth factor", "growth factor", "GENE_OR_GENE_PRODUCT", "p4"),
    ]
    pmid_to_queries = {"p1": {"q"}, "p2": {"q"}, "p3": {"q"}, "p4": {"q"}}
    records = _aggregate_entities(entities, pmid_to_queries, config)
    tiers = {r.preferred_label: r.candidate_tier for r in records}
    assert tiers.get("necrosis") == CandidateTier.H1_MORPHOLOGY
    assert tiers.get("lymphocyte") == CandidateTier.H2_CELL_TYPE
    assert tiers.get("kidney") == CandidateTier.ORGAN_VOCAB
    # GENE_OR_GENE_PRODUCT should be dropped
    assert "growth factor" not in tiers


def test_aggregate_synonyms_collected() -> None:
    config = PubMedHarvestConfig(
        queries=(PubMedQuery(name="q", query="test"),),
        output_path=Path("/tmp/x.parquet"),
        min_entity_occurrences=1,
        max_synonyms_per_entity=5,
    )
    entities = [
        _make_entity("Necrosis", "necrosis", "CANCER", "p1"),
        _make_entity("necrosis", "necrosis", "CANCER", "p2"),
        _make_entity("NECROSIS", "necrosis", "CANCER", "p3"),
    ]
    pmid_to_queries = {"p1": {"q"}, "p2": {"q"}, "p3": {"q"}}
    records = _aggregate_entities(entities, pmid_to_queries, config)
    assert len(records) == 1
    record = records[0]
    # Synonyms should contain the non-preferred surface forms
    assert len(record.synonyms) >= 1


def test_aggregate_occurrence_count_in_extra() -> None:
    config = PubMedHarvestConfig(
        queries=(PubMedQuery(name="q", query="test"),),
        output_path=Path("/tmp/x.parquet"),
        min_entity_occurrences=1,
    )
    entities = [_make_entity("necrosis", "necrosis", "CANCER", f"p{i}") for i in range(7)]
    pmid_to_queries = {f"p{i}": {"q"} for i in range(7)}
    records = _aggregate_entities(entities, pmid_to_queries, config)
    assert records[0].extra["occurrence_count"] == "7"


# ---------- harvest_pubmed end-to-end ----------


def _make_fake_abstract(pmid: str) -> PubMedAbstract:
    return PubMedAbstract(pmid=pmid, title=f"Study {pmid}", abstract="Necrosis observed.")


def _make_mock_client(abstracts: list[PubMedAbstract]) -> MagicMock:
    """Build a mock NCBIClient that yields the given abstracts."""
    client = MagicMock(spec=NCBIClient)
    client.__aenter__ = MagicMock(return_value=client)
    client.__aexit__ = MagicMock(return_value=None)

    client.esearch_pubmed = MagicMock(
        return_value=_coroutine_returning(
            ESearchResult(
                count=len(abstracts),
                pmids=tuple(a.pmid for a in abstracts),
                webenv="WE",
                query_key="1",
            )
        )
    )

    def _efetch_gen(**kwargs: object) -> AsyncGenerator[PubMedAbstract, None]:
        async def _gen() -> AsyncGenerator[PubMedAbstract, None]:
            for ab in abstracts:
                yield ab

        return _gen()

    client.efetch_pubmed_abstracts = MagicMock(side_effect=_efetch_gen)
    return client


def _coroutine_returning(value: object) -> object:
    async def _coro() -> object:
        return value

    return _coro()


class _FakePipeline(ScispacyPipeline):
    """Fake pipeline that emits known entities from abstract text."""

    def extract_entities(self, abstracts: object) -> object:  # type: ignore[override]
        from collections.abc import Iterator

        def _gen() -> Iterator[ScispacyEntity]:
            for ab in abstracts:  # type: ignore[union-attr]
                if "necrosis" in ab.abstract.lower() or "necrosis" in ab.title.lower():
                    yield ScispacyEntity(
                        text="Necrosis",
                        normalized="necrosis",
                        label="CANCER",
                        pmid=ab.pmid,
                    )

        return _gen()


async def test_harvest_pubmed_writes_parquet(tmp_path: Path) -> None:
    abstracts = [_make_fake_abstract(f"p{i}") for i in range(5)]
    client = _make_mock_client(abstracts)
    pipeline = _FakePipeline()

    config = PubMedHarvestConfig(
        queries=(PubMedQuery(name="test_q", query="necrosis"),),
        output_path=tmp_path / "pubmed.parquet",
        min_entity_occurrences=3,
    )

    count = await harvest_pubmed(config, client, pipeline=pipeline)  # type: ignore[arg-type]
    assert count >= 1
    assert (tmp_path / "pubmed.parquet").exists()
    records = read_raw_concepts(tmp_path / "pubmed.parquet")
    assert len(records) == count
    assert all(r.source_name == "pubmed" for r in records)


async def test_harvest_pubmed_empty_query_writes_empty(tmp_path: Path) -> None:
    client = _make_mock_client([])
    pipeline = _FakePipeline()

    config = PubMedHarvestConfig(
        queries=(PubMedQuery(name="empty_q", query="xyz_nonexistent_term"),),
        output_path=tmp_path / "pubmed.parquet",
        min_entity_occurrences=1,
    )

    count = await harvest_pubmed(config, client, pipeline=pipeline)  # type: ignore[arg-type]
    assert count == 0
    assert (tmp_path / "pubmed.parquet").exists()


async def test_harvest_pubmed_occurrence_count_in_extra(tmp_path: Path) -> None:
    abstracts = [_make_fake_abstract(f"p{i}") for i in range(10)]
    client = _make_mock_client(abstracts)
    pipeline = _FakePipeline()

    config = PubMedHarvestConfig(
        queries=(PubMedQuery(name="q", query="test"),),
        output_path=tmp_path / "pubmed.parquet",
        min_entity_occurrences=1,
    )

    await harvest_pubmed(config, client, pipeline=pipeline)  # type: ignore[arg-type]
    records = read_raw_concepts(tmp_path / "pubmed.parquet")
    for r in records:
        count_val = int(r.extra["occurrence_count"])
        assert count_val >= config.min_entity_occurrences
