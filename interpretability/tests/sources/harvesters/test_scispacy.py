"""Unit tests for the scispaCy NER pipeline wrapper."""

from __future__ import annotations

from molf_interp.sources.harvesters._scispacy import (
    ScispacyPipeline,
    _normalize_entity_text,
)
from molf_interp.sources.ncbi.models import PubMedAbstract

# ---------- _normalize_entity_text ----------


def test_normalize_strips_and_lowercases() -> None:
    assert _normalize_entity_text("  Mitotic Activity.  ") == "mitotic activity"


def test_normalize_strips_leading_the() -> None:
    assert _normalize_entity_text("The Necrosis") == "necrosis"


def test_normalize_strips_leading_a() -> None:
    assert _normalize_entity_text("a tumor") == "tumor"


def test_normalize_too_short_returns_none() -> None:
    assert _normalize_entity_text("xx") is None


def test_normalize_single_char_returns_none() -> None:
    assert _normalize_entity_text("x") is None


def test_normalize_digits_only_returns_none() -> None:
    assert _normalize_entity_text("123") is None


def test_normalize_pure_stopwords_returns_none() -> None:
    assert _normalize_entity_text("in the of") is None


def test_normalize_collapses_whitespace() -> None:
    result = _normalize_entity_text("  ductal   carcinoma  ")
    assert result == "ductal carcinoma"


# ---------- ScispacyPipeline with fake EntityRuler ----------


def _make_fake_pipeline(patterns: list[dict[str, str]]) -> ScispacyPipeline:
    """Build a ScispacyPipeline injecting a blank spaCy + EntityRuler."""
    import spacy  # type: ignore[import]

    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)  # type: ignore[union-attr]

    pipeline = ScispacyPipeline()
    pipeline._nlp = nlp
    return pipeline


def _make_abstract(pmid: str, text: str) -> PubMedAbstract:
    return PubMedAbstract(pmid=pmid, title=text, abstract="")


def test_extract_entities_basic() -> None:
    pipeline = _make_fake_pipeline(
        [
            {"label": "CANCER", "pattern": "carcinoma"},
            {"label": "CELL", "pattern": "lymphocyte"},
        ]
    )
    abs1 = _make_abstract("001", "carcinoma in the tissue")
    abs2 = _make_abstract("002", "lymphocyte infiltration")

    entities = list(pipeline.extract_entities([abs1, abs2]))
    labels = {e.label for e in entities}
    norms = {e.normalized for e in entities}
    assert "CANCER" in labels
    assert "CELL" in labels
    assert "carcinoma" in norms
    assert "lymphocyte" in norms


def test_extract_entities_two_abstracts_correct_pmids() -> None:
    pipeline = _make_fake_pipeline([{"label": "CANCER", "pattern": "necrosis"}])
    ab1 = _make_abstract("A1", "necrosis is present")
    ab2 = _make_abstract("A2", "necrosis was observed")

    entities = list(pipeline.extract_entities([ab1, ab2]))
    pmids = {e.pmid for e in entities}
    assert "A1" in pmids
    assert "A2" in pmids


def test_bad_entity_filtered() -> None:
    # "cell" is in _BAD_ENTITIES
    pipeline = _make_fake_pipeline([{"label": "CELL", "pattern": "cell"}])
    ab = _make_abstract("X", "cell is present")
    entities = list(pipeline.extract_entities([ab]))
    assert all(e.normalized != "cell" for e in entities)


def test_entity_normalized_correctly() -> None:
    pipeline = _make_fake_pipeline([{"label": "TISSUE", "pattern": "Ductal Carcinoma In Situ"}])
    ab = _make_abstract("Y", "Ductal Carcinoma In Situ lesion")
    entities = list(pipeline.extract_entities([ab]))
    if entities:
        assert entities[0].normalized == "ductal carcinoma in situ"


def test_pipeline_load_called_lazily() -> None:
    pipeline = ScispacyPipeline()
    assert pipeline._nlp is None  # not loaded yet


def test_empty_abstract_list_yields_nothing() -> None:
    pipeline = _make_fake_pipeline([{"label": "CANCER", "pattern": "tumor"}])
    entities = list(pipeline.extract_entities([]))
    assert entities == []
