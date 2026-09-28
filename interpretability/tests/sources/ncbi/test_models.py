"""Tests for NCBI models."""

from molf_interp.sources.ncbi.models import ESearchResult, PubMedAbstract, PubMedAuthor


def test_esearch_result_construction() -> None:
    r = ESearchResult(count=100, pmids=("1", "2", "3"), webenv="WE", query_key="1")
    assert r.count == 100
    assert r.pmids == ("1", "2", "3")
    assert r.webenv == "WE"
    assert r.query_key == "1"


def test_esearch_result_optional_fields() -> None:
    r = ESearchResult(count=0, pmids=())
    assert r.webenv is None
    assert r.query_key is None


def test_pubmed_abstract_full_text_property() -> None:
    ab = PubMedAbstract(pmid="123", title="Title here", abstract="Abstract body.")
    assert ab.full_text == "Title here\n\nAbstract body."


def test_pubmed_abstract_empty_abstract() -> None:
    ab = PubMedAbstract(pmid="999", title="Title only", abstract="")
    assert ab.full_text == "Title only\n\n"


def test_pubmed_abstract_authors_and_mesh() -> None:
    ab = PubMedAbstract(
        pmid="42",
        title="Study",
        abstract="Text",
        authors=(PubMedAuthor(last_name="Smith", fore_name="J"),),
        mesh_terms=("Necrosis", "Fibrosis"),
    )
    assert ab.authors[0].last_name == "Smith"
    assert "Necrosis" in ab.mesh_terms


def test_pubmed_abstract_defaults() -> None:
    ab = PubMedAbstract(pmid="1", title="T", abstract="A")
    assert ab.journal is None
    assert ab.pub_year is None
    assert ab.authors == ()
    assert ab.mesh_terms == ()
    assert ab.keywords == ()
    assert ab.doi is None
