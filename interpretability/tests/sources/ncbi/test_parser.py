"""Tests for PubMed XML parsers."""

import pytest

from molf_interp.sources.ncbi.exceptions import MalformedResponseError
from molf_interp.sources.ncbi.parser import (
    parse_efetch_pubmed_response,
    parse_esearch_response,
)
from tests.sources.ncbi.conftest import fake_efetch_xml, fake_esearch_xml

# ---------- parse_esearch_response ----------


def test_esearch_with_webenv() -> None:
    xml = fake_esearch_xml(count=5, pmids=["1", "2"], webenv="WE123", query_key="1")
    result = parse_esearch_response(xml)
    assert result.count == 5
    assert result.pmids == ("1", "2")
    assert result.webenv == "WE123"
    assert result.query_key == "1"


def test_esearch_without_webenv() -> None:
    xml = fake_esearch_xml(count=2, pmids=["10", "20"], webenv=None, query_key=None)
    result = parse_esearch_response(xml)
    assert result.webenv is None
    assert result.query_key is None


def test_esearch_empty_result() -> None:
    xml = fake_esearch_xml(count=0, pmids=[], webenv=None, query_key=None)
    result = parse_esearch_response(xml)
    assert result.count == 0
    assert result.pmids == ()


def test_esearch_malformed_xml() -> None:
    with pytest.raises(MalformedResponseError):
        parse_esearch_response("<not valid xml")


def test_esearch_error_body() -> None:
    xml = "<eSearchResult><ERROR>Invalid term</ERROR></eSearchResult>"
    with pytest.raises(MalformedResponseError, match="Invalid term"):
        parse_esearch_response(xml)


def test_esearch_missing_count() -> None:
    xml = "<eSearchResult><IdList></IdList></eSearchResult>"
    with pytest.raises(MalformedResponseError, match="missing <Count>"):
        parse_esearch_response(xml)


# ---------- parse_efetch_pubmed_response ----------


def test_efetch_three_articles() -> None:
    xml = fake_efetch_xml(
        [
            {"pmid": "1", "title": "T1", "abstract_text": "A1", "year": "2023"},
            {"pmid": "2", "title": "T2", "abstract_text": "A2", "year": "2022"},
            {"pmid": "3", "title": "T3", "abstract_text": "A3", "year": "2021"},
        ]
    )
    results = parse_efetch_pubmed_response(xml)
    assert len(results) == 3
    assert results[0].pmid == "1"
    assert results[0].title == "T1"
    assert results[0].abstract == "A1"
    assert results[0].pub_year == 2023


def test_efetch_structured_abstract() -> None:
    xml = fake_efetch_xml(
        [
            {
                "pmid": "99",
                "title": "Study",
                "abstract_text": [
                    {"label": "BACKGROUND", "text": "Background text."},
                    {"label": "METHODS", "text": "Methods text."},
                    {"label": "RESULTS", "text": "Results text."},
                ],
            }
        ]
    )
    results = parse_efetch_pubmed_response(xml)
    assert len(results) == 1
    ab = results[0]
    assert "BACKGROUND:" in ab.abstract
    assert "METHODS:" in ab.abstract
    assert "RESULTS:" in ab.abstract


def test_efetch_missing_abstract() -> None:
    xml = (
        "<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID>"
        "<Article><Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>"
        "<Title>J</Title></Journal>"
        "<ArticleTitle>Title without abstract</ArticleTitle></Article>"
        "</MedlineCitation></PubmedArticle></PubmedArticleSet>"
    )
    results = parse_efetch_pubmed_response(xml)
    assert len(results) == 1
    assert results[0].abstract == ""


def test_efetch_missing_year() -> None:
    xml = fake_efetch_xml([{"pmid": "55", "title": "T", "abstract_text": "A", "year": ""}])
    results = parse_efetch_pubmed_response(xml)
    assert results[0].pub_year is None


def test_efetch_extracts_mesh_and_keywords() -> None:
    xml = fake_efetch_xml(
        [
            {
                "pmid": "77",
                "title": "T",
                "abstract_text": "A",
                "mesh_terms": ["Necrosis", "Fibrosis"],
                "keywords": ["histology", "staining"],
            }
        ]
    )
    results = parse_efetch_pubmed_response(xml)
    assert "Necrosis" in results[0].mesh_terms
    assert "histology" in results[0].keywords


def test_efetch_doi_extracted() -> None:
    xml = fake_efetch_xml(
        [{"pmid": "88", "title": "T", "abstract_text": "A", "doi": "10.9999/test"}]
    )
    results = parse_efetch_pubmed_response(xml)
    assert results[0].doi == "10.9999/test"


def test_efetch_malformed_xml() -> None:
    with pytest.raises(MalformedResponseError):
        parse_efetch_pubmed_response("<broken xml <<")
