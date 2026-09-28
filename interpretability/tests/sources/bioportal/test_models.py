"""Tests for OntologyClass.from_bioportal_json and SearchResult."""

from typing import Any

import pytest

from molf_interp.sources.bioportal.models import OntologyClass
from tests.sources.bioportal.conftest import mk_class_json


def test_build_minimal() -> None:
    payload: dict[str, Any] = {"@id": "http://example.org/C1", "prefLabel": "MyClass"}
    cls = OntologyClass.from_bioportal_json(payload, "TEST")
    assert cls.iri == "http://example.org/C1"
    assert cls.pref_label == "MyClass"
    assert cls.synonyms == ()
    assert cls.definition is None
    assert cls.obsolete is False


def test_build_full_payload() -> None:
    payload = mk_class_json(
        iri="http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C3262",
        pref_label="Neoplasm",
        synonym=["Tumor", "Tumour"],
        definition=["A new growth of tissue."],
        cui=["C0027651"],
        semanticType=["T191"],
    )
    cls = OntologyClass.from_bioportal_json(payload, "NCIT")
    assert cls.pref_label == "Neoplasm"
    assert "Tumor" in cls.synonyms
    assert cls.definition == "A new growth of tissue."
    assert "C0027651" in cls.cui
    assert "T191" in cls.semantic_types


def test_missing_id_raises() -> None:
    with pytest.raises(ValueError, match="@id"):
        OntologyClass.from_bioportal_json({"prefLabel": "X"}, "TEST")


def test_missing_pref_label_raises() -> None:
    with pytest.raises(ValueError, match="prefLabel"):
        OntologyClass.from_bioportal_json({"@id": "http://example.org/C1"}, "TEST")


def test_empty_pref_label_raises() -> None:
    with pytest.raises(ValueError, match="prefLabel"):
        OntologyClass.from_bioportal_json(
            {"@id": "http://example.org/C1", "prefLabel": "  "}, "TEST"
        )


def test_drops_empty_synonyms() -> None:
    payload: dict[str, Any] = {
        "@id": "http://example.org/C1",
        "prefLabel": "Thing",
        "synonym": ["Valid", "", "  ", "Also valid"],
    }
    cls = OntologyClass.from_bioportal_json(payload, "TEST")
    assert cls.synonyms == ("Valid", "Also valid")


def test_picks_first_non_empty_definition() -> None:
    payload: dict[str, Any] = {
        "@id": "http://example.org/C1",
        "prefLabel": "Thing",
        "definition": ["", "  ", "The real definition."],
    }
    cls = OntologyClass.from_bioportal_json(payload, "TEST")
    assert cls.definition == "The real definition."


def test_strips_apikey_from_links() -> None:
    payload: dict[str, Any] = {
        "@id": "http://example.org/C1",
        "prefLabel": "Thing",
        "links": {
            "self": "https://data.bioontology.org/ontologies/TEST/classes/C1?apikey=secret123",
            "ontology": "https://data.bioontology.org/ontologies/TEST",
        },
    }
    cls = OntologyClass.from_bioportal_json(payload, "TEST")
    assert "secret123" not in cls.links.get("self", "")
    assert "apikey" not in cls.links.get("self", "")


def test_obsolete_preserved() -> None:
    payload: dict[str, Any] = {
        "@id": "http://example.org/C1",
        "prefLabel": "OldThing",
        "obsolete": True,
    }
    cls = OntologyClass.from_bioportal_json(payload, "TEST")
    assert cls.obsolete is True
