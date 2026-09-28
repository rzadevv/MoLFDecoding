"""Pydantic models for normalized BioPortal API responses."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from pydantic import Field

from molf_interp.io.config import BaseConfig


def _strip_apikey_from_links(links: dict[str, Any]) -> dict[str, str]:
    """Remove apikey query param from any URL values in the links dict."""
    clean: dict[str, str] = {}
    for k, v in links.items():
        if not isinstance(v, str):
            continue
        parsed = urlparse(v)
        qs = parse_qs(parsed.query, keep_blank_values=True)
        qs.pop("apikey", None)
        new_query = urlencode({k2: v2[0] for k2, v2 in qs.items()})
        clean[k] = urlunparse(parsed._replace(query=new_query))
    return clean


class OntologyClass(BaseConfig):
    """A single ontology class normalized from BioPortal JSON.

    Attributes:
        iri: The class IRI (@id field).
        ontology_acronym: Ontology this class belongs to, e.g. "NCIT".
        pref_label: Primary label (prefLabel).
        synonyms: Additional labels, whitespace-stripped, empties dropped.
        definition: First non-empty entry from the definition list, if any.
        parent_iris: IRIs of direct parents when present in the response.
        cui: UMLS CUI codes.
        semantic_types: UMLS semantic type codes.
        obsolete: Whether the class is marked obsolete.
        links: Raw link map with apikey stripped.
    """

    iri: str
    ontology_acronym: str
    pref_label: str
    synonyms: tuple[str, ...] = ()
    definition: str | None = None
    parent_iris: tuple[str, ...] = ()
    cui: tuple[str, ...] = ()
    semantic_types: tuple[str, ...] = ()
    obsolete: bool = False
    links: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_bioportal_json(cls, payload: dict[str, Any], ontology_acronym: str) -> OntologyClass:
        """Build from a single BioPortal class JSON object. Tolerates missing fields.

        Args:
            payload: Raw JSON dict from the BioPortal API.
            ontology_acronym: Ontology acronym to attach to the result.

        Returns:
            A validated OntologyClass instance.

        Raises:
            ValueError: If @id or prefLabel is missing or empty.
        """
        iri = payload.get("@id")
        if not iri:
            raise ValueError("BioPortal class payload missing required '@id' field")

        pref_label = payload.get("prefLabel", "")
        if not pref_label or not str(pref_label).strip():
            raise ValueError(f"BioPortal class {iri!r} has missing or empty 'prefLabel'")

        raw_synonyms: list[Any] = payload.get("synonym", []) or []
        synonyms = tuple(s.strip() for s in raw_synonyms if str(s).strip())

        raw_defs: list[Any] = payload.get("definition", []) or []
        definition: str | None = next((str(d).strip() for d in raw_defs if str(d).strip()), None)

        raw_parents: list[Any] = payload.get("parents", []) or []
        parent_iris = tuple(str(p) for p in raw_parents if p)

        raw_cui: list[Any] = payload.get("cui", []) or []
        cui = tuple(str(c) for c in raw_cui if c)

        raw_st: list[Any] = payload.get("semanticType", []) or []
        semantic_types = tuple(str(s) for s in raw_st if s)

        obsolete: bool = bool(payload.get("obsolete", False))

        raw_links: dict[str, Any] = payload.get("links", {}) or {}
        links = _strip_apikey_from_links(raw_links)

        return cls(
            iri=str(iri),
            ontology_acronym=ontology_acronym,
            pref_label=str(pref_label).strip(),
            synonyms=synonyms,
            definition=definition,
            parent_iris=parent_iris,
            cui=cui,
            semantic_types=semantic_types,
            obsolete=obsolete,
            links=links,
        )


class SearchResult(BaseConfig):
    """A single hit from /search.

    Attributes:
        iri: Class IRI.
        ontology_acronym: Ontology the hit belongs to.
        pref_label: Primary label.
        score: Search relevance score, if reported.
    """

    iri: str
    ontology_acronym: str
    pref_label: str
    score: float | None = None
