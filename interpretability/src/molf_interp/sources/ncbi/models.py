"""Pydantic models for NCBI E-utilities responses."""

from __future__ import annotations

from molf_interp.io.config import BaseConfig


class ESearchResult(BaseConfig):
    """Result of a PubMed ESearch call."""

    count: int
    pmids: tuple[str, ...]
    webenv: str | None = None
    query_key: str | None = None


class PubMedAuthor(BaseConfig):
    """One author entry from a PubMed record."""

    last_name: str | None = None
    fore_name: str | None = None


class PubMedAbstract(BaseConfig):
    """One PubMed record, normalised from XML."""

    pmid: str
    title: str
    abstract: str
    journal: str | None = None
    pub_year: int | None = None
    authors: tuple[PubMedAuthor, ...] = ()
    mesh_terms: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    doi: str | None = None

    @property
    def full_text(self) -> str:
        """Title + abstract joined with newline, used as NER input."""
        return f"{self.title}\n\n{self.abstract}"
