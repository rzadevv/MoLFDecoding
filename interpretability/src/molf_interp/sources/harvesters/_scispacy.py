"""scispaCy NER pipeline wrapper for biomedical entity extraction."""

from __future__ import annotations

import re
import string
import time
from collections.abc import Iterable, Iterator
from typing import Any

from loguru import logger

from molf_interp.io.config import BaseConfig
from molf_interp.sources.ncbi.models import PubMedAbstract

_BAD_ENTITIES: frozenset[str] = frozenset(
    {
        # Too generic to be useful
        "cell",
        "cells",
        "tissue",
        "tissues",
        "organ",
        "tumor",
        "tumour",
        "cancer",
        "patient",
        "patients",
        # PubMed NER junk
        "figure",
        "table",
        "image",
        "slide",
        "supplementary",
        # Additional generic/noisy terms
        "disease",
        "diseases",
        "tumors",
        "tumours",
        "cancers",
        "bone",
        "antitumor",
        "antitumour",
        "wound",
        "wounds",
        "neoplasm",
        "neoplasms",
        "lesion",
        "lesions",
    }
)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "to",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
    }
)

_LEADING_ARTICLES = ("the ", "a ", "an ")


class ScispacyEntity(BaseConfig):
    """One NER entity mention with provenance."""

    text: str
    normalized: str
    label: str
    pmid: str
    is_abbreviation: bool = False


def _normalize_entity_text(text: str) -> str | None:
    """Normalize a raw entity span to a canonical form for deduplication.

    Args:
        text: Raw surface form from the NER span.

    Returns:
        Lowercased, stripped, whitespace-collapsed form, or None if too short/noisy.
    """
    normalized = text.strip().strip(string.punctuation + string.whitespace)
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for article in _LEADING_ARTICLES:
        if normalized.startswith(article):
            normalized = normalized[len(article) :]
    if len(normalized) < 3:
        return None
    if normalized.isdigit():
        return None
    words = set(normalized.split())
    if words and words.issubset(_STOPWORDS):
        return None
    return normalized


class ScispacyPipeline:
    """Lazy-loading scispaCy NER pipeline with abbreviation detection.

    The real model is only loaded on the first call to extract_entities().
    For unit tests, inject a pre-loaded nlp via _nlp attribute before calling.
    """

    def __init__(
        self,
        model_name: str = "en_ner_bionlp13cg_md",
        n_process: int = 1,
        batch_size: int = 32,
    ) -> None:
        """Initialise pipeline configuration (model not loaded yet).

        Args:
            model_name: spaCy model name to load.
            n_process: Number of processes for spacy.pipe (1 = no multiprocessing).
            batch_size: spaCy pipe batch size.
        """
        self._model_name = model_name
        self._n_process = n_process
        self._batch_size = batch_size
        self._nlp: Any = None

    def load(self) -> None:
        """Load the scispaCy model and add AbbreviationDetector. Idempotent."""
        if self._nlp is not None:
            return
        import scispacy.abbreviation  # noqa: F401 — registers abbreviation_detector factory
        import spacy

        t0 = time.monotonic()
        nlp = spacy.load(self._model_name)
        nlp.add_pipe("abbreviation_detector")
        self._nlp = nlp
        elapsed = time.monotonic() - t0
        logger.info("Loaded scispaCy model '{}' in {:.1f}s", self._model_name, elapsed)

    def extract_entities(self, abstracts: Iterable[PubMedAbstract]) -> Iterator[ScispacyEntity]:
        """Run NER on a stream of abstracts.

        Calls load() on first use. Uses spacy.pipe() internally for throughput.

        Args:
            abstracts: Stream of PubMedAbstract objects to process.

        Yields:
            ScispacyEntity for each detected (and accepted) entity mention.
        """
        if self._nlp is None:
            self.load()

        abstract_list = list(abstracts)
        texts = [a.full_text for a in abstract_list]
        pmids = [a.pmid for a in abstract_list]

        nlp = self._nlp
        assert nlp is not None  # load() guarantees this
        for doc, pmid in zip(
            nlp.pipe(texts, batch_size=self._batch_size, n_process=self._n_process),
            pmids,
            strict=True,
        ):
            # Track entity texts seen in this doc for abbreviation cross-referencing
            entity_lower_to_label: dict[str, str] = {}
            entity_texts_seen: set[str] = set()

            for ent in doc.ents:
                norm = _normalize_entity_text(ent.text)
                if norm is None or norm in _BAD_ENTITIES:
                    continue
                entity_lower_to_label[ent.text.lower()] = ent.label_
                entity_texts_seen.add(ent.text.lower())
                yield ScispacyEntity(
                    text=ent.text,
                    normalized=norm,
                    label=ent.label_,
                    pmid=pmid,
                    is_abbreviation=False,
                )

            # Walk AbbreviationDetector results
            abbreviations = getattr(getattr(doc, "_", None), "abbreviations", [])
            for abbrev in abbreviations:
                try:
                    short_text: str = abbrev._.short_form.text
                    long_text: str = abbrev._.long_form.text
                except AttributeError:
                    continue

                short_lower = short_text.lower()
                long_lower = long_text.lower()
                short_in = short_lower in entity_texts_seen
                long_in = long_lower in entity_texts_seen

                if short_in and not long_in:
                    label = entity_lower_to_label.get(short_lower, "")
                    norm = _normalize_entity_text(long_text)
                    if norm and norm not in _BAD_ENTITIES and label:
                        yield ScispacyEntity(
                            text=long_text,
                            normalized=norm,
                            label=label,
                            pmid=pmid,
                            is_abbreviation=True,
                        )
                elif long_in and not short_in:
                    label = entity_lower_to_label.get(long_lower, "")
                    norm = _normalize_entity_text(short_text)
                    if norm and norm not in _BAD_ENTITIES and label:
                        yield ScispacyEntity(
                            text=short_text,
                            normalized=norm,
                            label=label,
                            pmid=pmid,
                            is_abbreviation=True,
                        )
