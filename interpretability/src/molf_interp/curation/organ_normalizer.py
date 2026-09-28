"""UBERON-based organ inference for H1_morphology concepts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from molf_interp.curation.sapbert import SapBert

# Curated list of ~30 high-frequency pathology organs
_PATHOLOGY_ORGANS: frozenset[str] = frozenset(
    {
        "lung",
        "breast",
        "liver",
        "kidney",
        "prostate",
        "colon",
        "colorectum",
        "brain",
        "skin",
        "pancreas",
        "bladder",
        "ovary",
        "uterus",
        "endometrium",
        "stomach",
        "thyroid",
        "head and neck",
        "bone marrow",
        "lymph node",
        "spleen",
        "heart",
        "muscle",
        "nerve",
        "cervix",
        "testis",
        "adrenal",
        "esophagus",
        "gallbladder",
        "peritoneum",
        "pleura",
        "thymus",
    }
)

# Aliases → canonical organ name
_ORGAN_ALIASES: dict[str, str] = {
    "pulmonary": "lung",
    "hepatic": "liver",
    "hepatocellular": "liver",
    "renal": "kidney",
    "gastric": "stomach",
    "colonic": "colon",
    "colorectal": "colon",
    "rectal": "colon",
    "mammary": "breast",
    "cerebral": "brain",
    "glioma": "brain",
    "glioblastoma": "brain",
    "cutaneous": "skin",
    "melanoma": "skin",
    "splenic": "spleen",
    "cardiac": "heart",
    "thymic": "thymus",
    "cervical": "cervix",
    "endometrial": "uterus",
    "ovarian": "ovary",
    "testicular": "testis",
    "prostatic": "prostate",
    "pancreatic": "pancreas",
    "adrenal": "adrenal",
    "esophageal": "esophagus",
}


def _build_synonym_dict(uberon_path: Path) -> dict[str, str]:
    """Build lowercase synonym → canonical name dict from UBERON parquet."""
    try:
        df = pd.read_parquet(uberon_path)
        result: dict[str, str] = {}
        if "preferred_label" not in df.columns:
            return result
        for _, row in df.iterrows():
            label = str(row["preferred_label"]).lower().strip()
            if label in _PATHOLOGY_ORGANS:
                result[label] = label
            syns_raw = row.get("synonyms")
            if syns_raw is None:
                syns: list[Any] = []
            elif isinstance(syns_raw, list | np.ndarray):
                syns = list(syns_raw)
            else:
                syns = []
            if isinstance(syns, list):
                for s in syns:
                    sl = str(s).lower().strip()
                    if label in _PATHOLOGY_ORGANS:
                        result[sl] = label
        return result
    except Exception as exc:
        logger.warning("Could not load UBERON table: {}; using keyword-only fallback", exc)
        return {}


class OrganNormalizer:
    """Map a free-text label to a canonical organ name using UBERON + keyword rules."""

    def __init__(
        self,
        uberon_path: Path,
        sapbert: SapBert | None = None,
    ) -> None:
        """Initialize with UBERON parquet path and optional SapBERT for fuzzy matching.

        Args:
            uberon_path: Path to UBERON organs parquet file.
            sapbert: Optional SapBert instance for fuzzy embedding-based matching.
        """
        self._synonym_dict = _build_synonym_dict(uberon_path)
        # Merge in hardcoded organs and aliases
        for organ in _PATHOLOGY_ORGANS:
            self._synonym_dict.setdefault(organ, organ)
        for alias, canonical in _ORGAN_ALIASES.items():
            self._synonym_dict.setdefault(alias, canonical)
        self._sapbert = sapbert
        # Pre-compute sorted organ list for fuzzy matching
        self._organs = sorted(_PATHOLOGY_ORGANS)
        self._organ_embeddings: npt.NDArray[Any] | None = None

    def _get_organ_embeddings(self) -> npt.NDArray[Any]:
        if self._organ_embeddings is None and self._sapbert is not None:
            self._organ_embeddings = self._sapbert.encode(self._organs)
        return self._organ_embeddings  # type: ignore[return-value]

    def normalize(self, label: str, *, fuzzy_threshold: float = 0.70) -> str | None:
        """Return canonical organ name or None.

        Args:
            label: Free-text label to normalize.
            fuzzy_threshold: Minimum cosine similarity for SapBERT fuzzy match.

        Returns:
            Canonical organ name, or None if no match found.
        """
        lower = label.lower().strip()

        # 1. Exact match in synonym dict
        if lower in self._synonym_dict:
            return self._synonym_dict[lower]

        # 2. Substring match: check if any organ name appears in the label
        for word, canonical in sorted(self._synonym_dict.items(), key=lambda x: -len(x[0])):
            pattern = r"\b" + re.escape(word) + r"\b"
            if re.search(pattern, lower):
                return canonical

        # 3. Fuzzy SapBERT match
        if self._sapbert is not None:
            emb = self._sapbert.encode([label])
            organ_embs = self._get_organ_embeddings()
            sims = self._sapbert.cosine_similarity(emb, organ_embs)[0]
            best_idx = int(np.argmax(sims))
            if float(sims[best_idx]) >= fuzzy_threshold:
                return self._organs[best_idx]

        return None

    def infer_from_concept(
        self,
        preferred_label: str,
        synonyms: list[str],
        default: str = "universal",
    ) -> str:
        """Infer organ from preferred_label, then synonyms.

        Args:
            preferred_label: Primary concept label.
            synonyms: List of synonym strings.
            default: Value to return if no organ found.

        Returns:
            Canonical organ name or default.
        """
        result = self.normalize(preferred_label)
        if result:
            return result
        for syn in synonyms:
            result = self.normalize(syn)
            if result:
                return result
        return default
