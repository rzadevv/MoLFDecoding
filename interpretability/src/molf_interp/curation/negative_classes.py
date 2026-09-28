"""Negative-class centroids for the Stage 3 visual filter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import yaml
from loguru import logger

from molf_interp.io.config import BaseConfig

if TYPE_CHECKING:
    from molf_interp.curation.sapbert import SapBert
    from molf_interp.sources.bioportal.client import BioPortalClient


class NegativeClassConfig(BaseConfig):
    """Configuration for negative-class centroid construction."""

    disease_bioportal_acronym: str = "NCIT"
    disease_subtree_iri: str = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C2991"
    disease_ncit_fallback_iri: str = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C2991"
    disease_max_terms: int = 500

    anatomy_uberon_path: Path = Path("data/cache/harvest/uberon_organs.parquet")
    anatomy_max_terms: int = 500

    procedure_ncit_iri: str = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25218"
    procedure_max_terms: int = 300

    clinical_metadata_yaml: Path = Path("configs/curation/negative_classes.yaml")

    cache_dir: Path = Path("data/cache/negative_classes")


_CENTROID_KEYS = ("disease", "anatomy", "procedure", "clinical_metadata")


class NegativeClassCentroids:
    """L2-normalized mean SapBERT embeddings for the four negative classes.

    Immutable value object — set attributes only via the constructor.
    """

    __slots__ = _CENTROID_KEYS

    def __init__(
        self,
        disease: npt.NDArray[Any],
        anatomy: npt.NDArray[Any],
        procedure: npt.NDArray[Any],
        clinical_metadata: npt.NDArray[Any],
    ) -> None:
        """Initialize centroids.

        Args:
            disease: L2-normalized mean embedding for disease terms [768].
            anatomy: L2-normalized mean embedding for anatomy terms [768].
            procedure: L2-normalized mean embedding for procedure terms [768].
            clinical_metadata: L2-normalized mean embedding for clinical-metadata terms [768].
        """
        object.__setattr__(self, "disease", disease)
        object.__setattr__(self, "anatomy", anatomy)
        object.__setattr__(self, "procedure", procedure)
        object.__setattr__(self, "clinical_metadata", clinical_metadata)

    def __setattr__(self, name: str, value: object) -> None:
        """Block mutation after construction."""
        raise AttributeError("NegativeClassCentroids is immutable")

    def as_dict(self) -> dict[str, npt.NDArray[Any]]:
        """Return a name → centroid mapping.

        Returns:
            Dict with keys disease, anatomy, procedure, clinical_metadata.
        """
        return {k: getattr(self, k) for k in _CENTROID_KEYS}

    def max_similarity(self, candidate_emb: npt.NDArray[Any]) -> tuple[str, float]:
        """Return (class_name, cosine_sim) for the most similar negative class.

        Args:
            candidate_emb: L2-normalized embedding of shape [768].

        Returns:
            Tuple of (nearest_class_name, cosine_similarity).
        """
        best_name = ""
        best_sim = -2.0
        for name, centroid in self.as_dict().items():
            sim = float(np.dot(candidate_emb, centroid))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        return best_name, best_sim

    def save(self, directory: Path) -> None:
        """Save centroids as individual .npy files under ``directory``.

        Args:
            directory: Target directory (created if absent).
        """
        directory.mkdir(parents=True, exist_ok=True)
        for name, centroid in self.as_dict().items():
            np.save(directory / f"{name}.npy", centroid)
        logger.info("Saved negative-class centroids to {}", directory)

    @classmethod
    def load(cls, directory: Path) -> NegativeClassCentroids:
        """Load centroids previously saved with :meth:`save`.

        Args:
            directory: Directory containing the .npy files.

        Returns:
            Loaded NegativeClassCentroids instance.

        Raises:
            FileNotFoundError: If any expected .npy file is missing.
        """
        arrays: dict[str, npt.NDArray[Any]] = {}
        for name in _CENTROID_KEYS:
            path = directory / f"{name}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing centroid file: {path}")
            arrays[name] = np.load(path)
        return cls(**arrays)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Term loading helpers
# ---------------------------------------------------------------------------


def _load_clinical_metadata_terms(yaml_path: Path) -> list[str]:
    with yaml_path.open() as fh:
        data = yaml.safe_load(fh)
    return [str(t) for t in data.get("clinical_metadata_terms", [])]


def _load_anatomy_terms(parquet_path: Path, max_terms: int) -> list[str]:
    import pandas as pd

    if not parquet_path.exists():
        logger.warning(
            "UBERON parquet not found at {}; anatomy centroid will be empty", parquet_path
        )
        return []
    df = pd.read_parquet(parquet_path, columns=["preferred_label"])
    terms: list[str] = df["preferred_label"].dropna().astype(str).tolist()
    return terms[:max_terms]


def _cache_path(config: NegativeClassConfig, class_name: str) -> Path:
    return config.cache_dir / f"{class_name}_terms.json"


def _load_cached_terms(config: NegativeClassConfig, class_name: str) -> list[str] | None:
    path = _cache_path(config, class_name)
    if not path.exists():
        return None
    try:
        result: list[str] = json.loads(path.read_text())
        return result
    except Exception:
        return None


def _save_cached_terms(config: NegativeClassConfig, class_name: str, terms: list[str]) -> None:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(config, class_name).write_text(json.dumps(terms))


async def _fetch_bioportal_subtree(
    bioportal_client: BioPortalClient,
    acronym: str,
    root_iri: str,
    max_terms: int,
) -> list[str]:
    """Yield up to max_terms preferred labels from a BioPortal subtree.

    Args:
        bioportal_client: Async BioPortal client.
        acronym: Ontology acronym.
        root_iri: Root class IRI.
        max_terms: Maximum number of terms to return.

    Returns:
        List of preferred-label strings.
    """
    terms: list[str] = []
    async for cls in bioportal_client.iter_descendants(acronym, root_iri):
        if cls.pref_label:
            terms.append(cls.pref_label)
        if len(terms) >= max_terms:
            break
    return terms


async def build_negative_class_terms(
    config: NegativeClassConfig,
    bioportal_client: BioPortalClient,
) -> dict[str, list[str]]:
    """Pull all four negative-class term lists.

    Anatomy and clinical_metadata are loaded locally.  Disease and procedure pull from
    BioPortal (with disk caching). If MeSH access returns 403, the disease list
    falls back to NCIt C2991.

    Args:
        config: NegativeClassConfig with paths and IRIs.
        bioportal_client: Async BioPortalClient for disease/procedure fetches.

    Returns:
        Dict mapping class name → list of term strings.
    """
    from molf_interp.sources.bioportal.exceptions import LicensedOntologyError

    # --- anatomy (local UBERON) ----------------------------------------
    anatomy_terms = _load_anatomy_terms(config.anatomy_uberon_path, config.anatomy_max_terms)

    # --- clinical_metadata (local YAML) ----------------------------------
    cm_terms = _load_clinical_metadata_terms(config.clinical_metadata_yaml)

    # --- disease (BioPortal MESH, cached) --------------------------------
    disease_terms = _load_cached_terms(config, "disease")
    if disease_terms is None:
        try:
            disease_terms = await _fetch_bioportal_subtree(
                bioportal_client,
                config.disease_bioportal_acronym,
                config.disease_subtree_iri,
                config.disease_max_terms,
            )
            _save_cached_terms(config, "disease", disease_terms)
        except LicensedOntologyError:
            logger.warning(
                "MeSH access returned 403; falling back to NCIt C2991 for disease centroid"
            )
            try:
                disease_terms = await _fetch_bioportal_subtree(
                    bioportal_client,
                    "NCIT",
                    config.disease_ncit_fallback_iri,
                    config.disease_max_terms,
                )
                _save_cached_terms(config, "disease", disease_terms)
            except Exception as exc:
                logger.warning("NCIt disease fallback failed: {}; using empty list", exc)
                disease_terms = []
        except Exception as exc:
            logger.warning("BioPortal disease fetch failed: {}; using empty list", exc)
            disease_terms = []

    # --- procedure (BioPortal NCIT, cached) ------------------------------
    procedure_terms = _load_cached_terms(config, "procedure")
    if procedure_terms is None:
        try:
            procedure_terms = await _fetch_bioportal_subtree(
                bioportal_client,
                "NCIT",
                config.procedure_ncit_iri,
                config.procedure_max_terms,
            )
            _save_cached_terms(config, "procedure", procedure_terms)
        except Exception as exc:
            logger.warning("BioPortal procedure fetch failed: {}; using empty list", exc)
            procedure_terms = []

    return {
        "disease": disease_terms,
        "anatomy": anatomy_terms,
        "procedure": procedure_terms,
        "clinical_metadata": cm_terms,
    }


def compute_negative_centroids(
    terms_by_class: dict[str, list[str]],
    sapbert: SapBert,
) -> NegativeClassCentroids:
    """Embed all terms per class and return L2-normalized mean centroids.

    Args:
        terms_by_class: Dict mapping class name → list of term strings.
        sapbert: SapBert instance used for encoding.

    Returns:
        NegativeClassCentroids with one 768-dim L2-normalized vector per class.
    """
    centroids: dict[str, npt.NDArray[Any]] = {}
    for class_name in _CENTROID_KEYS:
        terms = terms_by_class.get(class_name, [])
        if not terms:
            logger.warning("No terms for negative class '{}'; using zero centroid", class_name)
            centroids[class_name] = np.zeros(sapbert.embedding_dim, dtype=np.float32)
            continue
        logger.info("  Computing centroid for '{}' from {} terms…", class_name, len(terms))
        embs: npt.NDArray[Any] = sapbert.encode(terms)
        mean_emb: npt.NDArray[Any] = embs.mean(axis=0)
        norm = float(np.linalg.norm(mean_emb))
        if norm > 0:
            mean_emb = (mean_emb / norm).astype(np.float32)
        centroids[class_name] = mean_emb
    return NegativeClassCentroids(**centroids)  # type: ignore[arg-type]
