"""Tests for negative_classes module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from molf_interp.curation.negative_classes import (
    NegativeClassCentroids,
    _load_clinical_metadata_terms,
    compute_negative_centroids,
)
from tests.curation.conftest import make_mock_sapbert

# ---------------------------------------------------------------------------
# NegativeClassCentroids
# ---------------------------------------------------------------------------


def _make_centroids(dim: int = 768) -> NegativeClassCentroids:
    rng = np.random.default_rng(0)

    def _norm(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float32)
        return (v / np.linalg.norm(v)).astype(np.float32)

    return NegativeClassCentroids(
        disease=_norm(rng.standard_normal(dim)),
        anatomy=_norm(rng.standard_normal(dim)),
        procedure=_norm(rng.standard_normal(dim)),
        clinical_metadata=_norm(rng.standard_normal(dim)),
    )


def test_centroids_immutable() -> None:
    c = _make_centroids()
    with pytest.raises(AttributeError):
        c.disease = np.zeros(768, dtype=np.float32)  # type: ignore[misc]


def test_centroids_as_dict_keys() -> None:
    c = _make_centroids()
    d = c.as_dict()
    assert set(d) == {"disease", "anatomy", "procedure", "clinical_metadata"}


def test_centroids_max_similarity_returns_nearest() -> None:
    dim = 768
    disease = np.zeros(dim, dtype=np.float32)
    disease[0] = 1.0  # points along first axis
    anatomy = np.zeros(dim, dtype=np.float32)
    anatomy[1] = 1.0
    procedure = np.zeros(dim, dtype=np.float32)
    procedure[2] = 1.0
    cm = np.zeros(dim, dtype=np.float32)
    cm[3] = 1.0

    centroids = NegativeClassCentroids(
        disease=disease, anatomy=anatomy, procedure=procedure, clinical_metadata=cm
    )

    candidate = np.zeros(dim, dtype=np.float32)
    candidate[1] = 1.0  # closest to anatomy

    name, sim = centroids.max_similarity(candidate)
    assert name == "anatomy"
    assert abs(sim - 1.0) < 1e-5


def test_centroids_save_load_roundtrip(tmp_path: Path) -> None:
    c = _make_centroids()
    c.save(tmp_path)
    loaded = NegativeClassCentroids.load(tmp_path)
    for key in ("disease", "anatomy", "procedure", "clinical_metadata"):
        np.testing.assert_allclose(getattr(c, key), getattr(loaded, key), atol=1e-6)


def test_centroids_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        NegativeClassCentroids.load(tmp_path)


# ---------------------------------------------------------------------------
# compute_negative_centroids
# ---------------------------------------------------------------------------


def test_compute_negative_centroids_l2_normalized() -> None:
    sapbert = make_mock_sapbert()
    terms = {
        "disease": ["cancer", "tumor", "malignancy"],
        "anatomy": ["lung", "kidney", "liver"],
        "procedure": ["biopsy", "resection"],
        "clinical_metadata": ["patient age", "tumor stage I"],
    }
    centroids = compute_negative_centroids(terms, sapbert)
    for key in ("disease", "anatomy", "procedure", "clinical_metadata"):
        v = getattr(centroids, key)
        norm = float(np.linalg.norm(v))
        assert abs(norm - 1.0) < 1e-4, f"Centroid {key} not L2-normalized: norm={norm}"


def test_compute_negative_centroids_empty_class_gives_zero() -> None:
    sapbert = make_mock_sapbert()
    terms = {
        "disease": [],
        "anatomy": ["lung"],
        "procedure": ["biopsy"],
        "clinical_metadata": ["patient age"],
    }
    centroids = compute_negative_centroids(terms, sapbert)
    np.testing.assert_array_equal(centroids.disease, np.zeros(768, dtype=np.float32))


# ---------------------------------------------------------------------------
# _load_clinical_metadata_terms
# ---------------------------------------------------------------------------


def test_load_clinical_metadata_yaml(tmp_path: Path) -> None:
    yaml_content = "clinical_metadata_terms:\n  - tumor stage I\n  - patient age\n"
    p = tmp_path / "neg.yaml"
    p.write_text(yaml_content)
    terms = _load_clinical_metadata_terms(p)
    assert "tumor stage I" in terms
    assert "patient age" in terms


# ---------------------------------------------------------------------------
# build_negative_class_terms — cache hit, BioPortal fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_negative_class_terms_uses_clinical_yaml(tmp_path: Path) -> None:
    from molf_interp.curation.negative_classes import (
        NegativeClassConfig,
        build_negative_class_terms,
    )

    yaml_content = "clinical_metadata_terms:\n  - tumor stage I\n"
    yaml_path = tmp_path / "neg.yaml"
    yaml_path.write_text(yaml_content)

    # Create fake cached disease/procedure json files to skip BioPortal
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "disease_terms.json").write_text('["cancer", "tumor"]')
    (cache_dir / "procedure_terms.json").write_text('["biopsy"]')

    # Create fake uberon parquet
    import pandas as pd

    uberon_path = tmp_path / "uberon.parquet"
    pd.DataFrame({"preferred_label": ["lung", "kidney"]}).to_parquet(uberon_path)

    cfg = NegativeClassConfig(
        anatomy_uberon_path=uberon_path,
        clinical_metadata_yaml=yaml_path,
        cache_dir=cache_dir,
    )
    mock_client = MagicMock()
    terms = await build_negative_class_terms(cfg, mock_client)

    assert "tumor stage I" in terms["clinical_metadata"]
    assert "cancer" in terms["disease"]
    assert "biopsy" in terms["procedure"]
    assert "lung" in terms["anatomy"]
    # BioPortal was not called since cache files existed
    mock_client.iter_descendants.assert_not_called()


@pytest.mark.asyncio
async def test_build_negative_class_terms_mesh_403_fallback(tmp_path: Path) -> None:
    from molf_interp.curation.negative_classes import (
        NegativeClassConfig,
        build_negative_class_terms,
    )
    from molf_interp.sources.bioportal.exceptions import LicensedOntologyError

    yaml_path = tmp_path / "neg.yaml"
    yaml_path.write_text("clinical_metadata_terms: []")
    uberon_path = tmp_path / "uberon.parquet"
    import pandas as pd

    pd.DataFrame({"preferred_label": ["lung"]}).to_parquet(uberon_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Disease cache absent → will call BioPortal
    (cache_dir / "procedure_terms.json").write_text('["biopsy"]')

    cfg = NegativeClassConfig(
        anatomy_uberon_path=uberon_path,
        clinical_metadata_yaml=yaml_path,
        cache_dir=cache_dir,
    )

    async def _mesh_fail(*args, **kwargs):  # type: ignore[override]
        raise LicensedOntologyError("MESH", "403 license")
        yield  # make it an async generator

    async def _ncit_ok(*args, **kwargs):  # type: ignore[override]
        class _FakeCls:
            pref_label = "Carcinoma"
            obsolete = False

        yield _FakeCls()

    mock_client = MagicMock()
    # First call (MeSH disease) → raises 403; second call (NCIt fallback) → yields one term
    mock_client.iter_descendants.side_effect = [_mesh_fail(), _ncit_ok()]

    terms = await build_negative_class_terms(cfg, mock_client)
    assert "Carcinoma" in terms["disease"]
