"""Tests for UMAP projection and plot rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from molf_interp.comparison.config import ComparisonConfig
from molf_interp.comparison.matcher import match_banks
from molf_interp.comparison.merger import merge_banks
from molf_interp.comparison.visualization import (
    compute_umap_projection,
    render_umap_plot,
)
from tests.comparison.conftest import MockSapBert


def _make_merged_df(tiny_reference_bank, tiny_harvested_bank) -> pd.DataFrame:
    mock = MockSapBert()
    matches = match_banks(tiny_reference_bank, tiny_harvested_bank, mock, threshold=0.85)
    result = merge_banks(tiny_reference_bank, tiny_harvested_bank, matches)
    return result.merged_concepts


def test_umap_projection_shape(tiny_reference_bank, tiny_harvested_bank) -> None:
    """UMAP projection returns (N, 2) coords with no NaNs."""
    merged = _make_merged_df(tiny_reference_bank, tiny_harvested_bank)
    mock = MockSapBert()
    cfg = ComparisonConfig(umap_n_neighbors=2, umap_min_dist=0.1, umap_max_points=1000)
    umap_result = compute_umap_projection(merged, mock, cfg)  # type: ignore[arg-type]

    assert umap_result.coords.shape == (len(merged), 2)
    assert not np.any(np.isnan(umap_result.coords))


def test_umap_sampling_cap(tiny_reference_bank, tiny_harvested_bank) -> None:
    """With umap_max_points=5 and 14-row fixture, result is at most 5 points."""
    merged = _make_merged_df(tiny_reference_bank, tiny_harvested_bank)
    mock = MockSapBert()
    cfg = ComparisonConfig(umap_n_neighbors=2, umap_min_dist=0.1, umap_max_points=5)
    umap_result = compute_umap_projection(merged, mock, cfg)  # type: ignore[arg-type]

    assert umap_result.coords.shape[0] <= 5
    assert umap_result.coords.shape[0] > 0
    assert len(umap_result.labels_concept_id) == umap_result.coords.shape[0]


def test_plot_file_written(tmp_path: Path, tiny_reference_bank, tiny_harvested_bank) -> None:
    """render_umap_plot writes a non-empty PNG."""
    merged = _make_merged_df(tiny_reference_bank, tiny_harvested_bank)
    mock = MockSapBert()
    cfg = ComparisonConfig(umap_n_neighbors=2, umap_min_dist=0.1, umap_max_points=1000, plot_dpi=72)
    umap_result = compute_umap_projection(merged, mock, cfg)  # type: ignore[arg-type]

    out_png = tmp_path / "umap.png"
    render_umap_plot(umap_result, out_png, cfg)

    assert out_png.exists()
    assert out_png.stat().st_size > 1024


def test_coords_parquet_roundtrip(tmp_path: Path, tiny_reference_bank, tiny_harvested_bank) -> None:
    """umap_coords.parquet is written alongside the PNG and round-trips cleanly."""
    merged = _make_merged_df(tiny_reference_bank, tiny_harvested_bank)
    mock = MockSapBert()
    cfg = ComparisonConfig(umap_n_neighbors=2, umap_min_dist=0.1, umap_max_points=1000, plot_dpi=72)
    umap_result = compute_umap_projection(merged, mock, cfg)  # type: ignore[arg-type]

    out_png = tmp_path / "umap.png"
    render_umap_plot(umap_result, out_png, cfg)

    coords_path = tmp_path / "umap_coords.parquet"
    assert coords_path.exists()
    df = pd.read_parquet(coords_path)
    assert "umap_x" in df.columns
    assert "umap_y" in df.columns
    assert len(df) == len(umap_result.labels_concept_id)
