"""UMAP projection and matplotlib scatter plot for concept-space visualization."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from molf_interp.comparison.config import ComparisonConfig
from molf_interp.curation.sapbert import SapBert

_TIER_COLORS: dict[str, str] = {
    "H1_morphology": "#1f77b4",
    "H2_cell_type": "#ff7f0e",
    "H2_gene_program": "#2ca02c",
    "H2_pathway": "#d62728",
    "H2_niche": "#9467bd",
}
_DEFAULT_COLOR = "#7f7f7f"

_PROVENANCE_MARKERS: dict[str, str] = {
    "reference": "o",
    "harvested": "x",
    "matched": "*",
}


@dataclass(frozen=True)
class UmapResult:
    """Output of UMAP projection."""

    coords: npt.NDArray[Any]
    labels_concept_id: tuple[str, ...]
    labels_tier: tuple[str, ...]
    labels_provenance: tuple[str, ...]
    labels_cross_bank_status: tuple[str, ...]


def compute_umap_projection(
    merged_concepts: pd.DataFrame,
    sapbert: SapBert,
    config: ComparisonConfig,
) -> UmapResult:
    """Embed concept labels via SapBERT, then project to 2D via UMAP.

    If len(merged_concepts) > config.umap_max_points, randomly down-samples
    stratified by tier so each tier is represented.

    Args:
        merged_concepts: DataFrame with concept_name, tier, provenance, cross_bank_status columns.
        sapbert: Loaded SapBERT model.
        config: ComparisonConfig with UMAP hyperparameters.

    Returns:
        UmapResult with 2D coordinates aligned to the (possibly down-sampled) rows.
    """
    import umap as umap_lib

    df = merged_concepts.copy()

    if len(df) > config.umap_max_points:
        rng = np.random.RandomState(config.umap_random_state)
        tiers = df["tier"].unique().tolist()
        n_per_tier = max(1, config.umap_max_points // len(tiers))
        sampled_parts = []
        for tier in tiers:
            tier_df = df[df["tier"] == tier]
            n = min(n_per_tier, len(tier_df))
            sampled_parts.append(tier_df.sample(n=n, random_state=int(rng.randint(0, 2**31))))
        df = pd.concat(sampled_parts, ignore_index=True)
        # Trim to exact cap if slightly over
        if len(df) > config.umap_max_points:
            df = df.sample(n=config.umap_max_points, random_state=config.umap_random_state)

    labels = df["concept_name"].tolist()
    embeddings = sapbert.encode(labels)

    reducer = umap_lib.UMAP(
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        random_state=config.umap_random_state,
        n_components=2,
        metric="cosine",
    )
    coords: npt.NDArray[Any] = reducer.fit_transform(embeddings).astype(np.float32)

    return UmapResult(
        coords=coords,
        labels_concept_id=tuple(df["concept_id"].tolist()),
        labels_tier=tuple(df["tier"].tolist()),
        labels_provenance=tuple(df["provenance"].tolist()),
        labels_cross_bank_status=tuple(df["cross_bank_status"].tolist()),
    )


def render_umap_plot(
    umap_result: UmapResult,
    output_path: Path,
    config: ComparisonConfig,
) -> None:
    """Render the UMAP scatter plot as PNG.

    Color by tier, marker by provenance (reference=circle, harvested=x, matched=star).
    Also saves umap_coords.parquet next to the PNG.

    Args:
        umap_result: UmapResult from compute_umap_projection().
        output_path: Destination PNG path.
        config: ComparisonConfig for plot styling.
    """
    figsize: tuple[float, float] = config.plot_figsize
    fig, ax = plt.subplots(figsize=figsize, dpi=config.plot_dpi)
    ax.set_facecolor("#f8f8f8")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7, color="white")

    coords = umap_result.coords
    tiers = umap_result.labels_tier
    cross_statuses = umap_result.labels_cross_bank_status

    all_tiers = sorted(set(tiers))
    tier_handles = []
    for tier in all_tiers:
        color = _TIER_COLORS.get(tier, _DEFAULT_COLOR)
        idx = [i for i, t in enumerate(tiers) if t == tier]
        # Within tier, split by cross_bank_status for marker
        for status, marker, size, alpha in [
            ("matched", "*", 60, 0.9),
            ("reference_only", "o", 18, 0.7),
            ("harvested_only", "x", 18, 0.5),
        ]:
            pts = [i for i in idx if cross_statuses[i] == status]
            if not pts:
                continue
            ax.scatter(
                coords[pts, 0],
                coords[pts, 1],
                c=color,
                marker=marker,
                s=size,
                alpha=alpha,
                linewidths=0.5,
                label=f"{tier} ({status})",
            )
        # Add one tier entry for the legend color block
        tier_handles.append(
            mlines.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=8,
                label=tier,
            )
        )

    # Marker legend entries
    marker_handles = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#555",
            markersize=8,
            label="Reference only",
        ),
        mlines.Line2D([0], [0], marker="x", color="#555", markersize=8, label="Harvested only"),
        mlines.Line2D(
            [0], [0], marker="*", color="w", markerfacecolor="#555", markersize=10, label="Matched"
        ),
    ]

    leg1 = ax.legend(
        handles=tier_handles, title="Tier", loc="upper left", fontsize=7, title_fontsize=8
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=marker_handles, title="Provenance", loc="upper right", fontsize=7, title_fontsize=8
    )

    ax.set_title("Cross-bank concept embedding space (SapBERT → UMAP 2D)", fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot_dpi)
    plt.close(fig)

    # Save coords as parquet for re-plotting
    coords_df = pd.DataFrame(
        {
            "concept_id": list(umap_result.labels_concept_id),
            "tier": list(umap_result.labels_tier),
            "provenance": list(umap_result.labels_provenance),
            "cross_bank_status": list(umap_result.labels_cross_bank_status),
            "umap_x": coords[:, 0].tolist(),
            "umap_y": coords[:, 1].tolist(),
        }
    )
    coords_path = output_path.parent / "umap_coords.parquet"
    coords_df.to_parquet(coords_path, index=False)
