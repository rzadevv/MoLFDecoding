"""CLI sub-app for cross-bank comparison."""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
import typer
from loguru import logger

from molf_interp.comparison.config import ComparisonConfig
from molf_interp.comparison.lexicon_view import LexiconViewConfig, build_primary_lexicon_view
from molf_interp.comparison.matcher import compute_unmatched_nearest_sims, match_banks
from molf_interp.comparison.merger import merge_banks, write_merged_bank
from molf_interp.comparison.report import generate_report
from molf_interp.comparison.visualization import compute_umap_projection, render_umap_plot
from molf_interp.concepts.store import read_concept_bank
from molf_interp.curation.sapbert import SapBert
from molf_interp.io.config import load_config
from molf_interp.utils.logging import setup_logging

compare_app = typer.Typer(no_args_is_help=True, help="Cross-bank comparison.")

_DEFAULT_CONFIG = Path("configs/comparison/default.yaml")
_DEFAULT_LEXICON_CONFIG = Path("configs/comparison/lexicon_view.yaml")


@compare_app.command("run")
def cmd_run(
    config_path: Annotated[
        Path, typer.Option("--config", help="Config YAML path.")
    ] = _DEFAULT_CONFIG,
    skip_umap: Annotated[
        bool, typer.Option("--skip-umap", help="Skip UMAP (faster iteration).")
    ] = False,
    skip_lexicon_view: Annotated[
        bool,
        typer.Option("--skip-lexicon-view", help="Skip View A construction."),
    ] = False,
    lexicon_config_path: Annotated[
        Path,
        typer.Option("--lexicon-config", help="Lexicon view config YAML path."),
    ] = _DEFAULT_LEXICON_CONFIG,
) -> None:
    """Full comparison pipeline: match → merge → lexicon-view → report → UMAP.

    Outputs to data/outputs/comparison/:
      merged_concepts.parquet, merged_prompts.parquet, matches.parquet,
      summary.json, report.md, umap.png + umap_coords.parquet (unless --skip-umap),
      run_metadata.json

    The lexicon-view stage adds in_primary_lexicon column to merged_concepts.parquet.
    Use --skip-lexicon-view to iterate on report/UMAP only.
    """
    setup_logging("INFO")
    t0 = time.monotonic()

    cfg = load_config(config_path, ComparisonConfig)
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading banks …")
    reference_bank = read_concept_bank(cfg.reference_bank_dir)
    harvested_bank = read_concept_bank(cfg.harvested_bank_dir)
    logger.info(
        "Reference bank: {} concepts, harvested bank: {} concepts",
        len(reference_bank),
        len(harvested_bank),
    )

    logger.info("Initialising SapBERT …")
    sapbert = SapBert(cfg.sapbert_config)
    sapbert.load()

    logger.info("Matching banks …")
    cross_threshold: float | None = (
        cfg.cross_tier_match_threshold if cfg.enable_cross_tier_search else None
    )
    matches = match_banks(
        reference_bank,
        harvested_bank,
        sapbert,
        threshold=cfg.match_threshold,
        cross_tier_threshold=cross_threshold,
    )
    logger.info("Total matches: {}", len(matches))

    logger.info("Computing nearest sims for unmatched concepts …")
    unmatched_nearest = compute_unmatched_nearest_sims(
        reference_bank, harvested_bank, matches, sapbert
    )

    logger.info("Merging banks …")
    result = merge_banks(
        reference_bank, harvested_bank, matches, unmatched_nearest=unmatched_nearest
    )
    write_merged_bank(result, out_dir)

    if not skip_lexicon_view:
        logger.info("Building primary lexicon view (View A) …")
        lv_cfg = _load_lexicon_config(lexicon_config_path)
        lv_result = build_primary_lexicon_view(
            result.merged_concepts, result.matches, sapbert, lv_cfg
        )
        # Write merged_concepts back with in_primary_lexicon column
        mc = result.merged_concepts.copy()
        mc["in_primary_lexicon"] = (
            mc["concept_id"].astype(str).map(lv_result.in_primary_lexicon.to_dict()).fillna(False)
        )
        mc.to_parquet(out_dir / "merged_concepts.parquet", index=False)
        # Update summary in-place so generate_report sees view_a section
        result.summary["view_a"] = lv_result.summary
        # Re-write summary.json with view_a (strip unmatched_nearest which is large)
        summary_for_json = {k: v for k, v in result.summary.items() if k != "unmatched_nearest"}
        (out_dir / "summary.json").write_text(json.dumps(summary_for_json, indent=2))
        logger.info(
            "View A: {} / {} concepts",
            lv_result.summary["total"],
            len(result.merged_concepts),
        )

    sapbert.flush()

    logger.info("Generating report …")
    generate_report(result, cfg, out_dir / "report.md")

    if not skip_umap:
        logger.info("Computing UMAP projection …")
        sapbert.load()  # reload if flushed; cache makes this fast
        umap_result = compute_umap_projection(result.merged_concepts, sapbert, cfg)
        sapbert.flush()
        render_umap_plot(umap_result, out_dir / "umap.png", cfg)
        logger.info("UMAP written to {}/umap.png", out_dir)

    elapsed = time.monotonic() - t0
    _write_run_metadata(out_dir, cfg, len(matches), elapsed)
    typer.echo(f"Done in {elapsed:.0f}s. Outputs: {out_dir}")
    recall = result.summary["overall_reference_recall"]
    typer.echo(f"Reference recall: {recall:.1%} ({len(matches)} matches)")


@compare_app.command("build-lexicon-view")
def cmd_build_lexicon_view(
    config_path: Annotated[
        Path, typer.Option("--config", help="Comparison config YAML path.")
    ] = _DEFAULT_CONFIG,
    lexicon_config_path: Annotated[
        Path,
        typer.Option("--lexicon-config", help="Lexicon view config YAML path."),
    ] = _DEFAULT_LEXICON_CONFIG,
) -> None:
    """Build View A and update merged_concepts.parquet with in_primary_lexicon column.

    Requires that `compare run` has been executed first (merged bank must exist).
    Reads the existing merged_concepts.parquet and matches.parquet, computes View A,
    and re-writes merged_concepts.parquet with the new column.
    """
    setup_logging("INFO")

    cfg = load_config(config_path, ComparisonConfig)
    lv_cfg = _load_lexicon_config(lexicon_config_path)
    out_dir = cfg.output_dir

    mc_path = out_dir / "merged_concepts.parquet"
    matches_path = out_dir / "matches.parquet"

    if not mc_path.exists():
        typer.echo(f"No merged_concepts.parquet at {mc_path}. Run `compare run` first.", err=True)
        raise typer.Exit(1)

    logger.info("Loading merged bank …")
    mc = pd.read_parquet(mc_path)
    matches_df = pd.read_parquet(matches_path) if matches_path.exists() else pd.DataFrame()

    logger.info("Initialising SapBERT …")
    sapbert = SapBert(cfg.sapbert_config)
    sapbert.load()

    logger.info("Building lexicon view …")
    lv_result = build_primary_lexicon_view(mc, matches_df, sapbert, lv_cfg)
    sapbert.flush()

    mc["in_primary_lexicon"] = (
        mc["concept_id"].astype(str).map(lv_result.in_primary_lexicon.to_dict()).fillna(False)
    )
    mc.to_parquet(mc_path, index=False)

    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["view_a"] = lv_result.summary
        summary_path.write_text(json.dumps(summary, indent=2))

    typer.echo(f"View A: {lv_result.summary['total']} / {len(mc)} concepts. Updated {mc_path}")


@compare_app.command("inspect-matches")
def cmd_inspect_matches(
    n: Annotated[int, typer.Option("--n", help="Number of matches to show.")] = 10,
    tier: Annotated[str | None, typer.Option("--tier", help="Filter by tier.")] = None,
    min_similarity: Annotated[float | None, typer.Option("--min-sim")] = None,
    max_similarity: Annotated[float | None, typer.Option("--max-sim")] = None,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("data/outputs/comparison"),
) -> None:
    """Print N random matched pairs with optional filters.

    Useful for spot-checking match quality. Example:
      compare inspect-matches --tier H1_morphology --min-sim 0.85 --max-sim 0.90
    """
    matches_path = output_dir / "matches.parquet"
    if not matches_path.exists():
        typer.echo(
            f"No matches.parquet found at {matches_path}. Run `compare run` first.", err=True
        )
        raise typer.Exit(1)

    df = pd.read_parquet(matches_path)
    if tier:
        df = df[df["tier"] == tier]
    if min_similarity is not None:
        df = df[df["similarity"] >= min_similarity]
    if max_similarity is not None:
        df = df[df["similarity"] <= max_similarity]

    sample = df.sample(n=min(n, len(df)), random_state=42)
    for _, row in sample.iterrows():
        ct = " [cross-tier]" if row.get("is_cross_tier") else ""
        typer.echo(
            f"[{row['tier']}{ct}] {float(row['similarity']):.3f}  "
            f"Ref: {row['reference_label']!r} | Harv: {row['harvested_label']!r}"
        )


@compare_app.command("inspect-gaps")
def cmd_inspect_gaps(
    direction: Annotated[
        Literal["reference_only", "harvested_only"],
        typer.Argument(help="Which bank's gaps to show."),
    ],
    n: Annotated[int, typer.Option("--n", help="Number of concepts to show.")] = 10,
    tier: Annotated[str | None, typer.Option("--tier", help="Filter by tier.")] = None,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("data/outputs/comparison"),
) -> None:
    """Print N random concepts present in only one bank.

    Usage:
      compare inspect-gaps reference_only --tier H1_morphology --n 10
      compare inspect-gaps harvested_only --tier H2_gene_program --n 10
    """
    concepts_path = output_dir / "merged_concepts.parquet"
    if not concepts_path.exists():
        typer.echo(
            f"No merged_concepts.parquet at {concepts_path}. Run `compare run` first.", err=True
        )
        raise typer.Exit(1)

    df = pd.read_parquet(concepts_path)
    df = df[df["cross_bank_status"] == direction]
    if tier:
        df = df[df["tier"] == tier]

    sample = df.sample(n=min(n, len(df)), random_state=42)
    for _, row in sample.iterrows():
        typer.echo(f"[{row['tier']}] {row['concept_name']!r}  ({row['concept_id']})")


def _load_lexicon_config(path: Path) -> LexiconViewConfig:
    """Load lexicon view config, returning defaults if file not found."""
    if path.exists():
        return load_config(path, LexiconViewConfig)
    logger.warning("Lexicon config not found at {}; using defaults.", path)
    return LexiconViewConfig()


def _write_run_metadata(
    out_dir: Path,
    cfg: ComparisonConfig,
    n_matches: int,
    elapsed: float,
) -> None:
    meta = {
        "run_id": uuid.uuid4().hex,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "config": {
            "reference_bank_dir": str(cfg.reference_bank_dir),
            "harvested_bank_dir": str(cfg.harvested_bank_dir),
            "match_threshold": cfg.match_threshold,
            "cross_tier_match_threshold": cfg.cross_tier_match_threshold,
        },
        "n_matches": n_matches,
        "elapsed_seconds": round(elapsed, 1),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
