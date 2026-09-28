"""CLI sub-app for the curation pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from loguru import logger

from molf_interp.curation.config import CurationConfig, LLMAdjudicationConfig
from molf_interp.curation.pipeline import (
    _load_stage,
    run_curation_pipeline,
)
from molf_interp.io.config import load_config

curate_app = typer.Typer(name="curate", help="Curation pipeline commands.")

_DEFAULT_CONFIG = Path("configs/curation/default.yaml")


def _get_config(config_path: Path, skip_llm: bool = False) -> CurationConfig:
    cfg = load_config(config_path, CurationConfig)
    if skip_llm:
        cfg = cfg.model_copy(update={"llm": LLMAdjudicationConfig(enabled=False)})
    return cfg


@curate_app.command("run")
def cmd_run(
    config: Annotated[Path, typer.Option("--config", help="Config file path.")] = _DEFAULT_CONFIG,
    resume_from: Annotated[
        int | None,
        typer.Option("--resume-from", help="Resume from stage N (1-7)."),
    ] = None,
    skip_llm: Annotated[
        bool,
        typer.Option("--skip-llm", help="Override: disable LLM stage."),
    ] = False,
) -> None:
    """Run the full curation pipeline."""
    cfg = _get_config(config, skip_llm)
    bank, output_dir = run_curation_pipeline(cfg, resume_from=resume_from)
    summary = bank.summary()
    logger.info("Summary: {}", summary)
    typer.echo(f"Done. {len(bank)} concepts -> {output_dir}")


@curate_app.command("stage")
def cmd_stage(
    stage: Annotated[int, typer.Argument(help="Stage number 1-7.")],
    config: Annotated[Path, typer.Option("--config", help="Config file path.")] = _DEFAULT_CONFIG,
) -> None:
    """Run exactly one stage, loading the previous stage's cache as input."""
    if stage < 1 or stage > 7:
        typer.echo("Stage must be 1-7", err=True)
        raise typer.Exit(1)
    cfg = _get_config(config)
    run_curation_pipeline(cfg, resume_from=stage)


@curate_app.command("inspect")
def cmd_inspect(
    stage: Annotated[int, typer.Argument(help="Stage number 1-7.")],
    config: Annotated[Path, typer.Option("--config", help="Config file path.")] = _DEFAULT_CONFIG,
    n: Annotated[int, typer.Option("--n", help="Sample size.")] = 20,
) -> None:
    """Print N random rows from a stage's output parquet for spot-checking."""
    cfg = _get_config(config)
    try:
        df = _load_stage(cfg, stage)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    sample = df.sample(min(n, len(df)), random_state=42)
    cols = ["preferred_label", "candidate_tier", "source_name"]
    extra_cols = [
        "tier",
        "organ",
        "level",
        "concept_id",
        "reference_tier_similarity",
        "su_similarity",
        "visual_filter_decision",
    ]
    show_cols = [c for c in cols + extra_cols if c in sample.columns]
    typer.echo(sample[show_cols].to_string(index=False))


@curate_app.command("calibrate")
def cmd_calibrate(
    config: Annotated[Path, typer.Option("--config", help="Config file path.")] = _DEFAULT_CONFIG,
    output: Annotated[
        Path,
        typer.Option("--output", help="Output YAML path for calibrated thresholds."),
    ] = Path("configs/curation/calibrated_thresholds.yaml"),
) -> None:
    """Calibrate per-tier thresholds from the reference bank and negative-class centroids.

    Requires pre-computed negative-class centroids (run with --build-neg-centroids
    if they don't exist yet, or ensure BioPortal access is configured).
    """
    import asyncio

    from molf_interp.calibration_runner import run_calibration

    cfg = _get_config(config)
    thresholds = asyncio.run(run_calibration(cfg))
    from molf_interp.curation.calibration import save_calibrated_thresholds

    save_calibrated_thresholds(thresholds, output)
    typer.echo(f"Calibrated thresholds written to {output}")
    for tier in ("H1_morphology", "H2_cell_type", "H2_niche", "H2_gene_program", "H2_pathway"):
        t = thresholds.for_tier(tier)
        typer.echo(
            f"  {tier}: keep={t.keep_threshold:.4f}, drop={t.drop_threshold:.4f}, "
            f"knn_agree={t.knn_tier_agreement_threshold}"
        )


@curate_app.command("spotcheck")
def cmd_spotcheck(
    n_per_tier: Annotated[int, typer.Option("--n", help="Random samples per tier.")] = 10,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 42,
    bank_dir: Annotated[
        Path,
        typer.Option("--bank", help="ConceptBank directory."),
    ] = Path("data/cache/concept_bank/harvested"),
) -> None:
    """Print N random concepts per tier for manual quality review."""
    from molf_interp.concepts.store import read_concept_bank

    try:
        bank = read_concept_bank(bank_dir)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    import random

    rng = random.Random(seed)
    for tier_val in ("H1_morphology", "H2_cell_type", "H2_niche", "H2_gene_program", "H2_pathway"):
        from molf_interp.concepts.schemas import Tier

        tier_enum = Tier(tier_val)
        concepts = list(bank.filter_concepts(tier=tier_enum))
        sample = rng.sample(concepts, min(n_per_tier, len(concepts)))
        typer.echo(f"\n=== {tier_val} ({len(concepts)} total) ===")
        for c in sample:
            extra = ""
            if c.organ:
                extra += f"  organ={c.organ}"
            if c.level:
                extra += f"  level={c.level}"
            if c.category:
                extra += f"  category={c.category}"
            typer.echo(f"  [{c.concept_id}] {c.concept_name}{extra}")


@curate_app.command("diff")
def cmd_diff(
    bank_a: Annotated[Path, typer.Option("--a", help="Previous bank directory.")],
    bank_b: Annotated[Path, typer.Option("--b", help="Current bank directory.")],
) -> None:
    """Compare two curation runs and report concept-level differences."""
    from molf_interp.concepts.schemas import Tier
    from molf_interp.concepts.store import read_concept_bank

    try:
        a = read_concept_bank(bank_a)
        b = read_concept_bank(bank_b)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo("=== Concept counts by tier ===")
    for tier_val in ("H1_morphology", "H2_cell_type", "H2_niche", "H2_gene_program", "H2_pathway"):
        tier = Tier(tier_val)
        n_a = len(a.filter_concepts(tier=tier))
        n_b = len(b.filter_concepts(tier=tier))
        delta = n_b - n_a
        sign = "+" if delta >= 0 else ""
        typer.echo(f"  {tier_val}: A={n_a}  B={n_b}  ({sign}{delta})")

    ids_a = {c.concept_id: c for c in a.concepts}
    ids_b = {c.concept_id: c for c in b.concepts}
    names_a = {c.concept_name.lower(): c.concept_id for c in a.concepts}
    names_b = {c.concept_name.lower(): c.concept_id for c in b.concepts}

    # Lost (in A but not B by name)
    lost_names = set(names_a) - set(names_b)
    typer.echo(f"\n=== Lost in B ({len(lost_names)}) ===")
    for name in sorted(lost_names)[:20]:
        cid = names_a[name]
        c = ids_a.get(cid)
        tier_s = c.tier.value if c else "?"
        typer.echo(f"  [{cid}] {name}  ({tier_s})")
    if len(lost_names) > 20:
        typer.echo(f"  ... and {len(lost_names) - 20} more")

    # Gained (in B but not A by name)
    gained_names = set(names_b) - set(names_a)
    typer.echo(f"\n=== Gained in B ({len(gained_names)}) ===")
    for name in sorted(gained_names)[:20]:
        cid = names_b[name]
        c = ids_b.get(cid)
        tier_s = c.tier.value if c else "?"
        typer.echo(f"  [{cid}] {name}  ({tier_s})")
    if len(gained_names) > 20:
        typer.echo(f"  ... and {len(gained_names) - 20} more")

    # Changed tier/organ/level for shared names
    changed = []
    for name in set(names_a) & set(names_b):
        ca = ids_a.get(names_a[name])
        cb = ids_b.get(names_b[name])
        if ca is None or cb is None:
            continue
        diffs = []
        if ca.tier != cb.tier:
            diffs.append(f"tier: {ca.tier.value}→{cb.tier.value}")
        if ca.organ != cb.organ:
            diffs.append(f"organ: {ca.organ}→{cb.organ}")
        if ca.level != cb.level:
            diffs.append(f"level: {ca.level}→{cb.level}")
        if diffs:
            changed.append((name, ", ".join(diffs)))

    typer.echo(f"\n=== Changed in B ({len(changed)}) ===")
    for name, diff_str in sorted(changed)[:20]:
        typer.echo(f"  {name}: {diff_str}")
    if len(changed) > 20:
        typer.echo(f"  ... and {len(changed) - 20} more")


@curate_app.command("drop-tier")
def cmd_drop_tier(
    tier: Annotated[str, typer.Argument(help="Tier to remove entirely (e.g. H2_niche).")],
    bank_dir: Annotated[
        Path,
        typer.Option("--bank", help="ConceptBank directory."),
    ] = Path("data/cache/concept_bank/harvested"),
    confirm: Annotated[
        bool,
        typer.Option("--confirm", help="Required to actually perform the drop."),
    ] = False,
) -> None:
    """Remove all concepts of a given tier from a bank.

    Drops matching rows from BOTH concepts.parquet and prompts.parquet and writes
    a run_metadata.json recording the operation, row counts before/after, and the
    affected concept_ids.

    Without --confirm: dry-run mode, reports counts only.
    With --confirm: performs the drop and writes updated parquets.
    """
    concepts_path = bank_dir / "concepts.parquet"
    prompts_path = bank_dir / "prompts.parquet"

    if not concepts_path.exists():
        typer.echo(f"Error: concepts.parquet not found in {bank_dir}", err=True)
        raise typer.Exit(1)

    concepts_df = pd.read_parquet(concepts_path)
    prompts_df = pd.read_parquet(prompts_path) if prompts_path.exists() else pd.DataFrame()

    before_concepts = len(concepts_df)
    before_prompts = len(prompts_df)

    drop_mask = concepts_df["tier"] == tier
    n_drop = int(drop_mask.sum())

    if n_drop == 0:
        logger.info("drop-tier: tier '{}' not present in bank — no-op.", tier)
        typer.echo(f"Tier '{tier}' not found in bank. Nothing to drop.")
        return

    affected_ids = concepts_df.loc[drop_mask, "concept_id"].tolist()

    typer.echo(f"Tier '{tier}': {n_drop} concept(s) found → {before_concepts - n_drop} remain.")
    if not prompts_df.empty and "concept_id" in prompts_df.columns:
        prompt_drop = int(prompts_df["concept_id"].isin(affected_ids).sum())
        typer.echo(f"  Prompts: {prompt_drop} of {before_prompts} will be dropped.")
    else:
        prompt_drop = 0

    if not confirm:
        typer.echo("Dry-run (pass --confirm to apply).")
        return

    concepts_out = concepts_df[~drop_mask].copy()
    concepts_out.to_parquet(concepts_path, index=False)

    if not prompts_df.empty and "concept_id" in prompts_df.columns:
        prompts_out = prompts_df[~prompts_df["concept_id"].isin(affected_ids)].copy()
        prompts_out.to_parquet(prompts_path, index=False)
    else:
        prompts_out = pd.DataFrame()

    metadata = {
        "operation": "drop-tier",
        "tier": tier,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "concepts_before": before_concepts,
        "concepts_after": len(concepts_out),
        "prompts_before": before_prompts,
        "prompts_after": len(prompts_out) if not prompts_out.empty else before_prompts,
        "affected_concept_ids": affected_ids,
    }
    meta_path = bank_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    logger.info(
        "drop-tier '{}': removed {} concepts, {} prompts. Bank written to {}.",
        tier,
        n_drop,
        prompt_drop,
        bank_dir,
    )
    typer.echo(f"Done. Removed {n_drop} concept(s) and {prompt_drop} prompt(s).")
