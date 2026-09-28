"""CLI sub-app for VLM operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
import yaml

from molf_interp.concepts.schemas import ConceptPrompt, Species, SpeciesStatus, Tier
from molf_interp.io.cache import compute_cache_key
from molf_interp.io.config import load_config
from molf_interp.io.metadata import RunMetadata, _python_version, collect_git_info, write_metadata
from molf_interp.utils.logging import get_logger, setup_logging
from molf_interp.vlm.base import VLMConfig
from molf_interp.vlm.embedding_store import EmbeddingArtifact, write_embeddings
from molf_interp.vlm.registry import load_vlm
from molf_interp.vlm.view_prep import prepare_view_a

vlm_app = typer.Typer(no_args_is_help=True, help="VLM operations.")


@vlm_app.command()
def info(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to VLM config YAML."),
    ] = Path("configs/vlm/conch_v1.yaml"),
) -> None:
    """Print VLM model info without encoding anything."""
    setup_logging("INFO")
    log = get_logger("cli.vlm.info")

    cfg: VLMConfig = load_config(config, VLMConfig)
    log.info("VLMConfig loaded: model_name={}, device={}", cfg.model_name, cfg.device)

    vlm = load_vlm(cfg)
    log.info(
        "VLM loaded: {} on {} — text emb dim={}",
        vlm.__class__.__name__,
        vlm.device,
        vlm.encode_texts(["test"]).shape[1],
    )


@vlm_app.command("encode-texts")
def encode_texts(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to VLM config YAML."),
    ] = Path("configs/vlm/conch_v1.yaml"),
    texts: Annotated[
        list[str] | None,
        typer.Option("--text", "-t", help="Text strings to encode."),
    ] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", "-b")] = 32,
) -> None:
    """Encode free-form text strings and print their shape."""
    setup_logging("INFO")
    log = get_logger("cli.vlm.encode_texts")

    if not texts:
        log.warning("No texts provided — nothing to encode")
        return

    cfg: VLMConfig = load_config(config, VLMConfig)
    vlm = load_vlm(cfg)
    embeddings = vlm.encode_texts(texts, batch_size=batch_size)
    log.info(
        "Encoded {} texts → shape={}, dtype={}", len(texts), embeddings.shape, embeddings.dtype
    )


@vlm_app.command("concept-embed")
def concept_embed(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to concept embedding config YAML."),
    ] = Path("configs/vlm/concept_embedding.yaml"),
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing output file."),
    ] = False,
) -> None:
    """Precompute CONCH text embeddings for View A concept prompts."""
    setup_logging("INFO")
    log = get_logger("cli.vlm.concept_embed")

    with open(config, encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh)

    output_dir = Path(raw_cfg["output_dir"])
    output_name = raw_cfg["output_name"]
    out_path = output_dir / output_name

    if not force and out_path.exists():
        log.error("Output file already exists: {} (use --force to overwrite)", out_path)
        raise SystemExit(1)

    model_config_path = Path(raw_cfg["model_config"])
    vlm_name = raw_cfg["vlm_name"]
    text_batch_size = raw_cfg.get("text_batch_size", 32)
    ready_prompts_path = raw_cfg.get("ready_prompts_path")
    embedding_set_filter = raw_cfg.get("embedding_set_filter")

    if ready_prompts_path is not None:
        prompts_df = pd.read_parquet(Path(ready_prompts_path))
        log.info("Loaded ready prompts: {}", ready_prompts_path)
        if embedding_set_filter is not None:
            if embedding_set_filter not in {"visual", "transcriptomic"}:
                log.error("Unknown embedding_set_filter: {}", embedding_set_filter)
                raise SystemExit(1)
            prompts_df = prompts_df[prompts_df["embedding_set"] == embedding_set_filter].copy()
            log.info("Applied embedding_set_filter={}", embedding_set_filter)
    else:
        concepts_file = Path(raw_cfg["concepts_file"])
        prompts_file = Path(raw_cfg["prompts_file"])
        exclude_human_only = raw_cfg.get("exclude_human_only_for_mouse", True)

        concepts_df = pd.read_parquet(concepts_file)
        if "in_primary_lexicon" not in concepts_df.columns:
            log.error("concepts_file is missing 'in_primary_lexicon' column: {}", concepts_file)
            raise SystemExit(1)

        view_a_ids = set(concepts_df.loc[concepts_df["in_primary_lexicon"], "concept_id"].tolist())
        log.info("View A concept count: {}", len(view_a_ids))

        prompts_file = Path(raw_cfg["prompts_file"])
        prompts_df = pd.read_parquet(prompts_file)
        prompts_df = prompts_df[prompts_df["concept_id"].isin(view_a_ids)]

        if exclude_human_only:
            prompts_df = prompts_df[
                ~(
                    (prompts_df["species"] == "Mus_musculus")
                    & (prompts_df["species_status"] == "human_only")
                )
            ]

    log.info("Prompts to embed: {}", len(prompts_df))
    log.info("Species breakdown: {}", prompts_df["species"].value_counts().to_dict())
    if "embedding_set" in prompts_df.columns:
        log.info(
            "Embedding set breakdown: {}",
            prompts_df["embedding_set"].value_counts().to_dict(),
        )

    # Build ConceptPrompt objects
    prompts: list[ConceptPrompt] = []
    for _, row in prompts_df.iterrows():
        prompts.append(
            ConceptPrompt(
                concept_id=row["concept_id"],
                concept_name=row["concept_name"],
                tier=Tier(row["tier"]),
                species=Species(row["species"]),
                prompt_text=row["prompt_text"],
                species_status=SpeciesStatus(row["species_status"]),
            )
        )

    # Load VLM
    cfg: VLMConfig = load_config(model_config_path, VLMConfig)
    if cfg.model_name != vlm_name:
        log.error("model_config model_name {} != expected {}", cfg.model_name, vlm_name)
        raise SystemExit(1)

    vlm = load_vlm(cfg)

    # Embed in batches — collect artifacts as we go to report progress
    artifacts: list[EmbeddingArtifact] = []
    total = len(prompts)
    for i in range(0, total, text_batch_size):
        batch = prompts[i : i + text_batch_size]
        texts = [p.prompt_text for p in batch]
        emb_batch = vlm.encode_texts(texts, batch_size=text_batch_size)

        for j, prompt in enumerate(batch):
            artifacts.append(
                EmbeddingArtifact(
                    concept_id=prompt.concept_id,
                    species=prompt.species.value,
                    tier=prompt.tier.value,
                    view="A",
                    prompt_text=prompt.prompt_text,
                    embedding=emb_batch[j],
                )
            )

        log.info("Embedded {}/{} prompts", min(i + text_batch_size, total), total)

    write_embeddings(artifacts, out_path)
    log.info("Wrote {} embeddings to {}", len(artifacts), out_path)

    sha, dirty = collect_git_info()
    run_id = uuid.uuid4().hex
    meta = RunMetadata(
        run_id=run_id,
        git_sha=sha,
        git_dirty=dirty,
        config_hash=compute_cache_key(raw_cfg),
        started_at=datetime.now(tz=UTC),
        command="vlm.concept_embed",
        python_version=_python_version(),
    )
    write_metadata(meta, output_dir / f"run_metadata_{run_id}.json")
    log.info("Run metadata: run_id={}", run_id)


@vlm_app.command("prepare-view-a")
def cmd_prepare_view_a(
    concepts: Annotated[
        Path,
        typer.Option(
            "--concepts",
            help="Path to merged concepts parquet.",
        ),
    ] = Path("data/outputs/comparison/merged_concepts.parquet"),
    prompts: Annotated[
        Path,
        typer.Option(
            "--prompts",
            help="Path to merged prompts parquet.",
        ),
    ] = Path("data/outputs/comparison/merged_prompts.parquet"),
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory for cleaned View A embedding inputs.",
        ),
    ] = Path("data/cache/concept_embeddings"),
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Compute and log cleanup counts without writing files.",
        ),
    ] = False,
) -> None:
    """Prepare View A concepts and prompts for VLM text embedding."""
    setup_logging("INFO")
    log = get_logger("cli.vlm.prepare_view_a")

    output_concepts = output_dir / "merged_concepts_cleaned.parquet"
    output_prompts = output_dir / "merged_prompts_cleaned.parquet"
    ready_prompts = output_dir / "view_a_ready.parquet"
    dedup_map = output_dir / "view_a_deduplication_map.parquet"

    result = prepare_view_a(
        concepts_path=concepts,
        prompts_path=prompts,
        output_concepts_path=output_concepts,
        output_prompts_path=output_prompts,
        ready_prompts_path=ready_prompts,
        dedup_map_path=dedup_map,
        dry_run=dry_run,
    )

    log.info("Concepts before: {}", result.n_concepts_before)
    log.info("Clinical drops: {}", result.n_clinical_dropped)
    log.info("Dedup drops: {}", result.n_dedup_dropped)
    log.info("Concepts after: {}", result.n_concepts_after)
    log.info("Visual concepts: {}", result.n_visual_concepts)
    log.info("Transcriptomic concepts: {}", result.n_transcriptomic_concepts)
    log.info("Prompts before: {}", result.n_prompts_before)
    log.info("Ready prompts after: {}", result.n_prompts_after)
    log.info(
        "Human-only prompt rows before ready filtering: {}",
        result.n_human_only_prompt_rows_before,
    )
    log.info("Human-only mouse rows excluded: {}", result.n_human_only_mouse_rows_excluded)

    if dry_run:
        log.info("Dry run complete; no files written")
        return

    log.info("Cleaned concepts: {}", output_concepts)
    log.info("Cleaned prompts: {}", output_prompts)
    log.info("Ready prompts: {}", ready_prompts)
    log.info("Deduplication map: {}", dedup_map)
