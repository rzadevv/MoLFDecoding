"""Harvest CLI sub-app — one command per source plus a combined 'all' command."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from molf_interp.io.cache import compute_cache_key
from molf_interp.io.config import BaseConfig, load_config
from molf_interp.io.metadata import RunMetadata, _python_version, collect_git_info, write_metadata
from molf_interp.sources.bioportal.client import BioPortalClient
from molf_interp.sources.bioportal.config import BioPortalConfig
from molf_interp.sources.harvesters.cell_ontology import (
    CellOntologyHarvestConfig,
    harvest_cell_ontology,
)
from molf_interp.sources.harvesters.gene_ontology import (
    GeneOntologyHarvestConfig,
    harvest_gene_ontology,
)
from molf_interp.sources.harvesters.msigdb import MSigDBHarvestConfig, harvest_msigdb
from molf_interp.sources.harvesters.ncit import NCItHarvestConfig, harvest_ncit
from molf_interp.sources.harvesters.pubmed import PubMedHarvestConfig, harvest_pubmed
from molf_interp.sources.harvesters.snomed import SnomedHarvestConfig, harvest_snomed
from molf_interp.sources.harvesters.uberon import UberonHarvestConfig, harvest_uberon
from molf_interp.sources.ncbi.client import NCBIClient
from molf_interp.sources.ncbi.config import NCBIConfig
from molf_interp.sources.raw_concept import merge_raw_concept_files
from molf_interp.utils.logging import get_logger, setup_logging

harvest_app = typer.Typer(no_args_is_help=True, help="Concept harvesting from external sources.")

_DEFAULT_BP_CONFIG = Path("configs/sources/bioportal.yaml")
_DEFAULT_NCBI_CONFIG = Path("configs/sources/ncbi.yaml")


class _SourceEntry(BaseConfig):
    """Config entry for one source in the 'all' config."""

    enabled: bool = True
    config_path: Path


class HarvestAllConfig(BaseConfig):
    """Config schema for configs/harvest/all.yaml."""

    ncit: _SourceEntry
    cell_ontology: _SourceEntry
    gene_ontology: _SourceEntry
    uberon: _SourceEntry
    snomed: _SourceEntry
    msigdb: _SourceEntry
    pubmed: _SourceEntry
    output_merged_path: Path = Path("data/cache/harvest/all_raw_concepts.parquet")


def _write_run_meta(command: str, config_hash: str) -> None:
    sha, dirty = collect_git_info()
    run_id = uuid.uuid4().hex
    meta = RunMetadata(
        run_id=run_id,
        git_sha=sha,
        git_dirty=dirty,
        config_hash=config_hash,
        started_at=datetime.now(tz=UTC),
        command=command,
        python_version=_python_version(),
    )
    out = Path("data/outputs") / f"run_metadata_{run_id}.json"
    write_metadata(meta, out)


@harvest_app.command("ncit")
def cmd_ncit(
    config: Annotated[
        Path, typer.Option("--config", help="Path to NCItHarvestConfig YAML.")
    ] = Path("configs/harvest/ncit.yaml"),
    bioportal_config: Annotated[
        Path, typer.Option("--bioportal-config", help="Path to BioPortalConfig YAML.")
    ] = _DEFAULT_BP_CONFIG,
) -> None:
    """Harvest NCIt morphology subtrees. Writes ncit.parquet."""
    setup_logging("INFO")
    log = get_logger("cli.harvest.ncit")
    cfg = load_config(config, NCItHarvestConfig)
    bp_cfg = load_config(bioportal_config, BioPortalConfig)

    async def _run() -> int:
        async with BioPortalClient.from_config(bp_cfg) as client:
            return await harvest_ncit(cfg, client)

    count = asyncio.run(_run())
    log.info("NCIt: {} records written to {}", count, cfg.output_path)
    _write_run_meta("harvest-ncit", compute_cache_key(cfg.model_dump()))


@harvest_app.command("cell-ontology")
def cmd_cell_ontology(
    config: Annotated[
        Path, typer.Option("--config", help="Path to CellOntologyHarvestConfig YAML.")
    ] = Path("configs/harvest/cell_ontology.yaml"),
    bioportal_config: Annotated[
        Path, typer.Option("--bioportal-config", help="Path to BioPortalConfig YAML.")
    ] = _DEFAULT_BP_CONFIG,
) -> None:
    """Harvest Cell Ontology cell-type subtree. Writes cell_ontology.parquet."""
    setup_logging("INFO")
    log = get_logger("cli.harvest.cell_ontology")
    cfg = load_config(config, CellOntologyHarvestConfig)
    bp_cfg = load_config(bioportal_config, BioPortalConfig)

    async def _run() -> int:
        async with BioPortalClient.from_config(bp_cfg) as client:
            return await harvest_cell_ontology(cfg, client)

    count = asyncio.run(_run())
    log.info("Cell Ontology: {} records written to {}", count, cfg.output_path)
    _write_run_meta("harvest-cell-ontology", compute_cache_key(cfg.model_dump()))


@harvest_app.command("gene-ontology")
def cmd_gene_ontology(
    config: Annotated[
        Path, typer.Option("--config", help="Path to GeneOntologyHarvestConfig YAML.")
    ] = Path("configs/harvest/gene_ontology.yaml"),
    bioportal_config: Annotated[
        Path, typer.Option("--bioportal-config", help="Path to BioPortalConfig YAML.")
    ] = _DEFAULT_BP_CONFIG,
) -> None:
    """Harvest Gene Ontology biological-process subtrees. Writes gene_ontology.parquet."""
    setup_logging("INFO")
    log = get_logger("cli.harvest.gene_ontology")
    cfg = load_config(config, GeneOntologyHarvestConfig)
    bp_cfg = load_config(bioportal_config, BioPortalConfig)

    async def _run() -> int:
        async with BioPortalClient.from_config(bp_cfg) as client:
            return await harvest_gene_ontology(cfg, client)

    count = asyncio.run(_run())
    log.info("Gene Ontology: {} records written to {}", count, cfg.output_path)
    _write_run_meta("harvest-gene-ontology", compute_cache_key(cfg.model_dump()))


@harvest_app.command("uberon")
def cmd_uberon(
    config: Annotated[
        Path, typer.Option("--config", help="Path to UberonHarvestConfig YAML.")
    ] = Path("configs/harvest/uberon.yaml"),
    bioportal_config: Annotated[
        Path, typer.Option("--bioportal-config", help="Path to BioPortalConfig YAML.")
    ] = _DEFAULT_BP_CONFIG,
) -> None:
    """Harvest UBERON organ-level anatomy side table. Writes uberon_organs.parquet."""
    setup_logging("INFO")
    log = get_logger("cli.harvest.uberon")
    cfg = load_config(config, UberonHarvestConfig)
    bp_cfg = load_config(bioportal_config, BioPortalConfig)

    async def _run() -> int:
        async with BioPortalClient.from_config(bp_cfg) as client:
            return await harvest_uberon(cfg, client)

    count = asyncio.run(_run())
    log.info("UBERON: {} records written to {}", count, cfg.output_path)
    _write_run_meta("harvest-uberon", compute_cache_key(cfg.model_dump()))


@harvest_app.command("snomed")
def cmd_snomed(
    config: Annotated[
        Path, typer.Option("--config", help="Path to SnomedHarvestConfig YAML.")
    ] = Path("configs/harvest/snomed.yaml"),
    bioportal_config: Annotated[
        Path, typer.Option("--bioportal-config", help="Path to BioPortalConfig YAML.")
    ] = _DEFAULT_BP_CONFIG,
) -> None:
    """Harvest SNOMED CT morphology. Gracefully skips if not licensed."""
    setup_logging("INFO")
    log = get_logger("cli.harvest.snomed")
    cfg = load_config(config, SnomedHarvestConfig)
    bp_cfg = load_config(bioportal_config, BioPortalConfig)

    async def _run() -> int:
        async with BioPortalClient.from_config(bp_cfg) as client:
            return await harvest_snomed(cfg, client)

    count = asyncio.run(_run())
    log.info("SNOMED: {} records written to {}", count, cfg.output_path)
    _write_run_meta("harvest-snomed", compute_cache_key(cfg.model_dump()))


@harvest_app.command("msigdb")
def cmd_msigdb(
    config: Annotated[
        Path, typer.Option("--config", help="Path to MSigDBHarvestConfig YAML.")
    ] = Path("configs/harvest/msigdb.yaml"),
) -> None:
    """Download and parse MSigDB Hallmark, Reactome, KEGG collections."""
    setup_logging("INFO")
    log = get_logger("cli.harvest.msigdb")
    cfg = load_config(config, MSigDBHarvestConfig)
    count = harvest_msigdb(cfg)
    log.info("MSigDB: {} gene sets written to {}", count, cfg.output_path)
    _write_run_meta("harvest-msigdb", compute_cache_key(cfg.model_dump()))


@harvest_app.command("pubmed")
def cmd_pubmed(
    config: Annotated[
        Path, typer.Option("--config", help="Path to PubMedHarvestConfig YAML.")
    ] = Path("configs/harvest/pubmed.yaml"),
    ncbi_config: Annotated[
        Path, typer.Option("--ncbi-config", help="Path to NCBIConfig YAML.")
    ] = _DEFAULT_NCBI_CONFIG,
) -> None:
    """Fetch PubMed abstracts and extract entities with scispaCy NER."""
    setup_logging("INFO")
    log = get_logger("cli.harvest.pubmed")
    cfg = load_config(config, PubMedHarvestConfig)
    ncbi_cfg = load_config(ncbi_config, NCBIConfig)

    async def _run() -> int:
        async with NCBIClient.from_config(ncbi_cfg) as client:
            return await harvest_pubmed(cfg, client)

    count = asyncio.run(_run())
    log.info("PubMed: {} RawConcepts written to {}", count, cfg.output_path)
    _write_run_meta("harvest-pubmed", compute_cache_key(cfg.model_dump()))


@harvest_app.command("all")
def cmd_all(
    config: Annotated[Path, typer.Option("--config", help="Path to HarvestAllConfig YAML.")] = Path(
        "configs/harvest/all.yaml"
    ),
    bioportal_config: Annotated[
        Path,
        typer.Option("--bioportal-config", help="Path to BioPortalConfig YAML."),
    ] = _DEFAULT_BP_CONFIG,
    ncbi_config: Annotated[
        Path,
        typer.Option("--ncbi-config", help="Path to NCBIConfig YAML."),
    ] = _DEFAULT_NCBI_CONFIG,
    skip_on_error: Annotated[
        bool,
        typer.Option(
            help="Continue if one source fails. SNOMED license errors are always skipped."
        ),
    ] = True,
) -> None:
    """Run all enabled harvesters in sequence. Writes a merged all_raw_concepts.parquet."""
    setup_logging("INFO")
    log = get_logger("cli.harvest.all")
    cfg = load_config(config, HarvestAllConfig)
    bp_cfg = load_config(bioportal_config, BioPortalConfig)
    ncbi_cfg = load_config(ncbi_config, NCBIConfig)

    output_paths: list[Path] = []

    async def _run_bioportal() -> None:
        async with BioPortalClient.from_config(bp_cfg) as client:
            if cfg.ncit.enabled:
                try:
                    ncit_cfg = load_config(cfg.ncit.config_path, NCItHarvestConfig)
                    count = await harvest_ncit(ncit_cfg, client)
                    log.info("ncit: {} records", count)
                    output_paths.append(ncit_cfg.output_path)
                except Exception as exc:
                    if skip_on_error:
                        log.error("ncit harvester failed: {!r}", exc)
                    else:
                        raise
            else:
                log.info("Skipping ncit (disabled in config)")

            if cfg.cell_ontology.enabled:
                try:
                    cl_cfg = load_config(cfg.cell_ontology.config_path, CellOntologyHarvestConfig)
                    count = await harvest_cell_ontology(cl_cfg, client)
                    log.info("cell_ontology: {} records", count)
                    output_paths.append(cl_cfg.output_path)
                except Exception as exc:
                    if skip_on_error:
                        log.error("cell_ontology harvester failed: {!r}", exc)
                    else:
                        raise
            else:
                log.info("Skipping cell_ontology (disabled in config)")

            if cfg.gene_ontology.enabled:
                try:
                    go_cfg = load_config(cfg.gene_ontology.config_path, GeneOntologyHarvestConfig)
                    count = await harvest_gene_ontology(go_cfg, client)
                    log.info("gene_ontology: {} records", count)
                    output_paths.append(go_cfg.output_path)
                except Exception as exc:
                    if skip_on_error:
                        log.error("gene_ontology harvester failed: {!r}", exc)
                    else:
                        raise
            else:
                log.info("Skipping gene_ontology (disabled in config)")

            if cfg.uberon.enabled:
                try:
                    ub_cfg = load_config(cfg.uberon.config_path, UberonHarvestConfig)
                    count = await harvest_uberon(ub_cfg, client)
                    log.info("uberon: {} records", count)
                    output_paths.append(ub_cfg.output_path)
                except Exception as exc:
                    if skip_on_error:
                        log.error("uberon harvester failed: {!r}", exc)
                    else:
                        raise
            else:
                log.info("Skipping uberon (disabled in config)")

            if cfg.snomed.enabled:
                try:
                    sn_cfg = load_config(cfg.snomed.config_path, SnomedHarvestConfig)
                    count = await harvest_snomed(sn_cfg, client)
                    log.info("snomed: {} records", count)
                    output_paths.append(sn_cfg.output_path)
                except Exception as exc:
                    if skip_on_error:
                        log.error("snomed harvester failed: {!r}", exc)
                    else:
                        raise
            else:
                log.info("Skipping snomed (disabled in config)")

    asyncio.run(_run_bioportal())

    if cfg.msigdb.enabled:
        try:
            msigdb_cfg = load_config(cfg.msigdb.config_path, MSigDBHarvestConfig)
            count = harvest_msigdb(msigdb_cfg)
            log.info("msigdb: {} records", count)
            output_paths.append(msigdb_cfg.output_path)
        except Exception as exc:
            if skip_on_error:
                log.error("msigdb harvester failed: {!r}", exc)
            else:
                raise
    else:
        log.info("Skipping msigdb (disabled in config)")

    if cfg.pubmed.enabled:
        try:
            pubmed_cfg = load_config(cfg.pubmed.config_path, PubMedHarvestConfig)

            async def _run_pubmed() -> int:
                async with NCBIClient.from_config(ncbi_cfg) as client:
                    return await harvest_pubmed(pubmed_cfg, client)

            count = asyncio.run(_run_pubmed())
            log.info("pubmed: {} records", count)
            output_paths.append(pubmed_cfg.output_path)
        except Exception as exc:
            if skip_on_error:
                log.error("pubmed harvester failed: {!r}", exc)
            else:
                raise
    else:
        log.info("Skipping pubmed (disabled in config)")

    merged_count = merge_raw_concept_files(output_paths, cfg.output_merged_path)
    log.info("Merged {} total records into {}", merged_count, cfg.output_merged_path)
    _write_run_meta("harvest-all", compute_cache_key(cfg.model_dump()))
