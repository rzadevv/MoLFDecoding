"""BioPortal CLI sub-app for molf-interp."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from molf_interp.io.cache import compute_cache_key
from molf_interp.io.config import load_config
from molf_interp.io.metadata import RunMetadata, _python_version, collect_git_info, write_metadata
from molf_interp.sources.bioportal.client import BioPortalClient
from molf_interp.sources.bioportal.config import BioPortalConfig
from molf_interp.utils.logging import get_logger, setup_logging

bioportal_app = typer.Typer(no_args_is_help=True, help="BioPortal ontology operations.")

_DEFAULT_CONFIG = Path("configs/sources/bioportal.yaml")


def _load_cfg(config_path: Path) -> BioPortalConfig:
    return load_config(config_path, BioPortalConfig)


def _write_run_meta(command: str, cfg: BioPortalConfig) -> None:
    sha, dirty = collect_git_info()
    run_id = uuid.uuid4().hex
    meta = RunMetadata(
        run_id=run_id,
        git_sha=sha,
        git_dirty=dirty,
        config_hash=compute_cache_key(cfg.model_dump()),
        started_at=datetime.now(tz=UTC),
        command=command,
        python_version=_python_version(),
    )
    out = Path("data/outputs") / f"run_metadata_{run_id}.json"
    write_metadata(meta, out)


@bioportal_app.command()
def info(
    acronym: Annotated[str, typer.Argument(help="Ontology acronym, e.g. NCIT, CL, GO.")],
    config: Annotated[
        Path, typer.Option("--config", help="Path to BioPortalConfig YAML.")
    ] = _DEFAULT_CONFIG,
) -> None:
    """Fetch ontology metadata. Useful as an auth + access smoke test."""
    setup_logging("INFO")
    log = get_logger("cli.bioportal.info")
    cfg = _load_cfg(config)

    async def _run() -> None:
        async with BioPortalClient.from_config(cfg) as client:
            data = await client.get_ontology(acronym)
        log.info("Ontology {}: {}", acronym, json.dumps(data, indent=2, default=str))

    asyncio.run(_run())
    _write_run_meta("bioportal-info", cfg)


@bioportal_app.command()
def get_class(
    acronym: Annotated[str, typer.Argument(help="Ontology acronym.")],
    class_iri: Annotated[str, typer.Argument(help="Full class IRI.")],
    config: Annotated[
        Path, typer.Option("--config", help="Path to BioPortalConfig YAML.")
    ] = _DEFAULT_CONFIG,
) -> None:
    """Fetch and print one class as JSON."""
    setup_logging("INFO")
    log = get_logger("cli.bioportal.get_class")
    cfg = _load_cfg(config)

    async def _run() -> None:
        async with BioPortalClient.from_config(cfg) as client:
            cls = await client.get_class(acronym, class_iri)
        log.info("Class {}: {}", class_iri, cls.model_dump_json(indent=2))

    asyncio.run(_run())
    _write_run_meta("bioportal-get-class", cfg)


@bioportal_app.command()
def count_descendants(
    acronym: Annotated[str, typer.Argument(help="Ontology acronym.")],
    root_iri: Annotated[str, typer.Argument(help="Root class IRI.")],
    limit: Annotated[
        int, typer.Option("--limit", help="Stop after N descendants (0 = no limit).")
    ] = 0,
    config: Annotated[
        Path, typer.Option("--config", help="Path to BioPortalConfig YAML.")
    ] = _DEFAULT_CONFIG,
) -> None:
    """Count descendants under root_iri. Use --limit for a fast smoke test."""
    setup_logging("INFO")
    log = get_logger("cli.bioportal.count_descendants")
    cfg = _load_cfg(config)

    async def _run() -> int:
        count = 0
        async with BioPortalClient.from_config(cfg) as client:
            async for _ in client.iter_descendants(acronym, root_iri):
                count += 1
                if limit and count >= limit:
                    break
        return count

    n = asyncio.run(_run())
    log.info(
        "Descendants of {} in {}: {} (limit={})",
        root_iri,
        acronym,
        n,
        limit or "none",
    )
    _write_run_meta("bioportal-count-descendants", cfg)
