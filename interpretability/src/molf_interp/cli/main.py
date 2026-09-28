"""CLI entry point for molf-interp."""

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from molf_interp.cli.bioportal import bioportal_app
from molf_interp.cli.compare import compare_app
from molf_interp.cli.concepts import concepts_app
from molf_interp.cli.curate import curate_app
from molf_interp.cli.harvest import harvest_app
from molf_interp.cli.vlm import vlm_app
from molf_interp.io.cache import compute_cache_key
from molf_interp.io.config import HelloConfig, load_config
from molf_interp.io.metadata import (
    RunMetadata,
    _python_version,
    collect_git_info,
    write_metadata,
)
from molf_interp.utils.logging import get_logger, setup_logging

app = typer.Typer(no_args_is_help=True)
app.add_typer(concepts_app, name="concepts")
app.add_typer(bioportal_app, name="bioportal")
app.add_typer(harvest_app, name="harvest")
app.add_typer(curate_app, name="curate")
app.add_typer(compare_app, name="compare")
app.add_typer(vlm_app, name="vlm")


@app.callback()
def _callback() -> None:
    """MoLF Interpretability Pipeline CLI."""


@app.command()
def hello(
    config: Annotated[Path, typer.Option("--config", help="Path to YAML config file.")],
) -> None:
    """Greet a user by name using a YAML config.

    Args:
        config: Path to the HelloConfig YAML file.
    """
    setup_logging("INFO")
    log = get_logger("cli.hello")

    cfg = load_config(config, HelloConfig)
    log.info(f"{cfg.greeting}, {cfg.name}")

    sha, dirty = collect_git_info()
    run_id = uuid.uuid4().hex
    meta = RunMetadata(
        run_id=run_id,
        git_sha=sha,
        git_dirty=dirty,
        config_hash=compute_cache_key(cfg.model_dump()),
        started_at=datetime.now(tz=UTC),
        command="hello",
        python_version=_python_version(),
    )
    out = Path("data/outputs") / f"run_metadata_{run_id}.json"
    write_metadata(meta, out)
