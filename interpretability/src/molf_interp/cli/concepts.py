"""CLI sub-app for concept bank operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from molf_interp.concepts.ingesters.reference_tsv import load_reference_bank
from molf_interp.concepts.store import read_concept_bank, write_concept_bank
from molf_interp.io.cache import compute_cache_key
from molf_interp.io.metadata import RunMetadata, _python_version, collect_git_info, write_metadata
from molf_interp.utils.logging import get_logger, setup_logging

concepts_app = typer.Typer(no_args_is_help=True, help="Concept bank operations.")


@concepts_app.command()
def validate_reference(
    input_dir: Annotated[
        Path,
        typer.Option("--input", "-i", help="Directory with the three reference TSVs."),
    ] = Path("data/concept_bank"),
) -> None:
    """Load and validate the reference TSV bank, then print the summary."""
    setup_logging("INFO")
    log = get_logger("cli.concepts.validate_reference")
    bank = load_reference_bank(input_dir)
    log.info("Summary: {}", bank.summary())


@concepts_app.command()
def build_reference(
    input_dir: Annotated[
        Path,
        typer.Option("--input", "-i", help="Directory with the three reference TSVs."),
    ] = Path("data/concept_bank"),
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Directory to write parquet files."),
    ] = Path("data/cache/concept_bank/reference"),
) -> None:
    """Load and validate the reference TSV bank, write parquet and run metadata."""
    setup_logging("INFO")
    log = get_logger("cli.concepts.build_reference")
    bank = load_reference_bank(input_dir)
    write_concept_bank(bank, output_dir)
    log.info("Wrote concept bank to {}", output_dir)

    sha, dirty = collect_git_info()
    run_id = uuid.uuid4().hex
    meta = RunMetadata(
        run_id=run_id,
        git_sha=sha,
        git_dirty=dirty,
        config_hash=compute_cache_key({"input_dir": str(input_dir), "ingester": "reference_tsv"}),
        started_at=datetime.now(tz=UTC),
        command="concepts.build_reference",
        python_version=_python_version(),
    )
    write_metadata(meta, output_dir / f"run_metadata_{run_id}.json")
    log.info("Run metadata written: run_id={}", run_id)


@concepts_app.command()
def show(
    bank_dir: Annotated[
        Path,
        typer.Option("--bank", "-b", help="Directory with concepts.parquet and prompts.parquet."),
    ] = Path("data/cache/concept_bank/reference"),
) -> None:
    """Read a parquet ConceptBank from disk and print its summary."""
    setup_logging("INFO")
    log = get_logger("cli.concepts.show")
    bank = read_concept_bank(bank_dir)
    log.info("Summary: {}", bank.summary())
