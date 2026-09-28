"""Tests for the `curate drop-tier` CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from molf_interp.cli.main import app


def _write_bank(bank_dir: Path, tiers: list[str]) -> tuple[int, int]:
    """Write a minimal bank with one concept per tier, two prompts each."""
    bank_dir.mkdir(parents=True, exist_ok=True)
    concepts = []
    prompts = []
    for i, tier in enumerate(tiers):
        cid = f"TST-{i + 1:04d}"
        concepts.append(
            {
                "concept_id": cid,
                "concept_name": f"concept {i}",
                "tier": tier,
                "category": "test",
                "subcategory": "test",
                "source": "ncit",
                "provenance": "automated",
                "organ": None,
                "level": None,
                "concept_type": None,
            }
        )
        for sp in ("Homo_sapiens", "Mus_musculus"):
            prompts.append({"concept_id": cid, "tier": tier, "species": sp, "prompt_text": "p"})
    pd.DataFrame(concepts).to_parquet(bank_dir / "concepts.parquet", index=False)
    pd.DataFrame(prompts).to_parquet(bank_dir / "prompts.parquet", index=False)
    return len(concepts), len(prompts)


runner = CliRunner()


def test_drop_tier_removes_concepts_and_prompts(tmp_path: Path) -> None:
    bank_dir = tmp_path / "bank"
    _write_bank(bank_dir, ["H1_morphology", "H2_cell_type", "H2_niche"])

    result = runner.invoke(
        app, ["curate", "drop-tier", "H2_niche", "--bank", str(bank_dir), "--confirm"]
    )
    assert result.exit_code == 0, result.output

    concepts_df = pd.read_parquet(bank_dir / "concepts.parquet")
    prompts_df = pd.read_parquet(bank_dir / "prompts.parquet")

    assert "H2_niche" not in concepts_df["tier"].values
    assert len(concepts_df) == 2  # 3 - 1
    assert len(prompts_df) == 4  # (3 * 2) - 2


def test_drop_tier_dry_run_does_not_mutate(tmp_path: Path) -> None:
    bank_dir = tmp_path / "bank"
    n_concepts, n_prompts = _write_bank(bank_dir, ["H1_morphology", "H2_niche"])

    result = runner.invoke(app, ["curate", "drop-tier", "H2_niche", "--bank", str(bank_dir)])
    assert result.exit_code == 0
    assert "Dry-run" in result.output

    # Files must be unchanged
    concepts_df = pd.read_parquet(bank_dir / "concepts.parquet")
    prompts_df = pd.read_parquet(bank_dir / "prompts.parquet")
    assert len(concepts_df) == n_concepts
    assert len(prompts_df) == n_prompts


def test_drop_tier_missing_tier_is_noop(tmp_path: Path) -> None:
    bank_dir = tmp_path / "bank"
    n_concepts, _n_prompts = _write_bank(bank_dir, ["H1_morphology", "H2_cell_type"])

    result = runner.invoke(
        app, ["curate", "drop-tier", "H2_niche", "--bank", str(bank_dir), "--confirm"]
    )
    assert result.exit_code == 0
    assert "not found" in result.output.lower() or "nothing" in result.output.lower()

    concepts_df = pd.read_parquet(bank_dir / "concepts.parquet")
    assert len(concepts_df) == n_concepts


def test_drop_tier_writes_metadata(tmp_path: Path) -> None:
    bank_dir = tmp_path / "bank"
    _write_bank(bank_dir, ["H1_morphology", "H2_niche"])

    runner.invoke(app, ["curate", "drop-tier", "H2_niche", "--bank", str(bank_dir), "--confirm"])

    meta_path = bank_dir / "run_metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["tier"] == "H2_niche"
    assert meta["concepts_before"] == 2
    assert meta["concepts_after"] == 1
    assert len(meta["affected_concept_ids"]) == 1
