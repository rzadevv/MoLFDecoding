"""Tests for Stage 7: prompt generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from molf_interp.curation.config import CurationConfig
from molf_interp.curation.stage_7_prompt_gen import (
    _genes_for,
    run_stage_7,
)


def _cfg() -> CurationConfig:
    return CurationConfig()


def _h1_row(
    label: str = "coagulative necrosis",
    organ: str | None = "lung",
    level: str | None = "tissue",
    category: str = "organ_specific",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "concept_id": ["MOR-10001"],
            "preferred_label": [label],
            "tier": ["H1_morphology"],
            "organ": [organ],
            "level": [level],
            "category": [category],
            "concept_type": [None],
            "source_name": ["ncit"],
            "extra": [[]],
        }
    )


def _h2_row(
    tier: str,
    label: str = "macrophage",
    extra: object = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "concept_id": ["CTY-10001"],
            "preferred_label": [label],
            "tier": [tier],
            "organ": [None],
            "level": [None],
            "category": [tier.replace("H2_", "")],
            "concept_type": [tier.replace("H2_", "")],
            "source_name": ["ncit"],
            "extra": [extra if extra is not None else []],
        }
    )


# ---------------------------------------------------------------------------
# Species / volume
# ---------------------------------------------------------------------------


def test_two_species_per_concept() -> None:
    df = _h1_row()
    result = run_stage_7(df, _cfg())
    assert len(result) == 2  # 1 concept x 2 species


def test_all_prompts_non_empty() -> None:
    df = pd.concat(
        [
            _h1_row(category="organ_specific"),
            _h2_row("H2_cell_type"),
            _h2_row("H2_niche", label="tumor niche"),
            _h2_row("H2_pathway", label="PI3K pathway"),
            _h2_row("H2_gene_program", label="adipogenesis"),
        ],
        ignore_index=True,
    )
    result = run_stage_7(df, _cfg())
    assert (result["prompt_text"].str.len() > 0).all()


# ---------------------------------------------------------------------------
# H1 category branches
# ---------------------------------------------------------------------------


def test_h1_organ_specific_human() -> None:
    result = run_stage_7(_h1_row(category="organ_specific", organ="lung"), _cfg())
    human = result[result["species"] == "Homo_sapiens"].iloc[0]["prompt_text"]
    assert "lung" in human
    assert "human" in human
    assert "H&E" in human


def test_h1_organ_specific_mouse() -> None:
    result = run_stage_7(_h1_row(category="organ_specific", organ="lung"), _cfg())
    mouse = result[result["species"] == "Mus_musculus"].iloc[0]["prompt_text"]
    assert "lung" in mouse
    assert "murine" in mouse
    assert "mouse" not in mouse


def test_h1_default_category() -> None:
    result = run_stage_7(_h1_row(category="default", organ=None), _cfg())
    human = result[result["species"] == "Homo_sapiens"].iloc[0]["prompt_text"]
    assert "Histopathological feature" in human
    assert "observed" in human
    assert "H&E" in human


def test_h1_microenvironment_category() -> None:
    result = run_stage_7(_h1_row(category="microenvironment", organ=None), _cfg())
    human = result[result["species"] == "Homo_sapiens"].iloc[0]["prompt_text"]
    assert "microenvironment" in human.lower()
    assert "H&E" in human


def test_h1_artifact_category() -> None:
    result = run_stage_7(_h1_row(category="artifact", organ=None), _cfg())
    human = result[result["species"] == "Homo_sapiens"].iloc[0]["prompt_text"]
    assert "artifact" in human.lower()
    assert "H&E" in human


def test_h1_unknown_category_falls_back_to_default() -> None:
    result = run_stage_7(_h1_row(category="made_up_category", organ=None), _cfg())
    human = result[result["species"] == "Homo_sapiens"].iloc[0]["prompt_text"]
    assert "Histopathological feature" in human


# ---------------------------------------------------------------------------
# H2 branches
# ---------------------------------------------------------------------------


def test_h2_cell_type_without_markers() -> None:
    result = run_stage_7(_h2_row("H2_cell_type", label="macrophage"), _cfg())
    human = result[result["species"] == "Homo_sapiens"].iloc[0]["prompt_text"]
    assert "Cell type present" in human
    assert "macrophage" in human
    assert "human" in human


def test_h2_cell_type_mouse_uses_murine() -> None:
    result = run_stage_7(_h2_row("H2_cell_type", label="macrophage"), _cfg())
    mouse = result[result["species"] == "Mus_musculus"].iloc[0]["prompt_text"]
    assert "murine" in mouse
    assert "mouse" not in mouse


def test_h2_niche() -> None:
    result = run_stage_7(_h2_row("H2_niche", label="perivascular niche"), _cfg())
    human = result[result["species"] == "Homo_sapiens"].iloc[0]["prompt_text"]
    assert "niche" in human.lower()
    assert "perivascular niche" in human


def test_h2_pathway() -> None:
    result = run_stage_7(_h2_row("H2_pathway", label="PI3K-AKT signaling"), _cfg())
    human = result[result["species"] == "Homo_sapiens"].iloc[0]["prompt_text"]
    assert "Signaling pathway" in human
    assert "PI3K-AKT signaling" in human


def test_h2_gene_program_without_genes() -> None:
    result = run_stage_7(_h2_row("H2_gene_program", label="apoptosis"), _cfg())
    human = result[result["species"] == "Homo_sapiens"].iloc[0]["prompt_text"]
    assert "Transcriptional program" in human
    assert "apoptosis" in human
    assert "involving genes" not in human


def test_h2_gene_program_with_genes_human() -> None:
    genes_str = "FOXP3,CD8A,PDCD1,CTLA4,LAG3"
    extra = [np.array(["genes", genes_str], dtype=object)]
    result = run_stage_7(_h2_row("H2_gene_program", label="T cell exhaustion", extra=extra), _cfg())
    human = result[result["species"] == "Homo_sapiens"].iloc[0]["prompt_text"]
    assert "involving genes such as" in human
    assert "FOXP3" in human  # CAPS preserved for human


def test_h2_gene_program_with_genes_mouse_sentence_case() -> None:
    genes_str = "FOXP3,CD8A,PDCD1,CTLA4,LAG3"
    extra = [np.array(["genes", genes_str], dtype=object)]
    result = run_stage_7(_h2_row("H2_gene_program", label="T cell exhaustion", extra=extra), _cfg())
    mouse = result[result["species"] == "Mus_musculus"].iloc[0]["prompt_text"]
    assert "Foxp3" in mouse  # sentence-cased for mouse
    assert "FOXP3" not in mouse
    assert "murine" in mouse


def test_h2_gene_program_genes_capped_at_5() -> None:
    genes_str = "A,B,C,D,E,F,G,H"
    extra = [np.array(["genes", genes_str], dtype=object)]
    result = run_stage_7(_h2_row("H2_gene_program", label="prog", extra=extra), _cfg())
    human = result[result["species"] == "Homo_sapiens"].iloc[0]["prompt_text"]
    assert "F" not in human  # only first 5


# ---------------------------------------------------------------------------
# _genes_for unit tests
# ---------------------------------------------------------------------------


def _gp_row(genes_str: str) -> pd.Series:  # type: ignore[type-arg]
    return pd.Series({"extra": [np.array(["genes", genes_str], dtype=object)]})


def test_genes_for_human_keeps_caps() -> None:
    row = _gp_row("FOXP3,CD8A,PDCD1")
    result = _genes_for(row, "Homo_sapiens")
    assert result == "FOXP3, CD8A, PDCD1"


def test_genes_for_mouse_sentence_case() -> None:
    row = _gp_row("FOXP3,CD8A")
    result = _genes_for(row, "Mus_musculus")
    assert result == "Foxp3, Cd8a"


def test_genes_for_no_genes_key_returns_none() -> None:
    row = pd.Series({"extra": [np.array(["msigdb_url", "http://example.com"], dtype=object)]})
    assert _genes_for(row, "Homo_sapiens") is None


def test_genes_for_empty_extra_returns_none() -> None:
    row = pd.Series({"extra": []})
    assert _genes_for(row, "Homo_sapiens") is None
