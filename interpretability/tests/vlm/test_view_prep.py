"""Tests for View A embedding preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from molf_interp.vlm.view_prep import (
    CLINICAL_DROP_IDS,
    assign_embedding_set,
    compute_dedup_drops,
    compute_dedup_mapping,
    prepare_view_a,
)

READY_COLUMNS = [
    "concept_id",
    "concept_name",
    "tier",
    "embedding_set",
    "species",
    "prompt_text",
    "species_status",
]


def _concept(
    concept_id: str,
    name: str,
    tier: str,
    provenance: str = "reference",
    in_primary_lexicon: bool = True,
) -> dict[str, object]:
    return {
        "concept_id": concept_id,
        "concept_name": name,
        "tier": tier,
        "category": "category",
        "subcategory": "subcategory",
        "source": "test",
        "provenance": provenance,
        "organ": "lung",
        "level": "tissue",
        "concept_type": None,
        "cross_bank_status": "present",
        "in_primary_lexicon": in_primary_lexicon,
    }


def _prompt_rows(
    concept_id: str,
    name: str,
    tier: str,
    species_status: str = "shared",
) -> list[dict[str, str]]:
    return [
        {
            "concept_id": concept_id,
            "concept_name": name,
            "tier": tier,
            "species": "Homo_sapiens",
            "prompt_text": f"Human prompt for {name}",
            "species_status": species_status,
        },
        {
            "concept_id": concept_id,
            "concept_name": name,
            "tier": tier,
            "species": "Mus_musculus",
            "prompt_text": f"Mouse prompt for {name}",
            "species_status": species_status,
        },
    ]


def _write_inputs(
    tmp_path: Path,
    concepts: list[dict[str, object]],
    prompts: list[dict[str, str]],
) -> tuple[Path, Path]:
    concepts_path = tmp_path / "concepts.parquet"
    prompts_path = tmp_path / "prompts.parquet"
    pd.DataFrame(concepts).to_parquet(concepts_path, index=False)
    pd.DataFrame(prompts).to_parquet(prompts_path, index=False)
    return concepts_path, prompts_path


def _output_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    out = tmp_path / "out"
    return (
        out / "merged_concepts_cleaned.parquet",
        out / "merged_prompts_cleaned.parquet",
        out / "view_a_ready.parquet",
        out / "view_a_deduplication_map.parquet",
    )


def _run_prepare(
    tmp_path: Path,
    concepts: list[dict[str, object]],
    prompts: list[dict[str, str]],
    dry_run: bool = False,
):
    concepts_path, prompts_path = _write_inputs(tmp_path, concepts, prompts)
    output_concepts, output_prompts, ready_prompts, dedup_map = _output_paths(tmp_path)
    result = prepare_view_a(
        concepts_path=concepts_path,
        prompts_path=prompts_path,
        output_concepts_path=output_concepts,
        output_prompts_path=output_prompts,
        ready_prompts_path=ready_prompts,
        dedup_map_path=dedup_map,
        dry_run=dry_run,
    )
    return result, output_concepts, output_prompts, ready_prompts, dedup_map


def test_clinical_drop_ids_are_exact() -> None:
    assert (
        frozenset(
            {
                "MOR-11275",
                "MOR-11279",
                "MOR-11920",
                "MOR-12987",
            }
        )
        == CLINICAL_DROP_IDS
    )


@pytest.mark.parametrize("tier", ["H1_morphology", "H2_cell_type", "H2_niche"])
def test_assign_embedding_set_visual_tiers(tier: str) -> None:
    assert assign_embedding_set(tier) == "visual"


@pytest.mark.parametrize("tier", ["H2_gene_program", "H2_pathway"])
def test_assign_embedding_set_transcriptomic_tiers(tier: str) -> None:
    assert assign_embedding_set(tier) == "transcriptomic"


def test_assign_embedding_set_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown View A tier"):
        assign_embedding_set("unknown")


def test_compute_dedup_drops_reference_wins() -> None:
    df = pd.DataFrame(
        [
            _concept("MOR-1000", "Necrosis", "H1_morphology", "reference"),
            _concept("MOR-0001", "Necrosis", "H1_morphology", "harvested"),
        ]
    )
    assert compute_dedup_drops(df) == frozenset({"MOR-0001"})


def test_compute_dedup_drops_lower_id_wins_same_provenance() -> None:
    df = pd.DataFrame(
        [
            _concept("MOR-1000", "Necrosis", "H1_morphology", "reference"),
            _concept("MOR-0001", "Necrosis", "H1_morphology", "reference"),
        ]
    )
    assert compute_dedup_drops(df) == frozenset({"MOR-1000"})


def test_compute_dedup_drops_three_way_group() -> None:
    df = pd.DataFrame(
        [
            _concept("MOR-1000", "Necrosis", "H1_morphology", "reference"),
            _concept("MOR-0001", "Necrosis", "H1_morphology", "harvested"),
            _concept("MOR-0002", "Necrosis", "H1_morphology", "harvested"),
        ]
    )
    assert compute_dedup_drops(df) == frozenset({"MOR-0001", "MOR-0002"})


def test_compute_dedup_drops_case_insensitive() -> None:
    df = pd.DataFrame(
        [
            _concept("MOR-0001", "acinar pattern", "H1_morphology", "reference"),
            _concept("MOR-0002", "Acinar Pattern", "H1_morphology", "reference"),
        ]
    )
    assert compute_dedup_drops(df) == frozenset({"MOR-0002"})


def test_compute_dedup_drops_cross_tier_not_deduped() -> None:
    df = pd.DataFrame(
        [
            _concept("MOR-0001", "Fibrosis", "H1_morphology", "reference"),
            _concept("NIC-0001", "Fibrosis", "H2_niche", "reference"),
        ]
    )
    assert compute_dedup_drops(df) == frozenset()


def test_compute_dedup_mapping_has_expected_columns() -> None:
    df = pd.DataFrame(
        [
            _concept("MOR-0001", "Necrosis", "H1_morphology", "reference"),
            _concept("MOR-0002", "Necrosis", "H1_morphology", "harvested"),
        ]
    )
    mapping = compute_dedup_mapping(df)
    assert list(mapping.columns) == [
        "duplicate_concept_id",
        "canonical_concept_id",
        "tier",
        "concept_name",
        "duplicate_provenance",
        "canonical_provenance",
        "reason",
    ]


def test_prepare_view_a_drops_clinical_ids(tmp_path: Path) -> None:
    concepts = [
        _concept("MOR-11275", "Grade 2 clear cell renal cell carcinoma", "H1_morphology"),
        _concept("MOR-12986", "Squamous intraepithelial neoplasia", "H1_morphology"),
    ]
    prompts = [
        row
        for concept in concepts
        for row in _prompt_rows(
            str(concept["concept_id"]),
            str(concept["concept_name"]),
            str(concept["tier"]),
        )
    ]
    _, output_concepts, _, ready_prompts, _ = _run_prepare(tmp_path, concepts, prompts)

    cleaned = pd.read_parquet(output_concepts)
    ready = pd.read_parquet(ready_prompts)
    assert "MOR-11275" not in set(cleaned["concept_id"])
    assert "MOR-11275" not in set(ready["concept_id"])
    assert "MOR-12986" in set(cleaned["concept_id"])


def test_prepare_view_a_does_not_drop_reminted_id_with_clean_name(tmp_path: Path) -> None:
    concepts = [
        _concept("MOR-11275", "Granuloma", "H1_morphology"),
        _concept("MOR-0001", "Grade 2 clear cell renal cell carcinoma", "H1_morphology"),
    ]
    prompts = [
        row
        for concept in concepts
        for row in _prompt_rows(
            str(concept["concept_id"]),
            str(concept["concept_name"]),
            str(concept["tier"]),
        )
    ]
    result, output_concepts, _, ready_prompts, _ = _run_prepare(tmp_path, concepts, prompts)

    cleaned = pd.read_parquet(output_concepts)
    ready = pd.read_parquet(ready_prompts)
    assert result.n_clinical_dropped == 1
    assert set(cleaned["concept_id"]) == {"MOR-11275"}
    assert set(ready["concept_id"]) == {"MOR-11275"}


def test_prepare_view_a_dedup_applied(tmp_path: Path) -> None:
    concepts = [
        _concept("MOR-0001", "Necrosis", "H1_morphology", "reference"),
        _concept("MOR-0002", "Necrosis", "H1_morphology", "harvested"),
    ]
    prompts = [
        row
        for concept in concepts
        for row in _prompt_rows(
            str(concept["concept_id"]),
            str(concept["concept_name"]),
            str(concept["tier"]),
        )
    ]
    result, output_concepts, _, ready_prompts, dedup_map = _run_prepare(tmp_path, concepts, prompts)

    cleaned = pd.read_parquet(output_concepts)
    ready = pd.read_parquet(ready_prompts)
    mapping = pd.read_parquet(dedup_map)
    assert result.n_dedup_dropped == 1
    assert set(cleaned["concept_id"]) == {"MOR-0001"}
    assert set(ready["concept_id"]) == {"MOR-0001"}
    assert mapping.loc[0, "duplicate_concept_id"] == "MOR-0002"
    assert mapping.loc[0, "canonical_concept_id"] == "MOR-0001"


def test_prepare_view_a_embedding_set_column_present(tmp_path: Path) -> None:
    concepts = [
        _concept("MOR-0001", "Necrosis", "H1_morphology"),
        _concept("GPR-0001", "Hypoxia program", "H2_gene_program"),
    ]
    prompts = [
        row
        for concept in concepts
        for row in _prompt_rows(
            str(concept["concept_id"]),
            str(concept["concept_name"]),
            str(concept["tier"]),
        )
    ]
    _, output_concepts, output_prompts, _, _ = _run_prepare(tmp_path, concepts, prompts)

    cleaned = pd.read_parquet(output_concepts)
    cleaned_prompts = pd.read_parquet(output_prompts)
    assert set(cleaned["embedding_set"]) == {"visual", "transcriptomic"}
    assert set(cleaned_prompts["embedding_set"]) == {"visual", "transcriptomic"}


def test_prepare_view_a_ready_file_excludes_human_only_mouse(tmp_path: Path) -> None:
    concepts = [_concept("MOR-0001", "Necrosis", "H1_morphology")]
    prompts = _prompt_rows("MOR-0001", "Necrosis", "H1_morphology", "human_only")
    _, _, _, ready_prompts, _ = _run_prepare(tmp_path, concepts, prompts)

    ready = pd.read_parquet(ready_prompts)
    assert len(ready) == 1
    assert ready.loc[0, "species"] == "Homo_sapiens"


def test_prepare_view_a_cleaned_prompts_keep_traceability(tmp_path: Path) -> None:
    concepts = [_concept("MOR-0001", "Necrosis", "H1_morphology")]
    prompts = _prompt_rows("MOR-0001", "Necrosis", "H1_morphology", "human_only")
    _, _, output_prompts, ready_prompts, _ = _run_prepare(tmp_path, concepts, prompts)

    cleaned_prompts = pd.read_parquet(output_prompts)
    ready = pd.read_parquet(ready_prompts)
    assert len(cleaned_prompts) == 2
    assert len(ready) == 1
    assert "Mus_musculus" in set(cleaned_prompts["species"])
    assert "Mus_musculus" not in set(ready["species"])


def test_prepare_view_a_dry_run_writes_nothing(tmp_path: Path) -> None:
    concepts = [_concept("MOR-0001", "Necrosis", "H1_morphology")]
    prompts = _prompt_rows("MOR-0001", "Necrosis", "H1_morphology")
    result, output_concepts, output_prompts, ready_prompts, dedup_map = _run_prepare(
        tmp_path, concepts, prompts, dry_run=True
    )

    assert result.n_concepts_after == 1
    assert not output_concepts.exists()
    assert not output_prompts.exists()
    assert not ready_prompts.exists()
    assert not dedup_map.exists()


def test_prepare_view_a_counts_match_synthetic_fixture(tmp_path: Path) -> None:
    concepts = [
        _concept("MOR-11275", "Grade 2 clear cell renal cell carcinoma", "H1_morphology"),
        _concept("MOR-11279", "Grade 3 clear cell renal cell carcinoma", "H1_morphology"),
        _concept("MOR-11920", "Microsatellite stable endometrial carcinoma", "H1_morphology"),
        _concept("MOR-12987", "Squamous intraepithelial neoplasia grade 2", "H1_morphology"),
        _concept("MOR-0001", "Necrosis", "H1_morphology", "reference"),
        _concept("MOR-0002", "Necrosis", "H1_morphology", "harvested"),
        _concept("CTY-0001", "T cell", "H2_cell_type", "reference"),
        _concept("CTY-0002", "T cell", "H2_cell_type", "harvested"),
        _concept("NIC-0001", "Immune niche", "H2_niche"),
        _concept("GPR-0001", "Hypoxia program", "H2_gene_program"),
        _concept("PWY-0001", "MAPK pathway", "H2_pathway"),
    ]
    prompts = [
        row
        for concept in concepts
        for row in _prompt_rows(
            str(concept["concept_id"]),
            str(concept["concept_name"]),
            str(concept["tier"]),
        )
    ]
    result, _, _, ready_prompts, _ = _run_prepare(tmp_path, concepts, prompts)

    ready = pd.read_parquet(ready_prompts)
    assert result.n_concepts_before == 11
    assert result.n_clinical_dropped == 4
    assert result.n_dedup_dropped == 2
    assert result.n_concepts_after == 5
    assert result.n_visual_concepts == 3
    assert result.n_transcriptomic_concepts == 2
    assert len(ready) == 10


def test_ready_schema_columns(tmp_path: Path) -> None:
    concepts = [_concept("MOR-0001", "Necrosis", "H1_morphology")]
    prompts = _prompt_rows("MOR-0001", "Necrosis", "H1_morphology")
    _, _, _, ready_prompts, _ = _run_prepare(tmp_path, concepts, prompts)

    ready = pd.read_parquet(ready_prompts)
    assert list(ready.columns) == READY_COLUMNS


def test_ready_file_has_no_null_or_empty_prompts(tmp_path: Path) -> None:
    concepts = [_concept("MOR-0001", "Necrosis", "H1_morphology")]
    prompts = _prompt_rows("MOR-0001", "Necrosis", "H1_morphology")
    _, _, _, ready_prompts, _ = _run_prepare(tmp_path, concepts, prompts)

    ready = pd.read_parquet(ready_prompts)
    assert ready["prompt_text"].isna().sum() == 0
    assert ready["prompt_text"].astype(str).str.strip().eq("").sum() == 0


def test_ready_file_has_only_two_embedding_sets(tmp_path: Path) -> None:
    concepts = [
        _concept("MOR-0001", "Necrosis", "H1_morphology"),
        _concept("GPR-0001", "Hypoxia program", "H2_gene_program"),
    ]
    prompts = [
        row
        for concept in concepts
        for row in _prompt_rows(
            str(concept["concept_id"]),
            str(concept["concept_name"]),
            str(concept["tier"]),
        )
    ]
    _, _, _, ready_prompts, _ = _run_prepare(tmp_path, concepts, prompts)

    ready = pd.read_parquet(ready_prompts)
    assert set(ready["embedding_set"]) == {"visual", "transcriptomic"}
