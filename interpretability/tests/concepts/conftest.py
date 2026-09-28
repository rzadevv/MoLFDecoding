"""Shared fixture generators for concept tests. No fixture files committed."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DEFAULT_MORPHOLOGY: list[dict[str, str]] = [
    {
        "concept_id": "MOR-0001",
        "concept_name": "coagulative necrosis",
        "category": "necrosis",
        "subcategory": "tumor_necrosis",
        "level": "tissue",
        "organ": "universal",
        "source": "Robbins",
    },
    {
        "concept_id": "MOR-0002",
        "concept_name": "nuclear pleomorphism",
        "category": "nuclear",
        "subcategory": "atypia",
        "level": "cellular",
        "organ": "universal",
        "source": "WHO-Blue-Books",
    },
    {
        "concept_id": "MOR-0003",
        "concept_name": "Gleason pattern 5",
        "category": "glandular",
        "subcategory": "prostate",
        "level": "tissue",
        "organ": "prostate",
        "source": "WHO-Blue-Books",
    },
]

_DEFAULT_TRANSCRIPTOMICS: list[dict[str, str]] = [
    {
        "concept_id": "CTY-0001",
        "concept_name": "CD8+ effector T cell",
        "category": "immune_cell",
        "subcategory": "cytotoxic",
        "concept_type": "cell_type",
        "source": "HCA",
    },
    {
        "concept_id": "NIC-0001",
        "concept_name": "tertiary lymphoid structure",
        "category": "immune_niche",
        "subcategory": "TLS",
        "concept_type": "niche",
        "source": "literature",
    },
    {
        "concept_id": "GPR-0001",
        "concept_name": "HALLMARK_HYPOXIA",
        "category": "hallmark",
        "subcategory": "stress",
        "concept_type": "gene_program",
        "source": "MSigDB-Hallmark",
    },
    {
        "concept_id": "PWY-0001",
        "concept_name": "PI3K-Akt signaling pathway",
        "category": "signaling",
        "subcategory": "growth",
        "concept_type": "pathway",
        "source": "KEGG",
    },
]

_ALL_CONCEPT_IDS = [
    "MOR-0001",
    "MOR-0002",
    "MOR-0003",
    "CTY-0001",
    "NIC-0001",
    "GPR-0001",
    "PWY-0001",
]
_HIERARCHY_BY_ID = {
    "MOR-0001": "H1_morphology",
    "MOR-0002": "H1_morphology",
    "MOR-0003": "H1_morphology",
    "CTY-0001": "H2_cell_type",
    "NIC-0001": "H2_niche",
    "GPR-0001": "H2_gene_program",
    "PWY-0001": "H2_pathway",
}
_NAME_BY_ID = {
    "MOR-0001": "coagulative necrosis",
    "MOR-0002": "nuclear pleomorphism",
    "MOR-0003": "Gleason pattern 5",
    "CTY-0001": "CD8+ effector T cell",
    "NIC-0001": "tertiary lymphoid structure",
    "GPR-0001": "HALLMARK_HYPOXIA",
    "PWY-0001": "PI3K-Akt signaling pathway",
}

_HUMAN_ONLY_ID = "MOR-0003"


def _default_cross_species() -> list[dict[str, str]]:
    rows = []
    for cid in _ALL_CONCEPT_IDS:
        status = "human_only" if cid == _HUMAN_ONLY_ID else "shared"
        for sp in ("Homo_sapiens", "Mus_musculus"):
            rows.append(
                {
                    "concept_id": cid,
                    "concept_name": _NAME_BY_ID[cid],
                    "hierarchy": _HIERARCHY_BY_ID[cid],
                    "species": sp,
                    "prompt_text": f"Describe {_NAME_BY_ID[cid]} in {sp}.",
                    "species_status": status,
                }
            )
    return rows


def write_su_fixture(
    dst: Path,
    *,
    morphology_rows: list[dict[str, str]] | None = None,
    transcriptomics_rows: list[dict[str, str]] | None = None,
    cross_species_rows: list[dict[str, str]] | None = None,
) -> None:
    """Write a minimal reference-format fixture to dst.

    Args:
        dst: Target directory for the three TSV files.
        morphology_rows: Override morphology rows; defaults to 3 rows.
        transcriptomics_rows: Override transcriptomics rows; defaults to 4 rows.
        cross_species_rows: Override cross-species rows; defaults to 14 rows.
    """
    dst.mkdir(parents=True, exist_ok=True)
    morph = morphology_rows if morphology_rows is not None else _DEFAULT_MORPHOLOGY
    trans = transcriptomics_rows if transcriptomics_rows is not None else _DEFAULT_TRANSCRIPTOMICS
    cross = cross_species_rows if cross_species_rows is not None else _default_cross_species()

    pd.DataFrame(morph).to_csv(dst / "concept_bank_morphology.tsv", sep="\t", index=False)
    pd.DataFrame(trans).to_csv(dst / "concept_bank_transcriptomics.tsv", sep="\t", index=False)
    pd.DataFrame(cross).to_csv(dst / "concept_bank_cross_species.tsv", sep="\t", index=False)


def write_bad_duplicate_id(dst: Path) -> None:
    """Fixture: duplicate concept_id in morphology."""
    duped = [*_DEFAULT_MORPHOLOGY, _DEFAULT_MORPHOLOGY[0]]
    write_su_fixture(dst, morphology_rows=duped)


def write_bad_missing_in_cross_species(dst: Path) -> None:
    """Fixture: MOR-0001 missing from cross_species entirely."""
    cross = [r for r in _default_cross_species() if r["concept_id"] != "MOR-0001"]
    write_su_fixture(dst, cross_species_rows=cross)


def write_bad_extra_in_cross_species(dst: Path) -> None:
    """Fixture: cross_species has a concept_id not in concepts."""
    extra = [
        *_default_cross_species(),
        {
            "concept_id": "MOR-9999",
            "concept_name": "ghost concept",
            "hierarchy": "H1_morphology",
            "species": "Homo_sapiens",
            "prompt_text": "A ghost.",
            "species_status": "shared",
        },
        {
            "concept_id": "MOR-9999",
            "concept_name": "ghost concept",
            "hierarchy": "H1_morphology",
            "species": "Mus_musculus",
            "prompt_text": "A ghost.",
            "species_status": "shared",
        },
    ]
    write_su_fixture(dst, cross_species_rows=extra)


def write_bad_name_mismatch(dst: Path) -> None:
    """Fixture: concept_name in cross_species doesn't match concepts."""
    cross = [
        {**r, "concept_name": "WRONG NAME"} if r["concept_id"] == "MOR-0001" else r
        for r in _default_cross_species()
    ]
    write_su_fixture(dst, cross_species_rows=cross)


def write_bad_tier_mismatch(dst: Path) -> None:
    """Fixture: hierarchy in cross_species doesn't match actual tier."""
    cross = [
        {**r, "hierarchy": "H2_cell_type"} if r["concept_id"] == "MOR-0001" else r
        for r in _default_cross_species()
    ]
    write_su_fixture(dst, cross_species_rows=cross)


def write_bad_human_only_one_species_shared(dst: Path) -> None:
    """Fixture: MOR-0003 is human_only for human but shared for mouse."""
    cross = []
    for r in _default_cross_species():
        if r["concept_id"] == _HUMAN_ONLY_ID and r["species"] == "Mus_musculus":
            cross.append({**r, "species_status": "shared"})
        else:
            cross.append(r)
    write_su_fixture(dst, cross_species_rows=cross)


def write_bad_h1_missing_organ(dst: Path) -> None:
    """Fixture: H1 row with empty organ field."""
    morph = [{**_DEFAULT_MORPHOLOGY[0], "organ": ""}, *_DEFAULT_MORPHOLOGY[1:]]
    write_su_fixture(dst, morphology_rows=morph)


def write_bad_h2_wrong_concept_type(dst: Path) -> None:
    """Fixture: CTY row with concept_type='pathway'."""
    trans = [
        {**_DEFAULT_TRANSCRIPTOMICS[0], "concept_type": "pathway"},
        *_DEFAULT_TRANSCRIPTOMICS[1:],
    ]
    write_su_fixture(dst, transcriptomics_rows=trans)


def write_bad_invalid_level(dst: Path) -> None:
    """Fixture: H1 row with level='invalid'."""
    morph = [{**_DEFAULT_MORPHOLOGY[0], "level": "invalid"}, *_DEFAULT_MORPHOLOGY[1:]]
    write_su_fixture(dst, morphology_rows=morph)
