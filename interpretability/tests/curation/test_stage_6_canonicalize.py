"""Tests for Stage 6: canonicalize."""

from __future__ import annotations

import pandas as pd
import pytest

from molf_interp.curation.config import CanonicalizeConfig, CurationConfig
from molf_interp.curation.stage_6_canonicalize import _canonicalize_label, run_stage_6


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_name": ["ncit", "snomed"],
            "preferred_label": ["Coagulative Necrosis", "Fibrosis"],
            "synonyms": [["Necrosis"], []],
            "tier": ["H1_morphology", "H1_morphology"],
            "organ": ["lung", "liver"],
            "level": ["tissue", "tissue"],
            "concept_type": [None, None],
        }
    )


def test_ids_start_at_offset() -> None:
    cfg = CurationConfig(canonicalize=CanonicalizeConfig(id_offset=10000, sentence_case=False))
    df = _make_df()
    result = run_stage_6(df, cfg)
    ids = sorted(result["concept_id"].tolist())
    assert all(cid.startswith("MOR-") for cid in ids)
    numbers = [int(cid.split("-")[1]) for cid in ids]
    assert min(numbers) == 10001


def test_sentence_case_applied() -> None:
    cfg = CurationConfig(canonicalize=CanonicalizeConfig(id_offset=10000, sentence_case=True))
    df = _make_df()
    result = run_stage_6(df, cfg)
    labels = result["preferred_label"].tolist()
    # "Coagulative Necrosis" → "Coagulative necrosis" (first char uppercase, rest lower)
    assert all(lbl[0].isupper() or not lbl[0].isalpha() for lbl in labels)
    # Non-first words should be lowercased (no internal capitals from title-case source)
    for lbl in labels:
        words = lbl.split()
        if len(words) > 1:
            assert words[1][0].islower(), f"Internal capital survived in: {lbl!r}"


def test_synonym_dedup_case_insensitive() -> None:
    cfg = CurationConfig(canonicalize=CanonicalizeConfig(id_offset=10000))
    df = pd.DataFrame(
        {
            "source_name": ["ncit"],
            "preferred_label": ["necrosis"],
            "synonyms": [["Necrosis", "NECROSIS", "necrosis", "cell death"]],
            "tier": ["H1_morphology"],
            "organ": ["lung"],
            "level": ["tissue"],
            "concept_type": [None],
        }
    )
    result = run_stage_6(df, cfg)
    syns = result.iloc[0]["synonyms"]
    lower_syns = [s.lower() for s in syns]
    assert len(lower_syns) == len(set(lower_syns))


def test_stage_6_drops_high_confidence_clinical_h1_noise() -> None:
    cfg = CurationConfig(canonicalize=CanonicalizeConfig(id_offset=10000))
    df = pd.DataFrame(
        {
            "source_name": ["ncit", "ncit", "snomed", "ncit", "ncit", "snomed"],
            "preferred_label": [
                "Grade 2 clear cell renal cell carcinoma",
                "Microsatellite stable endometrial carcinoma",
                "Squamous intraepithelial neoplasia grade 2",
                "Her2/neu expression by immunohistochemistry 3+",
                "Necrotic area imaging finding",
                "Bronchial surgical margin",
            ],
            "synonyms": [[] for _ in range(6)],
            "tier": ["H1_morphology" for _ in range(6)],
            "organ": ["universal" for _ in range(6)],
            "level": ["tissue" for _ in range(6)],
            "concept_type": [None for _ in range(6)],
        }
    )

    result = run_stage_6(df, cfg)

    assert result.empty


def test_stage_6_keeps_valid_morphology_and_plain_disease_pattern_labels() -> None:
    cfg = CurationConfig(canonicalize=CanonicalizeConfig(id_offset=10000))
    df = pd.DataFrame(
        {
            "source_name": ["reference", "reference", "reference", "reference"],
            "preferred_label": [
                "Squamous intraepithelial neoplasia",
                "Tumor-stroma interface",
                "Tumor budding high grade",
                "Coagulative necrosis",
            ],
            "synonyms": [[] for _ in range(4)],
            "tier": ["H1_morphology" for _ in range(4)],
            "organ": ["universal" for _ in range(4)],
            "level": ["tissue" for _ in range(4)],
            "concept_type": [None for _ in range(4)],
        }
    )

    result = run_stage_6(df, cfg)

    assert set(result["preferred_label"]) == {
        "Squamous intraepithelial neoplasia",
        "Tumor-stroma interface",
        "Tumor budding high grade",
        "Coagulative necrosis",
    }


def test_ids_deterministic() -> None:
    cfg = CurationConfig(canonicalize=CanonicalizeConfig(id_offset=10000))
    df = _make_df()
    result1 = run_stage_6(df, cfg)
    result2 = run_stage_6(df, cfg)
    assert result1["concept_id"].tolist() == result2["concept_id"].tolist()


# _canonicalize_label unit tests


@pytest.mark.parametrize(
    "inp, expected",
    [
        ("pulmonary Infiltrate", "Pulmonary infiltrate"),
        ("breast Columnar Cell", "Breast columnar cell"),
        ("CD8-positive T cell", "CD8-positive T cell"),
        ("BRCA1 germline mutation", "BRCA1 germline mutation"),
        (
            "gastric Mixed Adenoneuroendocrine Carcinoma",
            "Gastric mixed adenoneuroendocrine carcinoma",
        ),
        ("RCC of clear cell type", "RCC of clear cell type"),
        ("necrosis", "Necrosis"),
        ("", ""),
    ],
)
def test_canonicalize_label_cases(inp: str, expected: str) -> None:
    assert _canonicalize_label(inp) == expected


def test_canonicalize_label_pd_l1() -> None:
    assert _canonicalize_label("PD-L1 expression") == "PD-L1 expression"


def test_canonicalize_label_nf_kb() -> None:
    assert _canonicalize_label("NF-kB signaling pathway") == "NF-kB signaling pathway"


def test_canonicalize_label_b_cell() -> None:
    assert _canonicalize_label("b cell lineage commitment") == "B cell lineage commitment"


def test_canonicalize_label_foxp3() -> None:
    assert _canonicalize_label("FOXP3 expression in regulatory t cells") == (
        "FoxP3 expression in regulatory T cells"
    )
