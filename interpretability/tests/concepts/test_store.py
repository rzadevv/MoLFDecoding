import pandas as pd
import pyarrow.parquet as pq
import pytest

from molf_interp.concepts.schemas import (
    Concept,
    ConceptBank,
    ConceptPrompt,
    ProvenanceSource,
    Species,
    SpeciesStatus,
    Tier,
)
from molf_interp.concepts.store import (
    _CONCEPTS_SCHEMA,
    _PROMPTS_SCHEMA,
    migrate_provenance_values,
    read_concept_bank,
    write_concept_bank,
)


def _small_bank() -> ConceptBank:
    c = Concept(
        concept_id="MOR-0001",
        concept_name="coagulative necrosis",
        tier=Tier.H1_MORPHOLOGY,
        category="necrosis",
        subcategory="tumor",
        source="Robbins",
        provenance=ProvenanceSource.REFERENCE,
        organ="universal",
        level="tissue",
    )
    ph = ConceptPrompt(
        concept_id="MOR-0001",
        concept_name="coagulative necrosis",
        tier=Tier.H1_MORPHOLOGY,
        species=Species.HOMO_SAPIENS,
        prompt_text="Describe coagulative necrosis in humans.",
        species_status=SpeciesStatus.SHARED,
    )
    pm = ConceptPrompt(
        concept_id="MOR-0001",
        concept_name="coagulative necrosis",
        tier=Tier.H1_MORPHOLOGY,
        species=Species.MUS_MUSCULUS,
        prompt_text="Describe coagulative necrosis in mouse.",
        species_status=SpeciesStatus.SHARED,
    )
    return ConceptBank(concepts=(c,), prompts=(ph, pm), provenance=ProvenanceSource.REFERENCE)


def test_write_then_read_preserves_bank(tmp_path):
    bank = _small_bank()
    write_concept_bank(bank, tmp_path)
    restored = read_concept_bank(tmp_path)
    assert len(restored) == len(bank)
    assert len(restored.prompts) == len(bank.prompts)
    assert restored.provenance == bank.provenance
    assert restored.concepts[0].concept_id == bank.concepts[0].concept_id
    assert restored.concepts[0].organ == bank.concepts[0].organ


def test_overwrite_existing_files(tmp_path):
    bank = _small_bank()
    write_concept_bank(bank, tmp_path)
    write_concept_bank(bank, tmp_path)
    restored = read_concept_bank(tmp_path)
    assert len(restored) == 1


def test_read_rejects_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_concept_bank(tmp_path)


def test_read_rejects_missing_prompts(tmp_path):
    bank = _small_bank()
    write_concept_bank(bank, tmp_path)
    (tmp_path / "prompts.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        read_concept_bank(tmp_path)


def test_parquet_schema_is_stable(tmp_path):
    bank = _small_bank()
    write_concept_bank(bank, tmp_path)
    ct = pq.read_table(tmp_path / "concepts.parquet")
    pt = pq.read_table(tmp_path / "prompts.parquet")
    assert ct.schema.equals(_CONCEPTS_SCHEMA)
    assert pt.schema.equals(_PROMPTS_SCHEMA)


def test_migrate_legacy_su_pathologist():
    df = pd.DataFrame({"provenance": ["su_pathologist", "su_pathologist"]})
    result = migrate_provenance_values(df)
    assert list(result["provenance"]) == ["reference", "reference"]


def test_migrate_legacy_automated_harvest():
    df = pd.DataFrame({"provenance": ["automated_harvest"]})
    result = migrate_provenance_values(df)
    assert list(result["provenance"]) == ["harvested"]


def test_migrate_current_values_pass_through():
    df = pd.DataFrame({"provenance": ["reference", "harvested"]})
    result = migrate_provenance_values(df)
    assert list(result["provenance"]) == ["reference", "harvested"]


def test_migrate_unknown_value_raises():
    df = pd.DataFrame({"provenance": ["unknown_source"]})
    with pytest.raises(ValueError, match="Unrecognised"):
        migrate_provenance_values(df)
