from pathlib import Path

import pytest
from pydantic import ValidationError

from molf_interp.concepts.ingesters.reference_tsv import load_reference_bank
from molf_interp.concepts.schemas import ConsistencyError, ProvenanceSource
from tests.concepts.conftest import (
    write_bad_duplicate_id,
    write_bad_extra_in_cross_species,
    write_bad_human_only_one_species_shared,
    write_bad_missing_in_cross_species,
    write_bad_name_mismatch,
    write_bad_tier_mismatch,
    write_su_fixture,
)


def test_load_good_fixture_returns_expected_counts(tmp_path):
    write_su_fixture(tmp_path)
    bank = load_reference_bank(tmp_path)
    assert len(bank) == 7
    assert len(bank.prompts) == 14


def test_load_missing_file_raises(tmp_path):
    write_su_fixture(tmp_path)
    (tmp_path / "concept_bank_morphology.tsv").unlink()
    with pytest.raises(FileNotFoundError):
        load_reference_bank(tmp_path)


def test_load_corrupts_into_pydantic_validation_error(tmp_path):
    from tests.concepts.conftest import write_bad_h1_missing_organ

    write_bad_h1_missing_organ(tmp_path)
    with pytest.raises((ValidationError, Exception)):
        load_reference_bank(tmp_path)


@pytest.mark.parametrize(
    "bad_writer",
    [
        write_bad_duplicate_id,
        write_bad_missing_in_cross_species,
        write_bad_extra_in_cross_species,
        write_bad_name_mismatch,
        write_bad_tier_mismatch,
        write_bad_human_only_one_species_shared,
    ],
)
def test_each_bad_scenario_raises_consistency_error(bad_writer, tmp_path):
    bad_writer(tmp_path)
    with pytest.raises(ConsistencyError):
        load_reference_bank(tmp_path)


def test_load_provenance_set_to_reference(tmp_path):
    write_su_fixture(tmp_path)
    bank = load_reference_bank(tmp_path)
    assert bank.provenance == ProvenanceSource.REFERENCE
    assert all(c.provenance == ProvenanceSource.REFERENCE for c in bank.concepts)


@pytest.mark.integration
def test_su_real_bank_loads_and_matches_expected_counts(tmp_path: Path) -> None:
    """Loads reference bank TSVs from data/concept_bank/. Skips if not present."""
    real_dir = Path("data/concept_bank")
    if not (real_dir / "concept_bank_cross_species.tsv").exists():
        pytest.skip("Reference bank not available")
    bank = load_reference_bank(real_dir)
    assert len(bank) == 1006
    assert len(bank.prompts) == 2012
    summary = bank.summary()
    assert summary["by_tier"]["H1_morphology"] == 418
    assert summary["totals"]["human_only"] == 13
