import pytest

from molf_interp.concepts.ingesters.reference_tsv import load_reference_bank
from molf_interp.concepts.schemas import (
    Concept,
    ConceptPrompt,
    ConsistencyError,
    ProvenanceSource,
    Species,
    SpeciesStatus,
    Tier,
)
from molf_interp.concepts.validation import (
    _check_cross_reference,
    _check_human_only_consistency,
    _check_name_consistency,
    _check_species_pairing,
    _check_tier_consistency,
    _check_unique_concept_ids,
    _check_unique_prompt_keys,
)
from tests.concepts.conftest import (
    write_bad_duplicate_id,
    write_bad_extra_in_cross_species,
    write_bad_human_only_one_species_shared,
    write_bad_missing_in_cross_species,
    write_bad_name_mismatch,
    write_bad_tier_mismatch,
)


def _c(cid="MOR-0001", name="necrosis", tier=Tier.H1_MORPHOLOGY):
    return Concept(
        concept_id=cid,
        concept_name=name,
        tier=tier,
        category="cat",
        subcategory="sub",
        source="src",
        provenance=ProvenanceSource.REFERENCE,
        organ="universal",
        level="tissue",
    )


def _p(
    cid="MOR-0001",
    name="necrosis",
    tier=Tier.H1_MORPHOLOGY,
    sp=Species.HOMO_SAPIENS,
    status=SpeciesStatus.SHARED,
):
    return ConceptPrompt(
        concept_id=cid,
        concept_name=name,
        tier=tier,
        species=sp,
        prompt_text=f"Describe {name}.",
        species_status=status,
    )


def _both_species(
    cid="MOR-0001", name="necrosis", tier=Tier.H1_MORPHOLOGY, status=SpeciesStatus.SHARED
):
    return [
        _p(cid, name, tier, Species.HOMO_SAPIENS, status),
        _p(cid, name, tier, Species.MUS_MUSCULUS, status),
    ]


# --- unit tests for each private check ---


def test_check_unique_concept_ids_ok():
    _check_unique_concept_ids([_c("MOR-0001"), _c("MOR-0002")])


def test_check_unique_concept_ids_fails():
    with pytest.raises(ConsistencyError, match="MOR-0001"):
        _check_unique_concept_ids([_c("MOR-0001"), _c("MOR-0001")])


def test_check_unique_prompt_keys_ok():
    _check_unique_prompt_keys(_both_species())


def test_check_unique_prompt_keys_fails():
    duped = [_p(), _p()]
    with pytest.raises(ConsistencyError, match="MOR-0001"):
        _check_unique_prompt_keys(duped)


def test_check_cross_reference_ok():
    _check_cross_reference([_c()], _both_species())


def test_check_cross_reference_missing_in_prompts():
    with pytest.raises(ConsistencyError, match="MOR-0001"):
        _check_cross_reference([_c()], [])


def test_check_cross_reference_extra_in_prompts():
    with pytest.raises(ConsistencyError, match="MOR-0001"):
        _check_cross_reference([], _both_species())


def test_check_name_consistency_ok():
    _check_name_consistency([_c()], _both_species())


def test_check_name_consistency_fails():
    with pytest.raises(ConsistencyError, match="MOR-0001"):
        _check_name_consistency([_c()], _both_species(name="WRONG"))


def test_check_tier_consistency_ok():
    _check_tier_consistency([_c()], _both_species())


def test_check_tier_consistency_fails():
    c = Concept(
        concept_id="CTY-0001",
        concept_name="x",
        tier=Tier.H2_CELL_TYPE,
        category="c",
        subcategory="s",
        source="s",
        provenance=ProvenanceSource.REFERENCE,
        concept_type="cell_type",
    )
    bad_prompts = [
        _p("CTY-0001", "x", Tier.H2_PATHWAY, Species.HOMO_SAPIENS),
        _p("CTY-0001", "x", Tier.H2_PATHWAY, Species.MUS_MUSCULUS),
    ]
    with pytest.raises(ConsistencyError, match="CTY-0001"):
        _check_tier_consistency([c], bad_prompts)


def test_check_species_pairing_ok():
    _check_species_pairing(_both_species())


def test_check_species_pairing_fails_missing_mouse():
    with pytest.raises(ConsistencyError, match="MOR-0001"):
        _check_species_pairing([_p()])


def test_check_human_only_consistency_ok():
    _check_human_only_consistency(_both_species(status=SpeciesStatus.HUMAN_ONLY))


def test_check_human_only_consistency_fails():
    mixed = [
        _p(status=SpeciesStatus.HUMAN_ONLY),
        _p(sp=Species.MUS_MUSCULUS, status=SpeciesStatus.SHARED),
    ]
    with pytest.raises(ConsistencyError, match="MOR-0001"):
        _check_human_only_consistency(mixed)


# --- bad-scenario integration tests ---


@pytest.mark.parametrize(
    "bad_writer,match",
    [
        (write_bad_duplicate_id, "MOR-0001"),
        (write_bad_missing_in_cross_species, "MOR-0001"),
        (write_bad_extra_in_cross_species, "MOR-9999"),
        (write_bad_name_mismatch, "MOR-0001"),
        (write_bad_tier_mismatch, "MOR-0001"),
        (write_bad_human_only_one_species_shared, "MOR-0003"),
    ],
)
def test_each_bad_scenario_raises_consistency_error(bad_writer, match, tmp_path):
    bad_writer(tmp_path)
    with pytest.raises(ConsistencyError, match=match):
        load_reference_bank(tmp_path)
