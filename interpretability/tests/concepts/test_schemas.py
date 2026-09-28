import pytest
from pydantic import ValidationError

from molf_interp.concepts.schemas import (
    Concept,
    ConceptBank,
    ConceptPrompt,
    ProvenanceSource,
    Species,
    SpeciesStatus,
    Tier,
)


def _h1(**kw):
    base = dict(
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
    base.update(kw)
    return base


def _h2(tier=Tier.H2_CELL_TYPE, cid="CTY-0001", ct="cell_type", **kw):
    base = dict(
        concept_id=cid,
        concept_name="CD8+ T cell",
        tier=tier,
        category="immune",
        subcategory="cytotoxic",
        source="HCA",
        provenance=ProvenanceSource.REFERENCE,
        concept_type=ct,
    )
    base.update(kw)
    return base


def _prompt(**kw):
    base = dict(
        concept_id="MOR-0001",
        concept_name="coagulative necrosis",
        tier=Tier.H1_MORPHOLOGY,
        species=Species.HOMO_SAPIENS,
        prompt_text="Describe coagulative necrosis.",
        species_status=SpeciesStatus.SHARED,
    )
    base.update(kw)
    return base


def _bank(n_concepts=1, n_human_only=0):
    concepts = [Concept(**_h1())]
    prompts = [
        ConceptPrompt(**_prompt()),
        ConceptPrompt(**_prompt(species=Species.MUS_MUSCULUS)),
    ]
    return ConceptBank(
        concepts=tuple(concepts),
        prompts=tuple(prompts),
        provenance=ProvenanceSource.REFERENCE,
    )


def test_concept_h1_requires_organ_and_level():
    with pytest.raises(ValidationError, match="organ"):
        Concept(**_h1(organ=None))


def test_concept_h1_rejects_invalid_level():
    with pytest.raises(ValidationError):
        Concept(**_h1(level="invalid"))


def test_concept_h2_requires_concept_type():
    with pytest.raises(ValidationError, match="concept_type"):
        Concept(**_h2(ct=None))


def test_concept_h2_concept_type_must_match_tier():
    with pytest.raises(ValidationError):
        Concept(**_h2(tier=Tier.H2_CELL_TYPE, ct="pathway"))


def test_concept_id_pattern_h1():
    with pytest.raises(ValidationError):
        Concept(**_h1(concept_id="X-0001"))


def test_concept_id_pattern_h2_prefix_matches_tier():
    with pytest.raises(ValidationError):
        Concept(**_h2(tier=Tier.H2_PATHWAY, cid="CTY-0001", ct="pathway"))


def test_concept_is_frozen():
    c = Concept(**_h1())
    with pytest.raises(ValidationError):
        c.concept_name = "mutated"  # type: ignore[misc]


def test_concept_prompt_empty_text_raises():
    with pytest.raises(ValidationError):
        ConceptPrompt(**_prompt(prompt_text="   "))


def test_concept_bank_filter_concepts_by_tier():
    c1 = Concept(**_h1())
    c2 = Concept(**_h2())
    p1 = ConceptPrompt(**_prompt())
    p1m = ConceptPrompt(**_prompt(species=Species.MUS_MUSCULUS))
    p2 = ConceptPrompt(
        concept_id="CTY-0001",
        concept_name="CD8+ T cell",
        tier=Tier.H2_CELL_TYPE,
        species=Species.HOMO_SAPIENS,
        prompt_text="Describe CD8+ T cell.",
        species_status=SpeciesStatus.SHARED,
    )
    p2m = ConceptPrompt(
        concept_id="CTY-0001",
        concept_name="CD8+ T cell",
        tier=Tier.H2_CELL_TYPE,
        species=Species.MUS_MUSCULUS,
        prompt_text="Describe CD8+ T cell in mouse.",
        species_status=SpeciesStatus.SHARED,
    )
    bank = ConceptBank(
        concepts=(c1, c2),
        prompts=(p1, p1m, p2, p2m),
        provenance=ProvenanceSource.REFERENCE,
    )
    result = bank.filter_concepts(tier=Tier.H1_MORPHOLOGY)
    assert len(result) == 1
    assert result[0].concept_id == "MOR-0001"


def test_concept_bank_filter_prompts_excludes_human_only_mouse_by_default():
    c = Concept(**_h1())
    ph = ConceptPrompt(**_prompt(species_status=SpeciesStatus.HUMAN_ONLY))
    pm = ConceptPrompt(
        **_prompt(species=Species.MUS_MUSCULUS, species_status=SpeciesStatus.HUMAN_ONLY)
    )
    bank = ConceptBank(concepts=(c,), prompts=(ph, pm), provenance=ProvenanceSource.REFERENCE)
    result = bank.filter_prompts()
    assert len(result) == 1
    assert result[0].species == Species.HOMO_SAPIENS


def test_concept_bank_filter_prompts_includes_human_only_mouse_when_flag_false():
    c = Concept(**_h1())
    ph = ConceptPrompt(**_prompt(species_status=SpeciesStatus.HUMAN_ONLY))
    pm = ConceptPrompt(
        **_prompt(species=Species.MUS_MUSCULUS, species_status=SpeciesStatus.HUMAN_ONLY)
    )
    bank = ConceptBank(concepts=(c,), prompts=(ph, pm), provenance=ProvenanceSource.REFERENCE)
    result = bank.filter_prompts(exclude_human_only_for_mouse=False)
    assert len(result) == 2


def test_concept_bank_get_concept_raises_keyerror_on_miss():
    bank = _bank()
    with pytest.raises(KeyError):
        bank.get_concept("DOES-NOT-EXIST")


def test_concept_bank_summary_counts_match():
    bank = _bank()
    s = bank.summary()
    assert s["totals"]["concepts"] == 1
    assert s["totals"]["prompts"] == 2
    assert s["by_tier"]["H1_morphology"] == 1


def test_concept_bank_len_returns_concept_count_not_prompt_count():
    bank = _bank()
    assert len(bank) == 1
