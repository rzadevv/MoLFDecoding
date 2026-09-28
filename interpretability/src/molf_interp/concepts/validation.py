"""Cross-table consistency checks for concept banks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from molf_interp.concepts.schemas import (
    Concept,
    ConceptPrompt,
    ConsistencyError,
    Species,
    SpeciesStatus,
)

_MAX_EXAMPLES = 5


def _fmt(ids: list[str]) -> str:
    shown = ids[:_MAX_EXAMPLES]
    suffix = f" (and {len(ids) - _MAX_EXAMPLES} more)" if len(ids) > _MAX_EXAMPLES else ""
    return ", ".join(shown) + suffix


def _check_unique_concept_ids(concepts: Sequence[Concept]) -> None:
    counts = Counter(c.concept_id for c in concepts)
    dupes = [cid for cid, n in counts.items() if n > 1]
    if dupes:
        raise ConsistencyError(f"Duplicate concept_ids in concepts: {_fmt(dupes)}")


def _check_unique_prompt_keys(prompts: Sequence[ConceptPrompt]) -> None:
    counts = Counter((p.concept_id, p.species) for p in prompts)
    dupes = [f"{cid}:{sp.value}" for (cid, sp), n in counts.items() if n > 1]
    if dupes:
        raise ConsistencyError(f"Duplicate (concept_id, species) pairs in prompts: {_fmt(dupes)}")


def _check_cross_reference(concepts: Sequence[Concept], prompts: Sequence[ConceptPrompt]) -> None:
    concept_ids = {c.concept_id for c in concepts}
    prompt_ids = {p.concept_id for p in prompts}
    missing_in_prompts = sorted(concept_ids - prompt_ids)
    extra_in_prompts = sorted(prompt_ids - concept_ids)
    msgs = []
    if missing_in_prompts:
        msgs.append(f"concept_ids in concepts but not prompts: {_fmt(missing_in_prompts)}")
    if extra_in_prompts:
        msgs.append(f"concept_ids in prompts but not concepts: {_fmt(extra_in_prompts)}")
    if msgs:
        raise ConsistencyError("; ".join(msgs))


def _check_name_consistency(concepts: Sequence[Concept], prompts: Sequence[ConceptPrompt]) -> None:
    name_by_id = {c.concept_id: c.concept_name for c in concepts}
    mismatches = sorted(
        p.concept_id
        for p in prompts
        if p.concept_id in name_by_id and p.concept_name != name_by_id[p.concept_id]
    )
    if mismatches:
        raise ConsistencyError(
            f"concept_name mismatch between concepts and prompts for: {_fmt(mismatches)}"
        )


def _check_tier_consistency(concepts: Sequence[Concept], prompts: Sequence[ConceptPrompt]) -> None:
    tier_by_id = {c.concept_id: c.tier for c in concepts}
    mismatches = sorted(
        f"{p.concept_id} (concepts={tier_by_id[p.concept_id].value}, prompts={p.tier.value})"
        for p in prompts
        if p.concept_id in tier_by_id and p.tier != tier_by_id[p.concept_id]
    )
    if mismatches:
        raise ConsistencyError(f"tier mismatch between concepts and prompts: {_fmt(mismatches)}")


def _check_species_pairing(prompts: Sequence[ConceptPrompt]) -> None:
    by_id: dict[str, set[Species]] = {}
    for p in prompts:
        by_id.setdefault(p.concept_id, set()).add(p.species)
    expected = set(Species)
    bad = sorted(cid for cid, sp_set in by_id.items() if sp_set != expected)
    if bad:
        raise ConsistencyError(f"concept_ids without exactly one prompt per species: {_fmt(bad)}")


def _check_human_only_consistency(prompts: Sequence[ConceptPrompt]) -> None:
    statuses: dict[str, set[SpeciesStatus]] = {}
    for p in prompts:
        statuses.setdefault(p.concept_id, set()).add(p.species_status)
    mixed = sorted(cid for cid, ss in statuses.items() if len(ss) > 1)
    if mixed:
        raise ConsistencyError(
            f"concept_ids with mixed species_status (must be uniform): {_fmt(mixed)}"
        )


def validate_consistency(
    concepts: Sequence[Concept],
    prompts: Sequence[ConceptPrompt],
) -> None:
    """Run all cross-table consistency checks.

    Args:
        concepts: Sequence of validated Concept objects.
        prompts: Sequence of validated ConceptPrompt objects.

    Raises:
        ConsistencyError: if any check fails, naming the rule and up to 5 offenders.
    """
    _check_unique_concept_ids(concepts)
    _check_unique_prompt_keys(prompts)
    _check_cross_reference(concepts, prompts)
    _check_name_consistency(concepts, prompts)
    _check_tier_consistency(concepts, prompts)
    _check_species_pairing(prompts)
    _check_human_only_consistency(prompts)
