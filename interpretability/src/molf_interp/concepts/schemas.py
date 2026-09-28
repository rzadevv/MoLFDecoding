"""Pydantic models and enums for the unified biomedical concept representation."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import field_validator, model_validator

from molf_interp.io.config import BaseConfig


class Tier(StrEnum):
    """Five-tier concept hierarchy for single-patch H&E interpretation."""

    H1_MORPHOLOGY = "H1_morphology"
    H2_CELL_TYPE = "H2_cell_type"
    H2_NICHE = "H2_niche"
    H2_GENE_PROGRAM = "H2_gene_program"
    H2_PATHWAY = "H2_pathway"


class Species(StrEnum):
    """Supported species for cross-species concept prompts."""

    HOMO_SAPIENS = "Homo_sapiens"
    MUS_MUSCULUS = "Mus_musculus"


class SpeciesStatus(StrEnum):
    """Whether a concept is biologically meaningful in mouse."""

    SHARED = "shared"
    HUMAN_ONLY = "human_only"


class ProvenanceSource(StrEnum):
    """Producer of a concept. Extend as new producers are added."""

    REFERENCE = "reference"
    HARVESTED = "harvested"


_MORPHOLOGY_ID_PATTERN = re.compile(r"^MOR-\d{4,}$")
_TRANSCRIPTOMICS_ID_PATTERNS: dict[Tier, re.Pattern[str]] = {
    Tier.H2_CELL_TYPE: re.compile(r"^CTY-\d{4,}$"),
    Tier.H2_NICHE: re.compile(r"^NIC-\d{4,}$"),
    Tier.H2_GENE_PROGRAM: re.compile(r"^GPR-\d{4,}$"),
    Tier.H2_PATHWAY: re.compile(r"^PWY-\d{4,}$"),
}
_ALLOWED_LEVELS: frozenset[str] = frozenset(
    {"tissue", "cellular", "subcellular", "extracellular", "stromal"}
)
_H2_CONCEPT_TYPE_BY_TIER: dict[Tier, str] = {
    Tier.H2_CELL_TYPE: "cell_type",
    Tier.H2_NICHE: "niche",
    Tier.H2_GENE_PROGRAM: "gene_program",
    Tier.H2_PATHWAY: "pathway",
}


class Concept(BaseConfig):
    """A single biomedical concept in the unified format.

    Concepts at tier H1_MORPHOLOGY must specify organ + level. Concepts at any H2 tier
    must specify concept_type matching the tier.
    """

    concept_id: str
    concept_name: str
    tier: Tier
    category: str
    subcategory: str
    source: str
    provenance: ProvenanceSource
    organ: str | None = None
    level: str | None = None
    concept_type: str | None = None

    @model_validator(mode="after")
    def _check_tier_fields(self) -> Concept:
        cid = self.concept_id
        if self.tier == Tier.H1_MORPHOLOGY:
            if not _MORPHOLOGY_ID_PATTERN.match(cid):
                raise ValueError(
                    f"concept_id {cid!r} does not match MOR-NNNN pattern for H1_MORPHOLOGY"
                )
            if not self.organ:
                raise ValueError(f"organ is required for H1_MORPHOLOGY (concept_id={cid!r})")
            if not self.level:
                raise ValueError(f"level is required for H1_MORPHOLOGY (concept_id={cid!r})")
            if self.level not in _ALLOWED_LEVELS:
                raise ValueError(
                    f"level={self.level!r} is not in {sorted(_ALLOWED_LEVELS)} (concept_id={cid!r})"
                )
        else:
            pattern = _TRANSCRIPTOMICS_ID_PATTERNS.get(self.tier)
            if pattern and not pattern.match(cid):
                raise ValueError(
                    f"concept_id {cid!r} does not match expected pattern for {self.tier.value}"
                )
            expected_type = _H2_CONCEPT_TYPE_BY_TIER.get(self.tier)
            if expected_type is not None:
                if self.concept_type is None:
                    raise ValueError(
                        f"concept_type is required for {self.tier.value} (concept_id={cid!r})"
                    )
                if self.concept_type != expected_type:
                    raise ValueError(
                        f"concept_type={self.concept_type!r} does not match"
                        f" expected {expected_type!r} for {self.tier.value}"
                        f" (concept_id={cid!r})"
                    )
        return self


class ConceptPrompt(BaseConfig):
    """A VLM-ready contextualized prompt for one concept in one species."""

    concept_id: str
    concept_name: str
    tier: Tier
    species: Species
    prompt_text: str
    species_status: SpeciesStatus

    @field_validator("prompt_text")
    @classmethod
    def _prompt_text_non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("prompt_text must be non-empty after stripping")
        return v


class ConsistencyError(ValueError):
    """Raised when concepts and prompts are not internally consistent."""


class ConceptBank(BaseConfig):
    """An immutable collection of concepts and their cross-species prompts."""

    concepts: tuple[Concept, ...]
    prompts: tuple[ConceptPrompt, ...]
    provenance: ProvenanceSource

    def filter_concepts(
        self,
        *,
        tier: Tier | None = None,
        organ: str | None = None,
        concept_type: str | None = None,
    ) -> tuple[Concept, ...]:
        """Filter concepts by optional tier, organ, or concept_type.

        Args:
            tier: If set, keep only concepts with this tier.
            organ: If set, keep only concepts with this organ value.
            concept_type: If set, keep only concepts with this concept_type.

        Returns:
            Filtered tuple of Concept objects.
        """
        result = self.concepts
        if tier is not None:
            result = tuple(c for c in result if c.tier == tier)
        if organ is not None:
            result = tuple(c for c in result if c.organ == organ)
        if concept_type is not None:
            result = tuple(c for c in result if c.concept_type == concept_type)
        return result

    def filter_prompts(
        self,
        *,
        tier: Tier | None = None,
        species: Species | None = None,
        exclude_human_only_for_mouse: bool = True,
    ) -> tuple[ConceptPrompt, ...]:
        """Filter prompts by tier and/or species with optional human-only exclusion.

        Args:
            tier: If set, keep only prompts with this tier.
            species: If set, keep only prompts for this species.
            exclude_human_only_for_mouse: When True (default), drops prompts where
                species == Mus_musculus AND species_status == HUMAN_ONLY.

        Returns:
            Filtered tuple of ConceptPrompt objects.
        """
        result = self.prompts
        if tier is not None:
            result = tuple(p for p in result if p.tier == tier)
        if species is not None:
            result = tuple(p for p in result if p.species == species)
        if exclude_human_only_for_mouse:
            result = tuple(
                p
                for p in result
                if not (
                    p.species == Species.MUS_MUSCULUS
                    and p.species_status == SpeciesStatus.HUMAN_ONLY
                )
            )
        return result

    def get_concept(self, concept_id: str) -> Concept:
        """Look up a concept by its ID.

        Args:
            concept_id: The concept identifier to look up.

        Returns:
            The matching Concept.

        Raises:
            KeyError: If concept_id is not in this bank.
        """
        for c in self.concepts:
            if c.concept_id == concept_id:
                return c
        raise KeyError(concept_id)

    def __len__(self) -> int:
        """Return the number of concepts (not prompts)."""
        return len(self.concepts)

    def summary(self) -> dict[str, dict[str, int]]:
        """Return a nested dict with counts by tier, species, provenance, and totals.

        Returns:
            Nested dict with keys: by_tier, by_species, by_provenance, totals.
        """
        by_tier: dict[str, int] = {}
        for c in self.concepts:
            by_tier[c.tier.value] = by_tier.get(c.tier.value, 0) + 1

        by_species: dict[str, int] = {}
        human_only = 0
        for p in self.prompts:
            by_species[p.species.value] = by_species.get(p.species.value, 0) + 1
            if p.species_status == SpeciesStatus.HUMAN_ONLY and p.species == Species.HOMO_SAPIENS:
                human_only += 1

        by_provenance: dict[str, int] = {}
        for c in self.concepts:
            by_provenance[c.provenance.value] = by_provenance.get(c.provenance.value, 0) + 1

        return {
            "by_tier": by_tier,
            "by_species": by_species,
            "by_provenance": by_provenance,
            "totals": {
                "concepts": len(self.concepts),
                "prompts": len(self.prompts),
                "human_only": human_only,
            },
        }
