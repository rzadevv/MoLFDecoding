"""Stage 7: cross-species prompt generation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from molf_interp.curation.config import CurationConfig
from molf_interp.io.config import BaseConfig, load_config


class PromptTemplates(BaseConfig):
    """Loaded prompt templates matching the generate_cross_species.py format exactly."""

    h1_by_category: dict[str, str]
    h2_cell_type_with_markers: str
    h2_cell_type_without_markers: str
    h2_niche: str
    h2_gene_program_with_genes: str
    h2_gene_program_without_genes: str
    h2_pathway: str
    species_word: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> PromptTemplates:
        """Load and validate templates from a YAML file."""
        return load_config(path, cls)


def _str_or_none(val: object) -> str | None:
    """Return string value or None, treating NaN/empty as None."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def _markers_for(row: pd.Series, species: str) -> str | None:  # type: ignore[type-arg]
    """Get marker genes for a cell type concept.

    Our automated harvest does not produce marker genes. Returns None for all
    rows in this pass. Future work: integrate CellMarker 2.0 or PanglaoDB.
    """
    return None


def _genes_for(row: pd.Series, species: str) -> str | None:  # type: ignore[type-arg]
    """Get representative genes for a gene program concept.

    Only MSigDB sources have gene lists (in extra["genes"], comma-separated).
    Returns the first 5 genes. For Mus_musculus applies naive sentence-casing
    (FOXP3 → Foxp3) as an approximation of mouse gene symbol style.

    Note: true human→mouse ortholog mapping is not performed; this is approximate.
    """
    extra: Any = row.get("extra")
    if extra is None:
        return None
    if not isinstance(extra, list | np.ndarray):
        return None
    for entry in extra:
        if not isinstance(entry, list | np.ndarray) or len(entry) < 2:
            continue
        if str(entry[0]) == "genes":
            raw = str(entry[1]).strip()
            if not raw:
                return None
            genes = [g.strip() for g in raw.split(",") if g.strip()][:5]
            if not genes:
                return None
            if species == "Mus_musculus":
                # Approximate human→mouse gene symbol casing: Uppercase first letter,
                # lowercase the rest. This is an approximation — true ortholog mapping
                # (e.g. TP53→Trp53) requires a lookup table. We guard against empty
                # symbols and single-char edge cases.
                genes = [g[0].upper() + g[1:].lower() if len(g) > 1 else g.upper() for g in genes]
            return ", ".join(genes)
    return None


def _build_prompt(
    row: pd.Series,  # type: ignore[type-arg]
    species: str,
    templates: PromptTemplates,
) -> str:
    """Build one prompt for one (concept, species) pair.

    Mirrors generate_cross_species.py branching:
      H1: branch on row.category
      H2_cell_type: branch on markers availability
      H2_gene_program: branch on representative genes availability
      H2_niche, H2_pathway: single template each
    """
    sw = templates.species_word.get(species, species)
    name = str(row["preferred_label"])
    tier = str(row["tier"])

    if tier == "H1_morphology":
        category = str(row.get("category") or "default")
        if "default" not in templates.h1_by_category:
            raise KeyError("PromptTemplates.h1_by_category must contain a 'default' key")
        tmpl = templates.h1_by_category.get(category, templates.h1_by_category["default"])
        organ = _str_or_none(row.get("organ")) or "universal"
        return tmpl.format(sw=sw, name=name, organ=organ)

    if tier == "H2_cell_type":
        markers = _markers_for(row, species)
        if markers:
            return templates.h2_cell_type_with_markers.format(sw=sw, name=name, markers=markers)
        return templates.h2_cell_type_without_markers.format(sw=sw, name=name)

    if tier == "H2_gene_program":
        genes = _genes_for(row, species)
        if genes:
            return templates.h2_gene_program_with_genes.format(sw=sw, name=name, genes=genes)
        return templates.h2_gene_program_without_genes.format(sw=sw, name=name)

    if tier == "H2_niche":
        return templates.h2_niche.format(sw=sw, name=name)

    if tier == "H2_pathway":
        return templates.h2_pathway.format(sw=sw, name=name)

    raise ValueError(f"Unknown tier: {tier}")


def run_stage_7(df: pd.DataFrame, config: CurationConfig) -> pd.DataFrame:
    """Generate cross-species prompts for all concepts.

    Args:
        df: DataFrame from Stage 6 with 'concept_id', 'preferred_label', 'tier',
            'organ', 'category', 'extra' columns.
        config: CurationConfig instance.

    Returns:
        Long-format DataFrame with one row per (concept, species).
    """
    t0 = time.monotonic()
    templates = PromptTemplates.load(config.prompt_gen.templates_path)
    species_list = list(config.prompt_gen.species)
    logger.info(
        "Stage 7: generating prompts for {} concepts x {} species…",
        len(df),
        len(species_list),
    )

    rows: list[dict[str, object]] = []
    for _, concept in df.iterrows():
        cid = str(concept["concept_id"])
        name = str(concept["preferred_label"])
        tier = str(concept["tier"])
        species_status = "shared"

        for sp in species_list:
            try:
                prompt_text = _build_prompt(concept, sp, templates)
            except (ValueError, KeyError) as exc:
                logger.warning("Could not render prompt for {}/{}: {}", cid, sp, exc)
                prompt_text = name

            rows.append(
                {
                    "concept_id": cid,
                    "concept_name": name,
                    "tier": tier,
                    "species": sp,
                    "prompt_text": prompt_text,
                    "species_status": species_status,
                }
            )

    result = pd.DataFrame(rows)
    elapsed = time.monotonic() - t0
    logger.info("Stage 7 done: {} prompt rows, {:.1f}s", len(result), elapsed)
    return result
