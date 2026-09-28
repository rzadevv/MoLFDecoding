"""Stage 5: tier reassignment + organ/level/category metadata."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

from molf_interp.concepts.schemas import ConceptBank, Tier
from molf_interp.concepts.store import read_concept_bank
from molf_interp.curation.config import CurationConfig
from molf_interp.curation.level_classifier import classify_level
from molf_interp.curation.organ_normalizer import OrganNormalizer
from molf_interp.curation.sapbert import SapBert

_TIER_TO_CONCEPT_TYPE: dict[str, str] = {
    "H2_cell_type": "cell_type",
    "H2_niche": "niche",
    "H2_gene_program": "gene_program",
    "H2_pathway": "pathway",
}

# Word-level matching (exact word, not substring) to avoid false positives from
# short terms appearing inside longer words (e.g. "ink" in "x-linked").
_ARTIFACT_KEYWORDS: frozenset[str] = frozenset(
    {
        "artifact",
        "fixation artifact",
        "crush artifact",
        "autolysis",
        "ink",
        "marker",
        "fold",
        "knife",
        "tear",
        "freezing artifact",
    }
)

# "infiltrat" deliberately excluded — too broad (matches "lymphocytic infiltration"
# which is a normal morphological feature, not a niche/microenvironment term).
_MICROENVIRONMENT_KEYWORDS: frozenset[str] = frozenset(
    {
        "microenvironment",
        "niche",
        "tertiary lymphoid",
        "immune exclus",
        "immune desert",
        "peritumoral",
        "stromal compartment",
    }
)


def _classify_h1_category(
    label: str,
    synonyms: Sequence[str],
    level: str | None,
    organ: str | None,
) -> Literal["organ_specific", "microenvironment", "artifact", "default"]:
    """Classify an H1 morphology concept into one of four categories.

    Strict positive-evidence rules (case-insensitive substring, ordered):
      1. artifact        — any term contains an _ARTIFACT_KEYWORDS phrase.
      2. microenvironment — any term contains a _MICROENVIRONMENT_KEYWORDS phrase.
      3. organ_specific  — organ != "universal" AND level == "tissue".
      4. default         — everything else.

    Args:
        label: Preferred label of the concept.
        synonyms: Synonym strings (may be empty).
        level: Level classification (e.g. "tissue", "cellular"). May be None.
        organ: Normalised organ name. May be None.

    Returns:
        One of "artifact", "microenvironment", "organ_specific", or "default".
    """
    terms = [label, *synonyms]
    for term in terms:
        t_lower = term.lower()
        if any(kw in t_lower for kw in _ARTIFACT_KEYWORDS):
            return "artifact"
    for term in terms:
        t_lower = term.lower()
        if any(kw in t_lower for kw in _MICROENVIRONMENT_KEYWORDS):
            return "microenvironment"
    if organ and organ != "universal" and level == "tissue":
        return "organ_specific"
    return "default"


_ALL_TIERS = list(Tier)
_TIER_VALUES: list[str] = [str(t) for t in _ALL_TIERS]

_DROP_TIERS = frozenset({"organ_vocab", "UNKNOWN"})


def _build_reference_embeddings(
    reference_bank: ConceptBank,
    sapbert: SapBert,
) -> tuple[npt.NDArray[Any], list[str]]:
    """Return embeddings and tier labels for all reference concepts."""
    names = [c.concept_name for c in reference_bank.concepts]
    tiers = [c.tier.value for c in reference_bank.concepts]
    embs = sapbert.encode(names)
    return embs, tiers


def run_stage_5(
    df: pd.DataFrame,
    config: CurationConfig,
    sapbert: SapBert,
    reference_bank: ConceptBank | None = None,
    organ_normalizer: OrganNormalizer | None = None,
) -> pd.DataFrame:
    """Tier reassignment + H1 metadata enrichment.

    Args:
        df: DataFrame from Stage 4.
        config: CurationConfig instance.
        sapbert: SapBert instance for embedding reference concepts.
        reference_bank: Optional pre-loaded ConceptBank; loaded from disk if None.
        organ_normalizer: Optional OrganNormalizer; created from config if None.

    Returns:
        DataFrame with 'tier', 'organ', 'level', 'concept_type' columns added.
    """
    t0 = time.monotonic()
    tr = config.tier_refine

    if reference_bank is None:
        reference_bank = read_concept_bank(config.visual_filter.reference_bank_dir)

    if organ_normalizer is None:
        organ_normalizer = OrganNormalizer(tr.uberon_table_path, sapbert)

    logger.info("Stage 5: tier refinement on {} rows…", len(df))

    if "embedding" not in df.columns:
        raise ValueError("Stage 5 requires 'embedding' column from Stage 1")

    # Drop organ_vocab and UNKNOWN tiers
    before = len(df)
    df = df[~df["candidate_tier"].isin(_DROP_TIERS)].copy()
    dropped_vocab = before - len(df)
    if dropped_vocab:
        logger.info("  Dropped {} organ_vocab/UNKNOWN rows", dropped_vocab)

    ref_embs, ref_tiers = _build_reference_embeddings(reference_bank, sapbert)
    k = tr.reassignment_min_neighbor_count

    # Pre-encode all unique labels + synonyms from H1 rows to avoid per-row cache misses
    h1_mask = df["candidate_tier"] == "H1_morphology"
    unique_texts: list[str] = []
    for _, row in df[h1_mask].iterrows():
        unique_texts.append(str(row["preferred_label"]))
        syns_raw = row.get("synonyms")
        if isinstance(syns_raw, list | np.ndarray):
            unique_texts.extend(str(s) for s in syns_raw)
    unique_texts = list(dict.fromkeys(unique_texts))  # deduplicate preserving order
    if unique_texts:
        logger.info("  Pre-encoding {} unique H1 labels+synonyms…", len(unique_texts))
        sapbert.encode(unique_texts)  # warm cache; single batch encode

    final_tiers: list[str] = []
    organs: list[str | None] = []
    levels: list[str | None] = []
    concept_types: list[str | None] = []
    categories: list[str] = []

    reassigned = 0

    # Vectorised cosine similarity: compute all candidate-vs-reference sims at once
    if len(df) > 0:
        all_embs = np.stack(df["embedding"].values)
        all_candidate_tiers = df["candidate_tier"].astype(str).tolist()
        if ref_embs.shape[0] > 0:
            all_sims: npt.NDArray[Any] = sapbert.cosine_similarity(all_embs, ref_embs)  # [N, M]
        else:
            all_sims = np.zeros((len(df), 0), dtype=np.float32)
    else:
        all_embs = np.zeros((0, sapbert.embedding_dim), dtype=np.float32)
        all_candidate_tiers = []
        all_sims = np.zeros((0, 0), dtype=np.float32)

    # Simple majority threshold: more than half of k neighbors must agree
    majority_threshold = k // 2 + 1

    for row_i, (_, row) in enumerate(df.iterrows()):
        current_tier = all_candidate_tiers[row_i]

        # Nearest neighbor reassignment
        new_tier = current_tier
        if ref_embs.shape[0] > 0:
            sims = all_sims[row_i]
            top_k_indices = np.argsort(sims)[-k:][::-1]
            top_k_sims = sims[top_k_indices]
            top_k_tier_list = [ref_tiers[i] for i in top_k_indices]

            if tr.reassignment_enabled and float(top_k_sims[0]) >= tr.reassignment_similarity_floor:
                # Majority vote among top-K neighbors (simple majority)
                tier_counts = Counter(top_k_tier_list)
                majority_tier, majority_count = tier_counts.most_common(1)[0]
                if (
                    majority_count >= majority_threshold
                    and majority_tier != current_tier
                    and majority_tier in _TIER_VALUES
                ):
                    new_tier = majority_tier
                    reassigned += 1

        final_tiers.append(new_tier)

        # Metadata
        if new_tier == "H1_morphology":
            syns_raw = row.get("synonyms")
            syns: list[Any] = list(syns_raw) if isinstance(syns_raw, list | np.ndarray) else []
            defn = row.get("definition")
            organ = organ_normalizer.infer_from_concept(
                str(row["preferred_label"]),
                syns,
                default=tr.organ_default,
            )
            level = classify_level(
                str(row["preferred_label"]),
                syns,
                str(defn) if defn else None,
            )
            category = _classify_h1_category(str(row["preferred_label"]), syns, level, organ)
            organs.append(organ)
            levels.append(level)
            concept_types.append(None)
            categories.append(category)
        else:
            concept_type = _TIER_TO_CONCEPT_TYPE.get(new_tier)
            organs.append(None)
            levels.append(None)
            concept_types.append(concept_type)
            categories.append(concept_type or new_tier)

    df = df.copy()
    df["tier"] = final_tiers
    df["organ"] = organs
    df["level"] = levels
    df["concept_type"] = concept_types
    df["category"] = categories

    elapsed = time.monotonic() - t0
    logger.info(
        "Stage 5 done: {} rows, {} reassigned, {:.1f}s",
        len(df),
        reassigned,
        elapsed,
    )
    logger.info("Tier distribution:\n{}", df["tier"].value_counts().to_string())
    return df
