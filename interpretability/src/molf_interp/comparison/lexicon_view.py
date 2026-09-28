"""Primary lexicon view (View A) construction for the merged concept bank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from molf_interp.curation.sapbert import SapBert
from molf_interp.io.config import BaseConfig


class LexiconViewConfig(BaseConfig):
    """Parameters for primary lexicon view (View A) construction."""

    reference_neighbor_similarity_threshold: float = 0.70
    """Min SapBERT cosine to nearest same-tier reference concept for harvested
    concepts not already matched to the reference bank to enter View A."""

    require_definition_or_synonyms: bool = True
    """If True and the merged_concepts DataFrame has 'definition'/'synonyms'
    columns, automated concepts must have at least one non-null definition OR a
    non-empty synonyms list. Skipped silently when those columns are absent
    (the Concept schema does not store them)."""

    exclude_labels_shorter_than: int = 4
    """Drop labels with fewer than N characters."""

    exclude_label_token_count_above: int = 8
    """Drop labels longer than N whitespace-separated tokens."""

    exclude_source_substrings: tuple[str, ...] = (
        "transmembrane transport",
        "downstream events",
        "stage I",
        "stage II",
        "stage III",
        "stage IV",
    )
    """Drop labels containing any of these substrings (case-insensitive).
    Targets Reactome reaction-level events and clinical staging qualifiers that
    are too specific for single-patch H&E lexicon use."""


@dataclass(frozen=True)
class LexiconViewResult:
    """Output of View A construction."""

    in_primary_lexicon: pd.Series
    """Bool Series indexed by concept_id; True = in View A."""

    reason_excluded: pd.Series
    """Str Series indexed by concept_id; empty string for included concepts."""

    summary: dict[str, Any]
    """Counts: total, by_tier, by_inclusion_reason, by_exclusion_reason."""


def compute_nearest_ref_within_tier(
    harvested_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    sapbert: SapBert,
) -> pd.Series:
    """Compute max cosine similarity to any same-tier reference concept for each harvested concept.

    Uses SapBERT's embedding cache; repeated calls for the same labels are
    instant. Computes tier-by-tier to bound the cross-product size.

    Args:
        harvested_df: DataFrame of harvested concepts (concept_id, concept_name,
            tier columns required).
        ref_df: DataFrame of reference concepts (concept_id, concept_name, tier columns
            required).
        sapbert: Loaded SapBERT model.

    Returns:
        Series indexed by concept_id with float similarity values. Concepts in
        tiers where the reference bank has zero concepts receive NaN.
    """
    if harvested_df.empty:
        return pd.Series(dtype=float, name="nearest_ref_sim")

    ref_by_tier: dict[str, list[str]] = {}
    for tier in ref_df["tier"].unique():
        ref_by_tier[str(tier)] = ref_df[ref_df["tier"] == tier]["concept_name"].tolist()

    sims: dict[str, float] = {}
    for tier in harvested_df["tier"].unique():
        tier_str = str(tier)
        harv_tier_df = harvested_df[harvested_df["tier"] == tier]
        ref_labels = ref_by_tier.get(tier_str, [])

        if not ref_labels:
            for cid in harv_tier_df["concept_id"]:
                sims[str(cid)] = float("nan")
            continue

        harv_labels = harv_tier_df["concept_name"].tolist()
        harv_emb = sapbert.encode(harv_labels)
        ref_emb = sapbert.encode(ref_labels)
        sim_matrix = sapbert.cosine_similarity(harv_emb, ref_emb)  # [N_harv, N_ref]
        max_sims = np.max(sim_matrix, axis=1)

        for i, cid in enumerate(harv_tier_df["concept_id"].tolist()):
            sims[str(cid)] = float(max_sims[i])

    return pd.Series(sims, name="nearest_ref_sim", dtype=float)


def build_primary_lexicon_view(
    merged_concepts: pd.DataFrame,
    matches: pd.DataFrame,
    sapbert: SapBert,
    config: LexiconViewConfig,
) -> LexiconViewResult:
    """Construct View A from the merged bank.

    Inclusion rules (applied in order; first exclusion reason wins):

      1. Reference concepts: ALL included unconditionally.
      2. Harvested concepts with a within-tier match (is_cross_tier=False in
         matches): included unconditionally — already validated by reference overlap.
      3. Cross-tier-matched harvested concepts: NOT auto-included; fall through
         to the same-tier neighbor check (rule 4).
      4. All other harvested concepts: included only if ALL of:
           a. label length >= exclude_labels_shorter_than
           b. token count <= exclude_label_token_count_above
           c. label does not contain any exclude_source_substrings
           d. if require_definition_or_synonyms AND 'definition'/'synonyms'
              columns exist: at least one non-null definition OR non-empty synonyms
           e. SapBERT nearest-same-tier-reference similarity >= threshold

    Exclusion reasons: "label_too_short" | "label_too_long" |
    "excluded_substring" | "no_definition_no_synonyms" |
    "low_reference_neighbor_similarity".

    Args:
        merged_concepts: Full merged bank DataFrame (concept_id, concept_name,
            tier, provenance, cross_bank_status columns required).
        matches: Matches DataFrame with is_cross_tier column.
        sapbert: Loaded SapBERT model.
        config: LexiconViewConfig controlling thresholds and filters.

    Returns:
        LexiconViewResult with inclusion flags, exclusion reasons, and counts.
    """
    su_mask = merged_concepts["provenance"] == "reference"
    su_df = merged_concepts[su_mask].copy()
    auto_df = merged_concepts[~su_mask].copy()

    # Within-tier matched automated concept_ids (these get auto-included)
    within_tier = matches[matches["is_cross_tier"] == False]  # noqa: E712
    within_tier_matched_ids: set[str] = set(
        within_tier["harvested_concept_id"].astype(str).tolist()
    )

    included: dict[str, bool] = {}
    reason: dict[str, str] = {}
    inclusion_su = 0
    inclusion_matched = 0
    inclusion_su_adjacent = 0

    # Rule 1: All reference concepts included
    for cid in su_df["concept_id"].astype(str):
        included[cid] = True
        reason[cid] = ""
        inclusion_su += 1

    # Rule 2: Within-tier matched automated → included
    matched_harv = auto_df[auto_df["concept_id"].astype(str).isin(within_tier_matched_ids)]
    for cid in matched_harv["concept_id"].astype(str):
        included[cid] = True
        reason[cid] = ""
        inclusion_matched += 1

    # Rules 3-4: Remaining automated (unmatched + cross-tier matched)
    unmatched_auto = auto_df[~auto_df["concept_id"].astype(str).isin(within_tier_matched_ids)]

    if unmatched_auto.empty:
        excluded_reasons: dict[str, int] = {
            "label_too_short": 0,
            "label_too_long": 0,
            "excluded_substring": 0,
            "no_definition_no_synonyms": 0,
            "low_reference_neighbor_similarity": 0,
        }
    else:
        excluded_reasons = _apply_filters(
            unmatched_auto, su_df, merged_concepts, sapbert, config, included, reason
        )
        for cid in unmatched_auto["concept_id"].astype(str):
            if included.get(cid, False):
                inclusion_su_adjacent += 1

    # Build Series aligned to all concept_ids
    all_ids = merged_concepts["concept_id"].astype(str).tolist()
    in_lex = pd.Series(
        {cid: included.get(cid, False) for cid in all_ids},
        name="in_primary_lexicon",
        dtype=bool,
    )
    reason_series = pd.Series(
        {cid: reason.get(cid, "") for cid in all_ids},
        name="reason_excluded",
        dtype=str,
    )

    view_a_total = int(in_lex.sum())
    by_tier: dict[str, int] = {}
    for tier in merged_concepts["tier"].unique():
        tier_mask = merged_concepts["tier"] == tier
        cids_in_tier = merged_concepts.loc[tier_mask, "concept_id"].astype(str)
        by_tier[str(tier)] = int(in_lex[in_lex.index.isin(cids_in_tier)].sum())

    summary: dict[str, Any] = {
        "total": view_a_total,
        "by_tier": by_tier,
        "by_inclusion_reason": {
            "reference": inclusion_su,
            "harvested_matched": inclusion_matched,
            "harvested_ref_adjacent": inclusion_su_adjacent,
        },
        "by_exclusion_reason": excluded_reasons,
    }

    logger.info(
        "View A: {} / {} concepts included (reference={} matched={} ref_adjacent={})",
        view_a_total,
        len(merged_concepts),
        inclusion_su,
        inclusion_matched,
        inclusion_su_adjacent,
    )
    return LexiconViewResult(
        in_primary_lexicon=in_lex,
        reason_excluded=reason_series,
        summary=summary,
    )


def _apply_filters(
    unmatched_auto: pd.DataFrame,
    su_df: pd.DataFrame,
    merged_concepts: pd.DataFrame,
    sapbert: SapBert,
    config: LexiconViewConfig,
    included: dict[str, bool],
    reason: dict[str, str],
) -> dict[str, int]:
    """Apply sequential label and similarity filters to unmatched automated concepts.

    Modifies ``included`` and ``reason`` in place. Returns per-reason counts.
    """
    excluded_reasons: dict[str, int] = {
        "label_too_short": 0,
        "label_too_long": 0,
        "excluded_substring": 0,
        "no_definition_no_synonyms": 0,
        "low_reference_neighbor_similarity": 0,
    }
    remaining_ids: set[str] = set(unmatched_auto["concept_id"].astype(str).tolist())

    def _mark(cid: str, r: str) -> None:
        included[cid] = False
        reason[cid] = r
        excluded_reasons[r] += 1
        remaining_ids.discard(cid)

    # Filter 1: label too short
    short_mask = unmatched_auto["concept_name"].str.len() < config.exclude_labels_shorter_than
    for cid in unmatched_auto.loc[short_mask, "concept_id"].astype(str):
        _mark(cid, "label_too_short")

    # Filter 2: label too long (token count)
    rem_df = unmatched_auto[unmatched_auto["concept_id"].astype(str).isin(remaining_ids)]
    max_tok = config.exclude_label_token_count_above
    long_mask = rem_df["concept_name"].str.split().str.len() > max_tok
    for cid in rem_df.loc[long_mask, "concept_id"].astype(str):
        _mark(cid, "label_too_long")

    # Filter 3: excluded substrings
    subs = tuple(s.lower() for s in config.exclude_source_substrings)
    rem_df = unmatched_auto[unmatched_auto["concept_id"].astype(str).isin(remaining_ids)]
    sub_mask = (
        rem_df["concept_name"].str.lower().apply(lambda label: any(sub in label for sub in subs))
    )
    for cid in rem_df.loc[sub_mask, "concept_id"].astype(str):
        _mark(cid, "excluded_substring")

    # Filter 4: definition/synonyms (only when columns present)
    if config.require_definition_or_synonyms:
        has_def_col = "definition" in merged_concepts.columns
        has_syn_col = "synonyms" in merged_concepts.columns
        if has_def_col or has_syn_col:
            rem_df = unmatched_auto[unmatched_auto["concept_id"].astype(str).isin(remaining_ids)]
            for _, row in rem_df.iterrows():
                cid = str(row["concept_id"])
                has_def = (
                    has_def_col
                    and pd.notna(row.get("definition"))
                    and str(row.get("definition", "")).strip()
                )
                has_syn = has_syn_col and bool(row.get("synonyms"))
                if not has_def and not has_syn:
                    _mark(cid, "no_definition_no_synonyms")
        # If neither column exists, skip this filter (data not available)

    # Filter 5: SapBERT nearest-reference similarity (only on remaining)
    if remaining_ids:
        rem_df = unmatched_auto[unmatched_auto["concept_id"].astype(str).isin(remaining_ids)]
        logger.info(
            "Computing nearest-reference similarity for {} harvested candidates …", len(rem_df)
        )
        nearest_sim = compute_nearest_ref_within_tier(rem_df, su_df, sapbert)
        for cid in list(remaining_ids):
            sim = nearest_sim.get(cid, float("nan"))
            if pd.isna(sim) or float(sim) < config.reference_neighbor_similarity_threshold:
                _mark(cid, "low_reference_neighbor_similarity")
            else:
                included[cid] = True
                reason[cid] = ""

    return excluded_reasons
