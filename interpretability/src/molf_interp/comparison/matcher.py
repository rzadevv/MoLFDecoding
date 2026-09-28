"""Cross-bank greedy bipartite concept matching."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from loguru import logger

from molf_interp.concepts.schemas import ConceptBank
from molf_interp.curation.sapbert import SapBert


@dataclass(frozen=True)
class ConceptMatch:
    """One matched pair of concepts across banks."""

    reference_concept_id: str
    harvested_concept_id: str
    reference_label: str
    harvested_label: str
    tier: str
    similarity: float
    is_cross_tier: bool = False
    reference_tier: str | None = None
    harvested_tier: str | None = None
    discordance_class: str = "exact"
    """Type of discordance. One of:
      "exact": within-tier match;
      "cross_tier_h1_vs_h2_cell_type": cells treated as morphology pattern;
      "cross_tier_program_vs_pathway": gene program vs signaling pathway;
      "cross_tier_other": any other cross-tier pair.
    """


def _classify_discordance(
    reference_tier: str | None,
    harvested_tier: str | None,
    is_cross_tier: bool,
) -> str:
    """Classify cross-tier matches into named discordance categories.

    Args:
        reference_tier: Reference concept tier (None for within-tier matches).
        harvested_tier: Harvested concept tier (None for within-tier matches).
        is_cross_tier: Whether this is a cross-tier match.

    Returns:
        Discordance class string.
    """
    if not is_cross_tier:
        return "exact"
    pair = frozenset({reference_tier, harvested_tier})
    if pair == frozenset({"H1_morphology", "H2_cell_type"}):
        return "cross_tier_h1_vs_h2_cell_type"
    if pair == frozenset({"H2_gene_program", "H2_pathway"}):
        return "cross_tier_program_vs_pathway"
    return "cross_tier_other"


def _greedy_match(
    ref_labels: list[str],
    harv_labels: list[str],
    ref_ids: list[str],
    harv_ids: list[str],
    sim_matrix: npt.NDArray[np.float32],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Apply greedy bipartite matching above threshold.

    Sort all (i, j) pairs by similarity descending.  Each index may be used at
    most once.  This is O(N*M*log(N*M)) but sub-second for N,M ≤ 5000.

    Greedy matching is not optimal (Hungarian/Jonker-Volgenant would be), but at
    high similarity the greedy answer is virtually identical to the optimal
    assignment while being ~100x faster at this scale.  The choice is deliberate:
    for biomedical label matching the top-1 greedy hit is almost always the
    globally-optimal one.

    Args:
        ref_labels: Reference concept labels.
        harv_labels: Harvested concept labels.
        ref_ids: Reference concept IDs aligned to ref_labels.
        harv_ids: Harvested concept IDs aligned to harv_labels.
        sim_matrix: Cosine similarity matrix [N_ref, N_harv].
        threshold: Minimum similarity to consider a match.

    Returns:
        List of (ref_idx, harv_idx, similarity) triples, sorted by descending sim.
    """
    n_su, n_auto = sim_matrix.shape
    candidates: list[tuple[float, int, int]] = []
    for i in range(n_su):
        for j in range(n_auto):
            s = float(sim_matrix[i, j])
            if s >= threshold:
                candidates.append((s, i, j))

    candidates.sort(reverse=True)
    used_su: set[int] = set()
    used_auto: set[int] = set()
    result: list[tuple[int, int, float]] = []
    for s, i, j in candidates:
        if i not in used_su and j not in used_auto:
            result.append((i, j, s))
            used_su.add(i)
            used_auto.add(j)
    return result


def match_banks(
    reference_bank: ConceptBank,
    harvested_bank: ConceptBank,
    sapbert: SapBert,
    *,
    threshold: float = 0.85,
    cross_tier_threshold: float | None = 0.92,
) -> list[ConceptMatch]:
    """Greedy bipartite matching across banks within each tier.

    Algorithm:
      1. For each tier T present in both banks:
         a. Embed all reference[T] and harvested[T] labels via SapBERT.
         b. Compute cosine similarity matrix [N_ref x N_harv].
         c. Greedy bipartite match at threshold.
      2. If cross_tier_threshold is not None:
         a. Collect unmatched concepts from both banks.
         b. Compute cross-tier similarities between them.
         c. Greedy match at cross_tier_threshold, marking is_cross_tier=True.

    Greedy matching is chosen over Hungarian/Jonker-Volgenant because at high
    similarity the answers converge and greedy is ~100x faster at this scale.

    Args:
        reference_bank: Reference (curated) bank.
        harvested_bank: Harvested bank.
        sapbert: Loaded SapBERT model.
        threshold: Within-tier match threshold.
        cross_tier_threshold: Cross-tier match threshold (None to disable).

    Returns:
        All matches sorted by tier then by similarity descending.
    """
    ref_by_tier: dict[str, list[tuple[str, str]]] = {}
    for c in reference_bank.concepts:
        ref_by_tier.setdefault(c.tier.value, []).append((c.concept_id, c.concept_name))

    harv_by_tier: dict[str, list[tuple[str, str]]] = {}
    for c in harvested_bank.concepts:
        harv_by_tier.setdefault(c.tier.value, []).append((c.concept_id, c.concept_name))

    all_tiers = sorted(set(ref_by_tier) | set(harv_by_tier))
    matches: list[ConceptMatch] = []
    matched_ref_ids: set[str] = set()
    matched_harv_ids: set[str] = set()

    for tier in all_tiers:
        ref_pairs = ref_by_tier.get(tier, [])
        harv_pairs = harv_by_tier.get(tier, [])
        if not ref_pairs or not harv_pairs:
            logger.info(
                "Tier {}: ref={} harv={} — skipping (one bank empty)",
                tier,
                len(ref_pairs),
                len(harv_pairs),
            )
            continue

        ref_ids, ref_labels = zip(*ref_pairs, strict=True)
        harv_ids, harv_labels = zip(*harv_pairs, strict=True)
        ref_ids_l = list(ref_ids)
        harv_ids_l = list(harv_ids)
        ref_labels_l = list(ref_labels)
        harv_labels_l = list(harv_labels)

        ref_emb = sapbert.encode(ref_labels_l)
        harv_emb = sapbert.encode(harv_labels_l)
        sim_matrix = sapbert.cosine_similarity(ref_emb, harv_emb)

        tier_matches = _greedy_match(
            ref_labels_l, harv_labels_l, ref_ids_l, harv_ids_l, sim_matrix, threshold
        )

        for ref_i, harv_j, sim in tier_matches:
            matches.append(
                ConceptMatch(
                    reference_concept_id=ref_ids_l[ref_i],
                    harvested_concept_id=harv_ids_l[harv_j],
                    reference_label=ref_labels_l[ref_i],
                    harvested_label=harv_labels_l[harv_j],
                    tier=tier,
                    similarity=sim,
                )
            )
            matched_ref_ids.add(ref_ids_l[ref_i])
            matched_harv_ids.add(harv_ids_l[harv_j])

        sims = [s for _, _, s in tier_matches]
        logger.info(
            "Tier {}: ref={} harv={} matched={} sim_mean={:.3f} sim_min={:.3f} sim_max={:.3f}",
            tier,
            len(ref_pairs),
            len(harv_pairs),
            len(tier_matches),
            float(np.mean(sims)) if sims else 0.0,
            float(np.min(sims)) if sims else 0.0,
            float(np.max(sims)) if sims else 0.0,
        )

    # Cross-tier matching for unmatched concepts
    if cross_tier_threshold is not None:
        unmatched_ref = [
            (c.concept_id, c.concept_name, c.tier.value)
            for c in reference_bank.concepts
            if c.concept_id not in matched_ref_ids
        ]
        unmatched_harv = [
            (c.concept_id, c.concept_name, c.tier.value)
            for c in harvested_bank.concepts
            if c.concept_id not in matched_harv_ids
        ]

        if unmatched_ref and unmatched_harv:
            ref_ids_u = [x[0] for x in unmatched_ref]
            ref_labels_u = [x[1] for x in unmatched_ref]
            ref_tiers_u = [x[2] for x in unmatched_ref]
            harv_ids_u = [x[0] for x in unmatched_harv]
            harv_labels_u = [x[1] for x in unmatched_harv]
            harv_tiers_u = [x[2] for x in unmatched_harv]

            ref_emb_u = sapbert.encode(ref_labels_u)
            harv_emb_u = sapbert.encode(harv_labels_u)
            sim_matrix_u = sapbert.cosine_similarity(ref_emb_u, harv_emb_u)

            # Only consider cross-tier pairs (same-tier unmatched already failed within-tier)
            for i in range(len(ref_ids_u)):
                for j in range(len(harv_ids_u)):
                    if ref_tiers_u[i] == harv_tiers_u[j]:
                        sim_matrix_u[i, j] = 0.0  # mask same-tier (already handled)

            ct_matches = _greedy_match(
                ref_labels_u,
                harv_labels_u,
                ref_ids_u,
                harv_ids_u,
                sim_matrix_u,
                cross_tier_threshold,
            )
            for ref_i, harv_j, sim in ct_matches:
                st = ref_tiers_u[ref_i]
                at = harv_tiers_u[harv_j]
                matches.append(
                    ConceptMatch(
                        reference_concept_id=ref_ids_u[ref_i],
                        harvested_concept_id=harv_ids_u[harv_j],
                        reference_label=ref_labels_u[ref_i],
                        harvested_label=harv_labels_u[harv_j],
                        tier=st,
                        similarity=sim,
                        is_cross_tier=True,
                        reference_tier=st,
                        harvested_tier=at,
                        discordance_class=_classify_discordance(st, at, True),
                    )
                )
            logger.info("Cross-tier matches: {}", len(ct_matches))

    matches.sort(key=lambda m: (m.tier, -m.similarity))
    return matches


def compute_unmatched_nearest_sims(
    reference_bank: ConceptBank,
    harvested_bank: ConceptBank,
    matches: Sequence[ConceptMatch],
    sapbert: SapBert,
) -> dict[str, tuple[str, float]]:
    """Compute the nearest cross-bank neighbor for every unmatched concept.

    For each reference-only concept, find its nearest harvested concept by cosine sim.
    For each harvested-only concept, find its nearest reference concept.

    Args:
        reference_bank: Reference (curated) bank.
        harvested_bank: Harvested bank.
        matches: Matched pairs (from match_banks).
        sapbert: Loaded SapBERT model.

    Returns:
        Dict mapping concept_id -> (nearest_cross_bank_id, similarity).
    """
    matched_ref = {m.reference_concept_id for m in matches}
    matched_harv = {m.harvested_concept_id for m in matches}

    su_unmatched = [
        (c.concept_id, c.concept_name)
        for c in reference_bank.concepts
        if c.concept_id not in matched_ref
    ]
    auto_unmatched = [
        (c.concept_id, c.concept_name)
        for c in harvested_bank.concepts
        if c.concept_id not in matched_harv
    ]

    result: dict[str, tuple[str, float]] = {}
    if not su_unmatched and not auto_unmatched:
        return result

    ref_u_ids = [x[0] for x in su_unmatched]
    ref_u_labels = [x[1] for x in su_unmatched]
    harv_u_ids = [x[0] for x in auto_unmatched]
    harv_u_labels = [x[1] for x in auto_unmatched]

    ref_emb = sapbert.encode(ref_u_labels)
    harv_emb = sapbert.encode(harv_u_labels)
    sapbert.cosine_similarity(ref_emb, harv_emb)  # pre-warm cache, not used directly

    # For each unmatched reference, find nearest harvested
    harv_all_ids = [c.concept_id for c in harvested_bank.concepts]
    if harv_all_ids:
        harv_all_labels = [c.concept_name for c in harvested_bank.concepts]
        harv_all_emb = sapbert.encode(harv_all_labels)
        sim_su_vs_all = sapbert.cosine_similarity(ref_emb, harv_all_emb)
        for i, su_id in enumerate(ref_u_ids):
            best_j = int(np.argmax(sim_su_vs_all[i]))
            result[su_id] = (harv_all_ids[best_j], float(sim_su_vs_all[i, best_j]))

    # For each unmatched harvested, find nearest reference
    ref_all_ids = [c.concept_id for c in reference_bank.concepts]
    if ref_all_ids:
        ref_all_labels = [c.concept_name for c in reference_bank.concepts]
        ref_all_emb = sapbert.encode(ref_all_labels)
        sim_auto_vs_all = sapbert.cosine_similarity(harv_emb, ref_all_emb)
        for j, auto_id in enumerate(harv_u_ids):
            best_i = int(np.argmax(sim_auto_vs_all[j]))
            result[auto_id] = (ref_all_ids[best_i], float(sim_auto_vs_all[j, best_i]))

    return result
