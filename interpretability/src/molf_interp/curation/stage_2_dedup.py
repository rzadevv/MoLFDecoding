"""Stage 2: cross-source dedup within each candidate_tier."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

from molf_interp.curation.config import CurationConfig, DedupConfig
from molf_interp.curation.gene_overlap import can_merge_gene_programs
from molf_interp.curation.sapbert import SapBert

_MAX_AGGLOMERATIVE = 8000  # fall back to HDBSCAN above this
_PCA_DIMS = 64  # reduce to this many dims before HDBSCAN when n > _PCA_THRESHOLD
_PCA_THRESHOLD = 5000  # apply PCA for tiers larger than this


def _source_priority_rank(source: str, priority: tuple[str, ...]) -> int:
    """Lower = higher priority."""
    for i, s in enumerate(priority):
        if s == source:
            return i
    return len(priority)


def _reduce_dims(embeddings: npt.NDArray[Any], n_components: int) -> npt.NDArray[Any]:
    """PCA dimensionality reduction on L2-normalized embeddings."""
    from sklearn.decomposition import TruncatedSVD

    n_components = min(n_components, embeddings.shape[1], embeddings.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced: npt.NDArray[Any] = np.asarray(svd.fit_transform(embeddings), dtype=np.float32)
    # Re-normalize after PCA
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    result: npt.NDArray[Any] = np.asarray(reduced / norms, dtype=np.float32)
    return result


def _cluster_hdbscan(embeddings: npt.NDArray[Any], min_cluster_size: int) -> npt.NDArray[Any]:
    import hdbscan  # type: ignore[import-untyped]

    # Reduce dims for large tiers to keep HDBSCAN tractable
    if embeddings.shape[0] > _PCA_THRESHOLD and embeddings.shape[1] > _PCA_DIMS:
        logger.info(
            "    PCA {}->{} dims for {} samples before HDBSCAN",
            embeddings.shape[1],
            _PCA_DIMS,
            embeddings.shape[0],
        )
        embeddings = _reduce_dims(embeddings, _PCA_DIMS)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )
    labels: npt.NDArray[Any] = clusterer.fit_predict(embeddings)
    # HDBSCAN noise points (-1) get their own singleton clusters
    max_label = int(labels.max()) if labels.max() >= 0 else -1
    for i, lbl in enumerate(labels):
        if lbl == -1:
            max_label += 1
            labels[i] = max_label
    return labels


def _cluster_agglomerative(
    embeddings: npt.NDArray[Any], similarity_threshold: float
) -> npt.NDArray[Any]:
    from sklearn.cluster import AgglomerativeClustering

    distance_threshold = 1.0 - similarity_threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
        compute_full_tree="auto",
    )
    labels: npt.NDArray[Any] = clustering.fit_predict(embeddings)
    return labels


def _pick_canonical(group_df: pd.DataFrame, config: DedupConfig) -> int:
    """Return the positional index (within group_df) of the canonical record."""
    priority = config.canonical_source_priority
    ranks = group_df["source_name"].apply(lambda s: _source_priority_rank(s, priority))
    min_rank = ranks.min()
    candidates = group_df[ranks == min_rank]
    # Tiebreak by shortest preferred_label
    lengths = candidates["preferred_label"].str.len()
    return int(lengths.idxmin())


def _should_merge_cluster_members(
    member_a: pd.Series,  # type: ignore[type-arg]
    member_b: pd.Series,  # type: ignore[type-arg]
    *,
    name_similarity: float,
    name_threshold: float,
    gene_overlap_threshold: float = 0.5,
) -> bool:
    """Decide whether two cluster members should actually merge.

    Default: name_similarity >= name_threshold.
    For MSigDB gene_program pairs that both have gene lists, ALSO require
    Jaccard gene overlap >= gene_overlap_threshold.

    Args:
        member_a: First cluster member row.
        member_b: Second cluster member row.
        name_similarity: Cosine similarity between the two embeddings.
        name_threshold: Minimum similarity to trigger a merge.
        gene_overlap_threshold: Applied only when both members are MSigDB gene_programs.

    Returns:
        True if the pair should be merged.
    """
    return can_merge_gene_programs(
        dict(member_a),
        dict(member_b),
        name_similarity=name_similarity,
        name_threshold=name_threshold,
        gene_overlap_threshold=gene_overlap_threshold,
    )


def _merge_cluster(cluster_df: pd.DataFrame, canonical_idx: int) -> dict[str, Any]:
    """Return merged metadata for the canonical record."""
    canonical = cluster_df.loc[canonical_idx]

    # Union synonyms from all cluster members
    all_syns: list[str] = []
    for _, row in cluster_df.iterrows():
        syns_raw = row.get("synonyms")
        if isinstance(syns_raw, list | np.ndarray):
            all_syns.extend(str(s) for s in syns_raw)
        if row["preferred_label"] != canonical["preferred_label"]:
            all_syns.append(str(row["preferred_label"]))

    # Deduplicate case-insensitively, preserve original casing
    seen_lower: set[str] = set()
    deduped_syns: list[str] = []
    for s in all_syns:
        sl = s.lower()
        if sl not in seen_lower:
            seen_lower.add(sl)
            deduped_syns.append(s)

    # Take longest non-None definition
    best_def: str | None = None
    best_len = -1
    for _, row in cluster_df.iterrows():
        d = row.get("definition")
        if d and isinstance(d, str) and len(d) > best_len:
            best_def = d
            best_len = len(d)

    return {
        "synonyms": sorted(deduped_syns),
        "definition": best_def,
    }


def _dedup_tier_group(
    tier_df: pd.DataFrame,
    config: DedupConfig,
) -> pd.DataFrame:
    """Dedup within a single tier. Returns deduplicated dataframe."""
    n = len(tier_df)
    if n == 0:
        return tier_df

    # Single-element tier: nothing to cluster, assign label 0
    if n == 1:
        tier_df = tier_df.copy()
        tier_df["_cluster"] = 0
        tier_df["dedup_cluster_id"] = "0"
        tier_df["dedup_cluster_size"] = 1
        return tier_df

    embeddings = np.stack(tier_df["embedding"].tolist())

    if config.method == "hdbscan" or n > _MAX_AGGLOMERATIVE:
        labels = _cluster_hdbscan(embeddings, config.min_cluster_size)
    else:
        labels = _cluster_agglomerative(embeddings, config.similarity_threshold)

    tier_df = tier_df.copy()
    tier_df["_cluster"] = labels

    # For MSigDB gene_program tier: check pairwise gene-overlap and split clusters
    # whose members have similar names but incompatible gene compositions.
    tier_name = str(tier_df["candidate_tier"].iloc[0]) if len(tier_df) > 0 else ""
    is_gene_program = tier_name == "H2_gene_program"

    def _split_if_gene_mismatch(group: pd.DataFrame) -> list[list[int]]:
        """Recursively split a cluster by gene-overlap compatibility.

        Returns a list of sub-cluster index lists.
        """
        if not is_gene_program or len(group) == 1:
            return [list(group.index)]
        idxs = list(group.index)
        # Build a compatibility graph: edge = should merge
        embs_group = np.stack([tier_df.loc[i, "embedding"] for i in idxs])
        sims = embs_group @ embs_group.T  # NxN cosine sims (L2-normed)
        # Union-Find to group compatible indices
        parent = {i: i for i in idxs}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a_pos, a_idx in enumerate(idxs):
            for b_pos in range(a_pos + 1, len(idxs)):
                b_idx = idxs[b_pos]
                sim = float(sims[a_pos, b_pos])
                if _should_merge_cluster_members(
                    tier_df.loc[a_idx],
                    tier_df.loc[b_idx],
                    name_similarity=sim,
                    name_threshold=config.similarity_threshold,
                    gene_overlap_threshold=config.gene_overlap_threshold,
                ):
                    ra, rb = find(a_idx), find(b_idx)
                    if ra != rb:
                        parent[rb] = ra

        # Collect sub-clusters
        sub: dict[int, list[int]] = {}
        for idx in idxs:
            root = find(idx)
            sub.setdefault(root, []).append(idx)
        return list(sub.values())

    rows_to_keep: list[int] = []
    overrides: dict[int, dict[str, Any]] = {}
    next_virtual_cluster = int(labels.max()) + 1 if len(labels) > 0 else 0

    for cluster_id, group in tier_df.groupby("_cluster"):
        sub_clusters = _split_if_gene_mismatch(group)
        for sub_group in sub_clusters:
            sub_df = tier_df.loc[sub_group]
            if len(sub_df) == 1:
                rows_to_keep.append(sub_df.index[0])
                tier_df.loc[sub_df.index[0], "dedup_cluster_id"] = f"{cluster_id}"
                tier_df.loc[sub_df.index[0], "dedup_cluster_size"] = 1
            else:
                canonical_idx = _pick_canonical(sub_df, config)
                merged = _merge_cluster(sub_df, canonical_idx)
                rows_to_keep.append(canonical_idx)
                overrides[canonical_idx] = merged
                cid_label = f"{cluster_id}" if len(sub_clusters) == 1 else f"{next_virtual_cluster}"
                if len(sub_clusters) > 1:
                    next_virtual_cluster += 1
                for idx in sub_df.index:
                    tier_df.loc[idx, "dedup_cluster_id"] = cid_label
                    tier_df.loc[idx, "dedup_cluster_size"] = len(sub_df)

    result = tier_df.loc[rows_to_keep].copy()
    for idx, override in overrides.items():
        if idx in result.index:
            result.at[idx, "synonyms"] = override["synonyms"]
            if override["definition"] is not None:
                result.at[idx, "definition"] = override["definition"]

    return result


def run_stage_2(
    df: pd.DataFrame,
    config: CurationConfig,
    sapbert: SapBert,
) -> pd.DataFrame:
    """Cross-source dedup within each candidate_tier.

    Args:
        df: DataFrame with 'embedding' column from Stage 1.
        config: CurationConfig instance.
        sapbert: SapBert instance (unused here, reserved for future use).

    Returns:
        Deduplicated DataFrame.
    """
    t0 = time.monotonic()
    logger.info(
        "Stage 2: dedup {} rows across {} tiers…",
        len(df),
        df["candidate_tier"].nunique(),
    )

    if "embedding" not in df.columns:
        raise ValueError("Stage 2 requires embeddings from Stage 1")

    df = df.copy()
    df["dedup_cluster_id"] = ""
    df["dedup_cluster_size"] = 1

    tier_results: list[pd.DataFrame] = []
    for tier, tier_df in df.groupby("candidate_tier"):
        logger.info("  Dedup tier {}: {} rows", tier, len(tier_df))
        deduped = _dedup_tier_group(tier_df, config.dedup)
        n_dropped = len(tier_df) - len(deduped)
        logger.info("    → {} kept, {} collapsed", len(deduped), n_dropped)
        tier_results.append(deduped)

    result = pd.concat(tier_results, ignore_index=True)
    elapsed = time.monotonic() - t0
    logger.info(
        "Stage 2 done: {} → {} rows (dropped {}), {:.1f}s",
        len(df),
        len(result),
        len(df) - len(result),
        elapsed,
    )
    return result
