"""Jaccard gene-overlap helpers for MSigDB dedup."""

from __future__ import annotations


def parse_gene_list(extra: object, key: str = "genes") -> set[str]:
    """Parse a comma-separated gene list from extra[key].

    Args:
        extra: The ``extra`` field value — either a list of [key, value] pairs or a
            plain dict.  Non-matching structure silently returns an empty set.
        key: Key to look for in the extra structure.

    Returns:
        Set of uppercase gene symbols, or empty set if the key is absent or the value
        is blank.
    """
    if extra is None:
        return set()
    if isinstance(extra, dict):
        raw = extra.get(key, "")
    elif isinstance(extra, list | tuple):
        # list-of-pairs format used in the pipeline parquet (entries may be lists or np.ndarray)
        raw = ""
        for entry in extra:
            try:
                entry_len = len(entry)  # type: ignore[arg-type]
            except TypeError:
                continue
            if not isinstance(entry, str) and entry_len >= 2 and str(entry[0]) == key:
                raw = str(entry[1])
                break
    else:
        return set()
    raw = str(raw).strip()
    if not raw:
        return set()
    return {g.strip().upper() for g in raw.split(",") if g.strip()}


def jaccard_overlap(genes_a: set[str], genes_b: set[str]) -> float:
    """Compute Jaccard similarity between two gene sets.

    Args:
        genes_a: First gene set.
        genes_b: Second gene set.

    Returns:
        intersection(A, B) / union(A, B), or 0.0 if both sets are empty.
    """
    if not genes_a and not genes_b:
        return 0.0
    union = genes_a | genes_b
    return len(genes_a & genes_b) / len(union)


def can_merge_gene_programs(
    record_a: dict[str, object],
    record_b: dict[str, object],
    *,
    name_similarity: float,
    name_threshold: float,
    gene_overlap_threshold: float = 0.5,
) -> bool:
    """Decide whether two H2_gene_program candidates should merge.

    Default: merge if name_similarity ≥ name_threshold.
    Exception: if BOTH records come from MSigDB AND both have gene lists, ALSO require
    Jaccard gene overlap ≥ gene_overlap_threshold.  This prevents collapsing
    Hallmark "EMT" with Gavish "EMT" — same name, different gene composition.

    Args:
        record_a: First candidate dict with 'source_name' and 'extra' keys.
        record_b: Second candidate dict with 'source_name' and 'extra' keys.
        name_similarity: Cosine similarity between the two concept embeddings.
        name_threshold: Minimum name similarity to merge.
        gene_overlap_threshold: Minimum Jaccard overlap (applied only when both have
            gene lists from MSigDB).

    Returns:
        True if the two records should be merged into a single canonical entry.
    """
    if name_similarity < name_threshold:
        return False

    src_a = str(record_a.get("source_name", ""))
    src_b = str(record_b.get("source_name", ""))
    both_msigdb = src_a.startswith("msigdb") and src_b.startswith("msigdb")

    if not both_msigdb:
        return True

    genes_a = parse_gene_list(record_a.get("extra"))
    genes_b = parse_gene_list(record_b.get("extra"))

    if not genes_a or not genes_b:
        # At least one lacks a gene list — fall back to name similarity alone
        return True

    return jaccard_overlap(genes_a, genes_b) >= gene_overlap_threshold
