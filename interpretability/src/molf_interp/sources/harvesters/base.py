"""Shared utilities for all harvesters."""

from __future__ import annotations

from collections.abc import Iterable

_NON_MORPHOLOGICAL_WORDS: frozenset[str] = frozenset(
    {
        "survival",
        "stage",
        "age",
        "patient",
        "history",
        "biopsy site",
        "prognosis",
        "treatment",
        "therapy",
        "clinical",
        "laboratory",
        "protein",
        "gene expression",
        "mrna",
        "variant",
        "mutation",
        "snp",
        "genetic",
        "molecular",
        "sequence",
        "assay",
        "biomarker",
        "diagnosis",
        "diagnostic",
        "epigenetic",
        "methylation",
    }
)


def normalize_synonyms(raw: Iterable[str], *, drop_label: str | None = None) -> tuple[str, ...]:
    """Clean, dedup, and sort synonyms.

    Args:
        raw: Raw synonym strings (may contain whitespace or duplicates).
        drop_label: Exact label to exclude from synonyms (case-insensitive).

    Returns:
        Sorted, deduplicated, non-empty synonym tuple preserving first-seen casing.
    """
    seen_lower: set[str] = set()
    result: list[str] = []
    drop_lower = drop_label.lower() if drop_label else None
    for s in raw:
        stripped = s.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if drop_lower and lower == drop_lower:
            continue
        if lower in seen_lower:
            continue
        seen_lower.add(lower)
        result.append(stripped)
    return tuple(sorted(result, key=str.casefold))


def truncate_definition(text: str | None, *, max_chars: int = 500) -> str | None:
    """Trim a definition to max_chars, breaking at the last sentence boundary if possible.

    Args:
        text: Raw definition text, or None.
        max_chars: Maximum character count (default 500).

    Returns:
        Truncated string, or None if input is None.
    """
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(". ")
    if last_period > 0:
        return truncated[: last_period + 1]
    return truncated


def is_obviously_non_morphological(label: str) -> bool:
    """Quick reject filter: returns True for labels that are clearly NOT H&E features.

    Heuristic — rejects labels containing clinical/molecular keywords. Defaults
    to False (keep) on ambiguous cases. Heavy lifting is left to SapBERT in curation.

    Args:
        label: The ontology class preferred label to evaluate.

    Returns:
        True if the label is obviously non-morphological and should be discarded.
    """
    lower = label.lower()
    return any(word in lower for word in _NON_MORPHOLOGICAL_WORDS)
