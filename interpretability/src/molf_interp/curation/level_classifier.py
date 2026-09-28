"""Rule-based classifier for H1_morphology anatomical level."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

_Level = Literal["tissue", "cellular", "subcellular", "extracellular", "stromal"]

# Keyword sets per level (word-boundary matched, case-insensitive)
_TISSUE_KEYWORDS: frozenset[str] = frozenset(
    {
        "fibrosis",
        "edema",
        "oedema",
        "necrosis",
        "infiltrate",
        "infiltration",
        "architecture",
        "glandular",
        "papillary",
        "solid",
        "nested",
        "tubular",
        "cribriform",
        "trabecular",
        "lobular",
        "alveolar",
        "acinar",
        "fascicular",
        "storiform",
        "whorl",
        "pseudoglandular",
        "lepidic",
        "micropapillary",
        "growth pattern",
        "desmoplastic",
        "myxoid",
        "hyalinized",
        "psammoma",
        "calcification",
        "hemorrhage",
        "infarct",
        "abscess",
        "granuloma",
        "granulomatous",
        "fibrotic",
        "sclerotic",
        "cystic",
        "pseudocyst",
        "necrotic",
        "inflammatory",
        "inflammation",
    }
)

_CELLULAR_KEYWORDS: frozenset[str] = frozenset(
    {
        "cell",
        "cellular",
        "mitosis",
        "mitotic",
        "mitoses",
        "pleomorphism",
        "anaplasia",
        "anaplastic",
        "hyperchromasia",
        "atypia",
        "atypical",
        "dysplasia",
        "dysplastic",
        "hyperplasia",
        "hypercellular",
        "binucleated",
        "multinucleated",
        "giant cell",
        "apoptosis",
        "apoptotic",
        "karyorrhexis",
        "karyolysis",
        "oncocytic",
        "signet ring",
        "spindle",
        "epithelioid",
        "lymphocyte",
        "macrophage",
        "neutrophil",
        "eosinophil",
        "plasma cell",
        "mast cell",
        "histiocyte",
    }
)

_SUBCELLULAR_KEYWORDS: frozenset[str] = frozenset(
    {
        "nucleus",
        "nuclear",
        "nucleolus",
        "nucleolar",
        "chromatin",
        "cytoplasm",
        "cytoplasmic",
        "organelle",
        "vesicle",
        "mitochondria",
        "mitochondrial",
        "vacuole",
        "vacuolization",
        "inclusion",
        "intranuclear",
        "intracytoplasmic",
        "karyomegaly",
        "anisonucleosis",
    }
)

_EXTRACELLULAR_KEYWORDS: frozenset[str] = frozenset(
    {
        "matrix",
        "mucin",
        "mucinous",
        "collagen",
        "fibrin",
        "amyloid",
        "hyaline",
        "hyalinization",
        "basement membrane",
        "extracellular",
        "secretion",
        "deposit",
        "deposition",
        "fibrinoid",
        "fibrinous",
        "proteinaceous",
        "eosinophilic material",
    }
)

_STROMAL_KEYWORDS: frozenset[str] = frozenset(
    {
        "stromal",
        "stroma",
    }
)


# Pre-compile regex patterns for performance (avoids recompilation on every call)
_COMPILED_PATTERNS: dict[str, re.Pattern[str]] = {}


def _get_compiled_pattern(keyword: str) -> re.Pattern[str]:
    """Return a compiled word-boundary regex for the given keyword (cached)."""
    if keyword not in _COMPILED_PATTERNS:
        _COMPILED_PATTERNS[keyword] = re.compile(r"\b" + re.escape(keyword) + r"\b")
    return _COMPILED_PATTERNS[keyword]


def _matches_any(text: str, keywords: frozenset[str]) -> bool:
    lower = text.lower()
    return any(_get_compiled_pattern(kw).search(lower) for kw in keywords)


def classify_level(
    label: str,
    synonyms: Sequence[str] = (),
    definition: str | None = None,
) -> _Level:
    """Classify anatomical level of a morphology concept. First-match rule order.

    Args:
        label: Primary concept label.
        synonyms: Optional synonym strings.
        definition: Optional concept definition text.

    Returns:
        One of: 'stromal', 'subcellular', 'extracellular', 'cellular', 'tissue'.
    """
    texts = [label] + list(synonyms) + ([definition] if definition else [])
    combined = " ".join(texts)

    if _matches_any(combined, _STROMAL_KEYWORDS):
        return "stromal"
    if _matches_any(combined, _SUBCELLULAR_KEYWORDS):
        return "subcellular"
    if _matches_any(combined, _EXTRACELLULAR_KEYWORDS):
        return "extracellular"
    if _matches_any(combined, _CELLULAR_KEYWORDS):
        return "cellular"
    # Tissue is the safe default (most common value in the reference bank)
    return "tissue"
