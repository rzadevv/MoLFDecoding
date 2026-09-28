"""Stage 6: concept_id minting + label canonicalization."""

from __future__ import annotations

import re
import time

import pandas as pd
from loguru import logger

from molf_interp.curation.config import CurationConfig

_TIER_PREFIXES: dict[str, str] = {
    "H1_morphology": "MOR",
    "H2_cell_type": "CTY",
    "H2_niche": "NIC",
    "H2_gene_program": "GPR",
    "H2_pathway": "PWY",
}

# Anchored patterns (match against start of lowercased name).
_NOISE_ANCHOR: list[re.Pattern[str]] = [
    re.compile(r"^\d+[pq]"),  # chromosomal loci: 1p34, 2p24, 4p, 3p21.31
    re.compile(r"^[XYxy][pq]\d*$"),  # sex chromosome arms: Xp, Yq, yp
    re.compile(r"^-"),  # negative-prefixed truncations: -negative tumors
    re.compile(r"^[a-z]\d{1,2}$"),  # AJCC staging codes: r0, s0, s3 (not 3-char abbrevs)
]

# Substring patterns (searched anywhere in original-case name).
_NOISE_SUBSTR: list[re.Pattern[str]] = [
    re.compile(r"\bTNM\b"),  # staging designations: "breast Cancer cN2 TNM Finding v8"
    re.compile(r"\bgrade\s+(?:[1-4]|[1-4]/[1-4])\b", re.IGNORECASE),
    re.compile(r"\b(?:who|nci)\s+grade\s+[1-4]\b", re.IGNORECASE),
    re.compile(r"\bctcae\b", re.IGNORECASE),
    re.compile(r"\bsurgical margin\b", re.IGNORECASE),
    re.compile(r"\bimaging finding\b", re.IGNORECASE),
    re.compile(r"\bmicrosatellite (?:stable|instability|instability signature)\b", re.IGNORECASE),
    re.compile(r"\bby immunohistochemistry\b", re.IGNORECASE),
    re.compile(r"\bimmunohistochemistry (?:present|[0-3]\+)\b", re.IGNORECASE),
]

# Map from uppercase token to its canonical display form.
# All-caps acronyms map to themselves; mixed-case forms are listed explicitly.
_CANON_CASING: dict[str, str] = {
    # Cancer subtypes / clinical
    "RCC": "RCC",
    "DCIS": "DCIS",
    "STIC": "STIC",
    "GIST": "GIST",
    "PECOMA": "PEComa",
    "NSCLC": "NSCLC",
    "SCLC": "SCLC",
    "AML": "AML",
    "CML": "CML",
    "ALL": "ALL",
    "CLL": "CLL",
    "HCC": "HCC",
    "ICC": "ICC",
    "CRC": "CRC",
    "GBM": "GBM",
    "LGG": "LGG",
    "TNBC": "TNBC",
    "HPV": "HPV",
    "EBV": "EBV",
    # T/B/NK cell designators (single-letter or short)
    "T": "T",
    "B": "B",
    "NK": "NK",
    # CD markers
    "CD3": "CD3",
    "CD4": "CD4",
    "CD8": "CD8",
    "CD20": "CD20",
    "CD45": "CD45",
    "CD68": "CD68",
    "CD138": "CD138",
    "FOXP3": "FoxP3",
    # Gene / protein names
    "MYC": "MYC",
    "BRCA": "BRCA",
    "TP53": "TP53",
    "EGFR": "EGFR",
    "KRAS": "KRAS",
    "BRAF": "BRAF",
    "PIK3CA": "PIK3CA",
    "ALK": "ALK",
    "ROS1": "ROS1",
    "ER": "ER",
    "PR": "PR",
    "HER2": "HER2",
    # Immune checkpoints (hyphened — matched as whole-word tokens only)
    "PD-L1": "PD-L1",
    "PD-1": "PD-1",
    "CTLA-4": "CTLA-4",
    # Growth factors
    "VEGF": "VEGF",
    "EGF": "EGF",
    "FGF": "FGF",
    "PDGF": "PDGF",
    "TGF": "TGF",
    "IGF": "IGF",
    # Signaling pathways
    "WNT": "WNT",
    "NF-KB": "NF-kB",
    "STAT": "STAT",
    "JAK": "JAK",
    "MAPK": "MAPK",
    "PI3K": "PI3K",
    "AKT": "AKT",
    "MTOR": "mTOR",
    # Imaging / methods
    "MRI": "MRI",
    "PET": "PET",
    "FDG": "FDG",
    "FISH": "FISH",
    "IHC": "IHC",
    "PCR": "PCR",
    "RNA": "RNA",
    "DNA": "DNA",
}

# Non-hyphen keys available for prefix matching (e.g. BRCA → BRCA1, CD8 → CD8+).
_PREFIX_CANON: list[tuple[str, str]] = [(k, v) for k, v in _CANON_CASING.items() if "-" not in k]


def _is_noise_concept(name: str) -> bool:
    lowered = name.lower()
    return any(p.match(lowered) for p in _NOISE_ANCHOR) or any(
        p.search(name) for p in _NOISE_SUBSTR
    )


def _canon_part(part: str) -> str:
    """Return canonically-cased form of a single non-hyphen token fragment."""
    upper = part.upper()
    if upper in _CANON_CASING:
        return _CANON_CASING[upper]
    # Prefix match: token starts with a known acronym followed only by digits/+
    for key, val in _PREFIX_CANON:
        if upper.startswith(key):
            suffix = part[len(key) :]
            if re.match(r"^[\d+]*$", suffix):
                return val + suffix
    return part.lower()


def _canon_word(word: str) -> str:
    """Return canonically-cased form of one whitespace-separated word."""
    upper = word.upper()
    # Whole-word match handles hyphened acronyms (PD-L1, NF-kB, CTLA-4).
    if upper in _CANON_CASING:
        return _CANON_CASING[upper]
    if "-" in word:
        return "-".join(_canon_part(p) for p in word.split("-"))
    return _canon_part(word)


def _canonicalize_label(label: str) -> str:
    """Apply sentence-case while preserving known acronyms and initialisms.

    Strategy:
      1. Lowercase the entire string.
      2. Split on whitespace; process each word with ``_canon_word``.
      3. Capitalize the first character of the first non-empty word.

    Examples:
      ``"pulmonary Infiltrate"`` → ``"Pulmonary infiltrate"``
      ``"breast Columnar Cell"`` → ``"Breast columnar cell"``
      ``"CD8-positive T cell"`` → ``"CD8-positive T cell"``
      ``"BRCA1 germline mutation"`` → ``"BRCA1 germline mutation"``

    Args:
        label: Raw preferred_label string.

    Returns:
        Sentence-cased label with known acronyms preserved.
    """
    if not label:
        return label
    words = label.lower().split()
    result = [_canon_word(w) for w in words]
    if result and result[0]:
        result[0] = result[0][0].upper() + result[0][1:]
    return " ".join(result)


def run_stage_6(df: pd.DataFrame, config: CurationConfig) -> pd.DataFrame:
    """Mint concept IDs and canonicalize labels.

    Args:
        df: DataFrame from Stage 5 with 'tier' column.
        config: CurationConfig instance.

    Returns:
        DataFrame with 'concept_id' column added and labels canonicalized.
    """
    t0 = time.monotonic()
    cfg = config.canonicalize
    offset = cfg.id_offset
    logger.info("Stage 6: canonicalizing {} rows (offset={})…", len(df), offset)

    # Drop noise H1 concepts (chromosomal loci, staging codes, truncated labels)
    before = len(df)
    h1_mask = df["tier"] == "H1_morphology"
    noise_mask = h1_mask & df["preferred_label"].apply(_is_noise_concept)
    if noise_mask.any():
        dropped = df[noise_mask]["preferred_label"].tolist()
        logger.info("  Dropping {} noise H1 concepts: {}", len(dropped), dropped)
        df = df[~noise_mask].copy()
    dropped_count = before - len(df)
    if dropped_count:
        logger.info("  {} rows remain after noise filter", len(df))

    # Sort for reproducibility: by tier then label alphabetically
    df = df.sort_values(["tier", "preferred_label"], ignore_index=True)

    # Mint IDs per tier
    tier_counters: dict[str, int] = {}
    concept_ids: list[str] = []

    for _, row in df.iterrows():
        tier = str(row["tier"])
        prefix = _TIER_PREFIXES.get(tier, "UNK")
        n = tier_counters.get(tier, 0)
        tier_counters[tier] = n + 1
        concept_id = f"{prefix}-{offset + n + 1:05d}"
        concept_ids.append(concept_id)

    df = df.copy()
    df["concept_id"] = concept_ids

    # Sentence-case preferred_label with acronym preservation
    if cfg.sentence_case:
        df["preferred_label"] = df["preferred_label"].apply(_canonicalize_label)

    # Deduplicate synonyms case-insensitively, cap at 20
    def _clean_syns(syns: object) -> list[str]:
        if not isinstance(syns, list):
            return []
        seen: set[str] = set()
        result: list[str] = []
        for s in syns:
            sl = str(s).lower()
            if sl not in seen:
                seen.add(sl)
                result.append(str(s))
        return sorted(result)[:20]

    df["synonyms"] = df["synonyms"].apply(_clean_syns)

    elapsed = time.monotonic() - t0
    logger.info("Stage 6 done: {} concepts minted, {:.1f}s", len(df), elapsed)
    return df
