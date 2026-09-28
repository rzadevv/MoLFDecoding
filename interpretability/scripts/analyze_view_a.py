"""Print summary statistics of View A for the concept-bank audit.

Reports tier and species counts, duplicate names, and terms matching the
clinical-label and generic-disease-label patterns (see docs/concept_bank.md).
Run from the interpretability/ directory: ``uv run python scripts/analyze_view_a.py``.
"""

from __future__ import annotations

import re

import pandas as pd

DATA = "data/outputs/comparison"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

mc = pd.read_parquet(f"{DATA}/merged_concepts.parquet")
mp = pd.read_parquet(f"{DATA}/merged_prompts.parquet")
va = mc[mc["in_primary_lexicon"]].copy()
va_ids = set(va["concept_id"])
vp = mp[mp["concept_id"].isin(va_ids)].copy()

print(f"View A concepts: {len(va)}")
print(f"View A prompts:  {len(vp)}")

# ---------------------------------------------------------------------------
# Tier counts
# ---------------------------------------------------------------------------

print("\n=== TIER COUNTS ===")
print(va["tier"].value_counts())

# ---------------------------------------------------------------------------
# Species breakdown
# ---------------------------------------------------------------------------

print("\n=== SPECIES BREAKDOWN (prompts) ===")
print(vp["species"].value_counts())
print("\nspecies_status breakdown:")
print(vp["species_status"].value_counts())

# ---------------------------------------------------------------------------
# H1 morphology sample
# ---------------------------------------------------------------------------

print("\n=== H1 MORPHOLOGY NAMES (sorted, first 200) ===")
h1 = va[va["tier"] == "H1_morphology"].sort_values("concept_name")
for n in h1["concept_name"].tolist():
    print(f"  {n}")

# ---------------------------------------------------------------------------
# H2 cell type sample
# ---------------------------------------------------------------------------

print("\n=== H2_CELL_TYPE NAMES (sorted) ===")
ct = va[va["tier"] == "H2_cell_type"].sort_values("concept_name")
for n in ct["concept_name"].tolist():
    print(f"  {n}")

# ---------------------------------------------------------------------------
# H2 niche sample
# ---------------------------------------------------------------------------

print("\n=== H2_NICHE NAMES ===")
niche = va[va["tier"] == "H2_niche"].sort_values("concept_name")
for n in niche["concept_name"].tolist():
    print(f"  {n}")

# ---------------------------------------------------------------------------
# H2 gene program sample
# ---------------------------------------------------------------------------

print("\n=== H2_GENE_PROGRAM NAMES ===")
gp = va[va["tier"] == "H2_gene_program"].sort_values("concept_name")
for n in gp["concept_name"].tolist():
    print(f"  {n}")

# ---------------------------------------------------------------------------
# H2 pathway sample
# ---------------------------------------------------------------------------

print("\n=== H2_PATHWAY NAMES ===")
pw = va[va["tier"] == "H2_pathway"].sort_values("concept_name")
for n in pw["concept_name"].tolist():
    print(f"  {n}")

# ---------------------------------------------------------------------------
# Prompts analysis
# ---------------------------------------------------------------------------

print("\n=== PROMPTS — by species and tier ===")
print(vp.groupby(["tier", "species"]).size())

print("\n=== PROMPTS — species_status ===")
print(vp["species_status"].value_counts())

print("\n=== SAMPLE PROMPTS — H1_morphology human ===")
h1_human = vp[(vp["tier"] == "H1_morphology") & (vp["species"] == "Homo_sapiens")]
for _, row in h1_human.head(15).iterrows():
    print(f"  [{row['concept_name']}] {row['prompt_text']}")

print("\n=== SAMPLE PROMPTS — H2_cell_type human ===")
ct_human = vp[(vp["tier"] == "H2_cell_type") & (vp["species"] == "Homo_sapiens")]
for _, row in ct_human.head(15).iterrows():
    print(f"  [{row['concept_name']}] {row['prompt_text']}")

print("\n=== SAMPLE PROMPTS — H2_gene_program human ===")
gp_human = vp[(vp["tier"] == "H2_gene_program") & (vp["species"] == "Homo_sapiens")]
for _, row in gp_human.head(15).iterrows():
    print(f"  [{row['concept_name']}] {row['prompt_text']}")

# ---------------------------------------------------------------------------
# Clinical / bad-term flags
# ---------------------------------------------------------------------------

print("\n=== BAD TERM SCREENING ===")

# Terms to flag
clinical_patterns = [
    r"\b(prognos|survival|mortality|outcome|response to treatment|clinical stage|grade [0-9]|tnm|ajcc)\b",  # noqa: E501
    r"\b(patient|pre-operative|post-operative|therapeutic|chemoradiation|recurrence|relapse)\b",
    r"\b(ki-67|p53|her2|er status|pr status|pd-l1|microsatellite|mutation load)\b",
    r"\b(biopsy|resection|surgical|mastectomy|lumpectomy| colectomy|excision)\b",
    r"\b(imaging|ct scan|mri pet|mammogram|ultrasound finding|lesion on imaging)\b",
    r"\b(grade [0-9]|stage [ivx]+|t[0-9]n[0-9]m[0-9])\b",
]

generic_patterns = [
    r"\b(tumor|cancer|malignancy|neoplasm|disease|lesion|abnormalit|abnormal growth)\b",
    r"\b(dysplasia|carcinoma in situ|atypia)\b",
]

assay_patterns = [
    r"\b(ihc|immunohistochemistry|immunostain|marker expression|expression level|staining intensity)\b",  # noqa: E501
    r"\b(pcr|sequencing|ngs|mutation|braf|egfr|alk rearrangement)\b",
]

# Flag H1 terms matching clinical
print("\n--- H1 clinical-pattern matches ---")
h1_names = va[va["tier"] == "H1_morphology"]["concept_name"].tolist()
clinical_h1 = []
for name in h1_names:
    for p in clinical_patterns:
        if re.search(p, name, re.IGNORECASE):
            clinical_h1.append(name)
            break
print(f"Clinical-pattern H1 terms ({len(clinical_h1)}):")
for n in clinical_h1[:50]:
    print(f"  {n}")

print("\n--- H1 generic-pattern matches ---")
generic_h1 = []
for name in h1_names:
    for p in generic_patterns:
        if re.search(p, name, re.IGNORECASE):
            generic_h1.append(name)
            break
print(f"Generic-pattern H1 terms ({len(generic_h1)}):")
for n in generic_h1[:50]:
    print(f"  {n}")

print("\n--- H1 assay-pattern matches ---")
assay_h1 = []
for name in h1_names:
    for p in assay_patterns:
        if re.search(p, name, re.IGNORECASE):
            assay_h1.append(name)
            break
print(f"Assay-pattern H1 terms ({len(assay_h1)}):")
for n in assay_h1:
    print(f"  {n}")

# ---------------------------------------------------------------------------
# Prompts with weak/no context
# ---------------------------------------------------------------------------

print("\n=== WEAK PROMPTS ===")
human_prompts = vp[vp["species"] == "Homo_sapiens"].copy()
# Find prompts that are just the concept name repeated
weak = human_prompts[
    human_prompts["prompt_text"].str.strip() == human_prompts["concept_name"].str.strip()
]
print(f"Prompts identical to concept name: {len(weak)}")

# Long prompts
human_prompts["prompt_len"] = human_prompts["prompt_text"].str.len()
long_prompts = human_prompts[human_prompts["prompt_len"] > 300]
print(f"Prompts > 300 chars: {len(long_prompts)}")
for _, row in long_prompts.head(10).iterrows():
    print(
        f"  [{row['concept_name']}] ({row['tier']}) len={row['prompt_len']}: {row['prompt_text'][:200]}"  # noqa: E501
    )

# ---------------------------------------------------------------------------
# Cross-tier redundancy (within same tier, similar names)
# ---------------------------------------------------------------------------

print("\n=== NEAR-DUPLICATE NAMES (within same tier) ===")
va_sorted = va.sort_values(["tier", "concept_name"])
va_sorted["name_lower"] = va_sorted["concept_name"].str.lower()
dups = va_sorted[va_sorted.duplicated(subset=["tier", "name_lower"], keep=False)]
print(f"Case-insensitive duplicate names (same tier): {len(dups)}")
for _, row in dups.iterrows():
    print(f"  [{row['tier']}] {row['concept_name']} ({row['concept_id']})")

# Check singular/plural pairs within H1
print("\n--- Singular/plural H1 candidates ---")
h1_names_lower = {n.lower() for n in h1["concept_name"].tolist()}
for name in h1["concept_name"].tolist():
    sing = name.rstrip("s")
    plur = name + "s"
    if sing in h1_names_lower and plur in h1_names_lower:
        print(f"  singular/plural pair: '{sing}' / '{plur}'")

# ---------------------------------------------------------------------------
# Context-required concepts (niche, gene programs)
# ---------------------------------------------------------------------------

print("\n=== CONTEXT-REQUIRED CONCEPTS ===")
print(f"H2_niche terms ({len(niche)}):")
for _, row in niche.iterrows():
    print(f"  {row['concept_name']}")

print(f"\nH2_gene_program terms ({len(gp)}):")
for _, row in gp.iterrows():
    print(f"  {row['concept_name']}")

# ---------------------------------------------------------------------------
# Tissue/organ mismatch — human vs mouse
# ---------------------------------------------------------------------------

print("\n=== HUMAN-SPECIFIC CLINICAL TERMS IN MOUSE PROMPTS ===")
mouse_prompts = vp[vp["species"] == "Mus_musculus"]
human_specific = mouse_prompts[
    mouse_prompts["prompt_text"].str.contains(r"\bhuman\b", case=False, na=False)
]
print(f"Mouse prompts mentioning 'human': {len(human_specific)}")
for _, row in human_specific.head(10).iterrows():
    print(f"  [{row['concept_name']}] {row['prompt_text']}")

# ---------------------------------------------------------------------------
# Level mismatches (if level column present)
# ---------------------------------------------------------------------------

print("\n=== LEVEL ANALYSIS ===")
print(va["level"].value_counts(dropna=False))
