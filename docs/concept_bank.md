# Concept bank

How the concept vocabulary used to interpret the experts was built, audited
and cleaned. The pipeline code is in [`../interpretability/`](../interpretability/).

## Sources

**Reference bank.** 1,006 concepts curated by a pathologist, in five tiers:
H1 morphology, and H2 cell type, niche, gene program and pathway
(`interpretability/data/concept_bank/`).

**Harvested bank.** Candidate concepts pulled from NCIt, SNOMED CT, Cell
Ontology and GO (via BioPortal), MSigDB gene sets and PubMed abstracts, then
filtered by the seven-stage curation pipeline to 4,372 concepts. UBERON
(also via BioPortal) supplies the organ names used during curation.

## View A

Matching the two banks gives a merged bank of 5,378 concepts
(`interpretability/data/outputs/comparison/report.md`). **View A** is the
subset used for interpretation: every reference concept, harvested concepts
that matched one, and harvested concepts that pass the label-quality checks and
are close (SapBERT similarity) to the reference vocabulary of the same tier.

| Tier | View A concepts |
|---|---:|
| H1 morphology | 1,303 |
| H2 cell type | 1,177 |
| H2 gene program | 245 |
| H2 pathway | 214 |
| H2 niche | 84 |
| **Total** | **3,023** |

Every concept has a human and a mouse prompt built from tier-specific templates,
e.g. *"Histopathological feature: comedo necrosis observed in H&E stained human
tissue"*.

## Audit

An audit of an earlier View A (2,758 concepts, 22 May 2026, statistics from
`interpretability/scripts/analyze_view_a.py`) found the prompts well formed and
the tier assignments sound, and raised five issues:

1. Four morphology entries were clinical labels rather than visual features
   (tumour grade, microsatellite status), e.g. *Grade 2 clear cell renal cell
   carcinoma*.
2. 201 same-tier duplicate names, mostly the same concept arriving from two
   sources (e.g. *Caseous necrosis* from both the reference bank and SNOMED CT).
3. 164 morphology terms name a disease rather than a visual pattern
   (e.g. *encapsulated tumor*, *nuclear atypia*); many are borderline and were
   marked for pathologist review.
4. Gene-program and pathway concepts describe transcriptional states that are
   not visible on H&E, so they do not belong in an image-text comparison.
5. Some concepts are human-specific (e.g. Gleason patterns; 13 in the final
   View A) and have no meaningful mouse prompt.

## Cleaning before embedding

`molf-interp vlm prepare-view-a` addresses issues 1, 2, 4 and 5
deterministically. On the final View A:

| Step | Effect |
|---|---|
| Drop clinical labels | none left to drop (the four were already removed upstream; the check remains as a guard) |
| Collapse same-tier duplicates by normalised name, keeping the reference concept, then the lowest ID | 115 dropped (60 morphology, 54 cell type, 1 pathway) |
| Split tiers into a *visual* set (morphology, cell type, niche) and a *transcriptomic* set (gene program, pathway) | 2,450 visual, 458 transcriptomic |
| Remove mouse prompts of human-specific concepts | 13 concepts |

This leaves 2,908 concepts and 5,803 prompts, all embedded with CONCH
(`interpretability/data/outputs/vlm/conch_text_embeddings/`). The expert
analysis uses the 2,450 visual concepts with their human prompts: 1,243
morphology, 1,123 cell type and 84 niche.

The disease-label terms (issue 3) were not reviewed by a pathologist and remain
in the morphology tier. They can appear among an expert's top concepts (e.g.
*Mixed medullary-follicular carcinoma* for Expert 0) and should be read as
"patches resembling tissue described this way", not as a diagnosis.

## Known problems and the filtered analysis

A review of the 2,450 scored concepts found three kinds of entries that cannot
be valid H&E findings:

| Problem | Concepts | Examples |
|---|---:|---|
| Cell types from other species | 48 | *Eggshell secreting cell*, *Fat body cell* (insects), *Nucleated thrombocyte* (non-mammalian), *Bergmann glial cell (mmus)* |
| Cytology-only finding | 1 | *Clue cell* (a cervicovaginal smear finding) |
| Cell subsets defined only by markers or functional state | 139 | *CD4-positive, alpha-beta memory T cell*, *regulatory T cell*, *TREM2-positive macrophage* |

All 48 cross-species entries come from the Cell Ontology, which covers every
animal; 33 were identified by reading all 895 Cell Ontology entries and 15
carry a species-restricted label such as "(mmus)" or "(sensu arthropoda)". The
curation did not filter by species, and its "visual filter" compares concept
*names* with the reference vocabulary (SapBERT similarity), so it measures how
much a name resembles pathology terminology rather than whether the concept is
visible. The marker-defined subsets are 115 harvested concepts and 24 from the
reference bank; on H&E they look like their parent cell type.

Several of these reached the reported lexicon: 6 of the 60 top-10 entries in
`results/summary/expert_lexicon.csv` (two T-cell subsets for Expert 0, *Clue
cell* for Experts 3 and 5, two insect cell types for Expert 4).

Because a concept's score does not depend on the other concepts,
`scripts/filter_concepts.py` recomputes the rankings without these 188
concepts from the per-slide scores. It first checks that, with nothing removed,
it reproduces the frozen aggregate exactly (global ranks, per-slide top-10
counts, stability). The frozen results are not modified; the filtered lexicon
is `results/summary/expert_lexicon_filtered.csv` (Fig. 6b) and the exclusion
list, with a reason for every entry, is `data/concept_exclusions.csv`.

Removing them leaves the themes unchanged: Experts 1 and 2 keep the same top
ten, Expert 4 becomes more consistently glandular and mucinous, and Expert 3
more clearly inflammatory. Cross-slide stability changes by at most 0.014 in
mean rank correlation for any expert and pair type.

Other weaknesses are not addressed by the filter:

- The per-tier thresholds in `configs/curation/calibrated_thresholds.yaml` are
  identical for all tiers and equal the defaults, so no tier-specific
  calibration took effect.
- 168 cell-type prompts list marker genes, which an image-text model cannot
  see, and every concept is scored with a single prompt rather than an
  ensemble of phrasings.
- The disease-label terms (audit issue 3) were not reviewed.

Fixing these at the source (a species filter for the Cell Ontology, a
visibility review, recalibrated thresholds, prompt ensembles) requires
re-embedding the concepts and re-running the expert analysis.

