# Cross-bank comparison: reference bank vs harvested bank

**Generated:** 2026-05-28T22:59:11Z
**Reference bank:** 1006 concepts (provenance: reference)
**Harvested bank:** 4372 concepts (provenance: harvested)
**Merged:** 5378 concepts
**Matches found:** 258 (within-tier: 255; cross-tier: 3)
**Overall reference recall:** 25.6% (fraction of reference concepts with a harvested match)

## 1. Per-tier coverage

| Tier | Reference | Harvested | Matched | Ref-only | Harv-only | Ref recall |
|---|---:|---:|---:|---:|---:|---:|
| H1_morphology | 418 | 2383 | 112 | 306 | 2270 | 26.8% |
| H2_cell_type | 170 | 1623 | 101 | 69 | 1522 | 59.4% |
| H2_gene_program | 195 | 96 | 12 | 183 | 83 | 6.2% |
| H2_niche | 80 | 28 | 0 | 80 | 28 | 0.0% |
| H2_pathway | 143 | 242 | 33 | 110 | 208 | 23.1% |
| **Total** | **1006** | **4372** | **258** | **748** | **4111** | **25.6%** |

> **Cross-tier matches:** 3 concepts matched across different tiers (see Section 8). 1 are 'cells as morphology pattern' vs cell taxonomy — a systematic tier-assignment difference between expert curation and ontology-based automation.

## 1a. Primary Lexicon View (View A) — input to Clinical Lexicon generation

View A is the subset of the merged bank for Task 3.1 Clinical Lexicon generation. Concepts here are either from the reference bank, validated by cross-bank overlap, or passed label quality gates AND SapBERT similarity to the reference bank's same-tier vocabulary.

View A total: **3023** concepts (56.2% of merged bank)

| Tier | Reference | Harv matched | Harv ref-adjacent | View A total |
|---|---:|---:|---:|---:|
| H1_morphology | 418 | 112 | 773 | 1303 |
| H2_cell_type | 170 | 101 | 906 | 1177 |
| H2_gene_program | 195 | 12 | 38 | 245 |
| H2_niche | 80 | 0 | 4 | 84 |
| H2_pathway | 143 | 33 | 38 | 214 |
| **Total** | **1006** | **258** | **1759** | **3023** |

Exclusions (automated concepts not entering View A):

| Reason | Count |
|---|---:|
| Label too short (< 4 chars) | 20 |
| Label too long (> 8 tokens) | 42 |
| Excluded substring | 1 |
| No definition or synonyms | 0 |
| Low ref-neighbor similarity (< 0.70) | 2292 |

## 2. Match similarity distribution

| Tier | Mean | Median | p25 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| H1_morphology | 0.928 | 0.918 | 0.875 | 1.000 | 1.000 |
| H2_cell_type | 0.962 | 1.000 | 0.907 | 1.000 | 1.000 |
| H2_gene_program | 0.920 | 0.915 | 0.908 | 0.931 | 0.937 |
| H2_niche | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| H2_pathway | 0.921 | 0.919 | 0.887 | 0.950 | 0.965 |

## 3. Matched examples (random samples per tier)

These are concepts where the reference label has a close harvested equivalent.

### H1_morphology

| Reference label | Harvested label | Similarity |
|---|---|---:|
| traditional serrated adenoma | Serrated adenoma | 0.932 |
| serous cystadenoma | Serous neoplasm | 0.856 |
| epithelioid morphology | Epithelioid | 0.968 |
| intraepithelial lymphocytes | Intraepithelial lymphocyte | 0.986 |
| pseudopalisading necrosis | Pseudopalisading necrosis | 1.000 |

### H2_cell_type

| Reference label | Harvested label | Similarity |
|---|---|---:|
| memory B cell | Memory B cell | 1.000 |
| Schwann cell | Schwann cell | 1.000 |
| pancreatic acinar cell | Acinar cell | 0.944 |
| granulocyte-monocyte progenitor | Granulocyte monocyte progenitor cell | 0.957 |
| goblet cell | Goblet cell | 1.000 |

### H2_gene_program

| Reference label | Harvested label | Similarity |
|---|---|---:|
| KRAS signaling up program | KRAS signaling up | 0.929 |
| spermatogenesis program | Spermatogenesis | 0.903 |
| epithelial-mesenchymal transition program | Epithelial mesenchymal transition | 0.918 |
| estrogen response early program | Estrogen response early | 0.908 |
| chemokine signaling program | Chemokine signaling pathway | 0.923 |

### H2_pathway

| Reference label | Harvested label | Similarity |
|---|---|---:|
| PPAR signaling pathway | Ppar signaling pathway | 1.000 |
| canonical Wnt developmental pathway | Signaling by WNT | 0.866 |
| PI3K-AKT-mTOR pathway | MTOR signalling | 0.887 |
| TLR4 signaling pathway | Myd88 independent tlr4 cascade | 0.882 |
| Activin signaling pathway | Signaling by activin | 0.926 |

## 4. Reference-only concepts (random samples per tier)

Concepts in the reference bank that the harvested bank did not match.
These represent gaps in automated curation.

### H1_morphology

- hemosiderin-laden macrophages
- tumor budding low grade
- apoptotic bodies
- gemistocytic astrocytes
- basaloid squamous cell carcinoma of head and neck

### H2_cell_type

- adaptive NK cell
- inflammatory CAF
- N2 pro-tumor neutrophil
- mature LAMP3-positive dendritic cell
- red pulp macrophage

### H2_gene_program

- pyrimidine synthesis program
- protein secretion meta-program
- basement membrane assembly program
- glycolysis program
- cilia meta-program

### H2_niche

- senescence niche
- alveolar niche
- mast cell-enriched niche
- subclonal mixing zone
- stem cell niche

### H2_pathway

- NLRP3 inflammasome pathway
- mitophagy pathway
- androgen receptor signaling pathway
- VEGF vascular development pathway
- purine metabolism pathway

## 5. Harvested-only concepts (random samples per tier)

Concepts the harvested bank contains with no reference near-equivalent.
These represent additions beyond the reference vocabulary.

### H1_morphology

- Hyperplastic scar
- Subpleural nodule
- Regenerative inflammation
- Diffuse-type
- Hamartoma

### H2_cell_type

- Glandular cell of the large intestine
- Colony forming unit – hill cell
- Appendix glandular cell
- T1 B cell
- Arachnoid barrier cell (mmus)

### H2_gene_program

- Interferon gamma response
- Fatty acid metabolism
- Tetrahydrobiopterin bh4 synthesis recycling salvage and regulation
- Focal calcium deposition
- Extracellular macromolecule aggregate alteration

### H2_niche

- Overlapping/multiple zones of prostate
- Medullary thymic epithelial cell type 1
- Non-contiguous intracranial parenchymal
- Tumor droplets present
- Periderm

### H2_pathway

- Renin angiotensin system
- Regulation of signaling by cbl
- Beta alanine metabolism
- Sos mediated signalling
- Adenylate cyclase inhibitory pathway

## 6. Top 20 hardest reference gaps

Reference concepts whose nearest harvested neighbor has the LOWEST similarity.
These are the highest-priority gaps for future automated curation work.

| Reference concept | Tier | Nearest harvested | Sim |
|---|---|---|---:|
| treatment effect general | H1_morphology | Physiological factors | 0.462 |
| normal fundic gland mucosa | H1_morphology | Gastric carcinoma in situ | 0.495 |
| drug efflux ABC transporter program | H2_gene_program | Unfolded protein response | 0.504 |
| growth factor independence program | H2_gene_program | Solid growth pattern | 0.507 |
| cross-presentation program | H2_gene_program | Large nested pattern | 0.513 |
| Homer Wright rosettes | H1_morphology | Flexner-wintersteiner rosette formation | 0.513 |
| immune cold desert niche | H2_niche | Blood island | 0.517 |
| extranodal extension in lymph node | H1_morphology | Atypical lymph node | 0.520 |
| purine salvage program | H2_gene_program | Rnd1 gtpase cycle | 0.524 |
| cytokine storm program | H2_gene_program | Alpha-beta T cell activation by superantigen | 0.525 |
| cDC1 | H2_cell_type | Mcf-7 | 0.526 |
| normal pyloric gland mucosa | H1_morphology | Atypical glandular epithelium resembling the normal endometrium present | 0.527 |
| tryptophan-kynurenine metabolism program | H2_gene_program | Interleukin 36 pathway | 0.527 |
| tryptophan IDO-TDO catabolism pathway | H2_pathway | Reactive oxygen species pathway | 0.528 |
| metal ion response meta-program | H2_gene_program | Miro gtpase cycle | 0.529 |
| wound healing program | H2_gene_program | Mucinous fibroplasia | 0.531 |
| normal peripheral nerve | H1_morphology | Schwann cell | 0.532 |
| miRNA processing pathway | H2_pathway | Miro gtpase cycle | 0.533 |
| immune evasion meta-program | H2_gene_program | Pre-B cell allelic exclusion | 0.539 |
| laminin program | H2_gene_program | Basement membrane-like material | 0.539 |

## 7. Top 20 most-novel harvested additions

Harvested concepts whose nearest reference neighbor has the LOWEST similarity.
Sanity check: are these legitimately novel, or noise that survived curation?

| Harvested concept | Tier | Nearest reference | Sim |
|---|---|---|---:|
| Calcitonin like ligand receptors | H2_pathway | R-loop resolution pathway | 0.409 |
| Eicosanoids | H2_gene_program | inflammatory CAF | 0.431 |
| Paranoia | H1_morphology | immune desert morphology | 0.454 |
| Physiological factors | H2_pathway | treatment effect general | 0.462 |
| E3 ubiquitin ligases ubiquitinate target proteins | H2_pathway | proteasome degradation pathway | 0.462 |
| Rsv host interactions | H2_pathway | type II interferon signaling pathway | 0.474 |
| Huntingtons disease | H2_pathway | Helicobacter-associated gastritis | 0.475 |
| Chl1 interactions | H2_pathway | Polycomb repressive complex pathway | 0.477 |
| Thecoma | H1_morphology | chicken-wire vasculature | 0.481 |
| Synaptic adhesion like molecules | H2_pathway | cell-cell adhesion program | 0.484 |
| Neurotransmitter receptors and postsynaptic signal transmission | H2_pathway | TCR signaling pathway | 0.485 |
| Crystal of reinke | H1_morphology | ceroid pigment | 0.488 |
| Synthesis of diphthamide eef2 | H2_pathway | RNA polymerase II elongation pathway | 0.491 |
| Nuclear import of rev protein | H2_pathway | nucleotide excision repair pathway | 0.491 |
| Synthesis secretion and deacylation of ghrelin | H2_gene_program | glutamine metabolism program | 0.496 |
| Regulated proteolysis of p75ntr | H2_pathway | NGF-Trk signaling pathway | 0.498 |
| Gigantiform cementoma | H1_morphology | foamy gland carcinoma | 0.498 |
| Gigantiform cementoma | H1_morphology | foamy gland carcinoma | 0.498 |
| Synthesis of ip2 ip and ins in the cytosol | H2_pathway | Activin signaling pathway | 0.500 |
| Pigment dispersion syndrome | H1_morphology | anthracotic pigment deposition | 0.500 |

## 8. Cross-tier matches

3 concepts matched across different tiers (above 0.92 similarity). These indicate either (a) ambiguous concepts spanning tiers, or (b) tier-assignment disagreements between banks.

| Reference label (tier) | Harvested label (tier) | Sim |
|---|---|---:|
| thyroid follicular cell (H2_cell_type) | Thyroid follicular cell (H1_morphology) | 1.000 |
| chemokine signaling program (H2_gene_program) | Chemokine signaling pathway (H2_pathway) | 0.923 |
| glycolysis pathway (H2_pathway) | Glycolysis (H2_gene_program) | 0.962 |

## 9. Discussion

- Tier-by-tier coverage and what gaps tell us about source diversity.
- Why H2_niche has 0% reference recall (documented harvest-side limitation).
- Why H2_gene_program has lower reference recall than H1_morphology.
- Cross-tier match patterns — are they consistent enough to suggest one bank's
  tier assignment is wrong?
- Harvested-only concepts — net positive or noise?

---

*Report generated by molf-interp v0.0.1 | Reference: data/cache/concept_bank/reference | Harvested: data/cache/concept_bank/harvested*
