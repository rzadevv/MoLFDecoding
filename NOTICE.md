# Third-party notices

This repository builds on, or uses, the code, models and data below. Several of
them are licensed for non-commercial use only.

## Code

**MoLF (Mixture of Latent Flows)**, © Su Hu et al., CC BY-NC 4.0,
https://github.com/susuhu/MoLF. The full license is in
`docs/upstream_molf_license.txt`. `conch_molf/` is built on part of this
codebase. Relative to upstream:

- unchanged: `models/baseline_models.py`
- modified: `train.py`, `models/flow_matching.py`, `utils/custom_dataset.py`,
  `utils/hest_utils.py`, `utils/training_utils.py`, `utils/pfm_preprocesing.py`,
  `utils/other_utils.py` (CONCH input features, GeneVAE-v2 latent target, no
  sequence positional encoding, deterministic Monte-Carlo validation,
  checkpoint and resume handling, configurable paths, removal of unused code)
- new: `models/genevae_v2.py`, `configs/training_config_conch_run2.yml`,
  `download_hest.py`

The upstream model weights, data and figures are not included.

## Models

- **CONCH v1**, © Mahmood Lab (Lu et al., *Nature Medicine* 2024),
  CC BY-NC-ND 4.0, https://github.com/mahmoodlab/CONCH. Image features for
  MoLF and text embeddings of the concept bank. Weights are not included.
- **UNI, UNI-v2, H-optimus-0, Virchow2, Prov-GigaPath**: optional alternative
  feature extractors in `conch_molf/utils/pfm_preprocesing.py`, each under its
  own (mostly gated, non-commercial) license. Not used for the reported
  results.
- **SapBERT** (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`), © Cambridge
  LTL (Liu et al., NAACL 2021), Apache 2.0. Embedding of concept labels during
  curation.
- **scispaCy** with `en_ner_bionlp13cg_md`, © Allen Institute for AI,
  Apache 2.0, https://allenai.github.io/scispacy/. Entity extraction from
  PubMed abstracts.
- **Qwen2.5-3B-Instruct**, © Alibaba Cloud, Qwen Research License
  (non-commercial), run locally via Ollama. Adjudication of borderline
  concepts during curation.

## Data

- **HEST-1k** (`MahmoodLab/hest`, metadata v1.1.0), © Mahmood Lab,
  CC BY-NC-SA 4.0. Source of all slides. Only slide identifiers and results
  derived from the data are included here.
- **Reference concept bank** (`interpretability/data/concept_bank/`),
  curated by a pathologist and provided through the project supervisor; used
  with permission for this research.
- **PubMed abstracts**, retrieved via NCBI E-utilities, National Library of
  Medicine, public domain.

## Ontologies and gene sets

- **NCI Thesaurus (NCIt)**, National Cancer Institute, public domain; via
  BioPortal.
- **SNOMED CT**, © SNOMED International, used under an affiliate license; via
  BioPortal.
- **MeSH**, National Library of Medicine, public domain; via BioPortal and
  PubMed.
- **Cell Ontology (CL)**, OBO Foundry, CC BY 4.0.
- **Gene Ontology (GO)**, © Gene Ontology Consortium, CC BY 4.0.
- **UBERON**, OBO Foundry, CC BY 3.0.
- **MSigDB** Hallmark, C2 Reactome and C2 KEGG legacy collections, © Broad
  Institute, MSigDB license; includes Reactome data (CC BY 4.0) and KEGG data
  (© Kanehisa Laboratories).
