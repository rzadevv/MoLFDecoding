# Data

Small metadata files needed to train and evaluate the model, and the concept
exclusion list. The slides and expression data themselves come from HEST-1k
(`conch_molf/download_hest.py`).

## splits/

HEST-1k slide splits used for Run 2 (504 slides).

| File | Contents |
|---|---|
| `train_split.csv`, `val_split.csv`, `test_split.csv` | 385 / 96 / 23 slides: `sample_id`, HEST file paths, OncoTree code (`oncotree_code_filled` is used for conditioning) |
| `run2_oncotree_code_list.txt` | the 29 OncoTree codes of the conditioning vocabulary |
| `final_504_integrity.tsv` | per-slide input check from the final audit (embedding, patch and expression files; patch count); all 504 pass |
| `RELATED_TEST_TRAIN_VAL_SPECIMENS.csv` | 16 pairs linking 4 test slides (the Xenium slides TENX99, TENX147, TENX148, TENX149) to training/validation slides from the same patient or study |

Slides do not overlap between splits, but related specimens do (see the last
file), so test results describe held-out slides, not held-out patients.

## genes/

`Hallmark_n_HVG50_genes.yaml` lists the 1,386 target genes: 1,169 genes from
the MSigDB Hallmark gene sets plus 217 highly variable genes, as selected in
upstream MoLF. The model was trained and evaluated on exactly this list; its
order matches the per-gene metrics in
`results/evaluation/runs/*/per_gene_metrics.csv`.

The Run-2 config refers to these files by their original locations; the
settings that point it here are listed in
[`../conch_molf/README.md`](../conch_molf/README.md#configuring-paths).

## concept_exclusions.csv

The 188 concepts left out of the filtered expert lexicon, each with a category
and a reason: 48 cell types from other species, 1 cytology-only finding, and
139 cell subsets defined only by molecular markers or functional state. It is
written by `scripts/filter_concepts.py` from the rules in that script; see
[`../docs/concept_bank.md`](../docs/concept_bank.md#known-problems-and-the-filtered-analysis).
