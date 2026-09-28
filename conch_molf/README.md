# CONCH-MoLF

Model, training and data-preparation code for CONCH-MoLF, the model reported in
this repository (*Run 2* in the run records). It is derived from the upstream
[MoLF](https://github.com/susuhu/MoLF) codebase (CC BY-NC 4.0); the changes are
listed in [`../NOTICE.md`](../NOTICE.md).

## Model

- **Input:** frozen CONCH image embeddings of 224×224 H&E patches (512-d,
  L2-normalised) with patch-context attention. Real x/y coordinates are not
  model inputs.
- **Target space:** the 128-d latent space of a frozen GeneVAE-v2
  (`models/genevae_v2.py`, 1,386 genes, best epoch 929). The flow is trained on
  posterior samples of the latent.
- **Velocity field:** Mixture of Experts with 6 experts (dim 256, 4 heads,
  1 layer each) and a top-2 router.
- **Conditioning:** image features and OncoTree cancer type, trained with 10%
  condition dropout for classifier-free guidance.
- **Training:** 385 slides, early stopping on a fixed Monte-Carlo validation
  loss (patience 80, min_delta 1e-4); best epoch 239 of 319.
- **Inference:** Euler, 2 steps, guidance scale 1.5, chosen on validation data
  only (`../results/evaluation/FINAL_INFERENCE_PROTOCOL.txt`).

## Files

| Path | Role |
|---|---|
| `download_hest.py` | fetch HEST-1k patches, expression and metadata for the split slides |
| `train.py` | training entry point |
| `models/flow_matching.py` | MoE latent flow model and CFG sampling |
| `models/genevae_v2.py` | GeneVAE-v2 gene autoencoder |
| `models/baseline_models.py` | baseline MLP (imported by `training_utils.py`) |
| `utils/custom_dataset.py` | HEST-1k dataset on precomputed patch embeddings |
| `utils/hest_utils.py` | gene lists, expression normalisation, HDF5 patch reading |
| `utils/training_utils.py` | model construction, losses, checkpointing, epoch loops |
| `utils/pfm_preprocesing.py` | patch-embedding extraction (CONCH, UNI, UNI-v2, ...) |
| `utils/other_utils.py` | seeding and nested-config helpers |
| `configs/training_config_conch_run2.yml` | the exact Run-2 configuration |
| `requirements.txt` | pinned dependencies (Python 3.11, PyTorch 2.5.1 with CUDA 12.1, `flow_matching` 1.0.10, CONCH) |

`train.py`, `models/flow_matching.py`, `models/genevae_v2.py` and
`utils/{custom_dataset,hest_utils,training_utils}.py` are exactly the code that
produced the results; their SHA-256 hashes are in
`../docs/audit/BUNDLE_SHA256SUMS.txt`. To keep those hashes valid, these files
are left as they ran, including some unused upstream helpers and cosmetic lint
(unused imports, `[DEBUG]` log lines), and are excluded from `ruff.toml`.

None of the following affects training, evaluation or interpretation:

- `normalize_adata` applies only `log1p`, although its docstring also mentions
  total-count normalisation.
- `HESTGraphDataset` has an outdated tuple interface and `save_hdf5` a typo in
  its append branch; neither is called.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

The Run-2 jobs ran on NVIDIA H100 GPUs. The CONCH weights are gated on
HuggingFace: request access, then set `HF_TOKEN`. HEST-1k downloads without a
token.

## Configuring paths

`configs/training_config_conch_run2.yml` is the exact Run-2 configuration, so
its paths point to the training cluster and the upstream MoLF layout. Before
running elsewhere, set the entries the code reads:

| Key | Set to |
|---|---|
| `paths_config.splits_path` | `../data/splits` |
| `paths_config.gene_list_path` | `../data/genes/Hallmark_n_HVG50_genes.yaml` |
| `paths_config.oncotree_vocab_path` | `../data/splits/run2_oncotree_code_list.txt` |
| `paths_config.patches_path` | `<hest dir>/patches` (from `download_hest.py`) |
| `paths_config.st_path` | `<hest dir>/st` |
| `paths_config.embeddings_path` | where the CONCH patch embeddings are written and read |
| `gene_vae_pretrained_path` | directory with the GeneVAE-v2 checkpoint |

`metadata_path`, `path_way_dict_path`, `root` and `wsis_path` are not used.

## Running

```bash
cd conch_molf

# 1. HEST-1k patches, expression and metadata for the 504 split slides
python download_hest.py --out-dir data/hest

# 2. CONCH patch embeddings (512-d, L2-normalised) for all slides in the splits
python utils/pfm_preprocesing.py --config configs/training_config_conch_run2.yml

# 3. Training (writes a log directory with checkpoints)
python train.py --config_path configs/training_config_conch_run2.yml

# 4. Resume an interrupted run
python train.py --log_dir <log_dir> --continue_training
```

HEST does not distribute CONCH embeddings, so step 2 is required. The Run-2
embeddings (504 slides, 1,076,882 patches) were checked by the forensic audit
for dimension, dtype, finiteness, L2 norm and patch/barcode order.

Validation-protocol search and the final test evaluation are in
[`../experiments/evaluation/`](../experiments/evaluation/).

## Data and weights

Included in the repository: the slide splits ([`../data/splits/`](../data/splits/))
and the 1,386 target genes ([`../data/genes/`](../data/genes/)).

Not included:

- HEST-1k patches (`<sample_id>.h5`) and expression data (`<sample_id>.h5ad`);
  `download_hest.py` fetches them.
- CONCH weights (`MahmoodLab/conch`, gated on HuggingFace; set `HF_TOKEN`).
- The GeneVAE-v2 checkpoint (`genevae_v2_best_epoch929.pt`) and the MoLF
  checkpoint; their SHA-256 hashes are in `../results/README.md`.

The GeneVAE-v2 training script was not preserved. The model definition, the
checkpoint hash and a reconstruction audit are
(`../experiments/audits/audit_genevae_reconstruction.py`, with results in
`../results/evaluation/GENEVAE_V2_RECON_*.csv`).
