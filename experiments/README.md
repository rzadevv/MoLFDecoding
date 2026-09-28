# Run-2 experiment jobs

The scripts and SLURM jobs in this directory are the ones that produced the
Run-2 results, kept byte-identical (their SHA-256 hashes are recorded in
`../docs/audit/` and in the per-slide interpretability manifests). They
contain the absolute paths of the HPC workspace they ran in; to re-run them
elsewhere, change the path constants at the top of each script and the
`#SBATCH` headers.

## evaluation/

Inference protocol selection used the 96 validation slides only; the 23 test
slides were evaluated once, afterwards, with the frozen protocol.

| Job | Purpose | Outputs in `results/evaluation/` |
|---|---|---|
| `run2_val_smoke.sbatch` | 2-slide smoke test | `runs/smoke_*` |
| `run2_val_stageA.sbatch` | solver / step count (Euler 10/25/50, RK4 10/25) at CFG 1.0 | `runs/stageA_*`, `STAGE_A_SOLVER_DECISION.txt` |
| `run2_val_stageA2.sbatch` | fewer Euler steps (2, 5) | `runs/stageA2_*` |
| `run2_val_stageB.sbatch` | guidance scale 1.25 / 1.5 / 2.0 | `runs/stageB_*` |
| `run2_val_stageC.sbatch` | CFG 1.0 / 1.5 / 2.0 over four more noise seeds | `runs/stageC_*`, `FINAL_INFERENCE_PROTOCOL.txt` |
| `run2_final_test.sbatch` | frozen protocol on the test set, 5 seeds | `runs/final_test_*`, `FINAL_TEST_*` |

`run2_validation_inference.py` and `run2_final_test_inference.py` are the
inference programs these jobs call. On the cluster they wrote each run to
`<evaluation dir>/results/<run>/`; here those folders are in
`results/evaluation/runs/`. Each run writes `summary.json`,
`per_slide_metrics.csv` and `per_gene_metrics.csv`. Metrics are computed on
measured genes only (masked), after `log1p`.

## interpretability/

Natural-routing interpretation of the 23 test slides.

- `run_all23_slide.py` — for one slide: runs the frozen model with the frozen
  protocol over 5 seeds, records router probabilities and top-2 route weights
  at both Euler states, selects each expert's 256 highest-affinity patches, and
  scores their CONCH profile against the 2,450 visual concept embeddings.
- `run_all23.sbatch` — array job over the 23 slides.
- `aggregate_all23.py` — cross-slide routing, semantic, stability and
  platform tables (`results/interpretability/aggregate/`).

## audits/

Independent checks run before the results were frozen.

| Script | Checks |
|---|---|
| `audit_run2_prezip_final.py` | provenance hashes, split/vocabulary/gene integrity, embeddings for all 504 slides, checkpoint selection, final metrics; report in `docs/audit/` |
| `audit_genevae_reconstruction.py` | GeneVAE-v2 reconstruction quality per slide and platform |
| `audit_target_distributions.py` | target-expression distributions per platform and split |
| `audit_all23_outputs.py`, `audit_final_all23_interpretability.py` | completeness and consistency of the interpretability outputs |
