#!/bin/bash
#SBATCH --job-name=r2-val-smoke
#SBATCH --account=p_pathology03
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/cat/ws/rzmi154h-molf-pathology/conch_molf_production/run2_evaluation/logs/smoke_%j.out
#SBATCH --error=/data/cat/ws/rzmi154h-molf-pathology/conch_molf_production/run2_evaluation/logs/smoke_%j.err

module --force purge
module load release/24.04 GCC/12.3.0 Python/3.11.3
source ~/molf-train-env/bin/activate
export PYTHONNOUSERSITE=1

CATROOT=/data/cat/ws/rzmi154h-molf-pathology/conch_molf_production
RUN2="$CATROOT/run2_code"
EVAL="$CATROOT/run2_evaluation"
FREEZE="$CATROOT/run2_frozen"

OUT="$EVAL/results/smoke_euler10_cfg1_seed41001"
rm -rf "$OUT"

python -u "$EVAL/run2_validation_inference.py" \
  --code-root "$RUN2" \
  --config "$FREEZE/config.yml" \
  --checkpoint "$FREEZE/run2_best_epoch239.pt" \
  --out-dir "$OUT" \
  --method euler \
  --num-steps 10 \
  --guidance-scale 1.0 \
  --noise-seed 41001 \
  --limit 2
