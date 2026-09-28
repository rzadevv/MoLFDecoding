#!/bin/bash
#SBATCH --job-name=r2-val-C
#SBATCH --account=p_pathology03
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=1-12%3
#SBATCH --output=/data/cat/ws/rzmi154h-molf-pathology/conch_molf_production/run2_evaluation/logs/stageC_%A_%a.out
#SBATCH --error=/data/cat/ws/rzmi154h-molf-pathology/conch_molf_production/run2_evaluation/logs/stageC_%A_%a.err

module --force purge
module load release/24.04 GCC/12.3.0 Python/3.11.3
source ~/molf-train-env/bin/activate
export PYTHONNOUSERSITE=1

CATROOT=/data/cat/ws/rzmi154h-molf-pathology/conch_molf_production
RUN2="$CATROOT/run2_code"
EVAL="$CATROOT/run2_evaluation"
FREEZE="$CATROOT/run2_frozen"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$EVAL/stageC_protocols.tsv")

METHOD=$(echo "$LINE" | awk '{print $1}')
STEPS=$(echo "$LINE" | awk '{print $2}')
GUIDANCE=$(echo "$LINE" | awk '{print $3}')
SEED=$(echo "$LINE" | awk '{print $4}')

OUT="$EVAL/results/stageC_${METHOD}${STEPS}_cfg${GUIDANCE}_seed${SEED}"
rm -rf "$OUT"

python -u "$EVAL/run2_validation_inference.py" \
  --code-root "$RUN2" \
  --config "$FREEZE/config.yml" \
  --checkpoint "$FREEZE/run2_best_epoch239.pt" \
  --out-dir "$OUT" \
  --method "$METHOD" \
  --num-steps "$STEPS" \
  --guidance-scale "$GUIDANCE" \
  --noise-seed "$SEED"
