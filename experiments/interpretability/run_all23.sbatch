#!/bin/bash
#SBATCH --job-name=run2-all23-interp
#SBATCH --account=p_pathology03
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-22%6
#SBATCH --output=/data/cat/ws/rzmi154h-molf-pathology/conch_molf_production/run2_interpretability/all23/logs/%A_%a.out
#SBATCH --error=/data/cat/ws/rzmi154h-molf-pathology/conch_molf_production/run2_interpretability/all23/logs/%A_%a.err

module --force purge
module load release/24.04 GCC/12.3.0 Python/3.11.3
source ~/molf-train-env/bin/activate
export PYTHONNOUSERSITE=1

CATROOT=/data/cat/ws/rzmi154h-molf-pathology/conch_molf_production
RUN2="$CATROOT/run2_code"
ALL23="$CATROOT/run2_interpretability/all23"

SLIDES=(
  INT13
  INT14
  INT15
  INT24
  MEND154
  MEND156
  MEND157
  MEND158
  MEND159
  MEND160
  MEND161
  MEND162
  NCBI643
  NCBI684
  TENX116
  TENX117
  TENX141
  TENX147
  TENX148
  TENX149
  TENX99
  ZEN48
  ZEN49
)

SID="${SLIDES[$SLURM_ARRAY_TASK_ID]}"

echo "========================================"
echo "ARRAY TASK: $SLURM_ARRAY_TASK_ID"
echo "SAMPLE: $SID"
echo "HOST: $(hostname)"
echo "========================================"

cd "$RUN2"

python -u \
  "$ALL23/run_all23_slide.py" \
  --sample-id "$SID"
