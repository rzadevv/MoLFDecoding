#!/bin/bash
#SBATCH --job-name=gv2-recon-audit
#SBATCH --account=p_pathology03
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/data/cat/ws/rzmi154h-molf-pathology/conch_molf_production/run2_evaluation/logs/gv2_recon_%j.out
#SBATCH --error=/data/cat/ws/rzmi154h-molf-pathology/conch_molf_production/run2_evaluation/logs/gv2_recon_%j.err

module --force purge
module load release/24.04 GCC/12.3.0 Python/3.11.3
source ~/molf-train-env/bin/activate
export PYTHONNOUSERSITE=1

CATROOT=/data/cat/ws/rzmi154h-molf-pathology/conch_molf_production
RUN2="$CATROOT/run2_code"
EVAL="$CATROOT/run2_evaluation"

cd "$RUN2"

python -u "$EVAL/audit_genevae_reconstruction.py"
