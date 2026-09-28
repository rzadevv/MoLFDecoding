#!/bin/bash
#SBATCH --job-name=run2-prezip-audit
#SBATCH --account=p_pathology03
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/cat/ws/rzmi154h-molf-pathology/conch_molf_production/prezip_final_audit/%j.out
#SBATCH --error=/data/cat/ws/rzmi154h-molf-pathology/conch_molf_production/prezip_final_audit/%j.err

module --force purge
module load release/24.04 GCC/12.3.0 Python/3.11.3

source ~/molf-train-env/bin/activate
export PYTHONNOUSERSITE=1

python -u \
  /data/cat/ws/rzmi154h-molf-pathology/conch_molf_production/prezip_final_audit/audit_run2_prezip_final.py
