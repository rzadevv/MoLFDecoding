#!/bin/bash
#SBATCH --job-name=r2-final-test
#SBATCH --account=p_pathology03
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=1-5%2
#SBATCH --output=/data/cat/ws/rzmi154h-molf-pathology/conch_molf_production/run2_evaluation/logs/final_test_%A_%a.out
#SBATCH --error=/data/cat/ws/rzmi154h-molf-pathology/conch_molf_production/run2_evaluation/logs/final_test_%A_%a.err

set -euo pipefail

module --force purge
module load release/24.04 GCC/12.3.0 Python/3.11.3
source ~/molf-train-env/bin/activate
export PYTHONNOUSERSITE=1

CATROOT=/data/cat/ws/rzmi154h-molf-pathology/conch_molf_production
EVAL="$CATROOT/run2_evaluation"
FREEZE="$CATROOT/run2_frozen"
RUN2="$CATROOT/run2_code"

check_sha () {
    FILE="$1"
    EXPECTED="$2"
    ACTUAL=$(sha256sum "$FILE" | awk '{print $1}')

    if [[ "$ACTUAL" != "$EXPECTED" ]]; then
        echo "SHA MISMATCH: $FILE"
        echo "expected: $EXPECTED"
        echo "actual:   $ACTUAL"
        exit 50
    fi
}

echo "=== FINAL TEST PRE-FLIGHT ==="

check_sha \
  "$EVAL/run2_validation_inference.py" \
  "7805c568c6b826aff839c0686f21d55a2169a6dfa389f6b48c1886bb9b59e085"

check_sha \
  "$EVAL/run2_final_test_inference.py" \
  "6ea99e689b1088ea5b38a35937752558d8f938e95fab1a808bc92b0c88ab3af0"

check_sha \
  "$EVAL/FINAL_INFERENCE_PROTOCOL.txt" \
  "e3939b92785ed85c1c15d54fa37204a16a030a301badca1b4cc8189862118dfc"

check_sha \
  "$EVAL/FINAL_TEST_EVALUATION_PLAN.txt" \
  "3ed9adb0554d7a1bfbe4a0fe3fcec8fcdaa0e81c7e965437c1997870fb2378ea"

check_sha \
  "$FREEZE/run2_best_epoch239.pt" \
  "dbe23d4e2f670749ee746272337a21d3d14e49fc6dab34dd43bea05fb727965d"

check_sha \
  "$FREEZE/config.yml" \
  "ba6e43d95a6acaa9a4a4fe0b2ca3f0e4a5d5d0fa8a35c0f233a5d7f3417cfd0c"

echo "ALL FROZEN HASHES: PASS"

SEED=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$EVAL/final_test_seeds.tsv")

case "$SEED" in
    41001|42001|43001|44001|45001) ;;
    *)
        echo "INVALID TEST SEED: $SEED"
        exit 51
        ;;
esac

OUT="$EVAL/results/final_test_euler2_cfg1.5_seed${SEED}"

if [[ -e "$OUT" ]]; then
    echo "REFUSING TO OVERWRITE EXISTING FINAL TEST OUTPUT:"
    echo "$OUT"
    exit 52
fi

echo
echo "FINAL TEST SEED=$SEED"
echo "OUTPUT=$OUT"
echo

python -u "$EVAL/run2_final_test_inference.py" \
  --code-root "$RUN2" \
  --config "$FREEZE/config.yml" \
  --checkpoint "$FREEZE/run2_best_epoch239.pt" \
  --out-dir "$OUT" \
  --noise-seed "$SEED"
