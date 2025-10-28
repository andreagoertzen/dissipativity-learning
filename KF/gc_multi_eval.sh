#!/bin/bash
# submit_all_evals.sh
# Usage: ./submit_all_evals.sh <BASE_DIR> <RE_NUMBER> [COSINE_EVAL_STEPS]
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 <BASE_DIR> <RE_NUMBER> [COSINE_EVAL_STEPS]"
  exit 1
fi

BASE_DIR="$1"
RE_NUMBER="$2"
COS_STEPS="${3:-}"

SBATCH_SCRIPT="gc_eval.sh"         # single-job script (updated above)
PARAM_BASENAME="model_params.npz"  # what to look for
COMPLETION_MARKER="results.npz"    # what eval.py actually writes
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

echo "Scanning: $BASE_DIR"
echo "Reynolds number: $RE_NUMBER"
[ -n "$COS_STEPS" ] && echo "Cosine eval steps: $COS_STEPS"
echo "Submitting jobs via: $SBATCH_SCRIPT"
echo "--------------------------------------------------"

job_count=0

# find safely handles spaces via -print0
while IFS= read -r -d '' PARAM_FILE; do
  MODEL_DIR="$(dirname "$PARAM_FILE")"
  MODEL_NAME="$(basename "$MODEL_DIR")"

  # Skip if results already exist
  if [ -f "$MODEL_DIR/$COMPLETION_MARKER" ]; then
    echo "✔ Skipping $MODEL_NAME (found $COMPLETION_MARKER)"
    continue
  fi

  echo "→ Submitting: $MODEL_NAME"
  if [ -n "$COS_STEPS" ]; then
    sbatch \
      --job-name="eval_Re${RE_NUMBER}_${MODEL_NAME}" \
      --output="${LOG_DIR}/eval_Re${RE_NUMBER}_${MODEL_NAME}_%j.out" \
      "$SBATCH_SCRIPT" "$PARAM_FILE" "$RE_NUMBER" "$COS_STEPS"
  else
    sbatch \
      --job-name="eval_Re${RE_NUMBER}_${MODEL_NAME}" \
      --output="${LOG_DIR}/eval_Re${RE_NUMBER}_${MODEL_NAME}_%j.out" \
      "$SBATCH_SCRIPT" "$PARAM_FILE" "$RE_NUMBER"
  fi

  ((job_count++))
done < <(find "$BASE_DIR" -type f -name "$PARAM_BASENAME" -print0)

echo "--------------------------------------------------"
echo "DONE. Submitted $job_count job(s)."
echo "Logs will appear under: $LOG_DIR/"
