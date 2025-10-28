#!/bin/bash
# submit_all_evals.sh
# Usage: ./submit_all_evals.sh <BASE_DIR> <RE_NUMBER> [COSINE_EVAL_STEPS]
set -uo pipefail   # <- drop `-e` so a single sbatch error doesn't kill the loop
set -x             # debug: echo commands as they run

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 <BASE_DIR> <RE_NUMBER> [COSINE_EVAL_STEPS]"
  exit 1
fi

BASE_DIR="$1"
RE_NUMBER="$2"
COS_STEPS="${3:-}"

SBATCH_SCRIPT="gc_eval.sh"
PARAM_BASENAME="model_params.npz"
COMPLETION_MARKER="results.npz"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

echo "Scanning: $BASE_DIR"
echo "Reynolds number: $RE_NUMBER"
[ -n "$COS_STEPS" ] && echo "Cosine eval steps: $COS_STEPS"
echo "Submitting jobs via: $SBATCH_SCRIPT"
echo "--------------------------------------------------"

job_count=0
fail_count=0

# helper: make a safe slug for job name and log filename (no spaces/quotes/etc.)
slugify () {
  # keep letters, digits, dot, underscore, dash; replace others with underscore and trim
  echo "$1" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-100
}

while IFS= read -r -d '' PARAM_FILE; do
  MODEL_DIR="$(dirname "$PARAM_FILE")"
  MODEL_NAME="$(basename "$MODEL_DIR")"

  # Skip if results already exist
  if [ -f "$MODEL_DIR/$COMPLETION_MARKER" ]; then
    echo "✔ Skipping $MODEL_NAME (found $COMPLETION_MARKER)"
    continue
  fi

  SAFE_NAME="$(slugify "$MODEL_NAME")"
  JOB_NAME="eval_Re${RE_NUMBER}_${SAFE_NAME}"
  LOG_FILE="${LOG_DIR}/eval_Re${RE_NUMBER}_${SAFE_NAME}_%j.out"

  echo "→ Submitting: $MODEL_NAME"
  echo "   JobName: $JOB_NAME"
  echo "   Log:     $LOG_FILE"

  if [ -n "$COS_STEPS" ]; then
    if ! sbatch --job-name="$JOB_NAME" --output="$LOG_FILE" \
        "$SBATCH_SCRIPT" "$PARAM_FILE" "$RE_NUMBER" "$COS_STEPS"; then
      echo "✗ sbatch failed for $MODEL_NAME"
      ((fail_count++))
      continue
    fi
  else
    if ! sbatch --job-name="$JOB_NAME" --output="$LOG_FILE" \
        "$SBATCH_SCRIPT" "$PARAM_FILE" "$RE_NUMBER"; then
      echo "✗ sbatch failed for $MODEL_NAME"
      ((fail_count++))
      continue
    fi
  fi

  ((job_count++))
  echo "--------------------------------------------------"

done < <(find -L "$BASE_DIR" -type f -name "$PARAM_BASENAME" -print0)

set +x
echo "--------------------------------------------------"
echo "DONE. Submitted $job_count job(s). Failures: $fail_count"
echo "Logs will appear under: $LOG_DIR/"
