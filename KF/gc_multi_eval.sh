#!/bin/bash
set -euo pipefail
mkdir -p logs

echo "STARTING JOB SUBMISSION SCRIPT"

# --- Configuration ---
# Base directory to search for model parameters
BASE_DIR="${1:-.}"
# The sbatch script to execute for each job
SBATCH_SCRIPT="gc_eval.sh"
# The filename that indicates an evaluation is complete
# Change this to match the actual output file of your eval.py script
COMPLETION_MARKER="eval_results.npz"

echo "Searching in: $BASE_DIR"
echo "--------------------------------------------------"

job_count=0
# Use find with -print0 and a while loop to safely handle all filenames
while IFS= read -r -d '' PARAM_FILE; do
    MODEL_DIR=$(dirname "$PARAM_FILE")
    
    # --- Check for Completion ---
    if [ -f "$MODEL_DIR/$COMPLETION_MARKER" ]; then
        echo "Skipping $MODEL_DIR (results file already exists)."
        continue
    fi

    # --- Extract Reynolds Number ---
    PARENT_DIR_NAME=$(basename "$MODEL_DIR")
    if [[ $PARENT_DIR_NAME =~ Re([0-9]+) ]]; then
        RE_NUMBER="${BASH_REMATCH[1]}"
    else
        echo "Could not extract Re number from folder: $PARENT_DIR_NAME. Skipping."
        continue
    fi

    echo "Found parameters for Re = $RE_NUMBER in $MODEL_DIR"
    echo "   Submitting job..."
    
    # --- Submit the Job ---
    sbatch "$SBATCH_SCRIPT" "$PARAM_FILE" "$RE_NUMBER"
    ((job_count++))
    echo "--------------------------------------------------"

done < <(find "$BASE_DIR" -type f -name "model_params.npz" -print0)

echo "DONE. Submitted a total of $job_count jobs."