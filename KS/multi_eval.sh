#!/usr/bin/env bash
set -euo pipefail

echo "STARTING EVAL BATCH"

BASE_DIR="${1:-.}"
if [[ $# -gt 0 ]]; then
    shift
fi

find "${BASE_DIR}" -type f -name "model_params.npz" -print0 | sort -z | while IFS= read -r -d '' PARAM_FILE; do
    MODEL_DIR="$(dirname "${PARAM_FILE}")"
    CHECKPOINT="${MODEL_DIR}/model_epoch_best.pt"

    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "Skipping ${MODEL_DIR}: missing model_epoch_best.pt"
        continue
    fi

    echo "Submitting ${MODEL_DIR}"
    sbatch eval_job.sh "${MODEL_DIR}" "$@"
done

echo "DONE"
