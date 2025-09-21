#!/bin/bash
echo "STARTING"

# Base directory to start search
BASE_DIR="${1:-.}"

# Find all params.npz files and process them
find "$BASE_DIR" -type f -name "model_params.npz" | while read -r PARAM_FILE; do
    echo "Processing $PARAM_FILE"

    PARENT_DIR=$(basename "$(dirname "$PARAM_FILE")")
    RE_NUMBER=$(echo "$PARENT_DIR" | grep -oE 'Re[0-9]+' | grep -oE '[0-9]+')

    if [[ -z "$RE_NUMBER" ]]; then
        echo "Could not extract Re number from folder name: $PARENT_DIR"
        continue
    fi

    echo "Extracted Re number: $RE_NUMBER"
    
    # Run your Python script and pass the path
    # python eval.py "$PARAM_FILE" "$RE_NUMBER" 
    sbatch eval.sh "$PARAM_FILE" "$RE_NUMBER" 

done
