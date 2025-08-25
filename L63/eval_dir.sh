#!/bin/bash
# ==============================================================================
# Check if a parent directory is provided as the first argument
if [ -z "$1" ]; then
    echo "Usage: eval_all_models <parent_directory> [data_path]"
    return 1
fi

PARENT_DIR=$1

# Check if the parent directory actually exists
if [ ! -d "$PARENT_DIR" ]; then
    echo "Error: Directory '$PARENT_DIR' not found."
    return 1
fi

# Find all subdirectories that contain a model checkpoint file
find "$PARENT_DIR" -name "model_epoch_best.pt" | while read -r CKPT_PATH; do
    MODEL_DIR=$(dirname "$CKPT_PATH")
    
    echo "-----------------------------------------------------"
    echo "Evaluating model in: $MODEL_DIR"
    echo "Using data from: $DATA_PATH"
    echo "-----------------------------------------------------"
    
    # Run the evaluation script with the correct paths
    python eval.py --model-dir "$MODEL_DIR"
    
    echo "Evaluation finished for $MODEL_DIR"
    echo ""
done

echo "All evaluations complete."
