#!/bin/bash

# --- Config ---
STAGES=7
EPOCHS=30
DATA="foe400"
TEST="cbsd68"
BASE_RESULT_DIR="ablation_results"

mkdir -p $BASE_RESULT_DIR

# Function to move results to a specific folder
organize_results() {
    local EXP_NAME=$1
    local DEST="$BASE_RESULT_DIR/$EXP_NAME"
    mkdir -p "$DEST"

    echo "Organizing results for $EXP_NAME into $DEST..."

    # 1. Move the specific checkpoints
    if [ -d "checkpoints/$EXP_NAME" ]; then
        mv "checkpoints/$EXP_NAME" "$DEST/checkpoints"
    fi

    # 2. Move the latest graph directory
    # Finds the newest directory in graphs/
    LATEST_GRAPH=$(ls -td graphs/*/ | head -1)
    if [ -n "$LATEST_GRAPH" ]; then
        mv "$LATEST_GRAPH" "$DEST/graphs"
    fi

    # 3. Move the outputs (images)
    # We move the whole outputs folder content and recreate it to keep it clean
    if [ -d "outputs" ]; then
        mv "outputs" "$DEST/outputs"
        mkdir -p outputs
    fi
}

# -----------------------------------------------------------------------------
# EXPERIMENT 1: Phi Battle (Soft-Thresholding)
# -----------------------------------------------------------------------------
EXP1="phi_soft_threshold"
echo ">>> Starting Experiment 1: $EXP1"
uv run main.py train \
    --phi "soft_threshold" \
    --stages $STAGES \
    --epochs $EPOCHS \
    --train-on $DATA \
    --test-on $TEST \
    --checkpoint-dir "checkpoints/$EXP1"

organize_results $EXP1

# -----------------------------------------------------------------------------
# EXPERIMENT 2: Fixed Physics (No Scalar Learning)
# -----------------------------------------------------------------------------
EXP2="fixed_physics"
echo ">>> Starting Experiment 2: $EXP2"
# Note: we use default phi (lorentzian) but set lr-scalars to 0
uv run main.py train \
    --lr-scalars 0.0 \
    --stages $STAGES \
    --epochs $EPOCHS \
    --train-on $DATA \
    --test-on $TEST \
    --checkpoint-dir "checkpoints/$EXP2"

organize_results $EXP2

echo "===================================================="
echo "All priority ablations complete!"
echo "Results located in: $BASE_RESULT_DIR/"
echo "===================================================="
