#!/bin/bash

# --- Config ---
STAGES=5
EPOCHS=20
DATA="foe400"
TEST="cbsd68"
LOG_FILE="ablation_runs.log"

# Clear log file
echo "=== Ablation Run Started: $(date) ===" | tee $LOG_FILE

# Function to run experiment and log it
run_exp() {
    local DESC=$1
    shift
    echo "" | tee -a $LOG_FILE
    echo ">>> EXPERIMENT: $DESC" | tee -a $LOG_FILE
    echo ">>> COMMAND: uv run main.py train $@" | tee -a $LOG_FILE
    
    # Run command, capture stdout/stderr, and show in terminal
    uv run main.py train "$@" 2>&1 | tee -a $LOG_FILE
}

# --- GROUP A: ARCHITECTURAL ---
run_exp "Gaussian Derivative Phi" --phi gaussian_deriv --stages $STAGES --epochs $EPOCHS

# --- GROUP B: PARAMETERS ---
run_exp "Filter Bank K=8"  --filters 8  --stages $STAGES --epochs $EPOCHS
run_exp "Filter Bank K=48" --filters 48 --stages $STAGES --epochs $EPOCHS

run_exp "Filter Size 3x3" --filter-size 3 --stages $STAGES --epochs $EPOCHS
run_exp "Filter Size 9x9" --filter-size 9 --stages $STAGES --epochs $EPOCHS

run_exp "Damping Gamma=0.1" --gamma 0.1 --stages $STAGES --epochs $EPOCHS
run_exp "Damping Gamma=0.9" --gamma 0.9 --stages $STAGES --epochs $EPOCHS

# --- GROUP C: TRAINING ---
run_exp "Depth Stages=3"  --stages 3  --epochs $EPOCHS
run_exp "Depth Stages=15" --stages 15 --epochs $EPOCHS

run_exp "Loss L1" --loss l1 --stages $STAGES --epochs $EPOCHS

echo "" | tee -a $LOG_FILE
echo "=== All experiments finished: $(date) ===" | tee -a $LOG_FILE
