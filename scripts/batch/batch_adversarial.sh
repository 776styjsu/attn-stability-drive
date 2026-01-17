#!/bin/bash
# Batch adversarial attack runner with multiple seeds
# Usage: ./batch_adversarial.sh

set -e

# Configuration - from 1 to 50
if [ -z "$SEEDS" ]; then
    SEEDS=(1 2 3 4 5 6 7 8 9 10)
else
    # Allow passing seeds as a space-separated string env var
    IFS=' ' read -r -a SEEDS <<< "$SEEDS"
fi

# Model configuration
MODEL_CLASS="attn_stability_drive.models.dave2.DAVE2v1"
if [ -z "$CKPT" ]; then
    CKPT="/home/buc9hh/work/research/pcla/agents/dave2/checkpoints/dave2v1_10000_per_town_ep50_aug/dave2v1_10000_per_town_ep50_aug.pt"
fi

# Input configuration (can be single image or directory)
# If IMAGE points to a directory, use --image-dir in the python call, otherwise --image
IMAGE="/scratch/buc9hh/carla_runs/closed_loop_4/10000_aug/Town10HD/images/Town10HD__2383583.png"

# Attack configuration
DIFF_METHOD="gradient_shap"
MAX_ITER=50
EPSILON=0.1
STEP_SIZE=0.01

# Loss weights
LAMBDA_EXP=1000.0
LAMBDA_OUT=1.0
LAMBDA_PERT=1.0

# Explanation Metrics
EXPLANATION_METRIC="cosine"
IG_STEPS=50
BG_COUNT=16

# Background data config (for gradient_shap)
DATA_JSONL="/scratch/buc9hh/carla_runs/closed_loop_4/10000_aug/Town10HD/measurements.jsonl"
IMG_ROOT="/scratch/buc9hh/carla_runs/closed_loop_4/10000_aug/Town10HD/"

# Optional Mask
# Set to "" to disable
# MASK="ef9bc9e8f3e3db46c194d5685458c5f2.png"

# Output configuration
BASE_OUT_DIR="output/adversarial/"

# Other flags
PLOT=true  # Set to true or false

# ==============================================================================

# Create timestamped subdirectory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_OUT="${BASE_OUT_DIR}/batch_${TIMESTAMP}"
mkdir -p "$BATCH_OUT"

echo "============================================"
echo "Batch Adversarial Attack"
echo "============================================"
echo "Model:      $MODEL_CLASS"
echo "Checkpoint: $CKPT"
echo "Image/Dir:  $IMAGE"
echo "Seeds:      ${SEEDS[*]}"
echo "Output:     $BATCH_OUT"
echo "Method:     $DIFF_METHOD"
echo "============================================"
echo ""

# Validate optional inputs to avoid cryptic python errors
if [[ ! -f "$CKPT" ]]; then
    echo "Error: Checkpoint not found: $CKPT"
    exit 1
fi

if [[ ! -e "$IMAGE" ]]; then
    echo "Error: Image path not found: $IMAGE"
    exit 1
fi

# Determine if image flag is --image or --image-dir
if [[ -d "$IMAGE" ]]; then
    IMG_FLAG="--image-dir"
else
    IMG_FLAG="--image"
fi

# Construct MASK argument if set
MASK_ARG=""
if [[ -n "$MASK" ]]; then
    MASK_ARG="--mask $MASK"
fi

# Construct PLOT argument
PLOT_ARG=""
if [[ "$PLOT" == "true" ]]; then
    PLOT_ARG="--plot"
fi

# Track results
TOTAL_RUNS=0
SUCCESSFUL_RUNS=0

# Run attack for each seed
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "============================================"
    echo "Running with seed: $SEED"
    echo "============================================"
    
    OUT_DIR="${BATCH_OUT}/seed_${SEED}"
    
    # We pass python arguments directly. 
    # Note: quoted variables preserve spaces, strict checking enabled by set -e
    python scripts/adversarial_attack.py \
        --model-class "$MODEL_CLASS" \
        --ckpt "$CKPT" \
        $IMG_FLAG "$IMAGE" \
        --out "$OUT_DIR" \
        --seed "$SEED" \
        --diff-method "$DIFF_METHOD" \
        --data-jsonl "$DATA_JSONL" \
        --img-root "$IMG_ROOT" \
        --bg-count "$BG_COUNT" \
        --max-iter "$MAX_ITER" \
        --epsilon "$EPSILON" \
        --step-size "$STEP_SIZE" \
        --lambda-exp "$LAMBDA_EXP" \
        --lambda-out "$LAMBDA_OUT" \
        --lambda-pert "$LAMBDA_PERT" \
        --ig-steps "$IG_STEPS" \
        --explanation-metric "$EXPLANATION_METRIC" \
        $PLOT_ARG \
        $MASK_ARG
    
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    
    # Check if attack succeeded by looking at summary (folder structure depends on single vs multi image)
    # The script saves to OUT_DIR/<image_name_stem>/attack_result.json
    if compgen -G "${OUT_DIR}/*/attack_summary.json" > /dev/null; then
         SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
    fi
done

echo ""
echo "============================================"
echo "Batch Complete!"
echo "============================================"
echo "Total seeds run: $TOTAL_RUNS"
echo "Results saved to: $BATCH_OUT"
echo ""
echo "To aggregate results, run:"
echo "  python scripts/aggregate_attack_results.py --input-dir $BATCH_OUT"
