#!/usr/bin/env bash
# Evaluate a converted PANORAMA checkpoint on the PanoCaps benchmark.
#
#   MODEL_PATH=/path/to/run/hf bash scripts/eval_panocaps.sh
#
# MODEL_PATH is the HF export produced by tools/convert_to_hf.py. Prediction runs on
# GPUS gpus (default 4, one node); the metric step runs on a single GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; [ -f .env ] && . ./.env; set +a

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the converted hf/ directory}"
GPUS="${GPUS:-4}"
IMAGE_DIR="${IMAGE_DIR:-${PANORAMA_DATA_ROOT:-data}/PanoCaps/images/test_val}"
ANN_FOLDER="${ANN_FOLDER:-${PANORAMA_DATA_ROOT:-data}/PanoCaps/annotations}"
SAVE_DIR="${SAVE_DIR:-$(dirname "$MODEL_PATH")/evals/panocaps}"

echo "=== PREDICT PanoCaps ==="
echo "  MODEL_PATH : $MODEL_PATH"
echo "  SAVE_DIR   : $SAVE_DIR"

torchrun --nnodes=1 --nproc_per_node="$GPUS" --master_port=$((20000 + RANDOM % 40000)) \
    -m eval.panocaps_eval "$MODEL_PATH" \
    --image-dir "$IMAGE_DIR" \
    --save-dir "$SAVE_DIR" \
    --launcher pytorch

echo "=== EVAL PanoCaps (val + test reported separately) ==="
CUDA_VISIBLE_DEVICES=0 python -m eval.metrics_panocaps \
    --split val test \
    --prediction_dir_path "$SAVE_DIR" \
    --gt_dir_path "$ANN_FOLDER"
