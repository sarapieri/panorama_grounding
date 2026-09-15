#!/usr/bin/env bash
# Evaluate a converted PANORAMA checkpoint on GroundingSuite-Eval.
#
#   MODEL_PATH=/path/to/run/hf bash scripts/eval_gseval.sh
#
# Needs PANORAMA_GSEVAL_ROOT in .env (the folder holding GroundingSuite-Eval.jsonl and the
# unlabeled2017/ images). Prediction runs on GPUS gpus (default 4, one node); the metric step
# runs on a single GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; [ -f .env ] && . ./.env; set +a

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the converted hf/ directory}"
GPUS="${GPUS:-4}"
: "${PANORAMA_GSEVAL_ROOT:?set PANORAMA_GSEVAL_ROOT in .env}"
SAVE_DIR="${SAVE_DIR:-$(dirname "$MODEL_PATH")/evals/groundingsuite}"

echo "=== PREDICT GroundingSuite-Eval ==="
echo "  MODEL_PATH  : $MODEL_PATH"
echo "  GSEVAL_ROOT : $PANORAMA_GSEVAL_ROOT"
echo "  SAVE_DIR    : $SAVE_DIR"

torchrun --nnodes=1 --nproc_per_node="$GPUS" --master_port=$((20000 + RANDOM % 40000)) \
    -m eval.groundingsuite_eval "$MODEL_PATH" \
    --save-dir "$SAVE_DIR" \
    --launcher pytorch

echo "=== EVAL GroundingSuite-Eval ==="
CUDA_VISIBLE_DEVICES=0 python -m eval.metrics_groundingsuite \
    --pred-file "$SAVE_DIR/predictions.jsonl"
