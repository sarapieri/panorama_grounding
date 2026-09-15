#!/usr/bin/env bash
# Evaluate a converted PANORAMA checkpoint on the GCG benchmark (GranD-f val / test).
#
#   MODEL_PATH=/path/to/run/hf bash scripts/eval_gcg.sh
#
# Needs PANORAMA_DATA_ROOT in .env (the dataset root holding glamm_data/). Prediction
# runs on GPUS gpus (default 4, one node) and covers both splits; the metric step runs on a
# single GPU for each split in SPLITS (default: val test).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; [ -f .env ] && . ./.env; set +a

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the converted hf/ directory}"
GPUS="${GPUS:-4}"
DATA_ROOT="${PANORAMA_DATA_ROOT:?set PANORAMA_DATA_ROOT in .env}"
SPLITS="${SPLITS:-val test}"
SAVE_DIR="${SAVE_DIR:-$(dirname "$MODEL_PATH")/evals/gcg}"

echo "=== PREDICT GCG ==="
echo "  MODEL_PATH : $MODEL_PATH"
echo "  DATA_ROOT  : $DATA_ROOT"
echo "  SAVE_DIR   : $SAVE_DIR"

torchrun --nnodes=1 --nproc_per_node="$GPUS" --master_port=$((20000 + RANDOM % 40000)) \
    -m eval.gcg_eval "$MODEL_PATH" \
    --save_dir "$SAVE_DIR" \
    --data_root "$DATA_ROOT" \
    --launcher pytorch

for SPLIT in $SPLITS; do
    echo "=== EVAL GCG metrics ($SPLIT) ==="
    CUDA_VISIBLE_DEVICES=0 python -m eval.metrics_gcg \
        --split "$SPLIT" \
        --prediction_dir_path "$SAVE_DIR" \
        --data_root "$DATA_ROOT"
done
