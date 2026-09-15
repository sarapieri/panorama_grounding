#!/usr/bin/env bash
# Evaluate a converted PANORAMA checkpoint on gRefCOCO (GRES: val, testA, testB).
#
#   MODEL_PATH=/path/to/run/hf bash scripts/eval_gres.sh
#
# Needs PANORAMA_DATA_ROOT (and PANORAMA_GREFCOCO_ROOT if gRefCOCO is not under
# <PANORAMA_DATA_ROOT>/ref_seg/grefcoco) in .env. Metrics are printed by eval.refcoco_eval at
# the end of each split. Override SPLITS to run a subset, e.g. SPLITS=val.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; [ -f .env ] && . ./.env; set +a

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the converted hf/ directory}"
GPUS="${GPUS:-4}"
DATA_ROOT="${PANORAMA_DATA_ROOT:?set PANORAMA_DATA_ROOT in .env}"
SPLITS="${SPLITS:-val testA testB}"

echo "=== EVAL gRefCOCO (GRES) ==="
echo "  MODEL_PATH   : $MODEL_PATH"
echo "  DATA_ROOT    : $DATA_ROOT"
echo "  SPLITS       : $SPLITS"

for SPLIT in $SPLITS; do
    echo "=== grefcoco / $SPLIT ==="
    torchrun --nnodes=1 --nproc_per_node="$GPUS" --master_port=$((20000 + RANDOM % 40000)) \
        -m eval.refcoco_eval "$MODEL_PATH" \
        --dataset grefcoco \
        --split "$SPLIT" \
        --data_root "$DATA_ROOT" \
        --launcher pytorch
done
