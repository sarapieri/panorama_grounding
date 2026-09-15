#!/usr/bin/env bash
# Evaluate a converted PANORAMA checkpoint on referring segmentation:
# RefCOCO / RefCOCO+ (val, testA, testB) and RefCOCOg (val, test).
#
#   MODEL_PATH=/path/to/run/hf bash scripts/eval_refcoco.sh
#
# Needs PANORAMA_DATA_ROOT in .env (the dataset root holding ref_seg/). Metrics are
# printed by eval.refcoco_eval itself at the end of each split. Override DATASETS / SPLITS to
# run a subset, e.g. DATASETS=refcocog SPLITS=val.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; [ -f .env ] && . ./.env; set +a

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the converted hf/ directory}"
GPUS="${GPUS:-4}"
DATA_ROOT="${PANORAMA_DATA_ROOT:?set PANORAMA_DATA_ROOT in .env}"
DATASETS="${DATASETS:-refcoco refcoco_plus refcocog}"

echo "=== EVAL RefCOCO family ==="
echo "  MODEL_PATH : $MODEL_PATH"
echo "  DATA_ROOT  : $DATA_ROOT"
echo "  DATASETS   : $DATASETS"

for DS in $DATASETS; do
    if [ -n "${SPLITS:-}" ]; then
        DS_SPLITS="$SPLITS"
    else
        case "$DS" in
            refcocog) DS_SPLITS="val test" ;;
            *)        DS_SPLITS="val testA testB" ;;
        esac
    fi
    for SPLIT in $DS_SPLITS; do
        echo "=== $DS / $SPLIT ==="
        torchrun --nnodes=1 --nproc_per_node="$GPUS" --master_port=$((20000 + RANDOM % 40000)) \
            -m eval.refcoco_eval "$MODEL_PATH" \
            --dataset "$DS" \
            --split "$SPLIT" \
            --data_root "$DATA_ROOT" \
            --launcher pytorch
    done
done
