#!/usr/bin/env bash
# Train PANORAMA on a single node.
#
#   CONFIG=src/configs/panorama_4b.py [GPUS=8] bash scripts/train.sh
#
# The paper's runs use 16 GPUs with an effective batch of 128 (batch_size 4 x
# accumulative_counts 2 in the config). With fewer GPUs, raise accumulative_counts in the
# config to keep the effective batch.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; [ -f .env ] && . ./.env; set +a

CONFIG="${CONFIG:?set CONFIG, e.g. CONFIG=src/configs/panorama_4b.py}"
GPUS="${GPUS:-8}"
CONFIG_NAME="$(basename "$CONFIG" .py)"
WORK_DIR="${WORK_DIR:-${PANORAMA_RESULTS_ROOT:?set PANORAMA_RESULTS_ROOT in .env}/$CONFIG_NAME/train}"
DEEPSPEED="${DEEPSPEED:-deepspeed_zero2}"
RESUME="${RESUME:-}"

EXTRA=()
[ -n "$RESUME" ] && EXTRA+=(--resume "$RESUME")

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "=== TRAIN PANORAMA ==="
echo "  CONFIG   : $CONFIG"
echo "  GPUS     : $GPUS"
echo "  WORK_DIR : $WORK_DIR"

torchrun --nnodes=1 --nproc_per_node="$GPUS" --master_port=$((20000 + RANDOM % 40000)) \
    tools/train.py "$CONFIG" \
    --launcher pytorch \
    --deepspeed "$DEEPSPEED" \
    --work-dir "$WORK_DIR" \
    "${EXTRA[@]}"

echo "Done. Next: CONFIG=$CONFIG bash scripts/convert.sh, then scripts/eval_panocaps.sh"
