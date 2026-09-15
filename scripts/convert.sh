#!/usr/bin/env bash
# Convert a trained PANORAMA checkpoint (.pth) to a Hugging Face directory for eval.
#
#   CONFIG=src/configs/panorama_4b.py bash scripts/convert.sh
#
# Reads <run>/train/last_checkpoint by default; override with PTH=/path/to/iter_N.pth.
# The output HF_DIR is what scripts/eval_panocaps.sh takes as MODEL_PATH. Weights are saved in
# bfloat16 (what evaluation loads); DTYPE=float32 keeps full precision.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; [ -f .env ] && . ./.env; set +a

CONFIG="${CONFIG:?set CONFIG, e.g. CONFIG=src/configs/panorama_4b.py}"
CONFIG_NAME="$(basename "$CONFIG" .py)"
RESULTS="${PANORAMA_RESULTS_ROOT:?set PANORAMA_RESULTS_ROOT in .env}"
WORK_DIR="${WORK_DIR:-$RESULTS/$CONFIG_NAME/train}"
PTH="${PTH:-$(cat "$WORK_DIR/last_checkpoint" 2>/dev/null || true)}"
HF_DIR="${HF_DIR:-$RESULTS/$CONFIG_NAME/hf}"

if [ -z "$PTH" ] || [ ! -e "$PTH" ]; then
    echo "ERROR: checkpoint not found (PTH='$PTH'). Set PTH or check $WORK_DIR/last_checkpoint." >&2
    exit 1
fi

echo "=== CONVERT -> HF ==="
echo "  CONFIG : $CONFIG"
echo "  PTH    : $PTH"
echo "  HF_DIR : $HF_DIR"
echo "  DTYPE  : ${DTYPE:-bfloat16}"

python tools/convert_to_hf.py "$CONFIG" "$PTH" --save-path "$HF_DIR" --dtype "${DTYPE:-bfloat16}"
