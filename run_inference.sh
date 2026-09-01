#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the verbalizer checkpoint directory}"
: "${SAE_PATH:?Set SAE_PATH to the 262k SAE params.safetensors}"
DATA_DIR="${DATA_DIR:-$ROOT/data/evaluation}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -u "$ROOT/inference.py" \
  --checkpoint "$CHECKPOINT_PATH" \
  --sae "$SAE_PATH" \
  --data-dir "$DATA_DIR" \
  --output-dir "$ROOT/outputs/predictions" \
  "$@"
