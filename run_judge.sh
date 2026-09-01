#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${JUDGE_MODEL_PATH:?Set JUDGE_MODEL_PATH to the independent judge model}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -u "$ROOT/judge.py" \
  --predictions "$ROOT/outputs/predictions/predictions.json" \
  --output "$ROOT/outputs/judged.json" \
  --judge-model "$JUDGE_MODEL_PATH" \
  "$@"
