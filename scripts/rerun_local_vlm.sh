#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_DIR="${1:-/app/storage/test_qwen35_base_group_fix}"
INPUT_PATH="${2:-/app/storage/inputs/os1-1.mp4}"

docker compose exec backend \
  python /app/storage/rerun_local_vlm_from_review.py \
  --run-dir "$RUN_DIR" \
  --input "$INPUT_PATH"
