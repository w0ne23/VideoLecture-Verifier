#!/usr/bin/env bash
# 백엔드 컨테이너 안에서 review_slides 기준 LocalVLM 이후 단계를 재실행하는 래퍼
set -euo pipefail

cd "$(dirname "$0")/.."

# 인자 미지정 시 기본 run_dir/입력 영상 사용
RUN_DIR="${1:-/app/storage/test_qwen35_base_group_fix}"
INPUT_PATH="${2:-/app/storage/inputs/os1-1.mp4}"

docker compose exec backend \
  python /app/storage/rerun_local_vlm_from_review.py \
  --run-dir "$RUN_DIR" \
  --input "$INPUT_PATH"
