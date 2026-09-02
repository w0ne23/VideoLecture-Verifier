#!/usr/bin/env bash
# 기존 전처리 산출물을 재사용해 검증 단계만 재실행하는 CLI 래퍼
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH="$PWD" python -m pipeline.cli.main run-verify "$@"
