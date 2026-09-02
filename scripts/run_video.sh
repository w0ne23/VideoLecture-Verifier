#!/usr/bin/env bash
# 영상 전체 파이프라인(전처리+검증)을 실행하는 CLI 래퍼
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH="$PWD" python -m pipeline.cli.main run-video "$@"
