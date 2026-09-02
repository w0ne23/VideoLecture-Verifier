#!/usr/bin/env bash
# 백엔드 개발 서버를 uvicorn으로 기동하는 스크립트
set -euo pipefail
cd "$(dirname "$0")/.."
# VLVerifier 등 다른 서비스가 8000을 쓰고 있으면 PORT=8001 ./scripts/dev_backend.sh 로 실행
PORT="${PORT:-8000}"
UVICORN="uvicorn"
[ -x .venv/bin/uvicorn ] && UVICORN=".venv/bin/uvicorn"
PYTHONPATH="$PWD:$PWD/backend" "$UVICORN" app.main:app --app-dir backend --host 0.0.0.0 --port "$PORT" --reload
