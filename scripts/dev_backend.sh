#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH="$PWD:$PWD/backend" uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload
