#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH="$PWD" python -m pipeline.cli.main run-video "$@"
