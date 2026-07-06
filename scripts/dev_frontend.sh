#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../frontend"
[ -d node_modules ] || npm install
npm run dev
