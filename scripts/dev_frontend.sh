#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../frontend"
if [ ! -d node_modules ] || [ ! -f node_modules/.package-lock.json ] \
  || [ package.json -nt node_modules/.package-lock.json ] \
  || [ package-lock.json -nt node_modules/.package-lock.json ]; then
  npm install
fi
npm run dev
