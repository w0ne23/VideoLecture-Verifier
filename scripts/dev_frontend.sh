#!/usr/bin/env bash
# 프론트엔드 개발 서버 기동, 의존성 변경 시 npm install 자동 실행
set -euo pipefail
cd "$(dirname "$0")/../frontend"
# package.json/package-lock.json이 node_modules보다 최신이면 재설치
if [ ! -d node_modules ] || [ ! -f node_modules/.package-lock.json ] \
  || [ package.json -nt node_modules/.package-lock.json ] \
  || [ package-lock.json -nt node_modules/.package-lock.json ]; then
  npm install
fi
npm run dev
