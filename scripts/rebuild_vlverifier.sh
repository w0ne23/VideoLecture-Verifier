#!/usr/bin/env bash
# VLVerifier 전체 서비스를 재빌드하고 강제 재생성으로 재기동
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

./scripts/vlverifier_compose.sh build
./scripts/vlverifier_compose.sh up -d --force-recreate
