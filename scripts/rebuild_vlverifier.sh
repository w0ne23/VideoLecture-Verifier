#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

./scripts/vlverifier_compose.sh build
./scripts/vlverifier_compose.sh up -d --force-recreate
