#!/usr/bin/env bash
# 호출자가 Compose override 파일을 따로 기억할 필요 없이 CPU/NVIDIA 스택 중 하나를 자동 선택
# VLVERIFIER_MODE=cpu 또는 VLVERIFIER_MODE=gpu로 자동 감지를 재정의 가능
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mode="${VLVERIFIER_MODE:-auto}"
case "$mode" in
  auto|cpu|gpu) ;;
  *)
    printf 'Invalid VLVERIFIER_MODE=%s; use auto, cpu, or gpu.\n' "$mode" >&2
    exit 2
    ;;
esac

gpu_ready=0
machine_arch="$(uname -m)"
if [ "$machine_arch" = "x86_64" ] || [ "$machine_arch" = "amd64" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    docker_runtimes="$(docker info --format '{{json .Runtimes}}' 2>/dev/null || true)"
    case "$docker_runtimes" in
      *nvidia*) gpu_ready=1 ;;
    esac
  fi
fi

if [ "$mode" = "gpu" ] && [ "$gpu_ready" -ne 1 ]; then
  printf 'VLVERIFIER_MODE=gpu was requested, but NVIDIA Docker GPU support was not detected.\n' >&2
  exit 1
fi
if [ "$mode" = "auto" ]; then
  if [ "$gpu_ready" -eq 1 ]; then
    mode="gpu"
  else
    mode="cpu"
  fi
fi

compose_args=(-f docker-compose.yml)
if [ "$mode" = "gpu" ]; then
  compose_args+=(-f docker-compose.gpu.yml --profile ocr)
  printf 'VLVerifier mode: GPU (CUDA decode, TensorRT, Nemotron OCR)\n'
else
  printf 'VLVerifier mode: CPU (OpenCV decode, CPU YOLO, embedded RapidOCR PP-OCRv5 Korean)\n'
fi

exec docker compose "${compose_args[@]}" "$@"
