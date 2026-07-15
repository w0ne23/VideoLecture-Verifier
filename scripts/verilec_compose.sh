#!/usr/bin/env bash
# Select the portable CPU stack or the NVIDIA stack without making callers
# remember Compose override files. Override automatic detection with
# VERILEC_MODE=cpu or VERILEC_MODE=gpu when needed.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mode="${VERILEC_MODE:-auto}"
case "$mode" in
  auto|cpu|gpu) ;;
  *)
    printf 'Invalid VERILEC_MODE=%s; use auto, cpu, or gpu.\n' "$mode" >&2
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
  printf 'VERILEC_MODE=gpu was requested, but NVIDIA Docker GPU support was not detected.\n' >&2
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
  printf 'VeriLec mode: GPU (CUDA decode, TensorRT, Nemotron OCR)\n'
else
  compose_args+=(--profile rapidocr)
  printf 'VeriLec mode: CPU (OpenCV decode, CPU YOLO, RapidOCR PP-OCRv5 Korean)\n'
fi

exec docker compose "${compose_args[@]}" "$@"
