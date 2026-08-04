"""로컬(Ollama) 모델의 VRAM 예산을 넘지 않는 선에서 병렬 배치를 묶고,
넘치면 다음 배치로 미뤄 순차 처리되게 하는 스케줄러.

클라우드 모델은 로컬 VRAM을 전혀 쓰지 않으므로 예산 계산에서 제외하고
항상 첫 배치에 포함시켜 동시 실행한다 — 이게 클라우드+로컬 혼합 구성이
코드 분기 없이 자연스럽게 되는 지점이다.
"""

from __future__ import annotations

import logging
import os

from .claim_common import _is_ollama_model, _resolve_ollama_model

log = logging.getLogger(__name__)

# 오늘 이 하드웨어(RTX 4090)에서 실측/추정한 값(GB). "실측"은 이 모델
# 하나만 로드한 상태에서 nvidia-smi 프로세스 목록으로 직접 확인한 값,
# 나머지는 `ollama ps`의 SIZE 컬럼 기준이라 신뢰도가 떨어진다. 표에 없는
# 모델은 예산 전체를 잡아먹는 걸로 간주해 항상 자기 배치를 독점하게
# 만든다 — 과소평가로 인한 사고(오늘 겪었던 것들)를 피하기 위한
# fail-safe 기본값이다. 새 모델을 실사용하게 되면 이 표에 실측치를
# 채워 넣을 것.
_KNOWN_LOCAL_MODEL_VRAM_GB: dict[str, float] = {
    "gemma3:12b": 14.5,  # 실측
    "gemma4:26b-a4b-it-q4_K_M": 21.9,  # 실측
    "qwen3:30b-a3b": 26.0,  # ollama ps SIZE
    "qwen3-vl:30b": 20.0,  # ollama ps SIZE
    "gemma4:31b": 25.0,  # ollama ps SIZE (실측 당시 이미 GPU에 못 들어가 CPU 오프로드 발생)
}

_DEFAULT_GB = float(os.getenv("VERILEC_OLLAMA_UNKNOWN_MODEL_VRAM_GB", "24"))


def _detect_gpu_total_vram_gb() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        return None


def _default_budget_gb() -> float:
    detected = _detect_gpu_total_vram_gb()
    if detected is None:
        # GPU를 못 찾으면 로컬 모델은 전부 순차 처리가 되도록 예산을 0으로 둔다.
        return 0.0
    reserve = float(os.getenv("VERILEC_OLLAMA_VRAM_RESERVE_GB", "1.5"))
    return max(0.0, detected - reserve)


OLLAMA_VRAM_BUDGET_GB = float(
    os.getenv("VERILEC_OLLAMA_VRAM_BUDGET_GB") or _default_budget_gb()
)


def _local_model_cost_gb(model_spec: str) -> float:
    resolved = _resolve_ollama_model(model_spec)
    return _KNOWN_LOCAL_MODEL_VRAM_GB.get(resolved, _DEFAULT_GB)


def plan_model_batches(models: list[str]) -> list[list[str]]:
    """models를 실행 배치로 나눈다.

    - 클라우드 모델은 전부 배치 0에 포함되어 항상 동시 실행된다(예산 계산 제외).
    - 로컬(ollama:) 모델은 큰 것부터 그리디로 예산 안에 채운다.
    - 혼자서도 예산을 넘는 모델은 막지 않고 경고만 남긴 뒤 자기 배치로 실행한다
      (느려지거나 실패할 수 있음 — 중단 여부는 추후 결정 사항).
    """
    cloud = [m for m in models if not _is_ollama_model(m)]
    local = sorted(
        (m for m in models if _is_ollama_model(m)),
        key=_local_model_cost_gb,
        reverse=True,
    )

    for model in local:
        cost = _local_model_cost_gb(model)
        if cost > OLLAMA_VRAM_BUDGET_GB:
            log.warning(
                "[model_capacity] %s 예상 VRAM(%.1fGB)이 감지된 예산(%.1fGB)을 "
                "초과합니다 — 느려지거나 실패할 수 있지만 일단 시도합니다.",
                model,
                cost,
                OLLAMA_VRAM_BUDGET_GB,
            )

    batches: list[list[str]] = [list(cloud)] if cloud else [[]]
    loads: list[float] = [0.0]

    for model in local:
        cost = _local_model_cost_gb(model)
        placed = False
        for i, load in enumerate(loads):
            if load + cost <= OLLAMA_VRAM_BUDGET_GB:
                batches[i].append(model)
                loads[i] = load + cost
                placed = True
                break
        if not placed:
            batches.append([model])
            loads.append(cost)

    return [batch for batch in batches if batch]
