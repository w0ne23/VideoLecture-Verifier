# Provider 중립적인 LLM endpoint 설정
# 이 모듈은 설정 데이터의 정규화와 검증만 담당, provider SDK 호출은 포함하지 않음
# API/UI가 provider 구현 방식을 몰라도 파이프라인 어댑터가 동일한 규격을 그대로 쓸 수 있도록 분리
from __future__ import annotations

from copy import deepcopy
from urllib.parse import urlparse


# llm_config 스키마 버전, 저장된 값 마이그레이션 판단에 사용
LLM_CONFIG_VERSION = 1

# endpoint가 지원할 수 있는 프로토콜 목록
SUPPORTED_PROTOCOLS = {
    "openai_chat_completions",
    "openai_responses",
    "anthropic_messages",
    "gemini_generate_content",
    "ollama_native",
    "custom",
}

# stage 바인딩에서 허용하는 파이프라인 stage 식별자 목록
STAGE_KEYS = {
    "claim",
    "claim_extract",
    "detect",
    "issue_detect",
    "classify",
    "issue_classify",
    "judge",
    "verify",
    "ground",
    "grounding",
    "slide",
}


# 문자열 정규화, 공백 제거 후 최대 길이 초과분 제거
def _text(value, *, default: str = "", max_length: int = 512) -> str:
    value = str(value or "").strip()
    return value[:max_length] if value else default


# 숫자 정규화, 변환 실패 시 기본값, 성공 시 min~max 범위로 제한
def _number(value, *, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


# http/https 스킴과 host를 갖춘 URL만 통과, 그 외는 빈 문자열
def _safe_url(value: object) -> str:
    url = _text(value, max_length=2048).rstrip("/")
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return url


# 문자열 dict 정규화, 항목 수와 값 길이를 제한
def _string_map(value: object, *, max_items: int = 32, max_length: int = 2048) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for key, item in list(value.items())[:max_items]:
        key = _text(key, max_length=128)
        if not key:
            continue
        result[key] = _text(item, max_length=max_length)
    return result


# dict 형태 값만 깊은 복사로 통과, 아니면 빈 dict
def _json_object(value: object) -> dict:
    return deepcopy(value) if isinstance(value, dict) else {}


# 허용된 capability 키만 골라 상태(status)를 검증해 통과
def _sanitize_capabilities(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    allowed = {"vision", "json_object", "json_schema", "tool_calling", "reasoning", "seed"}
    result: dict[str, object] = {}
    for key in allowed:
        item = value.get(key)
        if isinstance(item, bool):
            result[key] = item
        elif isinstance(item, dict):
            status = _text(item.get("status"), max_length=32).lower()
            if status in {"supported", "unsupported", "unknown"}:
                result[key] = {"status": status, "source": _text(item.get("source"), max_length=64)}
    return result


# endpoint 원본 데이터를 검증된 형태로 정규화, id가 비어있으면 index 기반으로 생성
def _sanitize_endpoint(raw: object, index: int) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    endpoint_id = _text(raw.get("id") or raw.get("endpoint_id"), max_length=128)
    if not endpoint_id:
        endpoint_id = f"endpoint-{index + 1}"

    protocol = _text(raw.get("protocol"), default="openai_chat_completions", max_length=64)
    if protocol not in SUPPORTED_PROTOCOLS:
        protocol = "custom"

    timeout = raw.get("timeout") if isinstance(raw.get("timeout"), dict) else {}
    retry = raw.get("retry") if isinstance(raw.get("retry"), dict) else {}
    return {
        "id": endpoint_id,
        "display_name": _text(raw.get("display_name"), max_length=160),
        "provider": _text(raw.get("provider"), default="custom", max_length=80),
        "protocol": protocol,
        "base_url": _safe_url(raw.get("base_url")),
        "credential_ref": _text(raw.get("credential_ref"), max_length=160),
        # API 비밀값은 이 객체에 절대 담지 않음, 서버 측 secret이나 환경 변수를 가리키는 참조만 저장
        "headers": _string_map(raw.get("headers"), max_length=512),
        "timeout": {
            "connect_sec": _number(timeout.get("connect_sec"), default=10.0, minimum=1.0, maximum=120.0),
            "read_sec": _number(timeout.get("read_sec"), default=180.0, minimum=1.0, maximum=1800.0),
        },
        "retry": {
            "max_attempts": int(_number(retry.get("max_attempts"), default=3, minimum=0, maximum=10)),
            "backoff_sec": _number(retry.get("backoff_sec"), default=2.0, minimum=0.0, maximum=120.0),
        },
        "capabilities": _sanitize_capabilities(raw.get("capabilities")),
        "provider_options": _json_object(raw.get("provider_options")),
        "enabled": raw.get("enabled", True) is not False,
    }


# stage-endpoint 바인딩 원본을 검증된 형태로 정규화, endpoint_ref/model이 없으면 None
def _sanitize_binding(raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None
    endpoint_ref = _text(raw.get("endpoint_ref") or raw.get("endpoint_id"), max_length=128)
    model = _text(raw.get("model") or raw.get("model_name"), max_length=256)
    if not endpoint_ref or not model:
        return None
    return {
        "endpoint_ref": endpoint_ref,
        "model": model,
        "weight": _number(raw.get("weight"), default=100.0, minimum=0.0, maximum=100.0),
    }


# endpoints/stage_bindings 전체를 검증된 형태로 정규화해 반환하는 최상위 진입점
def normalize_llm_config(raw: object) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    endpoints = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw.get("endpoints", []) if isinstance(raw.get("endpoints"), list) else []):
        endpoint = _sanitize_endpoint(item, index)
        if endpoint["id"] in seen_ids:
            endpoint["id"] = f"{endpoint['id']}-{index + 1}"
        seen_ids.add(endpoint["id"])
        endpoints.append(endpoint)

    endpoint_ids = {item["id"] for item in endpoints}
    stage_bindings: dict[str, list[dict]] = {}
    raw_bindings = raw.get("stage_bindings") if isinstance(raw.get("stage_bindings"), dict) else {}
    for stage, values in raw_bindings.items():
        stage = _text(stage, max_length=64)
        if stage not in STAGE_KEYS or not isinstance(values, list):
            continue
        bindings = []
        for item in values[:20]:
            binding = _sanitize_binding(item)
            if binding and binding["endpoint_ref"] in endpoint_ids:
                bindings.append(binding)
        if bindings:
            stage_bindings[stage] = bindings

    return {
        "version": LLM_CONFIG_VERSION,
        "endpoints": endpoints,
        "stage_bindings": stage_bindings,
    }
