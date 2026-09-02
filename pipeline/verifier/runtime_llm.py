"""웹 UI가 저장한 모델 설정을 읽어 런타임 LLM endpoint를 라우팅

웹 애플리케이션은 provider 중립적인 endpoint 문서를 ``VLVERIFIER_LLM_CONFIG_JSON``에
저장, 이 모듈은 하나의 stage/model 바인딩을 해석해 현재 UI가 지원하는 두 프로토콜
(OpenAI 호환 chat, Anthropic Messages)에 맞게 어댑팅
비밀값은 이 문서 자체에서 절대 읽지 않고, endpoint의 ``credential_ref``가 job 범위
credential map 또는(레거시 job의 경우) 서버 측 환경변수로 해석됨
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import urllib.error
import urllib.request
from typing import Any

try:
    from ..utils import anthropic_structured_output_request_kwargs, api_call_with_retry
except ImportError:
    from utils import anthropic_structured_output_request_kwargs, api_call_with_retry


# 파이프라인 stage 이름을 llm_config의 stage_bindings 키 후보 목록으로 매핑
_STAGE_ALIASES = {
    "extract": ("claim_extract", "claim"),
    "claim": ("claim_extract", "claim"),
    # claim_common은 1차 issue judge를 stage="judge"로 호출
    "judge": ("issue_detect", "detect", "judge"),
    "detect": ("issue_detect", "detect", "judge"),
    "classify": ("issue_classify", "classify"),
    "issue_classify": ("issue_classify", "classify"),
    "verify": ("verify", "judge"),
    "ground": ("grounding", "ground"),
    "grounding": ("grounding", "ground"),
    "slide_error": ("slide",),
    "slide_error_transcribe": ("slide",),
}

_DYNAMIC_LITELLM_LOCK = threading.Lock()
_DYNAMIC_LITELLM_MODELS: set[str] = set()


# LiteLLM 게이트웨이 사용 여부, LITELLM_ENABLED 환경변수로 제어
def _litellm_enabled() -> bool:
    return (os.getenv("LITELLM_ENABLED") or "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _gateway_model_name(model: str) -> str:
    """stage에 선택된 구체적인 모델명 반환

    provider 별칭이나 provider별 폴백 모델은 의도적으로 여기서 해석하지 않음,
    선택된 stage 바인딩에는 LiteLLM이 호출할 구체적인 모델 식별자가 이미 들어있어야 함
    """
    raw = str(model or "").strip()
    return raw


# LiteLLM 게이트웨이 경유용 runtime 딕셔너리 생성, 원본 endpoint를 LiteLLM 동적 모델로 등록
def _litellm_runtime(model: str, *, source_runtime: dict[str, Any] | None = None) -> dict[str, Any] | None:
    model_name = _gateway_model_name(model)
    if not model_name:
        return None
    source_endpoint = (source_runtime or {}).get("endpoint") or {}
    gateway_model = _ensure_litellm_model(model_name, source_endpoint)
    endpoint = {
        "id": "litellm",
        "display_name": "LiteLLM Gateway",
        "provider": "litellm",
        "protocol": "openai_chat_completions",
        "base_url": (os.getenv("LITELLM_BASE_URL") or "http://litellm:4000/v1").rstrip("/"),
        "credential_ref": "LITELLM_API_KEY",
        "headers": {},
        "timeout": source_endpoint.get("timeout") or {"read_sec": 180},
        "retry": source_endpoint.get("retry") or {"max_attempts": 3, "backoff_sec": 2},
        "capabilities": source_endpoint.get("capabilities") or {},
        # LiteLLM은 정규화된 OpenAI 호환 요청만 받음, 원본 endpoint의 provider별
        # 옵션이 여기로 새어 들어가면 안 됨
        "provider_options": {},
        "enabled": True,
    }
    binding = {
        "endpoint_ref": "litellm",
        "model": gateway_model,
        "weight": 100.0,
    }
    source_provider = _endpoint_provider(source_endpoint) if source_endpoint else ""
    return {
        "binding": binding,
        "endpoint": endpoint,
        "provider": "litellm",
        "protocol": "openai_chat_completions",
        "resolved_model": gateway_model,
        "endpoint_ref": "litellm",
        # 원래 선택 정보를 게이트웨이 바인딩과 함께 유지, 게이트웨이 엔드포인트
        # 자체는 OpenAI 호환이지만 web search는 상위 provider에 맞는 올바른
        # LiteLLM 요청 형태를 선택해야 함
        "source_provider": source_provider,
        "source_model": model_name,
        "source_protocol": str(source_endpoint.get("protocol") or "").strip().lower(),
    }


# 워커 프로세스 환경변수(VLVERIFIER_CREDENTIALS_JSON)에서 복호화된 credential map 로드
def _load_runtime_credentials() -> dict[str, str]:
    raw = str(os.getenv("VLVERIFIER_CREDENTIALS_JSON", "") or "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return {
        str(key): str(secret)
        for key, secret in value.items()
        if str(key).startswith("credential:") and isinstance(secret, str) and secret
    } if isinstance(value, dict) else {}


# LiteLLM base URL에서 /v1 접미사를 제거한 관리 API 루트 URL
def _litellm_api_root() -> str:
    base_url = (os.getenv("LITELLM_BASE_URL") or "http://litellm:4000/v1").rstrip("/")
    return re.sub(r"/v1$", "", base_url, flags=re.IGNORECASE)


def _provider_model_name(model: str, endpoint: dict[str, Any]) -> str:
    """카탈로그 모델 id를 LiteLLM의 provider 접두사 붙은 모델 id로 변환"""
    raw = str(model or "").strip()
    if not raw:
        return raw
    provider = _endpoint_provider(endpoint)
    first = raw.split("/", 1)[0].lower()
    known_prefixes = {
        "openai", "anthropic", "xai", "gemini", "vertex_ai", "deepseek",
        "openrouter", "ollama", "azure", "bedrock", "bedrock_converse",
        "bedrock_mantle", "groq", "huggingface", "nvidia_nim", "hosted_vllm",
    }
    if first in known_prefixes:
        return raw
    protocol = str(endpoint.get("protocol") or "").strip().lower()
    prefix = {
        "openai": "openai",
        "anthropic": "anthropic",
        "xai": "xai",
        "gemini": "gemini",
        "deepseek": "deepseek",
        "openrouter": "openrouter",
        "local": "openai",
    }.get(provider, provider)
    if provider in {"", "custom"}:
        if protocol in {"openai_chat_completions", "openai_responses"}:
            prefix = "openai"
        elif protocol == "anthropic_messages":
            prefix = "anthropic"
    return f"{prefix}/{raw}" if prefix else raw


def _ensure_litellm_model(model: str, source_endpoint: dict[str, Any]) -> str:
    """선택된 endpoint/model을 불투명한 LiteLLM deployment로 등록"""
    reference = str(source_endpoint.get("credential_ref") or "").strip()
    secret = (
        _load_runtime_credentials().get(reference, "")
        or (os.getenv(reference, "") if reference else "")
    )
    if not secret:
        raise RuntimeError(
            f"선택한 모델의 credential_ref를 확인할 수 없습니다: {reference or '미설정'}"
        )

    provider_model = _provider_model_name(model, source_endpoint)
    endpoint_identity = "\0".join((
        str(source_endpoint.get("id") or ""),
        str(source_endpoint.get("provider") or ""),
        str(source_endpoint.get("base_url") or ""),
    ))
    alias_digest = hashlib.sha256(
        f"{reference}\0{provider_model}\0{endpoint_identity}".encode("utf-8")
    ).hexdigest()[:24]
    alias = f"vlverifier-{alias_digest}"

    with _DYNAMIC_LITELLM_LOCK:
        if alias in _DYNAMIC_LITELLM_MODELS:
            return alias

        master_key = str(
            os.getenv("LITELLM_MASTER_KEY", "")
            or os.getenv("LITELLM_API_KEY", "")
            or ""
        ).strip()
        if not master_key:
            raise RuntimeError(
                "웹 API 키를 LiteLLM에 등록하려면 LITELLM_MASTER_KEY가 필요합니다."
            )

        params: dict[str, Any] = {
            "model": provider_model,
            "api_key": secret,
        }
        base_url = str(source_endpoint.get("base_url") or "").strip()
        if base_url:
            params["api_base"] = base_url
        body = json.dumps({
            "model_name": alias,
            "litellm_params": params,
            "model_info": {
                "id": alias,
                "managed_by": "verilec",
            },
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{_litellm_api_root()}/model/new",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status < 200 or response.status >= 300:
                    raise RuntimeError(
                        f"LiteLLM 동적 모델 등록 실패 (HTTP {response.status})"
                    )
        except urllib.error.HTTPError as exc:
            # 이전 워커/프로세스가 이미 같은 결정적(deterministic) alias를 등록했을 수 있음,
            # provider 설정 정보를 담을 수 있는 응답 본문은 노출하지 않음
            if exc.code not in {400, 409}:
                raise RuntimeError(
                    f"LiteLLM 동적 모델 등록 실패 (HTTP {exc.code})"
                ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError("LiteLLM 동적 모델 등록에 연결하지 못했습니다.") from exc

        _DYNAMIC_LITELLM_MODELS.add(alias)
    return alias


# 저장된 llm_config JSON 로드, 파싱 실패/미설정 시 빈 dict
def _load_config() -> dict[str, Any]:
    # VLVerifier는 백엔드 워커가 쓰는 런타임 전역 이름, 예전 VeriLec 표기는
    # 과거 job을 위한 읽기 전용 폴백으로 유지
    raw = str(
        os.getenv("VLVERIFIER_LLM_CONFIG_JSON", "")
        or os.getenv("VERILEC_LLM_CONFIG_JSON", "")
        or ""
    ).strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def configured_stage_models(stage: str) -> list[str]:
    """verifier stage에 선택된 구체적인 모델 ID 목록 반환"""
    config = _load_config()
    raw_bindings = config.get("stage_bindings")
    if not isinstance(raw_bindings, dict):
        return []
    for key in _STAGE_ALIASES.get(stage, (stage,)):
        values = raw_bindings.get(key)
        if not isinstance(values, list):
            continue
        models = [
            str(item.get("model") or "").strip()
            for item in values
            if isinstance(item, dict) and str(item.get("model") or "").strip()
        ]
        if models:
            return list(dict.fromkeys(models))
    return []


# 모델명/provider 문자열을 표준 provider family로 정규화
def _provider_family(value: str) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in {"gpt", "openai"} or lowered.startswith(("gpt", "o1", "o3")):
        return "openai"
    if lowered in {"claude", "anthropic"} or lowered.startswith(("claude", "sonnet", "haiku", "opus")):
        return "anthropic"
    if lowered in {"grok", "xai"} or lowered.startswith(("grok", "xai:")):
        return "xai"
    if lowered in {"gemini", "google"} or lowered.startswith(("gemini", "google:")):
        return "gemini"
    if lowered.startswith("deepseek"):
        return "deepseek"
    if lowered.startswith(("ollama", "qwen", "vllm", "local")):
        return "local"
    return lowered


# endpoint의 provider 필드를 표준 provider family로 정규화
def _endpoint_provider(endpoint: dict[str, Any]) -> str:
    return _provider_family(str(endpoint.get("provider") or ""))


# 대소문자 무시하고 모델명이 정확히 일치하는지 확인
def _model_matches(binding_model: str, requested_model: str) -> bool:
    left = str(binding_model or "").strip().lower()
    right = str(requested_model or "").strip().lower()
    if not left or not right:
        return False
    if left == right:
        return True
    return False


def resolve_runtime_binding(stage: str, model_spec: str = "") -> dict[str, Any] | None:
    """파이프라인 stage에 설정된 endpoint 바인딩 해석

    model_spec이 주어지면 정확히 일치하는 모델만 선택, provider 별칭이 조용히
    다른 설정된 모델을 고르지 않도록 방지
    """
    config = _load_config()
    endpoints = {
        str(item.get("id") or ""): item
        for item in config.get("endpoints", []) or []
        if isinstance(item, dict) and str(item.get("id") or "").strip()
        and item.get("enabled", True) is not False
    }
    if not endpoints:
        return _litellm_runtime(model_spec) if _litellm_enabled() else None

    raw_bindings = config.get("stage_bindings")
    if not isinstance(raw_bindings, dict):
        return _litellm_runtime(model_spec) if _litellm_enabled() else None
    bindings: list[dict[str, Any]] = []
    for key in _STAGE_ALIASES.get(stage, (stage,)):
        values = raw_bindings.get(key)
        if not isinstance(values, list):
            continue
        bindings = [
            item for item in values
            if isinstance(item, dict)
            and str(item.get("endpoint_ref") or "") in endpoints
            and str(item.get("model") or "").strip()
        ]
        if bindings:
            break
    if not bindings:
        return _litellm_runtime(model_spec) if _litellm_enabled() else None

    requested = str(model_spec or "").strip()
    exact = [item for item in bindings if _model_matches(item.get("model"), requested)]
    selected = exact
    if not selected and not requested and len(bindings) == 1:
        selected = bindings
    if len(selected) != 1:
        return None

    binding = selected[0]
    endpoint = endpoints[str(binding["endpoint_ref"])]
    runtime = {
        "binding": binding,
        "endpoint": endpoint,
        "provider": str(endpoint.get("provider") or "custom"),
        "protocol": str(endpoint.get("protocol") or "custom"),
        "resolved_model": str(binding.get("model") or "").strip(),
        "endpoint_ref": str(binding.get("endpoint_ref") or "").strip(),
    }
    # 활성화 시 LiteLLM이 모든 일반 verifier 호출의 단일 게이트웨이가 됨, 웹 UI에서
    # 선택한 provider와 무관, 선택된 바인딩이 여전히 모델 alias와 stage 소속을 결정
    return _litellm_runtime(str(binding.get("model") or model_spec), source_runtime=runtime) \
        if _litellm_enabled() else runtime


# endpoint의 credential_ref로 API 키 조회, 없으면 provider별 기본 환경변수로 폴백
def _credential(endpoint: dict[str, Any]) -> str:
    reference = str(endpoint.get("credential_ref") or "").strip()
    if reference:
        dynamic = _load_runtime_credentials().get(reference, "")
        if dynamic:
            return dynamic
        value = os.getenv(reference, "")
        if value:
            return value
    provider = _endpoint_provider(endpoint)
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY", "") or os.getenv("LITELLM_API_KEY", "")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY", "")
    if provider == "xai":
        return os.getenv("XAI_API_KEY", "")
    if provider == "deepseek":
        return os.getenv("DEEPSEEK_API_KEY", "")
    if provider == "local":
        return os.getenv("LOCAL_LLM_API_KEY", "") or os.getenv("OLLAMA_API_KEY", "") or "verilec-local"
    return ""


# endpoint 설정의 read timeout(초), 미설정/파싱 실패 시 기본 180초
def _timeout(endpoint: dict[str, Any]) -> float:
    timeout = endpoint.get("timeout") if isinstance(endpoint.get("timeout"), dict) else {}
    try:
        return max(1.0, float(timeout.get("read_sec", 180)))
    except (TypeError, ValueError):
        return 180.0


# 프롬프트/시스템 프롬프트/이미지를 OpenAI chat completions 메시지 형식으로 구성
def _messages(prompt: str, system_prompt: str | None, images: list[bytes]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if images:
        content: list[dict[str, Any]] = []
        for image in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image).decode()}"},
            })
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})
    return messages


# 응답 객체(dict 또는 속성 객체)에서 여러 후보 이름 중 첫 유효값을 int로 추출
def _usage_value(obj: Any, *names: str) -> int:
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


# OpenAI 호환 응답에서 토큰 사용량을 표준 usage dict로 변환
def _openai_usage(resp: Any, provider: str, model: str, stage: str) -> dict[str, Any]:
    usage = getattr(resp, "usage", None)
    details = getattr(usage, "completion_tokens_details", None) or getattr(usage, "output_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None) or getattr(usage, "input_tokens_details", None)
    input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
    output_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
    total_tokens = _usage_value(usage, "total_tokens") or input_tokens + output_tokens
    return {
        "provider": provider,
        "model": model,
        "stage": stage,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": _usage_value(details, "reasoning_tokens"),
        "tool_input_tokens": 0,
        "cached_input_tokens": _usage_value(prompt_details, "cached_tokens"),
        "cache_creation_input_tokens": 0,
        "total_tokens": total_tokens,
    }


# Anthropic 응답에서 토큰 사용량을 표준 usage dict로 변환
def _anthropic_usage(resp: Any, model: str, stage: str) -> dict[str, Any]:
    usage = getattr(resp, "usage", None)
    input_tokens = _usage_value(usage, "input_tokens")
    output_tokens = _usage_value(usage, "output_tokens")
    return {
        "provider": "anthropic",
        "model": model,
        "stage": stage,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": 0,
        "tool_input_tokens": 0,
        "cached_input_tokens": _usage_value(usage, "cache_read_input_tokens"),
        "cache_creation_input_tokens": _usage_value(usage, "cache_creation_input_tokens"),
        "total_tokens": input_tokens + output_tokens,
    }


# OpenAI 호환 응답에서 텍스트 콘텐츠 추출
def _response_text(resp: Any) -> str:
    choices = getattr(resp, "choices", []) or []
    if not choices:
        return ""
    content = getattr(getattr(choices[0], "message", None), "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text") or "") if isinstance(item, dict) else str(item) for item in content)
    return str(content)


# metadata 조회를 위해 SDK 응답 객체를 최선 노력으로 JSON 가능한 값으로 변환
def _jsonable(value: Any) -> Any:
    """metadata 조회를 위해 SDK 응답 객체를 최선 노력으로 변환"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump())
        except Exception:
            pass
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _jsonable(to_dict())
        except Exception:
            pass
    return str(value)


# 응답에 있으면 provider 중립적인 검색 횟수/출처 URL을 추출
def _web_search_metadata(resp: Any) -> dict[str, Any]:
    """응답에서 provider 중립적인 검색 횟수/출처 URL 추출(있는 경우)"""
    payload = _jsonable(resp)
    queries: list[str] = []
    sources: list[str] = []
    request_count = 0

    # 응답 구조를 재귀적으로 순회하며 web_search 호출 횟수와 URL/쿼리 문자열 수집
    def walk(value: Any) -> None:
        nonlocal request_count
        if isinstance(value, dict):
            kind = str(value.get("type") or "").lower()
            if "web_search" in kind or kind in {"search_call", "websearch_call"}:
                request_count += 1
            for key, item in value.items():
                lowered = str(key).lower()
                if lowered in {"url", "source_url", "sourceurl"}:
                    url = str(item or "").strip()
                    if url.startswith(("http://", "https://")) and url not in sources:
                        sources.append(url)
                elif lowered in {"query", "search_query", "searchquery"}:
                    query = str(item or "").strip()
                    if query and query not in queries:
                        queries.append(query)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    usage = getattr(resp, "usage", None)
    usage_payload = _jsonable(usage)
    if isinstance(usage_payload, dict):
        details = usage_payload.get("prompt_tokens_details") or usage_payload.get("input_tokens_details")
        if isinstance(details, dict):
            request_count = max(
                request_count,
                _usage_value(details, "web_search_requests", "web_search_queries"),
            )
    return {
        "web_search_requests": request_count,
        "web_search_queries": queries,
        "web_search_sources": sources,
    }


# 일부 모델은 temperature 파라미터를 지원하지 않음, 그 목록에 해당하는지 확인
def _openai_temperature_allowed(model: str) -> bool:
    lowered = str(model or "").strip().lower()
    return not lowered.startswith(("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"))


# OpenAI 호환 API로 LLM 호출, 서버가 특정 파라미터를 거부하면 그 파라미터만 제거하고 재시도
def _call_openai_compatible(
    *,
    runtime: dict[str, Any],
    prompt: str,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float | None,
    response_format: dict | None,
    images: list[bytes],
    model_spec: str,
    stage: str,
    web_search: bool = False,
    web_search_max_calls: int = 1,
    web_search_force: bool = False,
    web_search_context_size: str | None = None,
) -> tuple[str, dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다.") from exc

    endpoint = runtime["endpoint"]
    api_key = _credential(endpoint)
    provider = str(endpoint.get("provider") or "custom")
    if not api_key:
        reference = str(endpoint.get("credential_ref") or "")
        raise RuntimeError(f"{reference or provider} API 키가 설정되지 않았습니다.")
    # 네트워크/API 재시도 정책은 공유 파이프라인 wrapper(api_call_with_retry)가 담당
    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": _timeout(endpoint),
        "max_retries": 0,
    }
    base_url = str(endpoint.get("base_url") or "").strip()
    if base_url:
        client_kwargs["base_url"] = base_url
    headers = endpoint.get("headers") if isinstance(endpoint.get("headers"), dict) else {}
    if headers:
        client_kwargs["default_headers"] = headers
    client = OpenAI(**client_kwargs)
    model = runtime["resolved_model"]
    source_provider = str(
        runtime.get("source_provider")
        or endpoint.get("source_provider")
        or ""
    ).strip().lower()
    source_model = str(
        runtime.get("source_model")
        or endpoint.get("source_model")
        or model_spec
    ).strip()

    if web_search:
        context_size = str(web_search_context_size or "medium").strip().lower()
        if context_size not in {"low", "medium", "high"}:
            context_size = "medium"
        max_calls = max(1, int(web_search_max_calls or 1))

        # web search의 기능 경계는 LiteLLM, 애플리케이션이 provider 허용목록을
        # 직접 관리해서는 안 됨 — 새 provider와 provider 모델 변형은 게이트웨이에
        # 도달해 LiteLLM 자체의 adapter/capability registry가 수락·거부를 판단해야 함
        is_openai_search_model = source_provider == "openai" and any(
            marker in source_model.lower()
            for marker in ("search-preview", "search_api", "search-api")
        )
        if source_provider == "openai" and not is_openai_search_model:
            response_kwargs: dict[str, Any] = {
                "model": model,
                "input": prompt,
                "tools": [{
                    "type": "web_search_preview",
                    "search_context_size": context_size,
                }],
                "tool_choice": "required" if web_search_force else "auto",
                "max_tool_calls": max_calls,
                "max_output_tokens": max_tokens,
                "include": ["web_search_call.action.sources"],
            }
            response = api_call_with_retry(
                lambda: client.responses.create(**response_kwargs)
            )
            text = str(getattr(response, "output_text", "") or "")
            usage = _openai_usage(response, "litellm", model, stage)
        else:
            search_kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                # OpenAI SDK는 OpenAI 호환 프록시와 통신할 때 provider별 필드를
                # extra_body에 넣도록 요구함. 선택된 상위 provider로의 변환은
                # LiteLLM이 담당하므로, 이 요청은 여기서 필터링하지 않고 설정된
                # 모든 provider에 의도적으로 그대로 전송
                "extra_body": {
                    "web_search_options": {
                        "search_context_size": context_size,
                    }
                },
            }
            response = api_call_with_retry(
                lambda: client.chat.completions.create(**search_kwargs)
            )
            text = _response_text(response)
            usage = _openai_usage(response, "litellm", model, stage)

        usage.update(_web_search_metadata(response))
        usage["web_search_provider"] = source_provider
        usage["web_search_model"] = source_model
        usage["web_search_mode"] = "litellm_gateway"
        usage["web_search_requested"] = True
        return text, usage

    options = endpoint.get("provider_options") if isinstance(endpoint.get("provider_options"), dict) else {}
    raw_model = str(model_spec or "").strip()
    effort_match = re.search(r"-(low|medium|high|xhigh)$", raw_model.lower())
    reasoning_effort = effort_match.group(1) if effort_match else None
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": _messages(prompt, system_prompt, images),
    }
    if str(options.get("max_tokens_param") or "").strip().lower() == "max_tokens" or not model.lower().startswith(("gpt", "o1", "o3")):
        kwargs["max_tokens"] = max_tokens
    else:
        kwargs["max_completion_tokens"] = max_tokens
    if response_format is not None:
        if response_format.get("type") == "json_schema" and "json_schema" not in response_format:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": response_format.get("schema") or {"type": "object"},
                },
            }
        else:
            kwargs["response_format"] = response_format
    if temperature is not None and _openai_temperature_allowed(model) and not reasoning_effort:
        kwargs["temperature"] = temperature
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if isinstance(options.get("extra_body"), dict):
        kwargs["extra_body"] = options["extra_body"]

    # 선택된 호환 서버가 명시적으로 거부하는 파라미터만 제거하며 재시도,
    # 네트워크/API 재시도는 공유 헬퍼가 처리
    for _ in range(3):
        try:
            response = api_call_with_retry(lambda: client.chat.completions.create(**kwargs))
            return _response_text(response), _openai_usage(response, provider, model, stage)
        except Exception as exc:
            message = str(exc).lower()
            changed = False
            if "max_completion_tokens" in message and "max_completion_tokens" in kwargs:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                changed = True
            elif "response_format" in message and "response_format" in kwargs:
                kwargs.pop("response_format", None)
                changed = True
            elif "temperature" in message and "temperature" in kwargs:
                kwargs.pop("temperature", None)
                changed = True
            elif "reasoning_effort" in message and "reasoning_effort" in kwargs:
                kwargs.pop("reasoning_effort", None)
                changed = True
            elif "extra_body" in message and "extra_body" in kwargs:
                kwargs.pop("extra_body", None)
                changed = True
            if not changed:
                raise
    raise RuntimeError("OpenAI-compatible LLM 호출 실패")


# 축약된 별칭을 실제 Anthropic 모델 ID로 변환
def _anthropic_model_name(model: str) -> str:
    aliases = {
        "haiku-4.5": "claude-haiku-4-5-20251001",
        "claude-haiku-4.5": "claude-haiku-4-5-20251001",
        "claude-haiku-4-5": "claude-haiku-4-5-20251001",
        "sonnet-4.5": "claude-sonnet-4-5-20250929",
        "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
        "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
    }
    return aliases.get(str(model or "").strip(), str(model or "").strip())


# Anthropic Messages API로 LLM 호출, JSON schema 지정 시 강제 tool_use로 구조화된 응답 요청
def _call_anthropic_messages(
    *,
    runtime: dict[str, Any],
    prompt: str,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float | None,
    response_format: dict | None,
    images: list[bytes],
    stage: str,
) -> tuple[str, dict[str, Any]]:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic 패키지가 설치되어 있지 않습니다.") from exc

    endpoint = runtime["endpoint"]
    api_key = _credential(endpoint)
    if not api_key:
        reference = str(endpoint.get("credential_ref") or "ANTHROPIC_API_KEY")
        raise RuntimeError(f"{reference} API 키가 설정되지 않았습니다.")
    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": _timeout(endpoint),
        "max_retries": 0,
    }
    base_url = str(endpoint.get("base_url") or "").strip()
    if base_url:
        client_kwargs["base_url"] = base_url
    headers = endpoint.get("headers") if isinstance(endpoint.get("headers"), dict) else {}
    if headers:
        client_kwargs["default_headers"] = headers
    client = Anthropic(**client_kwargs)
    content: list[dict[str, Any]] = []
    for image in images:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(image).decode()},
        })
    content.append({"type": "text", "text": prompt})
    kwargs: dict[str, Any] = {
        "model": _anthropic_model_name(runtime["resolved_model"]),
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if temperature is not None and "sonnet-5" not in kwargs["model"].lower():
        kwargs["temperature"] = temperature
    if response_format is not None:
        schema = response_format.get("schema") if response_format.get("type") == "json_schema" else None
        if not isinstance(schema, dict):
            schema = {"type": "object", "additionalProperties": True}
        try:
            from anthropic import transform_schema

            schema = transform_schema(schema)
        except (ImportError, AttributeError, TypeError, ValueError):
            schema = dict(schema)
        kwargs.update(anthropic_structured_output_request_kwargs(schema, create_method=client.messages.create))

    response = api_call_with_retry(lambda: client.messages.create(**kwargs))
    blocks = list(getattr(response, "content", []) or [])
    tool_blocks = [block for block in blocks if getattr(block, "type", "") == "tool_use"]
    if tool_blocks and isinstance(getattr(tool_blocks[0], "input", None), dict):
        return json.dumps(tool_blocks[0].input, ensure_ascii=False), _anthropic_usage(response, kwargs["model"], stage)
    text = "".join(str(getattr(block, "text", "") or "") for block in blocks if getattr(block, "type", "") == "text")
    return text, _anthropic_usage(response, kwargs["model"], stage)


def call_runtime_llm(
    runtime: dict[str, Any],
    *,
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 8192,
    temperature: float | None = None,
    response_format: dict | None = None,
    image_bytes: bytes | None = None,
    image_bytes_list: list[bytes] | None = None,
    model_spec: str = "",
    stage: str = "default",
    web_search: bool = False,
    web_search_max_calls: int = 1,
    web_search_force: bool = False,
    web_search_context_size: str | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """설정된 runtime endpoint 호출, 지원하지 않는 프로토콜이면 None 반환"""
    protocol = str(runtime.get("protocol") or "").strip().lower()
    images = [item for item in (image_bytes_list or []) if item]
    if not images and image_bytes:
        images = [image_bytes]
    if protocol in {"openai_chat_completions", "custom"}:
        return _call_openai_compatible(
            runtime=runtime,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            images=images,
            model_spec=model_spec,
            stage=stage,
            web_search=web_search,
            web_search_max_calls=web_search_max_calls,
            web_search_force=web_search_force,
            web_search_context_size=web_search_context_size,
        )
    if protocol == "anthropic_messages":
        return _call_anthropic_messages(
            runtime=runtime,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            images=images,
            stage=stage,
        )
    # Gemini/native Ollama/Responses는 UI에서 endpoint별 SDK 계약이 선택되기 전까지
    # 기존 provider adapter를 그대로 사용
    return None
