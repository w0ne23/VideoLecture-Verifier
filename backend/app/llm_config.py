"""Provider-neutral LLM endpoint configuration.

This module only normalizes and validates configuration data.  It deliberately
does not contain provider SDK calls; the pipeline adapters can consume the
same contract without making the API/UI know how a provider is implemented.
"""

from __future__ import annotations

from copy import deepcopy
from urllib.parse import urlparse


LLM_CONFIG_VERSION = 1

SUPPORTED_PROTOCOLS = {
    "openai_chat_completions",
    "openai_responses",
    "anthropic_messages",
    "gemini_generate_content",
    "ollama_native",
    "custom",
}

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


def _text(value, *, default: str = "", max_length: int = 512) -> str:
    value = str(value or "").strip()
    return value[:max_length] if value else default


def _number(value, *, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _safe_url(value: object) -> str:
    url = _text(value, max_length=2048).rstrip("/")
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return url


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


def _json_object(value: object) -> dict:
    return deepcopy(value) if isinstance(value, dict) else {}


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
        # API secrets must never be placed in this object.  The value is only
        # a reference to a server-side secret or environment variable.
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


def normalize_llm_config(raw: object) -> dict:
    """Return a bounded, JSON-serializable endpoint/stage configuration."""
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
