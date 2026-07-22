"""Ollama adapter shared by verifier stages."""

from __future__ import annotations

import base64
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def is_ollama_model(model: str) -> bool:
    value = str(model or "").strip().lower()
    return value.startswith(("ollama:", "ollama/"))


def resolve_ollama_model(model: str) -> str:
    value = str(model or "").strip()
    if is_ollama_model(value):
        return value.split(value[6], 1)[1].strip()
    return value


def _base_url() -> str:
    return (
        os.getenv("VERIFIER_OLLAMA_BASE_URL")
        or os.getenv("GRAPHLEC_OLLAMA_BASE_URL")
        or "http://localhost:11434"
    ).rstrip("/")


def _timeout() -> float:
    try:
        return max(1.0, float(os.getenv("VERIFIER_OLLAMA_TIMEOUT_SEC", "600") or 600))
    except ValueError:
        return 600.0


def call_ollama(
    *,
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    image_bytes: bytes | None = None,
    image_bytes_list: list[bytes] | None = None,
    json_response: bool = True,
    stage: str = "default",
) -> tuple[str, dict[str, Any]]:
    resolved_model = resolve_ollama_model(model)
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_message: dict[str, Any] = {"role": "user", "content": prompt}
    images = [item for item in (image_bytes_list or []) if item]
    if not images and image_bytes:
        images = [image_bytes]
    if images:
        user_message["images"] = [base64.b64encode(item).decode("ascii") for item in images]
    messages.append(user_message)

    payload: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": os.getenv("VERIFIER_OLLAMA_KEEP_ALIVE", "10m"),
        "options": {
            "temperature": float(temperature),
            "num_predict": max(1, int(max_tokens)),
        },
    }
    if json_response:
        payload["format"] = "json"

    request = Request(
        f"{_base_url()}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=_timeout()) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    text = str((result.get("message") or {}).get("content") or "")
    input_tokens = int(result.get("prompt_eval_count", 0) or 0)
    output_tokens = int(result.get("eval_count", 0) or 0)
    usage = {
        "provider": "ollama",
        "model": resolved_model,
        "stage": stage,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": 0,
        "tool_input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "total_tokens": input_tokens + output_tokens,
        "total_duration_ns": int(result.get("total_duration", 0) or 0),
        "prompt_eval_duration_ns": int(result.get("prompt_eval_duration", 0) or 0),
        "eval_duration_ns": int(result.get("eval_duration", 0) or 0),
    }
    return text, usage
