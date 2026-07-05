"""
Pipeline API usage and estimated cost reporting.

This module intentionally records estimates, not billing-authoritative values.
Provider invoices can differ because of free tier, cached tokens, regional pricing,
batch mode, rounding, or model-specific image tokenization.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any, Optional


USD_TO_KRW = float(os.getenv("GRAPHLEC_USD_TO_KRW", "1400"))

TOKEN_PRICES_USD_PER_1M: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "cached_input": 0.03},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40, "cached_input": 0.01},
    "gemini-embedding-001": {"input": 0.15, "output": 0.0, "cached_input": 0.0},
    "gpt-5.4": {"input": 2.50, "output": 15.00, "cached_input": 0.25},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50, "cached_input": 0.075},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25, "cached_input": 0.02},
}

AUDIO_PRICES_USD_PER_HOUR: dict[str, float] = {
    "whisper-large-v3-turbo": 0.04,
    "whisper-large-v3": 0.111,
}

_lock = Lock()
_events: list[dict[str, Any]] = []
_configured: dict[str, Any] = {}


def reset() -> None:
    with _lock:
        _events.clear()


def configure(*, stem: str, output_dir: Path | str) -> None:
    with _lock:
        _configured.clear()
        _configured.update(
            {
                "stem": stem,
                "output_dir": str(output_dir),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _usage_from_gemini_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
    if usage is None:
        return {}
    return {
        "input_tokens": _safe_int(
            getattr(usage, "prompt_token_count", None) or getattr(usage, "promptTokenCount", None)
        ),
        "output_tokens": _safe_int(
            getattr(usage, "candidates_token_count", None) or getattr(usage, "candidatesTokenCount", None)
        ),
        "reasoning_tokens": _safe_int(
            getattr(usage, "thoughts_token_count", None) or getattr(usage, "thoughtsTokenCount", None)
        ),
        "cached_input_tokens": _safe_int(
            getattr(usage, "cached_content_token_count", None)
            or getattr(usage, "cachedContentTokenCount", None)
        ),
        "tool_input_tokens": _safe_int(
            getattr(usage, "tool_use_prompt_token_count", None)
            or getattr(usage, "toolUsePromptTokenCount", None)
        ),
        "total_tokens": _safe_int(
            getattr(usage, "total_token_count", None) or getattr(usage, "totalTokenCount", None)
        ),
    }


def _usage_from_openai_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {
        "input_tokens": _safe_int(getattr(usage, "prompt_tokens", None)),
        "output_tokens": _safe_int(getattr(usage, "completion_tokens", None)),
        "reasoning_tokens": _safe_int(getattr(completion_details, "reasoning_tokens", None)),
        "cached_input_tokens": _safe_int(getattr(prompt_details, "cached_tokens", None)),
        "tool_input_tokens": 0,
        "total_tokens": _safe_int(getattr(usage, "total_tokens", None)),
    }


def _estimate_token_cost_usd(model: str, usage: dict[str, int]) -> float:
    price = TOKEN_PRICES_USD_PER_1M.get(model)
    if not price:
        return 0.0
    cached = min(_safe_int(usage.get("cached_input_tokens")), _safe_int(usage.get("input_tokens")))
    uncached_input = max(_safe_int(usage.get("input_tokens")) - cached, 0)
    output_billable = _safe_int(usage.get("output_tokens")) + _safe_int(usage.get("reasoning_tokens"))
    return (
        uncached_input * price.get("input", 0.0)
        + cached * price.get("cached_input", price.get("input", 0.0))
        + output_billable * price.get("output", 0.0)
    ) / 1_000_000


def record_model_call(
    *,
    stage: str,
    provider: str,
    model: str,
    response: Any = None,
    usage: Optional[dict[str, int]] = None,
    image_count: int = 0,
    item_count: int = 0,
    prompt_chars: int = 0,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    if usage is None:
        if provider.lower() == "openai":
            usage = _usage_from_openai_response(response)
        else:
            usage = _usage_from_gemini_response(response)
    usage = usage or {}
    usage_estimated = False
    if not _safe_int(usage.get("input_tokens")) and prompt_chars:
        # Some APIs, notably embeddings in a few SDK versions, may not expose
        # usage metadata. Keep the report useful with a rough chars/4 estimate.
        usage = dict(usage)
        usage["input_tokens"] = max(int(prompt_chars / 4), 1)
        usage["total_tokens"] = max(_safe_int(usage.get("total_tokens")), usage["input_tokens"])
        usage_estimated = True
    cost_usd = _estimate_token_cost_usd(model, usage)
    event = {
        "kind": "model",
        "stage": stage,
        "provider": provider,
        "model": model,
        "input_tokens": _safe_int(usage.get("input_tokens")),
        "output_tokens": _safe_int(usage.get("output_tokens")),
        "reasoning_tokens": _safe_int(usage.get("reasoning_tokens")),
        "cached_input_tokens": _safe_int(usage.get("cached_input_tokens")),
        "tool_input_tokens": _safe_int(usage.get("tool_input_tokens")),
        "total_tokens": _safe_int(usage.get("total_tokens")),
        "image_count": int(image_count or 0),
        "item_count": int(item_count or 0),
        "prompt_chars": int(prompt_chars or 0),
        "estimated_cost_usd": cost_usd,
        "estimated_cost_krw": cost_usd * USD_TO_KRW,
    }
    if usage_estimated:
        event["usage_estimated_from"] = "prompt_chars/4"
    if metadata:
        event["metadata"] = metadata
    with _lock:
        _events.append(event)


def record_audio_call(
    *,
    stage: str,
    provider: str,
    model: str,
    audio_seconds: float,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    price_per_hour = AUDIO_PRICES_USD_PER_HOUR.get(model, 0.0)
    cost_usd = max(float(audio_seconds or 0.0), 0.0) / 3600.0 * price_per_hour
    event = {
        "kind": "audio",
        "stage": stage,
        "provider": provider,
        "model": model,
        "audio_seconds": float(audio_seconds or 0.0),
        "estimated_cost_usd": cost_usd,
        "estimated_cost_krw": cost_usd * USD_TO_KRW,
    }
    if metadata:
        event["metadata"] = metadata
    with _lock:
        _events.append(event)


def _add_event_to_bucket(bucket: dict[str, Any], event: dict[str, Any]) -> None:
    bucket["calls"] += 1
    bucket["estimated_cost_usd"] += float(event.get("estimated_cost_usd", 0.0) or 0.0)
    bucket["estimated_cost_krw"] += float(event.get("estimated_cost_krw", 0.0) or 0.0)
    for key in (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_input_tokens",
        "tool_input_tokens",
        "total_tokens",
        "image_count",
        "item_count",
        "prompt_chars",
    ):
        bucket[key] += int(event.get(key, 0) or 0)
    bucket["audio_seconds"] += float(event.get("audio_seconds", 0.0) or 0.0)


def _new_bucket() -> dict[str, Any]:
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cached_input_tokens": 0,
        "tool_input_tokens": 0,
        "total_tokens": 0,
        "image_count": 0,
        "item_count": 0,
        "prompt_chars": 0,
        "audio_seconds": 0.0,
        "estimated_cost_usd": 0.0,
        "estimated_cost_krw": 0.0,
    }


def _estimate_external_token_usage(token_usage: dict[str, Any], model_hint: str) -> dict[str, Any]:
    total = token_usage.get("total", {}) if isinstance(token_usage, dict) else {}
    usage = {
        "input_tokens": _safe_int(total.get("input_tokens")),
        "output_tokens": _safe_int(total.get("output_tokens")),
        "reasoning_tokens": _safe_int(total.get("reasoning_tokens")),
        "cached_input_tokens": _safe_int(total.get("cached_input_tokens")),
        "tool_input_tokens": _safe_int(total.get("tool_input_tokens")),
        "total_tokens": _safe_int(total.get("total_tokens")),
    }
    cost_usd = _estimate_token_cost_usd(model_hint, usage)
    return {
        **usage,
        "estimated_cost_usd": cost_usd,
        "estimated_cost_krw": cost_usd * USD_TO_KRW,
    }


def build_report(
    *,
    stem: str,
    output_dir: Path | str,
    timings: Optional[dict[str, float]] = None,
    analyzer_output_path: Optional[Path | str] = None,
) -> dict[str, Any]:
    with _lock:
        events = list(_events)

    by_stage: dict[str, dict[str, Any]] = defaultdict(_new_bucket)
    by_model: dict[str, dict[str, Any]] = defaultdict(_new_bucket)
    total = _new_bucket()
    for event in events:
        _add_event_to_bucket(by_stage[event.get("stage", "unknown")], event)
        model_key = f"{event.get('provider', 'unknown')}:{event.get('model', 'unknown')}"
        _add_event_to_bucket(by_model[model_key], event)
        _add_event_to_bucket(total, event)

    external: dict[str, Any] = {}
    if analyzer_output_path:
        p = Path(analyzer_output_path)
        if p.exists():
            try:
                analyzer = json.loads(p.read_text(encoding="utf-8"))
                token_usage = analyzer.get("token_usage", {})
                token_usage_per_model = analyzer.get("token_usage_per_model", {})
                external["analyzer"] = {
                    "path": str(p),
                    "token_usage": token_usage,
                    "token_usage_per_model": token_usage_per_model,
                    "estimated_by_model": {},
                }
                for model, usage in token_usage_per_model.items():
                    external["analyzer"]["estimated_by_model"][model] = _estimate_external_token_usage(
                        usage, model
                    )
                for estimate in external["analyzer"]["estimated_by_model"].values():
                    total["estimated_cost_usd"] += float(estimate.get("estimated_cost_usd", 0.0) or 0.0)
                    total["estimated_cost_krw"] += float(estimate.get("estimated_cost_krw", 0.0) or 0.0)
            except Exception as exc:
                external["analyzer"] = {"path": str(p), "error": str(exc)}
        else:
            external["analyzer"] = {"path": str(p), "status": "not_available"}

    return {
        "schema_version": 1,
        "description": "Estimated API usage/cost report. This is not a billing invoice.",
        "stem": stem,
        "output_dir": str(output_dir),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "currency": {"usd_to_krw": USD_TO_KRW},
        "pricing_assumptions": {
            "token_prices_usd_per_1m": TOKEN_PRICES_USD_PER_1M,
            "audio_prices_usd_per_hour": AUDIO_PRICES_USD_PER_HOUR,
            "notes": [
                "Gemini thinking tokens are estimated as output-billed tokens.",
                "Image inputs are counted through provider token usage when available.",
                "Free tier, quota credits, region, batch mode, and rounding are not reflected.",
            ],
        },
        "summary": total,
        "by_stage": dict(sorted(by_stage.items())),
        "by_model": dict(sorted(by_model.items())),
        "events": events,
        "external": external,
        "timings": timings or {},
        "configured": dict(_configured),
    }


def write_report(
    *,
    stem: str,
    output_dir: Path | str,
    timings: Optional[dict[str, float]] = None,
    analyzer_output_path: Optional[Path | str] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}_cost_report.json"
    report = build_report(
        stem=stem,
        output_dir=output_dir,
        timings=timings,
        analyzer_output_path=analyzer_output_path,
    )
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
