"""Final verifier for classified issue candidates.

This module consumes ``classified_issue_input.v1`` produced by
``issue_type_classifier.py`` and runs a category-specific final verifier over each
already-classified issue.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
import re
from pathlib import Path
from typing import Any

from .issue_type_classifier import (
    ISSUE_TYPES,
    TOKEN_USAGE_FIELDS,
    _aggregate_token_usage,
    _call_llm,
    _load_env,
    _parse_model_weights,
    _resolve_model_spec,
    _split_csv,
)


SCHEMA_VERSION = "classified_issue_verifier.v1"
DEFAULT_MODEL_WEIGHTS = "gpt=0.4,claude=0.4,grok=0.2"
DEFAULT_MODELS = ("gpt", "claude", "grok")
DEFAULT_CONTEXT_WINDOW = 5
CATEGORY_MODEL_WEIGHT_OVERRIDES = {
    "scope_overclaim": {"gpt": 0.3, "openai": 0.3, "claude": 0.5, "anthropic": 0.5, "grok": 0.2, "xai": 0.2},
    "confusing_explanation": {"gpt": 0.3, "openai": 0.3, "claude": 0.5, "anthropic": 0.5, "grok": 0.2, "xai": 0.2},
}
JUDGMENTS = {
    "valid_issue",
    "partially_resolved",
    "not_issue",
    "insufficient_context",
}

CATEGORY_LABELS = {
    "factual_error": "사실 오류",
    "temporal_error": "오래된 내용",
    "confusing_explanation": "혼동 가능 설명",
    "scope_overclaim": "과도한 일반화",
}


CATEGORY_DESCRIPTIONS = {
    "factual_error": (
        "정의, 용어, 동작 원리, 관계, 순서, 메커니즘, 수식, 인과관계 등 "
        "기준일과 무관하게 명제 자체가 객관적으로 틀린 사실 오류 resolved_claim입니다."
    ),
    "temporal_error": (
        "과거 어느 시점에는 맞았거나 자연스러웠을 수 있지만, 현재 기준으로는 "
        "더 이상 맞지 않거나 현재 학습자에게 outdated 정보로 전달될 수 있는 resolved_claim입니다."
    ),
    "confusing_explanation": (
        "명제가 명백히 틀렸다고 단정하기보다는, 비유/예시/생략/표현 방식 때문에 "
        "학생이 다른 의미로 해석하거나 잘못된 오개념을 만들 위험이 있는 resolved_claim입니다."
    ),
    "scope_overclaim": (
        "조건, 예외, 범위, 적용 대상을 닫아버려 과도하게 일반화한 resolved_claim입니다. "
        '"항상/오직/모든/유일한/전부/완전히/~만" 같은 범위 표현을 완화하면 '
        "대체로 맞는 명제가 되는 경우를 포함합니다."
    ),
}


CATEGORY_SCORE_GUIDES = {
    "factual_error": {
        "is_valid_issue": (
            "resolved_claim이 정의, 용어, 동작 원리, 관계, 순서, 수식, 인과관계 측면에서 "
            "객관적으로 틀렸을 가능성을 평가하세요. 문맥과 슬라이드를 포함해도 같은 잘못된 "
            "명제가 남아 있으면 높게 주고, 문맥상 표현이 바로 정정되었거나 정확한 의미로 "
            "좁혀지면 낮게 주세요. 영문/외래어 용어를 한글로 옮긴 발음 표기, 음차, 전사 "
            "흔들림만 문제이고 문맥상 어떤 원어와 개념을 가리키는지 명확하면 사실 오류로 "
            "높게 채점하지 마세요. 수치 표현에서 약, 한, 대략, 정도, 조금 같은 근사 표현이 "
            "있고 그 수치가 핵심 개념이 아니라 보조 설명, 감각적 환산, 예시로 쓰였으며 "
            "일반적으로 통용되는 근사라면 정확한 수치와 차이가 있어도 사실 오류로 높게 "
            "채점하지 마세요."
        ),
        "category_severity": (
            "학생이 핵심 개념, 작동 원리, 문제 풀이 방식, 구현 판단, 후속 개념 이해를 "
            "잘못 학습할 위험이 클수록 높게 주세요. 강의 흐름에 큰 영향을 주지 않는 "
            "사소한 부정확성, 발음 표기 차이, 음차/전사 흔들림, 보조 예시의 일반적 근사 "
            "표현은 낮게 주세요."
        ),
        "context_resolution": (
            "앞뒤 context와 슬라이드 텍스트가 잘못된 의미를 얼마나 정정, 보완, "
            "조건화하는지 평가하세요. 명시적 정정이나 충분한 보완이 있으면 높게 주고, "
            "문맥을 봐도 같은 오류가 그대로 남으면 낮게 주세요."
        ),
    },
    "temporal_error": {
        "is_valid_issue": (
            "resolved_claim이 기준일 현재 더 이상 맞지 않거나, 현재 학습자에게 outdated 정보로 "
            "전달될 가능성을 평가하세요. 단순히 강의가 오래되었거나 날짜 표현이 있다는 "
            "이유만으로 높게 주지 마세요."
        ),
        "category_severity": (
            "현재 학습자의 도구 선택, 구현 방식, 지원 여부, 정책/버전 판단, 통계나 시장 상황 "
            "이해에 실제 영향을 줄수록 높게 주세요. 역사적 배경 설명이거나 현재 학습에 영향이 적다면 낮게 주세요."
        ),
        "context_resolution": (
            "문맥상 과거 시점, 역사적 상황, 당시 기준의 설명으로 명확히 제한되어 있으면 높게 주세요. "
            "현재 사실처럼 제시되고 보완 설명이 없으면 낮게 주세요."
        ),
    },
    "confusing_explanation": {
        "is_valid_issue": (
            "비유, 예시, 생략, 모호한 지시어, 압축된 표현 때문에 학생이 resolved_claim을 다른 의미로 "
            "해석하거나 잘못된 mental model을 만들 가능성을 평가하세요. 단순히 더 친절한 "
            "설명이 가능하다는 이유만으로 높게 주지 마세요."
            "일반적으로 맞는 표현이며, 일반적으로 맞는 설명이면 낮은 점수를 주세요."
        ),
        "category_severity": (
            "그 오해가 핵심 개념, 절차, 원인-결과, 구성 요소의 역할 이해를 크게 왜곡할수록 "
            "높게 주세요. 잠깐 헷갈릴 수 있으나 뒤 학습에 거의 영향을 주지 않는 표현은 낮게 주세요."
        ),
        "context_resolution": (
            "앞뒤 설명이 오해 가능성을 얼마나 풀어주는지 평가하세요. 같은 슬라이드나 인접 context에서 "
            "정확한 의미가 충분히 설명되면 높게 주고, 모호한 표현만 남아 있으면 낮게 주세요."
        ),
    },
    "scope_overclaim": {
        "is_valid_issue": (
            "조건, 예외, 적용 범위, 대상 집합을 닫아버려 일반적으로 맞지 않는 설명이 실제로 남아 있다면 점수를 높게 주세요."
            "범위 표현이 강조나 수사에 그치거나 일반적인 사례로 읽히면 낮게 주세요."
            "강조 표현, 범위 표현, 단정 표현이 문맥상 적절하게 사용되었는지 판단하세요."
            "이러한 표현이 포함되었다는 이유만으로 점수를 높게 주지 마세요."
            "점수를 높게 주려면, 해당 표현이 실제로 잘못된 범위 제한, 예외 배제, 조건 누락, 결과 과장으로 이어져야 합니다."
            "하지만 일반적으로 맞는 표현이며, 설명이면 낮은 점수를 주세요."
            "단정, 과장으로 인해 일어날 수 있는 반례가 일반적이지 않은 상황에 대한 반례라면, 굉장히 낮은 점수를 주세요."
        ),
        "category_severity": (
            "그 과잉 단정이 학생의 적용 범위 판단, 예외, 가능/불가능 판단, 전체/일부 구분을 "
            "크게 틀리게 만들수록 높게 주세요."
            "하지만 그 과잉 단정 표현을 포함하여도, 통상적으로 맞는 지식이고, 일반적으로 맞는 설명이면 점수를 낮게 주세요."
        ),
        "context_resolution": (
            "앞뒤 context와 슬라이드가 조건, 예외, 적용 대상, 범위를 충분히 복원하는지 평가하세요. "
            "문맥상 범위 단정이 명확히 완화되거나 교육상 맥락에서 허용 가능한 설명으로 귀결되면 높게 주고, "
            "닫힌 범위가 그대로 남으면 낮게 주세요."
        ),
    },
}


def _status_from_severity(score: float) -> str:
    confirmed = _safe_float(os.getenv("CLASSIFIED_ISSUE_VERIFIER_CONFIRMED_THRESHOLD"), 0.80)
    rejected = _safe_float(os.getenv("CLASSIFIED_ISSUE_VERIFIER_REJECTED_THRESHOLD"), 0.20)
    if score >= confirmed:
        return "confirmed"
    if score <= rejected:
        return "rejected"
    return "professor_check"


def _confirmed_threshold() -> float:
    return _safe_float(os.getenv("CLASSIFIED_ISSUE_VERIFIER_CONFIRMED_THRESHOLD"), 0.80)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _strip_json_fence(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in {float("inf"), float("-inf")}:
        return default
    return number


def _clamp01(value: Any, default: float = 0.0) -> float:
    return round(max(0.0, min(1.0, _safe_float(value, default))), 6)


def _default_models() -> list[str]:
    _load_env()
    configured = _split_csv(os.getenv("CLASSIFIED_ISSUE_VERIFIER_MODELS"))
    return configured or list(DEFAULT_MODELS)


def _chunk(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _bundle_slide_number(item: dict[str, Any]) -> int | None:
    issue = item.get("issue") if isinstance(item.get("issue"), dict) else {}
    return _slide_number(issue)


def _chunk_by_slide_and_category(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[int | str, str], list[dict[str, Any]]] = {}
    order: list[tuple[int | str, str]] = []
    for item in items:
        slide_number = _bundle_slide_number(item)
        slide_key: int | str = slide_number if slide_number is not None else "__unknown_slide__"
        category = str(item.get("category") or "").strip() or "__unknown_category__"
        key = (slide_key, category)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(item)

    batches: list[list[dict[str, Any]]] = []
    for key in order:
        batches.extend(_chunk(grouped[key], size))
    return batches


def _batch_category_label(items: list[dict[str, Any]]) -> str:
    categories = []
    for item in items:
        category = str(item.get("category") or "").strip()
        if category and category not in categories:
            categories.append(category)
    if not categories:
        return "mixed"
    if len(categories) == 1:
        return categories[0]
    raise ValueError(f"final verifier batch contains mixed categories: {categories}")


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _slide_number(issue: dict[str, Any]) -> int | None:
    location = issue.get("location") if isinstance(issue.get("location"), dict) else {}
    value = location.get("slide_number", issue.get("slide_number"))
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _issue_context_ids(issue: dict[str, Any]) -> list[str]:
    context = issue.get("context") if isinstance(issue.get("context"), dict) else {}
    values = context.get("context_ids") or issue.get("context_ids")
    if isinstance(values, list):
        ids = [str(value).strip() for value in values if str(value).strip()]
        if ids:
            return ids
    single = str(context.get("context_id") or issue.get("context_id") or "").strip()
    return [single] if single else []


def _flatten_issues(payload: dict[str, Any], *, limit: int | None = None) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    issues_by_type = payload.get("issues_by_type") or {}
    if not isinstance(issues_by_type, dict):
        return refs
    for category in ISSUE_TYPES:
        rows = issues_by_type.get(category) or []
        if not isinstance(rows, list):
            continue
        for index, issue in enumerate(rows):
            if not isinstance(issue, dict):
                continue
            refs.append({
                "id": str(issue.get("issue_id") or f"{category}:{index + 1}"),
                "category": category,
                "category_label": CATEGORY_LABELS.get(category, category),
                "index": index,
                "issue": issue,
            })
    if limit is not None:
        return refs[: max(0, limit)]
    return refs


def _build_slide_lookup(*payloads: dict[str, Any]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for payload in payloads:
        candidates = []
        if isinstance(payload.get("slides"), list):
            candidates.extend(payload.get("slides") or [])
        if isinstance(payload.get("scenes"), list):
            candidates.extend(payload.get("scenes") or [])
        for slide in candidates:
            if not isinstance(slide, dict):
                continue
            try:
                number = int(slide.get("slide_number") or slide.get("slide_canonical_number") or 0)
            except (TypeError, ValueError):
                continue
            if number <= 0:
                continue
            base = lookup.setdefault(number, {})
            for key, value in slide.items():
                if value not in (None, "", [], {}):
                    base[key] = value
    return lookup


def _build_context_lookup(merged_payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_slide: dict[int, list[dict[str, Any]]] = {}
    containers: list[dict[str, Any]] = []
    for key in ("slides", "scenes"):
        rows = merged_payload.get(key) or []
        if isinstance(rows, list):
            containers.extend(row for row in rows if isinstance(row, dict))

    next_index_by_slide: dict[int, int] = {}
    for container in containers:
        if not isinstance(container, dict):
            continue
        try:
            slide_number = int(container.get("slide_number") or container.get("slide_canonical_number") or 0)
        except (TypeError, ValueError):
            continue
        contexts = container.get("contexts") or []
        if not isinstance(contexts, list):
            continue
        for context in contexts:
            if not isinstance(context, dict):
                continue
            global_index = next_index_by_slide.get(slide_number, 0)
            next_index_by_slide[slide_number] = global_index + 1
            row = dict(context)
            row.setdefault("slide_number", slide_number)
            row["context_index"] = global_index
            context_id = str(row.get("context_id") or "").strip()
            if not context_id:
                context_id = f"S{slide_number:03d}-C{global_index + 1:03d}"
                row["context_id"] = context_id
            if context_id:
                by_id[context_id] = row
            by_slide.setdefault(slide_number, []).append(row)
    for rows in by_slide.values():
        rows.sort(key=lambda item: (
            int(item.get("context_index", 0) or 0),
            float(item.get("start_time", 0) or 0),
        ))
    return by_id, by_slide


def _compact_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_id": context.get("context_id", ""),
        "context_index": context.get("context_index"),
        "text": context.get("text", ""),
    }


def _joined_context_text(contexts: list[dict[str, Any]]) -> str:
    lines = []
    for context in contexts:
        context_id = str(context.get("context_id") or "").strip()
        text = str(context.get("text") or "").strip()
        if not text:
            continue
        prefix = f"{context_id}: " if context_id else ""
        lines.append(f"{prefix}{text}")
    return "\n".join(lines).strip()


def _adjacent_slide_number(contexts_by_slide: dict[int, list[dict[str, Any]]], slide_number: int, step: int) -> int | None:
    numbers = sorted(number for number, rows in contexts_by_slide.items() if rows)
    if slide_number not in numbers:
        return None
    index = numbers.index(slide_number) + step
    if index < 0 or index >= len(numbers):
        return None
    return numbers[index]


def _context_window(
    issue: dict[str, Any],
    *,
    context_by_id: dict[str, dict[str, Any]],
    contexts_by_slide: dict[int, list[dict[str, Any]]],
    window: int,
) -> dict[str, Any]:
    context_ids = _issue_context_ids(issue)
    center_contexts = [context_by_id[cid] for cid in context_ids if cid in context_by_id]
    slide_number = _slide_number(issue)
    if slide_number is None and center_contexts:
        try:
            slide_number = int(center_contexts[0].get("slide_number") or 0)
        except (TypeError, ValueError):
            slide_number = None
    if not center_contexts and slide_number:
        rows = contexts_by_slide.get(slide_number) or []
        if rows:
            center_contexts = rows[:1]

    current_slide_contexts = contexts_by_slide.get(slide_number or -1) or []
    local_contexts = current_slide_contexts
    if current_slide_contexts:
        indices: list[int] = []
        for context in center_contexts:
            try:
                indices.append(int(context.get("context_index", 0) or 0))
            except (TypeError, ValueError):
                continue
        if indices:
            start = max(0, min(indices) - max(0, window))
            end = min(len(current_slide_contexts), max(indices) + max(0, window) + 1)
            local_contexts = current_slide_contexts[start:end]

    return {
        "target_context_ids": context_ids,
        "current_slide_transcript": _joined_context_text(local_contexts),
        "window_contexts": [_compact_context(item) for item in local_contexts],
    }


def _compact_slide(slide: dict[str, Any]) -> dict[str, Any]:
    return {
        key: slide.get(key)
        for key in ("slide_number", "title", "slide_text")
        if slide.get(key) not in (None, "", [], {})
    }


def _build_context_bundle(
    ref: dict[str, Any],
    *,
    domain: str,
    subdomain: str,
    slide_lookup: dict[int, dict[str, Any]],
    context_by_id: dict[str, dict[str, Any]],
    contexts_by_slide: dict[int, list[dict[str, Any]]],
    context_window: int,
) -> dict[str, Any]:
    issue = ref["issue"]
    slide_number = _slide_number(issue)
    slide = slide_lookup.get(slide_number or -1, {})
    return {
        "id": ref["id"],
        "category": ref["category"],
        "category_label": ref["category_label"],
        "domain": domain,
        "subdomain": subdomain,
        "issue": issue,
        "slide": _compact_slide(slide),
        "context_bundle": _context_window(
            issue,
            context_by_id=context_by_id,
            contexts_by_slide=contexts_by_slide,
            window=context_window,
        ),
    }


def _prompt_issue_brief(item: dict[str, Any]) -> dict[str, Any]:
    issue = item.get("issue") or {}
    context_bundle = item.get("context_bundle") if isinstance(item.get("context_bundle"), dict) else {}
    return {
        "id": item.get("id"),
        "issue_id": issue.get("issue_id", ""),
        "claim_id": issue.get("claim_id", ""),
        "claim_text": issue.get("claim_text", ""),
        "resolved_claim": issue.get("resolved_claim", ""),
        "target_context_ids": context_bundle.get("target_context_ids", []),
    }


def _prompt_batch_context(items: list[dict[str, Any]]) -> dict[str, Any]:
    first = items[0] if items else {}
    merged_contexts: dict[str, dict[str, Any]] = {}
    slides: dict[str, dict[str, Any]] = {}
    for item in items:
        slide = item.get("slide") if isinstance(item.get("slide"), dict) else {}
        slide_number = slide.get("slide_number")
        if slide_number not in (None, "", [], {}):
            slides[str(slide_number)] = slide
        context_bundle = item.get("context_bundle") if isinstance(item.get("context_bundle"), dict) else {}
        for context in context_bundle.get("window_contexts", []) or []:
            if not isinstance(context, dict):
                continue
            context_id = str(context.get("context_id") or "").strip()
            if not context_id:
                continue
            merged_contexts[context_id] = context
    ordered_contexts = sorted(
        merged_contexts.values(),
        key=lambda item: (
            int(item.get("slide_number", 0) or 0),
            int(item.get("context_index", 0) or 0),
            str(item.get("context_id") or ""),
        ),
    )
    return {
        "domain": first.get("domain", ""),
        "subdomain": first.get("subdomain", ""),
        "slides": sorted(
            slides.values(),
            key=lambda item: int(item.get("slide_number", 0) or 0),
        ),
        "context_bundle": {
            "current_slide_transcript": _joined_context_text(ordered_contexts),
        },
    }


def _prompt_payload(items: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "batch_context": _prompt_batch_context(items),
            "issues": [_prompt_issue_brief(item) for item in items],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _response_contract() -> str:
    return """응답은 JSON 객체 하나만 출력하세요. markdown fence는 쓰지 마세요.
모든 입력 id에 대해 judgments 항목을 하나씩 포함하세요.

{
  "judgments": [
    {
      "id": "입력 id",
      "judgment": "valid_issue | partially_resolved | not_issue | insufficient_context",
      "is_valid_issue": 0.0,
      "category_severity": 0.0,
      "context_resolution": 0.0,
      "reason": "판단 근거 1~2문장",
      "minimal_fix": "필요한 경우만 짧게, 없으면 빈 문자열"
    }
  ]
}

공통 출력 규칙:
- is_valid_issue, category_severity, context_resolution은 0.0 이상 1.0 이하 숫자입니다.
- context_resolution은 문맥이 issue를 해소하는 정도입니다. 0.0은 전혀 해소 안 됨, 1.0은 거의 완전히 해소됨입니다.
- 바로 뒤 또는 같은 슬라이드의 설명이 같은 대상/관계/조건을 정확히 풀어주면 context_resolution을 높게 주세요."""


def _build_prompt(category: str, items: list[dict[str, Any]], current_date: str) -> str:
    description = CATEGORY_DESCRIPTIONS.get(category, "")
    score_guide = CATEGORY_SCORE_GUIDES.get(category, {})
    return f"""당신은 강의 verifier의 최종 FACT check 심사자입니다.

오늘 날짜: {current_date}

공통 문맥 해소 판단 순서:
0. resolved_claim과 claim_text/target context의 관계를 먼저 확인하세요.
   resolved_claim이 target context의 최종 전달 의미를 올바르게 정리한 문장이고,
   claim_text의 어색함이 말실수, 전사 흔들림, 즉시 재표현, 자기수정 수준이라면
   claim_text의 표면적 어색함만으로 점수를 높이지 마세요.
   반대로 resolved_claim이 target context의 실제 전달 의미보다 더 강하거나 넓게 정리되었다면,
   resolved_claim의 강해진 부분을 그대로 믿지 말고 target context 기준으로 낮게 판단하세요.
1. 먼저 target context 안에서 claim이 실제로 어떤 의미로 사용되었는지 판단하세요.
2. 바로 앞뒤 context가 같은 대상, 같은 관계, 같은 조건을 설명하는 경우에만 해소 근거로 사용하세요.
3. 같은 context 또는 바로 인접 context에서 같은 대상의 속성, 조건, 반환값, 구성요소를 이어서 설명하는 경우,
   앞선 claim만 단독으로 판단하지 말고 이어지는 설명까지 포함해 최종적으로 학생에게 남는 의미를 판단하세요.
   이어지는 설명이 앞선 claim의 누락된 부분을 명확히 보완하여 전체 설명이 일반 도메인 지식 기준으로 자연스럽게 맞아진다면,
   context_resolution을 높게 줄 수 있습니다.
   다만 이어지는 설명이 단순히 같은 주제를 말하는 수준이거나, 앞선 claim의 핵심 오류를 직접 보완하지 못한다면
   문맥 해소로 보지 마세요.
4. 문맥이 단순히 같은 주제를 말하거나 일반 배경을 제공하는 정도라면 해소 근거로 보지 마세요.
5. 문맥이 claim의 강한 표현을 예시, 대비, 강조, 교육적 단순화로 좁혀 주면 context_resolution을 높게 주세요.
6. 반대로 문맥이 같은 강한 표현을 반복하거나 강화하면 context_resolution을 낮게 주세요.
7. 문맥이 양쪽으로 읽히면, claim 자체가 일반적으로 맞는 설명인지 먼저 보세요. 일반적으로 맞는 설명이면 해소 쪽으로, 일반적으로 틀린 설명이면 미해소 쪽으로 판단하세요.
8. slide_text는 target claim을 해석하고 문맥 해소 여부를 판단하기 위한 보조 근거입니다.
   slide_text에 관련 개념이나 강한 표현이 있다는 이유만으로 is_valid_issue를 높이지 마세요.
   slide_text가 target claim의 대상, 관계, 조건, 범위를 더 정확하게 설명하면 context_resolution을 높게 주세요.
   다만 slide_text 자체가 target claim과 같은 잘못된 명제를 직접 반복하거나 강화할 때만 issue를 높이는 근거로 사용할 수 있습니다.
9. 잘못된 용어/분류명을 직접 발화한 경우, 뒤에서 상위 범주나 포함 관계를 설명하더라도 그 설명이 해당 용어/분류명 자체를 바로잡는지 확인하세요.
   해당 용어가 직접 정정되지 않았고, 학생이 그 대상을 잘못된 범주명으로 외울 가능성이 남으면 context_resolution을 낮게 주세요.
   다만 뒤 문맥이나 slide_text가 같은 대상을 더 정확한 용어로 명시하고, 잘못된 용어가 단순 말실수나 재표현 과정으로 해소되면 context_resolution을 높게 줄 수 있습니다.
   단, 영문/외래어 용어의 한글 발음 표기, 음차, 전사 흔들림만 있고 문맥상 지칭하는 원어와 개념이 명확하면 잘못된 용어/분류명 오류로 보지 마세요.
10. 수치 claim에서 약, 한, 대략, 정도, 조금 같은 근사 표현이 있고, 해당 수치가 핵심 학습 대상이 아니라 보조 설명, 감각적 환산, 예시로 쓰인 경우에는 정확한 수치와 차이가 있어도 일반적으로 통용되는 근사인지 먼저 판단하세요.
   일반적으로 통용되는 근사이면 사실 오류로 높게 채점하지 말고, 문맥상 정확한 수치 판단이 핵심일 때만 높게 채점하세요.

입력으로 제공되는 정보:
- claim의 도메인/서브도메인
- resolved_claim과 원문 claim_text (전사본에서, claim단위로 구성하여 제공, resolved_claim은 지시어를 보강한 claim, claim_text는 원문기반 claim)
- 해당 context와 앞뒤 context (전사본을 문맥 단위로 나누어 제공)
- 해당 슬라이드의 slide_text(merged_clean에서 제공되는 기본 슬라이드 텍스트)

출력 점수:
- is_valid_issue: 이 분류 기준으로 실제 issue일 가능성. 0.0~1.0 issue일수록 1에 수렴.
- category_severity: 이 분류 안에서 오류가 얼마나 심각한지. 0.0~1.0 심각할수록 1에 수렴.
- context_resolution: 제공된 문맥이 issue를 얼마나 해소하는지. 0.0은 전혀 해소 안 됨, 1.0은 거의 완전히 해소됨.

판정 라벨:
- valid_issue: 이 분류 기준에서 유효한 issue
- partially_resolved: issue 가능성은 있으나 문맥으로 일부 해소됨
- not_issue: 이 분류 기준에서는 issue가 아님
- insufficient_context: 제공 자료만으로 판단하기 어려움

중요:
- 모든 입력 id에 대해 judgments 항목을 하나씩 포함하세요.
- 응답은 JSON 객체 하나만 출력하세요.
- 점수는 모두 0.0 이상 1.0 이하 숫자여야 합니다.
- reason은 한두 문장으로 쓰되, 반드시 다음 순서로 작성하세요: 1) claim이 일반 도메인 지식 기준으로 맞는지/틀린지, 2) 틀렸다면 현재 분류 기준 때문에 틀린 것인지, 3) 제공 문맥이 이를 해소했는지.
- minimal_fix는 가능하면 claim을 어떻게 완화/수정하면 되는지 짧게 쓰고, 없으면 빈 문자열로 두세요.
- 문맥이 issue를 해소하는 정도는 주로 context_resolution에 반영하세요.
- 문맥 해소를 이유로 is_valid_issue와 category_severity를 동시에 과도하게 낮추지 마세요.
- 다만 문맥을 포함했을 때 애초에 issue가 성립하지 않는다면 is_valid_issue도 낮출 수 있습니다.

{_response_contract()}

판정 분류: {CATEGORY_LABELS.get(category, category)} ({category})

아래 분류 설명과 점수별 판단 기준은 이 요청에서 유일하게 적용할 기준입니다.
다른 분류로 재분류하지 말고, 이 분류 기준 안에서만 issue의 유효성, 심각성, 문맥 해소 정도를 판단하세요.

판정 분류 설명:
{description}

이 분류에서 is_valid_issue 판단 기준:
{score_guide.get("is_valid_issue", "")}

이 분류에서 category_severity 판단 기준:
{score_guide.get("category_severity", "")}

이 분류에서 context_resolution 판단 기준:
{score_guide.get("context_resolution", "")}

입력 issue:
{_prompt_payload(items)}
"""


def _parse_response(text: str) -> list[dict[str, Any]]:
    payload = json.loads(_strip_json_fence(text))
    rows = payload.get("judgments", [])
    return rows if isinstance(rows, list) else []


def _final_model_score(
    *,
    category: str,
    judgment: str,
    is_valid_issue: float,
    category_severity: float,
    context_unresolved: float,
) -> float:
    return _clamp01(is_valid_issue * category_severity * context_unresolved)


def _normalize_judgment_row(
    row: dict[str, Any],
    *,
    ref: dict[str, Any],
    model: str,
    resolved: dict[str, str],
) -> dict[str, Any]:
    raw_judgment = str(row.get("judgment") or "").strip()
    judgment = raw_judgment if raw_judgment in JUDGMENTS else "insufficient_context"
    is_valid_issue = _clamp01(row.get("is_valid_issue"))
    category_severity = _clamp01(row.get("category_severity"))
    context_resolution = _clamp01(row.get("context_resolution"))
    context_unresolved = _clamp01(1.0 - context_resolution)
    final_model_score = _final_model_score(
        category=ref["category"],
        judgment=judgment,
        is_valid_issue=is_valid_issue,
        category_severity=category_severity,
        context_unresolved=context_unresolved,
    )
    reason = str(row.get("reason", "") or "").strip()
    return {
        "id": ref["id"],
        "model": model,
        "provider": resolved.get("provider", ""),
        "resolved_model": resolved.get("resolved_model", model),
        "category": ref["category"],
        "judgment": judgment,
        "is_valid_issue": is_valid_issue,
        "category_severity": category_severity,
        "context_resolution": context_resolution,
        "context_unresolved": context_unresolved,
        "final_model_score": final_model_score,
        "reason": reason,
        "minimal_fix": str(row.get("minimal_fix", "") or "").strip(),
        "status": "ok",
        "parse_error": "",
    }


def _parse_failed_row(ref: dict[str, Any], model: str, resolved: dict[str, str], error: str) -> dict[str, Any]:
    row = {
        "id": ref["id"],
        "model": model,
        "provider": resolved.get("provider", ""),
        "resolved_model": resolved.get("resolved_model", model),
        "category": ref["category"],
        "judgment": "insufficient_context",
        "is_valid_issue": 0.0,
        "category_severity": 0.0,
        "context_resolution": 0.0,
        "context_unresolved": 1.0,
        "final_model_score": 0.0,
        "reason": "",
        "minimal_fix": "",
        "status": "parse_failed",
        "parse_error": error,
    }
    return row


def _call_model_for_batch(
    *,
    model: str,
    category: str,
    batch: list[dict[str, Any]],
    current_date: str,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = _build_prompt(category, batch, current_date)
    text, usage, resolved = _call_llm(model_spec=model, prompt=prompt, max_tokens=max_tokens)
    try:
        rows = _parse_response(text)
        by_id = {str(row.get("id") or ""): row for row in rows if isinstance(row, dict)}
    except Exception as exc:
        by_id = {}
        parse_error = str(exc)
    else:
        parse_error = ""

    normalized = []
    for ref in batch:
        row = by_id.get(ref["id"])
        if isinstance(row, dict):
            normalized.append(_normalize_judgment_row(row, ref=ref, model=model, resolved=resolved))
        else:
            normalized.append(_parse_failed_row(ref, model, resolved, parse_error or "missing judgment row"))
    return normalized, usage


def _batch_worker(args: tuple) -> dict[str, Any]:
    model, category, batch, batch_index, total_batches, current_date, max_tokens = args
    resolved = _resolve_model_spec(model)
    ids = f"{batch[0]['id']}..{batch[-1]['id']}" if batch else "-"
    slide_number = _bundle_slide_number(batch[0]) if batch else None
    slide_label = f"slide {slide_number}" if slide_number else "slide ?"
    print(
        f"  [{model}] {slide_label} {category} batch {batch_index}/{total_batches} 요청 중: {ids}",
        flush=True,
    )
    rows, usage = _call_model_for_batch(
        model=model,
        category=category,
        batch=batch,
        current_date=current_date,
        max_tokens=max_tokens,
    )
    for row in rows:
        row["batch_index"] = batch_index
    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    print(
        f"  [{model}] {slide_label} {category} batch {batch_index}/{total_batches} 완료: {ok_count}/{len(rows)} parsed",
        flush=True,
    )
    return {
        "model": model,
        "provider": resolved["provider"],
        "resolved_model": resolved["resolved_model"],
        "category": category,
        "batch_index": batch_index,
        "judgments": rows,
        "token_usage": usage,
    }


def _dry_run_model_results(models: list[str]) -> dict[str, dict[str, Any]]:
    results = {}
    for model in models:
        try:
            resolved = _resolve_model_spec(model)
        except Exception:
            resolved = {"provider": "unknown", "resolved_model": model}
        results[model] = {
            "model": model,
            "provider": resolved["provider"],
            "resolved_model": resolved["resolved_model"],
            "status": "dry_run",
            "judgments": [],
            "token_usage": {},
        }
    return results


def _weighted_final_score(
    verdicts: list[dict[str, Any]],
    model_weights: dict[str, float],
) -> tuple[float, dict[str, float], float, float, bool]:
    score = 0.0
    used_weights: dict[str, float] = {}
    missing_weight = 0.0
    model_scores = []
    for verdict in verdicts:
        model = str(verdict.get("model") or "")
        weight = float(model_weights.get(model, 0.0) or 0.0)
        if verdict.get("status") != "ok":
            missing_weight += weight
            continue
        model_score = _clamp01(verdict.get("final_model_score"))
        score += model_score * weight
        used_weights[model] = weight
        model_scores.append(model_score)
    if not used_weights:
        return 0.0, {}, round(missing_weight, 6), 0.0, False
    disagreement = max(model_scores) - min(model_scores) if len(model_scores) > 1 else 0.0
    return round(score, 6), used_weights, round(missing_weight, 6), round(disagreement, 6), bool(disagreement >= 0.35)


def _verdict_model_family(verdict: dict[str, Any]) -> str:
    model = str(verdict.get("model") or "").strip().lower()
    provider = str(verdict.get("provider") or "").strip().lower()
    resolved_model = str(verdict.get("resolved_model") or "").strip().lower()
    if provider in {"openai", "anthropic", "xai"}:
        return provider
    if model.startswith(("gpt", "o1", "o3")) or resolved_model.startswith(("gpt", "o1", "o3")):
        return "gpt"
    if (
        model.startswith("claude")
        or resolved_model.startswith("claude")
        or any(token in model for token in ("sonnet", "haiku", "opus"))
        or any(token in resolved_model for token in ("sonnet", "haiku", "opus"))
    ):
        return "claude"
    if model.startswith("grok") or resolved_model.startswith("grok"):
        return "xai"
    return model


def _effective_model_weights(
    category: str,
    verdicts: list[dict[str, Any]],
    base_weights: dict[str, float],
) -> dict[str, float]:
    override = CATEGORY_MODEL_WEIGHT_OVERRIDES.get(category)
    if not override:
        return base_weights

    weights: dict[str, float] = {}
    seen_models = {str(verdict.get("model") or "") for verdict in verdicts if str(verdict.get("model") or "")}
    for model in seen_models:
        verdict = next((row for row in verdicts if str(row.get("model") or "") == model), {})
        family = _verdict_model_family(verdict)
        weights[model] = float(override.get(family, 0.0) or 0.0)

    total = sum(weights.values())
    if total <= 0:
        return base_weights
    return {model: round(weight / total, 6) for model, weight in weights.items()}


def _issue_result_record(
    ref: dict[str, Any],
    verdicts: list[dict[str, Any]],
    *,
    model_weights: dict[str, float],
) -> dict[str, Any]:
    issue = ref["issue"]
    effective_model_weights = _effective_model_weights(ref["category"], verdicts, model_weights)
    final_score, used_weights, missing_weight, disagreement, needs_manual_review = _weighted_final_score(
        verdicts,
        effective_model_weights,
    )
    final_status = _status_from_severity(final_score)
    ok_verdicts = [row for row in verdicts if row.get("status") == "ok"]
    avg_is_valid = sum(_clamp01(row.get("is_valid_issue")) for row in ok_verdicts) / len(ok_verdicts) if ok_verdicts else 0.0
    avg_severity = (
        sum(_clamp01(row.get("category_severity")) for row in ok_verdicts) / len(ok_verdicts)
        if ok_verdicts
        else 0.0
    )
    avg_context_unresolved = (
        sum(_clamp01(row.get("context_unresolved")) for row in ok_verdicts) / len(ok_verdicts)
        if ok_verdicts
        else 0.0
    )
    avg_context_resolution = (
        sum(_clamp01(row.get("context_resolution")) for row in ok_verdicts) / len(ok_verdicts)
        if ok_verdicts
        else 0.0
    )
    return {
        "id": ref["id"],
        "issue_id": issue.get("issue_id", ""),
        "claim_id": issue.get("claim_id", ""),
        "resolved_claim": issue.get("resolved_claim", ""),
        "claim_text": issue.get("claim_text", ""),
        "category": ref["category"],
        "category_label": ref["category_label"],
        "location": issue.get("location", {}),
        "context": issue.get("context", {}),
        "judge_context": {
            "domain": ref.get("domain", ""),
            "subdomain": ref.get("subdomain", ""),
            "slide": ref.get("slide", {}),
            "context_bundle": ref.get("context_bundle", {}),
        },
        "previous_classification": {
            "weighted_scores": issue.get("weighted_scores", {}),
            "ensemble_confidence": issue.get("ensemble_confidence", 0.0),
            "low_margin": bool(issue.get("low_margin")),
            "margin": issue.get("margin", 0.0),
        },
        "final_severity_score": final_score,
        "final_severity_percent": round(final_score * 100.0, 2),
        "average_is_valid_issue": round(avg_is_valid, 6),
        "average_category_severity": round(avg_severity, 6),
        "average_context_resolution": round(avg_context_resolution, 6),
        "average_context_unresolved": round(avg_context_unresolved, 6),
        "model_weights": used_weights,
        "missing_model_weight": missing_weight,
        "model_disagreement": disagreement,
        "model_disagreement_needs_review": needs_manual_review,
        "needs_manual_review": final_status == "professor_check",
        "model_judgments": verdicts,
    }


def _group_issue_results(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {category: [] for category in ISSUE_TYPES}
    for record in records:
        category = record.get("category")
        if category in grouped:
            grouped[category].append(record)
    for rows in grouped.values():
        rows.sort(key=lambda item: (
            -float(item.get("final_severity_score") or 0.0),
            float((item.get("location") or {}).get("start_time") or 0.0),
            str(item.get("issue_id") or ""),
        ))
    return grouped


def _summary(records: list[dict[str, Any]], model_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(record.get("category") or "unknown" for record in records)
    manual_review_count = sum(1 for record in records if record.get("needs_manual_review"))
    high_count = sum(
        1
        for record in records
        if float(record.get("final_severity_score") or 0.0) >= _confirmed_threshold()
    )
    return {
        "total_issue_count": len(records),
        "breakdown_by_type": {category: by_type.get(category, 0) for category in ISSUE_TYPES},
        "high_severity_count": high_count,
        "needs_manual_review_count": manual_review_count,
        "model_breakdown": {
            model: {
                "status": result.get("status", ""),
                "provider": result.get("provider", ""),
                "resolved_model": result.get("resolved_model", ""),
                "judgment_count": len(result.get("judgments", []) or []),
                "parse_failed_count": sum(
                    1 for row in result.get("judgments", []) or [] if row.get("status") != "ok"
                ),
            }
            for model, result in model_results.items()
        },
    }


def build_content_verification_view(result: dict[str, Any]) -> dict[str, Any]:
    """Convert severity output into the web verifier response shape.

    The frontend already knows how to render ``content_verification.v2`` style
    ``feedback_items``. Keeping this adapter here lets the new classified issue
    pipeline show up in the existing verifier page without a separate UI pass.
    """

    all_issues = result.get("all_issues", []) or []
    feedback_items = []
    claims = []
    for index, issue in enumerate(all_issues, start=1):
        if not isinstance(issue, dict):
            continue
        score = _clamp01(issue.get("final_severity_score"))
        status = _status_from_severity(score)
        issue_id = str(issue.get("issue_id") or f"I{index:04d}")
        claim_id = str(issue.get("claim_id") or issue_id)
        category = str(issue.get("category") or "")
        category_label = str(issue.get("category_label") or CATEGORY_LABELS.get(category, category))
        model_judgments = []
        for row in issue.get("model_judgments", []) or []:
            if not isinstance(row, dict):
                continue
            model_row = {
                "model": row.get("model", ""),
                "resolved_model": row.get("resolved_model", ""),
                "provider": row.get("provider", ""),
                "decision": row.get("judgment", ""),
                "status": row.get("status", ""),
                "confidence": row.get("final_model_score", 0.0),
                "vote_score": row.get("final_model_score", 0.0),
                "score": row.get("final_model_score", 0.0),
                "model_weight": (issue.get("model_weights") or {}).get(row.get("model", ""), 0.0),
                "reason": row.get("reason", ""),
                "is_valid_issue": row.get("is_valid_issue", 0.0),
            }
            if "minimal_fix" in row:
                model_row["minimal_fix"] = row.get("minimal_fix", "")
            if "category_severity" in row:
                model_row["category_severity"] = row.get("category_severity", 0.0)
            if "context_resolution" in row:
                model_row["context_resolution"] = row.get("context_resolution", 0.0)
            if "context_unresolved" in row:
                model_row["context_unresolved"] = row.get("context_unresolved", 0.0)
            model_judgments.append(model_row)
        reason = " / ".join(
            row.get("reason", "")
            for row in model_judgments
            if str(row.get("reason", "")).strip()
        )
        web_grounding = issue.get("web_grounding") if isinstance(issue.get("web_grounding"), dict) else {}
        grounding_reason = str(web_grounding.get("reason") or "").strip()
        grounding_sources = web_grounding.get("evidence_sources") if isinstance(web_grounding.get("evidence_sources"), list) else []
        if web_grounding.get("status") == "refutes_issue" and grounding_reason:
            reason = f"웹 근거로 기각: {grounding_reason}" + (f" / 기존 모델 판단: {reason}" if reason else "")
        elif web_grounding.get("status") == "supports_issue" and grounding_reason:
            reason = f"웹 근거로 확인: {grounding_reason}" + (f" / 기존 모델 판단: {reason}" if reason else "")
        minimal_fix = next(
            (
                row.get("minimal_fix", "")
                for row in model_judgments
                if str(row.get("minimal_fix", "")).strip()
            ),
            "",
        )
        location = issue.get("location") if isinstance(issue.get("location"), dict) else {}
        context = issue.get("context") if isinstance(issue.get("context"), dict) else {}
        claim = {
            "claim_id": claim_id,
            "claim_text": issue.get("claim_text", ""),
            "resolved_claim": issue.get("resolved_claim", ""),
            "claim_type": category,
            "context_id": context.get("context_id", ""),
            "context_ids": context.get("context_ids", []),
            "slide_number": location.get("slide_number"),
            "start_time": location.get("start_time"),
            "end_time": location.get("end_time"),
        }
        claims.append(claim)
        feedback_items.append(
            {
                "feedback_id": f"F{index:04d}",
                "issue_id": issue_id,
                "source_claim_id": claim_id,
                "status": status,
                "feedback_type": category,
                "feedback_label": category_label,
                "claim_text": issue.get("claim_text", ""),
                "resolved_claim": issue.get("resolved_claim", ""),
                "location": location,
                "severity_score": score,
                "severity_score_percent": round(score * 100.0, 2),
                "severity_status": status,
                "problem": {
                    "problematic_content": issue.get("resolved_claim") or issue.get("claim_text", ""),
                    "summary": reason or f"{category_label} 후보입니다.",
                    "why_wrong": reason,
                    "issue_basis": category_label,
                    "recommendation": minimal_fix,
                    "correct_info": minimal_fix,
                },
                "professor_feedback": {
                    "summary": reason,
                    "why_wrong": reason,
                    "teaching_note": minimal_fix,
                    "suggested_rephrase": minimal_fix,
                },
                "evidence": {
                    "slide_number": location.get("slide_number"),
                    "evidence_in_context": reason,
                    "web_grounding": web_grounding,
                    "web_sources": grounding_sources,
                    "source_issues": [issue],
                },
                "checks": {
                    "severity": {
                        "score": score,
                        "score_percent": round(score * 100.0, 2),
                        "status_by_score": status,
                        "verdict": status,
                        "model_results": model_judgments,
                    }
                },
                "classified_issue_verifier": {
                    "final_severity_score": score,
                    "final_severity_percent": issue.get("final_severity_percent", round(score * 100.0, 2)),
                    "average_is_valid_issue": issue.get("average_is_valid_issue", 0.0),
                    "average_context_resolution": issue.get("average_context_resolution", 0.0),
                    "model_disagreement": issue.get("model_disagreement", 0.0),
                    "needs_manual_review": bool(issue.get("needs_manual_review")),
                    "web_grounding": web_grounding,
                },
            }
        )
        if "average_context_resolution" in issue:
            feedback_items[-1]["problem"]["context_resolution"] = f"{issue.get('average_context_resolution', 0.0):.2f}"
            feedback_items[-1]["classified_issue_verifier"]["average_context_resolution"] = issue.get(
                "average_context_resolution",
                0.0,
            )
        if "average_context_unresolved" in issue:
            feedback_items[-1]["problem"]["context_unresolved"] = f"{issue.get('average_context_unresolved', 0.0):.2f}"
            feedback_items[-1]["classified_issue_verifier"]["average_context_unresolved"] = issue.get(
                "average_context_unresolved",
                0.0,
            )
        if "average_category_severity" in issue:
            feedback_items[-1]["classified_issue_verifier"]["average_category_severity"] = issue.get(
                "average_category_severity",
                0.0,
            )

    confirmed = [item for item in feedback_items if item.get("status") == "confirmed"]
    review = [item for item in feedback_items if item.get("status") == "professor_check"]
    rejected = [item for item in feedback_items if item.get("status") == "rejected"]
    breakdown = Counter(item.get("feedback_type") or "unknown" for item in feedback_items)
    return {
        "schema_version": "content_verification.v2",
        "mode": "classified_issue_verifier",
        "verification_date": result.get("generated_at", ""),
        "models": list((result.get("model_weights") or {}).keys()),
        "verifier_source_models": list((result.get("model_weights") or {}).keys()),
        "verifier_model_weights": result.get("model_weights", {}),
        "summary": {
            "total_feedback_count": len(feedback_items),
            "confirmed_feedback_count": len(confirmed),
            "review_needed_feedback_count": len(review),
            "rejected_feedback_count": len(rejected),
            "breakdown_by_type": dict(breakdown),
        },
        "counts": {
            "final_confirmed": len(confirmed),
            "needs_review": len(review),
            "rejected": len(rejected),
        },
        "claims": claims,
        "feedback_items": feedback_items,
        "views": {
            "classified_issue_verifier": result,
        },
        "final_confirmed_claims": confirmed,
        "needs_review_claims": review,
        "verifier_rejected_claims": rejected,
        "claim_decision_flow_summary": {
            "final_confirmed_claim_count": len(confirmed),
            "needs_review_claim_count": len(review),
            "verifier_rejected_claim_count": len(rejected),
        },
        "issues": all_issues,
        "classified_issue_verifier_path": result.get("output_path", ""),
    }


def judge_classified_issues(
    payload: dict[str, Any],
    *,
    input_path: str | Path,
    merged_clean_path: str | Path | None,
    slide_textualized_path: str | Path | None,
    slide_classified_path: str | Path | None,
    models: list[str],
    batch_size: int,
    current_date: str,
    max_tokens: int,
    max_workers: int,
    context_window: int,
    limit: int | None = None,
    dry_run: bool = False,
    model_weights_spec: str | None = None,
) -> dict[str, Any]:
    _load_env()
    refs = _flatten_issues(payload, limit=limit)
    merged_payload = _load_json(merged_clean_path)
    slide_lookup = _build_slide_lookup(merged_payload)
    context_by_id, contexts_by_slide = _build_context_lookup(merged_payload)
    domain = str(merged_payload.get("domain") or "")
    subdomain = str(merged_payload.get("subdomain") or "")
    bundles = [
        _build_context_bundle(
            ref,
            domain=domain,
            subdomain=subdomain,
            slide_lookup=slide_lookup,
            context_by_id=context_by_id,
            contexts_by_slide=contexts_by_slide,
            context_window=context_window,
        )
        for ref in refs
    ]

    model_results: dict[str, dict[str, Any]]
    if dry_run:
        model_results = _dry_run_model_results(models)
    else:
        model_results = {}
        for model in models:
            try:
                resolved = _resolve_model_spec(model)
            except Exception:
                resolved = {"provider": "unknown", "resolved_model": model}
            model_results[model] = {
                "model": model,
                "provider": resolved["provider"],
                "resolved_model": resolved["resolved_model"],
                "status": "ok",
                "judgments": [],
                "token_usage_by_batch": [],
                "batch_errors": [],
            }

        work_items = []
        for model in models:
            batches = _chunk_by_slide_and_category(bundles, batch_size)
            for batch_index, batch in enumerate(batches, start=1):
                category = _batch_category_label(batch)
                work_items.append((model, category, batch, batch_index, len(batches), current_date, max_tokens))

        if max_workers <= 1:
            for args in work_items:
                try:
                    result = _batch_worker(args)
                except Exception as exc:
                    model = args[0]
                    model_results[model]["status"] = "partial_failed"
                    model_results[model]["batch_errors"].append({"category": args[1], "batch_index": args[3], "error": str(exc)})
                    continue
                model_results[result["model"]]["judgments"].extend(result["judgments"])
                model_results[result["model"]]["token_usage_by_batch"].append(result["token_usage"])
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_batch_worker, args): args for args in work_items}
                for future in as_completed(futures):
                    args = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        model = args[0]
                        model_results[model]["status"] = "partial_failed"
                        model_results[model]["batch_errors"].append({
                            "category": args[1],
                            "batch_index": args[3],
                            "error": str(exc),
                        })
                        continue
                    model_results[result["model"]]["judgments"].extend(result["judgments"])
                    model_results[result["model"]]["token_usage_by_batch"].append(result["token_usage"])

        for result in model_results.values():
            usages = result.pop("token_usage_by_batch", [])
            result["token_usage"] = _aggregate_token_usage(usages)

    model_weights = _parse_model_weights(model_weights_spec, models, model_results)
    verdicts_by_id: dict[str, list[dict[str, Any]]] = {}
    for result in model_results.values():
        for row in result.get("judgments", []) or []:
            verdicts_by_id.setdefault(str(row.get("id") or ""), []).append(row)

    records = [
        _issue_result_record(bundle, verdicts_by_id.get(bundle["id"], []), model_weights=model_weights)
        for bundle in bundles
    ]
    grouped = _group_issue_results(records)
    token_usage = Counter()
    for result in model_results.values():
        usage = result.get("token_usage") or {}
        for key in TOKEN_USAGE_FIELDS:
            token_usage[key] += int(usage.get(key, 0) or 0)

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "classified_issue_verifier",
        "source_input_path": str(input_path),
        "source_classification_path": payload.get("source_classification_path", ""),
        "source_issue_path": payload.get("source_issue_path", ""),
        "merged_clean_path": str(merged_clean_path or ""),
        "slide_textualized_path": str(slide_textualized_path or ""),
        "slide_classified_path": str(slide_classified_path or ""),
        "generated_at": _now_iso(),
        "current_date": current_date,
        "categories": {category: CATEGORY_LABELS.get(category, category) for category in ISSUE_TYPES},
        "model_weights": model_weights,
        "summary": _summary(records, model_results),
        "issues_by_type": grouped,
        "all_issues": records,
        "model_results": model_results,
        "token_usage": dict(token_usage),
    }


def _default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("_classified_issues"):
        stem = stem[: -len("_classified_issues")]
    elif stem.endswith("_issue_judge"):
        stem = stem[: -len("_issue_judge")]
    return input_path.with_name(f"{stem}_classified_issue_verifier.json")


def _guess_related_path(input_path: Path, suffix: str) -> Path:
    stem = input_path.stem
    if stem.endswith("_classified_issues"):
        prefix = stem[: -len("_classified_issues")]
    else:
        prefix = stem
    if suffix.startswith("../"):
        return (input_path.parent / suffix).resolve()
    return input_path.with_name(f"{prefix}{suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the category-specific final verifier over classified issues.",
    )
    parser.add_argument("input_json", help="classified_issue_input.v1 JSON path")
    parser.add_argument("-o", "--output", help="output JSON path")
    parser.add_argument("--merged-clean", help="merged_clean JSON path")
    parser.add_argument("--slide-textualized", help="slide_textualized JSON path")
    parser.add_argument("--slide-classified", help="slide_classified JSON path")
    parser.add_argument(
        "--models",
        default=",".join(_default_models()),
        help="comma/space separated model list. Default: CLASSIFIED_ISSUE_VERIFIER_MODELS or preset",
    )
    parser.add_argument(
        "--model-weights",
        default=os.getenv("CLASSIFIED_ISSUE_VERIFIER_MODEL_WEIGHTS", DEFAULT_MODEL_WEIGHTS),
        help="comma/space separated weights, e.g. gpt=0.4,claude=0.4,grok=0.2",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("CLASSIFIED_ISSUE_VERIFIER_BATCH_SIZE", "4")),
    )
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("CLASSIFIED_ISSUE_VERIFIER_MAX_TOKENS", "8192")))
    parser.add_argument("--max-workers", type=int, default=int(os.getenv("CLASSIFIED_ISSUE_VERIFIER_MAX_WORKERS", "12")))
    parser.add_argument("--context-window", type=int, default=int(os.getenv("CLASSIFIED_ISSUE_VERIFIER_CONTEXT_WINDOW", str(DEFAULT_CONTEXT_WINDOW))))
    parser.add_argument("--current-date", default=os.getenv("CLASSIFIED_ISSUE_VERIFIER_CURRENT_DATE", "2026-05-14"))
    parser.add_argument("--limit", type=int, default=None, help="optional issue count limit for quick tests")
    parser.add_argument("--dry-run", action="store_true", help="validate input/output shape without calling LLMs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input_json)
    output_path = Path(args.output) if args.output else _default_output_path(input_path)
    merged_clean_path = Path(args.merged_clean) if args.merged_clean else _guess_related_path(input_path, "_merged_clean.json")
    slide_textualized_path = (
        Path(args.slide_textualized)
        if args.slide_textualized
        else _guess_related_path(input_path, "../" + input_path.parent.parent.name + "_slide_textualized.json")
    )
    slide_classified_path = (
        Path(args.slide_classified)
        if args.slide_classified
        else _guess_related_path(input_path, "../" + input_path.parent.parent.name + "_slide_classified.json")
    )

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("입력 JSON 최상위 객체는 dict여야 합니다.")

    models = _split_csv(args.models)
    if not models:
        raise ValueError("사용할 모델이 없습니다. --models 또는 CLASSIFIED_ISSUE_VERIFIER_MODELS를 설정하세요.")

    result = judge_classified_issues(
        payload,
        input_path=input_path,
        merged_clean_path=merged_clean_path,
        slide_textualized_path=slide_textualized_path,
        slide_classified_path=slide_classified_path,
        models=models,
        batch_size=max(1, args.batch_size),
        current_date=args.current_date,
        max_tokens=max(256, args.max_tokens),
        max_workers=max(1, args.max_workers),
        context_window=max(0, args.context_window),
        limit=args.limit,
        dry_run=args.dry_run,
        model_weights_spec=args.model_weights,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = result["summary"]
    print(f"입력 issue: {summary['total_issue_count']}건")
    print(f"모델: {', '.join(models)}")
    print(f"모델 가중치: {json.dumps(result.get('model_weights', {}), ensure_ascii=False)}")
    print(f"출력: {output_path}")
    print(f"유형별 분포: {json.dumps(summary['breakdown_by_type'], ensure_ascii=False)}")
    print(f"high severity: {summary['high_severity_count']}건")
    print(f"needs manual review: {summary['needs_manual_review_count']}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
