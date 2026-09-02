"""Final verifier for classified issue candidates.

This module consumes ``classified_issue_input.v2`` produced by
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
    ALL_ISSUE_TYPES,
    COMPOSITE_ISSUE_TYPE,
    ISSUE_TYPES,
    TOKEN_USAGE_FIELDS,
    _aggregate_token_usage,
    _call_llm,
    _load_env,
    _parse_model_weights,
    _resolve_model_spec,
    _split_csv,
)


SCHEMA_VERSION = "classified_issue_verifier.v3"
DEFAULT_CONTEXT_WINDOW = 5
FINAL_VERIFIER_MAX_ATTEMPTS = max(
    1,
    int(os.getenv("CLASSIFIED_ISSUE_VERIFIER_MAX_ATTEMPTS", "3") or "3"),
)
# Deliberately keep one weighting policy for every issue type. The previous
# category-specific overrides made scope/confusing issues use different model
# priors from factual/temporal issues, which contradicted the equal-weight
# verifier setting.
JUDGMENTS = {
    "valid_issue",
    "partially_resolved",
    "not_issue",
    "insufficient_context",
}
CATEGORY_LABELS = {
    "factual_error": "사실 오류",
    "temporal_error": "오래된 내용",
    "scope_overclaim": "과도한 일반화",
    "confusing_explanation": "혼동 가능 설명",
    COMPOSITE_ISSUE_TYPE: "복합 오류",
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
            "객관적으로 틀린 명제를 실제로 포함하는지 평가하세요. 문맥과 슬라이드를 포함해도 같은 잘못된 "
            "명제가 남아 있으면 높게 주세요. 강의자가 앞 발화의 오류를 명시적으로 취소·대체하여 "
            "정정한 경우에만 그 해소를 반영하세요. 영문/외래어 용어를 한글로 옮긴 발음 표기, 음차, 전사 "
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
            "앞뒤 전사 context에서 강의자가 잘못된 의미를 얼마나 명시적으로 정정·취소·대체하는지 "
            "평가하세요. 뒤에서 올바른 공식이나 결과를 별도로 설명한 것만으로는 앞 오류가 해소되지 않으며, "
            "슬라이드에 정확한 내용이 있어도 발화와 충돌할 뿐 강의자가 이를 정정하지 않았다면 "
            "해소로 보지 마세요."
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
            "앞뒤 전사 설명이 오해 가능성을 얼마나 풀어주는지 평가하세요. 인접 context에서 강의자가 "
            "정확한 의미를 충분히 설명하면 높게 주고, 슬라이드와 발화가 충돌하여 어느 쪽이 맞는지 "
            "학습자가 판단해야 한다면 해소로 보지 마세요."
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
            "앞뒤 전사 context에서 강의자가 조건, 예외, 적용 대상, 범위를 충분히 복원하는지 평가하세요. "
            "발화상 범위 단정이 명확히 완화되거나 교육상 맥락에서 허용 가능한 설명으로 귀결되면 높게 주고, "
            "슬라이드에만 정확한 범위가 있고 발화의 닫힌 범위가 정정되지 않았다면 낮게 주세요."
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
    try:
        from .runtime_llm import configured_stage_models
    except ImportError:
        from runtime_llm import configured_stage_models
    return configured_stage_models("verify")


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


def _web_evidence_lookup(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for item in payload.get("evidence_items", []) or []:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id") or "").strip()
        if candidate_id:
            lookup[candidate_id] = item
    return lookup


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


def _flatten_issues(
    payload: dict[str, Any],
    *,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    standard_refs: list[dict[str, Any]] = []
    composite_refs: list[dict[str, Any]] = []
    issues_by_type = payload.get("issues_by_type") or {}
    if not isinstance(issues_by_type, dict):
        return standard_refs, composite_refs
    for category in ISSUE_TYPES:
        rows = issues_by_type.get(category) or []
        if not isinstance(rows, list):
            continue
        for index, issue in enumerate(rows):
            if not isinstance(issue, dict):
                continue
            standard_refs.append({
                "id": str(issue.get("issue_id") or f"{category}:{index + 1}"),
                "category": category,
                "category_label": CATEGORY_LABELS.get(category, category),
                "index": index,
                "issue": issue,
            })
    composite_rows = issues_by_type.get(COMPOSITE_ISSUE_TYPE) or []
    if isinstance(composite_rows, list):
        for index, issue in enumerate(composite_rows):
            if not isinstance(issue, dict):
                continue
            composite_refs.append({
                "id": str(issue.get("issue_id") or f"{COMPOSITE_ISSUE_TYPE}:{index + 1}"),
                "category": COMPOSITE_ISSUE_TYPE,
                "category_label": "복합 오류",
                "index": index,
                "issue": issue,
            })
    if limit is not None:
        cap = max(0, limit)
        return standard_refs[:cap], composite_refs[: max(0, cap - len(standard_refs))]
    return standard_refs, composite_refs


def _composite_candidate_categories(issue: dict[str, Any]) -> list[str]:
    reasons = set(issue.get("routing_reasons") or [])
    candidates: set[str] = set()
    weighted_scores = issue.get("weighted_scores") or {}
    if "low_margin" in reasons:
        sorted_types = sorted(
            ISSUE_TYPES,
            key=lambda issue_type: float(weighted_scores.get(issue_type, 0.0) or 0.0),
            reverse=True,
        )
        for issue_type in sorted_types[:2]:
            if float(weighted_scores.get(issue_type, 0.0) or 0.0) > 0.0:
                candidates.add(issue_type)
    if "model_disagreement" in reasons:
        for verdict in issue.get("model_classifications") or []:
            if not isinstance(verdict, dict):
                continue
            top = str(verdict.get("top_issue_type") or "").strip()
            if top in ISSUE_TYPES:
                candidates.add(top)
    if not candidates:
        if weighted_scores:
            candidates.add(
                max(ISSUE_TYPES, key=lambda issue_type: float(weighted_scores.get(issue_type, 0.0) or 0.0))
            )
        else:
            candidates.add(ISSUE_TYPES[0])
    return [issue_type for issue_type in ISSUE_TYPES if issue_type in candidates]


def _candidate_bundle_id(base_id: str, category: str) -> str:
    return f"{base_id}::{category}"


def _composite_category_label(candidate_categories: list[str]) -> str:
    labels = [
        CATEGORY_LABELS.get(category, category)
        for category in candidate_categories
        if category in ISSUE_TYPES
    ]
    if not labels:
        return CATEGORY_LABELS[COMPOSITE_ISSUE_TYPE]
    return f"{CATEGORY_LABELS[COMPOSITE_ISSUE_TYPE]}({', '.join(labels)})"


def _normalized_composite_probabilities(
    candidate_categories: list[str],
    weighted_scores: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    raw = {
        category: _clamp01(weighted_scores.get(category))
        for category in candidate_categories
        if category in ISSUE_TYPES
    }
    if not raw:
        return {}, {}

    total = sum(raw.values())
    if total <= 0.0:
        equal = round(1.0 / len(raw), 6)
        normalized = {category: equal for category in raw}
    else:
        normalized = {
            category: round(value / total, 6)
            for category, value in raw.items()
        }

    categories = list(normalized)
    if categories:
        previous_sum = sum(normalized[category] for category in categories[:-1])
        normalized[categories[-1]] = round(max(0.0, 1.0 - previous_sum), 6)
    return raw, normalized


def _build_candidate_ref(ref: dict[str, Any], category: str) -> dict[str, Any]:
    return {
        "id": _candidate_bundle_id(ref["id"], category),
        "category": category,
        "category_label": CATEGORY_LABELS.get(category, category),
        "index": ref.get("index"),
        "issue": ref["issue"],
        "composite_base_id": ref["id"],
    }


def _merge_composite_verification(
    ref: dict[str, Any],
    candidate_categories: list[str],
    candidate_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_categories = [
        category
        for category in candidate_categories
        if category in ISSUE_TYPES and category in candidate_records
    ]
    if not candidate_categories:
        raise ValueError(f"composite issue {ref.get('id', '')} has no candidate verification records")
    issue = ref["issue"]
    raw_probabilities, normalized_probabilities = _normalized_composite_probabilities(
        candidate_categories,
        issue.get("weighted_scores") or {},
    )
    candidate_scores = {
        category: _clamp01(candidate_records[category].get("final_severity_score"))
        for category in candidate_categories
    }
    candidate_contributions = {
        category: round(normalized_probabilities.get(category, 0.0) * candidate_scores.get(category, 0.0), 6)
        for category in candidate_categories
    }
    final_score = round(sum(candidate_contributions.values()), 6)
    primary_category = max(
        candidate_categories,
        key=lambda category: (
            candidate_contributions.get(category, 0.0),
            candidate_scores.get(category, 0.0),
            normalized_probabilities.get(category, 0.0),
            -ISSUE_TYPES.index(category),
        ),
    )
    selected = dict(candidate_records[primary_category])
    composite_label = _composite_category_label(candidate_categories)
    final_status = _status_from_severity(final_score)
    selected.update({
        "original_final_issue_type": COMPOSITE_ISSUE_TYPE,
        "original_final_issue_type_label": "복합 오류",
        "routing_reasons": issue.get("routing_reasons", []),
        "composite_candidate_categories": list(candidate_categories),
        "candidate_verifications": dict(candidate_records),
        "primary_issue_type": primary_category,
        "primary_issue_type_label": CATEGORY_LABELS.get(primary_category, primary_category),
        "scored_as_composite": True,
        "composite_scoring": {
            "method": "weighted_expected_severity",
            "raw_probabilities": raw_probabilities,
            "normalized_probabilities": normalized_probabilities,
            "candidate_scores": candidate_scores,
            "candidate_contributions": candidate_contributions,
            "weighted_score": final_score,
            "primary_issue_type": primary_category,
            "primary_issue_type_label": CATEGORY_LABELS.get(primary_category, primary_category),
        },
        "category": COMPOSITE_ISSUE_TYPE,
        "category_label": composite_label,
        "final_severity_score": final_score,
        "final_severity_percent": round(final_score * 100.0, 2),
        "needs_manual_review": final_status == "professor_check",
        "id": ref["id"],
    })
    previous = dict(selected.get("previous_classification") or {})
    previous.update({
        "final_issue_type": issue.get("final_issue_type", COMPOSITE_ISSUE_TYPE),
        "routing_reasons": issue.get("routing_reasons", []),
        "routed_to_composite": True,
        "weighted_scores": issue.get("weighted_scores", {}),
        "ensemble_confidence": issue.get("ensemble_confidence", 0.0),
        "low_margin": bool(issue.get("low_margin")),
        "margin": issue.get("margin", 0.0),
    })
    selected["previous_classification"] = previous
    return selected


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
    brief = {
        "id": item.get("id"),
        "issue_id": issue.get("issue_id", ""),
        "claim_id": issue.get("claim_id", ""),
        "claim_text": issue.get("claim_text", ""),
        "resolved_claim": issue.get("resolved_claim", ""),
        "basis_code": issue.get("basis_code", ""),
        "target_context_ids": context_bundle.get("target_context_ids", []),
    }
    web_evidence = item.get("web_evidence") if isinstance(item.get("web_evidence"), dict) else {}
    evidence = web_evidence.get("evidence") if isinstance(web_evidence.get("evidence"), list) else []
    evidence_status = str(web_evidence.get("status") or "").strip()
    compact_evidence = []
    relations: set[str] = set()
    if evidence_status == "verified":
        for row in evidence:
            if not isinstance(row, dict):
                continue
            key_sentence = str(row.get("key_sentence") or "").strip()
            if not key_sentence:
                continue
            relation = str(row.get("relation_to_claim") or "").strip()
            if relation not in {"supports_claim", "contradicts_claim"}:
                continue
            relations.add(relation)
            relevance = str(row.get("document_relevance") or "").strip().lower()
            if relevance not in {"direct", "partial"}:
                relevance = "direct"
            compact_evidence.append({
                "source_type": str(
                    row.get("assessed_source_class")
                    or row.get("source_priority_label")
                    or ""
                ).strip(),
                "source_strength": str(
                    row.get("source_strength") or "strong"
                ).strip(),
                "relevance": relevance,
                "claim_relation": relation,
                "key_sentence": key_sentence,
            })
            if len(compact_evidence) >= 3:
                break
    if compact_evidence:
        compact_web_evidence = {
            "status": "verified",
            "evidence": compact_evidence,
        }
        if len(relations) == 1:
            compact_web_evidence["claim_relation"] = next(iter(relations))
            for row in compact_evidence:
                row.pop("claim_relation", None)
        brief["web_evidence"] = compact_web_evidence
    elif evidence_status == "insufficient_evidence":
        partial_rows = (
            web_evidence.get("partial_evidence")
            if isinstance(web_evidence.get("partial_evidence"), list)
            else []
        )
        compact_partial_evidence = []
        for row in partial_rows:
            if not isinstance(row, dict):
                continue
            key_sentence = str(row.get("key_sentence") or "").strip()
            if not key_sentence:
                continue
            compact_partial_evidence.append({
                "source_type": str(
                    row.get("assessed_source_class")
                    or row.get("source_priority_label")
                    or ""
                ).strip(),
                "source_strength": str(
                    row.get("source_strength") or "supporting"
                ).strip(),
                "relevance": "partial",
                "key_sentence": key_sentence,
            })
            if len(compact_partial_evidence) >= 2:
                break
        brief["web_evidence"] = {
            "status": "insufficient_evidence",
            "evidence": compact_partial_evidence,
        }
    elif evidence_status == "grounding_unavailable":
        brief["web_evidence"] = {
            "status": "grounding_unavailable",
            "evidence": [],
        }
    return brief


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
    return """JSON 객체만 출력하세요. 모든 입력 id를 한 번씩 포함하세요.
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
점수는 0.0~1.0입니다. reason과 minimal_fix는 한국어로 작성하고 enum과 key는 그대로 사용하세요."""


def _final_verifier_schema(ids: list[str] | None = None) -> dict[str, Any]:
    """Provider-neutral schema translated by each API adapter."""
    allowed_ids = [str(value) for value in (ids or []) if str(value)]
    id_schema: dict[str, Any] = {"type": "string"}
    if allowed_ids:
        id_schema["enum"] = allowed_ids
    judgment_items: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": id_schema,
            "judgment": {
                "type": "string",
                "enum": [
                    "valid_issue",
                    "partially_resolved",
                    "not_issue",
                    "insufficient_context",
                ],
            },
            "is_valid_issue": {"type": "number", "minimum": 0, "maximum": 1},
            "category_severity": {"type": "number", "minimum": 0, "maximum": 1},
            "context_resolution": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
            "minimal_fix": {"type": "string"},
        },
        "required": [
            "id",
            "judgment",
            "is_valid_issue",
            "category_severity",
            "context_resolution",
            "reason",
            "minimal_fix",
        ],
    }
    judgments_schema: dict[str, Any] = {
        "type": "array",
        "items": judgment_items,
    }
    if allowed_ids:
        judgments_schema["minItems"] = len(allowed_ids)
        judgments_schema["maxItems"] = len(allowed_ids)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "judgments": judgments_schema,
        },
        "required": ["judgments"],
    }


def _build_prompt(category: str, items: list[dict[str, Any]], current_date: str) -> str:
    description = CATEGORY_DESCRIPTIONS.get(category, "")
    score_guide = CATEGORY_SCORE_GUIDES.get(category, {})
    return f"""당신은 분류가 끝난 강의 이슈 후보의 최종 verifier입니다.
현재 날짜: {current_date}
대상 분류: {CATEGORY_LABELS.get(category, category)} ({category})
정의: {description}

목표
- 입력 후보를 다른 유형으로 재분류하지 말고 대상 분류 안에서만 판정하세요.
- claim_text, resolved_claim, 같은 슬라이드 전사와 slide_text를 함께 읽어 학생에게 최종적으로 남는 의미를 판단하세요.
- resolved_claim이 원문보다 강하거나 넓으면 원문 문맥을 우선하세요.
- resolved_claim에 주어가 없거나 "해당 화면", "이 작품", "그 기술"처럼 대상이 식별되지 않고,
  claim_text에도 지시 대상을 해소한 명시적 대상명이 없다면 resolved_claim만으로 판정하지 마세요.
  target_context_ids에 해당하는 전사, 그 앞뒤 context, slide_text를 함께 확인하여 무엇에 관한
  claim인지 먼저 식별한 뒤 판정하세요.
- 판정 대상의 중심은 claim_text와 resolved_claim이 가리키는 동일한 주장입니다. 문맥은 그 주장의
  주어·지시 대상·생략된 조건·범위를 해소하는 데 사용하고, 주변 context의 다른 주장을 새로운
  판정 대상으로 바꾸지 마세요.
- 주변 전사와 슬라이드에서 하나의 대상만 명확히 연결되면 그 구체적 대상명을 기준으로 claim의 각
  관계·수치·방향·조건을 검사하세요. 대상을 복원한 뒤에도 하나의 claim에 독립적인 하위 주장이
  여러 개 있으면 각각 확인하고, 일부가 맞다는 이유로 다른 핵심 오류를 무시하지 마세요.
- 전사와 slide_text는 주어·지시 대상과 강의 문맥을 해소하는 자료입니다. 그 내용이 claim과
  일치한다는 사실만으로 claim이 참이라고 간주하지 말고, 외부 근거·확립된 정의·계산으로 검증하세요.
- 제공된 전사와 슬라이드를 모두 확인해도 대상을 하나로 특정할 수 없을 때만 insufficient_context로
  판정하세요.
- slide_text는 claim의 대상·관계·조건을 해석하고 발화와의 충돌을 찾는 보조 자료입니다.
  전사와 슬라이드가 일치한다는 사실만으로 claim이 참이라고 판단하지 마세요.

문맥·근거 규칙
1. 같은 대상의 인접 설명이 claim을 명시적으로 정정·조건화·한정하면 context_resolution을 높이고,
   단순 배경 설명이거나 같은 오류를 반복하면 높이지 마세요.
   명시적 정정은 앞 발화가 잘못되었음을 밝히거나 그 발화를 취소·대체하는 연결이 드러나는 경우입니다.
   뒤에서 올바른 공식·계산·결과를 제시한 것만으로는 앞 오류를 사실상 정정한 것으로 간주하지 마세요.
   정확한 slide_text와 잘못된 발화가 충돌하더라도 강의자가 발화를 명시적으로 정정하지 않았다면
   오류가 해소된 것이 아닙니다. 학습자에게 상충하는 정보가 남으므로 is_valid_issue와
   category_severity를 유지하고 context_resolution을 높이지 마세요.
2. 문맥상 ASR 흔들림·외래어 음차로 인해 문제가 생긴 것이라면 이슈로 보지 마세요. 비핵심 수치의 통용 가능한
   근사는 허용하되, 핵심 계산·정의·범위에 영향을 주면 검증하세요.
   특히 외래어·고유명사를 자연스러운 원어로 보정하면 설명이 맞고 다른 대상을 뜻한다는 문맥 근거가 없으면
   not_issue, is_valid_issue=0, category_severity=0으로 판정하세요.
   전사에 나타난 영문 글자·변수·기호의 대문자/소문자 차이는 음성에서 확정할 수 없는 표기 흔들림입니다.
   `N/n`, `X/x`처럼 전사나 resolved_claim의 case가 슬라이드와 다르거나 모델이 이를 "대문자/소문자"로
   해석했더라도, 그 차이만 문제라면 반드시 not_issue, is_valid_issue=0, category_severity=0으로 판정하세요.
   대·소문자와 무관한 별도의 개념·계산·범위 오류가 문맥에 실제로 남아 있을 때만 그 독립된 오류를 검증하세요.
   강의자가 사실·정의·계산·범위를 명확히 잘못 말한 경우에는 단순한 말실수라는 이유로 제외하지 마세요.
   뒤에서 올바른 내용을 설명하더라도 앞의 오류를 명시적으로 정정하지 않았다면 이슈를 유지하세요.
3. web_evidence.status="verified"의 key_sentence와 claim_relation은 검증된 외부 근거이지만 최종 판정은 아닙니다.
4. status="insufficient_evidence"의 문장은 명시된 범위에서만 사용하세요. 다만 그 문장과 현재 날짜,
   claim의 수치·단위 또는 다른 근거로 결과가 결정론적으로 계산·도출되면 근거로 사용할 수 있으며,
   reason에 계산 또는 논리 연결을 적으세요.
5. direct는 단독 근거가 될 수 있습니다. partial은 누락 범위까지 확대하지 마세요. strong은 단독 사용 가능하고,
   supporting은 다른 근거를 보조할 때만 사용하세요. 빈 evidence나 grounding_unavailable은 어느 방향의 증거도 아닙니다.
6. factual_error·temporal_error를 인정하려면 정확한 사실·정의·계산·직접 반례를 제시하세요.
   이를 특정할 수 없으면 추측하지 말고 insufficient_context로 판정하세요.

점수
- is_valid_issue: 대상 분류의 실제 이슈인 정도
- category_severity: 학생의 오개념·적용 판단에 미치는 심각도
- context_resolution: 강의자의 앞뒤 전사 설명이 이슈를 명시적으로 해소한 정도
- 문맥이 일부 해소하면 주로 context_resolution에 반영하고, 애초에 이슈가 아니면 is_valid_issue도 낮추세요.

분류별 기준
- is_valid_issue: {score_guide.get("is_valid_issue", "")}
- category_severity: {score_guide.get("category_severity", "")}
- context_resolution: {score_guide.get("context_resolution", "")}

라벨
- valid_issue: 유효한 이슈
- partially_resolved: 이슈가 있으나 문맥이 일부 해소
- not_issue: 이 분류의 이슈가 아님
- insufficient_context: 제공 자료로 판정 불가

reason은 한국어 1~2문장으로 결론, 정확한 근거·계산, 문맥 해소 여부를 포함하세요.
minimal_fix는 필요한 경우만 짧게 작성하세요.

{_response_contract()}

입력:
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


def _parse_failed_row(
    ref: dict[str, Any],
    model: str,
    resolved: dict[str, str],
    error: str,
    response_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        "response_metadata": response_metadata or {},
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
    usages: list[dict[str, Any]] = []
    last_by_id: dict[str, dict[str, Any]] = {}
    last_error = ""
    last_metadata: dict[str, Any] = {}
    resolved: dict[str, Any] = _resolve_model_spec(model)

    for attempt in range(1, FINAL_VERIFIER_MAX_ATTEMPTS + 1):
        text, usage, resolved = _call_llm(
            model_spec=model,
            prompt=prompt,
            max_tokens=max_tokens,
            web_search=False,
            structured_schema=_final_verifier_schema([ref["id"] for ref in batch]),
            stage="verify",
        )
        if isinstance(usage, dict):
            usages.append(usage)
            last_metadata = usage.get("response_metadata", {}) or {}

        by_id: dict[str, dict[str, Any]] = {}
        if not str(text or "").strip():
            last_error = "empty_model_response"
        else:
            try:
                rows = _parse_response(text)
                by_id = {
                    str(row.get("id") or ""): row
                    for row in rows
                    if isinstance(row, dict) and str(row.get("id") or "").strip()
                }
                missing_ids = [ref["id"] for ref in batch if ref["id"] not in by_id]
                last_error = f"missing judgment row: {', '.join(missing_ids)}" if missing_ids else ""
            except Exception as exc:
                last_error = str(exc)

        last_by_id = by_id
        if not last_error:
            normalized = [
                _normalize_judgment_row(
                    last_by_id[ref["id"]], ref=ref, model=model, resolved=resolved
                )
                for ref in batch
            ]
            return normalized, _aggregate_token_usage(usages)

        if attempt < FINAL_VERIFIER_MAX_ATTEMPTS:
            print(
                f"  [{model}] {category} 파싱 실패 — 즉시 재시도 "
                f"({attempt + 1}/{FINAL_VERIFIER_MAX_ATTEMPTS}): {last_error}",
                flush=True,
            )

    # Some providers may return only a subset of a multi-item tool call even
    # when the schema is valid. Recover only the missing rows with singleton
    # requests so a provider omission never becomes a false parse failure.
    missing_refs = [ref for ref in batch if ref["id"] not in last_by_id]
    for ref in missing_refs:
        single_prompt = _build_prompt(category, [ref], current_date)
        for single_attempt in range(1, FINAL_VERIFIER_MAX_ATTEMPTS + 1):
            single_text, single_usage, resolved = _call_llm(
                model_spec=model,
                prompt=single_prompt,
                max_tokens=max_tokens,
                web_search=False,
                structured_schema=_final_verifier_schema([ref["id"]]),
                stage="verify",
            )
            if isinstance(single_usage, dict):
                usages.append(single_usage)
                last_metadata = single_usage.get("response_metadata", {}) or {}
            try:
                single_rows = _parse_response(single_text)
                single_by_id = {
                    str(row.get("id") or ""): row
                    for row in single_rows
                    if isinstance(row, dict) and str(row.get("id") or "").strip()
                }
            except Exception:
                single_by_id = {}
            if ref["id"] in single_by_id:
                last_by_id[ref["id"]] = single_by_id[ref["id"]]
                break
            if single_attempt < FINAL_VERIFIER_MAX_ATTEMPTS:
                print(
                    f"  [{model}] {category} 단일 항목 재시도 "
                    f"({single_attempt + 1}/{FINAL_VERIFIER_MAX_ATTEMPTS}): {ref['id']}",
                    flush=True,
                )

    normalized = []
    for ref in batch:
        row = last_by_id.get(ref["id"])
        if isinstance(row, dict):
            normalized.append(_normalize_judgment_row(row, ref=ref, model=model, resolved=resolved))
        else:
            normalized.append(
                _parse_failed_row(
                    ref,
                    model,
                    resolved,
                    last_error or "missing judgment row",
                    response_metadata=last_metadata,
                )
            )
    return normalized, _aggregate_token_usage(usages)


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


def _effective_model_weights(
    category: str,
    verdicts: list[dict[str, Any]],
    base_weights: dict[str, float],
) -> dict[str, float]:
    del category, verdicts
    return base_weights


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
            "basis_code": issue.get("basis_code", ""),
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
        "web_evidence": ref.get("web_evidence", {}),
        "model_judgments": verdicts,
    }


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
        "breakdown_by_type": {category: by_type.get(category, 0) for category in ALL_ISSUE_TYPES},
        "composite_resolved_count": sum(1 for record in records if record.get("scored_as_composite")),
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
                "web_search_requests": int(result.get("web_search_requests", 0) or 0),
                "web_search_judgment_count": sum(
                    1 for row in result.get("judgments", []) or [] if row.get("web_search_used")
                ),
            }
            for model, result in model_results.items()
        },
    }


def _slim_classified_issue_view(result: dict[str, Any]) -> dict[str, Any]:
    """Keep final response metadata without duplicating full issue records."""
    return {
        key: result.get(key)
        for key in (
            "schema_version",
            "stage",
            "generated_at",
            "current_date",
            "categories",
            "model_weights",
            "summary",
            "model_results",
            "token_usage",
            "web_evidence",
            "output_path",
        )
        if result.get(key) not in (None, "", [], {})
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
            if "web_search_used" in row:
                model_row["web_search_used"] = bool(row.get("web_search_used"))
                model_row["web_evidence_status"] = row.get("web_evidence_status", "not_used")
                model_row["evidence_sources"] = row.get("evidence_sources", [])
            model_judgments.append(model_row)
        reason = " / ".join(
            row.get("reason", "")
            for row in model_judgments
            if str(row.get("reason", "")).strip()
        )
        web_evidence = issue.get("web_evidence") if isinstance(issue.get("web_evidence"), dict) else {}
        evidence_rows = web_evidence.get("evidence") if isinstance(web_evidence.get("evidence"), list) else []
        evidence_sources = [
            str(row.get("url") or "")
            for row in evidence_rows
            if isinstance(row, dict) and str(row.get("url") or "").strip()
        ]
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
                },
                "evidence": {
                    "slide_number": location.get("slide_number"),
                    "web_evidence": web_evidence,
                    "web_sources": evidence_sources,
                    "source_issue_ids": [issue.get("id") or issue_id],
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
                    "web_evidence": web_evidence,
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
        if issue.get("scored_as_composite"):
            feedback_items[-1]["classified_issue_verifier"]["scored_as_composite"] = True
            feedback_items[-1]["classified_issue_verifier"]["original_final_issue_type"] = issue.get(
                "original_final_issue_type",
                COMPOSITE_ISSUE_TYPE,
            )
            feedback_items[-1]["classified_issue_verifier"]["routing_reasons"] = issue.get("routing_reasons", [])
            feedback_items[-1]["classified_issue_verifier"]["composite_candidate_categories"] = issue.get(
                "composite_candidate_categories",
                [],
            )
            feedback_items[-1]["classified_issue_verifier"]["candidate_verifications"] = issue.get(
                "candidate_verifications",
                {},
            )
            feedback_items[-1]["classified_issue_verifier"]["primary_issue_type"] = issue.get("primary_issue_type", "")
            feedback_items[-1]["classified_issue_verifier"]["composite_scoring"] = issue.get("composite_scoring", {})

    confirmed = [item for item in feedback_items if item.get("status") == "confirmed"]
    review = [item for item in feedback_items if item.get("status") == "professor_check"]
    rejected = [item for item in feedback_items if item.get("status") == "rejected"]
    breakdown = Counter(item.get("feedback_type") or "unknown" for item in feedback_items)
    summary = {
        "total_feedback_count": len(feedback_items),
        "confirmed_feedback_count": len(confirmed),
        "review_needed_feedback_count": len(review),
        "rejected_feedback_count": len(rejected),
        "breakdown_by_type": dict(breakdown),
    }
    source_summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    evidence_meta = result.get("web_evidence") if isinstance(result.get("web_evidence"), dict) else {}
    evidence_summary = evidence_meta.get("summary") if isinstance(evidence_meta.get("summary"), dict) else {}
    if evidence_meta.get("enabled"):
        summary["web_evidence_target_count"] = int(evidence_summary.get("target_count", 0) or 0)
        summary["web_evidence_verified_count"] = int(evidence_summary.get("verified_count", 0) or 0)
        summary["web_evidence_insufficient_count"] = int(
            evidence_summary.get("insufficient_evidence_count", 0) or 0
        )

    return {
        "schema_version": "content_verification.v2",
        "mode": "classified_issue_verifier",
        "verification_date": result.get("generated_at", ""),
        "models": list((result.get("model_weights") or {}).keys()),
        "verifier_source_models": list((result.get("model_weights") or {}).keys()),
        "verifier_model_weights": result.get("model_weights", {}),
        "summary": summary,
        "counts": {
            "final_confirmed": len(confirmed),
            "needs_review": len(review),
            "rejected": len(rejected),
        },
        "claims": claims,
        "feedback_items": feedback_items,
        "views": {
            "classified_issue_verifier": _slim_classified_issue_view(result),
        },
        "claim_decision_flow_summary": {
            "final_confirmed_claim_count": len(confirmed),
            "needs_review_claim_count": len(review),
            "verifier_rejected_claim_count": len(rejected),
        },
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
    issue_ids: list[str] | None = None,
    dry_run: bool = False,
    model_weights_spec: str | None = None,
    web_evidence_payload: dict[str, Any] | None = None,
    web_evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    _load_env()
    if not models:
        raise RuntimeError("최종 검증에 사용할 모델이 선택되지 않았습니다.")
    standard_refs, composite_refs = _flatten_issues(payload, limit=limit)
    if issue_ids:
        wanted = {str(issue_id).strip() for issue_id in issue_ids if str(issue_id).strip()}
        standard_refs = [ref for ref in standard_refs if str(ref.get("id") or "") in wanted]
        composite_refs = [ref for ref in composite_refs if str(ref.get("id") or "") in wanted]
    merged_payload = _load_json(merged_clean_path)
    slide_lookup = _build_slide_lookup(merged_payload)
    context_by_id, contexts_by_slide = _build_context_lookup(merged_payload)
    domain = str(merged_payload.get("domain") or "")
    subdomain = str(merged_payload.get("subdomain") or "")
    evidence_lookup = _web_evidence_lookup(web_evidence_payload)

    def _make_bundles(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        for bundle in bundles:
            evidence = evidence_lookup.get(str(bundle.get("id") or ""))
            if isinstance(evidence, dict):
                bundle["web_evidence"] = evidence
        return bundles

    standard_bundles = _make_bundles(standard_refs)
    composite_plans: list[tuple[dict[str, Any], list[str], list[dict[str, Any]]]] = []
    composite_candidate_bundles: list[dict[str, Any]] = []
    for ref in composite_refs:
        candidate_categories = _composite_candidate_categories(ref["issue"])
        candidate_refs = [_build_candidate_ref(ref, category) for category in candidate_categories]
        candidate_bundles = _make_bundles(candidate_refs)
        composite_plans.append((ref, candidate_categories, candidate_bundles))
        composite_candidate_bundles.extend(candidate_bundles)

    bundles = standard_bundles + composite_candidate_bundles
    if composite_refs:
        print(
            f"  composite issue {len(composite_refs)}건 → "
            f"후보 검증 {len(composite_candidate_bundles)}건",
            flush=True,
        )

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
                "response_metadata_by_batch": [],
            }

        work_items_by_model: dict[str, list[tuple]] = {}
        for model in models:
            batches = _chunk_by_slide_and_category(bundles, batch_size)
            work_items_by_model[model] = []
            for batch_index, batch in enumerate(batches, start=1):
                category = _batch_category_label(batch)
                work_items_by_model[model].append((model, category, batch, batch_index, len(batches), current_date, max_tokens))

        # ``max_workers``는 모델별 한도다. 모델마다 별도 pool을 두어 한
        # 공급자의 느린 요청이 다른 모델의 동시성을 줄이지 않게 한다.
        def _run_model_batches(model: str, work_items: list[tuple]) -> tuple[str, list[dict[str, Any]], list[tuple[tuple, Exception]]]:
            completed: list[dict[str, Any]] = []
            failed: list[tuple[tuple, Exception]] = []
            with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(work_items)))) as executor:
                futures = {executor.submit(_batch_worker, args): args for args in work_items}
                for future in as_completed(futures):
                    args = futures[future]
                    try:
                        completed.append(future.result())
                    except Exception as exc:
                        failed.append((args, exc))
            return model, completed, failed

        with ThreadPoolExecutor(max_workers=max(1, len(work_items_by_model))) as model_executor:
            model_futures = {
                model_executor.submit(_run_model_batches, model, work_items): model
                for model, work_items in work_items_by_model.items()
            }
            for model_future in as_completed(model_futures):
                model, completed, failed = model_future.result()
                for result in completed:
                    model_results[result["model"]]["judgments"].extend(result["judgments"])
                    model_results[result["model"]]["token_usage_by_batch"].append(result["token_usage"])
                    metadata = result["token_usage"].get("response_metadata", {})
                    if metadata:
                        model_results[result["model"]]["response_metadata_by_batch"].append(
                            {
                                "batch_index": result["batch_index"],
                                **metadata,
                            }
                        )
                for args, exc in failed:
                    model_results[model]["status"] = "partial_failed"
                    model_results[model]["batch_errors"].append({
                        "category": args[1],
                        "batch_index": args[3],
                        "error": str(exc),
                    })

        for result in model_results.values():
            usages = result.pop("token_usage_by_batch", [])
            result["web_search_requests"] = sum(
                int(usage.get("web_search_requests", 0) or 0)
                for usage in usages
            )
            result["web_search_queries"] = list(dict.fromkeys(
                str(query)
                for usage in usages
                for query in usage.get("web_search_queries", []) or []
                if str(query).strip()
            ))
            result["web_search_sources"] = list(dict.fromkeys(
                str(source)
                for usage in usages
                for source in usage.get("web_search_sources", []) or []
                if str(source).strip()
            ))
            result["token_usage"] = _aggregate_token_usage(usages)

    model_weights = _parse_model_weights(model_weights_spec, models, model_results)
    verdicts_by_id: dict[str, list[dict[str, Any]]] = {}
    for result in model_results.values():
        for row in result.get("judgments", []) or []:
            verdicts_by_id.setdefault(str(row.get("id") or ""), []).append(row)

    records = [
        _issue_result_record(bundle, verdicts_by_id.get(bundle["id"], []), model_weights=model_weights)
        for bundle in standard_bundles
    ]
    for ref, candidate_categories, candidate_bundles in composite_plans:
        candidate_records = {
            bundle["category"]: _issue_result_record(
                bundle,
                verdicts_by_id.get(bundle["id"], []),
                model_weights=model_weights,
            )
            for bundle in candidate_bundles
        }
        records.append(_merge_composite_verification(ref, candidate_categories, candidate_records))
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
        "categories": {category: CATEGORY_LABELS.get(category, category) for category in ALL_ISSUE_TYPES},
        "model_weights": model_weights,
        "summary": _summary(records, model_results),
        "all_issues": records,
        "model_results": model_results,
        "token_usage": dict(token_usage),
        "web_evidence": {
            "enabled": bool(web_evidence_payload),
            "source_path": str(web_evidence_path or ""),
            "summary": (
                web_evidence_payload.get("summary", {})
                if isinstance(web_evidence_payload, dict)
                else {}
            ),
            "token_usage": (
                web_evidence_payload.get("token_usage", {})
                if isinstance(web_evidence_payload, dict)
                else {}
            ),
        },
        "web_search": {
            "enabled": any(
                int(result.get("web_search_requests", 0) or 0) > 0
                for result in model_results.values()
            ),
            "request_count": sum(
                int(result.get("web_search_requests", 0) or 0)
                for result in model_results.values()
            ),
            "judgment_count": sum(
                1
                for result in model_results.values()
                for row in result.get("judgments", []) or []
                if row.get("web_search_used")
            ),
        },
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
    parser.add_argument("input_json", help="classified_issue_input.v2 JSON path")
    parser.add_argument("-o", "--output", help="output JSON path")
    parser.add_argument("--merged-clean", help="merged_clean JSON path")
    parser.add_argument("--slide-textualized", help="slide_textualized JSON path")
    parser.add_argument("--slide-classified", help="slide_classified JSON path")
    parser.add_argument(
        "--models",
        default=",".join(_default_models()),
        help="comma/space separated model list. Defaults to the verify stage bindings.",
    )
    parser.add_argument(
        "--model-weights",
        default=None,
        help="deprecated compatibility option; selected models always receive equal weight",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("CLASSIFIED_ISSUE_VERIFIER_BATCH_SIZE", "4")),
    )
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("CLASSIFIED_ISSUE_VERIFIER_MAX_TOKENS", "8192")))
    parser.add_argument("--max-workers", type=int, default=int(os.getenv("CLASSIFIED_ISSUE_VERIFIER_MAX_WORKERS", "20")))
    parser.add_argument("--context-window", type=int, default=int(os.getenv("CLASSIFIED_ISSUE_VERIFIER_CONTEXT_WINDOW", str(DEFAULT_CONTEXT_WINDOW))))
    parser.add_argument("--current-date", default=os.getenv("CLASSIFIED_ISSUE_VERIFIER_CURRENT_DATE", "2026-05-14"))
    parser.add_argument("--limit", type=int, default=None, help="optional issue count limit for quick tests")
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="only verify the specified issue IDs (useful for retrying failed batches)",
    )
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
        raise ValueError("verify 단계에서 사용할 모델을 선택해야 합니다.")

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
        issue_ids=args.ids,
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
