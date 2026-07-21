from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed


_CANONICAL_CLAIM_TYPES = {
    "definition",
    "numeric",
    "causal",
    "relationship",
    "currentness",
}


def _claim_extract_context_mode() -> str:
    mode = str(os.getenv("VERIFIER_CLAIM_EXTRACT_CONTEXT_MODE", "cards") or "cards").strip().lower()
    if mode in {"compact", "batch", "transcript"}:
        return "compact"
    if mode in {"card_lite", "card-lite", "lite", "cards_lite", "cards-lite"}:
        return "card_lite"
    return "cards"


def _claim_extract_batch_mode() -> str:
    mode = str(os.getenv("VERIFIER_CLAIM_EXTRACT_BATCH_MODE", "context") or "context").strip().lower()
    if mode in {"slide", "slides", "slide_batch", "slide-batch", "by_slide", "by-slide"}:
        return "slide"
    return "context"


def _claim_extract_context_window() -> tuple[int, int]:
    return (
        _read_int_env("VERIFIER_CLAIM_EXTRACT_CONTEXT_PREV", 1, minimum=0),
        _read_int_env("VERIFIER_CLAIM_EXTRACT_CONTEXT_NEXT", 1, minimum=0),
    )


def _claim_extract_context_group_size() -> int:
    return _read_int_env("VERIFIER_CLAIM_EXTRACT_CONTEXT_GROUP_SIZE", 3, minimum=1)


def _read_int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = str(os.getenv(name, str(default)) or str(default)).strip()
    try:
        return max(minimum, int(raw))
    except (TypeError, ValueError):
        return default


def _claim_extract_batch_size(default: int | None = None) -> int:
    if default is not None:
        return max(1, int(default))
    fallback = _read_int_env("VERIFIER_BATCH_SIZE", 4, minimum=1)
    return _read_int_env("VERIFIER_CLAIM_EXTRACT_BATCH_SIZE", fallback, minimum=1)


def _claim_extract_max_workers(default: int | None = None) -> int:
    if default is not None:
        return max(1, int(default))
    return _read_int_env("VERIFIER_CLAIM_EXTRACT_MAX_WORKERS", 4, minimum=1)


def _claim_extract_prompt_profile() -> str:
    raw = str(os.getenv("VERIFIER_CLAIM_EXTRACT_PROMPT_PROFILE", "auto") or "auto").strip().lower()
    if raw in {"gpt", "gpt_strict", "gpt-strict", "openai_strict", "openai-strict"}:
        return "gpt_strict"
    if raw in {"default", "base", "common", "none", "off"}:
        return "default"

    model = (
        os.getenv("VERIFIER_CLAIM_EXTRACT_MODEL", "")
        or os.getenv("VERIFIER_MODEL", "")
        or ""
    ).strip().lower()
    if model.startswith(("gpt", "o1", "o3")):
        return "gpt_strict"
    return "default"


def _model_specific_extract_rules() -> str:
    if _claim_extract_prompt_profile() != "gpt_strict":
        return ""
    return """
### GPT 계열 추가 규칙
GPT 계열 모델은 "빠뜨리지 말라"는 지시를 과하게 해석해 문장 조각이나 강의 진행 발언까지 claim으로 만들 수 있습니다.
따라서 아래 규칙을 우선 적용하세요.

1. 출력 기준
- claim은 학생이 그대로 외웠을 때 참/거짓을 검증할 수 있는 **완성 명제**여야 합니다.
- 단순히 강의자가 설명을 시작함, 예시를 들겠다고 함, 다음 내용을 예고함, 질문을 던짐, 강의 운영을 안내함은 claim이 아닙니다.
- 술어가 끝나지 않은 문장 조각이나 목적어/수식어만 있는 조각은 단독 claim으로 출력하지 마세요.

2. 전수 확인의 의미
- 검사 대상 context마다 claim 후보가 있는지 확인하라는 뜻이지, 모든 검사 대상 context를 claim으로 만들라는 뜻이 아닙니다.
- 검증 가능한 완성 명제가 없는 검사 대상 context는 과감히 출력하지 마세요.
- 강의 일정, 연락 방법, 수업 준비물, 공지 확인처럼 강의 운영에 관한 발언은 수치나 정책 자체가 핵심 검증 대상일 때만 추출하세요.

3. 문맥 결합
- 문맥은 지시어를 보수적으로 해소하고 현재 context의 의미 범위를 확인하기 위한 보조 정보입니다.
- 문맥을 이용해 현재 context가 실제로 말하지 않은 주체, 조건, 인과, 일반 법칙을 새로 만들지 마세요.
- 서로 다른 context를 하나의 claim으로 자동 병합하지 마세요.

4. resolved_claim 작성
- resolved_claim은 원문보다 넓어지면 안 됩니다.
- resolved_claim은 원문을 정답처럼 교정하는 필드가 아닙니다.
- 원문에 명시된 용어, 분류명, 주체, 대상이 틀린 것처럼 보여도 고치지 말고 그대로 보존하세요.
- 하나의 원문에 서로 다른 명제가 있으면 하나로 요약하지 말고 분리하세요.
- 원문에 있는 중요한 부정/한정 표현을 덮어쓰지 말고, 서로 다른 핵심 명제는 별도 claim으로 분리하세요.
"""


def _build_slide_references(slide_numbers: list[int], slide_ctx: dict) -> str:
    refs = []
    mode = _claim_extract_context_mode()
    for sn in slide_numbers:
        ctx = slide_ctx.get(sn, {})
        title = ctx.get("title", f"슬라이드 {sn}")
        time_range = ctx.get("time_range", "")
        slide_text = str(ctx.get("slide_text", "") or "")
        limit = 900 if mode in {"compact", "card_lite"} else 1200
        if len(slide_text) > limit:
            slide_text = slide_text[:limit] + "\n... (이하 생략)"
        refs.append(
            f"[슬라이드 {sn}] {title} ({time_range})\n"
            + (slide_text if slide_text else "(슬라이드 텍스트 없음)")
        )
    return "\n\n".join(refs)


def _build_slide_and_context_blocks(
    contexts: list[dict],
    slide_ctx: dict,
    target_context_ids: set[str] | None = None,
) -> tuple[str, str]:
    """슬라이드 참조와 context 문맥을 한 번의 순회로 생성."""
    from . import claim_common as cv

    total = len(contexts)
    seen_slides = OrderedDict()
    cards = []
    mode = _claim_extract_context_mode()

    if mode == "compact":
        transcript_lines = []
        for u in contexts:
            current_sn = int(u.get("slide_number", 0) or 0)
            if current_sn not in seen_slides:
                seen_slides[current_sn] = True
            transcript_lines.append(f"  {cv._format_context_for_prompt(u)}")
        context = (
            "[배치 context 전체]\n"
            + "\n".join(transcript_lines)
            + "\n\n[문맥 사용 규칙]\n"
            + "- 각 claim은 반드시 위 context의 context_id에 귀속하세요.\n"
            + "- 앞뒤 문맥은 위 context 순서 안에서 참고하세요.\n"
            + "- 지시어 선행사가 이 배치 밖에 있어 확정되지 않으면 원문 claim을 유지하고 resolved_claim도 claim_text와 동일하게 두세요."
        )
        return _build_slide_references(list(seen_slides.keys()), slide_ctx), context

    if mode == "card_lite":
        target_ids = target_context_ids or {str(u.get("context_id") or "") for u in contexts}
        transcript_lines = []
        target_lines = []
        for u in contexts:
            current_sn = int(u.get("slide_number", 0) or 0)
            if current_sn not in seen_slides:
                seen_slides[current_sn] = True
            cid = str(u.get("context_id") or "")
            marker = "*" if cid in target_ids else " "
            line = f"{marker} {cv._format_context_for_prompt(u)}"
            transcript_lines.append(line)
            if cid in target_ids:
                target_lines.append(f"- {cid} (슬라이드 {current_sn})")
        context = (
            "[배치 전체 전사]\n"
            + "\n".join(transcript_lines)
            + "\n\n[검사 대상 context]\n"
            + "\n".join(target_lines)
            + "\n\n[card-lite 추출 규칙]\n"
            + "- '*' 표시된 검사 대상 context는 각각 독립적으로 검토하세요.\n"
            + "- claim이 있는 검사 대상 context는 빠뜨리지 말고 claims 배열에 출력하세요.\n"
            + "- claim이 없는 검사 대상 context는 출력하지 않아도 됩니다.\n"
            + "- '*'가 없는 context는 앞뒤 문맥과 지시어 해소에만 사용하세요.\n"
            + "- '*'가 없는 context를 대표 context_id로 새 claim을 만들지 마세요.\n"
            + "- 검사 대상 context의 claim을 주변 context와 합쳐 새로운 claim으로 만들지 마세요.\n"
            + "- 긴 배치 전체에서 중요한 것만 선별하지 말고, 검사 대상 context마다 검증 가능한 정의/수치/인과/관계/현재성 주장이 있는지 전수 확인하세요."
        )
        return _build_slide_references(list(seen_slides.keys()), slide_ctx), context

    if target_context_ids:
        target_lines = []
        reference_lines = []
        for u in contexts:
            current_sn = int(u.get("slide_number", 0) or 0)
            if current_sn not in seen_slides:
                seen_slides[current_sn] = True
            cid = str(u.get("context_id") or "")
            line = f"  {cv._format_context_for_prompt(u)}"
            if cid in target_context_ids:
                target_lines.append(line)
            else:
                reference_lines.append(line)

        context = (
            "[검사 대상 context]\n"
            + ("\n".join(target_lines) if target_lines else "(없음)")
        )
        if reference_lines:
            context += (
                "\n\n[참고 context - 지시어 해소/문장 완결용]\n"
                + "\n".join(reference_lines)
            )
        context += (
            "\n\n[context-window 추출 규칙]\n"
            "- claim은 반드시 검사 대상 context에서 직접 말한 내용만 추출하세요.\n"
            "- 참고 context는 지시어 선행사 해소와 생략된 주어 확인에만 사용하세요.\n"
            "- 참고 context에만 있는 새 claim을 만들지 마세요.\n"
            "- 지시어가 단일 후보로 확실히 해소되면 resolved_claim에 최소한으로 반영하세요.\n"
            "- 후보가 둘 이상 가능하거나 검사 대상/참고 context/슬라이드 안에서 확정되지 않으면 claim_text를 보존하고 resolved_claim도 claim_text와 동일하게 두세요."
        )
        return _build_slide_references(list(seen_slides.keys()), slide_ctx), context

    for i, u in enumerate(contexts):
        current_sn = int(u.get("slide_number", 0) or 0)
        if current_sn not in seen_slides:
            seen_slides[current_sn] = True

        prev_context = contexts[max(0, i - 3):i]
        next_context = contexts[i + 1:min(total, i + 4)]
        same_slide_prev = [x for x in contexts[max(0, i - 6):i] if int(x.get("slide_number", 0) or 0) == current_sn]
        same_slide_next = [x for x in contexts[i + 1:min(total, i + 7)] if int(x.get("slide_number", 0) or 0) == current_sn]

        parts = [
            f"### 대상 context {u.get('context_id', '?')} (슬라이드 {current_sn})",
            f"[현재 context]\n{cv._format_context_for_prompt(u)}",
        ]
        if prev_context:
            parts.append("[직전 문맥]")
            parts.extend(f"  {cv._format_context_for_prompt(x)}" for x in prev_context)
        if next_context:
            parts.append("[직후 문맥]")
            parts.extend(f"  {cv._format_context_for_prompt(x)}" for x in next_context)
        if same_slide_prev or same_slide_next:
            parts.append("[같은 슬라이드 추가 문맥]")
            for x in same_slide_prev[-3:]:
                parts.append(f"  {cv._format_context_for_prompt(x)}")
            for x in same_slide_next[:3]:
                parts.append(f"  {cv._format_context_for_prompt(x)}")
        cards.append("\n".join(parts))

    return _build_slide_references(list(seen_slides.keys()), slide_ctx), "\n\n".join(cards)


def _build_extract_prompt(
    contexts: list[dict],
    current_date: str,
    hint: dict,
    slide_ctx: dict,
    target_context_ids: set[str] | None = None,
) -> str:
    slide_references, context_cards = _build_slide_and_context_blocks(
        contexts,
        slide_ctx,
        target_context_ids=target_context_ids,
    )

    return f"""당신은 강의 context에서 검증 가능한 사실 주장(claim)의 원문 목록을 추출하는 전문가입니다.
오늘 날짜: {current_date}
강의 도메인: {hint['label']}

아래는 슬라이드 참조 정보와 context 문맥입니다.
학생은 슬라이드와 강의 context를 동시에 받습니다.

[슬라이드 참조]
{slide_references}

[context 문맥]
{context_cards}

### 추출 대상
- definition: 정의, 의미, 용어, 기호, 개념 설명
- numeric: 수치, 단위, 기준값, 정량적 비교
- causal: 원인, 결과, 메커니즘, 작동 방식
- relationship: 분류, 포함, 비교, 상하위, 대응 관계
- currentness: 현재 시점의 유효성, 현행성, 주류성, 최신성

### 추출 제외
- 의견/감상, 교육적 지시, 구어적 필러
- 단순한 질문 제시만 있고 강의자가 답이나 기준을 제시하지 않은 경우
- "약/대략/정도" 붙은 수치는 claim_text/resolved_claim에 그 근사 표현을 그대로 남김

### 핵심 원칙: 1단계는 raw claim inventory입니다.
- 이 단계에서는 오류 여부, 오해 가능성, 교수 피드백, 반례를 판단하지 마세요.
- 반드시 **현재 context 자체가 명시한 주장**만 claim으로 추출하세요.
- claim_text는 현재 context에서 직접 가져온 원문 조각으로 쓰세요.
- claim_text를 만들 때, 원문에서 나온 주어, 예시, 설명들을 임의로 제거하거나 수정하지 마세요.
- resolved_claim은 원문 claim의 범위를 보존한 정리문입니다.
- resolved_claim은 지시어, 생략 주어, 담화 표지, 반복 표현을 문맥상 확실한 범위 안에서 풀어 검증 가능한 완성 명제로 만드는 필드입니다.
- 열거 대상과 지시어가 같은 context 안에서 명확히 연결될 때만, 그 범위 안에서 resolved_claim을 완성문으로 정리하세요.
- resolved_claim은 전사 오류, 용어 오류, 분류 오류, 사실 오류를 교정하는 필드가 아닙니다.
- 원문에 명시된 용어/분류명/주체/대상이 일반 지식이나 주변 문맥과 다르게 보이더라도 resolved_claim에서 고치지 마세요.
- resolved_claim에서 새로운 주체, 조건, 원인, 반례, 일반 법칙을 만들지 마세요.
- resolved_claim이 원문 주어, 대상, 분류명, 조건, 범위를 실제로 교정하거나 바꿀 위험이 있으면 claim_text와 동일하게 두세요.
- 단, "이런 것들", "그것", 생략 주어를 문맥에서 바로 확인되는 명시 대상명으로 바꾸는 것은 교정이 아니라 지시어 해소입니다.
- 주변 문맥과 슬라이드는 현재 context가 claim인지, 예시인지, 지시어가 명확한지만 판단하는 보조 정보입니다.
- 주변 문맥에 있는 더 강한 일반 명제를 현재 context에 덧씌우지 마세요.
- 현재 context가 예시/가정/비유/수사적 요약이면, resolved_claim에도 그 예시/가정/비유/요약 범위를 유지하세요.

### 지시어 처리
- "이것", "여기", "해당 항목", "얘", "이거", "그거" 같은 지시어는 단일 선행사가 확실할 때만 최소한으로 풀어 쓰세요.
- "이런 것들", "이것들", "얘네들"처럼 같은 context 안에서 바로 앞에 열거된 대상 전체를 가리키는 표현은, 열거 대상이 명확하면 resolved_claim에서 그 대상명으로 풀어 쓰세요.
- 둘 이상의 합리적 해석이 가능하면 특정 대상으로 확정하지 말고 원문 지시어를 유지하세요.
- 지시어 선행사가 제공된 문맥보다 앞에 있을 수 있어도, 현재 context에 검증 가능한 술어/정의/수치/관계가 있으면
  claim을 버리지 말고 원문 그대로 추출하세요. 이때 resolved_claim은 claim_text와 같게 두세요.
- 슬라이드나 앞뒤 context가 보이더라도, 현재 context가 실제로 말하지 않은 관계를 resolved_claim에 추가하지 마세요.
- 불완전한 조각 문장은 현재 context 안에서 검증 가능한 값/정의/관계가 완성되지 않으면 추출하지 마세요.
- 복원 결과가 애매하다는 이유만으로 "이것은 X이다", "얘는 Y로 처리된다" 같은 원문 claim을 버리지 마세요.
- 단, 현재 context에 검증 가능한 술어가 없고 지시어만 남은 경우는 추출하지 마세요.

### 복수 claim 추출
하나의 context에 여러 주장이 섞여 있으면 반드시 각각 별도 claim으로 추출하세요.
특히 "요즘/현재/최근/추세/주류" 표현은 별도 currentness claim으로 추출하세요.
하나의 context 안에 수치가 2개 이상 나오면, 각각이 독립적으로 검증 가능한 값인지 확인하고 가능한 한 분리하세요.
문장이 불완전해 보여도 직전 context와 결합하면 검증 가능한 수치/기준 claim이 되면 추출하세요.

### context 단위 원칙
- 입력 context는 이미 전사 발화를 의미 단위로 묶은 검증 단위입니다.
- 서로 다른 context를 이어 붙여 하나의 claim으로 만들지 마세요.
- "X는", "X라는 것은", "A를", "B하기 위해서", "희생한"처럼 현재 context 안에서 술어가 끝나지 않은 조각은 단독 claim으로 출력하지 마세요.

주의: 말실수나 용어 착각으로 보이더라도, 학생이 그대로 믿으면 틀린 지식이 되는 경우는 추출 대상입니다.
주의: 강의자가 예시 상황 안에서 기준값이나 정량적 조건을 말하면, 그 예시 범위 안의 검증 가능한 claim으로 추출하세요.
주의: 문장이 질문형으로 시작하더라도, 뒤에서 강의자가 특정 값이나 기준을 제시하면 그 제시된 값/기준은 claim으로 추출하세요.
주의: 다만 예시 속 기준값을 일반 상식/보편 법칙으로 확대 해석하지 마세요. 예시의 범위가 드러나면 resolved_claim에도 그 예시 범위를 남기세요.

{_model_specific_extract_rules()}

### 출력 (JSON만)
```json
{{
  "claims": [
    {{
      "context_id": "S001-SC0001-C001",
      "claim_type": "definition",
      "claim_text": "현재 context에서 직접 가져온 claim 원문",
      "resolved_claim": "원문 범위를 보존한 최소 정리문"
    }}
  ]
}}
```

지침:
- 검증 불가능한 주장은 추출하지 마세요.
- 하나의 context에서 여러 claim이 나올 수 있습니다.
- claim_type은 반드시 `definition`, `numeric`, `causal`, `relationship`, `currentness` 중 하나만 사용하세요.
- verification_question은 생성하지 마세요. 검증 질문은 후속 판정 단계에서 필요한 claim에만 만듭니다.
- resolved_claim을 쓰기 애매하면 claim_text와 동일하게 두세요.
- claim이 없으면 {{"claims": []}}만 출력하세요.
- JSON 외 텍스트를 출력하지 마세요.
"""


def _normalize_claim_type(value: str) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return "definition"

    lowered = raw.lower()
    korean = raw.strip()

    if lowered == "no_claim":
        return None

    if lowered in _CANONICAL_CLAIM_TYPES:
        return lowered

    if any(token in lowered for token in ("current", "outdated", "deprecated", "trend", "recent")):
        return "currentness"

    if any(token in lowered for token in ("causal", "cause", "effect", "mechanism")):
        return "causal"

    if (
        any(token in lowered for token in (
            "numeric", "numerical", "number", "quantity", "quantitative",
            "specification", "standard", "example_numeric", "standard/quantity",
        ))
        or korean in {"수치", "정량/기준값", "실무 예시 속 사실 주장"}
    ):
        return "numeric"

    if any(token in lowered for token in ("relation", "relationship", "relational", "comparison", "compare")) or korean == "관계":
        return "relationship"

    return "definition"


def assign_claim_display_ids(claims_by_batch: list[tuple]) -> None:
    """최종 추출 순서 기준으로 사람이 읽는 claim_id를 부여한다."""
    sequence = 1
    for _batch, claims in claims_by_batch:
        for claim in claims:
            claim["claim_id"] = f"CL{sequence:04d}"
            _order_claim_fields(claim)
            sequence += 1


def _order_claim_fields(claim: dict) -> None:
    preferred_keys = (
        "claim_id",
        "context_id",
        "claim_text",
        "resolved_claim",
        "claim_type",
    )
    ordered = {key: claim[key] for key in preferred_keys if key in claim}
    ordered.update({key: value for key, value in claim.items() if key not in ordered})
    claim.clear()
    claim.update(ordered)


def _text_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"\s+", str(value or "").strip()) if token}


def _similar_claim_text(a: dict, b: dict) -> bool:
    a_text = str(a.get("claim_text") or "")
    b_text = str(b.get("claim_text") or "")
    a_compact = re.sub(r"\s+", "", a_text)
    b_compact = re.sub(r"\s+", "", b_text)
    if not a_compact or not b_compact:
        return False
    if a_compact in b_compact or b_compact in a_compact:
        return True
    a_tokens = _text_tokens(a_text)
    b_tokens = _text_tokens(b_text)
    if not a_tokens or not b_tokens:
        return False
    return len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens))) >= 0.65


def _dedupe_overlapping_claims(claims: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    for claim in claims:
        claim_context_id = str(claim.get("context_id") or "")
        replaced = False
        for idx, existing in enumerate(deduped):
            if str(existing.get("claim_type") or "") != str(claim.get("claim_type") or ""):
                continue
            if str(existing.get("context_id") or "") != claim_context_id:
                continue
            if not _similar_claim_text(existing, claim):
                continue
            existing_text_len = len(str(existing.get("claim_text") or ""))
            claim_text_len = len(str(claim.get("claim_text") or ""))
            if claim_text_len > existing_text_len:
                deduped[idx] = claim
            replaced = True
            break
        if not replaced:
            deduped.append(claim)
    return deduped


def _context_batches_by_slide(contexts: list[dict]) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current_slide = None
    current_batch: list[dict] = []
    for context in contexts:
        slide_no = int(context.get("slide_number", 0) or 0)
        if current_batch and slide_no != current_slide:
            batches.append(current_batch)
            current_batch = []
        current_slide = slide_no
        current_batch.append(context)
    if current_batch:
        batches.append(current_batch)
    return batches


def _extract_claims(
    contexts: list[dict],
    current_date: str,
    hint: dict,
    slide_ctx: dict,
    target_context_ids: set[str] | None = None,
) -> tuple[list[dict], bool, int, dict]:
    from . import claim_common as cv

    if not contexts:
        return [], False, 0, cv._empty_token_usage()

    prompt = _build_extract_prompt(
        contexts,
        current_date,
        hint,
        slide_ctx,
        target_context_ids=target_context_ids,
    )
    api_calls = 0
    token_usage = cv._empty_token_usage()

    for attempt in range(cv.VERIFIER_PARSE_RETRIES + 1):
        text, call_usage = cv._call_llm(
            prompt,
            max_tokens=8192,
            temperature=0.0,
            thinking_budget=1024,
            stage="extract",
        )
        api_calls += 1
        cv._add_call_usage(token_usage, call_usage)
        try:
            payload = json.loads(cv._strip_json_fence(text.strip()))
            if isinstance(payload, list):
                claims = payload
            elif isinstance(payload, dict):
                claims = payload.get("claims", [])
            else:
                raise ValueError(f"unexpected JSON payload type: {type(payload).__name__}")
            if not isinstance(claims, list):
                raise ValueError(f"claims must be a list, got {type(claims).__name__}")
            cleaned = []
            for c in claims:
                if not isinstance(c, dict) or not c.get("context_id"):
                    continue
                claim_text = str(c.get("claim_text") or "").strip()
                if not claim_text:
                    continue
                resolved_claim = str(c.get("resolved_claim") or "").strip() or claim_text
                normalized = _normalize_claim_type(c.get("claim_type"))
                if normalized is None:
                    continue
                c["claim_type"] = normalized
                c["claim_text"] = claim_text
                c["resolved_claim"] = resolved_claim
                c.pop("claim_id", None)
                c["context_ids"] = [str(c.get("context_id") or "")]
                c.pop("resolution_status", None)
                c.pop("antecedent_context_ids", None)
                c.pop("claim_fingerprint", None)
                c.pop("is_approximate", None)
                c.pop("context_note", None)
                c.pop("verification_question", None)
                c.pop("verificationQuestion", None)
                cleaned.append(c)
            cleaned = _dedupe_overlapping_claims(cleaned)
            for claim in cleaned:
                allowed = {"context_id", "claim_text", "resolved_claim", "claim_type"}
                for key in list(claim.keys()):
                    if key not in allowed:
                        claim.pop(key, None)
            return cleaned, False, api_calls, token_usage
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            if attempt < cv.VERIFIER_PARSE_RETRIES:
                preview = " ".join(str(text or "").split())[:220]
                print(
                    f"    ↺ claim 추출 JSON 파싱 재시도 "
                    f"({attempt+1}/{cv.VERIFIER_PARSE_RETRIES}) — {type(e).__name__}: {e}"
                )
                if preview:
                    print(f"       응답 preview: {preview}")

    return [], True, api_calls, token_usage


def recover_claim_extraction(
    contexts: list[dict],
    current_date: str,
    hint: dict,
    slide_ctx: dict,
    label: str,
    target_context_ids: set[str] | None = None,
) -> tuple[list[dict], bool, int, dict, bool]:
    from . import claim_common as cv

    total_api_calls = 0
    total_token_usage = cv._empty_token_usage()
    last_claims: list[dict] = []
    parse_failed = False
    last_exc = None

    for attempt in range(cv.VERIFIER_BATCH_RECOVERY_RETRIES + 1):
        if attempt > 0:
            print(f"    ↺ {label} claim 추출 재처리 ({attempt}/{cv.VERIFIER_BATCH_RECOVERY_RETRIES})")
        try:
            claims, parse_failed, api_calls, token_usage = _extract_claims(
                contexts, current_date, hint, slide_ctx, target_context_ids=target_context_ids
            )
            total_api_calls += api_calls
            total_token_usage = cv._merge_token_usage(total_token_usage, token_usage)
            last_claims = claims
            if not parse_failed:
                return claims, False, total_api_calls, total_token_usage, True
        except Exception as e:
            last_exc = e

    split_items = _split_context_recovery_items(contexts, target_context_ids)
    if split_items:
        recovered_claims: list[dict] = []
        recovered_any = False
        print(f"    ↺ {label} claim 추출 분할 복구 ({len(split_items)}개 chunk)")
        for idx, (chunk_contexts, chunk_target_ids) in enumerate(split_items, start=1):
            chunk_label = f"{label} / split {idx}"
            claims, chunk_parse_failed, api_calls, token_usage, ok = recover_claim_extraction(
                chunk_contexts,
                current_date,
                hint,
                slide_ctx,
                chunk_label,
                target_context_ids=chunk_target_ids,
            )
            total_api_calls += api_calls
            total_token_usage = cv._merge_token_usage(total_token_usage, token_usage)
            if ok:
                recovered_any = True
            if claims:
                recovered_claims.extend(claims)
        if recovered_claims:
            recovered_claims = _dedupe_overlapping_claims(recovered_claims)
            return recovered_claims, not recovered_any, total_api_calls, total_token_usage, recovered_any

    if last_exc and not last_claims and total_api_calls == 0:
        raise last_exc
    return last_claims, True, total_api_calls, total_token_usage, False


def _split_context_recovery_items(
    contexts: list[dict],
    target_context_ids: set[str] | None,
) -> list[tuple[list[dict], set[str] | None]]:
    if len(contexts) <= 1:
        return []
    midpoint = max(1, len(contexts) // 2)
    chunks = [contexts[:midpoint], contexts[midpoint:]]
    items: list[tuple[list[dict], set[str] | None]] = []
    for chunk in chunks:
        if not chunk:
            continue
        if target_context_ids is None:
            chunk_target_ids = None
        else:
            chunk_ids = {str(context.get("context_id") or "") for context in chunk}
            chunk_target_ids = target_context_ids & chunk_ids
            if not chunk_target_ids:
                continue
        items.append((chunk, chunk_target_ids))
    return items


def extract_claims_only(
    contexts: list[dict],
    current_date: str,
    hint: dict,
    slide_ctx: dict,
    batch_size: int | None = None,
    max_workers: int | None = None,
) -> tuple[list[tuple], int, dict]:
    """1단계만 실행: claim 추출. (batch_list, total_claims) 반환."""
    from . import claim_common as cv

    batch_size = _claim_extract_batch_size(batch_size)
    max_workers = _claim_extract_max_workers(max_workers)

    all_claims_by_batch = []
    total_api = 0
    total_token_usage = cv._empty_token_usage()
    total = len(contexts)
    has_contexts = any(u.get("context_id") for u in contexts)
    context_window_prev, context_window_next = _claim_extract_context_window()
    if has_contexts:
        batch_mode = _claim_extract_batch_mode()
        if batch_mode == "slide":
            core_batches = _context_batches_by_slide(contexts)
        else:
            context_group_size = _claim_extract_context_group_size()
            core_batches = [
                contexts[i:min(total, i + context_group_size)]
                for i in range(0, total, context_group_size)
            ]
        print(
            f"  claim 추출 batch mode: {batch_mode}"
            + (
                f" (group_size={context_group_size}, prev={context_window_prev}, next={context_window_next})"
                if batch_mode == "context" else ""
            )
        )
    else:
        batch_mode = "context"
        core_batches = [
            contexts[i:min(total, i + batch_size)]
            for i in range(0, total, batch_size)
        ]
    context_mode = _claim_extract_context_mode()
    shared_context_mode = context_mode in {"compact", "card_lite"}

    cursor = 0
    work_items = []
    for i, core_batch in enumerate(core_batches):
        if not core_batch:
            continue
        start = cursor
        end = cursor + len(core_batch)
        cursor = end
        if has_contexts and batch_mode == "context":
            context_batch = contexts[
                max(0, start - context_window_prev):min(total, end + context_window_next)
            ]
            core_context_ids = {str(u.get("context_id") or "") for u in core_batch}
        elif shared_context_mode and not any(u.get("context_id") for u in core_batch):
            context_batch = contexts[max(0, start - 3):min(total, end + 5)]
            core_context_ids = {str(u.get("context_id") or "") for u in core_batch}
        else:
            context_batch = core_batch
            core_context_ids = None

        ids = f"{core_batch[0]['context_id']}..{core_batch[-1]['context_id']}"
        work_items.append({
            "index": i,
            "context_batch": context_batch,
            "core_context_ids": core_context_ids,
            "ids": ids,
        })

    def _run_item(item: dict) -> tuple[int, list[dict], list[dict], int, dict]:
        ids = item["ids"]
        print(f"    추출 [{item['index']+1}/{len(core_batches)}] {ids}", flush=True)
        claims, parse_failed, api_calls, token_usage, ok = recover_claim_extraction(
            item["context_batch"],
            current_date,
            hint,
            slide_ctx,
            f"배치 {item['index']+1} {ids}",
            target_context_ids=item["core_context_ids"],
        )
        if not ok:
            print(f"    ⚠️ claim 추출 실패: {ids} — 이 batch는 빈 결과로 기록됩니다.")
        if item["core_context_ids"] is not None:
            claims = [c for c in claims if str(c.get("context_id") or "") in item["core_context_ids"]]
        return item["index"], item["context_batch"], claims, api_calls, token_usage

    if work_items:
        worker_count = min(max_workers, len(work_items))
        print(
            f"  claim 추출 병렬 실행: batch_size={batch_size}, workers={worker_count}, batches={len(work_items)}",
            flush=True,
        )
    if len(work_items) <= 1 or max_workers <= 1:
        results = [_run_item(item) for item in work_items]
    else:
        results = [None] * len(work_items)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(work_items))) as executor:
            futures = {executor.submit(_run_item, item): item for item in work_items}
            for future in as_completed(futures):
                index, context_batch, claims, api_calls, token_usage = future.result()
                results[index] = (index, context_batch, claims, api_calls, token_usage)

    for _index, context_batch, claims, api_calls, token_usage in [r for r in results if r is not None]:
        all_claims_by_batch.append((context_batch, claims))
        total_api += api_calls
        total_token_usage = cv._merge_token_usage(total_token_usage, token_usage)

    total_claims = sum(len(c) for _, c in all_claims_by_batch)
    assign_claim_display_ids(all_claims_by_batch)
    print(f"  추출된 claim: {total_claims}개")
    return all_claims_by_batch, total_api, total_token_usage
