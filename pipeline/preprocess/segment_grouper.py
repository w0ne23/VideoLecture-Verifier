"""
전사 세그먼트를 맥락/문장·문단 단위로 묶기

1차: 시간/문장부호 기반으로 "호흡 단위" 그룹 생성
2차: LLM(Gemini)을 이용해 인접 그룹의 의미/주제가 이어지면 다시 병합

- 쉼/침묵: 인접 세그먼트 간 간격이 짧으면 같은 호흡으로 묶음
- 문장 끝: 마침표·물음표·느낌표 등에서 문장 경계로 분리
- 문단: 긴 침묵에서 문단 경계로 분리
- 최대 길이: 한 묶음이 너무 길어지지 않도록 상한
- 의미 기반 병합: LLM이 "같은 주제/맥락의 설명"이라고 판단하면 인접 그룹 병합
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.genai import types

from .config import GEMINI_GENERATIVE_MODEL, gemini_client_2, get_openai_client
from .text_processor import _is_ollama_model, _resolve_ollama_model
from .utils import api_call_with_retry


# 기존: 아래 세 LLM 호출(decide_semantic_merges/decide_context_breaks/decide_merge_pair)은
# GEMINI_GENERATIVE_MODEL(gemini-2.5-flash)로 하드코딩되어 있었음.
# 상용 API로 되돌리려면: GRAPHLEC_CONTEXT_GROUP_MODEL=gemini-2.5-flash (또는 GEMINI_GENERATIVE_MODEL)
CONTEXT_GROUP_MODEL = os.getenv("GRAPHLEC_CONTEXT_GROUP_MODEL", "ollama:qwen3.8:27b-q4_K_M").strip()

CONTEXT_PARALLEL_REQUESTS = max(
    1,
    int(os.getenv("VLVERIFIER_CONTEXT_PARALLEL_REQUESTS", "20")),
)
# Ollama로 라우팅되면 서버 동시 처리 한도에 맞춰 상한을 낮춘다 (text_processor._effective_parallel_requests와 동일 패턴).
CONTEXT_OLLAMA_PARALLEL_REQUESTS = max(
    1,
    int(os.getenv("MERGE_CORRECTION_OLLAMA_PARALLEL_REQUESTS", "4")),
)


def _effective_context_parallel_requests() -> int:
    if _is_ollama_model(CONTEXT_GROUP_MODEL):
        return min(CONTEXT_PARALLEL_REQUESTS, CONTEXT_OLLAMA_PARALLEL_REQUESTS)
    return CONTEXT_PARALLEL_REQUESTS


def _call_context_llm(prompt: str, *, max_output_tokens: int, json_mode: bool = True) -> str:
    """설정된 모델에 따라 Ollama, OpenAI 호환 API 또는 Gemini로 호출한다."""
    if _is_ollama_model(CONTEXT_GROUP_MODEL):
        from .config import get_ollama_client

        client = get_ollama_client()
        if client is None:
            raise RuntimeError("Ollama 클라이언트를 만들 수 없습니다 (openai 패키지 확인 필요).")
        resolved_model, think_override = _resolve_ollama_model(CONTEXT_GROUP_MODEL)

        def call_api():
            kwargs = {
                "model": resolved_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_completion_tokens": max_output_tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if think_override is not None:
                kwargs["extra_body"] = {"think": think_override}
            return client.chat.completions.create(**kwargs)

        response = api_call_with_retry(call_api)
        return (response.choices[0].message.content or "").strip()

    if CONTEXT_GROUP_MODEL.lower().startswith(("gpt-", "openai:")):
        client = get_openai_client()
        if client is None:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
        model = CONTEXT_GROUP_MODEL.removeprefix("openai:")

        def call_api():
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_output_tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kwargs)

        response = api_call_with_retry(call_api)
        return (response.choices[0].message.content or "").strip()

    def call_api():
        return gemini_client_2.models.generate_content(
            model=CONTEXT_GROUP_MODEL or GEMINI_GENERATIVE_MODEL,
            contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=max_output_tokens,
            ),
        )

    response = api_call_with_retry(call_api)
    return (response.text or "").strip()

CONTEXT_SOFT_MAX_SEGMENTS = max(
    1,
    int(os.getenv("VLVERIFIER_CONTEXT_SOFT_MAX_SEGMENTS", "20")),
)
CONTEXT_HARD_MAX_SEGMENTS = max(
    CONTEXT_SOFT_MAX_SEGMENTS,
    int(os.getenv("VLVERIFIER_CONTEXT_HARD_MAX_SEGMENTS", "20")),
)
CONTEXT_HARD_LOOKAHEAD_SEGMENTS = max(
    0,
    int(os.getenv("VLVERIFIER_CONTEXT_HARD_LOOKAHEAD_SEGMENTS", "0")),
)


# 문장 종결로 볼 문장부호 (한국어·영어·공백 제거 후 끝)
SENTENCE_END_PATTERN = re.compile(r".*[.!?\u3002\uFF01\uFF1F]\s*$")

# 그룹 상한: 이 시간(초)을 넘으면 새 그룹 시작
MAX_GROUP_DURATION = 30.0
# 이 간격(초) 이상이면 새 그룹 (문단/호흡 끊김)
PAUSE_PARAGRAPH = 1.2
# 이 간격(초) 이상이면 문장 경계 후보 (같은 문장이 아닐 가능성)
PAUSE_SENTENCE = 0.6
# 문장 끝 부호가 있으면 이 간격 이하여도 새 문장으로 분리 가능
PAUSE_AFTER_SENTENCE_END = 0.4


def _build_group_list_for_prompt(groups: list[dict], max_chars: int = 220) -> str:
    """LLM 프롬프트용 그룹 요약 문자열 생성"""
    lines = []
    for i, g in enumerate(groups):
        text = (g.get("text") or "").replace("\n", " ").strip()
        if len(text) > max_chars:
            text = text[: max_chars].rstrip() + "..."
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)


def decide_semantic_merges(groups: list[dict]) -> set[int]:
    """
    인접 그룹이 같은 주제/맥락인지 LLM으로 판단하여
    병합할 인덱스(i, i+1 병합)를 반환.
    """
    if len(groups) <= 1:
        return set()

    group_list = _build_group_list_for_prompt(groups)

    prompt = f"""당신은 대학 강의 전사를 분석하는 도우미입니다.
아래는 시간과 문장부호 기준으로 1차로 나눈 '맥락 그룹' 목록입니다.
각 그룹은 동일한 강의의 연속된 구간입니다.

목표: 인접한 두 그룹이 사실상 같은 주제/맥락의 설명이라면 하나로 다시 합치고 싶습니다.

특히, 다음과 같은 경우는 합쳐야 합니다.
- 같은 개념을 계속 설명하는데 시간/문장부호 때문에 잘린 경우
- 어떤 개념을 설명한 뒤 바로 그 예시/응용을 설명하는 경우

다음과 같은 경우는 합치지 마세요.
- 전혀 다른 개념/챕터/문단으로 넘어가는 경우
- 문제 풀이에서 완전히 다른 문제로 넘어가는 경우

아래 형식의 목록을 보고, i번째 그룹과 i+1번째 그룹을 합칠지 여부만 판단하세요.

그룹 목록:
{group_list}

출력은 다음 JSON 형식으로만 작성하세요:

```json
{{ "merge_after": [0, 1, 5] }}
```

여기서 merge_after의 각 숫자 i는 "그룹 i와 그룹 i+1을 병합하라"는 의미입니다.
설명은 쓰지 말고 JSON만 출력하세요.
"""

    try:
        text = _call_context_llm(prompt, max_output_tokens=2048)

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        data = json.loads(text)
        merge_after = data.get("merge_after", [])
        return {int(i) for i in merge_after if 0 <= int(i) < len(groups) - 1}
    except Exception:
        # 실패 시 의미 기반 병합 없이 진행
        return set()


def decide_context_breaks(groups: list[dict]) -> set[int]:
    """
    같은 scene 안의 연속 segment들을 의미/맥락 단위 context로 나누기 위해
    context가 끝나는 지점(i, i+1 사이 break)을 LLM으로 판단한다.

    기본 전제는 "인접 segment는 이어지는 강의 흐름"이며, LLM은 끊을 지점만 고른다.
    """
    if len(groups) <= 1:
        return set()

    group_list = _build_group_list_for_prompt(groups)

    prompt = f"""당신은 대학 강의 전사를 의미/맥락 단위 context로 나누는 도우미입니다.
아래는 같은 scene/slide 안에서 시간 순서대로 이어지는 발화 segment 목록입니다.

목표: segment들을 강의 설명의 의미 흐름이 유지되는 context 단위로 나누세요.

중요 원칙:
- 기본적으로 인접 segment는 같은 설명 흐름으로 보고 이어 붙입니다.
- scene 밖으로 넘어가는 병합은 이미 금지되어 있으므로, 아래 목록 내부에서만 판단하세요.
- 새 개념, 새 소주제, 새 설명 단계로 명확히 넘어가는 지점에서만 끊으세요.
- 수업 운영 멘트, 감사 인사, 녹음/마이크 안내, 쉬는 시간 안내, 다음 장/다음 주제로 넘어간다는 전환 멘트는
  본 설명 context와 섞지 말고 별도 context가 되도록 앞뒤를 끊으세요.

끊지 마세요:
- 같은 개념을 계속 설명하는 경우
- 앞 segment의 보충, 재진술, 원인/결과, 예시, 비교 설명인 경우
- "그러니까", "즉", "예를 들어", "이러한", "그런데", "그리고"처럼 이어지는 설명인 경우

segment 목록:
{group_list}

출력은 다음 JSON 형식으로만 작성하세요:

```json
{{ "break_after": [3, 8, 14] }}
```

break_after의 각 숫자 i는 "segment i까지 현재 context로 묶고, segment i+1부터 새 context를 시작하라"는 의미입니다.
끊을 지점이 없으면 빈 배열을 반환하세요. 설명은 쓰지 말고 JSON만 출력하세요.
"""

    try:
        text = _call_context_llm(prompt, max_output_tokens=2048)

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        data = json.loads(text)
        break_after = data.get("break_after", [])
        return {int(i) for i in break_after if 0 <= int(i) < len(groups) - 1}
    except Exception:
        # 실패 시 scene 내부 발화 흐름을 최대한 보존한다.
        return set()


_OPERATIONAL_CONTEXT_RE = re.compile(
    r"("
    r"감사합니다|수고하셨|고생하셨|"
    r"출석|쉬는\s*시간|잠깐\s*쉬|휴식|"
    r"녹음|마이크|소리\s*들리|"
    r"다음\s*영상|구독|좋아요|"
    r"질문\s*있|마치겠습니다|끝내겠습니다|"
    r"다음으로\s*(넘어|가|보)|이제\s*다음|넘어가겠습니다"
    r")"
)


def _is_operational_or_transition_segment(text: str) -> bool:
    """
    수업 운영/감사/짧은 전환 멘트를 본 설명 context와 섞지 않기 위한 보조 규칙.
    긴 내용 설명 안에 우연히 포함된 표현은 과도하게 분리하지 않도록 짧은 segment에만 적용한다.
    """
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return False
    return len(t) <= 90 and bool(_OPERATIONAL_CONTEXT_RE.search(t))


def _is_natural_context_boundary(groups: list[dict], boundary: int) -> bool:
    """세그먼트 경계가 문장/호흡상 안전한 분할 지점인지 판단한다."""
    if boundary < 0 or boundary >= len(groups) - 1:
        return False
    left = groups[boundary]
    right = groups[boundary + 1]
    left_text = (left.get("text") or "").strip()
    try:
        gap = float(right.get("start", 0.0)) - float(left.get("end", 0.0))
    except (TypeError, ValueError):
        gap = 0.0
    return bool(SENTENCE_END_PATTERN.match(left_text)) or gap >= PAUSE_SENTENCE


def _apply_context_segment_limit(
    groups: list[dict],
    llm_breaks: set[int],
) -> set[int]:
    """긴 context를 안전한 경계에서만 제한한다.

    soft limit에 도달하면 자연 경계를 찾고, hard limit에 도달한 뒤에도
    경계가 없으면 짧은 lookahead 후 세그먼트 경계에서 비상 분할한다.
    세그먼트 내부 텍스트는 절대 자르지 않는다.
    """
    if len(groups) <= CONTEXT_SOFT_MAX_SEGMENTS:
        return set(llm_breaks)

    semantic_breaks = {
        i for i in llm_breaks if 0 <= i < len(groups) - 1
    }
    natural_breaks = {
        i for i in range(len(groups) - 1)
        if _is_natural_context_boundary(groups, i)
    }
    safe_breaks = semantic_breaks | natural_breaks
    enforced = set(semantic_breaks)
    context_start = 0
    pending_break: int | None = None

    for boundary in range(len(groups) - 1):
        if boundary in semantic_breaks:
            enforced.add(boundary)
            context_start = boundary + 1
            pending_break = None
            continue

        if pending_break is not None:
            if boundary == pending_break:
                enforced.add(boundary)
                context_start = boundary + 1
                pending_break = None
            continue

        current_count = boundary - context_start + 1
        if current_count < CONTEXT_SOFT_MAX_SEGMENTS:
            continue

        if boundary in natural_breaks:
            enforced.add(boundary)
            context_start = boundary + 1
            continue

        if current_count >= CONTEXT_HARD_MAX_SEGMENTS:
            lookahead_end = min(
                len(groups) - 2,
                boundary + CONTEXT_HARD_LOOKAHEAD_SEGMENTS,
            )
            candidate = next(
                (i for i in range(boundary + 1, lookahead_end + 1) if i in safe_breaks),
                None,
            )
            if candidate is not None:
                pending_break = candidate
            else:
                # 세그먼트 내부가 아니라 세그먼트 사이에서만 비상 분할한다.
                enforced.add(boundary)
                context_start = boundary + 1

    return enforced


def decide_merge_pair(current_text: str, next_text: str, max_chars: int = 400) -> bool:
    """
    두 인접 구간(current_text, next_text)이 같은 맥락/설명의 연속인지 LLM으로 판단.
    - True: 같은 맥락으로 보고 병합
    - False: 다른 맥락으로 보고 분리
    """
    cur = (current_text or "").replace("\n", " ").strip()
    nxt = (next_text or "").replace("\n", " ").strip()
    if not cur or not nxt:
        return False
    if len(cur) > max_chars:
        cur = cur[:max_chars].rstrip() + "..."
    if len(nxt) > max_chars:
        nxt = nxt[:max_chars].rstrip() + "..."

    prompt = f"""당신은 대학 강의 전사를 분석하는 도우미입니다.

아래에는 같은 슬라이드 안에서 연속된 두 구간의 전사 텍스트가 주어집니다.

[이전 구간] (이미 하나의 맥락으로 묶여 있는 설명 전체)
---
{cur}
---

[다음 구간] (바로 이어지는 설명)
---
{nxt}
---

판단 기준:

1. 다음과 같은 경우라면 "같은 맥락의 설명"으로 보고 합쳐야 합니다.
   - 이전 구간에서 소개한 개념/정의/기능에 대한 설명이 다음 구간에서 계속 이어지는 경우
   - 다음 구간이 이전 구간의 내용을 다시 말하거나 정리하는 경우
     (예: "다시 말해서", "다시 말하면", "정리하면", "즉", "한마디로" 등으로 시작)
   - 다음 구간이 이전 구간에서 소개한 개념의 예시/응용/비유를 설명하는 경우
     (예: "예를 들어", "예를 들면", "하나의 예로" 등)
   - 이전 구간이 "~에 대해 알아보겠습니다", "~의 정의는" 처럼 앞으로 설명할 것을 예고하고,
     다음 구간이 그 실제 설명인 경우

2. 다음과 같은 경우라면 "다른 맥락"으로 보고 합치지 말아야 합니다.
   - 이전 구간과 완전히 다른 개념/챕터/슬라이드의 내용을 시작하는 경우
   - 강의의 주요 주제(예: 운영체제, 프로세스, 메모리, 파일, 자원, 사용자 등)와 관계없는
     짧은 잡담, 수업 운영 멘트(출석, 쉬는 시간 안내, 농담 등)만으로 이루어진 경우
   - 이전 구간의 설명이 사실상 끝났고, 새로운 주제를 소개하는 경우
     (예: "다음으로", "이제 다른 주제로", "이제 ~를 보겠습니다" 등)

출력 형식:
- 두 구간이 같은 맥락의 하나의 설명으로 이어져 있다고 판단되면
  MERGE
  라고만 출력하세요.
- 서로 다른 맥락이라고 판단되면
  SPLIT
  라고만 출력하세요.

추가 설명이나 다른 문장은 쓰지 말고, "MERGE" 또는 "SPLIT" 중 하나만 출력하세요."""

    try:
        text = _call_context_llm(prompt, max_output_tokens=256, json_mode=False).upper()
        if "MERGE" in text and "SPLIT" not in text:
            return True
        if "SPLIT" in text and "MERGE" not in text:
            return False
        # 애매하면 보수적으로 분리
        return False
    except Exception:
        # 실패 시 보수적으로 분리
        return False


def group_segments_by_context(
    segments: list[dict],
    *,
    max_group_duration: float = MAX_GROUP_DURATION,
    pause_paragraph: float = PAUSE_PARAGRAPH,
    pause_sentence: float = PAUSE_SENTENCE,
    pause_after_sentence_end: float = PAUSE_AFTER_SENTENCE_END,
    use_semantic_merge: bool = True,
) -> list[dict]:
    """
    인접 세그먼트를 맥락/문장·문단 단위로 묶는다.

    각 세그먼트는 { 'start', 'end', 'text' } 형태.
    반환되는 각 그룹은:
      - start, end: 구간 시간
      - text: 묶인 텍스트 (공백으로 연결)
      - segment_indices: 원본 segments 인덱스 리스트
    """
    if not segments:
        return []

    # 1차: 시간/문장부호 기반 그룹화
    groups: list[dict] = []
    current_indices = [0]
    current_start = segments[0]["start"]
    current_end = segments[0]["end"]
    current_texts = [segments[0]["text"].strip()]

    for i in range(1, len(segments)):
        seg = segments[i]
        prev_end = segments[i - 1]["end"]
        gap = seg["start"] - prev_end
        prev_text = (segments[i - 1]["text"] or "").strip()
        prev_ends_sentence = bool(SENTENCE_END_PATTERN.match(prev_text))

        # 문단 수준 침묵 -> 무조건 새 그룹
        if gap >= pause_paragraph:
            text = " ".join(t for t in current_texts if t)
            groups.append({
                "start": current_start,
                "end": current_end,
                "text": text,
                "segment_indices": current_indices.copy(),
            })
            current_indices = [i]
            current_start = seg["start"]
            current_end = seg["end"]
            current_texts = [seg["text"].strip()]
            continue

        # 그룹 길이 상한 초과 -> 새 그룹
        if (seg["end"] - current_start) > max_group_duration:
            text = " ".join(t for t in current_texts if t)
            groups.append({
                "start": current_start,
                "end": current_end,
                "text": text,
                "segment_indices": current_indices.copy(),
            })
            current_indices = [i]
            current_start = seg["start"]
            current_end = seg["end"]
            current_texts = [seg["text"].strip()]
            continue

        # 문장 끝 + (짧은 침묵이거나 다음으로 이어짐) -> 문장 경계로 새 그룹
        if prev_ends_sentence and gap >= pause_after_sentence_end:
            text = " ".join(t for t in current_texts if t)
            groups.append({
                "start": current_start,
                "end": current_end,
                "text": text,
                "segment_indices": current_indices.copy(),
            })
            current_indices = [i]
            current_start = seg["start"]
            current_end = seg["end"]
            current_texts = [seg["text"].strip()]
            continue

        # 문장 수준 침묵 (문장 끝 부호 없음) -> 새 그룹
        if gap >= pause_sentence:
            text = " ".join(t for t in current_texts if t)
            groups.append({
                "start": current_start,
                "end": current_end,
                "text": text,
                "segment_indices": current_indices.copy(),
            })
            current_indices = [i]
            current_start = seg["start"]
            current_end = seg["end"]
            current_texts = [seg["text"].strip()]
            continue

        # 같은 호흡/문맥으로 묶음 (1차 기준)
        current_indices.append(i)
        current_end = seg["end"]
        current_texts.append(seg["text"].strip())

    if current_indices:
        text = " ".join(t for t in current_texts if t)
        groups.append({
            "start": current_start,
            "end": current_end,
            "text": text,
            "segment_indices": current_indices.copy(),
        })

    # 2차: 의미/주제 기반 병합 (LLM)
    if not use_semantic_merge or len(groups) <= 1:
        return groups

    merge_after = decide_semantic_merges(groups)
    if not merge_after:
        return groups

    merged: list[dict] = []
    current = {
        "start": groups[0]["start"],
        "end": groups[0]["end"],
        "text": groups[0]["text"],
        "segment_indices": list(groups[0]["segment_indices"]),
    }

    for i in range(len(groups) - 1):
        g_next = groups[i + 1]

        if i in merge_after:
            # 의미적으로 같은 맥락 -> 병합
            current["end"] = g_next["end"]
            current["text"] = f"{current['text']} {g_next['text']}".strip()
            current["segment_indices"].extend(g_next["segment_indices"])
        else:
            merged.append(current)
            current = {
                "start": g_next["start"],
                "end": g_next["end"],
                "text": g_next["text"],
                "segment_indices": list(g_next["segment_indices"]),
            }

    if current:
        merged.append(current)

    return merged


def expand_group_annotations_to_segments(
    annotated_groups: list[dict],
    segments: list[dict],
    groups: list[dict],
) -> list[dict]:
    """
    그룹 단위 강조 결과를 원본 세그먼트 단위로 펼친다.

    - annotated_groups: 그룹별 강조 정보가 붙은 리스트 (각 항목에 'start'로 그룹 식별)
    - segments: 원본 전사 세그먼트 리스트
    - groups: group_segments_by_context() 반환값 (각 그룹에 segment_indices 있음)

    반환: segments와 같은 길이·순서의 리스트. 각 세그먼트에 해당 그룹의
    audio_emphasis, 점수 산출용 필드 등이 복사됨.
    """
    group_by_start = {g["start"]: g for g in groups}
    annotated_by_start = {s["start"]: s for s in annotated_groups}

    # segment index -> annotated group (that contains this segment)
    index_to_annotated = {}
    for g in groups:
        start = g["start"]
        ann = annotated_by_start.get(start)
        if ann is None:
            continue
        for idx in g["segment_indices"]:
            index_to_annotated[idx] = ann

    result = []
    for i, seg in enumerate(segments):
        seg_copy = seg.copy()
        ann = index_to_annotated.get(i)
        if ann:
            for key in (
                "emphasis_methods",
                "emphasis_reasons",
                "detection_count",
                "confidence",
                "emphasis_signals",
                "emphasis_keywords",
                "emphasis_keywords_by_method",
                "emphasis_detail",
                "audio_emphasis",
            ):
                if key in ann:
                    seg_copy[key] = ann[key]
        else:
            seg_copy["detection_count"] = 0
        result.append(seg_copy)

    return result


# ---------------------------------------------------------------------------
# scene 기반 그룹화 (metadata.json 사용)
# ---------------------------------------------------------------------------

def load_slide_ranges(metadata_path: str, duration_sec: float) -> list[dict]:
    """
    metadata.json에서 scene occurrence별 시간 구간 계산.
    같은 scene_index의 base(annot_index=0)가 한 scene 시작.
    반환: [ {"scene_index": 1, "slide_number": 1, "start_sec": 0.07, "end_sec": 46.73}, ... ]
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    bases = [
        x for x in items
        if x.get("capture_type") == "base"
        or (x.get("annot_index") == 0 and x.get("capture_type") not in {"build", "annotation", "annot"})
    ]

    def _scene_index(item: dict):
        return item.get("scene_index") if item.get("scene_index") is not None else item.get("slide_index")

    bases = sorted(
        bases,
        key=lambda x: (
            _scene_index(x),
            x["timestamp_sec"],
        ),
    )
    seen = set()
    unique_bases = []
    for b in bases:
        scene_idx = _scene_index(b)
        if scene_idx is None:
            continue
        if scene_idx in seen:
            continue
        seen.add(scene_idx)
        unique_bases.append({
            "scene_index": scene_idx,
            "slide_number": b.get("slide_number"),
            "slide_canonical_index": b.get("slide_canonical_index", b.get("same_slide_canonical")),
            "slide_visit_order": b.get("slide_visit_order", b.get("same_slide_visit_order", 1)),
            "slide_is_revisit": bool(b.get("slide_is_revisit", b.get("same_slide_is_revisit", False))),
            "timestamp_sec": b["timestamp_sec"],
        })
    unique_bases.sort(key=lambda x: x["timestamp_sec"])
    ranges = []
    for i, b in enumerate(unique_bases):
        start = b["timestamp_sec"]
        end = unique_bases[i + 1]["timestamp_sec"] if i + 1 < len(unique_bases) else duration_sec
        ranges.append({
            "scene_index": b["scene_index"],
            "slide_number": b.get("slide_number"),
            "slide_canonical_index": b.get("slide_canonical_index"),
            "slide_visit_order": b.get("slide_visit_order", 1),
            "slide_is_revisit": b.get("slide_is_revisit", False),
            "start_sec": start,
            "end_sec": end,
        })
    return ranges


def group_segments_by_scene_and_context(
    segments: list[dict],
    scene_ranges: list[dict],
    duration_sec: float,
    *,
    use_llm_merge: bool = True,
    use_pause_sentence: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    먼저 scene occurrence별로 세그먼트를 나누고, 각 scene 내부에서 의미 전환점 기준으로 컨텍스트 구성.
    use_pause_sentence: True면 침묵/문장끝 기준 분할 추가 (나중에 사용할 옵션).

    반환: (groups_flat, scenes_structure)
    - groups_flat: 강조 분석용 그룹 리스트 (start, end, text, segment_indices, scene_index, context_index_in_scene)
    - scenes_structure: 최종 JSON용 [ { scene_index, slide_number, start_sec, end_sec, text, contexts: [...] } ]
    """
    if not segments or not scene_ranges:
        return [], []

    seg_to_scene = []
    for i, seg in enumerate(segments):
        t = seg["start"]
        scene_idx = None
        for r in scene_ranges:
            if r["start_sec"] <= t < r["end_sec"]:
                scene_idx = r["scene_index"]
                break
        if scene_idx is None and scene_ranges:
            if t < scene_ranges[0]["start_sec"]:
                scene_idx = scene_ranges[0]["scene_index"]
            else:
                scene_idx = scene_ranges[-1]["scene_index"]
        seg_to_scene.append(scene_idx)

    # Scene별 의미 경계 판단은 서로 독립적인 LLM 호출이므로 병렬 실행한다.
    # 결과는 scene_ranges 순서로 다시 조립해 기존 JSON 순서를 유지한다.
    scene_entries = []
    for r in scene_ranges:
        sidx = r["scene_index"]
        indices_in_scene = [i for i in range(len(segments)) if seg_to_scene[i] == sidx]
        initial_groups = []
        for i in indices_in_scene:
            seg = segments[i]
            initial_groups.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": (seg.get("text") or "").strip(),
                "segment_indices": [i],
            })
        scene_entries.append({"range": r, "initial_groups": initial_groups})

    break_after_by_entry: dict[int, set[int]] = {}
    llm_jobs = [
        (entry_index, entry["initial_groups"])
        for entry_index, entry in enumerate(scene_entries)
        if use_llm_merge and len(entry["initial_groups"]) > 1
    ]
    if llm_jobs:
        worker_count = min(_effective_context_parallel_requests(), len(llm_jobs))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(decide_context_breaks, groups): entry_index
                for entry_index, groups in llm_jobs
            }
            for future in as_completed(future_map):
                entry_index = future_map[future]
                try:
                    break_after_by_entry[entry_index] = set(future.result() or set())
                except Exception:
                    break_after_by_entry[entry_index] = set()

    scenes_structure = []
    groups_flat = []
    for entry_index, entry in enumerate(scene_entries):
        r = entry["range"]
        sidx = r["scene_index"]
        start_sec = r["start_sec"]
        end_sec = r["end_sec"]
        initial_groups = entry["initial_groups"]
        if not initial_groups:
            scenes_structure.append({
                "scene_id": f"scene/{int(sidx):04d}",
                "scene_index": sidx,
                "slide_number": r.get("slide_number"),
                "slide_canonical_index": r.get("slide_canonical_index"),
                "slide_visit_order": r.get("slide_visit_order", 1),
                "slide_is_revisit": r.get("slide_is_revisit", False),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "text": "",
                "contexts": [],
            })
            continue

        break_after = _apply_context_segment_limit(
            initial_groups,
            break_after_by_entry.get(entry_index, set()),
        )

        # 수업 운영/감사/짧은 전환 멘트는 본 설명 context와 섞이지 않도록 앞뒤를 끊는다.
        for j, g in enumerate(initial_groups):
            if not _is_operational_or_transition_segment(g.get("text", "")):
                continue
            if j > 0:
                break_after.add(j - 1)
            if j < len(initial_groups) - 1:
                break_after.add(j)

        context_groups = []
        cur = {
            "start": initial_groups[0]["start"],
            "end": initial_groups[0]["end"],
            "text": initial_groups[0]["text"],
            "segment_indices": list(initial_groups[0]["segment_indices"]),
        }
        for j in range(len(initial_groups) - 1):
            g_next = initial_groups[j + 1]
            if j in break_after:
                context_groups.append(cur)
                cur = {
                    "start": g_next["start"],
                    "end": g_next["end"],
                    "text": g_next["text"],
                    "segment_indices": list(g_next["segment_indices"]),
                }
            else:
                cur["end"] = g_next["end"]
                cur["text"] = f"{cur['text']} {g_next['text']}".strip()
                cur["segment_indices"].extend(g_next["segment_indices"])
        context_groups.append(cur)

        scene_text = " ".join((g["text"] for g in context_groups if g["text"])).strip()
        contexts_for_scene = []
        for cix, g in enumerate(context_groups):
            g_flat = {
                "start": g["start"],
                "end": g["end"],
                "text": g["text"],
                "segment_indices": g["segment_indices"],
                "scene_index": sidx,
                "slide_number": r.get("slide_number"),
                "context_index_in_scene": cix,
            }
            groups_flat.append(g_flat)
            segs_in_context = [
                {"start": segments[i]["start"], "end": segments[i]["end"], "text": (segments[i].get("text") or "").strip()}
                for i in g["segment_indices"]
            ]
            contexts_for_scene.append({
                "context_index": cix,
                "start": g["start"],
                "end": g["end"],
                "text": g["text"],
                "segment_indices": g["segment_indices"],
                "segments": segs_in_context,
            })
        scenes_structure.append({
            "scene_id": f"scene/{int(sidx):04d}",
            "scene_index": sidx,
            "slide_number": r.get("slide_number"),
            "slide_canonical_index": r.get("slide_canonical_index"),
            "slide_visit_order": r.get("slide_visit_order", 1),
            "slide_is_revisit": r.get("slide_is_revisit", False),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "text": scene_text,
            "contexts": contexts_for_scene,
        })

    return groups_flat, scenes_structure
