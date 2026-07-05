"""
키워드 기반 강조 구간 감지

- 가중치 키워드: 하드코딩된 중요/요약/시험 관련 표현 매칭
- 주제 키워드 반복: 전사문 전체에서 반복 등장하는 내용어 기반

키워드 추출: kiwipiepy/Kiwi 형태소 분석 필수(명사만). 미설치 시 중단.
주제 키워드 LLM 필터: Gemini로 강의 흐름과 무관한 후보 제거.

진단: EMPHASIS_DEBUG=1 python main.py ... 로 실행하면 주제 키워드 진단 결과를 output/ 폴더에 저장.

max_keywords / min_freq 조정 (기본값 20/5):
  [1] emphasis_keyword.py
      - _get_topic_keywords(..., min_freq=5, max_keywords=20)  함수 정의부 기본값
      - get_topic_keywords_list(..., min_freq=5, max_keywords=20)  함수 정의부 기본값
      - detect_emphasis_by_topic_keyword_repetition(..., min_freq=5, max_keywords=20)  함수 정의부 기본값
      - diagnose_topic_keywords 내부 _get_topic_keywords(..., min_freq=5, max_keywords=20)  호출 인자
  [2] main.py
      - get_topic_keywords_list(groups, min_freq=5, max_keywords=20, ...)  키워드 보고서용 호출
  위 네 군데를 같은 값으로 맞춰서 바꾸면 됨.
"""

import json
import os
import re
from collections import Counter
from typing import Optional, Set


# ---------------------------------------------------------------------------
# 가중치 키워드 감지
# ---------------------------------------------------------------------------

KEYWORDS_WEIGHTED = {
    'exam':    (['시험', '문제', '출제', '나옵니다'], 5),
    'strong':  (['중요', '핵심', '반드시', '꼭', '필수'], 3),
    'summary': (['정리하면', '요약하면', '다시 말하면', '즉'], 1),
}


def detect_emphasis_by_keywords_weighted(segments: list[dict]) -> list[dict]:
    """중요 표현 키워드 점수 계산. 카테고리별로 한 번만 가산한다."""
    print("  [가중치 키워드] 분석 중...")
    emphasis_segments = []

    for seg in segments:
        text = seg['text']
        matched_keywords_by_category: dict[str, list[str]] = {}
        category_scores = {}
        matched_categories = []
        keyword_score = 0

        for category, (keywords, weight) in KEYWORDS_WEIGHTED.items():
            matched = [keyword for keyword in keywords if keyword in text]
            matched_keywords_by_category[category] = matched
            if matched:
                matched_categories.append(category)
                category_scores[category] = weight
                keyword_score += weight
            else:
                category_scores[category] = 0

        flat_matched_keywords = []
        for keywords in matched_keywords_by_category.values():
            for keyword in keywords:
                if keyword not in flat_matched_keywords:
                    flat_matched_keywords.append(keyword)

        emphasis_segments.append({
            'start': seg['start'],
            'end': seg['end'],
            'text': text,
            'emphasis_score': float(keyword_score),
            'importance_keyword_score': int(keyword_score),
            'category_scores': category_scores,
            'matched_categories': matched_categories,
            'matched_keywords': flat_matched_keywords,
            'matched_keywords_by_category': matched_keywords_by_category,
            'detected': keyword_score > 0,
            'detection_method': 'keyword_weighted',
        })

    detected_count = sum(1 for s in emphasis_segments if s.get("detected"))
    ratio = detected_count / len(segments) * 100 if segments else 0.0
    print(f"    -> 가중치 키워드: {detected_count}개 detected (전체의 {ratio:.1f}%), 점수 {len(emphasis_segments)}개")
    return emphasis_segments


# ---------------------------------------------------------------------------
# 주제 키워드 추출 (Kiwi 형태소 분석 필수)
# ---------------------------------------------------------------------------

_KIWI = None


def _get_kiwi():
    global _KIWI
    if _KIWI is None:
        try:
            from kiwipiepy import Kiwi
        except ImportError as exc:
            raise RuntimeError(
                "kiwipiepy가 설치되어 있지 않아 Kiwi 기반 키워드 추출을 중단합니다. "
                "app/backend/pipeline/requirements.txt를 설치하거나 `pip install kiwipiepy`를 실행하세요."
            ) from exc
        _KIWI = Kiwi()
    return _KIWI


_MINIMAL_STOP = {
    "그", "이", "저", "것", "수", "등", "및", "또는", "그리고",
    "있다", "하다", "되다", "이다", "없다", "않다",
    "위해", "통해", "대한", "있는", "하는", "되는", "라는",
    "이제", "그래서", "그러면", "그럼", "이렇게", "그렇게",
    "어떤", "무슨", "어디", "언제", "왜", "몇", "네", "예", "아니요",
    "이런", "저런", "그런", "여러", "바로", "우리", "다음", "이거", "그거", "저거",
    "이런것", "그런것", "저런것", "무엇", "어떤것",
    "문제", "경우", "방법", "이유", "결과", "사실", "정말", "대부분", "보통", "항상",
    "그런데", "그러나", "따라서", "즉시", "아마", "혹시", "좀", "더", "매우", "너무", "잘", "많이", "적게",
}

_ENDINGS = (
    "했습니다", "됐습니다", "였습니다", "습니다", "ㅂ니다",
    "입니다", "합니다", "됩니다",
    "이에요", "이예요", "예요", "에요", "네요", "군요", "어요", "아요", "해요", "되요",
    "되는", "하는", "있는", "라는", "이라는", "된", "할", "하게", "하지",
    "는데", "니까", "거나", "지만", "에서", "으로", "고", "면", "며", "죠", "네",
    "부터", "까지", "처럼", "대로", "마저", "조차", "랑", "들",
    "가", "를", "을", "의", "로", "에", "와", "과", "는", "은", "도", "만", "이",
)

_NOUN_TAGS = ("NNG", "NNP", "SL")


def _strip_endings(word: str) -> str:
    if not word or len(word) < 2:
        return word
    for ending in _ENDINGS:
        if word.endswith(ending) and len(word) > len(ending):
            stem = word[: -len(ending)]
            if len(stem) >= 2:
                return stem
    return word


def _tokenize(text: str) -> list[str]:
    text = (text or "").strip()
    text = re.sub(r"[^\w\s\u3131-\uD7A3]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) >= 2]


def _content_stems(tokens: list[str], min_length: int = 2) -> list[str]:
    stems = []
    for t in tokens:
        if t in _MINIMAL_STOP or len(t) < min_length:
            continue
        s = _strip_endings(t)
        if len(s) >= min_length and s not in _MINIMAL_STOP:
            stems.append(s)
    return stems


def _add_compound_nouns_from_seq(seq: list[str], out: list[str], min_length: int) -> None:
    if len(seq) < 2:
        return
    max_n = 3
    n = len(seq)
    for size in range(2, min(max_n, n) + 1):
        for i in range(0, n - size + 1):
            comp = "".join(seq[i: i + size])
            if len(comp) >= min_length and comp not in _MINIMAL_STOP:
                out.append(comp)


def _extract_content_words(text: str, min_length: int = 3) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    kiwi = _get_kiwi()
    try:
        tokens = kiwi.tokenize(text)
    except Exception as exc:
        raise RuntimeError("Kiwi 형태소 분석 중 오류가 발생해 키워드 추출을 중단합니다.") from exc

    words: list[str] = []
    noun_seq: list[str] = []
    for t in tokens:
        tag = getattr(t, "tag", None)
        form = getattr(t, "form", None)
        if tag in _NOUN_TAGS and form and len(form) >= min_length and form not in _MINIMAL_STOP:
            words.append(form)
            noun_seq.append(form)
        else:
            if noun_seq:
                _add_compound_nouns_from_seq(noun_seq, words, min_length)
                noun_seq = []
    if noun_seq:
        _add_compound_nouns_from_seq(noun_seq, words, min_length)
    return words


def extract_content_words(text: str, min_length: int = 3) -> list[str]:
    """Kiwi 기반 내용어 추출 공개 API. Kiwi 미설치/분석 실패 시 예외를 올린다."""
    return _extract_content_words(text, min_length=min_length)


def extract_contiguous_content_words(text: str, min_length: int = 3) -> list[str]:
    """
    원문에서 공백/기호 없이 붙어 있던 명사열만 복합어로 묶어 반환한다.

    슬라이드 강조 키워드처럼 사용자에게 직접 노출되는 후보용이다.
    예: "운영체제 개념" -> ["운영체제", "개념"]
        "응용소프트웨어 운영체제" -> ["응용소프트웨어", "운영체제"]
    """
    text = (text or "").strip()
    if not text:
        return []
    kiwi = _get_kiwi()
    try:
        tokens = kiwi.tokenize(text)
    except Exception as exc:
        raise RuntimeError("Kiwi 형태소 분석 중 오류가 발생해 키워드 추출을 중단합니다.") from exc

    result: list[str] = []
    run: list = []
    prev_end: Optional[int] = None

    def flush_run() -> None:
        nonlocal run
        if not run:
            return
        forms = [getattr(t, "form", "") for t in run]
        if len(forms) >= 2:
            compound = "".join(forms)
            if len(compound) >= min_length and compound not in _MINIMAL_STOP:
                result.append(compound)
        else:
            form = forms[0]
            if len(form) >= min_length and form not in _MINIMAL_STOP:
                result.append(form)
        run = []

    for t in tokens:
        tag = getattr(t, "tag", None)
        form = getattr(t, "form", None)
        start = getattr(t, "start", None)
        length = getattr(t, "len", None)
        is_content = (
            tag in _NOUN_TAGS
            and form
            and len(form) >= min_length
            and form not in _MINIMAL_STOP
            and start is not None
            and length is not None
        )
        if not is_content:
            flush_run()
            prev_end = None
            continue

        if run and prev_end == start:
            run.append(t)
        else:
            flush_run()
            run = [t]
        prev_end = int(start) + int(length)

    flush_run()
    return result


def _select_representative_keywords(candidates: set[str]) -> set[str]:
    if not candidates:
        return set()
    reps: list[str] = []
    for w in sorted(candidates, key=len, reverse=True):
        if any(w in r and w != r for r in reps):
            continue
        reps.append(w)
    return set(reps)


def _get_topic_keywords(
    segments: list[dict],
    min_freq: int = 5,
    max_keywords: int = 20,
    max_segment_ratio: float = 1.0,
    min_length: int = 2,
) -> set[str]:
    if not segments:
        return set()
    total_segments = len(segments)
    counter: Counter = Counter()
    segment_presence: Counter = Counter()

    for seg in segments:
        text = (seg.get("text") or "").strip()
        words = _extract_content_words(text, min_length=min_length)
        seen_here = set()
        for s in words:
            counter[s] += 1
            seen_here.add(s)
        for s in seen_here:
            segment_presence[s] += 1

    threshold_segments = total_segments if max_segment_ratio >= 1.0 else max(2, int(total_segments * max_segment_ratio))
    topic = set()
    for w, cnt in counter.most_common(max_keywords * 2):
        if cnt < min_freq:
            continue
        if segment_presence[w] > threshold_segments:
            continue
        topic.add(w)
        if len(topic) >= max_keywords:
            break
    return _select_representative_keywords(topic)


def _filter_topic_keywords_by_llm(segments: list[dict], candidate_keywords: set[str]) -> set[str]:
    if not candidate_keywords:
        return candidate_keywords
    try:
        from google.genai import types
        from .config import GEMINI_GENERATIVE_MODEL, gemini_client_2
        from .utils import api_call_with_retry
    except ImportError:
        return candidate_keywords

    parts = []
    n = len(segments)
    step = max(1, n // 8) if n > 8 else 1
    for i in range(0, min(n, 50), step):
        parts.append((segments[i].get("text") or "").strip())
    context = " ".join(parts).strip()[:4000]
    keyword_list = sorted(candidate_keywords)

    prompt = f"""당신은 강의 전사문을 보고, 그 강의의 **주제와 흐름에 맞는 키워드**만 골라주는 도우미입니다.

아래는 이 강의 전사의 앞·중간 일부입니다 (맥락 파악용):
---
{context}
---

아래는 전사문에서 빈도 기반으로 뽑은 **키워드 후보** 목록입니다. 이 중에서:
- 이 강의의 주제·내용·흐름과 **관련 있는** 키워드만 남기고,
- 주제와 **동떨어진** 단어(다른 분야 용어, 말실수/오타로 나온 단어, 강의와 무관한 단어)는 제외해주세요.

키워드 후보 (쉼표로 구분):
{", ".join(keyword_list)}

출력은 반드시 아래 JSON 형식만 사용하세요. 설명 없이 JSON만 출력하세요.
```json
{{ "keywords": ["키워드1", "키워드2", ...] }}
```
선택한 키워드만 배열에 넣으면 됩니다. 반드시 위 후보 목록에 있던 단어만 포함하세요."""

    def call_api():
        return gemini_client_2.models.generate_content(
            model=GEMINI_GENERATIVE_MODEL,
            contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=2048),
        )

    try:
        from .utils import api_call_with_retry
        response = api_call_with_retry(call_api)
        try:
            from .cost_report import record_model_call

            record_model_call(
                stage="stage3b_emphasis_keyword",
                provider="google",
                model=GEMINI_GENERATIVE_MODEL,
                response=response,
                prompt_chars=len(prompt),
            )
        except Exception:
            pass
        text = (response.text or "").strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        data = json.loads(text)
        filtered = [str(k).strip() for k in data.get("keywords", [])]
        result = set(filtered) & candidate_keywords
        if result:
            print(f"    -> LLM 필터: {len(candidate_keywords)}개 후보 -> {len(result)}개 (주제 관련만 유지)")
            return result
    except Exception:
        pass
    return candidate_keywords


# ---------------------------------------------------------------------------
# 주제 키워드 공개 API
# ---------------------------------------------------------------------------

def get_topic_keywords_filtered(
    segments: list[dict],
    *,
    min_freq: int = 5,
    max_keywords: int = 20,
    max_segment_ratio: float = 1.0,
    min_keyword_len: int = 2,
    use_llm_filter: bool = True,
) -> set[str]:
    """주제 키워드 추출 + LLM 필터 적용. 여러 min_keyword_count 변형 실행 시 1회만 호출."""
    kw_set = _get_topic_keywords(
        segments,
        min_freq=min_freq,
        max_keywords=max_keywords,
        max_segment_ratio=max_segment_ratio,
        min_length=min_keyword_len,
    )
    if use_llm_filter and kw_set:
        kw_set = _filter_topic_keywords_by_llm(segments, kw_set)
    return kw_set


def get_topic_keywords_filtered_v2(
    segments: list[dict],
    *,
    min_freq: int = 5,
    max_keywords: int = 20,
    max_segment_ratio: float = 1.0,
    min_keyword_len: int = 2,
    candidate_pool_size: int = 80,
    use_llm_filter: bool = True,
) -> set[str]:
    """
    v2: 전체 후보를 더 많이 모은 뒤 LLM으로 주제 무관만 제거하고,
    남은 것 중 빈도 순 상위 max_keywords개를 반환.
    """
    if not segments:
        return set()
    total_segments = len(segments)
    counter: Counter = Counter()
    segment_presence: Counter = Counter()
    for seg in segments:
        text = (seg.get("text") or "").strip()
        words = _extract_content_words(text, min_length=min_keyword_len)
        seen_here = set()
        for s in words:
            counter[s] += 1
            seen_here.add(s)
        for s in seen_here:
            segment_presence[s] += 1

    threshold_segments = total_segments if max_segment_ratio >= 1.0 else max(2, int(total_segments * max_segment_ratio))
    candidate_pool: set[str] = set()
    for w, cnt in counter.most_common(candidate_pool_size * 3):
        if cnt < min_freq:
            continue
        if segment_presence[w] > threshold_segments:
            continue
        candidate_pool.add(w)
        if len(candidate_pool) >= candidate_pool_size:
            break

    if not candidate_pool:
        return set()

    filtered = _filter_topic_keywords_by_llm(segments, candidate_pool) if use_llm_filter else candidate_pool

    sorted_by_freq = sorted(filtered, key=lambda w: counter[w], reverse=True)
    top_expanded = sorted_by_freq[: max(max_keywords * 2, 40)]
    representative = _select_representative_keywords(set(top_expanded))
    repr_sorted = sorted(representative, key=lambda w: counter[w], reverse=True)[:max_keywords]
    return set(repr_sorted)


def get_topic_keyword_count_map(
    segments: list[dict],
    *,
    min_freq: int = 5,
    max_keywords: int = 20,
    max_segment_ratio: float = 1.0,
    min_keyword_len: int = 2,
    candidate_pool_size: int = 80,
    use_llm_filter: bool = True,
    _topic_keywords_override: Optional[Set[str]] = None,
) -> dict[str, int]:
    """
    주제 키워드별 전체 등장 횟수를 반환.

    반환값은 최종 주제 키워드로 살아남은 단어만 포함한다.
    개별 context/slide 집계에서는 같은 키워드가 여러 번 등장해도 이 total_count를 한 번만 더한다.
    """
    if not segments:
        return {}

    counter: Counter = Counter()
    for seg in segments:
        text = (seg.get("text") or "").strip()
        words = _extract_content_words(text, min_length=min_keyword_len)
        for s in words:
            counter[s] += 1

    if _topic_keywords_override is not None:
        topic_keywords = set(_topic_keywords_override)
    else:
        topic_keywords = get_topic_keywords_filtered_v2(
            segments,
            min_freq=min_freq,
            max_keywords=max_keywords,
            max_segment_ratio=max_segment_ratio,
            min_keyword_len=min_keyword_len,
            candidate_pool_size=candidate_pool_size,
            use_llm_filter=use_llm_filter,
        )

    return {
        kw: int(counter.get(kw, 0))
        for kw in sorted(topic_keywords, key=lambda w: (-counter.get(w, 0), w))
        if counter.get(kw, 0) >= min_freq
    }


def get_topic_keyword_score_map(topic_count_map: dict[str, int]) -> dict[str, int]:
    """반복 키워드 count 내림차순으로 4개씩 5~1점을 부여한다."""
    if not topic_count_map:
        return {}
    sorted_items = sorted(topic_count_map.items(), key=lambda item: (-int(item[1]), item[0]))
    result = {}
    for idx, (kw, _count) in enumerate(sorted_items[:20]):
        result[kw] = max(1, 5 - idx // 4)
    return result


def topic_keyword_count_items(topic_count_map: dict[str, int]) -> list[dict]:
    """JSON 저장용 [{keyword,total_count}] 목록으로 변환."""
    score_map = get_topic_keyword_score_map(topic_count_map)
    return [
        {"keyword": kw, "total_count": int(total), "score": int(score_map.get(kw, 0))}
        for kw, total in sorted(topic_count_map.items(), key=lambda item: (-item[1], item[0]))
    ]


def summarize_topic_keyword_counts_for_text(
    text: str,
    topic_count_map: dict[str, int],
    *,
    min_keyword_len: int = 2,
) -> dict:
    """
    특정 text에 포함된 반복 키워드와 그 키워드들의 total_count 합을 반환.

    같은 text 안에 같은 키워드가 여러 번 나와도 total_count는 한 번만 더한다.
    """
    if not text or not topic_count_map:
        return {"keywords": [], "total_count_sum": 0, "keyword_scores": {}, "score_sum": 0}
    words = set(_extract_content_words(text, min_length=min_keyword_len))
    keywords = sorted(words & set(topic_count_map), key=lambda w: (-topic_count_map[w], w))
    score_map = get_topic_keyword_score_map(topic_count_map)
    keyword_scores = {kw: int(score_map.get(kw, 0)) for kw in keywords}
    return {
        "keywords": keywords,
        "total_count_sum": int(sum(topic_count_map[kw] for kw in keywords)),
        "keyword_scores": keyword_scores,
        "score_sum": int(sum(keyword_scores.values())),
    }


def get_topic_keywords_list(
    segments: list[dict],
    *,
    min_freq: int = 5,
    max_keywords: int = 20,
    max_segment_ratio: float = 1.0,
    min_keyword_len: int = 2,
) -> list[str]:
    """주제 키워드 반복 감지에 사용되는 키워드 목록 반환 (확인용)."""
    kw_set = _get_topic_keywords(
        segments,
        min_freq=min_freq,
        max_keywords=max_keywords,
        max_segment_ratio=max_segment_ratio,
        min_length=min_keyword_len,
    )
    return sorted(kw_set)


# ---------------------------------------------------------------------------
# 주제 키워드 반복 감지
# ---------------------------------------------------------------------------

def detect_emphasis_by_topic_keyword_repetition(
    segments: list[dict],
    *,
    window: int = 2,
    min_keyword_len: int = 2,
    max_segment_ratio: float = 1.0,
    min_freq: int = 5,
    max_keywords: int = 20,
    use_llm_filter: bool = True,
    min_keyword_count: int = 1,
    _topic_keywords_override: Optional[Set[str]] = None,
    _topic_keyword_count_map: Optional[dict[str, int]] = None,
    _topic_keyword_score_map: Optional[dict[str, int]] = None,
) -> list[dict]:
    """
    전사문 전체에서 반복되는 주제 키워드가 등장하는 구간을 강조로 표시.

    - min_freq: 전사문 전체 등장 횟수 기준.
    - use_llm_filter: True면 Gemini로 주제 무관 키워드 제거.
    - _topic_keywords_override: 이미 추출된 키워드 집합을 넘기면 재추출/LLM 필터 생략.
    """
    label = f"min_kw>={min_keyword_count}"
    _get_kiwi()
    print(f"  [주제 키워드 반복 ({label})] 분석 중 (전사 전체 기준, Kiwi 명사만 사용)...")
    if not segments:
        return []

    if _topic_keywords_override is not None:
        topic_keywords = _topic_keywords_override
    else:
        topic_keywords = _get_topic_keywords(
            segments,
            min_freq=min_freq,
            max_keywords=max_keywords,
            max_segment_ratio=max_segment_ratio,
            min_length=min_keyword_len,
        )
        if use_llm_filter and topic_keywords:
            topic_keywords = _filter_topic_keywords_by_llm(segments, topic_keywords)
    if not topic_keywords:
        print(f"    -> 주제 키워드 없음 (반복 단어 부족)")
        return []
    topic_keyword_score_map = _topic_keyword_score_map
    if topic_keyword_score_map is None:
        topic_keyword_score_map = get_topic_keyword_score_map(_topic_keyword_count_map or {})

    emphasis_segments = []
    for seg in segments:
        words = _extract_content_words(seg.get("text") or "", min_length=min_keyword_len)
        here = set(words) & topic_keywords
        repeated_words = sorted(here)
        topic_total_count_sum = 0
        if _topic_keyword_count_map:
            # 같은 context 안에 같은 키워드가 여러 번 나와도 total_count는 한 번만 더한다.
            topic_total_count_sum = int(sum(_topic_keyword_count_map.get(w, 0) for w in repeated_words))
        keyword_scores = {w: int(topic_keyword_score_map.get(w, 0)) for w in repeated_words}
        topic_keyword_score = int(sum(keyword_scores.values()))
        emphasis_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "emphasis_score": float(topic_keyword_score),
            "repeated_topic_keywords": repeated_words,
            "repeated_topic_keyword_scores": keyword_scores,
            "audio_topic_total_count_sum": topic_total_count_sum,
            "audio_topic_keyword_score": topic_keyword_score,
            "detected": len(here) >= min_keyword_count and topic_keyword_score > 0,
            "detection_method": "topic_keyword_repeat",
        })

    detected_count = sum(1 for s in emphasis_segments if s.get("detected"))
    print(f"    -> 주제 키워드 반복 ({label}): {detected_count}개 (전체의 {detected_count/len(segments)*100:.1f}%)")
    return emphasis_segments


# ---------------------------------------------------------------------------
# 진단 도구 (EMPHASIS_DEBUG=1 시 사용)
# ---------------------------------------------------------------------------

def diagnose_topic_keywords(
    segments: list[dict],
    check_words: tuple[str, ...] = ("운영체제", "운영", "체제"),
    min_length: int = 2,
    max_segment_ratio: float = 1.0,
) -> dict:
    """주제 키워드가 왜 잡히지 않는지 진단."""
    total = len(segments)
    kiwi = _get_kiwi()
    lines = []
    lines.append("=== 주제 키워드 진단 ===")
    lines.append(f"전체 구간 수: {total}")
    lines.append(f"min_length={min_length}, max_segment_ratio={max_segment_ratio}")
    lines.append("")

    if kiwi is not None:
        try:
            sample = "운영체제의 목적과 기능"
            tokens = kiwi.tokenize(sample)
            lines.append(f"[Kiwi 분석 예시] '{sample}'")
            for t in tokens:
                form = getattr(t, "form", "?")
                tag = getattr(t, "tag", "?")
                lines.append(f"  form='{form}' tag={tag} len(form)={len(form)} NNG/NNP? {tag in _NOUN_TAGS} min_length 통과? {len(form) >= min_length} stop? {form in _MINIMAL_STOP}")
            lines.append("")
        except Exception as e:
            lines.append(f"Kiwi 예시 분석 실패: {e}")
            lines.append("")
    counter: Counter = Counter()
    segment_presence: Counter = Counter()
    for seg in segments:
        text = (seg.get("text") or "").strip()
        words = _extract_content_words(text, min_length=min_length)
        seen = set(words)
        for s in words:
            counter[s] += 1
        for s in seen:
            segment_presence[s] += 1

    threshold = total if max_segment_ratio >= 1.0 else max(2, int(total * max_segment_ratio))
    lines.append(f"threshold_segments (이 값 초과 시 제외): {threshold}")
    lines.append("")

    topic = _get_topic_keywords(
        segments, min_freq=5, max_keywords=20,
        max_segment_ratio=max_segment_ratio, min_length=min_length,
    )
    lines.append(f"topic_keywords 개수: {len(topic)}")
    lines.append("")

    for w in check_words:
        cnt = counter.get(w, 0)
        pres = segment_presence.get(w, 0)
        in_topic = w in topic
        lines.append(f"[{w}]")
        lines.append(f"  전체 등장 횟수: {cnt}, 등장한 구간 수: {pres}")
        lines.append(f"  threshold 초과? {pres > threshold} -> 제외됨: {pres > threshold}")
        lines.append(f"  min_freq(5) 미달? {cnt < 5}")
        lines.append(f"  topic_keywords 포함: {in_topic}")
        if not in_topic and cnt > 0:
            reason = []
            if cnt < 5:
                reason.append("min_freq 미달")
            if pres > threshold:
                reason.append("구간 비율 초과로 제외")
            if w in _MINIMAL_STOP:
                reason.append("stopword")
            if len(w) < min_length:
                reason.append(f"길이 {len(w)} < min_length {min_length}")
            lines.append(f"  -> 추정 원인: {', '.join(reason) or '확인 필요 (Kiwi 품사/길이 등)'}")
        lines.append("")

    sample_list = sorted(topic)[:30]
    lines.append("topic_keywords 샘플 (처음 30개): " + ", ".join(sample_list))
    return {"lines": lines, "topic_count": len(topic), "in_topic": {w: w in topic for w in check_words}}
