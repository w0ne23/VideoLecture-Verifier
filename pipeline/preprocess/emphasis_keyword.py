"""
슬라이드 텍스트 반복 주제 키워드 추출

- 주제 키워드 반복: 슬라이드 텍스트화 결과(t1) 전체에서 반복 등장하는 내용어 기반

키워드 추출: kiwipiepy/Kiwi 형태소 분석 필수(명사만), 미설치 시 중단
주제 키워드 LLM 필터: Gemini로 강의 흐름과 무관한 후보 제거

max_keywords / min_freq 조정 (기본값 20/5):
  - get_topic_keywords_filtered_v2(..., min_freq=5, max_keywords=20)  함수 정의부 기본값
  - get_topic_keyword_count_map(..., min_freq=5, max_keywords=20)  함수 정의부 기본값
  - slide_textualizer.py의 get_topic_keyword_count_map(...) 호출 인자
  위 세 군데를 같은 값으로 맞춰서 변경
"""

import json
from collections import Counter
from typing import Optional, Set


# ---------------------------------------------------------------------------
# 주제 키워드 추출 (Kiwi 형태소 분석 필수)
# ---------------------------------------------------------------------------

_KIWI = None


# Kiwi 형태소 분석기 지연 초기화, 미설치 시 명확한 예외
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


# 키워드 후보에서 제외할 불용어 최소 집합
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

# 명사로 취급할 Kiwi 품사 태그
_NOUN_TAGS = ("NNG", "NNP", "SL")


# 연속된 명사 시퀀스에서 2~3어절 복합명사 후보 생성
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


# 텍스트에서 명사(+복합명사) 내용어 추출, 불용어/최소 길이 미만 제외
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


# 서로 포함관계인 후보 중 더 긴 대표 키워드만 남김
def _select_representative_keywords(candidates: set[str]) -> set[str]:
    if not candidates:
        return set()
    reps: list[str] = []
    for w in sorted(candidates, key=len, reverse=True):
        if any(w in r and w != r for r in reps):
            continue
        reps.append(w)
    return set(reps)


# LLM(Gemini)으로 강의 맥락과 무관한 키워드 후보 제거, 실패 시 원본 후보 그대로 반환
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
    남은 것 중 빈도 순 상위 max_keywords개를 반환
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
    주제 키워드별 전체 등장 횟수를 반환

    반환값은 최종 주제 키워드로 살아남은 단어만 포함
    개별 context/slide 집계에서는 같은 키워드가 여러 번 등장해도 이 total_count는 한 번만 더함
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
    """반복 키워드 count 내림차순으로 4개씩 5~1점 부여"""
    if not topic_count_map:
        return {}
    sorted_items = sorted(topic_count_map.items(), key=lambda item: (-int(item[1]), item[0]))
    result = {}
    for idx, (kw, _count) in enumerate(sorted_items[:20]):
        result[kw] = max(1, 5 - idx // 4)
    return result


def topic_keyword_count_items(topic_count_map: dict[str, int]) -> list[dict]:
    """JSON 저장용 [{keyword,total_count}] 목록으로 변환"""
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
    특정 text에 포함된 반복 키워드와 그 키워드들의 total_count 합을 반환

    같은 text 안에 같은 키워드가 여러 번 나와도 total_count는 한 번만 더함
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
