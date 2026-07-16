"""
텍스트 교정 엔진.

Kiwi pre-pass: 슬라이드 용어사전과 고신뢰 유사도 기반으로 명백한 ASR 오인식만 선교정
Pass 1: Gemini로 슬라이드 제목 + 용어 사전 기반 ASR 오인식 후보 생성
Pass 2: GPT로 슬라이드 전체 텍스트 컨텍스트 기반 후보 보강
Pass 3: GPT-5.4로 Pass 1/2 후보 중 적용할 후보만 선택
"""

import json
import difflib
import os
import re
import time
import base64
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .config import GEMINI_GENERATIVE_MODEL, gemini_client_2

load_dotenv()

GEMINI_MODEL = GEMINI_GENERATIVE_MODEL
PASS1_TEXT_MODEL = os.getenv("GRAPHLEC_TEXT_PROCESSOR_PASS1_MODEL", "gemini-3-flash-preview").strip()
PASS2_TEXT_MODEL = os.getenv("GRAPHLEC_TEXT_PROCESSOR_PASS2_MODEL", "gpt-5.4-mini").strip()
PASS3_TEXT_MODEL = os.getenv("GRAPHLEC_TEXT_PROCESSOR_PASS3_MODEL", "gpt-5.4").strip()
TEXT_REASONING_EFFORT = os.getenv("GRAPHLEC_TEXT_PROCESSOR_REASONING_EFFORT", "minimal").strip().lower()
PASS2_REASONING_EFFORT = os.getenv("GRAPHLEC_TEXT_PROCESSOR_PASS2_REASONING_EFFORT", "minimal").strip().lower()
PASS3_REASONING_EFFORT = os.getenv("GRAPHLEC_TEXT_PROCESSOR_PASS3_REASONING_EFFORT", "low").strip().lower()
TEXT_MAX_OUTPUT_TOKENS = int(os.getenv("GRAPHLEC_TEXT_PROCESSOR_TEXT_MAX_OUTPUT_TOKENS", "8192"))
PASS2_MAX_OUTPUT_TOKENS = int(os.getenv("GRAPHLEC_TEXT_PROCESSOR_PASS2_MAX_OUTPUT_TOKENS", "4096"))
PASS3_MAX_OUTPUT_TOKENS = int(os.getenv("GRAPHLEC_TEXT_PROCESSOR_PASS3_MAX_OUTPUT_TOKENS", "2048"))
PASS3_ITEM_BATCH_SIZE = max(1, int(os.getenv("GRAPHLEC_TEXT_PROCESSOR_PASS3_ITEM_BATCH_SIZE", "12")))
PASS2_FULL_SLIDE_CONTEXT = os.getenv("GRAPHLEC_TEXT_PROCESSOR_PASS2_FULL_SLIDE_CONTEXT", "1").strip() == "1"
IMAGE_PROVIDER = os.getenv("GRAPHLEC_TEXT_PROCESSOR_IMAGE_PROVIDER", "text").strip().lower()
IMAGE_MODEL = os.getenv("GRAPHLEC_TEXT_PROCESSOR_IMAGE_MODEL", "gpt-4.1-mini").strip()

BATCH_SIZE = int(os.getenv("MERGE_CORRECTION_BATCH_SIZE", "12"))
PARALLEL_REQUESTS = max(1, int(os.getenv("MERGE_CORRECTION_PARALLEL_REQUESTS", "8")))
TRANSITION_LEAD_SEC = float(os.getenv("MERGE_TRANSITION_LEAD_SEC", "1.0"))
TRANSITION_TAIL_SEC = float(os.getenv("MERGE_TRANSITION_TAIL_SEC", "0.2"))
ASSIGN_MAX_GAP_SEC = float(os.getenv("MERGE_ASSIGN_MAX_GAP_SEC", "3.0"))
DEBUG_CORRECTION_OUTPUT = os.getenv("MERGE_CORRECTION_DEBUG_OUTPUT", "0").strip() == "1"

# Slide terminology is evidence for the correction model, not a replacement
# dictionary.  Automatic substitutions can turn valid Korean lecture prose
# into mixed Korean/English text, so keep the pre-pass opt-in only.
KIWI_PREPASS_ENABLED = os.getenv("GRAPHLEC_TEXT_PROCESSOR_KIWI_PREPASS", "0").strip() == "1"
KIWI_GLOSSARY_ENABLED = os.getenv("GRAPHLEC_TEXT_PROCESSOR_KIWI_GLOSSARY", "1").strip() == "1"
KIWI_PREPASS_MIN_SCORE = float(os.getenv("GRAPHLEC_TEXT_PROCESSOR_KIWI_MIN_SCORE", "0.84"))
KIWI_PREPASS_MIN_MARGIN = float(os.getenv("GRAPHLEC_TEXT_PROCESSOR_KIWI_MIN_MARGIN", "0.08"))
KIWI_PREPASS_MAX_EDITS_PER_SEGMENT = max(
    1, int(os.getenv("GRAPHLEC_TEXT_PROCESSOR_KIWI_MAX_EDITS_PER_SEGMENT", "2"))
)
# Likewise, do not collapse Korean and English slide terms to one spelling by
# default.  Both forms remain available to the LLM as reference candidates.
G2P_NORMALIZATION_ENABLED = os.getenv("GRAPHLEC_TEXT_PROCESSOR_G2P_NORMALIZATION", "0").strip() == "1"
G2P_NORMALIZATION_MIN_SCORE = float(os.getenv("GRAPHLEC_TEXT_PROCESSOR_G2P_MIN_SCORE", "0.93"))
G2P_LEGACY_MIN_SCORE = float(os.getenv("GRAPHLEC_TEXT_PROCESSOR_G2P_LEGACY_MIN_SCORE", "0.50"))

DOMAIN_CHOICES = [
    "engineering",
    "natural_science",
    "humanities",
    "social_science",
    "arts",
    "health_sciences",
    "sports",
    "education",
    "etc",
]
SUBDOMAIN_CHOICES = [
    "computer_science", "electrical_engineering", "mechanical_engineering",
    "physics", "chemistry", "biology", "mathematics",
    "history", "philosophy", "literature",
    "economics", "political_science", "sociology",
    "nursing", "public_health", "sports_science", "education",
]

_token_usage: dict[str, int] = {"input": 0, "output": 0, "calls": 0}
_stage_token_usage: dict[str, dict[str, int]] = {}
_stage_counts: dict[str, int] = {
    "pass1_candidates": 0,
    "pass2_candidates": 0,
    "pass3_candidates": 0,
    "pass3_accepted": 0,
}
_pass3_candidate_audit: list[dict] = []
_token_usage_lock = Lock()
_override_client: Optional[genai.Client] = gemini_client_2
PASS3_REASON_CODES = {
    1: "asr_phonetic",
    2: "domain_term",
    3: "slide_term",
    4: "broken_phrase",
    5: "particle_repair",
    6: "english_or_code",
    7: "proper_noun",
}

def set_client(client: genai.Client) -> None:
    global _override_client
    _override_client = client


def _get_client() -> genai.Client:
    if _override_client is None:
        raise RuntimeError("Gemini client가 설정되지 않았습니다.")
    return _override_client


def _add_stage_count(key: str, value: int) -> None:
    with _token_usage_lock:
        _stage_counts[key] = int(_stage_counts.get(key, 0) or 0) + int(value or 0)


def _add_pass3_candidate_audit(records: list[dict]) -> None:
    if not records:
        return
    with _token_usage_lock:
        _pass3_candidate_audit.extend(records)


def _add_usage(response, stage: str = "stage3b_text_processor") -> None:
    usage = getattr(response, "usage_metadata", None)
    with _token_usage_lock:
        if usage:
            input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
            _token_usage["input"] += input_tokens
            _token_usage["output"] += output_tokens
            bucket = _stage_token_usage.setdefault(stage, {"input": 0, "output": 0, "calls": 0})
            bucket["input"] += input_tokens
            bucket["output"] += output_tokens
            bucket["calls"] += 1
        else:
            bucket = _stage_token_usage.setdefault(stage, {"input": 0, "output": 0, "calls": 0})
            bucket["calls"] += 1
        _token_usage["calls"] += 1
    try:
        from .cost_report import record_model_call

        record_model_call(
            stage=stage,
            provider="google",
            model=GEMINI_MODEL,
            response=response,
        )
    except Exception:
        pass


def _add_openai_usage(response, *, stage: str, model: str, prompt_chars: int = 0) -> None:
    usage = getattr(response, "usage", None)
    with _token_usage_lock:
        if usage:
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            _token_usage["input"] += input_tokens
            _token_usage["output"] += output_tokens
            bucket = _stage_token_usage.setdefault(stage, {"input": 0, "output": 0, "calls": 0})
            bucket["input"] += input_tokens
            bucket["output"] += output_tokens
            bucket["calls"] += 1
        else:
            bucket = _stage_token_usage.setdefault(stage, {"input": 0, "output": 0, "calls": 0})
            bucket["calls"] += 1
        _token_usage["calls"] += 1
    try:
        from .cost_report import record_model_call

        record_model_call(
            stage=stage,
            provider="openai",
            model=model,
            response=response,
            prompt_chars=prompt_chars,
        )
    except Exception:
        pass


def _call_openai_text_correction(
    prompt: str,
    *,
    stage: str,
    model: str,
    system_prompt: str,
    max_output_tokens: int = 8192,
) -> str:
    from .config import get_openai_client

    client = get_openai_client()
    if client is None and os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if client is None:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    def call():
        reasoning_effort = TEXT_REASONING_EFFORT
        if "pass2" in stage:
            reasoning_effort = PASS2_REASONING_EFFORT
        elif "pass3" in stage:
            reasoning_effort = PASS3_REASONING_EFFORT
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": max_output_tokens,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        temperature = os.getenv("GRAPHLEC_TEXT_PROCESSOR_OPENAI_TEMPERATURE", "0").strip()
        if temperature:
            kwargs["temperature"] = float(temperature)

        while True:
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as exc:
                message = str(exc).lower()
                removed = False
                if "temperature" in kwargs and "temperature" in message and (
                    "unsupported" in message or "not support" in message
                ):
                    kwargs.pop("temperature", None)
                    removed = True
                if "reasoning_effort" in kwargs and (
                    "reasoning" in message or "reasoning_effort" in message
                ) and ("unsupported" in message or "not support" in message):
                    kwargs.pop("reasoning_effort", None)
                    removed = True
                if removed:
                    continue
                raise

    response = api_call_with_retry(call)
    _add_openai_usage(response, stage=stage, model=model, prompt_chars=len(prompt))
    return response.choices[0].message.content or ""


def _call_openai_image_correction(prompt: str, image_bytes: bytes):
    from .config import get_openai_client
    from .utils import api_call_with_retry

    client = get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "high",
            },
        },
        {"type": "text", "text": prompt},
    ]

    def call():
        return client.chat.completions.create(
            model=IMAGE_MODEL,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            max_completion_tokens=8192,
            temperature=0,
        )

    response = api_call_with_retry(call)
    try:
        from .cost_report import record_model_call

        record_model_call(
            stage="stage3b_text_processor_pass2",
            provider="openai",
            model=IMAGE_MODEL,
            response=response,
            image_count=1,
            prompt_chars=len(prompt),
        )
    except Exception:
        pass
    return response.choices[0].message.content or ""


def api_call_with_retry(func, max_retries: int | None = None, initial_wait: int | None = None):
    if max_retries is None:
        max_retries = int(os.getenv("GRAPHLEC_TEXT_PROCESSOR_API_MAX_RETRIES", "0"))
    if initial_wait is None:
        initial_wait = int(os.getenv("GRAPHLEC_TEXT_PROCESSOR_API_INITIAL_WAIT", "10"))
    max_wait = float(os.getenv("GRAPHLEC_API_RETRY_MAX_WAIT_SEC", "60"))
    infinite = max_retries <= 0
    attempt = 0
    while infinite or attempt < max_retries:
        try:
            return func()
        except Exception as exc:
            attempt += 1
            err = str(exc)
            retryable = ["429", "503", "500", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded"]
            if any(code in err for code in retryable) and (infinite or attempt < max_retries):
                wait = min(initial_wait * attempt, max_wait)
                total = "∞" if infinite else str(max_retries - 1)
                print(f"  재시도 ({attempt}/{total}): {err[:60]}, {wait:g}초 대기")
                time.sleep(wait)
                continue
            raise
    raise Exception("API 호출 실패")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _applied(text: str, risk: str, reason: str) -> dict:
    return {
        "candidate_text": text,
        "applied_text": text,
        "risk": risk,
        "apply": True,
        "reason": reason,
    }


def classify_lecture_domain(slide_titles: list[str], transcript_sample: str) -> dict:
    titles_block = "\n".join(f"- {title}" for title in slide_titles[:10] if title)
    transcript_block = transcript_sample[:1500]

    prompt = f"""아래 강의 슬라이드 제목과 전사 내용을 보고 도메인과 서브도메인을 분류하세요.

## 슬라이드 제목
{titles_block}

## 전사 내용 (일부)
{transcript_block}

## 도메인 선택지
{", ".join(DOMAIN_CHOICES)}

## 서브도메인 선택지
{", ".join(SUBDOMAIN_CHOICES)}

위 선택지에서 가장 적합한 것을 하나씩 골라 JSON만 출력하세요.
판단이 애매하거나 선택지에 없으면 domain은 "etc", subdomain은 ""로 두세요.
{{"domain": "...", "subdomain": "..."}}"""

    def call():
        return _get_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=256,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

    try:
        response = api_call_with_retry(call)
        _add_usage(response, stage="stage3b_text_processor_domain")
        raw = (response.text or "").strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        parsed = json.loads(raw)
        domain = str(parsed.get("domain", "") or "").strip().lower().replace("-", "_")
        subdomain = str(parsed.get("subdomain", "") or "").strip().lower().replace("-", "_")
        if domain not in DOMAIN_CHOICES:
            domain = "etc"
        if subdomain not in SUBDOMAIN_CHOICES:
            subdomain = ""
        return {"domain": domain, "subdomain": subdomain}
    except Exception as exc:
        print(f"  [도메인 분류 오류] {exc}")
        return {"domain": "etc", "subdomain": ""}


def _build_occurrence_index(scene_occurrences: dict[int, list[dict]]) -> list[dict]:
    occ_index = []
    for scene_no, occs in scene_occurrences.items():
        for occ in occs:
            occ_index.append(
                {
                    "scene_no": int(scene_no),
                    "start_sec": float(occ["start_sec"]),
                    "end_sec": float(occ["end_sec"]),
                }
            )
    occ_index.sort(key=lambda item: (item["start_sec"], item["end_sec"], item["scene_no"]))
    return occ_index


def _assign_segment_occurrence(seg: dict, occ_index: list[dict]) -> Optional[int]:
    if not occ_index:
        return None

    seg_start = float(seg.get("start", 0) or 0)
    seg_end = float(seg.get("end", seg_start) or seg_start)
    if seg_end < seg_start:
        seg_end = seg_start

    overlaps: list[tuple[int, float]] = []
    for idx, occ in enumerate(occ_index):
        overlap = min(seg_end, occ["end_sec"]) - max(seg_start, occ["start_sec"])
        if overlap > 0:
            overlaps.append((idx, overlap))

    if overlaps:
        overlap_map = {idx: overlap for idx, overlap in overlaps}
        overlaps.sort(key=lambda item: (item[1], occ_index[item[0]]["start_sec"]), reverse=True)
        best_idx, best_overlap = overlaps[0]

        later_candidates = [
            idx
            for idx, _ in overlaps
            if occ_index[idx]["start_sec"] > seg_start and occ_index[idx]["start_sec"] <= seg_end
        ]
        if later_candidates:
            later_idx = min(later_candidates, key=lambda item: occ_index[item]["start_sec"])
            later_start = occ_index[later_idx]["start_sec"]
            later_overlap = overlap_map.get(later_idx, 0.0)
            started_near_boundary = seg_start >= (later_start - TRANSITION_LEAD_SEC)
            carried_after_boundary = (seg_end - later_start) >= TRANSITION_TAIL_SEC
            if started_near_boundary or later_overlap >= (best_overlap * 0.75) or carried_after_boundary:
                return later_idx

        return best_idx

    anchor = seg_start + (seg_end - seg_start) * 0.65
    best_idx = None
    best_dist = float("inf")
    for idx, occ in enumerate(occ_index):
        if occ["start_sec"] <= anchor <= occ["end_sec"]:
            return idx
        dist = min(abs(anchor - occ["start_sec"]), abs(anchor - occ["end_sec"]))
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    if best_idx is None or best_dist > ASSIGN_MAX_GAP_SEC:
        return None
    return best_idx


def parse_batch_response(text: str) -> dict[int, dict]:
    if not text:
        return {}
    raw = text.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}

    corrections: dict[int, dict] = {}
    for correction in parsed.get("corrections", []):
        if not isinstance(correction, dict):
            continue
        idx = correction.get("index")
        text_value = correction.get("text", "")
        decision = str(correction.get("decision", "") or "").strip().lower()
        risk = str(correction.get("risk", "") or "").strip().lower()
        reason = str(correction.get("reason", "") or "").strip()
        if isinstance(idx, int) and text_value:
            corrections[idx] = {
                "text": text_value,
                "decision": decision,
                "risk": risk,
                "reason": reason,
            }
    return corrections


def _extract_glossary_terms_rules(slide_texts: dict[int, dict]) -> list[str]:
    """Fallback extractor used when Kiwi is unavailable."""
    ordered_terms: dict[str, str] = {}
    predicate_endings = re.compile(
        r"(?:합니다|됩니다|있습니다|없습니다|구성됩니다|위치하며|관리하고|제어하고|사용합니다|부릅니다|수행합니다)[.!?。！？]?$"
    )

    def add_term(term: str, *, allow_sentence: bool = False) -> None:
        cleaned = re.sub(r"\s+", " ", str(term or "")).strip()
        cleaned = re.sub(r"^[\d\s\.\)\]\-•●○■□▶]+", "", cleaned).strip()
        cleaned = cleaned.strip(" \t\r\n,.;:()[]{}<>\"'")
        if not cleaned or len(cleaned) < 2 or len(cleaned) > 40:
            return
        if not allow_sentence and predicate_endings.search(cleaned):
            return
        if not allow_sentence and re.search(
            r"[가-힣](?:은|는|이|가|을|를|에|에서|으로|에게|와|과|도)\s",
            cleaned,
        ):
            return
        if len(cleaned.split()) > 6:
            return
        key = _glossary_key(cleaned)
        if key and key not in ordered_terms:
            ordered_terms[key] = cleaned

    for _, extracted in slide_texts.items():
        title = str(extracted.get("title", "") or "")
        body = str(extracted.get("glossary_text", extracted.get("t1", extracted.get("text", ""))) or "")
        add_term(title, allow_sentence=True)
        for raw_line in re.split(r"[\n\r]+", body):
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue

            # Parenthesized labels are explicit slide terminology.
            for parenthetical in re.findall(r"\(([^()]{1,50})\)", line):
                add_term(parenthetical, allow_sentence=True)

            # Keep independent bullet/label fragments, not the whole explanation.
            parts = re.split(r"\s*[-–—:/]\s*", line)
            for part in parts:
                add_term(part)
            if len(parts) == 1:
                add_term(line)

            # Preserve acronyms, code, and English labels shown on the slide.
            for english_term in re.findall(r"\b[A-Za-z][A-Za-z0-9_+#./-]*\b", line):
                add_term(english_term, allow_sentence=True)

    max_terms = int(os.getenv("GRAPHLEC_TEXT_PROCESSOR_MAX_CANONICAL_TERMS", "300"))
    return list(ordered_terms.values())[:max_terms]


def extract_glossary_term_records(slide_texts: dict[int, dict]) -> list[dict]:
    """Extract glossary candidates and retain their source slide/scene IDs."""
    if not KIWI_GLOSSARY_ENABLED:
        return [
            {"term": term, "source_slides": [], "source_scenes": []}
            for term in _extract_glossary_terms_rules(slide_texts)
        ]
    try:
        from kiwipiepy import Kiwi
    except Exception as exc:
        print(f"    Kiwi 용어 추출기 비활성화: kiwipiepy 로드 실패 ({exc})")
        return [
            {"term": term, "source_slides": [], "source_scenes": []}
            for term in _extract_glossary_terms_rules(slide_texts)
        ]

    kiwi = Kiwi()
    keep_tags = {"NNG", "NNP", "NNB", "NP", "SL", "SN", "XR"}
    stop_terms = {
        "것", "수", "등", "등등", "및", "나", "또", "이", "그", "저", "여러", "예",
        "예시", "방법", "경우", "부분", "모양", "때문", "관련", "통해", "대해", "위해",
    }
    token_stop_forms = stop_terms | {"어로", "시", "때"}
    single_stop_terms = {
        "관리", "운영", "체제", "자원", "컴퓨터", "소프트웨어", "하드웨어", "프로그램",
        "사용", "처리", "기능", "목적", "정의", "개념", "발전", "과정", "시작", "이해",
        "강의", "단어", "목표", "핵심", "차이", "역할", "처음", "이후", "시절", "발단",
        "원시", "출현", "영향", "종류", "특징", "사이", "중계", "실체", "나머지", "특별",
        "다양", "언어", "개발", "실행", "저장", "생성", "요청", "적재", "할당", "위치",
        "관리자", "보안", "기타", "통계", "오류", "대응", "보호", "종료", "편리", "효율",
        "접속", "시간", "외부", "침입", "반환", "삭제", "속성", "정보", "달성", "공통",
        "서비스", "제공", "요구", "충족", "설계", "특정", "수행", "명령", "배타",
    }
    predicate_endings = re.compile(
        r"(?:합니다|됩니다|있습니다|없습니다|구성됩니다|위치하며|관리하고|제어하고|사용합니다|부릅니다|수행합니다)[.!?。！？]?$"
    )
    records: dict[str, dict] = {}
    sequence = 0
    current_source_slide: Optional[int] = None
    current_source_scenes: set[int] = set()

    def add_term(
        term: str,
        *,
        priority: int = 1,
        explicit: bool = False,
        single: bool = False,
    ) -> None:
        nonlocal sequence
        cleaned = re.sub(r"\s+", " ", str(term or "")).strip()
        cleaned = re.sub(r"^[\d\s\.\)\]\-•●○■□▶]+", "", cleaned).strip()
        cleaned = cleaned.strip(" \t\r\n,.;:()[]{}<>\"'")
        cleaned = cleaned.strip("/")
        if not cleaned or len(cleaned) < 2 or len(cleaned) > 40:
            return
        if cleaned == "C/C":
            return
        if len(cleaned) == 1:
            return
        if predicate_endings.search(cleaned):
            return
        if len(cleaned.split()) > 5:
            return
        if re.search(r"[가-힣](?:은|는|이|가|을|를|에|에서|으로|에게|와|과|도)\s", cleaned):
            return
        if any(word in stop_terms for word in cleaned.split()):
            return
        key = _glossary_key(cleaned)
        if not key:
            return
        existing = records.get(key)
        if existing is None:
            records[key] = {
                "term": cleaned,
                "priority": priority + (2 if explicit else 0),
                "frequency": 1,
                "sequence": sequence,
                "single": single,
                "explicit": explicit,
                "source_slides": ([current_source_slide] if isinstance(current_source_slide, int) else []),
                "source_scenes": sorted(current_source_scenes),
            }
            sequence += 1
        else:
            existing["frequency"] += 1
            existing["priority"] = max(existing["priority"], priority + (2 if explicit else 0))
            existing["single"] = bool(existing["single"] and single)
            existing["explicit"] = bool(existing["explicit"] or explicit)
            if isinstance(current_source_slide, int) and current_source_slide not in existing["source_slides"]:
                existing["source_slides"].append(current_source_slide)
            existing["source_scenes"] = sorted(
                set(existing.get("source_scenes", [])) | current_source_scenes
            )

    def add_token_spans(text: str, *, priority: int = 1, explicit: bool = False) -> None:
        if not text.strip():
            return
        tokens = kiwi.tokenize(text)
        lexical = [
            token for token in tokens
            if token.len > 0 and str(token.tag) in keep_tags
        ]
        for token_index, token in enumerate(tokens):
            if token.len <= 0 or str(token.tag) not in keep_tags:
                continue
            form = text[token.start:token.start + token.len]
            if form in token_stop_forms:
                continue
            if re.fullmatch(r"[가-힣]", form):
                continue
            adjacent_compound = False
            for neighbor_index in (token_index - 1, token_index + 1):
                if not 0 <= neighbor_index < len(tokens):
                    continue
                neighbor = tokens[neighbor_index]
                gap_start = min(token.start + token.len, neighbor.start + neighbor.len)
                gap_end = max(token.start, neighbor.start)
                if str(neighbor.tag) in keep_tags and not text[gap_start:gap_end]:
                    adjacent_compound = True
                    break
            if not adjacent_compound:
                add_term(form, priority=priority, explicit=explicit, single=True)

        # Add short contiguous noun/foreign-token compounds, but stop at
        # particles, conjunctions, and other non-lexical tokens.
        for start, token in enumerate(tokens):
            if token.len <= 0 or str(token.tag) not in keep_tags:
                continue
            end = start
            while end + 1 < len(tokens):
                current = tokens[end]
                following = tokens[end + 1]
                gap = text[current.start + current.len:following.start]
                if str(following.tag) not in keep_tags:
                    break
                if gap and (not gap.isspace() or not explicit):
                    break
                if end - start >= 1:
                    break
                next_token = tokens[end + 1]
                if any(
                    text[item.start:item.start + item.len] in token_stop_forms
                    or re.fullmatch(r"[가-힣]", text[item.start:item.start + item.len])
                    for item in (token, next_token)
                ):
                    break
                end += 1
                candidate = text[token.start:tokens[end].start + tokens[end].len]
                add_term(candidate, priority=priority + 1, explicit=explicit)

    for slide_no, extracted in slide_texts.items():
        current_source_slide = slide_no if isinstance(slide_no, int) else None
        current_source_scenes = {
            int(scene_no)
            for scene_no in extracted.get("scene_numbers", [])
            if isinstance(scene_no, int)
        }
        title = str(extracted.get("title", "") or "")
        body = str(extracted.get("glossary_text", extracted.get("t1", extracted.get("text", ""))) or "")
        if title and not re.search(r"그리고|및", title):
            add_term(title, priority=3, explicit=True)
        add_token_spans(title, priority=3, explicit=False)

        for raw_line in re.split(r"[\n\r]+", body):
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            for parenthetical in re.findall(r"\(([^()]{1,60})\)", line):
                add_token_spans(parenthetical, priority=3, explicit=True)
                for english_term in re.findall(r"\b[A-Za-z][A-Za-z0-9_+#./-]*\b", parenthetical):
                    add_term(english_term, priority=3, explicit=True)

            # Labels around a colon or bullet are stronger glossary evidence.
            parts = re.split(r"\s*[-–—:/]\s*", line)
            line_is_label = len(line.split()) <= 4 and not predicate_endings.search(line)
            for part in parts:
                add_token_spans(part, priority=2, explicit=line_is_label or len(parts) > 1)
            add_token_spans(line, priority=1)

            # Kiwi may classify symbols such as C++ as punctuation; retain
            # explicit Latin/code labels separately.
            for english_term in re.findall(r"\b[A-Za-z][A-Za-z0-9_+#./-]*\b", line):
                add_term(english_term, priority=2, explicit=True)

    max_terms = int(os.getenv("GRAPHLEC_TEXT_PROCESSOR_MAX_CANONICAL_TERMS", "300"))
    ranked = sorted(
        (
            item for item in records.values()
            if not (
                item["single"]
                and item["frequency"] < 2
                and not item["explicit"]
            )
            and not (
                item["single"]
                and item["term"] in single_stop_terms
            )
        ),
        key=lambda item: (-int(item["priority"]), -int(item["frequency"]), int(item["sequence"])),
    )
    extracted_records = [
        {
            "term": item["term"],
            "source_slides": sorted(set(item.get("source_slides", []))),
            "source_scenes": sorted(set(item.get("source_scenes", []))),
            "frequency": int(item.get("frequency", 0) or 0),
        }
        for item in ranked[:max_terms]
    ]
    return _canonicalize_glossary_term_records(extracted_records)


def extract_glossary_terms(slide_texts: dict[int, dict]) -> list[str]:
    """Compatibility wrapper returning only canonical term strings."""
    return [record["term"] for record in extract_glossary_term_records(slide_texts)]


def _kiwi_similarity_key(value: str) -> str:
    """Return a comparison key that keeps Hangul phoneme components visible."""
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[^0-9a-zA-Z가-힣ᄀ-ᇿ]", "", value)
    return unicodedata.normalize("NFD", value)


_HANGUL_INITIAL_SOUND = ("g", "k", "n", "d", "t", "r", "m", "b", "p", "s", "s", "", "j", "j", "ch", "k", "t", "p", "h")
_HANGUL_FINAL_SOUND = ("", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "p", "l", "l", "l", "l", "l", "p", "p", "t", "t", "ng", "t", "ch", "k", "t", "p", "h")
_HANGUL_VOWEL_SOUND = (
    "a", "e", "ya", "ye", "eo", "e", "yeo", "ye", "o", "wa", "we", "we", "yo",
    "u", "wo", "we", "wi", "yu", "eu", "ui", "i",
)
_ARPABET_SOUND = {
    "AA": "a", "AE": "e", "AH": "eo", "AO": "o", "AW": "au", "AY": "ai",
    "B": "p", "CH": "ch", "D": "t", "DH": "d", "EH": "e", "ER": "eo",
    "EY": "ei", "F": "p", "G": "g", "HH": "h", "IH": "i", "IY": "i",
    "JH": "j", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ng",
    "OW": "o", "OY": "oi", "P": "p", "R": "r", "S": "s", "SH": "s",
    "T": "t", "TH": "s", "UH": "u", "UW": "u", "V": "p", "W": "w",
    "Y": "y", "Z": "s", "ZH": "j",
}


def _hangul_consonant_key(value: str) -> str:
    """한글식 외래어와 영문 용어의 보수적 발음 비교용 자음 골격."""
    parts: list[str] = []
    for char in str(value or ""):
        code = ord(char) - 0xAC00
        if 0 <= code < 11172:
            initial = code // 588
            final = code % 28
            parts.append(_HANGUL_INITIAL_SOUND[initial])
            parts.append(_HANGUL_FINAL_SOUND[final])
    return re.sub(r"(.)\1+", r"\1", "".join(parts))


def _latin_consonant_key(value: str) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"ph", "f", text)
    text = re.sub(r"c(?=[eiy])", "s", text)
    text = text.replace("ck", "k").replace("qu", "k").replace("x", "ks")
    text = re.sub(r"[^a-z]", "", text)
    text = re.sub(r"[aeiouy]", "", text)
    return re.sub(r"(.)\1+", r"\1", text)


@lru_cache(maxsize=1)
def _korean_g2p():
    """Return KSS's maintained Korean G2P callable, or None when unavailable."""
    if not G2P_NORMALIZATION_ENABLED:
        return None
    try:
        from kss import Kss

        return Kss("g2p")
    except Exception as exc:
        print(f"    G2P 정규화 비활성화: KSS 로드 실패 ({exc})")
        return None


@lru_cache(maxsize=2048)
def _hangul_g2p_key(value: str) -> str:
    g2p = _korean_g2p()
    if g2p is None:
        return ""
    try:
        pronounced = str(g2p(value, descriptive=True, group_vowels=True, num_workers=1))
    except Exception:
        return ""
    parts: list[str] = []
    for char in pronounced:
        code = ord(char) - 0xAC00
        if not 0 <= code < 11172:
            continue
        initial = code // 588
        vowel = (code % 588) // 28
        final = code % 28
        parts.extend((_HANGUL_INITIAL_SOUND[initial], _HANGUL_VOWEL_SOUND[vowel], _HANGUL_FINAL_SOUND[final]))
    return "".join(parts)


@lru_cache(maxsize=2048)
def _english_g2p_keys(value: str) -> tuple[str, ...]:
    if not re.fullmatch(r"[A-Za-z]+", str(value or "")):
        return ()
    try:
        import pronouncing

        pronunciations = pronouncing.phones_for_word(value.lower())
    except Exception:
        return ()
    keys: list[str] = []
    for pronunciation in pronunciations:
        key = "".join(
            _ARPABET_SOUND.get(re.sub(r"\d", "", phone), "")
            for phone in pronunciation.split()
        )
        if key and key not in keys:
            keys.append(key)
    return tuple(keys)


def _g2p_similarity_score(source: str, target: str) -> float:
    """Compare Korean and English terms after each is converted to pronunciation."""
    if len(re.findall(r"[가-힣]", source)) < 2:
        return 0.0
    source_key = _hangul_g2p_key(source)
    target_keys = _english_g2p_keys(target)
    if len(source_key) < 3 or not target_keys:
        return 0.0

    variants = {source_key}
    # English words that end in a consonant are commonly written with a final
    # epenthetic vowel in Korean (cache→캐시). Compare that form as well.
    if source_key[-1:] in {"i", "u", "e"}:
        variants.add(source_key[:-1])
    return max(
        difflib.SequenceMatcher(None, source_variant, target_key).ratio()
        for source_variant in variants
        for target_key in target_keys
    )


def _cross_script_phonetic_score(source: str, target: str) -> float:
    # Two-syllable Korean common nouns (e.g. 머리, 위치, 바탕) collide too
    # easily with English consonant skeletons. Keep this auto-normalization
    # deliberately conservative; shorter terms remain available to later
    # context-aware correction rather than being rewritten here.
    g2p_score = _g2p_similarity_score(source, target)
    if g2p_score >= G2P_NORMALIZATION_MIN_SCORE:
        return g2p_score
    # The older consonant skeleton is retained only for 3+ syllable terms;
    # it is useful for process/프로세스 but unsafe for short common nouns.
    if len(re.findall(r"[가-힣]", source)) < 3:
        return 0.0
    # A consonant-only match can still rescue longer transliterations such as
    # 프로세서→process, but it must agree at least loosely with real G2P.
    if g2p_score < G2P_LEGACY_MIN_SCORE:
        return 0.0
    hangul_key = _hangul_consonant_key(source)
    latin_key = _latin_consonant_key(target)
    if len(hangul_key) < 2 or len(latin_key) < 2:
        return 0.0
    return difflib.SequenceMatcher(None, hangul_key, latin_key).ratio()


def _canonicalize_glossary_term_records(records: list[dict]) -> list[dict]:
    """한글식 발음 용어가 확실히 같은 영문 용어면 영문 표기로 사전을 통일한다."""
    english = [item for item in records if re.search(r"[A-Za-z]", str(item.get("term", ""))) and not re.search(r"[가-힣]", str(item.get("term", "")))]
    if not english:
        return records
    canonical_by_term = {str(item.get("term", "")): dict(item) for item in english}
    retained: list[dict] = []
    for item in records:
        term = str(item.get("term", ""))
        if not re.search(r"[가-힣]", term):
            continue
        ranked = sorted(
            ((_cross_script_phonetic_score(term, str(candidate.get("term", ""))), candidate) for candidate in english),
            key=lambda pair: pair[0], reverse=True,
        )
        best_score, best = ranked[0] if ranked else (0.0, None)
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        if best is not None and best_score >= 0.92 and best_score - second_score >= 0.08:
            canonical = canonical_by_term[str(best.get("term", ""))]
            canonical["source_slides"] = sorted(set(canonical.get("source_slides", [])) | set(item.get("source_slides", [])))
            canonical["source_scenes"] = sorted(set(canonical.get("source_scenes", [])) | set(item.get("source_scenes", [])))
            canonical["frequency"] = int(canonical.get("frequency", 0) or 0) + int(item.get("frequency", 0) or 0)
        else:
            retained.append(item)
    english_terms = {str(item.get("term", "")) for item in english}
    ordered = [canonical_by_term[str(item.get("term", ""))] for item in records if str(item.get("term", "")) in english_terms]
    seen: set[str] = set()
    return [item for item in ordered + retained if not (str(item.get("term", "")) in seen or seen.add(str(item.get("term", ""))))]


def _kiwi_candidate_score(source: str, target: str) -> float:
    if bool(re.search(r"[가-힣]", source)) != bool(re.search(r"[가-힣]", target)):
        return _cross_script_phonetic_score(source, target)
    source_key = _kiwi_similarity_key(source)
    target_key = _kiwi_similarity_key(target)
    if not source_key or not target_key:
        return 0.0
    if source_key == target_key:
        return 1.0

    if not re.search(r"[가-힣]", source) and not re.search(r"[가-힣]", target):
        latin_source = re.sub(r"(.)\1+", r"\1", source_key.replace("y", "i"))
        latin_target = re.sub(r"(.)\1+", r"\1", target_key.replace("y", "i"))
        if latin_source == latin_target:
            return 1.0

    syllable_score = difflib.SequenceMatcher(
        None,
        unicodedata.normalize("NFKC", source).casefold().replace(" ", ""),
        unicodedata.normalize("NFKC", target).casefold().replace(" ", ""),
    ).ratio()
    phoneme_score = difflib.SequenceMatcher(None, source_key, target_key).ratio()
    length_ratio = min(len(source_key), len(target_key)) / max(len(source_key), len(target_key))
    return (0.45 * syllable_score) + (0.40 * phoneme_score) + (0.15 * length_ratio)


def _kiwi_is_comparable(source: str, target: str) -> bool:
    source_key = _kiwi_similarity_key(source)
    target_key = _kiwi_similarity_key(target)
    if not source_key or not target_key:
        return False
    source_hangul = bool(re.search(r"[가-힣]", source))
    target_hangul = bool(re.search(r"[가-힣]", target))
    source_latin = bool(re.search(r"[A-Za-z]", source))
    target_latin = bool(re.search(r"[A-Za-z]", target))
    if source_hangul != target_hangul:
        return _cross_script_phonetic_score(source, target) >= 0.92
    if target_latin and not source_latin:
        return False
    # Pre-pass is allowed to restore a typo, not append a new modifier or
    # predicate. A target may be at most one decomposed character longer.
    if len(target_key) - len(source_key) > 1:
        return False
    if abs(len(source_key) - len(target_key)) > (3 if source_hangul else 4):
        return False
    return True


def _kiwi_prepass_text(
    text: str,
    terms: list[str],
    kiwi,
) -> dict:
    """Apply only high-confidence glossary substitutions before the LLM passes."""
    original = str(text or "")
    if not original or not terms:
        return {"text": original, "edits": [], "candidates": []}

    # Tokenization is intentionally done once per segment. Candidate spans use
    # Kiwi's morpheme boundaries, so a particle attached to an ASR error (e.g.
    # ``운영체재를``) is preserved while only the damaged noun is replaced.
    kiwi_tokens = kiwi.tokenize(original)
    term_by_key = {
        _glossary_key(term): str(term).strip()
        for term in terms
        if str(term or "").strip()
    }
    term_values = list(term_by_key.values())
    if not term_values:
        return {"text": original, "edits": [], "candidates": []}

    lexical_tags = {"NNG", "NNP", "NNB", "NP", "SL", "SN", "XR"}
    token_spans: list[tuple[int, int]] = []
    for token_index, token in enumerate(kiwi_tokens):
        if token.len <= 0 or str(token.tag) not in lexical_tags:
            continue
        span_start = token.start
        span_end = token.start + token.len
        next_index = token_index + 1
        while next_index < len(kiwi_tokens):
            next_token = kiwi_tokens[next_index]
            next_start = next_token.start
            if next_start != span_end or str(next_token.tag) not in lexical_tags:
                break
            span_end = next_start + next_token.len
            next_index += 1
        token_spans.append((span_start, span_end))
    candidate_spans: list[tuple[int, int]] = list(token_spans)
    if not candidate_spans:
        return {"text": original, "edits": [], "candidates": []}

    candidates: list[dict] = []
    seen_spans: set[tuple[int, int]] = set()
    for span_start, span_end in candidate_spans:
        if (span_start, span_end) in seen_spans:
            continue
        seen_spans.add((span_start, span_end))
        source = original[span_start:span_end].strip(" ,.;:!?。！？()[]{}\"'")
        if len(source) < 2:
            continue
        source_key = _glossary_key(source)
        if not source_key or source_key in term_by_key:
            continue

        source_word_count = len(source.split())
        ranked: list[tuple[float, str]] = []
        for target in term_values:
            if abs(len(target.split()) - source_word_count) > 1:
                continue
            if not _kiwi_is_comparable(source, target):
                continue
            score = _kiwi_candidate_score(source, target)
            if score >= KIWI_PREPASS_MIN_SCORE:
                ranked.append((score, target))
        if not ranked:
            continue
        ranked.sort(reverse=True, key=lambda item: (item[0], len(item[1])))
        best_score, best_target = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        if len(ranked) > 1 and best_score - second_score < KIWI_PREPASS_MIN_MARGIN:
            continue
        if _normalize_text(source) == _normalize_text(best_target):
            continue
        candidates.append(
            {
                "from_text": source,
                "to_text": best_target,
                "start": span_start,
                "end": span_end,
                "score": round(best_score, 4),
                "second_score": round(second_score, 4),
                "method": "kiwi_glossary_similarity",
            }
        )

    # Prefer longer spans and higher scores, then discard overlaps.
    selected: list[dict] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (item["end"] - item["start"], item["score"]),
        reverse=True,
    ):
        if any(
            candidate["start"] < selected_item["end"]
            and selected_item["start"] < candidate["end"]
            for selected_item in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= KIWI_PREPASS_MAX_EDITS_PER_SEGMENT:
            break

    corrected = original
    for edit in sorted(selected, key=lambda item: item["start"], reverse=True):
        corrected = corrected[:edit["start"]] + edit["to_text"] + corrected[edit["end"]:]
        edit["status"] = "applied"
    selected.sort(key=lambda item: item["start"])
    return {"text": corrected, "edits": selected, "candidates": candidates}


def _apply_kiwi_prepass(
    segments: list[dict],
    seg_logical_slide: dict[int, int],
    slide_texts: dict[int, dict],
    all_terms: list[str],
) -> list[dict]:
    if not KIWI_PREPASS_ENABLED:
        return [seg.copy() for seg in segments]
    try:
        from kiwipiepy import Kiwi
    except Exception as exc:
        print(f"    Kiwi pre-pass 비활성화: kiwipiepy 로드 실패 ({exc})")
        return [seg.copy() for seg in segments]

    started_at = time.perf_counter()
    kiwi = Kiwi()
    for term in all_terms:
        try:
            kiwi.add_user_word(str(term), "NNP")
        except Exception:
            continue

    local_terms_by_slide: dict[int, list[str]] = {}
    for slide_no in slide_texts:
        local_terms_by_slide[slide_no] = extract_glossary_terms(
            {
                sno: slide_texts[sno]
                for sno in range(slide_no - 1, slide_no + 2)
                if sno in slide_texts
            }
        )

    result: list[dict] = []
    applied_count = 0
    candidate_count = 0
    for index, segment in enumerate(segments):
        current = segment.copy()
        original = str(segment.get("text_original", segment.get("text", "")) or "")
        slide_no = seg_logical_slide.get(index)
        terms = local_terms_by_slide.get(slide_no, []) if isinstance(slide_no, int) else []
        if not terms:
            terms = all_terms
        prepass = _kiwi_prepass_text(original, terms, kiwi)
        prepass_text = prepass["text"]
        current["text_original_asr"] = original
        current["text_prepass"] = prepass_text
        current["text_original"] = prepass_text
        current["text"] = prepass_text
        current["kiwi_prepass_candidates"] = prepass["candidates"]
        current["kiwi_prepass_edits"] = prepass["edits"]
        if prepass["edits"]:
            applied_count += len(prepass["edits"])
        candidate_count += len(prepass["candidates"])
        result.append(current)

    print(
        f"    Kiwi pre-pass: 후보 {candidate_count}개, 적용 {applied_count}개 "
        f"(임계값 {KIWI_PREPASS_MIN_SCORE:.2f}, {time.perf_counter() - started_at:.3f}초)"
    )
    return result


def build_local_glossary(
    slide_texts: dict[int, dict],
    slide_no: Optional[int],
    *,
    window: int = 1,
) -> str:
    if not isinstance(slide_no, int):
        return ""
    local_slide_texts = {
        sno: slide_texts[sno]
        for sno in range(slide_no - window, slide_no + window + 1)
        if sno in slide_texts
    }
    terms = extract_glossary_terms(local_slide_texts)
    if not terms:
        return ""
    return "## 현재/인접 슬라이드 용어 사전 (표기 참조용)\n" + ", ".join(terms)


def _glossary_key(value: str) -> str:
    value = str(value or "").casefold()
    return re.sub(r"[\s\.,;:!?。．，、·()\[\]{}<>\"'`]+", "", value)


def _batch_text_chunks(batch: list[tuple[int, dict]]) -> list[str]:
    chunks: list[str] = []
    for _, segment in batch:
        text = str(segment.get("text_original", segment.get("text", "")) or "")
        words = text.split()
        for size in range(1, min(4, len(words)) + 1):
            chunks.extend(" ".join(words[start:start + size]) for start in range(len(words) - size + 1))
    return chunks


def build_batch_glossary(
    batch: list[tuple[int, dict]],
    slide_texts: dict[int, dict],
    slide_no: Optional[int],
    all_terms: list[str],
    *,
    window: int = 1,
    term_records: Optional[list[dict]] = None,
    scene_indices: Optional[set[int]] = None,
) -> str:
    """Retrieve only terms relevant to one transcript batch.

    The full slide vocabulary is built once, then filtered locally without an
    additional model call. Exact matches and close spelling/ASR matches win;
    nearby-slide terms provide context for Korean/English transliterations
    that cannot be matched by character similarity alone.
    """
    max_terms = max(1, int(os.getenv("GRAPHLEC_TEXT_PROCESSOR_BATCH_GLOSSARY_TERMS", "20")))
    fuzzy_threshold = float(os.getenv("GRAPHLEC_TEXT_PROCESSOR_GLOSSARY_FUZZY_THRESHOLD", "0.78"))
    latin_fuzzy_threshold = float(
        os.getenv("GRAPHLEC_TEXT_PROCESSOR_GLOSSARY_LATIN_FUZZY_THRESHOLD", "0.60")
    )
    batch_text = " ".join(
        str(segment.get("text_original", segment.get("text", "")) or "")
        for _, segment in batch
    )
    batch_key = _glossary_key(batch_text)
    chunks = [_glossary_key(chunk) for chunk in _batch_text_chunks(batch) if _glossary_key(chunk)]

    if term_records is None:
        term_records = extract_glossary_term_records(slide_texts)
    record_by_key = {
        _glossary_key(record.get("term", "")): record
        for record in term_records
        if _glossary_key(record.get("term", ""))
    }
    if not all_terms:
        all_terms = [str(record.get("term", "")) for record in term_records]

    local_records: list[dict] = []
    if isinstance(slide_no, int):
        local_slide_numbers = set(range(slide_no - window, slide_no + window + 1))
        local_records = [
            record for record in term_records
            if local_slide_numbers.intersection(
                int(value) for value in record.get("source_slides", [])
                if isinstance(value, int)
            )
        ]
    local_keys = {_glossary_key(record.get("term", "")) for record in local_records}

    scored: dict[str, tuple[float, dict]] = {}
    for term in all_terms:
        key = _glossary_key(term)
        if len(key) < 2:
            continue
        score = 0.0
        if key and key in batch_key:
            score = 100.0
        elif len(key) >= 3 and chunks:
            best_ratio = max(
                difflib.SequenceMatcher(None, key, chunk).ratio()
                for chunk in chunks
                if abs(len(key) - len(chunk)) <= max(4, int(len(key) * 0.5))
            ) if any(abs(len(key) - len(chunk)) <= max(4, int(len(key) * 0.5)) for chunk in chunks) else 0.0
            threshold = latin_fuzzy_threshold if re.search(r"[a-z]", key) else fuzzy_threshold
            if best_ratio >= threshold:
                score = best_ratio * 10.0
        if key in local_keys:
            score += 2.0
        record = record_by_key.get(key, {"term": term, "source_slides": [], "source_scenes": []})
        source_scenes = {
            int(value) for value in record.get("source_scenes", []) if isinstance(value, int)
        }
        if scene_indices and source_scenes.intersection(scene_indices):
            score += 4.0
        if score > 0:
            previous = scored.get(key)
            previous_term = str(previous[1].get("term", "")) if previous else ""
            if previous is None or score > previous[0] or (score == previous[0] and len(term) > len(previous_term)):
                scored[key] = (score, record)

    selected_records = [
        record for _, record in sorted(
            scored.values(),
            key=lambda item: (
                -item[0],
                -len(str(item[1].get("term", ""))),
                str(item[1].get("term", "")),
            ),
        )[:max_terms]
    ]
    if not selected_records:
        # Keep a small local context when the ASR text has no lexical overlap.
        selected_records = local_records[:max_terms]
    if not selected_records:
        return ""

    def format_record(record: dict) -> str:
        term = str(record.get("term", "") or "")
        sources: list[str] = []
        source_slides = sorted({int(value) for value in record.get("source_slides", []) if isinstance(value, int)})
        source_scenes = sorted({int(value) for value in record.get("source_scenes", []) if isinstance(value, int)})
        if source_slides:
            sources.append("slide " + "/".join(str(value) for value in source_slides))
        if source_scenes:
            sources.append("scene " + "/".join(str(value) for value in source_scenes))
        return f"{term} ({'; '.join(sources)})" if sources else term

    return "## 배치 관련 용어 사전 (표기 참조용)\n" + ", ".join(
        format_record(record) for record in selected_records
    )


def _load_slide_occurrences_from_metadata(metadata: list[dict]) -> tuple[dict[int, list[dict]], dict[int, int]]:
    scene_occurrences: dict[int, list[dict]] = {}
    scene_to_slide_no: dict[int, int] = {}
    seen: set[int] = set()

    for entry in metadata:
        scene_no = entry.get("scene_index")
        logical_slide_no = entry.get("slide_number")
        start_sec = entry.get("scene_start_sec")
        end_sec = entry.get("scene_end_sec")
        if not isinstance(scene_no, int) or scene_no in seen:
            continue
        if start_sec is None or end_sec is None:
            continue
        start = float(start_sec)
        end = float(end_sec)
        if end < start:
            end = start
        scene_occurrences[scene_no] = [{
            "start_sec": start,
            "end_sec": end,
            "duration": round(end - start, 3),
            "is_dup": False,
        }]
        if isinstance(logical_slide_no, int):
            scene_to_slide_no[scene_no] = logical_slide_no
        seen.add(scene_no)

    return scene_occurrences, scene_to_slide_no


def _build_scene_metadata_index(metadata: list[dict]) -> dict[int, dict]:
    scene_meta: dict[int, dict] = {}
    for entry in metadata:
        if entry.get("capture_type") != "base" and int(entry.get("annot_index", 0) or 0) != 0:
            continue
        scene_idx = entry.get("scene_index")
        if not isinstance(scene_idx, int) or scene_idx in scene_meta:
            continue
        slide_number = entry.get("slide_number", entry.get("slide_canonical_index"))
        slide_canonical_index = entry.get(
            "slide_canonical_index",
            entry.get("same_slide_canonical"),
        )
        scene_meta[scene_idx] = {
            "scene_index": scene_idx,
            "slide_number": slide_number if isinstance(slide_number, int) else None,
            "slide_canonical_index": (
                slide_canonical_index if isinstance(slide_canonical_index, int) else None
            ),
            "slide_visit_order": int(entry.get("slide_visit_order", entry.get("same_slide_visit_order", 1)) or 1),
            "slide_is_revisit": bool(
                entry.get("slide_is_revisit", entry.get("same_slide_is_revisit", False))
            ),
        }
    return scene_meta


def _load_integrated_slide_texts(
    integrated_data: dict,
    base_dir: Path,
    scene_meta_by_index: Optional[dict[int, dict]] = None,
) -> dict[int, dict]:
    result: dict[int, dict] = {}
    slide_rows = integrated_data.get("scenes") or integrated_data.get("slides") or []
    for slide in slide_rows:
        scene_no = slide.get(
            "scene_number",
            slide.get("representative_scene_number", slide.get("slide_number")),
        )
        slide_no = slide.get("slide_number")
        if scene_meta_by_index and isinstance(scene_no, int):
            logical_slide_no = scene_meta_by_index.get(scene_no, {}).get("slide_number")
        else:
            logical_slide_no = slide_no
        if not isinstance(logical_slide_no, int):
            continue
        raw_image_path = str(slide.get("image_path", "") or "")
        image_path = ""
        if raw_image_path:
            candidate = Path(raw_image_path)
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
            image_path = str(candidate)
        text_parts = []
        if slide.get("t1"):
            text_parts.append(str(slide.get("t1")))
        if slide.get("t1_structure"):
            text_parts.append(str(slide.get("t1_structure")))
        current = result.get(logical_slide_no)
        candidate_entry = {
            "title": str(slide.get("title", "") or ""),
            "text": "\n".join(part for part in text_parts if part),
            "glossary_text": str(slide.get("t1", "") or ""),
            "image_path": image_path,
            "scene_number": scene_no if isinstance(scene_no, int) else None,
            "scene_numbers": [scene_no] if isinstance(scene_no, int) else [],
        }
        if current is None:
            result[logical_slide_no] = candidate_entry
            continue
        merged_scene_numbers = sorted(
            set(current.get("scene_numbers", [])) | set(candidate_entry.get("scene_numbers", []))
        )
        current_len = len(current.get("text", "")) + len(current.get("title", ""))
        candidate_len = len(candidate_entry.get("text", "")) + len(candidate_entry.get("title", ""))
        if candidate_len > current_len:
            candidate_entry["scene_numbers"] = merged_scene_numbers
            result[logical_slide_no] = candidate_entry
        else:
            current["scene_numbers"] = merged_scene_numbers
    return result


def _correct_batch_pass1(
    batch: list[tuple[int, dict]],
    slide_title: str = "",
    glossary: str = "",
) -> dict[int, str]:
    if not batch:
        return {}

    seg_lines: list[str] = []
    for local_i, (_, seg) in enumerate(batch):
        original_text = str(seg.get("text_original", seg["text"]) or "")
        pass1_candidate = str(seg.get("pass1_candidate", "") or "")
        if pass1_candidate and _normalize_text(pass1_candidate) != _normalize_text(original_text):
            seg_lines.append(
                f"[{local_i}]\n원문: {original_text}\nPass1 후보: {pass1_candidate}"
            )
        else:
            seg_lines.append(f"[{local_i}]\n원문: {original_text}")
    seg_text = "\n\n".join(seg_lines)
    topic_hint = f"\n## 현재 구간 주제\n{slide_title}\n" if slide_title.strip() else ""
    glossary_block = f"\n{glossary}\n" if glossary.strip() else ""

    prompt = f"""강의 음성 전사본을 문맥과 용어 후보들을 보고 ASR 오인식 수정 후보만 생성하세요. 슬라이드 원본은 제공되지 않습니다.
{topic_hint}{glossary_block}
## 전사 (교정 대상)
{seg_text}

## 출력 (JSON만)
{{"corrections": [{{"index": 0, "text": "교정된 텍스트"}}, ...]}}
수정 대상 index만 corrections에 넣으세요.
전체 전사본을 다시 출력하지 마세요.
각 correction의 text에는 해당 index의 교정 후 전체 문장만 넣으세요.
원문과 동일하게 유지할 index, 단순 자연화 후보는 출력하지 마세요.
ASR 오인식 가능성이 있는 후보는 Pass3가 검증할 수 있도록 출력하세요.
후보가 없으면 {{"corrections": []}}를 출력하세요.

### 교정 범위
- ASR 오인식 교정 (깨진 텍스트를 문맥에 맞는 단어로 복원)
- 전문용어 철자 교정 (슬라이드를 참고하여 정확한 표기로)
- 맞춤법, 띄어쓰기, 조사 오류 교정

### 후보 생성 근거
- 후보는 반드시 다음 중 하나 이상의 근거가 있을 때만 생성한다: 음가/철자 유사 ASR 오인식, 형태소·조사 결합 오류, 또는 인접 발화 문맥에서만 해소되는 국소적 손상.
- 문맥 복원은 원문 어절과 후보 어절의 발음 또는 형태가 가까우며, 같은 문장 역할을 유지할 때만 허용한다.
- 슬라이드·용어 사전은 표기를 뒷받침하는 보조 근거일 뿐, 그것만으로 후보를 만들지 않는다.
- 일반어를 더 좁은 전문용어로 바꾸거나, 원문에 없는 정보·수식어·서술어를 추가하는 후보는 만들지 않는다.

### 핵심 원칙 — 강의자의 실제 발화 의미를 보존하라
- 전사 원문의 의미가 기준이다. 슬라이드는 용어 철자 확인용 참고 자료일 뿐이다
- 문장 길이와 정보량은 원문과 거의 똑같게 유지
- 원문의 손상된 표현을 삭제하거나 더 일반적인 개념어로 추상화하지 말 것
- 원문에 있는 정보 단위가 후보에서 사라지면 출력하지 말 것
- 비문을 자연스럽게 요약하지 말고, 깨진 어절을 같은 위치의 어절로 복원할 것
- 요약, 재서술, 슬라이드 bullet 복사 금지
- 강의자의 발화 중 오인식된 단어가 있다면 해당 단어에 대해서만 교체하는 수준이다
- 슬라이드와 발화가 완전히 다르다면, 발화를 따르도록 할 것.
- 전사본을 따라갔을 때 강의 내용이 틀려 보이더라도 정답으로 고치지 말 것
- 단, 슬라이드에서 영어 원표기가 동적으로 확인되는 고유명사·제품명·OS명·코드명은 한국어 음역보다 해당 영어 원표기를 우선할 수 있다.
- 원문에 없는 수식어·제품군명·단어를 추가하지 않는다. 동적으로 확인된 표기라도 원문의 발화 범위를 넘어 확장하지 않는다.
- 슬라이드의 정답/문맥에 맞추기 위해 강의자의 한국어 개념어를 반대 개념으로 바꾸지 말 것
- 각 index의 원문만 수정. 다른 index 내용과 섞지 말 것

### 중요 원칙
- 강의자의 발화 구조를 절대적으로 따라가라
- 일반 한국어·한국어 기술어를 임의로 영어로 번역하지 말 것. 영어 고유명사·제품명·OS명·코드명만 정식 영어 표기로 복원할 것.
- 내용 오류, 개념 오류, 슬라이드와 발화의 불일치는 verifier가 확인할 문제이므로 전사 보정에서 제거하지 말 것"""

    def call():
        return _get_client().models.generate_content(
            model=PASS1_TEXT_MODEL,
            contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=8192,
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
            ),
        )

    try:
        response = api_call_with_retry(call)
        _add_usage(response, stage="stage3b_text_processor_pass1")
        local_corrections = parse_batch_response(response.text or "")
    except Exception as exc:
        print(f"  [Pass1 오류 무시] {exc}")
        return {}

    result: dict[int, str] = {}
    for local_i, corr_payload in local_corrections.items():
        if 0 <= local_i < len(batch):
            global_i = batch[local_i][0]
            original = batch[local_i][1].get("text_original", batch[local_i][1]["text"])
            cleaned = _normalize_text(str(corr_payload.get("text", "") or ""))
            if cleaned and cleaned != _normalize_text(original):
                result[global_i] = cleaned
    _add_stage_count("pass1_candidates", len(result))
    return result


def _correct_batch_pass2(
    batch: list[tuple[int, dict]],
    slide_context: str,
    slide_image_path: Optional[str] = None,
    glossary: str = "",
) -> dict[int, str]:
    if not batch:
        return {}

    seg_lines: list[str] = []
    for local_i, (_, seg) in enumerate(batch):
        original_text = str(seg.get("text_original", seg["text"]) or "")
        pass1_candidate = str(seg.get("pass1_candidate", "") or "")
        if pass1_candidate and _normalize_text(pass1_candidate) != _normalize_text(original_text):
            seg_lines.append(
                f"[{local_i}]\n원문: {original_text}\nPass1 후보: {pass1_candidate}"
            )
        else:
            seg_lines.append(f"[{local_i}]\n원문: {original_text}")
    seg_text = "\n\n".join(seg_lines)
    has_image = IMAGE_PROVIDER == "openai" and bool(slide_image_path and Path(slide_image_path).exists())
    if not PASS2_FULL_SLIDE_CONTEXT:
        ref_block = "\n## 강의자료 전체 텍스트\n이번 실험에서는 전체 슬라이드 텍스트를 제공하지 않습니다. 아래 용어사전만 표기 참고로 사용하세요.\n"
    elif has_image:
        ref_block = "\n## 강의 슬라이드 이미지 (첨부됨)\n이미지에 보이는 용어, 수식, 다이어그램을 참고하여 전사를 교정하세요.\n"
        if slide_context.strip():
            ref_block += f"\n## 강의자료 텍스트 (추가 참조)\n{slide_context[:1500]}\n"
    else:
        ref_block = f"\n## 강의자료 (용어 참조)\n{slide_context[:2000]}\n" if slide_context.strip() else ""
    glossary_block = f"\n## 현재/인접 슬라이드 용어 사전\n{glossary}\n" if glossary.strip() else ""

    prompt = f"""강의 음성 전사본을 슬라이드와 문맥을 참고하여 STT/ASR 오인식 가능성이 높은 수정 후보를 생성하세요.
{ref_block}{glossary_block}
## 전사 (교정 대상)
{seg_text}

## 출력 (JSON만)
{{"corrections": [{{"index": 0, "text": "교정된 텍스트"}}, ...]}}
수정 대상 index만 corrections에 넣으세요.
전체 전사본을 다시 출력하지 마세요.
각 correction의 text에는 해당 index의 교정 후 전체 문장만 넣으세요.
원문과 동일하게 유지할 index, 단순 자연화 후보는 출력하지 마세요.
ASR 오인식 가능성이 있는 후보는 Pass3가 검증할 수 있도록 출력하세요.
후보가 없으면 {{"corrections": []}}를 출력하세요.

### 역할
- 이것은 전사 후처리의 Standard Fix 단계다. 필러 제거, 요약, 구조화, 문장 정리는 하지 않는다.
- 슬라이드와 용어 사전은 correction dictionary로만 사용한다. 슬라이드 문장을 베껴 전사를 더 좋은 설명문으로 만들지 않는다.
- 영어 자체가 정식 명칭인 고유명사·제품명·OS명·코드명은 슬라이드에서 동적으로 확인된 영어 원표기를 우선한다. 단, 원문에 없는 단어를 덧붙이거나 제품명을 임의로 확장하지 않는다.
- Pass2는 최종 승인 단계가 아니라 후보 생성 단계다. 원문이 깨져 있고 슬라이드/용어 사전/주변 문맥이 지지하는 후보는 출력한다. 최종 적용 여부는 Pass3가 판단한다.
- Pass2의 출력 text는 반드시 원문을 기준으로 교정한 전체 문장이어야 한다.

### 후보 생성 근거
- 후보는 반드시 음가/철자 유사 ASR 오인식, 형태소·조사 결합 오류, 또는 인접 발화 문맥에서만 해소되는 국소적 손상 중 하나로 설명될 수 있어야 한다.
- 문맥은 손상된 어절의 후보를 고르는 데만 사용한다. 문맥에 맞는 더 좋은 설명문을 새로 만들 근거가 되지 않는다.
- 슬라이드·용어 사전의 일치는 표기 확인을 보조할 뿐, 일반어를 전문용어로 치환하거나 개념 범위를 바꿀 근거가 되지 않는다.
- 위 원칙의 예외로, 영어 정식 명칭의 음역을 영어 원표기로 복원하는 것은 표기 교정에 해당한다.

### 수정할 것
- STT/ASR 오인식 가능성이 높은 단어/구절: 발음은 비슷하지만 문맥상 한국어로 성립하기 어렵거나 강의 용어와 맞지 않는 경우
- 슬라이드 또는 용어 사전에 실제로 보이는 강의 용어의 철자/표기 오류
- 전문용어가 조사/접사와 붙어 깨진 경우, 전문용어가 포함된 어절 전체를 후보로 복원
- 한국어식 발음이나 깨진 음가로 들어온 영문/외래어/전문용어의 원 표기 복원
- 맞춤법/띄어쓰기는 강의 용어 표기 또는 오인식 복원에 필요한 경우만 수정

### 수정하지 말 것
- 문장 어미, 말투, 격식, 발화 순서, 문장 구조 변경
- 단순 자연화, 표현 개선, 중복 제거, 필러 제거
- 설명 추가, 의미 보충, 정의문/뜻풀이로 확장
- 슬라이드 정답에 맞추기 위한 개념 교정
- 한국어로 정상 발화한 일반어를 영어/전문어로 번역하는 것
- 다른 index의 내용을 섞거나, 한 index의 내용을 다른 index로 옮기는 것

### 기준
- 전사 원문의 의미가 기준이다. 슬라이드는 용어와 표기 확인용 참고 자료다.
- 수정은 가능한 최소 span이어야 한다.
- 후보 문장은 원문과 길이, 정보량, 구어체가 거의 같아야 한다.
- 단순 취향 차이나 자연화 후보는 출력하지 않는다.
- 원문 구절이 깨져 있고 참조 자료나 주변 문맥이 지지하면, 적용 여부 판단은 Pass3에 맡기고 후보로 출력한다."""

    img_bytes = None
    if has_image and IMAGE_PROVIDER == "openai":
        with open(slide_image_path, "rb") as f:
            img_bytes = f.read()

    try:
        if has_image and IMAGE_PROVIDER == "openai":
            response_text = _call_openai_image_correction(prompt, img_bytes or b"")
        else:
            response_text = _call_openai_text_correction(
                prompt,
                stage="stage3b_text_processor_pass2",
                model=PASS2_TEXT_MODEL,
                system_prompt=(
                    "당신은 한국어 강의 STT 오류 후보 생성기입니다. "
                    "슬라이드와 용어 사전을 교정 사전으로만 사용하고, "
                    "오인식 가능성이 높은 후보를 JSON으로 출력하세요."
                ),
                max_output_tokens=PASS2_MAX_OUTPUT_TOKENS,
            )
        local_corrections = parse_batch_response(response_text)
    except Exception as exc:
        print(f"  [Pass2 오류 무시] {exc}")
        return {}

    result: dict[int, str] = {}
    for local_i, corr_payload in local_corrections.items():
        if 0 <= local_i < len(batch):
            global_i = batch[local_i][0]
            original = batch[local_i][1].get("text_original", batch[local_i][1]["text"])
            cleaned = _normalize_text(str(corr_payload.get("text", "") or ""))
            if cleaned and cleaned != _normalize_text(original):
                result[global_i] = cleaned
    _add_stage_count("pass2_candidates", len(result))
    return result


def parse_pass3_response(text: str) -> dict[str, list[dict]]:
    if not text:
        return {"accepted_changes": []}
    raw = text.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"accepted_changes": []}
    if not isinstance(parsed, dict):
        return {"accepted_changes": []}

    raw_items = parsed.get("changes", parsed.get("accepted_changes", parsed.get("accepted", [])))
    accepted_changes: list[dict] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        idx = item.get("i", item.get("index"))
        from_text = str(item.get("from", item.get("from_text", "")) or "")
        to_text = str(item.get("to", item.get("to_text", "")) or "")
        if not isinstance(idx, int) or not from_text or not to_text:
            continue
        candidate_number = item.get("c", item.get("candidate_number"))
        occurrence = item.get("occ", item.get("occurrence", 1))
        reason_code = item.get("reason_code", item.get("r"))
        if isinstance(reason_code, str) and reason_code.isdigit():
            reason_code = int(reason_code)
        if not isinstance(reason_code, int) or reason_code not in PASS3_REASON_CODES:
            reason_code = 0
        reason = PASS3_REASON_CODES.get(reason_code, "")
        if not reason:
            reason = str(item.get("reason", "") or "").strip()
        accepted_changes.append({
            "index": idx,
            "candidate_number": candidate_number if isinstance(candidate_number, int) else None,
            "from_text": from_text,
            "to_text": to_text,
            "occurrence": occurrence,
            "reason_code": reason_code,
            "reason": reason,
        })
    return {"accepted_changes": accepted_changes}


def _build_pass3_items(
    batch: list[tuple[int, dict]],
    pass1: dict[int, str],
    pass2: dict[int, str],
) -> list[dict]:
    raw_by_global = {
        global_i: str(seg.get("text_original", seg["text"]) or "")
        for global_i, seg in batch
    }
    sorted_globals = sorted(raw_by_global.keys())
    position_by_global = {global_i: pos for pos, global_i in enumerate(sorted_globals)}
    items: list[dict] = []
    for global_i in sorted(set(pass1.keys()) | set(pass2.keys())):
        raw = raw_by_global.get(global_i, "")
        raw_norm = _normalize_text(raw)
        candidates: list[dict] = []
        p1 = _normalize_text(pass1.get(global_i, "") or "")
        p2 = _normalize_text(pass2.get(global_i, "") or "")
        if p1 and p1 != raw_norm:
            candidates.append({"source": "pass1", "sources": ["pass1"], "text": p1})
        if p2 and p2 != raw_norm:
            same_text = next((candidate for candidate in candidates if candidate["text"] == p2), None)
            if same_text is not None:
                # 동일 후보를 두 pass가 만들었다는 사실도 보존해야 사후 품질 분석이 가능하다.
                same_text.setdefault("sources", [same_text.get("source", "pass1")]).append("pass2")
            else:
                candidates.append({"source": "pass2", "sources": ["pass2"], "text": p2})
        if candidates:
            pos = position_by_global.get(global_i, -1)
            prev_text = raw_by_global.get(sorted_globals[pos - 1], "") if pos > 0 else ""
            next_text = raw_by_global.get(sorted_globals[pos + 1], "") if 0 <= pos < len(sorted_globals) - 1 else ""
            items.append({
                "global_index": global_i,
                "raw_text": raw,
                "prev_text": prev_text,
                "next_text": next_text,
                "candidates": candidates,
            })
    return items


def _format_pass3_items(pass3_items: list[dict]) -> str:
    blocks: list[str] = []
    for local_i, item in enumerate(pass3_items):
        lines = [
            f"[{local_i}]",
            f"이전 발화: {item.get('prev_text', '')}",
            f"원문: {item.get('raw_text', '')}",
            f"다음 발화: {item.get('next_text', '')}",
            "후보 문장:",
        ]
        for candidate_number, candidate in enumerate(item.get("candidates", []), start=1):
            lines.append(
                f"- 후보 {candidate_number} ({candidate.get('source', '')}): "
                f"{candidate.get('text', '')}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _validate_candidates_pass3(
    pass3_items: list[dict],
    slide_context: str = "",
) -> dict[int, dict]:
    if not pass3_items:
        return {}
    total_candidates = sum(len(item.get("candidates", [])) for item in pass3_items)
    _add_stage_count("pass3_candidates", total_candidates)

    ref_block = f"\n## 강의자료 참고\n{slide_context[:1000]}\n" if slide_context.strip() else ""
    items_text = _format_pass3_items(pass3_items)
    prompt = f"""역할: 한국어 강의 STT 후보 edit 추출기.
목표: 후보 문장을 적용하지 말고, 원문에서 후보로 바꿀 수 있는 최소 from_text/to_text 쌍만 모두 출력한다.
절대 기준은 각 항목의 '원문'이다. 원문이 실제 Whisper 원전사이며, 후보 문장이나 슬라이드 문장이 정답이 아니다.
원문과 후보의 차이를 글자 단위로 대조하고, 원문에 실제로 존재하는 국소적인 ASR 손상만 교체 대상으로 삼아라.
확신이 없거나 원문과 후보의 대응 위치를 특정할 수 없으면 아무 변경도 출력하지 말고 원문을 유지하라.
출력에 없는 변경은 적용하지 않는다.
{ref_block}

## 검증 대상
{items_text}

## 출력(JSON만)
{{
  "changes":[
    {{"i":0,"c":1,"from":"원문 연속 문자열","to":"후보 근거 문자열","occ":1,"r":1}}
  ]
}}

reason_code는 아래 번호 중 하나만 사용한다. 번호를 고를 수 없으면 그 변경은 출력하지 않는다.
1 발음/음가 유사 오인식
2 강의 도메인 전문용어 복원
3 슬라이드 표기 용어 복원
4 문맥상 깨진 구절 복원
5 깨진 용어에 붙은 조사까지 복원
6 영문/코드/약어 오인식 복원
7 고유명사 복원

## 적용 전 필수 판정
각 변경은 아래 조건을 모두 만족할 때만 출력한다.
1. 원문 어절에 ASR 손상 근거가 있다. 근거는 음가/철자 유사, 형태소·조사 결합 오류, 또는 인접 발화와의 연결에서 드러나는 국소적 문맥 손상이다. 후보가 더 자연스럽거나 슬라이드에 있다는 이유만으로는 부족하다.
2. 후보는 원문과 같은 문장 역할과 정보량을 유지한다. 일반어를 더 좁은 개념으로 바꾸거나, 원문에 없던 정보·수식어·서술어를 보태거나, 정보를 삭제하면 안 된다.
3. 슬라이드·용어 사전은 일반어 교정의 보조 근거일 뿐이다. 단, 영어 정식 고유명사·제품명·OS명·코드명은 슬라이드에 영어 원표기가 있으면 그 표기를 적용 근거로 사용할 수 있다.
4. 위 근거가 불충분하면 후보가 자연스러워 보여도 출력하지 않는다.

## 실행 순서
1. 각 후보를 원문과 왼쪽부터 오른쪽까지 끝까지 비교한다.
2. 원문의 깨진 단어/구절과 후보의 같은 위치 대응어를 찾는다.
3. 하나를 찾은 뒤에도 멈추지 말고, 같은 항목의 남은 차이를 계속 검사한다.
4. 서로 독립된 오류는 하나의 큰 span으로 합치지 말고 별도 changes로 출력한다.
5. 후보 전체가 자연화/재작성이어도, 명확한 단어/구절 대응만 따로 출력한다.
6. from_text는 원문에 실제로 존재해야 하고, to_text는 해당 후보에 실제로 존재해야 한다.
7. 한국어 오류 span은 한 글자나 어절 내부 일부만 자르지 말고, 깨진 어절/용어 단위로 잡는다.
   - 금지: 외 -> 회, 한해 -> 환산, 재활 -> 재화
   - 허용: 관리외계 -> 관리회계, 외계에서 -> 회계에서, 한해봐야겠죠 -> 환산해봐야겠죠, 재활을 -> 재화를
8. 조사/어미와 결합했을 때 최종 문장이 깨지면, 조사/어미까지 포함한 span을 출력한다.

출력 키 의미:
- i: 검증 대상의 [] 번호
- c: 후보 번호
- from: 원문에 실제로 있는 연속 문자열
- to: 후보 문장에 실제로 있는 문자열
- occ: 같은 from이 원문에 여러 번 있으면 몇 번째인지, 모두면 "all"
- r: reason_code

## accepted 기준
- 고유명사, 영문 토큰, 코드/함수명, 제품명, 슬라이드 표기 용어, 운영체제/컴퓨터 구조 전문용어의 오인식은 적극 출력한다.
- 영어 자체가 정식 명칭인 항목은 한국어 음역 대신 영어 원표기를 선택한다. 일반 한국어 기술어를 영어로 바꾸는 근거로 사용하지 않는다.
- 같은 강의에서 반복되는 전문용어의 오인식도 적극 출력한다.
- 일반 명사구/술어구는 원문이 문맥상 성립하기 어렵고, 후보가 같은 위치에서 음가/문맥상 자연스럽게 복원할 때만 출력한다.
- 비문은 전체 문장을 고치지 말고, 비문을 만든 깨진 단어/구절만 출력한다.
- 용어 복원은 원문 어절 자체에 음가·철자·형태상 손상 근거가 있을 때만 출력한다.

## rejected span
- 문장 전체 또는 절 전체
- 어순 변경, 주어/목적어/서술어 보충
- 원문 단어 삭제, 설명 추가, 의미 추가/삭제
- 단순 자연화, 구어체 정리, 문장부호/띄어쓰기만 변경
- 후보에 없는 새 표현
- 최소 대응을 확정할 수 없는 변경
"""

    try:
        response_text = _call_openai_text_correction(
            prompt,
            stage="stage3b_text_processor_pass3",
            model=PASS3_TEXT_MODEL,
            system_prompt=(
                "당신은 한국어 강의 STT 후보 edit 추출기입니다. "
                "각 항목의 원문(실제 Whisper ASR)을 유일한 기준으로 삼고 후보 문장을 정답으로 취급하지 마세요. "
                "원문에 근거가 있는 최소 from_text/to_text만 출력하고, 불확실하면 변경하지 마세요. "
                "후보 문장 전체, 절 전체, 새 문장은 출력하지 마세요. JSON만 출력하세요."
            ),
            max_output_tokens=PASS3_MAX_OUTPUT_TOKENS,
        )
        accepted = parse_pass3_response(response_text)
    except Exception as exc:
        print(f"  [Pass3 오류 무시] {exc}")
        return {}

    accepted_changes_by_local: dict[int, list[dict]] = {}
    for judgment in accepted.get("accepted_changes", []):
        local_i = judgment.get("index")
        if not isinstance(local_i, int):
            continue
        accepted_changes_by_local.setdefault(local_i, []).append(judgment)

    audit_records: list[dict] = []
    result: dict[int, dict] = {}
    for local_i, item in enumerate(pass3_items):
        global_i = int(item["global_index"])
        raw = str(item.get("raw_text", "") or "")
        accepted_edits: list[dict] = []
        audit_changes: list[dict] = []
        for change in accepted_changes_by_local.get(local_i, []):
            expanded_changes = _expand_llm_change(raw, change, item.get("candidates", []))
            if expanded_changes:
                accepted_edits.extend(expanded_changes)
                audit_changes.extend([{**edit, "accepted": True} for edit in expanded_changes])
            else:
                audit_changes.append({
                    "candidate_number": change.get("candidate_number"),
                    "from_text": str(change.get("from_text", "") or ""),
                    "to_text": str(change.get("to_text", "") or ""),
                    "occurrence": change.get("occurrence", 1),
                    "reason_code": change.get("reason_code", 0),
                    "reason": str(change.get("reason", "") or ""),
                    "accepted": False,
                })
        audit_records.append({
            "index": global_i,
            "pass3_local_index": local_i,
            "raw_text": raw,
            "prev_text": str(item.get("prev_text", "") or ""),
            "next_text": str(item.get("next_text", "") or ""),
            "candidate_full_texts": item.get("candidates", []),
            "accepted_changes": audit_changes,
        })

        if accepted_edits:
            result[global_i] = {
                "accepted_edits": accepted_edits,
                "raw_text": raw,
            }
    _add_pass3_candidate_audit(audit_records)
    _add_stage_count("pass3_accepted", len(result))
    return result


def _is_surface_only_edit(edit: dict) -> bool:
    from_text = str(edit.get("from_text", "") or "")
    to_text = str(edit.get("to_text", "") or "")
    if not from_text or not to_text:
        return False
    strip_chars = r"[\s\.,;:!\?。．，、…]+"
    return re.sub(strip_chars, "", from_text) == re.sub(strip_chars, "", to_text)


def _is_hangul_char(value: str) -> bool:
    return bool(value) and "가" <= value <= "힣"


def _is_unsafe_short_korean_span(raw_text: str, from_text: str, to_text: str, raw_start: int, raw_end: int) -> bool:
    if len(from_text) > 2 and len(to_text) > 2:
        return False
    if not re.search(r"[가-힣]", from_text + to_text):
        return False
    left = raw_text[raw_start - 1] if raw_start > 0 else ""
    right = raw_text[raw_end] if raw_end < len(raw_text) else ""
    return _is_hangul_char(left) or _is_hangul_char(right)


def _is_broken_phrase_content_expansion(from_text: str, to_text: str, reason_code: int) -> bool:
    """Reject a reason-4 edit that preserves a phrase and merely appends content to it."""
    if reason_code != 4:
        return False
    source = re.sub(r"\s+", "", from_text)
    target = re.sub(r"\s+", "", to_text)
    if not source or len(target) <= len(source):
        return False
    return target.startswith(source) or target.endswith(source)


def _expand_llm_change(raw_text: str, change: dict, candidates: list[dict]) -> list[dict]:
    raw_text = str(raw_text or "")
    from_text = str(change.get("from_text", "") or "")
    to_text = str(change.get("to_text", "") or "")
    if not from_text or not to_text:
        return []
    if _is_surface_only_edit({"from_text": from_text, "to_text": to_text}):
        return []

    candidate_number = change.get("candidate_number")
    candidate_sources: list[str] = []
    if isinstance(candidate_number, int) and 1 <= candidate_number <= len(candidates):
        candidate = candidates[candidate_number - 1]
        candidate_text = str(candidate.get("text", "") or "")
        if to_text not in candidate_text:
            return []
        candidate_sources = [
            str(source) for source in candidate.get("sources", [candidate.get("source", "")])
            if str(source)
        ]

    starts: list[int] = []
    start = raw_text.find(from_text)
    while start != -1:
        starts.append(start)
        start = raw_text.find(from_text, start + max(1, len(from_text)))
    if not starts:
        return []

    occurrence = change.get("occurrence", 1)
    if isinstance(occurrence, str) and occurrence.lower() == "all":
        selected_starts = starts
    elif isinstance(occurrence, int) and 1 <= occurrence <= len(starts):
        selected_starts = [starts[occurrence - 1]]
    elif len(starts) == 1:
        selected_starts = starts
    else:
        return []

    reason = str(change.get("reason", "") or "")
    reason_code = change.get("reason_code", 0)
    if _is_broken_phrase_content_expansion(from_text, to_text, reason_code):
        return []
    expanded: list[dict] = []
    for raw_start in selected_starts:
        raw_end = raw_start + len(from_text)
        if _is_unsafe_short_korean_span(raw_text, from_text, to_text, raw_start, raw_end):
            continue
        expanded.append({
            "source": "pass3_llm",
            "candidate_number": candidate_number if isinstance(candidate_number, int) else 0,
            "candidate_sources": candidate_sources,
            "from_text": from_text,
            "to_text": to_text,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "reason_code": reason_code if isinstance(reason_code, int) else 0,
            "reason": reason,
            "accepted": True,
            "accepted_via": "llm_change",
        })
    return expanded


def _apply_accepted_edits(raw_text: str, edits: list[dict]) -> tuple[str, list[dict]]:
    raw_text = str(raw_text or "")
    valid_edits: list[dict] = []
    occupied: list[tuple[int, int]] = []

    for edit in sorted(edits, key=lambda e: (int(e.get("raw_start", 0) or 0), int(e.get("raw_end", 0) or 0))):
        raw_start = int(edit.get("raw_start", 0) or 0)
        raw_end = int(edit.get("raw_end", raw_start) or raw_start)
        from_text = str(edit.get("from_text", "") or "")
        if raw_start < 0 or raw_end < raw_start or raw_end > len(raw_text):
            continue
        if raw_text[raw_start:raw_end] != from_text:
            continue
        overlaps = any(not (raw_end <= used_start or raw_start >= used_end) for used_start, used_end in occupied)
        if overlaps and raw_start != raw_end:
            continue
        occupied.append((raw_start, raw_end))
        valid_edits.append(edit)

    if not valid_edits:
        return raw_text, []

    corrected = raw_text
    for edit in sorted(valid_edits, key=lambda e: int(e.get("raw_start", 0) or 0), reverse=True):
        raw_start = int(edit.get("raw_start", 0) or 0)
        raw_end = int(edit.get("raw_end", raw_start) or raw_start)
        to_text = str(edit.get("to_text", "") or "")
        corrected = corrected[:raw_start] + to_text + corrected[raw_end:]
    return corrected, valid_edits


def merge_two_passes(
    batch: list[tuple[int, dict]],
    pass1: dict[int, str],
    pass2: dict[int, str],
    subdomain: str = "",
    slide_context: str = "",
) -> dict[int, dict]:
    pass3_items = _build_pass3_items(batch, pass1, pass2)
    return merge_pass3_items(pass3_items, slide_context=slide_context)


def merge_pass3_items(
    pass3_items: list[dict],
    slide_context: str = "",
) -> dict[int, dict]:
    corrections: dict[int, dict] = {}
    accepted = _validate_candidates_pass3(pass3_items, slide_context=slide_context)

    for global_i, judgment in accepted.items():
        raw_text = str(judgment.get("raw_text", "") or "")
        accepted_edits = list(judgment.get("accepted_edits", []) or [])
        applied_text, applied_edits = _apply_accepted_edits(raw_text, accepted_edits)
        selected_text = _normalize_text(applied_text)
        raw_norm = _normalize_text(raw_text)
        if not selected_text or selected_text == raw_norm or not applied_edits:
            continue
        sources = sorted({str(edit.get("source", "") or "candidate") for edit in applied_edits})
        reasons = [str(edit.get("reason", "") or "") for edit in applied_edits if str(edit.get("reason", "") or "")]
        corrections[global_i] = {
            "candidate_text": selected_text,
            "applied_text": selected_text,
            "risk": "low",
            "apply": True,
            "reason": "; ".join(reasons) or "pass3 accepted edits",
            "source": "+".join(sources) if sources else "candidate",
            "accepted_edits": applied_edits,
        }
    return corrections


def _build_pass2_batch_from_pass1(
    batch: list[tuple[int, dict]],
    pass1: dict[int, str],
) -> list[tuple[int, dict]]:
    return [(global_i, seg.copy()) for global_i, seg in batch]


def correct_segments_two_pass(
    segments: list[dict],
    metadata: list[dict],
    textualized_data: dict,
    textualized_dir: Path,
    use_pass2: bool = True,
) -> list[dict]:
    if not segments:
        return []

    original_segments = [segment.copy() for segment in segments]

    scene_occurrences, _ = _load_slide_occurrences_from_metadata(metadata)
    scene_meta_by_index = _build_scene_metadata_index(metadata)
    extracted_slide_texts = _load_integrated_slide_texts(
        textualized_data,
        textualized_dir,
        scene_meta_by_index=scene_meta_by_index,
    )

    occ_index = _build_occurrence_index(scene_occurrences)
    seg_scene: dict[int, int] = {}
    seg_logical_slide: dict[int, int] = {}
    for i, seg in enumerate(segments):
        scene_no = seg.get("scene_index")
        if isinstance(scene_no, int):
            seg_scene[i] = scene_no
            logical_slide_no = scene_meta_by_index.get(scene_no, {}).get("slide_number")
            if isinstance(logical_slide_no, int):
                seg_logical_slide[i] = logical_slide_no
            continue
        occ_idx = _assign_segment_occurrence(seg, occ_index)
        if occ_idx is not None:
            scene_no = occ_index[occ_idx]["scene_no"]
            seg_scene[i] = scene_no
            logical_slide_no = scene_meta_by_index.get(scene_no, {}).get("slide_number")
            if isinstance(logical_slide_no, int):
                seg_logical_slide[i] = logical_slide_no

    glossary_window = max(0, int(os.getenv("GRAPHLEC_TEXT_PROCESSOR_GLOSSARY_SLIDE_WINDOW", "1")))
    all_glossary_records = extract_glossary_term_records(extracted_slide_texts)
    all_glossary_terms = [record["term"] for record in all_glossary_records]
    if all_glossary_records:
        print(
            f"    용어 사전: 전체 {len(all_glossary_records)}개 용어, "
            f"batch별 관련 용어만 검색 (현재±{glossary_window} 슬라이드 보강)"
        )

    # Pass 1이 원문 그대로가 아니라, 고신뢰 용어 후보가 반영된 텍스트를
    # 입력으로 받도록 한다. 원문은 original_segments와 text_original_asr에 보존한다.
    segments = _apply_kiwi_prepass(
        segments,
        seg_logical_slide,
        extracted_slide_texts,
        all_glossary_terms,
    )

    # Pre-pass 이후의 segment 사본으로 슬라이드 그룹을 다시 만든다.
    groups = {}
    group_scene_indices = {}
    no_slide = []
    for i, seg in enumerate(segments):
        if i in seg_logical_slide:
            logical_slide_no = seg_logical_slide[i]
            groups.setdefault(logical_slide_no, []).append((i, seg))
            if i in seg_scene:
                group_scene_indices.setdefault(logical_slide_no, set()).add(seg_scene[i])
        else:
            no_slide.append((i, seg))

    slide_titles = [extracted_slide_texts.get(sno, {}).get("title", "") for sno in sorted(extracted_slide_texts.keys())]
    transcript_sample = " ".join(seg.get("text", "") for seg in segments[:30])
    domain_info = classify_lecture_domain(slide_titles, transcript_sample)
    subdomain = domain_info.get("subdomain", "")
    print(f"    도메인: {domain_info['domain']}, 서브도메인: {subdomain}")

    def process_sub_batch(
        sub: list[tuple[int, dict]],
        context: str,
        slide_title: str = "",
        glossary: str = "",
    ) -> tuple[str, list[dict]]:
        pass1 = _correct_batch_pass1(sub, slide_title=slide_title, glossary=glossary)
        if not use_pass2:
            return context, _build_pass3_items(sub, pass1, {})

        pass2_batch = _build_pass2_batch_from_pass1(sub, pass1)
        pass2 = _correct_batch_pass2(pass2_batch, context, glossary=glossary)
        return context, _build_pass3_items(sub, pass1, pass2)

    parallel_jobs: list[tuple[list[tuple[int, dict]], str, str, str]] = []
    for logical_slide_no in sorted(extracted_slide_texts.keys()):
        group = groups.get(logical_slide_no, [])
        if not group:
            continue
        extracted = extracted_slide_texts.get(logical_slide_no, {})
        context = f"슬라이드 제목: {extracted.get('title', '')}\n{extracted.get('text', '')}"
        slide_title = extracted.get("title", "")
        scene_list = sorted(group_scene_indices.get(logical_slide_no, set()))
        if len(scene_list) <= 3:
            scene_label = ",".join(f"{scene_idx}" for scene_idx in scene_list)
        else:
            scene_label = f"{scene_list[0]},{scene_list[1]},...,{scene_list[-1]}"
        sub_batches = [group[b:b + BATCH_SIZE] for b in range(0, len(group), BATCH_SIZE)]
        print(
            f"    slide {logical_slide_no:3d} (scenes {scene_label:>7s}, {slide_title[:22]:22s}): "
            f"{len(group):3d}개, {len(sub_batches)}배치",
            flush=True,
        )

        for sub in sub_batches:
            glossary = build_batch_glossary(
                sub,
                extracted_slide_texts,
                logical_slide_no,
                all_glossary_terms,
                window=glossary_window,
                term_records=all_glossary_records,
                scene_indices=set(scene_list),
            )
            parallel_jobs.append((sub, context, slide_title, glossary))

    if no_slide:
        print(f"    미매핑 {len(no_slide)}개 교정 중...")
        for offset in range(0, len(no_slide), BATCH_SIZE):
            sub = no_slide[offset:offset + BATCH_SIZE]
            parallel_jobs.append((
                sub,
                "",
                "",
                build_batch_glossary(
                    sub,
                    extracted_slide_texts,
                    None,
                    all_glossary_terms,
                    term_records=all_glossary_records,
                ),
            ))

    print(
        f"    총 {len(parallel_jobs)}개 batch를 최대 {PARALLEL_REQUESTS}개 병렬로 교정 중...",
        end="",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=PARALLEL_REQUESTS) as executor:
        futures = [
            executor.submit(process_sub_batch, sub, context, slide_title, glossary)
            for sub, context, slide_title, glossary in parallel_jobs
        ]
        pass3_groups: dict[str, list[dict]] = {}
        for future in as_completed(futures):
            context, pass3_items = future.result()
            if pass3_items:
                pass3_groups.setdefault(context, []).extend(pass3_items)
            print(".", end="", flush=True)
    print()

    pass3_jobs: list[tuple[list[dict], str]] = []
    for context, items in pass3_groups.items():
        items.sort(key=lambda item: int(item.get("global_index", 0) or 0))
        for offset in range(0, len(items), PASS3_ITEM_BATCH_SIZE):
            pass3_jobs.append((items[offset:offset + PASS3_ITEM_BATCH_SIZE], context))

    all_corrections: dict[int, dict] = {}
    total_pass3_items = sum(len(items) for items, _ in pass3_jobs)
    if pass3_jobs:
        print(
            f"    Pass3 후보 {total_pass3_items}개를 {len(pass3_jobs)}개 batch로 검증 중...",
            end="",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=PARALLEL_REQUESTS) as executor:
            futures = [
                executor.submit(merge_pass3_items, items, context)
                for items, context in pass3_jobs
            ]
            for future in as_completed(futures):
                all_corrections.update(future.result())
                print(".", end="", flush=True)
        print()

    corrected_segments: list[dict] = []
    applied_count = 0
    for i, seg in enumerate(segments):
        original_seg = original_segments[i]
        corrected = original_seg.copy()
        scene_idx = seg_scene.get(i)
        if isinstance(scene_idx, int):
            corrected["scene_index"] = scene_idx
            scene_meta = scene_meta_by_index.get(scene_idx, {})
            logical_slide_no = scene_meta.get("slide_number")
            if isinstance(logical_slide_no, int):
                corrected["slide_number"] = logical_slide_no
            slide_canonical_index = scene_meta.get("slide_canonical_index")
            if isinstance(slide_canonical_index, int):
                corrected["slide_canonical_index"] = slide_canonical_index
            corrected["slide_visit_order"] = int(scene_meta.get("slide_visit_order", 1) or 1)
            corrected["slide_is_revisit"] = bool(scene_meta.get("slide_is_revisit", False))
        original = str(original_seg.get("text_original", original_seg.get("text", "")) or "")
        prepass_text = str(seg.get("text_prepass", seg.get("text_original", original)) or original)
        corrected["text_raw"] = original
        corrected["text_prepass"] = prepass_text
        corrected["text_corrected"] = prepass_text
        corrected["text_original"] = original
        corrected["text_original_asr"] = original
        if seg.get("kiwi_prepass_candidates"):
            corrected["kiwi_prepass_candidates"] = seg["kiwi_prepass_candidates"]
        if seg.get("kiwi_prepass_edits"):
            corrected["kiwi_prepass_edits"] = seg["kiwi_prepass_edits"]

        corr_info = all_corrections.get(i)
        if corr_info:
            accepted_edits = corr_info.get("accepted_edits")
            if accepted_edits:
                corrected["accepted_edits"] = accepted_edits
            corrected["correction_risk"] = str(corr_info.get("risk", "none") or "none")
            corrected["correction_reason"] = str(corr_info.get("reason", "") or "")
            candidate_text = str(corr_info.get("candidate_text", "") or "").strip()
            if candidate_text:
                corrected["text_corrected_candidate"] = candidate_text
            if DEBUG_CORRECTION_OUTPUT:
                corrected["correction_source"] = str(corr_info.get("source", "") or "")

            if corr_info.get("apply") and corr_info.get("applied_text"):
                corrected_text = str(corr_info["applied_text"])
                corrected["text"] = corrected_text
                corrected["text_corrected"] = corrected_text
                corrected["correction_status"] = "applied"
                applied_count += 1
            else:
                corrected["text"] = prepass_text
                corrected["text_corrected"] = prepass_text
                corrected["correction_status"] = (
                    "prepass_applied"
                    if _normalize_text(prepass_text) != _normalize_text(original)
                    else "unchanged"
                )
        else:
            corrected["text"] = prepass_text
            corrected["correction_status"] = (
                "prepass_applied" if _normalize_text(prepass_text) != _normalize_text(original) else "unchanged"
            )
            corrected["correction_risk"] = "low" if corrected["correction_status"] == "prepass_applied" else "none"
            corrected["correction_reason"] = (
                "kiwi glossary pre-pass" if corrected["correction_status"] == "prepass_applied" else ""
            )

        corrected_segments.append(corrected)

    print(f"    교정 적용: {applied_count}건 / 후보 출력: 0건 / 전체 {len(segments)}건")
    return corrected_segments
