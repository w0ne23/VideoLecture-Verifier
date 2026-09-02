"""
텍스트 교정 엔진

Pass 1: Gemini로 슬라이드 제목 + 용어 사전 기반 ASR 오인식 후보 생성
Pass 2: GPT로 슬라이드 전체 텍스트 컨텍스트 기반 후보 보강
Pass 3: GPT-5.4로 Pass 1/2 후보 중 적용할 후보만 선택
"""

import json
import os
import re
import time
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Callable, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .config import GEMINI_GENERATIVE_MODEL, gemini_client_2

load_dotenv()

GEMINI_MODEL = GEMINI_GENERATIVE_MODEL
# 상용 API로 되돌리려면: PASS1=gemini-3-flash-preview, PASS2=gpt-5.4-mini, PASS3=gpt-5.4
PASS1_TEXT_MODEL = os.getenv("VLVERIFIER_TEXT_PROCESSOR_PASS1_MODEL", "ollama:qwen3.8:27b-q4_K_M").strip()
PASS2_TEXT_MODEL = os.getenv("VLVERIFIER_TEXT_PROCESSOR_PASS2_MODEL", "ollama:qwen3.8:27b-q4_K_M").strip()
PASS3_TEXT_MODEL = os.getenv("VLVERIFIER_TEXT_PROCESSOR_PASS3_MODEL", "ollama:qwen3.8:27b-q4_K_M").strip()
TEXT_REASONING_EFFORT = os.getenv("VLVERIFIER_TEXT_PROCESSOR_REASONING_EFFORT", "minimal").strip().lower()
PASS2_REASONING_EFFORT = os.getenv("VLVERIFIER_TEXT_PROCESSOR_PASS2_REASONING_EFFORT", "minimal").strip().lower()
PASS3_REASONING_EFFORT = os.getenv("VLVERIFIER_TEXT_PROCESSOR_PASS3_REASONING_EFFORT", "low").strip().lower()
TEXT_MAX_OUTPUT_TOKENS = int(os.getenv("VLVERIFIER_TEXT_PROCESSOR_TEXT_MAX_OUTPUT_TOKENS", "8192"))
PASS2_MAX_OUTPUT_TOKENS = int(os.getenv("VLVERIFIER_TEXT_PROCESSOR_PASS2_MAX_OUTPUT_TOKENS", "4096"))
PASS3_MAX_OUTPUT_TOKENS = int(os.getenv("VLVERIFIER_TEXT_PROCESSOR_PASS3_MAX_OUTPUT_TOKENS", "2048"))
PASS3_ITEM_BATCH_SIZE = max(1, int(os.getenv("VLVERIFIER_TEXT_PROCESSOR_PASS3_ITEM_BATCH_SIZE", "12")))
IMAGE_PROVIDER = os.getenv("VLVERIFIER_TEXT_PROCESSOR_IMAGE_PROVIDER", "text").strip().lower()
IMAGE_MODEL = os.getenv("VLVERIFIER_TEXT_PROCESSOR_IMAGE_MODEL", "gpt-4.1-mini").strip()

BATCH_SIZE = int(os.getenv("MERGE_CORRECTION_BATCH_SIZE", "12"))
PARALLEL_REQUESTS = max(1, int(os.getenv("MERGE_CORRECTION_PARALLEL_REQUESTS", "20")))
# Ollama는 OLLAMA_NUM_PARALLEL(기본 4)개까지만 서버에서 동시 처리, 그보다
# 많은 요청을 동시에 쏘면 대기열에 밀려 클라이언트 타임아웃(600초)을 넘겨버려서,
# 로컬 모델을 쓸 때는 서버 동시 처리 한도에 맞춰 요청 수를 낮춤
OLLAMA_PARALLEL_REQUESTS = max(1, int(os.getenv("MERGE_CORRECTION_OLLAMA_PARALLEL_REQUESTS", "4")))
TRANSITION_LEAD_SEC = float(os.getenv("MERGE_TRANSITION_LEAD_SEC", "1.0"))
TRANSITION_TAIL_SEC = float(os.getenv("MERGE_TRANSITION_TAIL_SEC", "0.2"))
ASSIGN_MAX_GAP_SEC = float(os.getenv("MERGE_ASSIGN_MAX_GAP_SEC", "3.0"))
DEBUG_CORRECTION_OUTPUT = os.getenv("MERGE_CORRECTION_DEBUG_OUTPUT", "0").strip() == "1"

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

# 테스트/오버라이드용 Gemini 클라이언트 주입
def set_client(client: genai.Client) -> None:
    global _override_client
    _override_client = client


# 현재 설정된 Gemini 클라이언트 반환, 없으면 예외
def _get_client() -> genai.Client:
    if _override_client is None:
        raise RuntimeError("Gemini client가 설정되지 않았습니다.")
    return _override_client


# 스테이지별 카운터에 값 누적 (스레드 안전)
def _add_stage_count(key: str, value: int) -> None:
    with _token_usage_lock:
        _stage_counts[key] = int(_stage_counts.get(key, 0) or 0) + int(value or 0)


# Pass3 판정 감사 로그 레코드 누적
def _add_pass3_candidate_audit(records: list[dict]) -> None:
    if not records:
        return
    with _token_usage_lock:
        _pass3_candidate_audit.extend(records)


# Gemini 응답의 토큰 사용량을 전역/스테이지별 집계에 반영하고 cost_report에도 기록
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


# OpenAI 호환 응답의 토큰 사용량을 전역/스테이지별 집계에 반영하고 cost_report에도 기록
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


# Ollama 모델 여부 판별
def _is_ollama_model(model: str) -> bool:
    spec = str(model or "").strip().lower()
    return spec.startswith("ollama:") or spec.startswith("ollama/")


def _effective_parallel_requests(*models: str) -> int:
    """models 중 하나라도 Ollama면 서버 동시 처리 한도(OLLAMA_PARALLEL_REQUESTS)를,
    전부 상용 API면 기존 PARALLEL_REQUESTS를 반환
    """
    if any(_is_ollama_model(model) for model in models):
        return OLLAMA_PARALLEL_REQUESTS
    return PARALLEL_REQUESTS


def _resolve_ollama_model(model: str) -> tuple[str, Optional[bool]]:
    """`ollama:` prefix를 떼어내고, `#think`/`#nothink` 접미사가 있으면 (모델, think여부)로 반환"""
    spec = str(model or "").strip()
    lowered = spec.lower()
    if lowered.startswith("ollama:"):
        spec = spec.split(":", 1)[1].strip()
    elif lowered.startswith("ollama/"):
        spec = spec.split("/", 1)[1].strip()
    if spec.endswith("#nothink"):
        return spec[: -len("#nothink")].strip(), False
    if spec.endswith("#think"):
        return spec[: -len("#think")].strip(), True
    return spec, None


# OpenAI 호환(Ollama 포함) API로 텍스트 교정 프롬프트 호출, 서버가 특정 파라미터를 거부하면 그것만 제거하고 재시도
def _call_openai_text_correction(
    prompt: str,
    *,
    stage: str,
    model: str,
    system_prompt: str,
    max_output_tokens: int = 8192,
) -> str:
    is_ollama = _is_ollama_model(model)
    if is_ollama:
        from .config import get_ollama_client

        client = get_ollama_client()
        if client is None:
            raise RuntimeError("Ollama 클라이언트를 만들 수 없습니다 (openai 패키지 확인 필요).")
        resolved_model, think_override = _resolve_ollama_model(model)
    else:
        from .config import get_openai_client

        client = get_openai_client()
        if client is None and os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if client is None:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
        resolved_model, think_override = model, None

    def call():
        reasoning_effort = TEXT_REASONING_EFFORT
        if "pass2" in stage:
            reasoning_effort = PASS2_REASONING_EFFORT
        elif "pass3" in stage:
            reasoning_effort = PASS3_REASONING_EFFORT
        kwargs = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": max_output_tokens,
        }
        if is_ollama:
            if think_override is not None:
                kwargs["extra_body"] = {"think": think_override}
        elif reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        temperature = os.getenv("VLVERIFIER_TEXT_PROCESSOR_OPENAI_TEMPERATURE", "0").strip()
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


# 슬라이드 이미지를 첨부해 OpenAI Vision으로 텍스트 교정 후보 생성 (Pass2 이미지 참조 모드)
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


# 재시도 가능한 오류 코드가 감지되면 지수적으로 대기하며 재시도, max_retries<=0이면 무한 재시도
def api_call_with_retry(func, max_retries: int | None = None, initial_wait: int | None = None):
    if max_retries is None:
        max_retries = int(os.getenv("VLVERIFIER_TEXT_PROCESSOR_API_MAX_RETRIES", "0"))
    if initial_wait is None:
        initial_wait = int(os.getenv("VLVERIFIER_TEXT_PROCESSOR_API_INITIAL_WAIT", "10"))
    max_wait = float(os.getenv("VLVERIFIER_API_RETRY_MAX_WAIT_SEC", "60"))
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


# 연속 공백을 하나로 정규화
def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


# 적용 확정된 교정 결과 dict 생성
def _applied(text: str, risk: str, reason: str) -> dict:
    return {
        "candidate_text": text,
        "applied_text": text,
        "risk": risk,
        "apply": True,
        "reason": reason,
    }


# 슬라이드 제목+전사 일부를 보고 강의 도메인/서브도메인 분류
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


# scene별 occurrence를 시간순 평면 리스트로 변환
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


# 전사 세그먼트를 겹치는 시간 구간(occurrence)에 배정, 겹치는 구간이 여러 개면 전환
# 경계 근처(TRANSITION_LEAD_SEC/TAIL_SEC) 휴리스틱으로 다음 구간에 배정할지 판단,
# 겹치는 구간이 없으면 세그먼트 65% 지점에 가장 가까운 구간을 ASSIGN_MAX_GAP_SEC 이내에서 선택
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


# Pass1/2 LLM 응답에서 corrections 배열을 {index: 교정정보} dict로 파싱
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


# 슬라이드 텍스트에서 한글 용어/코드 식별자 후보를 추출해 용어 사전 목록 생성
def extract_glossary_terms(slide_texts: dict[int, dict]) -> list[str]:
    ordered_terms: dict[str, None] = {}

    def add_term(term: str) -> None:
        cleaned = re.sub(r"\s+", " ", str(term or "")).strip()
        cleaned = re.sub(r"^[\d\s\.\)\]-]+", "", cleaned).strip()
        cleaned = cleaned.strip(" \t\r\n,.;:()[]{}<>\"'")
        variants = [cleaned]
        if re.search(r"[가-힣]", cleaned) and re.search(r"[\(\[]", cleaned):
            variants.insert(0, re.split(r"\s*[\(\[]", cleaned, maxsplit=1)[0].strip())
        for variant in variants:
            variant = variant.strip(" \t\r\n,.;:()[]{}<>\"'")
            if len(variant) < 2 or len(variant) > 40:
                continue
            ordered_terms.setdefault(variant, None)

    all_text_parts: list[str] = []
    for _, extracted in slide_texts.items():
        all_text_parts.extend([
            str(extracted.get("title", "") or ""),
            str(extracted.get("text", "") or ""),
            str(extracted.get("t1", "") or ""),
            str(extracted.get("t1_structure", "") or ""),
        ])
    all_text = "\n".join(all_text_parts)

    for raw_unit in re.split(r"[\n\r\t,;:|•●○■□▶]+", all_text):
        unit = re.sub(r"\s+", " ", raw_unit).strip()
        unit = re.sub(r"^[\d\s\.\)\]-]+", "", unit).strip()
        if not re.search(r"[가-힣]", unit):
            continue
        if len(unit) <= 28:
            add_term(unit)

        words = unit.split()
        if len(words) > 6:
            continue
        for size in range(1, min(3, len(words)) + 1):
            for start in range(0, len(words) - size + 1):
                phrase = " ".join(words[start:start + size])
                if re.search(r"[가-힣]", phrase) and (size > 1 or len(phrase) >= 3):
                    add_term(phrase)

    code_terms = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_.()]*\b", all_text))
    func_terms = set(re.findall(r"\b[A-Za-z_]\w*\s*\(", all_text))
    func_terms = {term.strip().rstrip("(").strip() for term in func_terms}
    for term in sorted(code_terms | func_terms):
        if term.isupper() or re.search(r"[\d_.()]", term):
            add_term(term)

    max_terms = int(os.getenv("VLVERIFIER_TEXT_PROCESSOR_MAX_GLOSSARY_TERMS", "800"))
    return list(ordered_terms.keys())[:max_terms]


# 지정 슬라이드 및 인접 슬라이드(window)의 용어만 골라 프롬프트용 용어 사전 블록 생성
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


# metadata에서 scene별 시간 구간과 scene->논리 슬라이드 번호 매핑 추출
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


# metadata에서 scene_index별 슬라이드 번호/canonical/visit 정보 인덱스 생성
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


# 텍스트화 결과에서 논리 슬라이드 번호별 대표 텍스트(가장 정보량 많은 scene) 선택
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
            "image_path": image_path,
            "scene_number": scene_no if isinstance(scene_no, int) else None,
        }
        if current is None:
            result[logical_slide_no] = candidate_entry
            continue
        current_len = len(current.get("text", "")) + len(current.get("title", ""))
        candidate_len = len(candidate_entry.get("text", "")) + len(candidate_entry.get("title", ""))
        if candidate_len > current_len:
            result[logical_slide_no] = candidate_entry
    return result


# Pass1: 슬라이드 참고 없이 문맥+용어 사전만으로 ASR 오인식 교정 후보 생성
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
확신이 없어도 조금이라도 어색하거나 조사/어미/발음이 의심되면 후보로 출력할 것.
Pass1에서 후보를 놓치는 것(recall)이 후보를 과하게 내는 것보다 훨씬 나쁘다 — 최종 판단은 Pass3가 하므로 애매하면 반드시 후보로 낼 것.
후보가 없으면 {{"corrections": []}}를 출력하세요.

### 교정 범위
- ASR 오인식 교정 (원문 발음과 유사하고 문맥에도 맞는 단어로 깨진 텍스트를 복원)
- 전문용어 철자 교정 (같은 발화를 옮긴 표기라는 전제에서 슬라이드를 참고하여 정확한 표기로)
- 맞춤법, 띄어쓰기, 조사 오류 교정
- 불필요한 추임새(자, 뭐, 어, 그) 제거 (의미가 유지될 때만)
- 가장 발음이 인접한 후보를 통해 완전한 문맥이 형성된다면 해당 후보만 적용
- 그리스 문자, 수학 기호, 단위, 전문 기호는 표준 표기로 복원 (예: 뮤→μ, 시그마→σ, 세타헷→θ̂, 시그마제곱→σ², 도씨→°C, 퍼센트→%, 마이크로→μ, 옴→Ω 등 발화된 기호/단위 명칭을 실제 기호·단위 표기로)
- 단어·개념어 교체는 ① 원래 발화를 그렇게 잘못 인식했을 만한 발음·음가 유사성과
  ② 교체 후 앞뒤 문맥이 자연스럽고 의미가 맞는다는 조건이 모두 성립할 때만 후보로 출력
- 영문 음차, 고유명사, 코드·약어, 띄어쓰기·철자 교정은 같은 발화의 표기 복원이므로 문자 모양의
  유사성 대신 실제로 같은 소리를 옮긴 표기인지 판단

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
- 슬라이드의 정답/문맥에 맞추기 위해 강의자의 한국어 개념어를 반대 개념으로 바꾸지 말 것
- 발음·음가 근거 없이 문맥이나 슬라이드에 더 잘 맞는다는 이유만으로 정상적인 한국어 개념어를
  의미가 다른 개념어로 바꾸지 말 것. 두 조건 중 하나라도 불확실하면 후보를 출력하지 말 것
- 각 index의 원문만 수정. 다른 index 내용과 섞지 말 것
- 합성어/전문용어의 띄어쓰기가 붙여써도 띄어써도 둘 다 표준으로 허용되는 경우(예: 응용 프로그램/응용프로그램, 다중 프로그래밍/다중프로그래밍)는 수정하지 말 것 — ASR 오류가 아니라 정상 표기 변형이다
- 받침(종성)을 임의로 추가하거나 제거하지 말 것. 특히 동사 어간의 받침 삭제는 금지(예: "끌 때까지"를 "끄 때까지"로 바꾸는 것처럼 존재하던 받침을 지우면 새로운 비문이 생긴다)

### 중요 원칙
- 강의자의 발화 구조를 절대적으로 따라가라
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
        if _is_ollama_model(PASS1_TEXT_MODEL):
            response_text = _call_openai_text_correction(
                prompt,
                stage="stage3b_text_processor_pass1",
                model=PASS1_TEXT_MODEL,
                system_prompt=(
                    "당신은 한국어 강의 STT 오인식 후보 생성기입니다. "
                    "슬라이드 제목과 용어 사전만 참고하여 ASR 오인식 가능성이 높은 교정 후보를 JSON으로 출력하세요."
                ),
                max_output_tokens=TEXT_MAX_OUTPUT_TOKENS,
            )
        else:
            response = api_call_with_retry(call)
            _add_usage(response, stage="stage3b_text_processor_pass1")
            response_text = response.text or ""
        local_corrections = parse_batch_response(response_text)
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


# Pass2: 슬라이드(텍스트 또는 이미지)를 참고해 Pass1 후보를 보강한 교정 후보 생성
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
    if has_image:
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
- Pass2는 최종 승인 단계가 아니라 후보 생성 단계다. 원문을 그렇게 잘못 들었을 만한 음가 유사성이 있고
  슬라이드/용어 사전/주변 문맥도 같은 복원을 지지하는 후보만 출력한다. 최종 적용 여부는 Pass3가 판단한다.
- Pass2의 출력 text는 반드시 원문을 기준으로 교정한 전체 문장이어야 한다.

### 수정할 것
- STT/ASR 오인식 가능성이 높은 단어/구절: 원문과 후보의 발음·음가가 유사하고, 동시에 원문은 문맥상
  성립하기 어렵지만 후보는 강의 문맥에 맞는 경우. 음가 유사성과 문맥 적합성을 모두 충족해야 한다.
- 슬라이드 또는 용어 사전에 실제로 보이는 강의 용어의 철자/표기 오류. 단, 원문과 후보가 같은 발화를
  옮긴 표기여야 하며 슬라이드에 있다는 사실만으로 다른 개념어를 가져오지 않는다.
- 전문용어가 조사/접사와 붙어 깨진 경우, 전문용어가 포함된 어절 전체를 후보로 복원
- 한국어식 발음이나 깨진 음가로 들어온 영문/외래어/전문용어의 원 표기 복원
- 맞춤법/띄어쓰기는 강의 용어 표기 또는 오인식 복원에 필요한 경우만 수정
- 각 문장을 끝까지 읽고 조사(을/를/이/가/은/는/의/에/에게/야 등)가 빠졌거나 잘못된 종류로 쓰인 곳, 종결어미(-다/-라고/-까/-니 등)가 문법적으로 안 맞는 곳이 있으면 후보로 낼 것

### 수정하지 말 것
- 문장 어미, 말투, 격식, 발화 순서, 문장 구조 변경
- 단순 자연화, 표현 개선, 중복 제거, 필러 제거
- 설명 추가, 의미 보충, 정의문/뜻풀이로 확장
- 슬라이드 정답에 맞추기 위한 개념 교정
- 원문과 발음·음가가 유사하지 않은 다른 개념어를 문맥 또는 슬라이드에 맞는다는 이유만으로 대입하는 것
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
                    "원문과 음가가 유사하고 문맥에도 맞는 오인식 후보만 JSON으로 출력하세요. "
                    "문맥이나 슬라이드 정답에 맞는다는 이유만으로 의미가 다른 개념어를 대입하지 마세요."
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


# Pass3 LLM 응답에서 broken 판정 + 채택된 변경(changes) 목록 파싱, broken에 없는 index는 무시
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

    broken_raw = parsed.get("broken", [])
    broken_indices = {v for v in broken_raw if isinstance(v, int)} if isinstance(broken_raw, list) else set()

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
        if isinstance(parsed.get("broken"), list) and idx not in broken_indices:
            # 1단계(원문 자체 판정)에서 broken으로 표시하지 않은 i는
            # 2단계(후보 선택) 결과를 신뢰하지 않고 무시
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


# Pass1/2 후보가 있는 세그먼트만 모아 Pass3 검증용 item(원문+앞뒤 발화+후보 목록) 구성
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
            candidates.append({"source": "pass1", "text": p1})
        if p2 and p2 != raw_norm and p2 not in [c["text"] for c in candidates]:
            candidates.append({"source": "pass2", "text": p2})
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


# Pass3 item 목록을 프롬프트용 텍스트 블록으로 포맷
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


# Pass3: 원문 자체가 깨졌는지 먼저 판정(broken) 후, 깨진 항목에 한해 후보에서 최소 교체 span만 추출
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
출력에 없는 변경은 적용하지 않는다.
{ref_block}

## 검증 대상
{items_text}

## 판단 절차 (반드시 이 순서로, 두 단계를 분리해서 판단한다)

1단계 — 원문 자체 판정 (후보는 참고하지 않는다):
각 [i]의 "원문"만 읽고, 후보가 있든 없든 상관없이 원문 자체가 그 자리에서 한국어로 성립하기 어려운지 스스로 판단한다.
- 성립하기 어려운 경우만: 발음이 이상한 단어, 전문용어/슬라이드 용어 표기 오류, 문맥상 깨진 어절/구절, 조사가 안 맞는 경우, 영문/코드/고유명사 오인식, 같은 단어가 자기 자신을 가리키며 반복돼 의미가 중복되는 경우(예: "응용 소프트웨어는... 다양한 소프트웨어를 사용한다").
- 이 판단에는 후보 내용을 근거로 쓰지 않는다. "후보가 다르게 썼으니까 원문이 이상하다"는 판단 금지.
- 성립하기 어렵다고 판단한 i만 "broken"에 넣는다. 나머지 i는 "broken"에 넣지 않는다.

2단계 — 후보에서 최소 span 추출 (1단계에서 broken으로 판단한 i에 대해서만 수행):
1. 그 i의 후보를 원문과 왼쪽부터 오른쪽까지 끝까지 비교한다.
2. 원문의 깨진 단어/구절과 후보의 같은 위치 대응어를 찾는다.
3. 하나를 찾은 뒤에도 멈추지 말고, 같은 항목의 남은 차이를 계속 검사한다.
4. 서로 독립된 오류는 하나의 큰 span으로 합치지 말고 별도 changes로 출력한다.
5. 후보 전체가 자연화/재작성이어도, 명확한 단어/구절 대응만 따로 출력한다.
6. from_text는 원문에 실제로 존재해야 하고, to_text는 해당 후보에 실제로 존재해야 한다.
7. 한국어 오류 span은 한 글자나 어절 내부 일부만 자르지 말고, 깨진 어절/용어 단위로 잡는다.
   - 금지: 외 -> 회, 한해 -> 환산, 재활 -> 재화
   - 허용: 관리외계 -> 관리회계, 외계에서 -> 회계에서, 한해봐야겠죠 -> 환산해봐야겠죠, 재활을 -> 재화를
8. 조사/어미와 결합했을 때 최종 문장이 깨지면, 조사/어미까지 포함한 span을 출력한다.
9. broken에 없는 i는 changes에 넣지 않는다. 여러 후보가 상충하면, 1단계에서 찾은 원문의 문제를 실제로 해소하는 후보를 우선한다.

## 출력(JSON만)
{{
  "broken":[0,3],
  "changes":[
    {{"i":0,"c":1,"from":"원문 연속 문자열","to":"후보 근거 문자열","occ":1,"r":1}}
  ]
}}

reason_code는 아래 번호 중 하나만 사용한다. 번호를 고를 수 없으면 그 변경은 출력하지 않는다.
1 발음/음가 유사 오인식
2 강의 도메인 전문용어 복원
3 슬라이드 표기 용어 복원
4 음가상 오인식 가능성이 있고 문맥상 깨진 구절 복원. 문맥만 맞는 다른 개념어 치환에는 사용 금지
5 깨진 용어에 붙은 조사까지 복원
6 영문/코드/약어 오인식 복원
7 고유명사 복원

reason_code 2·3·4·6·7도 음가 근거 없이 다른 개념을 넣는 예외가 아니다. 전문용어·슬라이드 용어·영문
음차·고유명사는 원문과 후보가 같은 발화를 옮긴 표기일 때만 해당 코드를 사용할 수 있다.

## 실행 순서
1. 후보는 정답이 아니다. 각 변경을 원문에 적용한 완성 문장을 이전·다음 발화와 함께 먼저 읽고,
   해당 변경이 ASR 복원인지 의미가 다른 개념을 문맥에 맞춰 넣은 것인지 판단한다.
2. 일반 단어·개념어 교체는 원문을 후보처럼 잘못 인식했을 만한 발음·음가 유사성과, 교체 후 문맥
   적합성이 모두 명확한 경우에만 출력한다. 문맥 또는 슬라이드만 맞으면 출력하지 않는다.
3. 각 후보를 원문과 왼쪽부터 오른쪽까지 끝까지 비교한다.
4. 원문의 깨진 단어/구절과 후보의 같은 위치 대응어를 찾는다.
5. 하나를 찾은 뒤에도 멈추지 말고, 같은 항목의 남은 차이를 계속 검사한다.
6. 서로 독립된 오류는 하나의 큰 span으로 합치지 말고 별도 changes로 출력한다.
7. 후보 전체가 자연화/재작성이어도, 위 두 조건을 충족하는 명확한 단어/구절 대응만 따로 출력한다.
8. from_text는 원문에 실제로 존재해야 하고, to_text는 해당 후보에 실제로 존재해야 한다.
9. 한국어 오류 span은 한 글자나 어절 내부 일부만 자르지 말고, 깨진 어절/용어 단위로 잡는다.
   - 금지: 외 -> 회, 한해 -> 환산, 재활 -> 재화
   - 허용: 관리외계 -> 관리회계, 외계에서 -> 회계에서, 한해봐야겠죠 -> 환산해봐야겠죠, 재활을 -> 재화를
10. 조사/어미와 결합했을 때 최종 문장이 깨지면, 조사/어미까지 포함한 span을 출력한다.

출력 키 의미:
- broken: 1단계에서 원문 자체가 성립하기 어렵다고 판단한 [] 번호 목록
- i: 검증 대상의 [] 번호 (반드시 broken에 포함된 번호여야 한다)
- c: 후보 번호
- from: 원문에 실제로 있는 연속 문자열
- to: 후보 문장에 실제로 있는 문자열
- occ: 같은 from이 원문에 여러 번 있으면 몇 번째인지, 모두면 "all"
- r: reason_code

## accepted 기준
- 고유명사, 영문 토큰, 코드/함수명, 제품명, 슬라이드 표기 용어, 운영체제/컴퓨터 구조 전문용어는
  원문과 후보가 같은 발화를 옮긴 표기라는 조건 아래 오인식을 적극 출력한다.
- 같은 강의에서 반복되는 전문용어의 오인식도 적극 출력한다.
- 영문 음차·고유명사·코드·약어·철자·띄어쓰기 복원은 같은 발화를 다른 표기로 옮긴 경우에만 출력한다.
- 일반 명사구/술어구/개념어는 원문이 문맥상 성립하기 어렵고, 후보가 같은 위치에서 음가상 유사하며
  문맥상으로도 자연스럽게 복원될 때만 출력한다. 음가와 문맥은 선택 조건이 아니라 모두 필요한 조건이다.
- 비문은 전체 문장을 고치지 말고, 비문을 만든 깨진 단어/구절만 출력한다.

## rejected span
- 문장 전체 또는 절 전체
- 어순 변경, 주어/목적어/서술어 보충
- 원문 단어 삭제, 설명 추가, 의미 추가/삭제
- 발음·음가 근거 없이 문맥·슬라이드에 맞추기 위해 정상적인 개념어를 의미가 다른 개념어로 바꾸는 변경
- 적용 후 같은 개념어가 부자연스럽게 반복되거나, 앞뒤 발화와 새로운 모순을 만드는 변경
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
                "먼저 후보를 보지 않고 원문 자체가 성립하기 어려운 항목만 broken으로 판정한 뒤, "
                "broken인 항목에 대해서만 후보에서 실제로 교체할 최소 from_text/to_text를 출력하세요. "
                "후보 문장 전체를 승인/거절하지 마세요. "
                "후보는 정답이 아니므로 각 편집을 적용한 완성 문장을 먼저 검증하세요. "
                "일반 단어와 개념어는 원문과 음가가 유사하고 문맥에도 맞는 경우에만 승인하며, "
                "문맥이나 슬라이드 정답만으로 의미가 다른 개념을 대입한 편집은 거절하세요. "
                "후보 전체를 일괄 승인하지 말고 검증을 통과한 최소 from_text/to_text만 출력하세요. "
                "문장 전체, 절 전체, 새 문장은 출력하지 마세요. "
                "JSON만 출력하세요."
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


# 공백/문장부호 차이만 있는 표면적 편집인지 확인 (실질적 변경이 아니면 True)
def _is_surface_only_edit(edit: dict) -> bool:
    from_text = str(edit.get("from_text", "") or "")
    to_text = str(edit.get("to_text", "") or "")
    if not from_text or not to_text:
        return False
    strip_chars = r"[\s\.,;:!\?。．，、…]+"
    return re.sub(strip_chars, "", from_text) == re.sub(strip_chars, "", to_text)


# 완성형 한글 음절 여부 확인
def _is_hangul_char(value: str) -> bool:
    return bool(value) and "가" <= value <= "힣"


# 1~2글자 짧은 한글 교체가 앞뒤 한글과 붙어 다른 단어를 오염시킬 위험이 있는지 확인
def _is_unsafe_short_korean_span(raw_text: str, from_text: str, to_text: str, raw_start: int, raw_end: int) -> bool:
    if len(from_text) > 2 and len(to_text) > 2:
        return False
    if not re.search(r"[가-힣]", from_text + to_text):
        return False
    left = raw_text[raw_start - 1] if raw_start > 0 else ""
    right = raw_text[raw_end] if raw_end < len(raw_text) else ""
    return _is_hangul_char(left) or _is_hangul_char(right)


# LLM이 낸 from/to 변경 1건을 원문 내 실제 위치(raw_start/end)로 확장, 후보 문장에 없는
# to_text·표면적 편집·불안전한 짧은 span은 제외
def _expand_llm_change(raw_text: str, change: dict, candidates: list[dict]) -> list[dict]:
    raw_text = str(raw_text or "")
    from_text = str(change.get("from_text", "") or "")
    to_text = str(change.get("to_text", "") or "")
    if not from_text or not to_text:
        return []
    if _is_surface_only_edit({"from_text": from_text, "to_text": to_text}):
        return []

    candidate_number = change.get("candidate_number")
    if isinstance(candidate_number, int) and 1 <= candidate_number <= len(candidates):
        candidate_text = str(candidates[candidate_number - 1].get("text", "") or "")
        if to_text not in candidate_text:
            return []

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
    expanded: list[dict] = []
    for raw_start in selected_starts:
        raw_end = raw_start + len(from_text)
        if _is_unsafe_short_korean_span(raw_text, from_text, to_text, raw_start, raw_end):
            continue
        expanded.append({
            "source": "pass3_llm",
            "candidate_number": candidate_number if isinstance(candidate_number, int) else 0,
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


_BATCHIM_PARTICLE_PAIRS = [("을", "를"), ("은", "는"), ("이", "가"), ("과", "와")]
_BATCHIM_PARTICLE_ONE_CHARS = {ch for pair in _BATCHIM_PARTICLE_PAIRS for ch in pair}


def _hangul_batchim_index(ch: str) -> Optional[int]:
    """완성형 한글 음절의 종성(받침) 인덱스. 0=받침없음, 1~27=받침 있음. 한글 음절이 아니면 None."""
    if not ch:
        return None
    code = ord(ch)
    if not (0xAC00 <= code <= 0xD7A3):
        return None
    return (code - 0xAC00) % 28


def _fix_trailing_batchim_particle(text: str, insert_end: int, last_inserted_char: str) -> str:
    """to_text 삽입 직후 원문에 남아있던 조사(을/를, 은/는, 이/가, 과/와, 으로/로)를
    방금 삽입한 단어의 받침 유무에 맞게 결정론적으로 교정

    최소 span 교체 방식상 후보 단어만 바뀌고 뒤에 남은 조사는 원문 그대로 남는데,
    받침 유무가 바뀌는 치환(예: 욕구를->목적을)에서 조사가 안 맞게 되는 문제를 막음
    한글 음절이 아니거나(영어/숫자/기호) 판정 불가능한 경우는 건드리지 않음
    """
    final_index = _hangul_batchim_index(last_inserted_char)
    if final_index is None:
        return text
    has_batchim = final_index != 0

    one = text[insert_end:insert_end + 1]
    two = text[insert_end:insert_end + 2]
    if two == "으로" or one == "로":
        current_len = 2 if two == "으로" else 1
        # ㄹ받침(final_index==8)은 으로가 아니라 로를 씀
        want_short = (not has_batchim) or final_index == 8
        correct = "로" if want_short else "으로"
        current = two if current_len == 2 else one
        if current != correct:
            return text[:insert_end] + correct + text[insert_end + current_len:]
        return text

    if one in _BATCHIM_PARTICLE_ONE_CHARS:
        for with_batchim, without_batchim in _BATCHIM_PARTICLE_PAIRS:
            if one in (with_batchim, without_batchim):
                correct = with_batchim if has_batchim else without_batchim
                if one != correct:
                    return text[:insert_end] + correct + text[insert_end + 1:]
                break
    return text


# 검증된 편집들을 겹치지 않는 것만 골라 원문에 적용, 적용 후 받침 조사도 함께 보정
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
        if to_text:
            corrected = _fix_trailing_batchim_particle(corrected, raw_start + len(to_text), to_text[-1])
    return corrected, valid_edits


# Pass1/2 후보로 Pass3 item을 만들고 곧바로 검증까지 실행하는 편의 함수
def merge_two_passes(
    batch: list[tuple[int, dict]],
    pass1: dict[int, str],
    pass2: dict[int, str],
    subdomain: str = "",
    slide_context: str = "",
) -> dict[int, dict]:
    pass3_items = _build_pass3_items(batch, pass1, pass2)
    return merge_pass3_items(pass3_items, slide_context=slide_context)


# Pass3 검증 결과를 실제 원문에 적용해 최종 교정 dict 구성
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


# Pass2 입력용 배치 복사 (현재는 Pass1 결과를 별도 반영하지 않고 그대로 복제)
def _build_pass2_batch_from_pass1(
    batch: list[tuple[int, dict]],
    pass1: dict[int, str],
) -> list[tuple[int, dict]]:
    return [(global_i, seg.copy()) for global_i, seg in batch]


def _prepare_pass3_jobs(
    segments: list[dict],
    metadata: list[dict],
    textualized_data: dict,
    textualized_dir: Path,
    use_pass2: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> dict:
    """Pass1/Pass2를 실행해 Pass3 검증용 후보 job 생성

    Pass3 재현성 테스트 등에서 동일한 Pass1/2 후보 세트를 재사용하기 위해
    correct_segments_three_pass에서 Pass1/2 단계만 분리

    progress_callback(band, done, total)이 주어지면 job이 하나씩 끝날 때마다
    "pass1"/"pass2" 두 band로 나눠 보고 — 한 job 안에서 Pass1을 마친 시점과
    (use_pass2면) Pass2까지 마친 시점을 따로 신호
    """
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

    groups: dict[int, list[tuple[int, dict]]] = {}
    group_scene_indices: dict[int, set[int]] = {}
    no_slide: list[tuple[int, dict]] = []
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

    glossary_window = max(0, int(os.getenv("VLVERIFIER_TEXT_PROCESSOR_GLOSSARY_SLIDE_WINDOW", "1")))
    total_glossary_terms = len(extract_glossary_terms(extracted_slide_texts))
    if total_glossary_terms:
        print(
            f"    용어 사전: 전체 {total_glossary_terms}개 용어, "
            f"batch별 현재±{glossary_window} 슬라이드만 사용"
        )

    progress_lock = Lock()
    pass1_done = 0
    pass2_done = 0

    def _tick(band: str) -> None:
        # parallel_jobs는 이 함수가 호출되는 시점(=executor 제출 이후)에는 이미 다
        # 채워져 있으므로, 총량으로 그냥 len(parallel_jobs)를 읽어도 됨
        nonlocal pass1_done, pass2_done
        if not progress_callback:
            return
        with progress_lock:
            if band == "pass1":
                pass1_done += 1
                done = pass1_done
            else:
                pass2_done += 1
                done = pass2_done
        progress_callback(band, done, len(parallel_jobs))

    def process_sub_batch(
        sub: list[tuple[int, dict]],
        context: str,
        slide_title: str = "",
        glossary: str = "",
    ) -> tuple[str, list[dict]]:
        pass1 = _correct_batch_pass1(sub, slide_title=slide_title, glossary=glossary)
        _tick("pass1")
        if not use_pass2:
            # Pass2를 아예 안 쓰는 실행이면 그 band는 각 job이 끝나는 대로 바로
            # 다 찬 것으로 봄 — Pass1과 동시에 완료 신호를 보냄
            _tick("pass2")
            return context, _build_pass3_items(sub, pass1, {})

        pass2_batch = _build_pass2_batch_from_pass1(sub, pass1)
        pass2 = _correct_batch_pass2(pass2_batch, context, glossary=glossary)
        _tick("pass2")
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
            glossary = build_local_glossary(
                extracted_slide_texts,
                logical_slide_no,
                window=glossary_window,
            )
            parallel_jobs.append((sub, context, slide_title, glossary))

    if no_slide:
        print(f"    미매핑 {len(no_slide)}개 교정 중...")
        for offset in range(0, len(no_slide), BATCH_SIZE):
            sub = no_slide[offset:offset + BATCH_SIZE]
            parallel_jobs.append((sub, "", "", ""))

    if not parallel_jobs and progress_callback:
        # 교정 대상 자체가 없으면 job이 하나도 안 돌아 _tick이 안 불림 — 두 band를
        # 즉시 완료 처리해서 뒤 단계(그룹화)가 기다리지 않게 함
        progress_callback("pass1", 0, 0)
        progress_callback("pass2", 0, 0)

    pass12_workers = _effective_parallel_requests(PASS1_TEXT_MODEL, PASS2_TEXT_MODEL)
    print(
        f"    총 {len(parallel_jobs)}개 batch를 최대 {pass12_workers}개 병렬로 교정 중...",
        end="",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=pass12_workers) as executor:
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

    return {
        "pass3_jobs": pass3_jobs,
        "seg_scene": seg_scene,
        "scene_meta_by_index": scene_meta_by_index,
    }


def _run_pass3_jobs(
    pass3_jobs: list[tuple[list[dict], str]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict[int, dict]:
    """Pass3(merge_pass3_items)만 실행

    PASS3_TEXT_MODEL을 바꿔가며 동일한 pass3_jobs(=동일 Pass1/2 후보)를 재사용해
    재현성을 테스트할 때 사용
    """
    all_corrections: dict[int, dict] = {}
    total_pass3_items = sum(len(items) for items, _ in pass3_jobs)
    if pass3_jobs:
        pass3_workers = _effective_parallel_requests(PASS3_TEXT_MODEL)
        total_jobs = len(pass3_jobs)
        done_jobs = 0
        print(
            f"    Pass3 후보 {total_pass3_items}개를 {total_jobs}개 batch로 검증 중...",
            end="",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=pass3_workers) as executor:
            futures = [
                executor.submit(merge_pass3_items, items, context)
                for items, context in pass3_jobs
            ]
            for future in as_completed(futures):
                all_corrections.update(future.result())
                print(".", end="", flush=True)
                if progress_callback:
                    done_jobs += 1
                    progress_callback(done_jobs, total_jobs)
        print()
    elif progress_callback:
        # 검증할 Pass3 후보가 아예 없으면 band를 즉시 완료 처리
        progress_callback(0, 0)
    return all_corrections


# 원본 세그먼트에 scene/슬라이드 정보와 최종 교정 결과(적용/미적용)를 반영해 완성
def _assemble_corrected_segments(
    segments: list[dict],
    seg_scene: dict[int, int],
    scene_meta_by_index: dict[int, dict],
    all_corrections: dict[int, dict],
) -> list[dict]:
    corrected_segments: list[dict] = []
    applied_count = 0
    for i, seg in enumerate(segments):
        corrected = seg.copy()
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
        original = seg.get("text_original", seg["text"])
        corrected["text_raw"] = original
        corrected["text_corrected"] = original
        corrected["text_original"] = original

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
                corrected["text"] = original
                corrected["correction_status"] = "unchanged"
        else:
            corrected["text"] = original
            corrected["correction_status"] = "unchanged"
            corrected["correction_risk"] = "none"
            corrected["correction_reason"] = ""

        corrected_segments.append(corrected)

    print(f"    교정 적용: {applied_count}건 / 후보 출력: 0건 / 전체 {len(segments)}건")
    return corrected_segments


def correct_segments_three_pass(
    segments: list[dict],
    metadata: list[dict],
    textualized_data: dict,
    textualized_dir: Path,
    use_pass2: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> list[dict]:
    """progress_callback(band, done, total)이 주어지면 "pass1"/"pass2"/"pass3" 세 band로
    나눠 실측 진행률을 보고, band별 총량은 그 band가 시작되는 시점에만 알면 됨
    (Pass1/2 총량은 job 리스트가 다 만들어진 직후, Pass3 총량은 Pass1/2가 끝난 직후)
    """
    if not segments:
        if progress_callback:
            progress_callback("pass1", 0, 0)
            progress_callback("pass2", 0, 0)
            progress_callback("pass3", 0, 0)
        return []

    prepared = _prepare_pass3_jobs(
        segments, metadata, textualized_data, textualized_dir, use_pass2,
        progress_callback=progress_callback,
    )
    pass3_callback = None
    if progress_callback:
        pass3_callback = lambda done, total: progress_callback("pass3", done, total)
    all_corrections = _run_pass3_jobs(prepared["pass3_jobs"], progress_callback=pass3_callback)
    return _assemble_corrected_segments(
        segments,
        prepared["seg_scene"],
        prepared["scene_meta_by_index"],
        all_corrections,
    )


# 외부 호출 호환용 별칭, 구현은 현재 3-pass
correct_segments_two_pass = correct_segments_three_pass
