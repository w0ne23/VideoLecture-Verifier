# 슬라이드 전환 판정을 보조하는 OCR 힌트 조회/비교 유틸, OCR 서비스(RapidOCR/Nemotron)와 통신
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# OCR 서비스 설정용 환경변수 키 목록
_OCR_PROVIDER_ENV = "VLVERIFIER_SLIDE_OCR_PROVIDER"
_OCR_BASE_URL_ENV = "VLVERIFIER_SLIDE_OCR_BASE_URL"
_OCR_MODEL_DIR_ENV = "VLVERIFIER_SLIDE_OCR_MODEL_DIR"
_OCR_LANG_ENV = "VLVERIFIER_SLIDE_OCR_LANG"
_OCR_TIMEOUT_ENV = "VLVERIFIER_SLIDE_OCR_TIMEOUT_SEC"
_OCR_SIMILARITY_THRESHOLD_ENV = "VLVERIFIER_SLIDE_OCR_SIMILARITY_THRESHOLD"


# 환경변수를 bool로 파싱, 1/true/yes/on만 True
def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# 환경변수를 float로 파싱, 실패/미설정 시 default
def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# 슬라이드 동일 판정에 쓰는 OCR 텍스트 유사도 임계값, 0~1로 clamp
def ocr_similarity_threshold() -> float:
    return max(0.0, min(1.0, env_float(_OCR_SIMILARITY_THRESHOLD_ENV, 0.83)))


# 설정된 OCR provider 이름, 미설정이면 'none'
def ocr_provider() -> str:
    return str(os.getenv(_OCR_PROVIDER_ENV, "none") or "none").strip().lower()


# OCR 힌트 기능 활성화 여부
def ocr_enabled() -> bool:
    return ocr_provider() not in {"", "none", "off", "false", "0"}


# OCR 서비스 base URL, override 우선 적용
def ocr_base_url(override: str | None = None) -> str:
    if override:
        return str(override).strip().rstrip("/")
    return str(os.getenv(_OCR_BASE_URL_ENV, "http://ocr:8000") or "http://ocr:8000").strip().rstrip("/")


# OCR 모델 디렉터리 경로
def ocr_model_dir() -> str:
    return str(os.getenv(_OCR_MODEL_DIR_ENV, "") or "").strip()


# OCR 언어 설정
def ocr_lang() -> str:
    return str(os.getenv(_OCR_LANG_ENV, "multilingual") or "multilingual").strip().lower()


# 비교를 위해 OCR 텍스트를 정규화 (NFC, 소문자, 공백/특수문자 제거)
def normalize_ocr_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or "")).lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w가-힣0-9]+", "", text)
    return text.strip()


# 두 문자열 간 편집 거리 계산
def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for i, left_ch in enumerate(left, start=1):
        current = [i]
        for j, right_ch in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if left_ch == right_ch else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


# 편집 거리 기반 유사도(0~1), 거리, 최대 길이 반환
def levenshtein_similarity(left: str, right: str) -> tuple[float, int, int]:
    max_len = max(len(left), len(right))
    if max_len == 0:
        return 1.0, 0, 0
    distance = levenshtein_distance(left, right)
    return max(0.0, 1.0 - (distance / max_len)), distance, max_len


# 로컬 경로를 OCR 서비스 컨테이너 내부 경로(/app 기준)로 변환
def _service_path_for(path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        rel = path.resolve().relative_to(repo_root)
    except Exception:
        return str(path.resolve())
    return str(Path("/app") / rel)


# 다양한 형태의 OCR 응답을 문자열 라인 목록으로 재귀 정규화
def _normalize_text_lines(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, (list, tuple)):
        lines: list[str] = []
        for item in value:
            lines.extend(_normalize_text_lines(item))
        return lines
    if isinstance(value, dict):
        lines: list[str] = []
        for key in ("lines", "text", "ocr_txts", "texts", "paragraphs", "content"):
            if key in value:
                lines.extend(_normalize_text_lines(value.get(key)))
        return lines

    for attr in ("lines", "text", "ocr_txts", "texts", "paragraphs", "content"):
        if hasattr(value, attr):
            return _normalize_text_lines(getattr(value, attr))

    if hasattr(value, "to_dict"):
        try:
            return _normalize_text_lines(value.to_dict())
        except Exception:
            return []

    text = str(value).strip()
    return [text] if text else []


# 라인 중복 제거 및 길이 제한
def _compact_lines(lines: list[str], *, max_lines: int = 80, max_chars: int = 6000) -> str:
    compact: list[str] = []
    prev = ""
    for line in lines:
        line = " ".join(str(line).split()).strip()
        if not line:
            continue
        if line == prev:
            continue
        compact.append(line)
        prev = line
        if len(compact) >= max_lines:
            compact.append("...")
            break
    text = "\n".join(compact).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


# OCR 응답 body에서 텍스트 콘텐츠 추출, 알려진 키 순서대로 시도
def _response_content(body: dict[str, Any]) -> str:
    for key in ("text", "content", "ocr_text"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("lines", "ocr_txts", "texts", "paragraphs"):
        value = body.get(key)
        if value:
            return "\n".join(_normalize_text_lines(value))
    return ""


# 이미지 경로+mtime+size+설정을 캐시 키로 삼아 OCR 결과를 메모리 캐시, 실패 시 빈 문자열
@lru_cache(maxsize=2048)
def _ocr_hint_for_path_cached(
    image_path: str,
    mtime_ns: int,
    size: int,
    provider: str,
    base_url: str,
    model_dir: str,
    lang: str,
) -> str:
    if provider in {"", "none", "off", "false", "0"}:
        return ""

    payload = {
        "image_path": image_path,
        "model_dir": model_dir,
        "lang": lang,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/ocr",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = env_float(_OCR_TIMEOUT_ENV, 300.0)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return ""

    if not isinstance(body, dict):
        return ""
    text = _response_content(body)
    if not text:
        text = _compact_lines(_normalize_text_lines(body))
    return _compact_lines(_normalize_text_lines(text))


# OCR 서비스에 이미지를 요청해 텍스트 결과 조회 (캐시 없이 즉시 호출)
def _fetch_slide_ocr_text(image_path: str | Path, *, base_url: str | None = None) -> str:
    path = Path(image_path)
    service_path = _service_path_for(path)
    payload = {
        "image_path": service_path,
        "model_dir": ocr_model_dir(),
        "lang": ocr_lang(),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{ocr_base_url(base_url)}/ocr",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = env_float(_OCR_TIMEOUT_ENV, 300.0)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict):
        return ""
    text = _response_content(body)
    if not text:
        text = _compact_lines(_normalize_text_lines(body))
    return _compact_lines(_normalize_text_lines(text))


# 슬라이드 이미지의 OCR 힌트 텍스트 조회, 비활성화 상태거나 파일 없으면 빈 문자열, 캐시 활용
def get_slide_ocr_hint(
    image_path: str | Path,
    *,
    base_url: str | None = None,
    force: bool = False,
) -> str:
    enabled = force or ocr_enabled()
    if not enabled:
        return ""
    path = Path(image_path)
    if not path.exists():
        return ""
    service_path = _service_path_for(path)
    try:
        stat = path.stat()
    except OSError:
        return ""
    return _ocr_hint_for_path_cached(
        service_path,
        int(stat.st_mtime_ns),
        int(stat.st_size),
        ocr_provider() if ocr_enabled() else ("http" if base_url else ocr_provider()),
        ocr_base_url(base_url),
        ocr_model_dir(),
        ocr_lang(),
    )


# 두 OCR 텍스트를 정규화 후 Levenshtein 유사도로 비교해 merge/reject/uncertain 판정
def compare_ocr_texts(left_text: str, right_text: str) -> dict[str, Any]:
    left_norm = normalize_ocr_text(left_text)
    right_norm = normalize_ocr_text(right_text)
    threshold = ocr_similarity_threshold()
    distance = None
    max_len = max(len(left_norm), len(right_norm))
    if not left_norm or not right_norm:
        similarity = 0.0
        decision = "unavailable"
    else:
        similarity, distance, max_len = levenshtein_similarity(left_norm, right_norm)
        if left_norm == right_norm:
            decision = "merge"
        elif similarity >= threshold:
            decision = "merge"
        elif similarity <= threshold - 0.05:
            decision = "reject"
        else:
            decision = "uncertain"
    return {
        "algorithm": "levenshtein",
        "distance": distance,
        "max_length": max_len,
        "left_text": left_text,
        "right_text": right_text,
        "left_normalized": left_norm,
        "right_normalized": right_norm,
        "similarity": similarity,
        "threshold": threshold,
        "decision": decision,
    }


# 두 슬라이드 이미지의 OCR 힌트를 조회해 비교, base_url 지정 시 캐시 없이 직접 조회
def compare_slide_ocr(left_path: str | Path, right_path: str | Path, *, base_url: str | None = None) -> dict[str, Any]:
    if base_url:
        left = _fetch_slide_ocr_text(left_path, base_url=base_url)
        right = _fetch_slide_ocr_text(right_path, base_url=base_url)
    else:
        left = get_slide_ocr_hint(left_path)
        right = get_slide_ocr_hint(right_path)
    result = compare_ocr_texts(left, right)
    result["left_path"] = str(left_path)
    result["right_path"] = str(right_path)
    if base_url:
        result["base_url"] = str(base_url)
    return result


# LLM 프롬프트에 삽입할 OCR 힌트 블록 텍스트 생성
def format_ocr_hint_block(image_path: str | Path, *, label: str | None = None) -> str:
    hint = get_slide_ocr_hint(image_path)
    if not hint:
        return ""
    label_text = label or Path(image_path).name
    return (
        "[Pre-extracted OCR hint - use only as a hint, trust the image if they conflict]\n"
        f"Image: {label_text}\n"
        f"{hint}\n"
    )
