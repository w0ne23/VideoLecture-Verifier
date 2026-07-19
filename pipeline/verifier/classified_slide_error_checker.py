"""Slide error checker for the classified issue pipeline.

The classified pipeline keeps its own output shape. The slide image is the
source of truth, OCR text is only a hint, and whitespace/layout/style
suggestions are not reportable slide errors.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .issue_type_classifier import (
    TOKEN_USAGE_FIELDS,
    _load_env,
    _split_csv,
)


SCHEMA_VERSION = "classified_slide_error_checker.v3"
DEFAULT_MODELS = ("gpt",)
DEFAULT_BATCH_SIZE = 5
DEFAULT_MIN_SCORE = 0.0

SLIDE_ERROR_TYPES = {
    "text_error": "철자/표기 오류",
    "numeric_unit": "숫자/단위 표기 오류",
    "code_syntax": "코드/수식 문법 오류",
    "visual_defect": "이미지 깨짐·텍스트 겹침 등 시각적 결함",
    "other": "기타 슬라이드 표면 오류",
}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def _strip_json_fence(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _default_models() -> list[str]:
    _load_env()
    configured = (
        _split_csv(os.getenv("CLASSIFIED_SLIDE_ERROR_MODELS"))
        or _split_csv(os.getenv("VERIFIER_CLASSIFIED_SLIDE_ERROR_MODELS"))
    )
    return configured or list(DEFAULT_MODELS)


def _chunk(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _norm_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _compact_no_space(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _existing_path(raw_path: object) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    candidates = [path]
    if str(path).startswith("/pipeline/"):
        try:
            candidates.append(Path.cwd() / path.relative_to("/pipeline"))
        except ValueError:
            pass
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path if path.is_absolute() else None


def _find_slide_image(img_dir: str | None, slide_no: int) -> Path | None:
    if not img_dir:
        return None
    base = Path(img_dir)
    candidates = (
        base / f"slide_{slide_no:03d}_base.jpg",
        base / f"slide_{slide_no:03d}_base.png",
        base / f"slide_{slide_no:03d}_start.jpg",
        base / f"slide_{slide_no:03d}_start.png",
        base / f"slide_{slide_no:03d}_end.jpg",
        base / f"slide_{slide_no:03d}_end.png",
    )
    return next((path for path in candidates if path.exists()), None)


def _resolve_img_dir(merged_payload: dict[str, Any], merged_path: str | Path | None) -> str | None:
    detector_log = str(merged_payload.get("source_detector_log", "") or "").strip()
    candidates: list[Path] = []
    if detector_log:
        candidates.append(Path(detector_log).parent)
    if merged_path:
        candidates.append(Path(merged_path).resolve().parent / "slides")
        candidates.append(Path(merged_path).resolve().parent.parent / "slides")
    for img_dir in candidates:
        if img_dir.is_dir() and (
            any(img_dir.glob("slide_*_base.*"))
            or any(img_dir.glob("slide_*_start.*"))
            or any(img_dir.glob("scene_*_base.*"))
            or any(img_dir.glob("scene_*_annot_*.jpg"))
        ):
            return str(img_dir)
    return None


def _slide_number(slide: dict[str, Any]) -> int | None:
    for key in ("slide_number", "slide_canonical_number", "scene_number"):
        try:
            number = int(slide.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return None


def _slides_from(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("slides")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    rows = payload.get("scenes")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _image_candidates(*payloads: dict[str, Any]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for payload in payloads:
        for item in _slides_from(payload):
            image_path = str(item.get("image_path", "") or "").strip()
            if not image_path:
                continue
            text = _norm_text(
                "\n".join(
                    part
                    for part in (
                        str(item.get("slide_text", "") or ""),
                        str(item.get("t1", "") or ""),
                        str(item.get("t1_structure", "") or ""),
                    )
                    if part
                )
            )
            candidates.append(
                {
                    "title": _norm_text(item.get("title")),
                    "text": text,
                    "image_path": image_path,
                }
            )
    return candidates


def _attach_slide_image_paths(
    slides: list[dict[str, Any]],
    textualized_payload: dict[str, Any],
    classified_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = _image_candidates(textualized_payload, classified_payload)
    if not candidates:
        return slides

    enriched: list[dict[str, Any]] = []
    for slide in slides:
        item = dict(slide)
        if item.get("image_path"):
            enriched.append(item)
            continue

        title = _norm_text(item.get("title") or item.get("slide_title"))
        slide_text = _norm_text(item.get("slide_text") or item.get("text"))
        matches = [candidate for candidate in candidates if candidate["title"] == title]
        if not matches:
            enriched.append(item)
            continue

        def score(candidate: dict[str, str]) -> tuple[int, int]:
            candidate_text = candidate["text"]
            if candidate_text and (candidate_text in slide_text or slide_text in candidate_text):
                overlap = max(len(candidate_text), len(slide_text))
            else:
                slide_tokens = set(slide_text.split())
                candidate_tokens = set(candidate_text.split())
                overlap = len(slide_tokens & candidate_tokens)
            return (overlap, len(candidate_text))

        best = max(matches, key=score)
        item["image_path"] = best["image_path"]
        enriched.append(item)
    return enriched


def _slides_for_check(
    *,
    merged_payload: dict[str, Any],
    textualized_payload: dict[str, Any],
    classified_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    slides = _slides_from(merged_payload)
    if not slides:
        slides = _slides_from(textualized_payload) or _slides_from(classified_payload)
    rows = [dict(slide) for slide in slides]
    rows = _attach_slide_image_paths(rows, textualized_payload, classified_payload)
    rows.sort(key=lambda item: int(item.get("slide_number") or item.get("slide_canonical_number") or 0))
    return rows


def _build_slide_error_prompt(slide_no: int, title: str, slide_text: str) -> str:
    return f"""당신은 강의 슬라이드에서 눈에 보이는 오류를 찾는 검수자입니다.

중요:
- 슬라이드 이미지가 원본입니다.
- 아래 OCR 텍스트는 보조 정보일 뿐이며, 이미지와 다르면 이미지를 우선하세요.
- 내용의 사실성, 더 좋은 표현, 문체 개선, 미적 취향(배치·색상 선택 등 디자인 문제)은 보고하지 마세요.
- 애매하면 보고하지 마세요.

대상 슬라이드:
- 번호: {slide_no}
- OCR 제목: {title}
- OCR 본문:
{slide_text[:3000]}

보고할 것 (error_type별 기준):
- text_error: 이미지에서 명백하게 보이는 한글/영문 철자·표기 오류
- numeric_unit: 숫자/단위 오기. 단위 기호 자체가 틀린 경우뿐 아니라, 강의 맥락이나 도메인 지식으로
  해석할 필요 없이 이미 그 자체로 하나의 값이 고정되어 있는 수치(상수, 정의상 고정된 환산 비율 등)가
  실제 값과 다르게 적힌 경우도 포함합니다. 이런 고정값의 오기는 아래 "사실 오류" 배제 대상이
  아닙니다 — 문맥 판단이 필요 없는, 이미 정답이 하나로 확정된 값이기 때문입니다.
- code_syntax: 코드나 수식이 실제 문법 규칙을 명백히 위반한 경우(괄호 불일치, 연산자 누락 등).
  의사코드나 개념 설명을 위해 의도적으로 단순화한 표기는 문법 오류로 보지 마세요.
  괄호·중괄호·대괄호 짝은 전체적인 느낌만으로 판단하지 말고, 코드나 수식이 있는 줄마다 여는 기호
  (소괄호·대괄호·중괄호)와 닫는 기호를 하나씩 순서대로 대응시켜 개수를 세어보세요. 한 줄이 끝났는데
  대응하지 못한 여는 기호나 닫는 기호가 남아 있으면, 그 줄을 code_syntax로 보고하세요.
  이 대응 확인은 프로그래밍 코드뿐 아니라 제곱근·분수·지수가 섞인 수식에도 똑같이 적용하세요.
  수식에서는 괄호 기호가 작거나 다른 기호(근호, 분수선 등)와 붙어 있어 놓치기 쉬우니, 여는 기호가
  나올 때마다 그 수식이 끝날 때까지 대응하는 닫는 기호가 실제로 나오는지 끝까지 따라가며 확인하세요.
- visual_defect: 이미지가 깨지거나 로드되지 않아 내용을 알아볼 수 없는 경우, 텍스트·도형이
  서로 겹쳐 내용을 읽을 수 없는 경우, 콘텐츠가 슬라이드 경계에 잘려 일부가 안 보이는 경우.
  미적 취향의 문제가 아니라 콘텐츠를 읽을 수 없게 만드는 기능적 결함일 때만 보고하세요.

보고하지 말 것:
- OCR이 잘못 읽은 텍스트 자체
- 용어 선택/문체/표현 선호
- 사실 오류나 개념 오류 (단, 강의 맥락과 무관하게 이미 하나의 값으로 고정된 수치가 명백히 틀리게
  적힌 경우는 예외입니다 — numeric_unit으로 보고하세요)
- 띄어쓰기, 줄바꿈, 글자 간격
- 배치·색상 선택 등 미적 취향의 디자인 문제 (읽는 데 지장이 없다면)
- 약어, 고유명사, 표기 관례처럼 오류로 단정하기 어려운 것
- 복합어 띄어쓰기 관례
- 조사/어미/접속 표현 교정
- 외래어를 한국어로 순화하는 교정
- 쉼표 추가, 문장 자연화, 더 좋은 표현 제안

출력 형식은 JSON만 허용합니다.

```json
{{
  "slide_errors": [
    {{
      "error_type": "text_error | numeric_unit | code_syntax | visual_defect",
      "problematic_text": "문제가 있는 원문 또는 위치·대상 설명",
      "corrected_text": "수정 표현 (visual_defect면 빈 문자열)",
      "confidence": 0.0,
      "reason": "한두 문장 근거"
    }}
  ]
}}
```

지침:
1. 확신이 0.80 미만이면 출력하지 마세요.
2. "더 자연스럽다", "더 적절하다" 수준이면 출력하지 마세요.
3. error_type은 반드시 text_error, numeric_unit, code_syntax, visual_defect 중 하나만 사용하세요.
4. visual_defect는 corrected_text를 빈 문자열로 두고, problematic_text에 문제 위치·대상을 설명하세요.
5. 오류가 없으면 {{"slide_errors": []}}만 출력하세요.
6. JSON 외 텍스트 금지.
7. 슬라이드 하나에 오류가 여러 개 있을 수 있습니다. 하나를 찾았다고 바로 멈추지 말고, 슬라이드 전체를
   끝까지 마저 확인해서 발견되는 오류를 모두 slide_errors 배열에 각각 출력하세요.
"""


def _parse_response(text: str) -> list[dict[str, Any]]:
    payload = json.loads(_strip_json_fence(text))
    rows = payload.get("slide_errors", [])
    return rows if isinstance(rows, list) else []


def _is_reportable_slide_error(
    problematic: str,
    corrected: str,
    reason: str = "",
    error_type: str = "text_error",
) -> bool:
    p = str(problematic or "").strip()
    r = str(reason or "").strip().lower()

    if error_type == "visual_defect":
        # 고칠 "텍스트"가 없는 유형이라, 문제 위치/대상 설명과 근거만 있으면 됩니다.
        return bool(p) and bool(r)

    c = str(corrected or "").strip()
    if not p or not c or p == c:
        return False
    p_compact = _compact_no_space(p)
    c_compact = _compact_no_space(c)
    if not p_compact or p_compact == c_compact:
        return False
    if re.sub(r"\S+", "", p) != re.sub(r"\S+", "", c):
        return False

    style_markers = (
        "더 적절",
        "더 자연",
        "자연스럽",
        "어색",
        "문법",
        "조사",
        "순화",
        "용어",
        "표현 선호",
        "더 좋은 표현",
        "문체",
        "선택의 의미",
        "나열",
        "병렬",
        "쉼표",
        "구분",
        "별개",
        "띄어쓰기",
        "공백",
        "줄바꿈",
        "글자 간격",
    )
    if any(marker in r for marker in style_markers):
        return False
    if "중복" in r and re.search(r"(을|를|이|가|은|는)\s+\S+(을|를|이|가|은|는)", p):
        return False
    return True


def _normalize_error_type(raw_value: object, problematic: str, corrected: str) -> str:
    """LLM이 준 error_type을 검증하고, 없거나 잘못됐을 때만 텍스트로 추론합니다."""
    value = str(raw_value or "").strip().lower()
    if value in SLIDE_ERROR_TYPES:
        return value
    if not corrected:
        return "visual_defect"
    if re.search(r"\d|[%℃°]|(?:ms|sec|kb|mb|gb|hz|khz|mhz|ghz)\b", f"{problematic} {corrected}", re.IGNORECASE):
        return "numeric_unit"
    return "text_error"


def _normalize_error(
    row: dict[str, Any],
    *,
    slide: dict[str, Any],
    slide_number: int,
    model: str,
) -> dict[str, Any] | None:
    problematic = str(row.get("problematic_text", "") or "").strip()
    corrected = str(row.get("corrected_text", "") or "").strip()
    reason = str(row.get("reason", "") or "").strip()
    confidence = _clamp01(row.get("confidence"))
    if confidence < 0.80:
        return None
    error_type = _normalize_error_type(row.get("error_type"), problematic, corrected)
    if not _is_reportable_slide_error(problematic, corrected, reason, error_type):
        return None
    severity_score = confidence
    digest = hashlib.sha1(
        json.dumps(
            {
                "slide_number": slide_number,
                "error_type": error_type,
                "problematic_text": problematic,
                "corrected_text": corrected,
                "model": model,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:10]
    return {
        "slide_error_id": f"S{slide_number:03d}:{digest}",
        "slide_number": slide_number,
        "slide_title": slide.get("title", ""),
        "slide_image_path": slide.get("image_path", ""),
        "problematic_text": problematic,
        "corrected_text": corrected,
        "reason": reason,
        "confidence": confidence,
        "severity_score": severity_score,
        "error_type": error_type,
        "error_type_label": SLIDE_ERROR_TYPES.get(error_type, error_type),
        "is_reportable": True,
        "source": "classified_slide_error_checker",
        "model": model,
    }


def _check_single_slide(
    *,
    model: str,
    slide: dict[str, Any],
    img_dir: str | None,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], bool, int, dict[str, Any]]:
    from . import claim_common as cc

    slide_number = _slide_number(slide) or 0
    title = str(slide.get("title", "") or "")
    slide_text = str(slide.get("slide_text") or slide.get("t1") or "")
    prompt = _build_slide_error_prompt(slide_number, title, slide_text)
    img_path = _existing_path(slide.get("image_path")) or _find_slide_image(img_dir, slide_number)
    img_bytes = img_path.read_bytes() if img_path and img_path.exists() else None
    response_format = {"type": "json_object"} if cc._supports_json_object_response_format(model) else None
    token_usage = cc._empty_token_usage()
    api_calls = 0

    for attempt in range(cc.VERIFIER_PARSE_RETRIES + 1):
        text, call_usage = cc._call_llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            image_bytes=img_bytes,
            thinking_budget=0,
            response_format=response_format,
            stage="slide_error",
        )
        api_calls += 1
        cc._add_call_usage(token_usage, call_usage)
        try:
            rows = _parse_response(text)
            errors = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                error = _normalize_error(row, slide=slide, slide_number=slide_number, model=model)
                if error:
                    errors.append(error)
            return errors, False, api_calls, token_usage
        except Exception:
            if attempt < cc.VERIFIER_PARSE_RETRIES:
                print(f"    ↺ 슬라이드 {slide_number} 오류 JSON 파싱 재시도 ({attempt+1}/{cc.VERIFIER_PARSE_RETRIES})")
    return [], True, api_calls, token_usage


def detect_classified_slide_errors(
    *,
    merged_clean_path: str | Path,
    slide_textualized_path: str | Path | None,
    slide_classified_path: str | Path | None,
    models: list[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = 1,
    max_tokens: int = 4096,
    current_date: str | None = None,
    min_score: float | None = None,
) -> dict[str, Any]:
    from . import claim_common as cc

    _load_env()
    current_date = current_date or datetime.now().date().isoformat()
    model = str(cc._resolve_stage_model("slide_error") or "").strip()
    models = [model] if model else (models or _default_models())
    min_score = DEFAULT_MIN_SCORE if min_score is None else max(0.0, min(1.0, min_score))

    merged_payload = _load_json(merged_clean_path)
    textualized_payload = _load_json(slide_textualized_path)
    classified_payload = _load_json(slide_classified_path)
    slides = _slides_for_check(
        merged_payload=merged_payload,
        textualized_payload=textualized_payload,
        classified_payload=classified_payload,
    )
    img_dir = _resolve_img_dir(merged_payload, merged_clean_path)

    model_results: dict[str, dict[str, Any]] = {
        item: {
            "model": item,
            "status": "ok",
            "slide_errors": [],
            "batch_errors": [],
            "api_calls": 0,
            "token_usage": cc._empty_token_usage(),
        }
        for model in models
        for item in [model]
    }
    work_items: list[tuple[str, dict[str, Any]]] = []
    for model in models:
        for slide in slides:
            work_items.append((model, slide))

    def worker(args: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        model, slide = args
        slide_number = _slide_number(slide) or "?"
        print(f"    슬라이드 오타 검사 [{slide_number}]", flush=True)
        errors, parse_failed, api_calls, usage = _check_single_slide(
            model=model,
            slide=slide,
            img_dir=img_dir,
            max_tokens=max_tokens,
        )
        return {
            "model": model,
            "slide_number": slide_number,
            "slide_errors": errors,
            "parse_failed": parse_failed,
            "api_calls": api_calls,
            "token_usage": usage,
        }

    if max_workers <= 1:
        for args in work_items:
            try:
                result = worker(args)
            except Exception as exc:
                model_results[args[0]]["status"] = "partial_failed"
                model_results[args[0]]["batch_errors"].append(
                    {"slide_number": _slide_number(args[1]), "error": str(exc)}
                )
                continue
            model_results[result["model"]]["slide_errors"].extend(result["slide_errors"])
            model_results[result["model"]]["api_calls"] += int(result.get("api_calls", 0) or 0)
            model_results[result["model"]]["token_usage"] = cc._merge_token_usage(
                model_results[result["model"]].get("token_usage"),
                result.get("token_usage"),
            )
            if result.get("parse_failed"):
                model_results[result["model"]]["status"] = "partial_failed"
                model_results[result["model"]]["batch_errors"].append(
                    {"slide_number": result.get("slide_number"), "error": "json_parse_failed"}
                )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker, args): args for args in work_items}
            for future in as_completed(futures):
                args = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    model_results[args[0]]["status"] = "partial_failed"
                    model_results[args[0]]["batch_errors"].append(
                        {"slide_number": _slide_number(args[1]), "error": str(exc)}
                    )
                    continue
                model_results[result["model"]]["slide_errors"].extend(result["slide_errors"])
                model_results[result["model"]]["api_calls"] += int(result.get("api_calls", 0) or 0)
                model_results[result["model"]]["token_usage"] = cc._merge_token_usage(
                    model_results[result["model"]].get("token_usage"),
                    result.get("token_usage"),
                )
                if result.get("parse_failed"):
                    model_results[result["model"]]["status"] = "partial_failed"
                    model_results[result["model"]]["batch_errors"].append(
                        {"slide_number": result.get("slide_number"), "error": "json_parse_failed"}
                    )

    all_errors: list[dict[str, Any]] = []
    token_usage = Counter()
    for result in model_results.values():
        usage_total = (result.get("token_usage") or {}).get("total", {})
        for key in TOKEN_USAGE_FIELDS:
            token_usage[key] += int(usage_total.get(key, 0) or 0)
        all_errors.extend(result.get("slide_errors", []) or [])

    dedup: dict[tuple[Any, str, str], dict[str, Any]] = {}
    for error in all_errors:
        key = (
            error.get("slide_number"),
            str(error.get("problematic_text", "")).strip().lower(),
            str(error.get("corrected_text", "")).strip().lower(),
        )
        prev = dedup.get(key)
        if prev is None or float(error.get("confidence", 0) or 0) > float(prev.get("confidence", 0) or 0):
            dedup[key] = error
    reportable = [
        error
        for error in dedup.values()
        if error.get("is_reportable") and float(error.get("severity_score", 0) or 0) >= min_score
    ]
    reportable.sort(key=lambda item: (
        int(item.get("slide_number", 0) or 0),
        -float(item.get("severity_score", 0) or 0),
        str(item.get("problematic_text", "")),
    ))

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "classified_slide_error_checker",
        "generated_at": _now_iso(),
        "current_date": current_date,
        "merged_clean_path": str(merged_clean_path),
        "slide_textualized_path": str(slide_textualized_path or ""),
        "slide_classified_path": str(slide_classified_path or ""),
        "models": models,
        "min_score": min_score,
        "summary": {
            "total_slide_count": len(slides),
            "raw_error_count": len(all_errors),
            "deduped_error_count": len(dedup),
            "reportable_error_count": len(reportable),
            "breakdown_by_type": dict(Counter(error.get("error_type", "other") for error in reportable)),
        },
        "slide_errors": reportable,
        "raw_slide_errors": all_errors,
        "model_results": model_results,
        "token_usage": dict(token_usage),
    }
