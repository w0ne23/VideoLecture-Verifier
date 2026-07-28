"""Slide error checker for the classified issue pipeline.

The classified pipeline keeps its own output shape. The slide image is the
source of truth, and whitespace/layout/style suggestions are not reportable
slide errors.

Earlier stages' OCR-derived text (t1/slide_text) is never passed straight
into the error-judging prompt: it can already contain misreads from that
stage's own OCR hint, and re-presenting it here would anchor the judgment
call on the same mistake a second time. Instead, each slide is re-transcribed
here from the image alone (`_transcribe_slide_text`) before judging, mirroring
the transcribe-then-judge split already used for code_syntax.

Even with that split, the judging model can still misread the same subtle
glyph on its own (independent of the transcription), especially on small text
in a downscaled full-slide image. text_error/numeric_unit candidates are
therefore re-verified against a cropped, zoomed-in region of the original
image (`_verify_error_with_crop`) before being finalized, since a focused
crop avoids the resolution loss a vision API applies when downsampling the
whole slide.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image

from .issue_type_classifier import (
    TOKEN_USAGE_FIELDS,
    _load_env,
    _split_csv,
)


SCHEMA_VERSION = "classified_slide_error_checker.v5"
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
        base / f"scene_{slide_no:03d}_base.jpg",
        base / f"scene_{slide_no:03d}_base.png",
        base / f"slide_{slide_no:03d}_base.jpg",
        base / f"slide_{slide_no:03d}_base.png",
        base / f"slide_{slide_no:03d}_start.jpg",
        base / f"slide_{slide_no:03d}_start.png",
        base / f"slide_{slide_no:03d}_end.jpg",
        base / f"slide_{slide_no:03d}_end.png",
    )
    return next((path for path in candidates if path.exists()), None)


_BASE_VARIANT_RE = re.compile(r"^(scene_\d+|slide_\d+)_.*(\.[A-Za-z0-9]+)$")


def _force_base_image_path(
    image_path: object, img_dir: str | None, slide_no: int
) -> Path | None:
    """annot(필기)·build(애니메이션 진행) 변형 이미지가 아니라 항상 base 이미지로
    판정하도록 강제합니다. annot/build 프레임은 필기나 미완성 애니메이션 상태 때문에
    텍스트/도형이 겹쳐 보여 visual_defect 오탐을 유발하므로, base 한 장만 채점 대상으로
    삼습니다. annot/build도 결국 base에서 파생된 프레임이라 base만 봐도 충분합니다."""
    candidates: list[Path] = []
    raw = str(image_path or "").strip()
    if raw:
        match = _BASE_VARIANT_RE.match(Path(raw).name)
        if match:
            candidates.append(Path(raw).parent / f"{match.group(1)}_base{match.group(2)}")
    if img_dir:
        base = Path(img_dir)
        candidates.extend(
            [
                base / f"scene_{slide_no:03d}_base.jpg",
                base / f"scene_{slide_no:03d}_base.png",
                base / f"slide_{slide_no:03d}_base.jpg",
                base / f"slide_{slide_no:03d}_base.png",
            ]
        )
    for candidate in candidates:
        resolved = _existing_path(str(candidate))
        if resolved and resolved.exists():
            return resolved
    return None


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
- 슬라이드 이미지가 유일한 원본이자 진실입니다.
- 아래 "1차 전사 텍스트"는 별도 모델이 이 이미지를 보고 미리 옮겨 적어둔 참고용 초안일 뿐이며,
  자동 인식 과정에서 생긴 오독을 포함할 수 있습니다. 특히 한글은 자모 하나 차이(예: 늬/니, 왜/외,
  웨/훼)를 자동 인식이 잘못 읽는 경우가 흔합니다.
- text_error, numeric_unit 등 철자·표기 오류를 판단할 때 이 1차 전사 텍스트를 정답 기준으로 삼지
  마세요. 반드시 슬라이드 이미지 속 글자를 네가 직접 다시 읽어서 판단하세요.
- 1차 전사 텍스트와 이미지가 다르게 보인다면, 그것은 100% 1차 전사(자동 인식)의 실수이지 슬라이드의
  오류가 아닙니다. 이런 차이 자체는 절대 오류로 보고하지 마세요.
- text_error를 보고하려면, problematic_text는 1차 전사 텍스트를 그대로 옮기지 말고 네가 이미지에서
  직접 확인한 글자를 근거로 작성하세요. 이미지 속 글자가 실제로 잘못 쓰여 있다고 스스로 확신할 때만
  보고하세요.
- 내용의 사실성, 더 좋은 표현, 문체 개선, 미적 취향(배치·색상 선택 등 디자인 문제)은 보고하지 마세요.
- 애매하면 보고하지 마세요.
- 아래 "번호"는 영상에서 화면이 바뀔 때마다 시스템이 자동으로 매긴 내부 색인일 뿐이며, 슬라이드
  자료 자체에 인쇄된 페이지·목차 번호와 무관합니다. 이 번호가 슬라이드 이미지 안에 보이는 번호와
  다르다는 이유로는 절대 오류를 보고하지 마세요.

대상 슬라이드:
- 번호: {slide_no}
- 1차 전사 제목(참고용, 오류 가능): {title}
- 1차 전사 본문(참고용, 오류 가능):
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
  단, 목록 항목의 번호·기호로 쓰인 단독 괄호(예: 항목 앞에 숫자나 문자 뒤에 닫는 괄호 하나만
  붙는 표기)는 코드나 수식이 아니므로 이 괄호 짝 검사 대상에서 제외하세요.
- visual_defect: 아래 중 하나에 해당하고, 그로 인해 실제 "글자(텍스트)"를 읽을 수 없게 된
  경우만 보고하세요 — 이미지가 깨지거나 로드되지 않아 전체 내용을 알아볼 수 없는 경우, 텍스트가
  다른 도형·이미지·비디오에 가려지거나 슬라이드 경계에 잘려서 그 글자 자체가 안 보이는 경우.
  * 도형이나 디자인 요소끼리 겹쳐 배치된 것 자체는 오류가 아닙니다 — 강조 테두리, 코너
    브래킷, 강사 화면(PIP) 등은 원래 그렇게 겹치거나 인접하도록 디자인된 경우가 많습니다.
    겹침·인접 배치가 보인다는 이유만으로 보고하지 말고, 그로 인해 실제 텍스트 내용이 사라지거나
    읽을 수 없게 됐는지를 반드시 확인하세요.
  * "화면 경계에 잘려 안 보인다"고 보고하기 전에, 그 상자/텍스트가 실제로 잘려서 일부가
    사라졌는지 이미지를 다시 확인하세요. 글자와 테두리가 전부 온전히 보인다면(다른 요소와
    가깝게 배치되어 있을 뿐이라면) 잘린 것이 아니므로 보고하지 마세요.
  미적 취향의 문제가 아니라, 텍스트를 읽을 수 없게 만드는 기능적 결함일 때만 보고하세요.

보고하지 말 것:
- 1차 전사(자동 인식)가 잘못 읽은 텍스트 자체 — 이미지 속 글자가 실제로 맞게 쓰여 있다면 1차
  전사와 다르더라도 오류가 아닙니다
- 용어 선택/문체/표현 선호
- 사실 오류나 개념 오류 (단, 강의 맥락과 무관하게 이미 하나의 값으로 고정된 수치가 명백히 틀리게
  적힌 경우는 예외입니다 — numeric_unit으로 보고하세요)
- 띄어쓰기, 줄바꿈, 글자 간격
- 배치·색상 선택 등 미적 취향의 디자인 문제 (읽는 데 지장이 없다면)
- 도형·이미지·강사 화면(PIP) 등이 서로 겹치거나 가깝게 배치된 것 자체 (그로 인해 실제
  텍스트가 가려지거나 잘리지 않았다면 오류가 아닙니다)
- 약어, 고유명사, 표기 관례처럼 오류로 단정하기 어려운 것
- 복합어 띄어쓰기 관례
- 조사/어미/접속 표현 교정
- 외래어를 한국어로 순화하는 교정
- 쉼표 추가, 문장 자연화, 더 좋은 표현 제안
- 시스템이 매긴 슬라이드 번호와 슬라이드 이미지에 인쇄된 번호가 다른 경우
- 번호 매기기·항목 기호로 쓰인 단독 괄호(예: 목록 항목 앞에 숫자나 문자 뒤에 닫는 괄호 하나만
  오는 표기)를 코드나 수식의 괄호 짝 오류로 보는 것 — 이런 표기는 코드·수식이 아닙니다

출력 형식은 JSON만 허용합니다.

```json
{{
  "slide_errors": [
    {{
      "error_type": "text_error | numeric_unit | code_syntax | visual_defect",
      "problematic_text": "문제가 있는 원문 또는 위치·대상 설명",
      "corrected_text": "수정 표현 (visual_defect면 빈 문자열)",
      "confidence": 0.0,
      "reason": "한두 문장 근거",
      "confirmed_after_recheck": true,
      "bbox": {{"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}}
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
8. confirmed_after_recheck: 배열에 넣기 직전에 이미지를 한 번 더 확인해서 "정말로 오류가 맞는지"
   최종 결정한 결과를 담으세요. 재확인 결과 오류가 아니라고 판단되면 그 항목은 배열에서 완전히
   빼세요 — confirmed_after_recheck를 false로 적어서 배열에 남겨두지 마세요. 즉 배열에 실제로
   들어가는 항목은 전부 confirmed_after_recheck가 true여야 합니다.
9. bbox: problematic_text(또는 visual_defect 위치)가 이미지 안에서 있는 대략적인 사각형 영역을
   0~1 정규화 좌표로 표시하세요. (x0, y0)는 좌상단, (x1, y1)은 우하단 모서리이며, 이미지 전체
   너비/높이를 1.0으로 봅니다. 정확하지 않아도 되니 최대한 근접하게 표시하세요.
"""


def _parse_response(text: str) -> list[dict[str, Any]]:
    payload = json.loads(_strip_json_fence(text))
    rows = payload.get("slide_errors", [])
    return rows if isinstance(rows, list) else []


_BRACKET_OPENERS = {"(": ")", "[": "]", "{": "}"}
_BRACKET_CLOSERS = {")": "(", "]": "[", "}": "{"}


def _find_bracket_mismatch(text: str) -> str | None:
    """스택 기반으로 괄호 짝을 확인합니다. 사람이 눈으로 세는 대신 결정론적으로 계산하므로
    LLM에게 판단을 맡기는 것보다 정확합니다. 문제가 있으면 설명을, 없으면 None을 반환합니다."""
    stack: list[str] = []
    for ch in text:
        if ch in _BRACKET_OPENERS:
            stack.append(ch)
        elif ch in _BRACKET_CLOSERS:
            if not stack:
                return f"'{ch}' 앞에 대응하는 여는 기호가 없습니다"
            top = stack.pop()
            if _BRACKET_OPENERS[top] != ch:
                return f"'{top}'로 열었는데 '{ch}'로 닫혔습니다"
    if stack:
        return f"닫히지 않은 여는 기호가 남아 있습니다: {''.join(stack)}"
    return None


def _build_text_transcription_prompt(slide_no: int, title_hint: str) -> str:
    return f"""당신은 강의 슬라이드 이미지 속 텍스트를 있는 그대로 옮겨 적는 전사자입니다.

대상 슬라이드 번호: {slide_no}
참고용 제목 힌트(부정확할 수 있으니 참고만 하세요): {title_hint}

작업:
- 이 슬라이드 이미지에 보이는 제목과 본문 텍스트를 처음부터 끝까지 빠짐없이 옮겨 적으세요.
- 옳고 그름을 판단하거나 고치지 마세요. 이미지에 실제로 보이는 글자 그대로 옮기는 것이 유일한 목표입니다.
- 한글 자모 하나 차이로 헷갈리기 쉬운 글자(예: 늬/니, 왜/외, 웨/훼)는 획을 하나씩 확인하듯 주의
  깊게 보고 옮겨 적으세요.
- 표/도형/그림 안의 텍스트도 함께 옮겨 적으세요.
- 텍스트가 전혀 없으면 빈 문자열을 출력하세요.

출력 형식은 JSON만 허용합니다.

```json
{{
  "title": "이미지에 보이는 그대로의 제목",
  "body": "이미지에 보이는 그대로의 본문 텍스트"
}}
```

JSON 외 텍스트는 출력하지 마세요.
"""


def _build_code_transcription_prompt(slide_no: int, title: str) -> str:
    return f"""당신은 강의 슬라이드 이미지 속 코드나 수식을 있는 그대로 옮겨 적는 전사자입니다.

대상 슬라이드 번호: {slide_no}
제목: {title}

작업:
- 이 슬라이드 이미지 안에 프로그래밍 코드 또는 괄호가 포함된 수식이 있는지 확인하세요.
- 목록 항목의 번호·기호로 쓰인 단독 괄호(예: 항목 앞에 숫자나 문자 뒤에 닫는 괄호 하나만 붙는
  표기)는 코드나 수식이 아니므로 전사 대상이 아닙니다.
- 있다면, 보이는 그대로 한 글자도 빠짐없이 옮겨 적으세요. 특히 소괄호()·대괄호[]·중괄호{{}}는
  절대 생략하거나 요약하지 말고, 실제 이미지에 있는 개수와 위치 그대로 옮기세요.
- 옳고 그름을 판단하거나 고치지 마세요. 이미지에 실제로 보이는 그대로 옮겨 적는 것이 유일한 목표입니다.
- 코드/수식 블록이 여러 개면 각각 따로 옮겨 적으세요.
- 코드나 수식이 전혀 없으면 빈 배열을 출력하세요.

출력 형식은 JSON만 허용합니다.

```json
{{
  "blocks": [
    {{"transcription": "이미지에 보이는 그대로의 코드 또는 수식 텍스트"}}
  ]
}}
```

JSON 외 텍스트는 출력하지 마세요.
"""


def _check_code_syntax_mechanical(
    *,
    model: str,
    slide: dict[str, Any],
    img_dir: str | None,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """VLM에게는 '보이는 그대로 옮겨 적기'만 시키고, 괄호 짝 판단은 결정론적 알고리즘이
    맡습니다. 시각 인식과 문법 판단을 한 번에 묶어서 시켰던 기존 방식보다 정확도가 높을
    것으로 기대되는 별도 경로입니다.

    OCR 텍스트로 코드/수식 슬라이드를 미리 걸러내지 않고 모든 슬라이드에 대해 무조건
    전사를 시도합니다 — OCR이 코드 블록을 놓치면 걸러내기 자체가 실패해 탐지가 통째로
    스킵되는 위험이 있었기 때문입니다. 대신 판단이 필요 없는 단순 전사 작업이라는 점을
    이용해 저렴한 전용 모델(VERIFIER_SLIDE_ERROR_TRANSCRIBE_MODEL, 기본 gpt-5.4-mini)을
    써서 전체 슬라이드에 걸어도 비용 부담이 크지 않게 합니다."""
    from . import claim_common as cc

    slide_number = _slide_number(slide) or 0
    img_path = (
        _force_base_image_path(slide.get("image_path"), img_dir, slide_number)
        or _existing_path(slide.get("image_path"))
        or _find_slide_image(img_dir, slide_number)
    )
    img_bytes = img_path.read_bytes() if img_path and img_path.exists() else None
    if not img_bytes:
        return [], 0, cc._empty_token_usage()

    transcribe_model = cc._resolve_stage_model("slide_error_transcribe") or model
    title = str(slide.get("title", "") or "")
    prompt = _build_code_transcription_prompt(slide_number, title)
    response_format = (
        {"type": "json_object"} if cc._supports_json_object_response_format(transcribe_model) else None
    )
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
            stage="slide_error_transcribe",
        )
        api_calls += 1
        cc._add_call_usage(token_usage, call_usage)
        try:
            payload = json.loads(_strip_json_fence(text))
            blocks = payload.get("blocks", [])
            if not isinstance(blocks, list):
                blocks = []
            errors: list[dict[str, Any]] = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                transcription = str(block.get("transcription", "") or "").strip()
                if not transcription:
                    continue
                issue = _find_bracket_mismatch(transcription)
                if not issue:
                    continue
                digest = hashlib.sha1(
                    json.dumps(
                        {
                            "slide_number": slide_number,
                            "error_type": "code_syntax",
                            "transcription": transcription,
                            "model": transcribe_model,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()[:10]
                errors.append(
                    {
                        "slide_error_id": f"S{slide_number:03d}:{digest}",
                        "slide_number": slide_number,
                        "slide_title": slide.get("title", ""),
                        "slide_image_path": slide.get("image_path", ""),
                        "problematic_text": transcription,
                        "corrected_text": "",
                        "reason": f"괄호/기호 짝이 맞지 않습니다 ({issue}).",
                        "confidence": 0.95,
                        "severity_score": 0.95,
                        "error_type": "code_syntax",
                        "error_type_label": SLIDE_ERROR_TYPES["code_syntax"],
                        "is_reportable": True,
                        "source": "classified_slide_error_checker.mechanical",
                        "model": transcribe_model,
                    }
                )
            return errors, api_calls, token_usage
        except Exception:
            if attempt < cc.VERIFIER_PARSE_RETRIES:
                continue
    return [], api_calls, token_usage


def _transcribe_slide_text(
    *,
    model: str,
    slide: dict[str, Any],
    img_dir: str | None,
    max_tokens: int,
) -> tuple[str, str, int, dict[str, Any]]:
    """이미지만 보고 제목/본문을 새로 전사합니다. 이전 단계(t1 추출)에서 OCR 힌트에
    이끌려 잘못 옮겨졌을 수 있는 텍스트를 오탈자 판정 프롬프트에 그대로 넘기면, 그 오독이
    두 번째 판정 단계에도 앵커링되어 실제로는 이미지에 맞게 적힌 글자를 오류로 보고하는
    문제가 생긴다. 그래서 판정용 텍스트를 기존 slide_text(t1)에서 가져오지 않고, OCR 힌트
    없이 이미지만 보는 별도 호출로 다시 뽑는다."""
    from . import claim_common as cc

    slide_number = _slide_number(slide) or 0
    img_path = (
        _force_base_image_path(slide.get("image_path"), img_dir, slide_number)
        or _existing_path(slide.get("image_path"))
        or _find_slide_image(img_dir, slide_number)
    )
    img_bytes = img_path.read_bytes() if img_path and img_path.exists() else None
    if not img_bytes:
        return "", "", 0, cc._empty_token_usage()

    transcribe_model = cc._resolve_stage_model("slide_error_transcribe") or model
    title_hint = str(slide.get("title", "") or "")
    prompt = _build_text_transcription_prompt(slide_number, title_hint)
    response_format = (
        {"type": "json_object"} if cc._supports_json_object_response_format(transcribe_model) else None
    )
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
            stage="slide_error_transcribe",
        )
        api_calls += 1
        cc._add_call_usage(token_usage, call_usage)
        try:
            payload = json.loads(_strip_json_fence(text))
            title = str(payload.get("title", "") or "").strip()
            body = str(payload.get("body", "") or "").strip()
            return title, body, api_calls, token_usage
        except Exception:
            if attempt < cc.VERIFIER_PARSE_RETRIES:
                continue
    return "", "", api_calls, token_usage


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
        "쉼표",
        "띄어쓰기",
        "공백",
        "줄바꿈",
        "글자 간격",
    )
    if any(marker in r for marker in style_markers):
        return False
    if "중복" in r and re.search(r"(을|를|이|가|은|는)\s+\S+(은|는|이|가|을|를)", p):
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
    confirmed_raw = row.get("confirmed_after_recheck", True)
    confirmed = confirmed_raw if isinstance(confirmed_raw, bool) else str(confirmed_raw).strip().lower() not in {
        "false",
        "no",
        "0",
    }
    if not confirmed:
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


def _extract_bbox(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = row.get("bbox")
    if not isinstance(bbox, dict):
        return None
    try:
        x0, y0, x1, y1 = (
            float(bbox.get("x0")),
            float(bbox.get("y0")),
            float(bbox.get("x1")),
            float(bbox.get("y1")),
        )
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _crop_and_zoom_image_bytes(
    image_path: Path,
    bbox: tuple[float, float, float, float],
    *,
    pad_ratio: float = 0.15,
    min_width: int = 1000,
) -> bytes | None:
    """bbox 주변을 여유 있게 크롭해서 확대합니다. 전체 슬라이드를 한 번에 보낼 때는 비전
    API가 이미지를 내부적으로 다운스케일하면서 작은 글자의 디테일을 뭉갤 수 있는데, 문제
    구간만 잘라 확대해 다시 보내면 그 구간이 다운스케일의 영향을 덜 받아 더 선명하게
    보입니다."""
    try:
        image = Image.open(image_path)
        image.load()
    except Exception:
        return None
    width, height = image.size
    x0, y0, x1, y1 = bbox
    # bbox 자체가 한 단어처럼 아주 좁으면 bbox 크기에 비례한 여백만으로는 주변 문맥이
    # 부족해 잘려 나가기 쉽다(글자가 안 보여 "판독 불가"로 되돌아옴). 전체 이미지 폭의
    # 일정 비율을 최소 여백으로 강제해 항상 읽을 수 있는 크기의 크롭을 확보한다.
    pad_x = max((x1 - x0) * pad_ratio, 0.04)
    pad_y = max((y1 - y0) * pad_ratio, 0.04)
    left = max(0, int((x0 - pad_x) * width))
    top = max(0, int((y0 - pad_y) * height))
    right = min(width, int((x1 + pad_x) * width))
    bottom = min(height, int((y1 + pad_y) * height))
    if right - left < 4 or bottom - top < 4:
        return None
    crop = image.convert("RGB").crop((left, top, right, bottom))
    if crop.width < min_width:
        scale = min_width / crop.width
        crop = crop.resize((min_width, max(1, int(crop.height * scale))), Image.LANCZOS)
    buf = BytesIO()
    crop.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _build_crop_verify_prompt(problematic_text: str, corrected_text: str, error_type: str) -> str:
    return f"""당신은 강의 슬라이드에서 발견된 오류 후보를 확대된 이미지로 최종 확인하는
검수자입니다.

이전 단계에서 아래 오류 후보가 보고되었습니다:
- error_type: {error_type}
- 문제로 지목된 표기: "{problematic_text}"
- 제안된 수정: "{corrected_text}"

지금 보여주는 이미지는 그 부분을 확대(zoom)한 것입니다. 한 글자씩 주의 깊게 읽고 그대로
옮겨 적은 뒤(actual_text), 그 결과를 근거로 오류 후보가 실제로 유효한지 판단하세요.

verdict 값 정의 (이 단어 뜻 그대로 사용하세요):
- "error_confirmed": actual_text가 "문제로 지목된 표기"와 같고, 그것이 실제로 잘못된 표기가
  맞습니다 → 오류 신고를 그대로 유지합니다.
- "not_an_error": 확대해서 다시 보니 실제로는 "{corrected_text}"처럼 이미 올바르게 쓰여 있거나,
  애초에 오류가 아니었습니다 → 오류 신고를 취소합니다.
- "inconclusive": 이미지가 잘리거나 흐려서 판단할 근거가 부족합니다 → 오류 신고를 그대로
  유지합니다(판단 보류).

출력 형식은 JSON만 허용합니다.

```json
{{
  "actual_text": "확대 이미지에서 실제로 보이는 글자 그대로",
  "verdict": "error_confirmed | not_an_error | inconclusive"
}}
```

JSON 외 텍스트는 출력하지 마세요.
"""


def _verify_error_with_crop(
    *,
    model: str,
    image_path: Path,
    bbox: tuple[float, float, float, float],
    error: dict[str, Any],
    max_tokens: int,
) -> tuple[bool, int, dict[str, Any]]:
    from . import claim_common as cc

    crop_bytes = _crop_and_zoom_image_bytes(image_path, bbox)
    if not crop_bytes:
        return True, 0, cc._empty_token_usage()

    verify_model = cc._resolve_stage_model("slide_error_transcribe") or model
    prompt = _build_crop_verify_prompt(
        error.get("problematic_text", ""), error.get("corrected_text", ""), error.get("error_type", "")
    )
    response_format = (
        {"type": "json_object"} if cc._supports_json_object_response_format(verify_model) else None
    )
    token_usage = cc._empty_token_usage()
    api_calls = 0

    for attempt in range(cc.VERIFIER_PARSE_RETRIES + 1):
        text, call_usage = cc._call_llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            image_bytes=crop_bytes,
            thinking_budget=0,
            response_format=response_format,
            stage="slide_error_transcribe",
        )
        api_calls += 1
        cc._add_call_usage(token_usage, call_usage)
        try:
            payload = json.loads(_strip_json_fence(text))
            verdict = str(payload.get("verdict", "") or "").strip().lower()
            confirmed = verdict != "not_an_error"
            return confirmed, api_calls, token_usage
        except Exception:
            if attempt < cc.VERIFIER_PARSE_RETRIES:
                continue
    return True, api_calls, token_usage


def _check_single_slide(
    *,
    model: str,
    slide: dict[str, Any],
    img_dir: str | None,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], bool, int, dict[str, Any]]:
    from . import claim_common as cc

    slide_number = _slide_number(slide) or 0
    fallback_title = str(slide.get("title", "") or "")
    fallback_text = str(slide.get("slide_text") or slide.get("t1") or "")
    transcribed_title, transcribed_text, transcribe_calls, transcribe_usage = _transcribe_slide_text(
        model=model,
        slide=slide,
        img_dir=img_dir,
        max_tokens=max_tokens,
    )
    title = transcribed_title or fallback_title
    slide_text = transcribed_text or fallback_text
    prompt = _build_slide_error_prompt(slide_number, title, slide_text)
    img_path = (
        _force_base_image_path(slide.get("image_path"), img_dir, slide_number)
        or _existing_path(slide.get("image_path"))
        or _find_slide_image(img_dir, slide_number)
    )
    img_bytes = img_path.read_bytes() if img_path and img_path.exists() else None
    response_format = {"type": "json_object"} if cc._supports_json_object_response_format(model) else None
    token_usage = transcribe_usage
    api_calls = transcribe_calls

    errors: list[dict[str, Any]] | None = None
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
                if not error:
                    continue
                if error.get("error_type") in {"text_error", "numeric_unit"} and img_path and img_path.exists():
                    bbox = _extract_bbox(row)
                    if bbox:
                        confirmed, verify_calls, verify_usage = _verify_error_with_crop(
                            model=model,
                            image_path=img_path,
                            bbox=bbox,
                            error=error,
                            max_tokens=max_tokens,
                        )
                        api_calls += verify_calls
                        cc._add_call_usage(token_usage, verify_usage)
                        if not confirmed:
                            continue
                errors.append(error)
            break
        except Exception:
            if attempt < cc.VERIFIER_PARSE_RETRIES:
                print(f"    ↺ 슬라이드 {slide_number} 오류 JSON 파싱 재시도 ({attempt+1}/{cc.VERIFIER_PARSE_RETRIES})")

    if errors is None:
        return [], True, api_calls, token_usage

    mech_errors, mech_calls, mech_usage = _check_code_syntax_mechanical(
        model=model,
        slide=slide,
        img_dir=img_dir,
        max_tokens=max_tokens,
    )
    errors.extend(mech_errors)
    api_calls += mech_calls
    token_usage = cc._merge_token_usage(token_usage, mech_usage)
    return errors, False, api_calls, token_usage


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
    work_items_by_model: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for model in models:
        work_items_by_model[model] = []
        for slide in slides:
            work_items_by_model[model].append((model, slide))

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

    # ``max_workers``는 모델별 한도다. 현재 기본은 단일 모델이지만, 다중
    # 모델 설정에서도 각 모델이 독립적으로 같은 동시성을 사용한다.
    def _run_model_slides(model: str, work_items: list[tuple[str, dict[str, Any]]]) -> tuple[str, list[dict[str, Any]], list[tuple[tuple[str, dict[str, Any]], Exception]]]:
        completed: list[dict[str, Any]] = []
        failed: list[tuple[tuple[str, dict[str, Any]], Exception]] = []
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(work_items)))) as executor:
            futures = {executor.submit(worker, args): args for args in work_items}
            for future in as_completed(futures):
                args = futures[future]
                try:
                    completed.append(future.result())
                except Exception as exc:
                    failed.append((args, exc))
        return model, completed, failed

    with ThreadPoolExecutor(max_workers=max(1, len(work_items_by_model))) as model_executor:
        model_futures = {
            model_executor.submit(_run_model_slides, model, work_items): model
            for model, work_items in work_items_by_model.items()
        }
        for model_future in as_completed(model_futures):
            model, completed, failed = model_future.result()
            for result in completed:
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
            for args, exc in failed:
                model_results[model]["status"] = "partial_failed"
                model_results[model]["batch_errors"].append(
                    {"slide_number": _slide_number(args[1]), "error": str(exc)}
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
    slim_model_results = {
        model: {
            key: value
            for key, value in result.items()
            if key != "slide_errors"
        }
        for model, result in model_results.items()
    }

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
        "model_results": slim_model_results,
        "token_usage": dict(token_usage),
    }
