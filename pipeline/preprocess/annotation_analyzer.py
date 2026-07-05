"""
annotation_analyzer.py
=======================
    slide_extractor.py의 출력(clean final + annot 프레임 쌍)을 받아
각 필기/강조가 어떤 콘텐츠를 타겟하는지 Gemini Vision으로 분석합니다.

Input : output_slides/ (slide_extractor.py 출력 디렉토리)
Output: output/annotation_analysis.json

분석 흐름:
  1. clean final + annot 이미지 쌍 구성
  2. pixel diff로 필기 영역 마스크 생성
  3. Gemini에게 (base, annot, mask) 세 장 전달 → 구조화 JSON 반환
  4. 결과를 knowledge graph Stage 1 (t1_structure) 형식과 호환되도록 저장

Usage:
    python annotation_analyzer.py --slides output_slides/ --output output/annotation_analysis.json
"""

import os
import re
import time
import cv2
import json
import base64
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# 외부 라이브러리 노이즈 로그 억제
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.ai.generativelanguage").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)

from .config import GEMINI_GENERATIVE_MODEL

VLM_MODEL = os.getenv("ANNOTATION_VLM_MODEL", GEMINI_GENERATIVE_MODEL)
ANNOTATION_THINKING_BUDGET = int(os.getenv("ANNOTATION_THINKING_BUDGET", "512"))


def _sanitize_raw(raw: str) -> str:
    """Gemini 응답에서 JSON 파싱을 방해하는 요소 제거"""
    # 마크다운 코드블록 제거
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    # JSON 문자열 내부의 unescaped 제어문자 제거
    raw = re.sub(r'(?<!\\)\n', ' ', raw)
    raw = re.sub(r'(?<!\\)\r', ' ', raw)
    raw = re.sub(r'(?<!\\)\t', ' ', raw)
    return raw


def _parse_json_safe(raw: str) -> dict:
    """
    1차: 표준 json.loads
    2차: json_repair 라이브러리 (잘린 JSON, 특수문자 복구)
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        from json_repair import repair_json
        result = json.loads(repair_json(raw))
        log.warning("  json_repair로 복구 성공")
        return result
    except Exception as e:
        raise json.JSONDecodeError(f"json_repair도 실패: {e}", raw, 0)


_LEGACY_EVENT_KEYS = (
    "slide_number",
    "annot_index",
    "timestamp_sec",
    "base_path",
    "annot_path",
    "region_bboxes",
    "annotations",
    "slide_summary",
)

_LEGACY_REGION_KEYS = ("id", "x", "y", "w", "h")
_LEGACY_ANNOTATION_KEYS = (
    "id",
    "type",
    "target_type",
    "target_content",
    "handwritten_content",
    "emphasis_intent",
    "annotation_bbox",
    "target_bbox",
    "position",
    "confidence",
)


def _legacy_region_bbox(region: dict) -> dict:
    return {key: region.get(key) for key in _LEGACY_REGION_KEYS if key in region}


def _legacy_annotation(annotation: dict) -> dict:
    return {
        key: annotation.get(key)
        for key in _LEGACY_ANNOTATION_KEYS
        if key in annotation
    }


def _legacy_annotation_event(event: dict) -> dict:
    legacy = {key: event.get(key) for key in _LEGACY_EVENT_KEYS if key in event}
    legacy["region_bboxes"] = [
        _legacy_region_bbox(region)
        for region in (event.get("region_bboxes") or [])
        if isinstance(region, dict)
    ]
    legacy["annotations"] = [
        _legacy_annotation(annotation)
        for annotation in (event.get("annotations") or [])
        if isinstance(annotation, dict)
    ]
    legacy["slide_summary"] = str(event.get("slide_summary") or "")
    return legacy


def _to_legacy_annotation_schema(events: list[dict]) -> list[dict]:
    return [_legacy_annotation_event(event) for event in events]


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
class Config:
    DIFF_PIXEL_THRESHOLD  = 30    # 픽셀 diff 절댓값 임계 (그레이스케일)
                                  # 양쪽 모두 동일 해상도로 다운스케일 후 비교하므로
                                  # 보간 아티팩트가 없어 낮은 값으로도 충분
                                  # (slide_002 기준 p99≈23, 실제 필기는 >> 30)
    DIFF_RESIZE_WIDTH     = 960   # diff 계산용 다운스케일 너비
                                  # Gemini 전송 이미지는 원본 해상도 유지
    DIFF_DILATE_KERNEL    = 5     # 마스크 팽창 커널 크기 (획 주변 여백 확보)
    DIFF_CLOSE_KERNEL     = 5     # morphological closing 커널 (인접 획을 하나의 컴포넌트로 합침)
    MASK_MIN_AREA         = 50    # 이보다 작은 연결 컴포넌트는 노이즈로 제거
                                  # 마우스 포인터(점) 감지를 위해 낮게 유지


# ──────────────────────────────────────────────
# 이미지 쌍 로딩
# ──────────────────────────────────────────────
def load_slide_pairs(slides_dir: str) -> list[dict]:
    """
    output_slides/ 디렉토리에서 (base, [annot_01, annot_02, ...]) 쌍을 구성합니다.
    반환 형식:
        [
          {
            "slide_number": 1,
            "base_path": "...",
            "annot_paths": ["...", "..."]
          },
          ...
        ]
    """
    slides_path = Path(slides_dir)
    metadata_path = slides_path / "metadata.json"

    if metadata_path.exists():
        # metadata.json이 있으면 정확히 파싱
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        pairs: dict[int, dict] = {}
        last_path_by_scene: dict[int, str] = {}
        seen_scenes: set[int] = set()

        ordered_metadata = sorted(
            metadata,
            key=lambda entry: (
                float(entry.get("timestamp_sec", 0.0) or 0.0),
                int(entry.get("annot_index", 0) or 0),
            ),
        )

        for entry in ordered_metadata:
            if entry.get("scene_type") == "video":
                continue

            scene_idx = entry.get("scene_index")
            logical_slide_no = entry.get(
                "slide_number",
                entry.get("slide_canonical_index"),
            )
            if not isinstance(scene_idx, int) or not isinstance(logical_slide_no, int):
                continue

            pair = pairs.setdefault(
                logical_slide_no,
                {
                    "slide_number": logical_slide_no,
                    "scene_indices": [],
                    "base_path": None,
                    "annot_paths": [],
                    "annot_timestamps": [],
                    "annot_prev_paths": [],
                    "annot_indices": [],
                },
            )
            if scene_idx not in pair["scene_indices"]:
                pair["scene_indices"].append(scene_idx)
            seen_scenes.add(scene_idx)

            full_path = str(slides_path / entry["filename"])
            capture_type = entry.get("capture_type")
            annot_idx = int(entry.get("annot_index", 0) or 0)
            if capture_type == "base":
                if pair["base_path"] is None:
                    pair["base_path"] = full_path
                pair["original_base_path"] = full_path
                last_path_by_scene[scene_idx] = full_path
                continue

            if capture_type not in ("annot", "annotation"):
                if annot_idx == 0 and pair["base_path"] is None:
                    pair["base_path"] = full_path
                    last_path_by_scene[scene_idx] = full_path
                continue

            prev_path = last_path_by_scene.get(scene_idx)
            if prev_path is None:
                prev_path = pair["base_path"] or full_path

            pair["annot_paths"].append(full_path)
            pair["annot_timestamps"].append(float(entry.get("timestamp_sec", 0.0) or 0.0))
            pair["annot_prev_paths"].append(prev_path)
            pair["annot_indices"].append(
                int(entry.get("slide_annot_index", len(pair["annot_paths"])) or len(pair["annot_paths"]))
            )
            last_path_by_scene[scene_idx] = full_path

        result = [v for v in pairs.values() if v["base_path"] is not None]
        result.sort(key=lambda x: x["slide_number"])
    else:
        # metadata 없을 경우 파일명 패턴으로 폴백
        log.warning("metadata.json 없음 - 파일명 패턴으로 쌍 구성")
        result = _load_pairs_by_filename(slides_path)

    total_annots = sum(len(p["annot_paths"]) for p in result)
    scene_total = len({scene_idx for pair in result for scene_idx in pair.get("scene_indices", [])})
    if scene_total:
        log.info(f"logical slide {len(result)}개 로드 완료 (scene {scene_total}개, 총 annot: {total_annots}개)")
    else:
        log.info(f"슬라이드 {len(result)}개 로드 완료 (총 annot: {total_annots}개)")
    return result


def _load_pairs_by_filename(slides_path: Path) -> list[dict]:
    bases = sorted(slides_path.glob("scene_*_base.jpg"))
    pairs = []
    for base in bases:
        idx = int(base.name.split("_")[1])
        annots = sorted(slides_path.glob(f"scene_{idx:03d}_annot_*.jpg"))
        pairs.append({
            "slide_number": idx,
            "base_path": str(base),
            "original_base_path": str(base),
            "annot_paths": [str(a) for a in annots],
        })
    return pairs


# ──────────────────────────────────────────────
# Diff 마스크 생성
# ──────────────────────────────────────────────
def build_diff_mask(base_path: str, annot_path: str, cfg: Config) -> tuple[np.ndarray, list[dict]]:
    """
    base와 annot 이미지의 픽셀 차이로 필기 영역 마스크를 만듭니다.

    diff 계산 전략:
      - Gemini 전송용 이미지는 원본 그대로 유지 (분석 품질 보장)
      - diff 계산만 양쪽 모두 DIFF_RESIZE_WIDTH로 다운스케일하여 수행
        → base(원본 해상도)와 annot(영상 캡처본) 간 해상도 차이로 인한
          리사이즈 보간 아티팩트를 원천 제거
      - bbox 좌표는 정규화(0.0~1.0) 비율이므로 해상도 무관하게 원본에 적용 가능

    반환:
      mask_bgr     : 3채널 BGR 마스크 이미지 (흰색=필기 영역, 검정=배경)
                     Gemini 전송용이므로 DIFF_RESIZE_WIDTH 해상도
      region_bboxes: 각 필기 영역의 정규화 bbox 리스트
                     [{"id": 1, "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.05}, ...]
    """
    base  = cv2.imread(base_path)
    annot = cv2.imread(annot_path)

    # diff 계산용: 양쪽 모두 동일 해상도로 다운스케일
    def _resize_for_diff(img: np.ndarray, width: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = width / w
        return cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)

    base_small  = _resize_for_diff(base,  cfg.DIFF_RESIZE_WIDTH)
    annot_small = _resize_for_diff(annot, cfg.DIFF_RESIZE_WIDTH)

    H, W = base_small.shape[:2]

    # 그레이스케일 diff (동일 해상도 기준)
    base_gray  = cv2.cvtColor(base_small,  cv2.COLOR_BGR2GRAY).astype(np.int16)
    annot_gray = cv2.cvtColor(annot_small, cv2.COLOR_BGR2GRAY).astype(np.int16)
    diff       = np.abs(base_gray - annot_gray).astype(np.uint8)

    # 임계값 이진화
    _, mask = cv2.threshold(diff, cfg.DIFF_PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY)

    # closing: 인접한 필기 획을 하나의 컴포넌트로 합침
    # (밑줄처럼 연속된 획이 여러 조각으로 쪼개지는 것 방지)
    close_kernel = np.ones((cfg.DIFF_CLOSE_KERNEL, cfg.DIFF_CLOSE_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # 노이즈 제거: 작은 컴포넌트 제거 + bbox 수집
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    region_bboxes = []
    region_id = 0
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= cfg.MASK_MIN_AREA:
            clean_mask[labels == label] = 255
            region_id += 1
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            region_bboxes.append({
                "id": region_id,
                "x": float(round(x / W, 4)),
                "y": float(round(y / H, 4)),
                "w": float(round(w / W, 4)),
                "h": float(round(h / H, 4)),
            })

    # 팽창: 필기 획 주변으로 마스크 확장
    kernel = np.ones((cfg.DIFF_DILATE_KERNEL, cfg.DIFF_DILATE_KERNEL), np.uint8)
    dilated = cv2.dilate(clean_mask, kernel, iterations=1)

    # 3채널로 변환 (Gemini 전송용)
    return cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR), region_bboxes


# ──────────────────────────────────────────────
# 시간 단계별 색상 팔레트 (BGR)
# ──────────────────────────────────────────────
TEMPORAL_COLORS = [
    (0, 0, 255),      # 빨강 (step 1)
    (255, 0, 0),      # 파랑 (step 2)
    (0, 255, 0),      # 초록 (step 3)
    (0, 255, 255),    # 노랑 (step 4)
    (255, 0, 255),    # 마젠타 (step 5)
    (255, 255, 0),    # 시안 (step 6)
    (0, 165, 255),    # 주황 (step 7)
    (128, 0, 128),    # 보라 (step 8)
    (0, 128, 255),    # 다크 오렌지 (step 9)
    (255, 128, 0),    # 스카이 블루 (step 10)
    (128, 255, 0),    # 라임 (step 11)
    (0, 128, 128),    # 올리브 (step 12)
    (128, 128, 255),  # 살몬 (step 13)
    (255, 128, 128),  # 라이트 블루 (step 14)
    (128, 255, 128),  # 라이트 그린 (step 15)
]

TEMPORAL_COLOR_NAMES = [
    "빨강", "파랑", "초록", "노랑", "마젠타", "시안",
    "주황", "보라", "다크오렌지", "스카이블루", "라임",
    "올리브", "살몬", "라이트블루", "라이트그린",
]


def _describe_position(x: float, y: float, w: float, h: float) -> str:
    """정규화 좌표로부터 사람이 읽을 수 있는 위치 설명 생성"""
    cx, cy = x + w / 2, y + h / 2
    if cy < 0.33:
        v = "상단"
    elif cy < 0.66:
        v = "중앙"
    else:
        v = "하단"
    if cx < 0.33:
        h_pos = "좌측"
    elif cx < 0.66:
        h_pos = "중앙"
    else:
        h_pos = "우측"
    if v == "중앙" and h_pos == "중앙":
        return "슬라이드 중앙부"
    return f"슬라이드 {v} {h_pos}"


def _describe_size(w: float, h: float) -> str:
    """정규화 bbox 크기로부터 상대적 크기 설명 생성"""
    area = w * h
    if area > 0.05:
        return "큰 영역"
    elif area > 0.01:
        return "중간 영역"
    elif area > 0.002:
        return "작은 영역"
    else:
        return "점/미세 영역"


def _annot_source_label(annot_index: int) -> str:
    return f"annot_{annot_index:02d}"


def build_composite_diff_mask(
    base_path: str,
    annot_paths: list[str],
    annot_timestamps: list[float],
    annot_prev_paths: Optional[list[str]],
    annot_indices: Optional[list[int]],
    cfg: Config,
) -> tuple[np.ndarray, str, list[dict], list[dict]]:
    """
    슬라이드의 모든 어노테이션 프레임에 대해 증분 diff를 계산하고,
    시간 단계별 색상으로 구분된 합성 마스크와 변화 로그를 생성합니다.

    반환:
      composite_mask : 3채널 BGR 합성 마스크 (시간 단계별 색상 구분)
      change_log     : Gemini 프롬프트에 삽입할 변화 로그 JSON 문자열
      per_step_data  : 각 단계별 메타데이터 리스트
                       [{"annot_index", "timestamp_sec", "annot_path",
                         "prev_path", "region_bboxes", "diff_mask"}, ...]
      change_tracks  : 로컬에서 누적한 좌표 변화 track 목록
    """
    per_step_data = []

    # 1단계: 각 증분 diff 계산
    for step_idx, (annot_path, ts) in enumerate(zip(annot_paths, annot_timestamps)):
        prev_path = (
            annot_prev_paths[step_idx]
            if annot_prev_paths and step_idx < len(annot_prev_paths)
            else base_path
        )
        logical_annot_index = (
            int(annot_indices[step_idx])
            if annot_indices and step_idx < len(annot_indices)
            else step_idx + 1
        )
        diff_mask, region_bboxes = build_diff_mask(prev_path, annot_path, cfg)
        per_step_data.append({
            "annot_index": logical_annot_index,
            "timestamp_sec": ts,
            "annot_path": annot_path,
            "prev_path": prev_path,
            "region_bboxes": [dict(r) for r in region_bboxes],
            "diff_mask": diff_mask,
        })

    # 2단계: bbox를 시간 축으로 연결하여 change track 생성
    change_tracks: list[dict] = []
    next_track_num = 1

    for step in per_step_data:
        for region in step["region_bboxes"]:
            best_track = None
            best_score = float("-inf")
            for track in change_tracks:
                step_gap = step["annot_index"] - track["last_seen_annot_index"]
                if step_gap > 1:
                    continue

                candidate_bbox = track["step_history"][-1]["bbox"]
                iou = _bbox_iou(region, candidate_bbox)
                inter_min = _bbox_intersection_over_min(region, candidate_bbox)
                gap = _bbox_gap(region, candidate_bbox)

                if step_gap == 0:
                    if iou <= 0.0 and inter_min < 0.6:
                        continue
                    score = max(iou, inter_min) - gap
                else:
                    gap_limit = max(
                        0.008,
                        min(_bbox_diag(region), _bbox_diag(candidate_bbox)) * 0.18,
                    )
                    if iou <= 0.0 and inter_min < 0.2 and gap > gap_limit:
                        continue
                    score = max(iou, inter_min) - gap

                if score > best_score:
                    best_score = score
                    best_track = track

            step_entry = {
                "annot_index": step["annot_index"],
                "annot_source": _annot_source_label(step["annot_index"]),
                "timestamp_sec": round(float(step["timestamp_sec"]), 2),
                "bbox": {
                    "x": region["x"],
                    "y": region["y"],
                    "w": region["w"],
                    "h": region["h"],
                },
            }

            if best_track is None:
                track_id = f"track_{next_track_num:03d}"
                next_track_num += 1
                best_track = {
                    "track_id": track_id,
                    "first_seen_annot_index": step["annot_index"],
                    "first_seen_annot_source": _annot_source_label(step["annot_index"]),
                    "first_seen_timestamp_sec": round(float(step["timestamp_sec"]), 2),
                    "last_seen_annot_index": step["annot_index"],
                    "last_seen_annot_source": _annot_source_label(step["annot_index"]),
                    "last_seen_timestamp_sec": round(float(step["timestamp_sec"]), 2),
                    "merged_bbox": {
                        "x": region["x"],
                        "y": region["y"],
                        "w": region["w"],
                        "h": region["h"],
                    },
                    "step_history": [step_entry],
                }
                change_tracks.append(best_track)
            else:
                best_track["last_seen_annot_index"] = step["annot_index"]
                best_track["last_seen_annot_source"] = _annot_source_label(step["annot_index"])
                best_track["last_seen_timestamp_sec"] = round(float(step["timestamp_sec"]), 2)
                best_track["merged_bbox"] = _bbox_union(best_track["merged_bbox"], region)
                best_track["step_history"].append(step_entry)

            region["track_id"] = best_track["track_id"]

    # 2단계: 합성 마스크 생성 (시간 단계별 색상)
    # 첫 번째 diff mask의 크기를 기준으로 합성 마스크 생성
    h, w = per_step_data[0]["diff_mask"].shape[:2]
    composite = np.zeros((h, w, 3), dtype=np.uint8)

    for step_idx, step in enumerate(per_step_data):
        color = TEMPORAL_COLORS[step_idx % len(TEMPORAL_COLORS)]
        mask_gray = cv2.cvtColor(step["diff_mask"], cv2.COLOR_BGR2GRAY)
        # 해당 단계의 필기 영역을 색상으로 칠함 (이후 단계가 덮어씀)
        composite[mask_gray > 0] = color

    # 3단계: 변화 로그 JSON 생성 (LLM용)
    prompt_tracks = []
    for track in change_tracks:
        merged = track["merged_bbox"]
        prompt_tracks.append({
            "track_id": track["track_id"],
            "first_seen_annot": track["first_seen_annot_source"],
            "first_seen_timestamp_sec": track["first_seen_timestamp_sec"],
            "last_seen_annot": track["last_seen_annot_source"],
            "last_seen_timestamp_sec": track["last_seen_timestamp_sec"],
            "merged_bbox": merged,
            "position_hint": _describe_position(merged["x"], merged["y"], merged["w"], merged["h"]),
            "size_hint": _describe_size(merged["w"], merged["h"]),
            "steps": track["step_history"],
        })

    change_log = json.dumps(
        {"change_tracks": prompt_tracks},
        ensure_ascii=False,
        indent=2,
    )

    return composite, change_log, per_step_data, change_tracks


def encode_image_b64(image_path: str) -> str:
    """이미지 파일을 base64 인코딩"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_ndarray_b64(img: np.ndarray) -> str:
    """numpy 배열을 JPEG base64 인코딩"""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _call_visual_json_model(
    prompt: str,
    image_bytes_list: list[bytes],
    max_output_tokens: int,
) -> str:
    """
    이미지 + 프롬프트를 VLM에 보내 JSON 문자열을 반환한다.
    현재 기본 모델은 Gemini Vision(gemini-2.5-flash)이며, 필요 시 다른 VLM도 지원한다.
    """
    model = str(VLM_MODEL or "").strip()

    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        from openai import OpenAI
        from .config import OPENAI_API_KEY

        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

        content = []
        for img in image_bytes_list:
            b64 = base64.b64encode(img).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})

        last_error = None
        for attempt in range(1, 5):
            try:
                client = OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    max_completion_tokens=max_output_tokens,
                    response_format={"type": "json_object"},
                )
                try:
                    from .cost_report import record_model_call

                    record_model_call(
                        stage="stage3a_annotation_analyzer",
                        provider="openai",
                        model=model,
                        response=response,
                        image_count=len(image_bytes_list),
                        prompt_chars=len(prompt),
                    )
                except Exception:
                    pass
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                last_error = e
                err_msg = str(e)
                retryable = (
                    "missing_scope" in err_msg
                    or "model.request" in err_msg
                    or "Rate limit" in err_msg
                    or "429" in err_msg
                    or "timed out" in err_msg.lower()
                    or "connection" in err_msg.lower()
                )
                if attempt >= 4 or not retryable:
                    raise
                sleep_sec = min(2 ** (attempt - 1), 6)
                log.warning(
                    f"  OpenAI 호출 재시도 {attempt}/4 ({sleep_sec}s 대기): {type(e).__name__}: {err_msg}"
                )
                time.sleep(sleep_sec)

        raise last_error

    from google.genai import types
    from .config import gemini_client as client

    parts = []
    for idx, img in enumerate(image_bytes_list, start=1):
        parts.append(types.Part(text=f"Image {idx}:"))
        parts.append(types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=img)))
    parts.append(types.Part(text=prompt))

    response = client.models.generate_content(
        model=model,
        contents=[types.Content(parts=parts)],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_budget=ANNOTATION_THINKING_BUDGET
            ),
        )
    )
    try:
        from .cost_report import record_model_call

        record_model_call(
            stage="stage3a_annotation_analyzer",
            provider="google",
            model=model,
            response=response,
            image_count=len(image_bytes_list),
            prompt_chars=len(prompt),
        )
    except Exception:
        pass
    return (response.text or "").strip()


# ──────────────────────────────────────────────
# Gemini 분석
# ──────────────────────────────────────────────
ANALYSIS_PROMPT_TEMPLATE = """
당신은 강의 슬라이드 분석 전문가입니다.
세 장의 이미지가 제공됩니다:
  - Image 1 (BASE): 필기 전 원본 슬라이드
  - Image 2 (ANNOTATED): 교수가 필기/강조를 추가한 슬라이드
  - Image 3 (DIFF MASK): 흰색 영역이 필기가 추가된 위치를 나타내는 마스크

아래는 DIFF MASK에서 검출된 필기 영역의 번호와 정규화 좌표(0.0~1.0, 좌상단 기준)입니다:
{region_bboxes_str}

각 필기 표시를 분석하여 아래 JSON 형식으로만 응답하세요.
JSON 외의 설명, 마크다운 코드블록을 절대 포함하지 마세요.

{{
  "annotations": [
    {{
      "id": 1,
      "type": "circle" | "underline" | "arrow" | "handwritten_text" | "bracket" | "cross" | "other",
      "target_type": "text" | "image" | "diagram_element" | "none",
      "target_content": "강조 대상이 되는 원본 슬라이드의 텍스트를 하나의 문자열로 그대로 기재 (리스트 불가, target_keywords 필드 사용 금지). 필기 텍스트라면 null",
      "handwritten_content": "필기 텍스트인 경우 읽히는 내용. 강조 표시라면 null",
      "emphasis_intent": "이 표시가 전달하는 의도를 한 문장으로 (예: 'CPU와 캐시를 하드웨어 자원으로 강조', '1.5GB 메모리 크기 추가 설명')",
      "annotation_bbox": {{"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}},
      "target_bbox": {{"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}},
      "position": "top-left" | "top-right" | "bottom-left" | "bottom-right" | "center" | "top" | "bottom" | "left" | "right",
      "confidence": "high" | "medium" | "low"
    }}
  ],
  "slide_summary": "이 슬라이드에서 필기를 통해 교수가 강조한 전체적인 포인트를 2-3문장으로 요약"
}}

필드 설명:

[annotation_bbox]
  위에 제공된 필기 영역 번호 중 이 annotation에 해당하는 영역의 좌표를 그대로 사용하세요.
  값은 정규화 좌표 {{"x": float, "y": float, "w": float, "h": float}} 형식.

[target_bbox]
  강조 대상(target_content)이 슬라이드에서 차지하는 위치를 BASE 이미지 기준으로 추정하세요.
  target_type이 "none"이면 null.
  값은 정규화 좌표 {{"x": float, "y": float, "w": float, "h": float}} 형식.

분류 기준:
- circle: 텍스트나 요소를 동그라미로 감쌈
- underline: 텍스트 아래에 선을 그음
- arrow: 화살표로 특정 요소를 가리킴
- handwritten_text: 새로운 단어/숫자/문장을 직접 씀
- bracket: 중괄호나 대괄호 형태로 묶음
- cross: X 표시로 취소하거나 부정
- other: 위 유형에 해당하지 않는 경우

confidence 기준:
- high: 마스크와 타겟이 명확히 일치
- medium: 타겟 추정 가능하나 불확실
- low: 마스크가 불명확하거나 필기가 판독 불가

중요: target_content는 반드시 문자열(string)이어야 하며, 리스트나 배열 형태로 출력하지 말 것.
target_keywords 같은 임의 필드를 추가하지 말 것. 위 JSON 스키마를 정확히 따를 것.
"""

# ──────────────────────────────────────────────
# change-track 통합 분석 프롬프트 (슬라이드당 1회 호출, 2장 이미지)
# ──────────────────────────────────────────────
CHANGE_TRACK_ANALYSIS_PROMPT_TEMPLATE = """
당신은 강의 슬라이드 분석 전문가입니다.
두 장의 이미지가 제공됩니다:
  - Image 1 (BASE): 필기 전 원본 슬라이드
  - Image 2 (FINAL): 교수의 모든 필기가 완료된 최종 슬라이드

교수는 강의 중 여러 시점에 걸쳐 필기를 추가했습니다.
아래 JSON은 base→annot_01→annot_02 ... 순서의 증분 diff를 로컬에서 누적해 만든 change track 목록입니다.
각 track은 같은 위치에서 이어진 필기 변화들을 하나로 묶은 것이며,
track마다 처음 등장한 annot 시점과 누적 bbox가 포함되어 있습니다.

{change_tracks_json}

Image 1 (BASE)에 없고 Image 2 (FINAL)에만 있는 표시가 교수의 필기입니다.
위 change track을 참고하여 FINAL 이미지에서 실제로 보이는 교수 필기를 semantic annotation 단위로 묶어 분석하세요.
하나의 semantic annotation이 여러 track으로 구성될 수 있습니다. 예: 화살표 몸통+머리, 두 조각으로 나뉜 밑줄.

아래 JSON 형식으로만 응답하세요. JSON 외의 설명, 마크다운 코드블록을 절대 포함하지 마세요.

{{
  "annotations": [
    {{
      "id": 1,
      "track_ids": ["track_001"],
      "type": "circle" | "underline" | "arrow" | "handwritten_text" | "bracket" | "cross" | "other",
      "target_type": "text" | "image" | "diagram_element" | "none",
      "target_content": "강조 대상이 되는 원본 슬라이드의 텍스트를 하나의 문자열로 그대로 기재 (리스트 불가). 필기 텍스트라면 null",
      "handwritten_content": "필기 텍스트인 경우 읽히는 내용. 강조 표시라면 null",
      "emphasis_intent": "이 표시가 전달하는 의도를 한 문장으로",
      "annotation_bbox": {{"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}},
      "target_bbox": {{"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}},
      "position": "top-left" | "top-right" | "bottom-left" | "bottom-right" | "center" | "top" | "bottom" | "left" | "right",
      "confidence": "high" | "medium" | "low"
    }}
  ],
  "slide_summary": "이 슬라이드에서 필기를 통해 교수가 강조한 전체적인 포인트를 2-3문장으로 요약"
}}

필드 설명:

[track_ids]
  반드시 위 JSON의 track_id만 사용하세요.
  하나의 semantic annotation이 여러 change track으로 구성된 경우, 해당 track_id를 모두 배열에 넣으세요.
  예: ["track_003", "track_004"]

[annotation_bbox]
  track의 좌표를 참고하되, Image 2 (FINAL)에서 실제 필기 위치를 확인하여 정확한 좌표를 기재하세요.
  값은 정규화 좌표 {{"x": float, "y": float, "w": float, "h": float}} 형식.

[target_bbox]
  강조 대상(target_content)이 슬라이드에서 차지하는 위치를 BASE 이미지 기준으로 추정하세요.
  target_type이 "none"이면 null.

분류 기준:
- circle: 텍스트나 요소를 동그라미로 감쌈
- underline: 텍스트 아래에 선을 그음
- arrow: 화살표로 특정 요소를 가리킴
- handwritten_text: 새로운 단어/숫자/문장을 직접 씀
- bracket: 중괄호나 대괄호 형태로 묶음
- cross: X 표시로 취소하거나 부정
- other: 위 유형에 해당하지 않는 경우

confidence 기준:
- high: 마스크와 타겟이 명확히 일치
- medium: 타겟 추정 가능하나 불확실
- low: 마스크가 불명확하거나 필기가 판독 불가

중요:
- 동일한 필기를 중복으로 포함하지 마세요. 하나의 밑줄, 하나의 동그라미는 1개의 annotation입니다.
- 보이는 필기가 여러 change track으로 나뉘어 있어도 하나의 semantic annotation이면 1개로 묶으세요.
- 존재하지 않는 track_id를 만들지 마세요.
- target_content는 반드시 문자열(string)이어야 하며, 리스트나 배열 형태로 출력하지 말 것.
- 위 JSON 스키마를 정확히 따를 것.
"""


def analyze_annotation_pair(
    base_path: str,
    annot_path: str,
    diff_mask: np.ndarray,
    region_bboxes: list[dict],
    slide_number: int,
    annot_index: int,
    timestamp_sec: float = 0.0,
) -> dict:
    """
    (base, annot, diff_mask) 세 장과 필기 영역 bbox 목록을 VLM에게 전달하여
    분석 결과를 반환합니다.

    region_bboxes: build_diff_mask()가 반환한 정규화 bbox 리스트.
                   VLM이 annotation_bbox 할당에 사용.
    """
    log.info(f"  분석 중: slide_{slide_number:03d}_annot_{annot_index:02d} "
             f"(필기 영역 {len(region_bboxes)}개)")

    # bbox 목록 문자열 생성 (프롬프트 주입용)
    # 면적(w×h) 기준 상위 TOP_N_REGIONS개만 사용:
    #   - 실제 필기(밑줄, 동그라미 등)는 면적이 크고, 렌더링 노이즈는 작음
    #   - 전체를 주입하면 프롬프트/응답 토큰이 과도하게 늘어 json_repair 빈도 증가
    TOP_N_REGIONS = 10
    top_bboxes = sorted(region_bboxes, key=lambda r: r["w"] * r["h"], reverse=True)[:TOP_N_REGIONS]

    if top_bboxes:
        bbox_lines = [
            f"  영역 {r['id']}: x={r['x']}, y={r['y']}, w={r['w']}, h={r['h']}"
            for r in top_bboxes
        ]
        region_bboxes_str = "\n".join(bbox_lines)
        if len(region_bboxes) > TOP_N_REGIONS:
            region_bboxes_str += f"\n  (전체 {len(region_bboxes)}개 중 면적 상위 {TOP_N_REGIONS}개만 표시)"
    else:
        region_bboxes_str = "  (검출된 필기 영역 없음 — source='slide' 항목만 존재할 수 있음)"

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(region_bboxes_str=region_bboxes_str)

    try:
        with open(base_path, "rb") as f:
            base_bytes = f.read()
        with open(annot_path, "rb") as f:
            annot_bytes = f.read()
        _, mask_buf = cv2.imencode(".jpg", diff_mask, [cv2.IMWRITE_JPEG_QUALITY, 90])
        raw = _call_visual_json_model(
            prompt=prompt,
            image_bytes_list=[base_bytes, annot_bytes, mask_buf.tobytes()],
            max_output_tokens=8192,
        )
        raw = _sanitize_raw(raw)
        result = _parse_json_safe(raw)
        result["slide_number"]   = slide_number
        result["annot_index"]    = annot_index
        result["timestamp_sec"]  = timestamp_sec
        result["base_path"]      = base_path
        result["annot_path"]     = annot_path
        result["region_bboxes"]  = region_bboxes   # 원시 마스크 영역 목록 (디버그용)
        return result

    except json.JSONDecodeError as e:
        log.warning(f"  JSON 파싱 실패 (slide {slide_number}, annot {annot_index}): {e}")
        return {
            "slide_number":  slide_number,
            "annot_index":   annot_index,
            "timestamp_sec": timestamp_sec,
            "base_path":     base_path,
            "annot_path":    annot_path,
            "region_bboxes": region_bboxes,
            "error":         f"JSON parse error: {e}",
            "raw_response":  raw if 'raw' in locals() else "no response",
            "annotations":   [],
            "slide_summary": "",
        }
    except Exception as e:
        log.error(f"  VLM API 오류 (slide {slide_number}, annot {annot_index}): {e}")
        return {
            "slide_number":  slide_number,
            "annot_index":   annot_index,
            "timestamp_sec": timestamp_sec,
            "region_bboxes": region_bboxes,
            "error": str(e),
            "annotations":   [],
            "slide_summary": "",
        }


# ──────────────────────────────────────────────
# 슬라이드 change track 통합 분석
# ──────────────────────────────────────────────
def analyze_slide_change_tracks(
    base_path: str,
    annot_paths: list[str],
    annot_timestamps: list[float],
    annot_prev_paths: Optional[list[str]],
    annot_indices: Optional[list[int]],
    slide_number: int,
    cfg: Config,
    save_masks: bool = False,
    mask_dir: Optional[Path] = None,
) -> list[dict]:
    """
    한 슬라이드의 모든 어노테이션을 change track 기준으로 통합 분석합니다.

    로컬에서 base→annot 증분 diff를 change track으로 누적한 뒤,
    VLM에는 base/final 이미지와 change track JSON만 전달합니다.
    통합 분석 요청이 실패해도 per-annot 방식으로 자동 폴백하지 않습니다.
    비용 보호를 위해 local change track 결과만 저장합니다.

    반환: 기존 analyze_all()과 동일한 형식의 list[dict]
    """
    log.info(
        f"  [change-track 분석] slide_{slide_number:03d}: "
        f"{len(annot_paths)}개 annot → 1회 요청 ({VLM_MODEL}, base/final + change tracks)"
    )

    # 증분 diff → per_step_data 생성 (로컬 처리)
    composite_mask, change_log, per_step_data, change_tracks = build_composite_diff_mask(
        base_path, annot_paths, annot_timestamps, annot_prev_paths, annot_indices, cfg
    )

    # 마스크 저장 (디버그용)
    if save_masks and mask_dir:
        cv2.imwrite(
            str(mask_dir / f"slide_{slide_number:03d}_composite_mask.jpg"),
            composite_mask
        )
        for step in per_step_data:
            mask_fname = f"slide_{slide_number:03d}_annot_{step['annot_index']:02d}_mask.jpg"
            cv2.imwrite(str(mask_dir / mask_fname), step["diff_mask"])

    if not change_tracks:
        log.warning(
            f"  [change-track 분석] slide_{slide_number:03d}: "
            "local change track 0개 → local 결과만 저장"
        )
        return _build_local_track_results(
            per_step_data=per_step_data,
            change_tracks=[],
            slide_number=slide_number,
            error="No local change tracks detected",
        )

    prompt = CHANGE_TRACK_ANALYSIS_PROMPT_TEMPLATE.format(change_tracks_json=change_log)

    # Content parts 구성: BASE(1장) + FINAL(1장) + 텍스트 프롬프트
    final_annot_path = annot_paths[-1]  # 최종 필기 완성 이미지
    with open(base_path, "rb") as f:
        base_bytes = f.read()
    with open(final_annot_path, "rb") as f:
        final_bytes = f.read()

    try:
        raw = _call_visual_json_model(
            prompt=prompt,
            image_bytes_list=[base_bytes, final_bytes],
            max_output_tokens=16384,
        )
        raw = _sanitize_raw(raw)
        track_analysis_result = _parse_json_safe(raw)

        return _build_results_from_track_analysis(
            track_analysis_result=track_analysis_result,
            per_step_data=per_step_data,
            slide_number=slide_number,
            base_path=base_path,
            change_tracks=change_tracks,
        )

    except json.JSONDecodeError as e:
        log.warning(
            f"  [change-track 분석 파싱 실패] slide {slide_number}: {e} "
            "→ local change track 결과만 저장"
        )
        return _build_local_track_results(
            per_step_data=per_step_data,
            change_tracks=change_tracks,
            slide_number=slide_number,
            error=f"Track analysis JSON parse error: {e}",
        )
    except Exception as e:
        err_msg = str(e)
        log.error(
            f"  [change-track 분석 API 오류] slide {slide_number}: {e} "
            "→ local change track 결과만 저장"
        )
        return _build_local_track_results(
            per_step_data=per_step_data,
            change_tracks=change_tracks,
            slide_number=slide_number,
            error=err_msg,
        )


def _bbox_iou(a: dict, b: dict) -> float:
    """두 정규화 bbox 간 IoU(Intersection over Union) 계산"""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _bbox_area(b: dict) -> float:
    return max(0.0, b.get("w", 0.0)) * max(0.0, b.get("h", 0.0))


def _bbox_union(a: dict, b: dict) -> dict:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    x2, y2 = max(ax2, bx2), max(ay2, by2)
    return {
        "x": float(round(x1, 4)),
        "y": float(round(y1, 4)),
        "w": float(round(max(0.0, x2 - x1), 4)),
        "h": float(round(max(0.0, y2 - y1), 4)),
    }


def _bbox_intersection_over_min(a: dict, b: dict) -> float:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    min_area = min(_bbox_area(a), _bbox_area(b))
    return inter / min_area if min_area > 0 else 0.0


def _bbox_distance(a: dict, b: dict) -> float:
    """두 bbox 중심점 간 거리"""
    acx, acy = a["x"] + a["w"] / 2, a["y"] + a["h"] / 2
    bcx, bcy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def _bbox_diag(b: dict) -> float:
    return (b["w"] ** 2 + b["h"] ** 2) ** 0.5


def _bbox_gap(a: dict, b: dict) -> float:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]
    dx = max(ax1 - bx2, bx1 - ax2, 0.0)
    dy = max(ay1 - by2, by1 - ay2, 0.0)
    return (dx ** 2 + dy ** 2) ** 0.5


def _normalize_track_ids(raw_track_ids, change_tracks_by_id: dict[str, dict]) -> list[str]:
    if isinstance(raw_track_ids, str):
        candidates = [raw_track_ids]
    elif isinstance(raw_track_ids, list):
        candidates = raw_track_ids
    else:
        candidates = []

    track_ids: list[str] = []
    seen = set()
    for item in candidates:
        track_id = str(item or "").strip()
        if not track_id or track_id in seen or track_id not in change_tracks_by_id:
            continue
        seen.add(track_id)
        track_ids.append(track_id)
    return track_ids


def _match_annotation_to_track_ids(annotation_bbox: Optional[dict], change_tracks: list[dict]) -> list[str]:
    if not annotation_bbox:
        return []

    ranked: list[tuple[float, str]] = []
    for track in change_tracks:
        merged_bbox = track["merged_bbox"]
        iou = _bbox_iou(annotation_bbox, merged_bbox)
        inter_min = _bbox_intersection_over_min(annotation_bbox, merged_bbox)
        gap = _bbox_gap(annotation_bbox, merged_bbox)
        gap_limit = max(
            0.01,
            min(_bbox_diag(annotation_bbox), _bbox_diag(merged_bbox)) * 0.35,
        )
        if iou <= 0.0 and inter_min < 0.2 and gap > gap_limit:
            continue
        score = max(iou, inter_min) - gap
        ranked.append((score, track["track_id"]))

    ranked.sort(reverse=True)
    return [track_id for _, track_id in ranked[:3]]


def _build_annotation_track_metadata(track_ids: list[str], change_tracks_by_id: dict[str, dict]) -> dict:
    tracks = [change_tracks_by_id[track_id] for track_id in track_ids if track_id in change_tracks_by_id]
    if not tracks:
        return {}

    tracks_sorted = sorted(
        tracks,
        key=lambda t: (t["first_seen_timestamp_sec"], t["first_seen_annot_index"], t["track_id"]),
    )
    primary = tracks_sorted[0]

    union_bbox = tracks_sorted[0]["merged_bbox"]
    combined_steps = []
    for track in tracks_sorted:
        union_bbox = _bbox_union(union_bbox, track["merged_bbox"])
        combined_steps.extend(track["step_history"])

    combined_steps.sort(key=lambda s: (s["timestamp_sec"], s["annot_index"], s["annot_source"]))
    last_track = max(
        tracks_sorted,
        key=lambda t: (t["last_seen_timestamp_sec"], t["last_seen_annot_index"], t["track_id"]),
    )

    return {
        "primary_track_id": primary["track_id"],
        "annot_source": primary["first_seen_annot_source"],
        "first_seen_annot_index": primary["first_seen_annot_index"],
        "first_seen_timestamp_sec": primary["first_seen_timestamp_sec"],
        "last_seen_annot_index": last_track["last_seen_annot_index"],
        "last_seen_timestamp_sec": last_track["last_seen_timestamp_sec"],
        "track_steps": combined_steps,
        "track_union_bbox": union_bbox,
    }


def _build_results_from_track_analysis(
    track_analysis_result: dict,
    per_step_data: list[dict],
    slide_number: int,
    base_path: str,
    change_tracks: list[dict],
) -> list[dict]:
    """
    change-track 통합 분석 응답의 flat annotations[]를 로컬 change track과 결합하여
    annot 첫 등장 시점 기준 per-step 결과로 분할합니다.
    """
    annotations = track_analysis_result.get("annotations", [])
    slide_summary = track_analysis_result.get("slide_summary", "")
    change_tracks_by_id = {track["track_id"]: track for track in change_tracks}

    # step별 annotations 할당 결과
    step_annotations = {step["annot_index"]: [] for step in per_step_data}

    for annot in annotations:
        normalized = dict(annot)
        track_ids = _normalize_track_ids(normalized.get("track_ids"), change_tracks_by_id)
        if not track_ids:
            track_ids = _match_annotation_to_track_ids(normalized.get("annotation_bbox"), change_tracks)

        metadata = _build_annotation_track_metadata(track_ids, change_tracks_by_id)
        if not metadata:
            fallback_step = per_step_data[0]["annot_index"]
            normalized["track_ids"] = []
            step_annotations[fallback_step].append(normalized)
            continue

        normalized["track_ids"] = track_ids
        normalized.update(metadata)
        normalized["timestamp_sec"] = metadata["first_seen_timestamp_sec"]
        if not normalized.get("annotation_bbox"):
            normalized["annotation_bbox"] = metadata["track_union_bbox"]
        step_annotations[metadata["first_seen_annot_index"]].append(normalized)

    # per-annot 형식으로 조립
    results = []
    for step in per_step_data:
        annot_idx = step["annot_index"]
        step_tracks = [
            {
                "track_id": track["track_id"],
                "first_seen_annot_index": track["first_seen_annot_index"],
                "first_seen_timestamp_sec": track["first_seen_timestamp_sec"],
                "last_seen_annot_index": track["last_seen_annot_index"],
                "last_seen_timestamp_sec": track["last_seen_timestamp_sec"],
                "merged_bbox": track["merged_bbox"],
                "step_history": track["step_history"],
            }
            for track in change_tracks
            if track["first_seen_annot_index"] == annot_idx
        ]
        results.append({
            "slide_number":  slide_number,
            "annot_index":   annot_idx,
            "timestamp_sec": step["timestamp_sec"],
            "base_path":     step["prev_path"],
            "annot_path":    step["annot_path"],
            "region_bboxes": step["region_bboxes"],
            "detected_change_tracks": step_tracks,
            "annotations":   step_annotations.get(annot_idx, []),
            "slide_summary": slide_summary,
        })

    return results


def _build_local_track_results(
    per_step_data: list[dict],
    change_tracks: list[dict],
    slide_number: int,
    error: Optional[str] = None,
) -> list[dict]:
    results = []
    for step in per_step_data:
        annot_idx = step["annot_index"]
        step_tracks = [
            {
                "track_id": track["track_id"],
                "first_seen_annot_index": track["first_seen_annot_index"],
                "first_seen_timestamp_sec": track["first_seen_timestamp_sec"],
                "last_seen_annot_index": track["last_seen_annot_index"],
                "last_seen_timestamp_sec": track["last_seen_timestamp_sec"],
                "merged_bbox": track["merged_bbox"],
                "step_history": track["step_history"],
            }
            for track in change_tracks
            if track["first_seen_annot_index"] == annot_idx
        ]
        payload = {
            "slide_number": slide_number,
            "annot_index": annot_idx,
            "timestamp_sec": step["timestamp_sec"],
            "base_path": step["prev_path"],
            "annot_path": step["annot_path"],
            "region_bboxes": step["region_bboxes"],
            "detected_change_tracks": step_tracks,
            "annotations": [],
            "slide_summary": "",
        }
        if error:
            payload["error"] = error
        results.append(payload)
    return results


def _step_change_tracks(change_tracks: list[dict], annot_idx: int) -> list[dict]:
    return [
        {
            "track_id": track["track_id"],
            "first_seen_annot_index": track["first_seen_annot_index"],
            "first_seen_timestamp_sec": track["first_seen_timestamp_sec"],
            "last_seen_annot_index": track["last_seen_annot_index"],
            "last_seen_timestamp_sec": track["last_seen_timestamp_sec"],
            "merged_bbox": track["merged_bbox"],
            "step_history": track["step_history"],
        }
        for track in change_tracks
        if track["first_seen_annot_index"] == annot_idx
    ]


def _enrich_results_with_change_tracks(results: list[dict], change_tracks: list[dict]) -> list[dict]:
    change_tracks_by_id = {track["track_id"]: track for track in change_tracks}

    for event in results:
        annot_idx = int(event.get("annot_index", 0) or 0)
        event["detected_change_tracks"] = _step_change_tracks(change_tracks, annot_idx)

        annotations = []
        for annot in event.get("annotations", []):
            normalized = dict(annot)
            track_ids = _normalize_track_ids(normalized.get("track_ids"), change_tracks_by_id)
            if not track_ids:
                track_ids = _match_annotation_to_track_ids(normalized.get("annotation_bbox"), change_tracks)
            metadata = _build_annotation_track_metadata(track_ids, change_tracks_by_id)
            if metadata:
                normalized["track_ids"] = track_ids
                normalized.update(metadata)
                normalized["timestamp_sec"] = metadata["first_seen_timestamp_sec"]
                if not normalized.get("annotation_bbox"):
                    normalized["annotation_bbox"] = metadata["track_union_bbox"]
            annotations.append(normalized)
        event["annotations"] = annotations

    return results


def _run_per_annot_mode(
    base_path: str,
    per_step_data: list[dict],
    slide_number: int,
    cfg: Config,
    save_masks: bool,
    mask_dir: Optional[Path],
    change_tracks: Optional[list[dict]] = None,
) -> list[dict]:
    """annot별 개별 호출 방식 실행"""
    log.info(f"  [per-annot 모드] slide_{slide_number:03d}: annot별 개별 호출 실행")
    results = []
    for step in per_step_data:
        result = analyze_annotation_pair(
            step["prev_path"],
            step["annot_path"],
            step["diff_mask"],
            step["region_bboxes"],
            slide_number,
            step["annot_index"],
            timestamp_sec=step["timestamp_sec"],
        )
        results.append(result)
    if change_tracks:
        return _enrich_results_with_change_tracks(results, change_tracks)
    return results


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────
def analyze_all(
    slides_dir: str,
    output_path: str,
    save_masks: bool = False,
    per_annot_mode: bool = False,
):
    """
    슬라이드 필기 분석 메인 함수.

    per_annot_mode=False (기본): 슬라이드당 1회 API 호출 (change track 통합 분석)
    per_annot_mode=True:         annot당 1회 API 호출
    """
    from .config import gemini_client

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cfg   = Config()
    pairs = load_slide_pairs(slides_dir)
    results = []

    mask_dir = Path(slides_dir) / "diff_masks" if save_masks else None
    if mask_dir:
        mask_dir.mkdir(exist_ok=True)

    for pair in pairs:
        slide_no = pair["slide_number"]
        base_path = pair["base_path"]
        scene_indices = pair.get("scene_indices", [])
        scene_label = ",".join(str(scene_idx) for scene_idx in scene_indices[:4])
        if len(scene_indices) > 4:
            scene_label += ",..."

        if not pair["annot_paths"]:
            if scene_label:
                log.info(f"[slide {slide_no} | scenes {scene_label}] 필기 없음 - 스킵")
            else:
                log.info(f"[슬라이드 {slide_no}] 필기 없음 - 스킵")
            continue

        annot_timestamps = pair.get("annot_timestamps", [0.0] * len(pair["annot_paths"]))
        if scene_label:
            log.info(f"[slide {slide_no} | scenes {scene_label}] {len(pair['annot_paths'])}개 annot 분석 시작")
        else:
            log.info(f"[슬라이드 {slide_no}] {len(pair['annot_paths'])}개 annot 분석 시작")

        if per_annot_mode:
            # ── 기존 방식: annot당 1회 호출 ──
            annot_prev_paths = pair.get("annot_prev_paths", [])
            annot_indices = pair.get("annot_indices", [])
            for step_idx, (annot_path, ts) in enumerate(
                zip(pair["annot_paths"], annot_timestamps)
            ):
                prev_path = annot_prev_paths[step_idx] if step_idx < len(annot_prev_paths) else base_path
                annot_idx = (
                    int(annot_indices[step_idx])
                    if step_idx < len(annot_indices)
                    else step_idx + 1
                )
                diff_mask, region_bboxes = build_diff_mask(prev_path, annot_path, cfg)

                if save_masks:
                    mask_fname = f"slide_{slide_no:03d}_annot_{annot_idx:02d}_mask.jpg"
                    cv2.imwrite(str(mask_dir / mask_fname), diff_mask)

                result = analyze_annotation_pair(
                    prev_path, annot_path, diff_mask, region_bboxes,
                    slide_no, annot_idx, timestamp_sec=ts
                )
                results.append(result)
        else:
            # ── change track 통합 분석: 슬라이드당 1회 호출 (2장 이미지) ──
            track_analysis_results = analyze_slide_change_tracks(
                base_path,
                pair["annot_paths"],
                annot_timestamps,
                pair.get("annot_prev_paths"),
                pair.get("annot_indices"),
                slide_no, cfg, save_masks, mask_dir,
            )
            results.extend(track_analysis_results)

    legacy_results = _to_legacy_annotation_schema(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(legacy_results, f, ensure_ascii=False, indent=2)

    total_annots = sum(len(r.get("annotations", [])) for r in legacy_results)
    log.info(f"\n완료: {len(results)}개 분석, 총 {total_annots}개 annotation 추출 → {output_path}")
    return legacy_results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from .config import DEFAULT_SLIDES_DIR, DEFAULT_OUTPUT_DIR
    parser = argparse.ArgumentParser(description="슬라이드 필기 VLM 분석기")
    parser.add_argument("--slides", "-s", default=str(DEFAULT_SLIDES_DIR),
                        help=f"slide_extractor.py 출력 디렉토리 (default: {DEFAULT_SLIDES_DIR})")
    parser.add_argument("--output", "-o",
                        default=str(DEFAULT_OUTPUT_DIR / "annotation_analysis.json"),
                        help="분석 결과 JSON 경로")
    parser.add_argument("--masks", action="store_true", help="diff 마스크 이미지 저장 (디버그용)")
    parser.add_argument(
        "--per-annot-mode",
        dest="per_annot_mode",
        action="store_true",
        help="change-track 통합 분석 대신 annot별 개별 호출 방식으로 실행 (디버깅용)",
    )
    parser.add_argument("--legacy-per-annot", dest="per_annot_mode", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--no-batch", dest="per_annot_mode", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    analyze_all(
        args.slides,
        args.output,
        save_masks=args.masks,
        per_annot_mode=args.per_annot_mode,
    )
