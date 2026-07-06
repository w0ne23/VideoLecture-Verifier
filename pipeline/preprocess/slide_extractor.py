"""
slide_extractor.py
==================
PPT 기반 강의 영상에서 scene + 필기 완료 시점 프레임을 추출합니다.

용어:
  - scene: 영상 타임라인에서 연속적으로 등장하는 하나의 방문 구간
  - slide: 원본 장표 identity (재방문 scene들을 하나로 묶는 논리 단위)

Input : input/lecture.mp4
Output: output_slides/
        ├── scene_001_base.jpg      # scene 최초 등장 프레임 (장면 전환)
        ├── scene_001_annot_01.jpg  # 필기 안정화 캡처
        └── ...

scene idx가 증가하는 경우:
  1. Cut 전환 (직전 프레임 대비 급격한 MSE + phash 변화)
  2. Fade 전환 (sliding window 내 oldest ↔ current 누적 변화)
  3. scene base 구조 비교 (base 대비 changed ratio + edge 보존율)
  4. scene base pHash 보조 비교

Usage:
    python slide_extractor.py --input input/lecture.mp4 --output output_slides/
    python slide_extractor.py --input input/lecture.mp4 --output output_slides/ --debug
    python slide_extractor.py --input input/lecture.mp4 --tune
"""

import cv2
import numpy as np
import imagehash
import os
import platform
import ctypes
import shutil
import subprocess
import tempfile
from PIL import Image
from pathlib import Path
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import json
import logging
import math

try:
    from .person_masks import masked_pair, resize_mask
except ImportError:  # pragma: no cover - allows direct script execution
    from person_masks import masked_pair, resize_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
_FFMPEG_HWACCEL_DEVICE_CACHE: dict[str, bool] = {}

# 외부 라이브러리 노이즈 로그 억제
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.ai.generativelanguage").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)

# ──────────────────────────────────────────────
# 설정값 (튜닝 포인트)
# ──────────────────────────────────────────────
class Config:
    # ── 슬라이드 전환 감지 (Cut) ─────────────────────────────────────
    SLIDE_CHANGE_MSE_THRESHOLD   = 500   # MSE 임계 (직전 프레임 대비)
    SLIDE_CHANGE_HASH_THRESHOLD  = 10    # phash hamming distance (0~64)

    # ── 슬라이드 전환 감지 (Fade - Sliding Window) ───────────────────
    FADE_WINDOW_SEC              = 1.0   # 페이드 감지 윈도우 크기 (초)
    FADE_MSE_THRESHOLD           = 400   # oldest ↔ current MSE 임계
    FADE_HASH_THRESHOLD          = 8     # oldest ↔ current phash 거리 임계

    # ── scene base 구조 비교 ─────────────────────────────────────────
    # 반복 PPT 템플릿에서 pHash만으로 놓치는 장면 전환을 보완한다.
    BASE_HASH_THRESHOLD          = 8     # scene_base_phash ↔ current phash 거리 임계
    SCENE_BASE_MSE_THRESHOLD     = float(os.getenv("GRAPHLEC_SCENE_BASE_MSE_THRESHOLD", "350"))
    SCENE_BASE_CHANGED_RATIO     = float(os.getenv("GRAPHLEC_SCENE_BASE_CHANGED_RATIO", "0.045"))
    SCENE_STRONG_CHANGED_RATIO   = float(os.getenv("GRAPHLEC_SCENE_STRONG_CHANGED_RATIO", "0.10"))
    CONTENT_CROP_LEFT            = float(os.getenv("GRAPHLEC_CONTENT_CROP_LEFT", "0.12"))
    CONTENT_CROP_TOP             = float(os.getenv("GRAPHLEC_CONTENT_CROP_TOP", "0.06"))
    CONTENT_CROP_RIGHT           = float(os.getenv("GRAPHLEC_CONTENT_CROP_RIGHT", "0.94"))
    CONTENT_CROP_BOTTOM          = float(os.getenv("GRAPHLEC_CONTENT_CROP_BOTTOM", "0.90"))
    SAME_SCENE_EDGE_PRESERVE_THRESHOLD = float(
        os.getenv("GRAPHLEC_SAME_SCENE_EDGE_PRESERVE_THRESHOLD", "0.64")
    )
    SAME_SCENE_CHANGED_RATIO_MAX = float(
        os.getenv("GRAPHLEC_SAME_SCENE_CHANGED_RATIO_MAX", "0.32")
    )

    # ── 중복 슬라이드 감지 (후처리) ──────────────────────────────────
    # compute_phash_hires (256비트, hash_size=16) 기준. 최대 거리 = 256.
    # 실행 후 로그의 슬라이드 간 거리 분포를 보고 튜닝:
    #   - 실제 동일 슬라이드 쌍의 dist → 이 값보다 크게
    #   - 실제 다른 슬라이드 쌍의 dist → 이 값보다 작게
    DUPLICATE_HASH_THRESHOLD     = 30    # 초기값, 로그 확인 후 조정 필요
    DUPLICATE_DHASH_THRESHOLD    = int(os.getenv("GRAPHLEC_DUPLICATE_DHASH_THRESHOLD", "34"))
    DUPLICATE_CONTENT_HASH_THRESHOLD = int(os.getenv("GRAPHLEC_DUPLICATE_CONTENT_HASH_THRESHOLD", "18"))
    DUPLICATE_CONTENT_DHASH_THRESHOLD = int(os.getenv("GRAPHLEC_DUPLICATE_CONTENT_DHASH_THRESHOLD", "24"))
    DUPLICATE_CONTENT_CHANGED_RATIO_MAX = float(
        os.getenv("GRAPHLEC_DUPLICATE_CONTENT_CHANGED_RATIO_MAX", "0.10")
    )
    DUPLICATE_CONTENT_EDGE_OVERLAP_MIN = float(
        os.getenv("GRAPHLEC_DUPLICATE_CONTENT_EDGE_OVERLAP_MIN", "0.90")
    )
    DUPLICATE_CONTENT_MSE_MAX = float(os.getenv("GRAPHLEC_DUPLICATE_CONTENT_MSE_MAX", "0.025"))
    DUPLICATE_CONTENT_HIST_MIN = float(os.getenv("GRAPHLEC_DUPLICATE_CONTENT_HIST_MIN", "0.97"))
    DUPLICATE_FULL_HIST_MIN = float(os.getenv("GRAPHLEC_DUPLICATE_FULL_HIST_MIN", "0.95"))
    AGENDA_TEXT_GUARD_ENABLED = os.getenv("GRAPHLEC_AGENDA_TEXT_GUARD_ENABLED", "1") != "0"
    AGENDA_TEXT_MISMATCH_MAX = float(os.getenv("GRAPHLEC_AGENDA_TEXT_MISMATCH_MAX", "0.18"))
    AGENDA_TEXT_XOR_MAX = float(os.getenv("GRAPHLEC_AGENDA_TEXT_XOR_MAX", "0.045"))
    BUILD_CANDIDATE_PREV_EDGE_PRESERVE_MIN = float(
        os.getenv("GRAPHLEC_BUILD_CANDIDATE_PREV_EDGE_PRESERVE_MIN", "0.90")
    )
    BUILD_CANDIDATE_CHANGED_RATIO_MIN = float(
        os.getenv("GRAPHLEC_BUILD_CANDIDATE_CHANGED_RATIO_MIN", "0.08")
    )
    BUILD_CANDIDATE_CHANGED_RATIO_MAX = float(
        os.getenv("GRAPHLEC_BUILD_CANDIDATE_CHANGED_RATIO_MAX", "0.55")
    )
    BUILD_CANDIDATE_CONTENT_MSE_MAX = float(
        os.getenv("GRAPHLEC_BUILD_CANDIDATE_CONTENT_MSE_MAX", "0.022")
    )
    BUILD_CANDIDATE_CONTENT_HIST_MIN = float(
        os.getenv("GRAPHLEC_BUILD_CANDIDATE_CONTENT_HIST_MIN", "0.80")
    )
    BUILD_CANDIDATE_CONTENT_HASH_MAX = int(
        os.getenv("GRAPHLEC_BUILD_CANDIDATE_CONTENT_HASH_MAX", "90")
    )

    # ── 필기 감지 ────────────────────────────────────────────────────
    ANNOT_DIFF_THRESHOLD         = 15    # 픽셀 변화 판정 절댓값 임계
    ANNOT_CUMULATIVE_RATIO       = 0.0005  # base 대비 0.05% 이상 변화 → 필기 시작
    ANNOT_INSTANT_RATIO          = 0.0001  # 직전 프레임 대비 변화 → 펜 움직임 여부

    # ── 안정화 판단 ──────────────────────────────────────────────────
    STABILITY_WINDOW_SEC         = 0.7
    MIN_ANNOT_DURATION_SEC       = 0.2
    SCENE_CAPTURE_DELAY_SEC      = float(os.getenv("GRAPHLEC_SCENE_CAPTURE_DELAY_SEC", "0.8"))
    SCENE_STABLE_MSE_THRESHOLD   = float(os.getenv("GRAPHLEC_SCENE_STABLE_MSE_THRESHOLD", "80"))
    SCENE_STABLE_HASH_THRESHOLD  = int(os.getenv("GRAPHLEC_SCENE_STABLE_HASH_THRESHOLD", "4"))
    SCENE_PENDING_MAX_SEC        = float(os.getenv("GRAPHLEC_SCENE_PENDING_MAX_SEC", "2.0"))

    # ── 처리 성능 ────────────────────────────────────────────────────
    PROCESS_EVERY_N_FRAMES       = 2
    # 전역 판정/annotation 감지는 이 해상도로 수행한다.
    # 후처리 duplicate 판정용 phash_hires보다 더 작은 폭을 사용해도 충분한 경우가 많다.
    DECISION_RESIZE_WIDTH        = int(os.getenv("GRAPHLEC_SLIDE_DECISION_WIDTH", "768"))
    RESIZE_WIDTH                 = 960
    DECODE_BACKEND               = os.getenv("GRAPHLEC_SLIDE_DECODE_BACKEND", "auto")
    FFMPEG_HWACCEL               = os.getenv("GRAPHLEC_FFMPEG_HWACCEL", "cuda")
    # 서버/로컬 공통 정책: 슬라이드 추출은 항상 5분 단위 청크 병렬 처리
    EXTRACT_CHUNK_SEC            = 300.0
    EXTRACT_CHUNK_OVERLAP_SEC    = 3.0
    EXTRACT_WORKERS              = int(os.getenv("GRAPHLEC_SLIDE_EXTRACT_WORKERS", "0"))


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
def compute_mse(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    if frame_a.ndim == 2:
        a = frame_a.astype(np.float32)
    else:
        a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if frame_b.ndim == 2:
        b = frame_b.astype(np.float32)
    else:
        b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean((a - b) ** 2))


def compute_phash(frame: np.ndarray) -> imagehash.ImageHash:
    """실시간 슬라이드 전환 감지용 (64비트, 속도 우선)"""
    if frame.ndim == 2:
        pil_img = Image.fromarray(frame)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img)


def compute_phash_int(frame: np.ndarray) -> int:
    return int(str(compute_phash(frame)), 16)


def phash_distance_int(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


def compute_phash_hires(frame: np.ndarray) -> imagehash.ImageHash:
    """중복 슬라이드 후처리 감지용 (256비트, 정밀도 우선).

    PPT 템플릿처럼 레이아웃이 동일한 슬라이드들은 64비트 phash로는
    콘텐츠 차이를 구분하기 어렵다. hash_size=16 (256비트)으로 세밀한
    콘텐츠 차이를 포착한다. 최대 거리는 256.
    임계값 튜닝 기준: 실제 동일 슬라이드의 거리를 로그로 확인 후 설정.
    """
    if frame.ndim == 2:
        pil_img = Image.fromarray(frame)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img, hash_size=16)


def compute_dhash_hires(frame: np.ndarray) -> imagehash.ImageHash:
    """중복 슬라이드 후처리 보조 해시 (256비트, edge/gradient 변화에 민감)."""
    if frame.ndim == 2:
        pil_img = Image.fromarray(frame)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.dhash(pil_img, hash_size=16)


def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def count_changed_pixels(frame_a: np.ndarray, frame_b: np.ndarray, threshold: int) -> float:
    diff = cv2.absdiff(frame_a, frame_b)
    if diff.ndim == 2:
        max_diff = diff
    else:
        max_diff = np.max(diff, axis=2)
    return np.sum(max_diff > threshold) / max_diff.size


def _edge_mask(frame: np.ndarray) -> np.ndarray:
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1) > 0


def edge_preservation_ratio(reference: np.ndarray, frame: np.ndarray) -> float:
    ref_edges = _edge_mask(reference)
    ref_count = int(ref_edges.sum())
    if ref_count <= 0:
        return 0.0

    frame_edges = _edge_mask(frame)
    kernel = np.ones((5, 5), np.uint8)
    frame_edges_dilated = cv2.dilate(frame_edges.astype(np.uint8), kernel, iterations=1) > 0
    return float(np.logical_and(ref_edges, frame_edges_dilated).sum() / ref_count)


def symmetric_edge_overlap(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    return min(
        edge_preservation_ratio(frame_a, frame_b),
        edge_preservation_ratio(frame_b, frame_a),
    )


def normalized_mse(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    return compute_mse(frame_a, frame_b) / (255.0 * 255.0)


def grayscale_hist_correlation(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    gray_a = frame_a if frame_a.ndim == 2 else cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = frame_b if frame_b.ndim == 2 else cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    hist_a = cv2.calcHist([gray_a], [0], None, [64], [0, 256])
    hist_b = cv2.calcHist([gray_b], [0], None, [64], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def content_region(frame: np.ndarray, cfg: Config) -> np.ndarray:
    """반복 템플릿/하단 footer 영향을 줄이고 실제 장표 본문 중심으로 비교한다."""
    h, w = frame.shape[:2]
    x0 = max(0, min(w - 1, int(w * cfg.CONTENT_CROP_LEFT)))
    y0 = max(0, min(h - 1, int(h * cfg.CONTENT_CROP_TOP)))
    x1 = max(x0 + 1, min(w, int(w * cfg.CONTENT_CROP_RIGHT)))
    y1 = max(y0 + 1, min(h, int(h * cfg.CONTENT_CROP_BOTTOM)))
    return frame[y0:y1, x0:x1]


def scene_content_metrics(reference: np.ndarray, frame: np.ndarray, cfg: Config) -> dict:
    ref = content_region(reference, cfg)
    cur = content_region(frame, cfg)
    return {
        "mse": compute_mse(ref, cur),
        "changed_ratio": count_changed_pixels(ref, cur, cfg.ANNOT_DIFF_THRESHOLD),
        "edge_preserve": edge_preservation_ratio(ref, cur),
        "symmetric_edge": symmetric_edge_overlap(ref, cur),
        "hash_dist": compute_phash(ref) - compute_phash(cur),
    }


def is_same_scene_content(reference: np.ndarray, frame: np.ndarray, cfg: Config) -> bool:
    metrics = scene_content_metrics(reference, frame, cfg)
    if metrics["changed_ratio"] > cfg.SAME_SCENE_CHANGED_RATIO_MAX:
        return False
    return metrics["edge_preserve"] >= cfg.SAME_SCENE_EDGE_PRESERVE_THRESHOLD


def duplicate_frame_features(frame: np.ndarray, cfg: Config, mask: np.ndarray | None = None) -> dict:
    full = resize_frame(frame, cfg.RESIZE_WIDTH)
    content = content_region(full, cfg)
    full_mask = resize_mask(mask, full.shape[:2]) if mask is not None else None
    content_mask = content_region(full_mask.astype(np.uint8), cfg).astype(bool) if full_mask is not None else None
    return {
        "frame": full,
        "content": content,
        "mask": full_mask,
        "content_mask": content_mask,
        "phash": compute_phash_hires(full),
        "dhash": compute_dhash_hires(full),
        "content_phash": compute_phash_hires(content),
        "content_dhash": compute_dhash_hires(content),
    }


def _agenda_white_components(content: np.ndarray, ignore_mask: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    hsv = cv2.cvtColor(content, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 175)
    if ignore_mask is not None:
        white &= ~ignore_mask

    candidate = white.astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    h, w = candidate.shape[:2]
    keep = np.zeros_like(candidate)
    component_count = 0
    for label in range(1, count):
        x, y, comp_w, comp_h, area = stats[label]
        aspect = comp_w / max(comp_h, 1)
        if area < 0.012 * h * w:
            continue
        if area > 0.50 * h * w:
            continue
        if not (0.40 <= aspect <= 2.60):
            continue
        keep[labels == label] = 255
        component_count += 1

    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8), iterations=1)
    return keep.astype(bool), component_count


def agenda_text_guard_metrics(rep_a: dict, rep_b: dict) -> dict:
    """Detect agenda/table-of-contents slides whose circle text changed.

    Person masks intentionally remove lecturer bodies first, then the guard
    compares dark text inside large white agenda/table regions. This catches
    slides that share the same template but have different numbered items.
    """
    content_a = rep_a["content"]
    content_b = rep_b["content"]
    ignore_mask = None
    mask_a = rep_a.get("content_mask")
    mask_b = rep_b.get("content_mask")
    if mask_a is not None or mask_b is not None:
        ignore_mask = np.zeros(content_a.shape[:2], dtype=bool)
        if mask_a is not None:
            ignore_mask |= mask_a.astype(bool)
        if mask_b is not None:
            ignore_mask |= mask_b.astype(bool)
        ignore_mask = cv2.dilate(
            ignore_mask.astype(np.uint8) * 255,
            np.ones((15, 15), np.uint8),
            iterations=1,
        ).astype(bool)

    white_a, components_a = _agenda_white_components(content_a, ignore_mask)
    white_b, components_b = _agenda_white_components(content_b, ignore_mask)
    shared_region = white_a | white_b
    if ignore_mask is not None:
        shared_region &= ~ignore_mask

    shared_area = float(np.mean(shared_region))
    if components_a < 2 or components_b < 2 or shared_area < 0.08:
        return {
            "agenda_like": False,
            "agenda_components_a": int(components_a),
            "agenda_components_b": int(components_b),
            "agenda_shared_area": shared_area,
        }

    def _dark_text_mask(content: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
        text = ((gray < 165) & shared_region).astype(np.uint8) * 255
        text = cv2.morphologyEx(text, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        return text.astype(bool)

    text_a = _dark_text_mask(content_a)
    text_b = _dark_text_mask(content_b)
    text_count = max(int(text_a.sum()), int(text_b.sum()), 1)

    kernel = np.ones((3, 3), np.uint8)
    dilated_a = cv2.dilate(text_a.astype(np.uint8) * 255, kernel, iterations=1).astype(bool)
    dilated_b = cv2.dilate(text_b.astype(np.uint8) * 255, kernel, iterations=1).astype(bool)
    tolerant_overlap = int(((text_a & dilated_b) | (text_b & dilated_a)).sum())
    mismatch_ratio = max(0.0, 1.0 - (tolerant_overlap / text_count))
    xor_ratio = float((text_a ^ text_b).sum() / max(int(shared_region.sum()), 1))

    return {
        "agenda_like": True,
        "agenda_components_a": int(components_a),
        "agenda_components_b": int(components_b),
        "agenda_shared_area": shared_area,
        "agenda_text_pixels_a": int(text_a.sum()),
        "agenda_text_pixels_b": int(text_b.sum()),
        "agenda_text_mismatch": float(mismatch_ratio),
        "agenda_text_xor": xor_ratio,
    }


def duplicate_pair_decision(rep_a: dict, rep_b: dict, cfg: Config) -> tuple[bool, dict]:
    full_a, full_b = masked_pair(rep_a["frame"], rep_a.get("mask"), rep_b["frame"], rep_b.get("mask"))
    content_a, content_b = masked_pair(
        rep_a["content"],
        rep_a.get("content_mask"),
        rep_b["content"],
        rep_b.get("content_mask"),
    )
    phash_a = compute_phash_hires(full_a)
    phash_b = compute_phash_hires(full_b)
    dhash_a = compute_dhash_hires(full_a)
    dhash_b = compute_dhash_hires(full_b)
    content_phash_a = compute_phash_hires(content_a)
    content_phash_b = compute_phash_hires(content_b)
    content_dhash_a = compute_dhash_hires(content_a)
    content_dhash_b = compute_dhash_hires(content_b)

    metrics = {
        "phash": int(phash_a - phash_b),
        "dhash": int(dhash_a - dhash_b),
        "changed": float(count_changed_pixels(full_a, full_b, cfg.ANNOT_DIFF_THRESHOLD)),
        "edge": float(symmetric_edge_overlap(full_a, full_b)),
        "mse": float(normalized_mse(full_a, full_b)),
        "hist": float(grayscale_hist_correlation(full_a, full_b)),
        "content_phash": int(content_phash_a - content_phash_b),
        "content_dhash": int(content_dhash_a - content_dhash_b),
        "content_changed": float(count_changed_pixels(content_a, content_b, cfg.ANNOT_DIFF_THRESHOLD)),
        "content_edge": float(symmetric_edge_overlap(content_a, content_b)),
        "content_mse": float(normalized_mse(content_a, content_b)),
        "content_hist": float(grayscale_hist_correlation(content_a, content_b)),
        "person_masked": bool(rep_a.get("mask") is not None or rep_b.get("mask") is not None),
    }

    strict_phash = max(8, min(int(cfg.DUPLICATE_HASH_THRESHOLD), 18))
    strict_match = (
        metrics["phash"] <= strict_phash
        and metrics["dhash"] <= 24
        and metrics["content_changed"] <= 0.14
        and metrics["content_mse"] <= 0.030
        and metrics["content_edge"] >= 0.72
    )
    near_identical = (
        metrics["phash"] <= cfg.DUPLICATE_HASH_THRESHOLD
        and metrics["dhash"] <= cfg.DUPLICATE_DHASH_THRESHOLD
        and metrics["changed"] <= 0.055
        and metrics["mse"] <= 0.018
        and metrics["edge"] >= 0.86
        and metrics["hist"] >= 0.985
    )
    content_match = (
        metrics["content_phash"] <= cfg.DUPLICATE_CONTENT_HASH_THRESHOLD
        and metrics["content_dhash"] <= cfg.DUPLICATE_CONTENT_DHASH_THRESHOLD
        and metrics["content_changed"] <= cfg.DUPLICATE_CONTENT_CHANGED_RATIO_MAX
        and metrics["content_mse"] <= cfg.DUPLICATE_CONTENT_MSE_MAX
        and metrics["content_edge"] >= cfg.DUPLICATE_CONTENT_EDGE_OVERLAP_MIN
        and metrics["content_hist"] >= cfg.DUPLICATE_CONTENT_HIST_MIN
        and metrics["hist"] >= cfg.DUPLICATE_FULL_HIST_MIN
    )

    if strict_match:
        metrics["reason"] = "strict"
    elif near_identical:
        metrics["reason"] = "near-identical"
    elif content_match:
        metrics["reason"] = "content"
    else:
        metrics["reason"] = ""

    is_duplicate = bool(strict_match or near_identical or content_match)
    if is_duplicate and cfg.AGENDA_TEXT_GUARD_ENABLED:
        agenda_metrics = agenda_text_guard_metrics(rep_a, rep_b)
        metrics.update(agenda_metrics)
        if (
            agenda_metrics.get("agenda_like")
            and (
                agenda_metrics.get("agenda_text_mismatch", 0.0) > cfg.AGENDA_TEXT_MISMATCH_MAX
                or agenda_metrics.get("agenda_text_xor", 0.0) > cfg.AGENDA_TEXT_XOR_MAX
            )
        ):
            metrics["duplicate_veto"] = "agenda_text_changed"
            metrics["reason"] = ""
            is_duplicate = False

    return is_duplicate, metrics


def build_pair_decision(prev_rep: dict, curr_rep: dict, cfg: Config) -> tuple[bool, dict]:
    """Return whether adjacent base frames are plausible same-slide build steps.

    This is intentionally a candidate detector, not an automatic merge rule.
    A build step should preserve most of the previous slide structure while
    adding or revealing a meaningful amount of content.
    """
    prev_full, curr_full = masked_pair(prev_rep["frame"], prev_rep.get("mask"), curr_rep["frame"], curr_rep.get("mask"))
    prev_content, curr_content = masked_pair(
        prev_rep["content"],
        prev_rep.get("content_mask"),
        curr_rep["content"],
        curr_rep.get("content_mask"),
    )
    content_phash_prev = compute_phash_hires(prev_content)
    content_phash_curr = compute_phash_hires(curr_content)
    content_dhash_prev = compute_dhash_hires(prev_content)
    content_dhash_curr = compute_dhash_hires(curr_content)
    phash_prev = compute_phash_hires(prev_full)
    phash_curr = compute_phash_hires(curr_full)
    dhash_prev = compute_dhash_hires(prev_full)
    dhash_curr = compute_dhash_hires(curr_full)
    metrics = {
        "prev_edge_preserve": float(edge_preservation_ratio(prev_content, curr_content)),
        "curr_edge_preserve": float(edge_preservation_ratio(curr_content, prev_content)),
        "content_changed": float(count_changed_pixels(prev_content, curr_content, cfg.ANNOT_DIFF_THRESHOLD)),
        "content_mse": float(normalized_mse(prev_content, curr_content)),
        "content_hist": float(grayscale_hist_correlation(prev_content, curr_content)),
        "content_phash": int(content_phash_prev - content_phash_curr),
        "content_dhash": int(content_dhash_prev - content_dhash_curr),
        "phash": int(phash_prev - phash_curr),
        "dhash": int(dhash_prev - dhash_curr),
        "person_masked": bool(prev_rep.get("mask") is not None or curr_rep.get("mask") is not None),
    }
    additive_change = (
        metrics["prev_edge_preserve"] >= cfg.BUILD_CANDIDATE_PREV_EDGE_PRESERVE_MIN
        and cfg.BUILD_CANDIDATE_CHANGED_RATIO_MIN
        <= metrics["content_changed"]
        <= cfg.BUILD_CANDIDATE_CHANGED_RATIO_MAX
        and metrics["content_mse"] <= cfg.BUILD_CANDIDATE_CONTENT_MSE_MAX
        and metrics["content_hist"] >= cfg.BUILD_CANDIDATE_CONTENT_HIST_MIN
        and metrics["content_phash"] <= cfg.BUILD_CANDIDATE_CONTENT_HASH_MAX
    )
    metrics["reason"] = "additive-build-candidate" if additive_change else ""
    return bool(additive_change), metrics




def to_decision_frame(frame: np.ndarray, width: int) -> np.ndarray:
    small = resize_frame(frame, width)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _video_metadata(input_path: str) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0.0 or width <= 0 or height <= 0:
        raise RuntimeError(f"영상 메타데이터를 읽지 못했습니다: {input_path}")
    return fps, total_frames, width, height


def _ffmpeg_hwaccels() -> set[str]:
    if shutil.which("ffmpeg") is None:
        return set()
    try:
        result = subprocess.run(
            ["ffmpeg", "-hwaccels"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return set()

    accels: set[str] = set()
    for line in (result.stdout or "").splitlines():
        token = line.strip().lower()
        if token and token.isascii() and " " not in token and token != "hardware acceleration methods:":
            accels.add(token)
    return accels


def _cuda_runtime_available() -> bool:
    system = platform.system().lower()
    if system == "darwin":
        return False

    nvidia_markers = (
        "/dev/nvidiactl",
        "/dev/nvidia0",
        "/proc/driver/nvidia/version",
    )
    if any(Path(marker).exists() for marker in nvidia_markers):
        return True

    if shutil.which("nvidia-smi") is not None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and (result.stdout or "").strip():
                return True
        except Exception:
            pass

    try:
        ctypes.CDLL("libcuda.so.1")
        return True
    except OSError:
        return False


def _ffmpeg_hwaccel_device_available(hwaccel: str) -> bool:
    hwaccel = (hwaccel or "").strip().lower()
    if not hwaccel:
        return False

    cached = _FFMPEG_HWACCEL_DEVICE_CACHE.get(hwaccel)
    if cached is not None:
        return cached

    if hwaccel != "cuda":
        _FFMPEG_HWACCEL_DEVICE_CACHE[hwaccel] = True
        return True

    if shutil.which("ffmpeg") is None:
        _FFMPEG_HWACCEL_DEVICE_CACHE[hwaccel] = False
        return False

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-init_hw_device",
                "cuda=graphlec_cuda",
                "-f",
                "lavfi",
                "-i",
                "nullsrc=s=16x16:d=0.01",
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        available = result.returncode == 0
        if not available:
            err_text = (result.stderr or "").strip()
            log.warning(f"ffmpeg cuda 초기화 확인 실패로 CUDA 디코드를 비활성화합니다: {err_text}")
    except Exception as e:
        available = False
        log.warning(f"ffmpeg cuda 초기화 확인 실패로 CUDA 디코드를 비활성화합니다: {e}")

    _FFMPEG_HWACCEL_DEVICE_CACHE[hwaccel] = available
    return available


def _resolve_decode_backend(preferred_backend: str) -> tuple[str, str | None]:
    backend = (preferred_backend or "auto").strip().lower()
    hwaccels = _ffmpeg_hwaccels()
    system = platform.system().lower()
    cuda_usable = (
        "cuda" in hwaccels
        and _cuda_runtime_available()
        and _ffmpeg_hwaccel_device_available("cuda")
    )

    if backend == "ffmpeg-cuda":
        if cuda_usable:
            return "ffmpeg", "cuda"
        return "opencv", None

    if backend == "ffmpeg-videotoolbox":
        if "videotoolbox" in hwaccels:
            return "ffmpeg", "videotoolbox"
        return "opencv", None

    if backend == "auto":
        if cuda_usable:
            return "ffmpeg", "cuda"
        if system == "darwin" and "videotoolbox" in hwaccels:
            return "ffmpeg", "videotoolbox"
        return "opencv", None

    return "opencv", None


def _iter_processed_frames_opencv(input_path: str, cfg: Config, fps: float):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {input_path}")

    try:
        frame_no = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            if frame_no % cfg.PROCESS_EVERY_N_FRAMES != 0:
                continue
            yield frame_no, frame_no / fps, frame
    finally:
        cap.release()


def _iter_processed_frames_opencv_range(
    input_path: str,
    cfg: Config,
    fps: float,
    start_sec: float,
    end_sec: float,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {input_path}")

    start_frame = max(0, int(start_sec * fps))
    end_frame = max(start_frame, int(end_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    try:
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % cfg.PROCESS_EVERY_N_FRAMES != 0:
                continue
            yield frame_idx, frame_idx / fps, frame
    finally:
        cap.release()


def _iter_processed_frames_ffmpeg_hwaccel(
    input_path: str,
    cfg: Config,
    fps: float,
    width: int,
    height: int,
    hwaccel: str,
):
    select_filter = (
        f"select='not(mod(n+1\\,{cfg.PROCESS_EVERY_N_FRAMES}))',format=bgr24"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        hwaccel,
        "-i",
        input_path,
        "-vf",
        select_filter,
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10 ** 8,
    )

    frame_size = width * height * 3
    frame_no = cfg.PROCESS_EVERY_N_FRAMES

    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                raise RuntimeError("ffmpeg rawvideo 출력이 중간에 잘렸습니다.")

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
            yield frame_no, frame_no / fps, frame
            frame_no += cfg.PROCESS_EVERY_N_FRAMES
    finally:
        if proc.stdout:
            proc.stdout.close()

    stderr = b""
    if proc.stderr is not None:
        stderr = proc.stderr.read()
        proc.stderr.close()
    ret = proc.wait()
    if ret != 0:
        err_text = stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg hwaccel 디코드 실패 (hwaccel={hwaccel}, exit={ret}): {err_text}")


def _iter_processed_frames_ffmpeg_hwaccel_range(
    input_path: str,
    cfg: Config,
    fps: float,
    width: int,
    height: int,
    hwaccel: str,
    start_sec: float,
    end_sec: float,
):
    select_filter = f"select='not(mod(n+1\\,{cfg.PROCESS_EVERY_N_FRAMES}))',format=bgr24"
    duration = max(0.0, end_sec - start_sec)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-hwaccel",
        hwaccel,
        "-i",
        input_path,
        "-t",
        f"{duration:.3f}",
        "-vf",
        select_filter,
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10 ** 8,
    )

    frame_size = width * height * 3
    start_frame = max(0, int(start_sec * fps))
    absolute_frame_no = start_frame + 1
    while absolute_frame_no % cfg.PROCESS_EVERY_N_FRAMES != 0:
        absolute_frame_no += 1

    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                raise RuntimeError("ffmpeg rawvideo 출력이 중간에 잘렸습니다.")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
            timestamp = absolute_frame_no / fps
            yield absolute_frame_no, timestamp, frame
            absolute_frame_no += cfg.PROCESS_EVERY_N_FRAMES
    finally:
        if proc.stdout:
            proc.stdout.close()

    stderr = b""
    if proc.stderr is not None:
        stderr = proc.stderr.read()
        proc.stderr.close()
    ret = proc.wait()
    if ret != 0:
        err_text = stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg hwaccel 디코드 실패 (hwaccel={hwaccel}, exit={ret}): {err_text}")


def _iter_with_opencv_fallback(ffmpeg_iter, opencv_iter_factory, hwaccel: str):
    last_frame_no = 0
    try:
        for frame_no, timestamp, frame in ffmpeg_iter:
            last_frame_no = max(last_frame_no, int(frame_no))
            yield frame_no, timestamp, frame
        return
    except Exception as e:
        log.warning(f"ffmpeg hwaccel 디코드 실패로 OpenCV 디코드로 폴백합니다 (hwaccel={hwaccel}): {e}")

    for frame_no, timestamp, frame in opencv_iter_factory():
        if int(frame_no) <= last_frame_no:
            continue
        yield frame_no, timestamp, frame


def _frame_iterator(
    input_path: str,
    cfg: Config,
    fps: float,
    width: int,
    height: int,
    decode_backend: str,
):
    resolved_backend, hwaccel = _resolve_decode_backend(decode_backend or cfg.DECODE_BACKEND)

    if resolved_backend == "ffmpeg" and hwaccel:
        log.info(f"프레임 디코드 백엔드: ffmpeg ({hwaccel})")
        return (
            _iter_with_opencv_fallback(
                _iter_processed_frames_ffmpeg_hwaccel(input_path, cfg, fps, width, height, hwaccel),
                lambda: _iter_processed_frames_opencv(input_path, cfg, fps),
                hwaccel,
            ),
            f"ffmpeg-{hwaccel}/opencv-fallback",
        )

    log.info("프레임 디코드 백엔드: opencv")
    return _iter_processed_frames_opencv(input_path, cfg, fps), "opencv"


def _frame_iterator_range(
    input_path: str,
    cfg: Config,
    fps: float,
    width: int,
    height: int,
    decode_backend: str,
    start_sec: float,
    end_sec: float,
):
    resolved_backend, hwaccel = _resolve_decode_backend(decode_backend or cfg.DECODE_BACKEND)

    if resolved_backend == "ffmpeg" and hwaccel:
        log.info(
            f"청크 프레임 디코드 백엔드: ffmpeg ({hwaccel}) [{start_sec:.2f}s ~ {end_sec:.2f}s]"
        )
        return (
            _iter_with_opencv_fallback(
                _iter_processed_frames_ffmpeg_hwaccel_range(
                    input_path, cfg, fps, width, height, hwaccel, start_sec, end_sec
                ),
                lambda: _iter_processed_frames_opencv_range(
                    input_path, cfg, fps, start_sec, end_sec
                ),
                hwaccel,
            ),
            f"ffmpeg-{hwaccel}/opencv-fallback",
        )

    log.info(f"청크 프레임 디코드 백엔드: opencv [{start_sec:.2f}s ~ {end_sec:.2f}s]")
    return _iter_processed_frames_opencv_range(input_path, cfg, fps, start_sec, end_sec), "opencv"


# ──────────────────────────────────────────────
# 슬라이드 전환 감지기 (Cut + Fade + Base 이중 비교)
# ──────────────────────────────────────────────
class SlideChangeDetector:
    """
    네 가지 방식으로 scene 전환을 감지:

      1. Cut:  prev_frame ↔ current MSE + phash 급등
      2. Scene base 구조 비교: scene_base ↔ current의 changed ratio + edge 보존율
         - pHash가 유사하게 나오는 반복 PPT 템플릿에서도 scene 변경을 잡기 위함
      3. Base pHash 보조 비교: scene_base_phash ↔ current 비교
      4. Fade: sliding window의 oldest ↔ current 누적 변화
         - 연속 프레임 간 diff가 작아 Cut을 통과하는 페이드 감지

    슬라이드 전환 시 반드시 reset() 호출.
    """
    def __init__(self, cfg: Config, fps: float):
        self.cfg = cfg
        buf_size = max(2, int(cfg.FADE_WINDOW_SEC * fps / cfg.PROCESS_EVERY_N_FRAMES))
        self.frame_buffer     = deque(maxlen=buf_size)
        self.prev_frame       = None
        self.prev_phash       = None
        self.scene_base_frame = None
        self.scene_base_phash = None

    def reset(self, frame: np.ndarray):
        phash = compute_phash(frame)
        self.prev_frame       = frame.copy()
        self.prev_phash       = phash
        self.scene_base_frame = frame.copy()
        self.scene_base_phash = phash
        self.frame_buffer.clear()
        self.frame_buffer.append((frame.copy(), phash))

    def is_slide_change(self, frame: np.ndarray, curr_phash: imagehash.ImageHash | None = None) -> bool:
        if self.prev_frame is None:
            self.reset(frame)
            return False

        if curr_phash is None:
            curr_phash = compute_phash(frame)
        metrics = None
        base_hash_dist = 0
        same_content = None
        if self.scene_base_frame is not None:
            metrics = scene_content_metrics(self.scene_base_frame, frame, self.cfg)
            base_hash_dist = (
                self.scene_base_phash - curr_phash
                if self.scene_base_phash is not None
                else 0
            )
            same_content = is_same_scene_content(self.scene_base_frame, frame, self.cfg)

        # ── 1. Cut 전환 ────────────────────────────────────────────────
        mse = compute_mse(self.prev_frame, frame)
        if same_content is not True and mse >= self.cfg.SLIDE_CHANGE_MSE_THRESHOLD:
            if (self.prev_phash - curr_phash) >= self.cfg.SLIDE_CHANGE_HASH_THRESHOLD:
                self._update(frame, curr_phash)
                return True

        # ── 2. Scene base 구조 비교 ───────────────────────────────────
        if metrics is not None:
            structure_changed = (
                not same_content
                and metrics["mse"] >= self.cfg.SCENE_BASE_MSE_THRESHOLD
                and metrics["changed_ratio"] >= self.cfg.SCENE_BASE_CHANGED_RATIO
            )
            strongly_changed = (
                not same_content
                and metrics["changed_ratio"] >= self.cfg.SCENE_STRONG_CHANGED_RATIO
                and (
                    metrics["hash_dist"] >= max(4, self.cfg.BASE_HASH_THRESHOLD // 2)
                    or base_hash_dist >= max(4, self.cfg.BASE_HASH_THRESHOLD // 2)
                )
            )
            if structure_changed or strongly_changed:
                self._update(frame, curr_phash)
                return True

        # ── 3. Base pHash 보조 비교 ───────────────────────────────────
        if self.scene_base_phash is not None and same_content is not True:
            if (self.scene_base_phash - curr_phash) >= self.cfg.BASE_HASH_THRESHOLD:
                if mse >= self.cfg.SLIDE_CHANGE_MSE_THRESHOLD * 0.5:
                    self._update(frame, curr_phash)
                    return True

        # ── 4. Fade 전환 ───────────────────────────────────────────────
        if same_content is not True and len(self.frame_buffer) == self.frame_buffer.maxlen:
            oldest_frame, oldest_phash = self.frame_buffer[0]
            if compute_mse(oldest_frame, frame) >= self.cfg.FADE_MSE_THRESHOLD:
                if (oldest_phash - curr_phash) >= self.cfg.FADE_HASH_THRESHOLD:
                    self._update(frame, curr_phash)
                    return True

        self._update(frame, curr_phash)
        return False

    def _update(self, frame: np.ndarray, phash: imagehash.ImageHash | None = None):
        if phash is None:
            phash = compute_phash(frame)
        self.prev_frame = frame.copy()
        self.prev_phash = phash
        self.frame_buffer.append((frame.copy(), phash))


# ──────────────────────────────────────────────
# 필기 안정화 감지기
# ──────────────────────────────────────────────
class AnnotationStabilityDetector:
    """
    scene 내부 필기 안정화만 감지한다.

    상태 머신:
      STABLE    → (ratio ≥ ANNOT_RATIO) → WRITING
      WRITING   → (안정화) → CAPTURE_ANNOT → STABLE (base 갱신)

    화면 자체가 바뀐 후보는 이 감지기에서 새 base로 만들지 않고,
    main loop의 annotation 저장 직전 가드에서 scene 전환으로 승격한다.
    """
    def __init__(self, cfg: Config, fps: float):
        self.cfg = cfg
        self.stability_frames = int(cfg.STABILITY_WINDOW_SEC * fps / cfg.PROCESS_EVERY_N_FRAMES)
        self.min_annot_frames = int(cfg.MIN_ANNOT_DURATION_SEC * fps / cfg.PROCESS_EVERY_N_FRAMES)
        self.reset()

    def reset(self, base_frame: np.ndarray = None):
        self.state            = "STABLE"
        self.base_frame       = base_frame
        self.prev_frame       = base_frame
        self.stable_count     = 0
        self.writing_count    = 0
        self.last_annot_frame = None
        self.last_annot_frame_no = None
        self.last_annot_timestamp = None

    def process(
        self,
        frame: np.ndarray,
        frame_no: int | None = None,
        timestamp: float | None = None,
    ) -> str:
        """반환값: "CAPTURE_ANNOT" | "NONE" """
        if self.base_frame is None or self.prev_frame is None:
            self.prev_frame = frame.copy()
            return "NONE"

        cumulative_ratio = count_changed_pixels(
            self.base_frame, frame, self.cfg.ANNOT_DIFF_THRESHOLD
        )
        instant_ratio = count_changed_pixels(
            self.prev_frame, frame, self.cfg.ANNOT_DIFF_THRESHOLD
        )
        is_active = instant_ratio >= self.cfg.ANNOT_INSTANT_RATIO

        result = "NONE"

        if self.state == "STABLE":
            if cumulative_ratio >= self.cfg.ANNOT_CUMULATIVE_RATIO:
                self.state         = "WRITING"
                self.writing_count = 1
                self.stable_count  = 0
                self.last_annot_frame = frame.copy()
                self.last_annot_frame_no = frame_no
                self.last_annot_timestamp = timestamp

        elif self.state == "WRITING":
            if cumulative_ratio >= self.cfg.ANNOT_CUMULATIVE_RATIO:
                self.writing_count += 1
                if is_active:
                    self.stable_count = 0
                    self.last_annot_frame = frame.copy()
                    self.last_annot_frame_no = frame_no
                    self.last_annot_timestamp = timestamp
                else:
                    self.stable_count += 1

                if self.stable_count >= self.stability_frames:
                    result = "CAPTURE_ANNOT"
                    self.base_frame = frame.copy()
                    self.state         = "STABLE"
                    self.stable_count  = 0
                    self.writing_count = 0
            else:
                self.state         = "STABLE"
                self.stable_count  = 0
                self.writing_count = 0

        self.prev_frame = frame.copy()
        return result

    def get_capture_frame(self, current_frame: np.ndarray) -> np.ndarray:
        return self.last_annot_frame if self.last_annot_frame is not None else current_frame

    def get_capture_frame_no(self, current_frame_no: int) -> int:
        return self.last_annot_frame_no if self.last_annot_frame_no is not None else current_frame_no

    def get_capture_timestamp(self, current_timestamp: float) -> float:
        return self.last_annot_timestamp if self.last_annot_timestamp is not None else current_timestamp


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────
def _run_slide_decision_pass(
    frame_iter,
    cfg: Config,
    fps: float,
    duration: float,
    debug: bool = False,
):
    slide_detector = SlideChangeDetector(cfg, fps)
    annot_detector = AnnotationStabilityDetector(cfg, fps)

    scene_idx        = 0
    annot_idx        = 0
    processed_frames = 0
    first_frame      = True
    metadata         = []
    scene_base_frame = None
    scene_base_phash = None
    pending_scene    = None
    scene_stable_frames = max(
        2,
        int(cfg.SCENE_CAPTURE_DELAY_SEC * fps / cfg.PROCESS_EVERY_N_FRAMES),
    )
    scene_pending_max_frames = max(
        scene_stable_frames,
        int(cfg.SCENE_PENDING_MAX_SEC * fps / cfg.PROCESS_EVERY_N_FRAMES),
    )
    progress_interval = max(1, int((duration * fps / cfg.PROCESS_EVERY_N_FRAMES) / 20)) if duration > 0 else 500

    def register_new_base(frame_no, small, timestamp, reason):
        nonlocal scene_idx, annot_idx, scene_base_frame, scene_base_phash
        scene_idx += 1
        annot_idx  = 0
        scene_base_frame = small.copy()
        scene_base_phash = compute_phash(small)
        fname = f"scene_{scene_idx:03d}_base.jpg"
        metadata.append(_meta(fname, scene_idx, timestamp, "base", annot_index=0, frame_no=frame_no))
        log.info(f"[씬 {scene_idx}] base ({reason}) @ {timestamp:.2f}s")
        slide_detector.reset(small)
        annot_detector.reset(base_frame=small)

    def should_suppress_duplicate_base(candidate_small):
        return (
            scene_base_frame is not None
            and is_same_scene_content(scene_base_frame, candidate_small, cfg)
        )

    def start_pending_scene(frame_no, timestamp, small, reason):
        nonlocal pending_scene
        pending_scene = {
            "start_frame_no": frame_no,
            "start_timestamp": timestamp,
            "frame_no": frame_no,
            "timestamp": timestamp,
            "frame": small.copy(),
            "phash": compute_phash(small),
            "stable_count": 1,
            "observed_count": 1,
            "reason": reason,
        }
        annot_detector.reset(base_frame=None)
        log.info(f"  [전환 후보] base 안정화 대기 ({reason}) @ {timestamp:.2f}s")

    def confirm_pending_scene(reason):
        nonlocal pending_scene
        if pending_scene is None:
            return

        candidate = pending_scene
        pending_scene = None
        candidate_frame = candidate["frame"]

        if should_suppress_duplicate_base(candidate_frame):
            log.info(
                "  [전환 후보 생략] 안정화 후 직전 scene과 같은 본문 화면입니다 "
                f"@ {candidate['timestamp']:.2f}s"
            )
            slide_detector.reset(scene_base_frame)
            annot_detector.reset(base_frame=scene_base_frame)
            return

        register_new_base(
            candidate["frame_no"],
            candidate_frame,
            candidate["start_timestamp"],
            reason,
        )

    def update_pending_scene(frame_no, timestamp, small):
        nonlocal pending_scene
        if pending_scene is None:
            return False

        candidate = pending_scene
        candidate["observed_count"] += 1
        curr_phash = compute_phash(small)
        mse = compute_mse(candidate["frame"], small)
        hash_dist = candidate["phash"] - curr_phash

        if mse <= cfg.SCENE_STABLE_MSE_THRESHOLD and hash_dist <= cfg.SCENE_STABLE_HASH_THRESHOLD:
            candidate["stable_count"] += 1
        else:
            candidate["frame_no"] = frame_no
            candidate["timestamp"] = timestamp
            candidate["frame"] = small.copy()
            candidate["phash"] = curr_phash
            candidate["stable_count"] = 1

        if candidate["stable_count"] >= scene_stable_frames:
            confirm_pending_scene("slide_change_stabilized")
            return True

        if candidate["observed_count"] >= scene_pending_max_frames:
            log.info(
                "  [전환 후보 확정] 최대 대기 시간을 넘어 현재 후보를 base로 사용 "
                f"@ {candidate['timestamp']:.2f}s"
            )
            confirm_pending_scene("slide_change_pending_timeout")
            return True

        return False

    def is_annotation_screen_change(candidate_small):
        if scene_base_frame is None or scene_base_phash is None:
            return False

        return not is_same_scene_content(scene_base_frame, candidate_small, cfg)

    for item in frame_iter:
        frame_no, timestamp, small = item[0], item[1], item[2]
        processed_frames += 1

        if pending_scene is not None:
            update_pending_scene(frame_no, timestamp, small)
            if debug and frame_no % (int(fps) * 10) == 0:
                log.debug(f"  처리 중: {timestamp:.1f}s / {duration:.1f}s")
            elif processed_frames % progress_interval == 0:
                ratio = min(100.0, (timestamp / duration) * 100.0) if duration > 0 else 0.0
                log.info(
                    f"  [전역 판정 진행] sampled_frames={processed_frames}, "
                    f"time={timestamp:.1f}s/{duration:.1f}s ({ratio:.1f}%)"
                )
            continue

        # ── scene 전환 감지 (Cut / Fade / Base 이중 비교) ─────────
        slide_change_detected = first_frame or slide_detector.is_slide_change(small)

        if slide_change_detected:
            first_frame = False
            if scene_idx == 0:
                register_new_base(frame_no, small, timestamp, "first_frame")
            else:
                start_pending_scene(frame_no, timestamp, small, "slide_change")
            continue

        # ── 필기 감지 ────────────────────────────────────────────────
        event = annot_detector.process(small, frame_no=frame_no, timestamp=timestamp)

        if event == "CAPTURE_ANNOT":
            capture_frame_no = annot_detector.get_capture_frame_no(frame_no)
            capture_ts = annot_detector.get_capture_timestamp(timestamp)
            candidate_small = annot_detector.get_capture_frame(small)

            if is_annotation_screen_change(candidate_small):
                start_pending_scene(capture_frame_no, capture_ts, candidate_small, "annotation_screen_change")
                continue

            annot_idx += 1
            fname = f"scene_{scene_idx:03d}_annot_{annot_idx:02d}.jpg"
            metadata.append(
                _meta(fname, scene_idx, capture_ts, "annotation",
                      annot_index=annot_idx, frame_no=capture_frame_no)
            )
            log.info(f"  [필기 완료] {fname} @ {capture_ts:.2f}s")

        if debug and frame_no % (int(fps) * 10) == 0:
            log.debug(f"  처리 중: {timestamp:.1f}s / {duration:.1f}s")
        elif processed_frames % progress_interval == 0:
            ratio = min(100.0, (timestamp / duration) * 100.0) if duration > 0 else 0.0
            log.info(
                f"  [전역 판정 진행] sampled_frames={processed_frames}, "
                f"time={timestamp:.1f}s/{duration:.1f}s ({ratio:.1f}%)"
            )

    if pending_scene is not None:
        log.info("  [전환 후보 flush] 영상 종료로 pending scene을 정리합니다.")
        confirm_pending_scene("slide_change_end_flush")

    log.info(f"  처리 프레임={processed_frames}")
    return metadata


def _extract_slides_core(
    input_path: str,
    output_dir: str,
    debug: bool = False,
    decode_backend: str | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    run_postprocess: bool = True,
):
    fps, total_frames, frame_width, frame_height = _video_metadata(input_path)
    duration = total_frames / fps if total_frames > 0 else 0.0

    log.info(f"영상 로드: {input_path}")
    log.info(f"  FPS={fps:.2f}, 총 프레임={total_frames}, 길이={duration:.1f}초")
    log.info(f"  해상도={frame_width}x{frame_height}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    if start_sec is None or end_sec is None:
        frame_iter, active_backend = _frame_iterator(
            input_path,
            cfg,
            fps,
            frame_width,
            frame_height,
            decode_backend or cfg.DECODE_BACKEND,
        )
    else:
        frame_iter, active_backend = _frame_iterator_range(
            input_path,
            cfg,
            fps,
            frame_width,
            frame_height,
            decode_backend or cfg.DECODE_BACKEND,
            start_sec,
            end_sec,
        )

    decision_width = max(160, min(cfg.DECISION_RESIZE_WIDTH, cfg.RESIZE_WIDTH))
    decision_iter = (
        (frame_no, timestamp, to_decision_frame(frame, decision_width))
        for frame_no, timestamp, frame in frame_iter
    )
    metadata = _run_slide_decision_pass(
        frame_iter=decision_iter,
        cfg=cfg,
        fps=fps,
        duration=duration,
        debug=debug,
    )

    _materialize_metadata_frames(input_path, out_path, metadata)
    metadata = add_slide_time_ranges(metadata, duration)
    metadata = mark_clean_final_frames(metadata)
    metadata = mark_visual_duplicates(metadata, out_path, cfg)
    if run_postprocess:
        metadata = maybe_run_local_vlm_review(metadata, out_path)
    metadata = finalize_scene_slide_metadata(metadata)
    log_scene_slide_summary(metadata)

    if not run_postprocess:
        return metadata

    meta_path = out_path / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    scene_slide_map_path = out_path / "scene_slide_map.json"
    with open(scene_slide_map_path, "w", encoding="utf-8") as f:
        json.dump(build_scene_slide_map(metadata), f, ensure_ascii=False, indent=2)

    canonical_slide_annotations_path = out_path / "canonical_slide_annotations.json"
    with open(canonical_slide_annotations_path, "w", encoding="utf-8") as f:
        json.dump(build_canonical_slide_annotations(metadata), f, ensure_ascii=False, indent=2)
    log.info(f"  디코드 백엔드={active_backend}")
    log.info(f"\n완료: scene {len({m['scene_index'] for m in metadata})}개, 총 {len(metadata)}개 프레임 저장 → {output_dir}")
    return metadata


def _extract_sampled_frames_chunk_worker(
    input_path: str,
    chunk_dir: str,
    decode_backend: str,
    start_sec: float,
    end_sec: float,
):
    cfg = Config()
    fps, total_frames, frame_width, frame_height = _video_metadata(input_path)
    duration = total_frames / fps if total_frames > 0 else 0.0
    chunk_path = Path(chunk_dir)
    chunk_path.mkdir(parents=True, exist_ok=True)

    frame_iter, active_backend = _frame_iterator_range(
        input_path,
        cfg,
        fps,
        frame_width,
        frame_height,
        decode_backend or cfg.DECODE_BACKEND,
        start_sec,
        end_sec,
    )

    sampled_fps = max(1.0, fps / max(1, cfg.PROCESS_EVERY_N_FRAMES))
    decision_width = max(160, min(cfg.DECISION_RESIZE_WIDTH, cfg.RESIZE_WIDTH))
    small_height = int(frame_height * (decision_width / frame_width))
    video_filename = "sampled_frames.avi"
    video_path = chunk_path / video_filename
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        sampled_fps,
        (decision_width, small_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"샘플 프레임 비디오를 열 수 없습니다: {video_path}")

    manifest: list[dict] = []
    processed_frames = 0
    try:
        for frame_no, timestamp, frame in frame_iter:
            processed_frames += 1
            small = resize_frame(frame, decision_width)
            decision_small = cv2.GaussianBlur(
                cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                (3, 3),
                0,
            )
            writer.write(small)
            manifest.append({
                "frame_no": frame_no,
                "timestamp_sec": round(timestamp, 4),
                "phash_int": compute_phash_int(decision_small),
            })
    finally:
        writer.release()

    payload = {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": duration,
        "decode_backend": active_backend,
        "processed_frames": processed_frames,
        "decision_resize_width": decision_width,
        "video_filename": video_filename,
        "frames": manifest,
    }
    manifest_path = chunk_path / "sampled_frames.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(manifest_path)


def _ordered_sampled_frames(manifest_paths: list[Path]):
    seen_frame_nos: set[int] = set()
    for manifest_path in manifest_paths:
        chunk_dir = manifest_path.parent
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        video_filename = payload.get("video_filename")
        if not video_filename:
            raise ValueError(f"video_filename이 없는 sampled manifest입니다: {manifest_path}")

        video_path = chunk_dir / video_filename
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"샘플 프레임 비디오를 열 수 없습니다: {video_path}")

        try:
            for item in payload.get("frames", []):
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise RuntimeError(f"샘플 프레임 비디오를 끝까지 읽지 못했습니다: {video_path}")
                frame_no = int(item["frame_no"])
                if frame_no in seen_frame_nos:
                    continue
                seen_frame_nos.add(frame_no)
                decision_frame = cv2.GaussianBlur(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    (3, 3),
                    0,
                )
                phash_int = item.get("phash_int")
                if phash_int is None:
                    phash_int = compute_phash_int(decision_frame)
                yield (
                    frame_no,
                    float(item["timestamp_sec"]),
                    decision_frame,
                    int(phash_int),
                )
        finally:
            cap.release()


def _group_chunk_metadata(metadata: list[dict], source_dir: Path) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    current_scene_idx = None
    for item in metadata:
        normalized = dict(item)
        normalized["_source_dir"] = str(source_dir)
        scene_idx = item["scene_index"]
        if current_scene_idx is None or scene_idx != current_scene_idx:
            if current:
                groups.append(current)
            current = [normalized]
            current_scene_idx = scene_idx
        else:
            current.append(normalized)
    if current:
        groups.append(current)
    return groups


def _item_source_path(item: dict) -> Path:
    return Path(item["_source_dir"]) / item["filename"]


def _group_representative_path(group: list[dict]) -> Path:
    annotations = [item for item in group if item.get("capture_type") == "annotation"]
    target = annotations[-1] if annotations else group[0]
    return _item_source_path(target)


def _group_base_path(group: list[dict]) -> Path:
    return _item_source_path(group[0])


def _image_hash_for_merge(path: Path, resize_width: int) -> imagehash.ImageHash:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return compute_phash_hires(resize_frame(img, resize_width))


def _groups_match_for_merge(
    prev_group: list[dict],
    curr_group: list[dict],
    cfg: Config,
) -> bool:
    prev_end = max(item["timestamp_sec"] for item in prev_group)
    curr_start = min(item["timestamp_sec"] for item in curr_group)
    if curr_start - prev_end > max(1.0, cfg.EXTRACT_CHUNK_OVERLAP_SEC + 0.5):
        return False

    prev_rep = _image_hash_for_merge(_group_representative_path(prev_group), cfg.RESIZE_WIDTH)
    curr_rep = _image_hash_for_merge(_group_representative_path(curr_group), cfg.RESIZE_WIDTH)
    if (prev_rep - curr_rep) < cfg.DUPLICATE_HASH_THRESHOLD:
        return True

    prev_base = _image_hash_for_merge(_group_base_path(prev_group), cfg.RESIZE_WIDTH)
    curr_base = _image_hash_for_merge(_group_base_path(curr_group), cfg.RESIZE_WIDTH)
    return (prev_base - curr_base) < cfg.DUPLICATE_HASH_THRESHOLD


def _merge_group_frames(prev_group: list[dict], curr_group: list[dict]) -> list[dict]:
    seen = {
        (item.get("capture_type"), round(float(item.get("timestamp_sec", 0.0)), 2))
        for item in prev_group
    }
    merged = list(prev_group)
    for item in curr_group:
        key = (item.get("capture_type"), round(float(item.get("timestamp_sec", 0.0)), 2))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    merged.sort(key=lambda x: (x["timestamp_sec"], 0 if x["capture_type"] == "base" else 1))
    return merged


def _copy_merged_groups(merged_groups: list[list[dict]], out_path: Path) -> list[dict]:
    for stale in out_path.glob("slide_*.jpg"):
        stale.unlink(missing_ok=True)
    for stale in out_path.glob("scene_*.jpg"):
        stale.unlink(missing_ok=True)

    metadata: list[dict] = []
    for new_scene_idx, group in enumerate(merged_groups, start=1):
        annot_idx = 0
        for item in sorted(group, key=lambda x: (x["timestamp_sec"], 0 if x["capture_type"] == "base" else 1)):
            capture_type = item["capture_type"]
            if capture_type == "base":
                fname = f"scene_{new_scene_idx:03d}_base.jpg"
            else:
                annot_idx += 1
                fname = f"scene_{new_scene_idx:03d}_annot_{annot_idx:02d}.jpg"
            shutil.copy2(_item_source_path(item), out_path / fname)
            metadata.append(
                _meta(
                    fname,
                    new_scene_idx,
                    item["timestamp_sec"],
                    capture_type,
                    annot_index=annot_idx if capture_type == "annotation" else 0,
                    frame_no=item.get("frame_no"),
                )
            )
    return metadata


def _chunk_specs(duration: float, cfg: Config, workers: int) -> list[dict]:
    if duration <= cfg.EXTRACT_CHUNK_SEC:
        return []

    chunk_sec = max(30.0, cfg.EXTRACT_CHUNK_SEC)
    overlap = max(0.5, min(cfg.EXTRACT_CHUNK_OVERLAP_SEC, chunk_sec / 4))
    specs: list[dict] = []
    chunk_count = max(1, math.ceil(duration / chunk_sec))
    for idx in range(chunk_count):
        core_start = idx * chunk_sec
        core_end = min(duration, (idx + 1) * chunk_sec)
        start_sec = 0.0 if idx == 0 else max(0.0, core_start - overlap)
        end_sec = duration if idx == chunk_count - 1 else min(duration, core_end + overlap)
        specs.append({
            "chunk_index": idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "core_start_sec": core_start,
            "core_end_sec": core_end,
        })
    return specs


def _extract_chunk_worker(
    input_path: str,
    chunk_dir: str,
    decode_backend: str,
    start_sec: float,
    end_sec: float,
    debug: bool = False,
):
    metadata = _extract_slides_core(
        input_path=input_path,
        output_dir=chunk_dir,
        debug=debug,
        decode_backend=decode_backend,
        start_sec=start_sec,
        end_sec=end_sec,
        run_postprocess=False,
    )
    chunk_meta_path = Path(chunk_dir) / "chunk_metadata.json"
    with open(chunk_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return str(chunk_meta_path)


def _extract_slides_legacy(
    input_path: str,
    output_dir: str,
    debug: bool = False,
    decode_backend: str | None = None,
    extract_workers: int | None = None,
):
    cfg = Config()
    requested_workers = cfg.EXTRACT_WORKERS if extract_workers is None else int(extract_workers)
    fps, total_frames, _, _ = _video_metadata(input_path)
    duration = total_frames / fps if total_frames > 0 else 0.0

    specs = _chunk_specs(duration, cfg, requested_workers)
    if not specs:
        log.info(
            f"슬라이드 추출 단일 청크 실행: duration={duration:.1f}s <= "
            f"chunk_sec={cfg.EXTRACT_CHUNK_SEC:.1f}s"
        )
        return _extract_slides_core(input_path, output_dir, debug=debug, decode_backend=decode_backend)

    if requested_workers <= 0:
        workers = len(specs)
    else:
        # 5분 초과로 청크가 2개 이상 생긴 경우에는 항상 병렬로 처리한다.
        # 서버 환경에서 잘못된 설정값(예: 1)으로 병렬성이 꺼지는 일을 막는다.
        workers = min(max(2, requested_workers), len(specs))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log.info(
        f"슬라이드 추출 청크 병렬 실행: duration={duration:.1f}s, "
        f"chunk_sec={cfg.EXTRACT_CHUNK_SEC:.1f}s, requested_workers={requested_workers}, "
        f"workers={workers}, chunks={len(specs)}"
        + (" (auto)" if requested_workers <= 0 else "")
    )

    with tempfile.TemporaryDirectory(prefix="graphlec_slide_chunks_") as temp_root:
        temp_root_path = Path(temp_root)
        manifest_paths: dict[int, Path] = {}

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {}
            for spec in specs:
                chunk_dir = temp_root_path / f"chunk_{spec['chunk_index']:03d}"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                log.info(
                    f"  [청크 시작 {spec['chunk_index'] + 1}/{len(specs)}] "
                    f"{spec['start_sec']:.1f}s ~ {spec['end_sec']:.1f}s"
                )
                future = executor.submit(
                    _extract_sampled_frames_chunk_worker,
                    input_path,
                    str(chunk_dir),
                    decode_backend or cfg.DECODE_BACKEND,
                    spec["start_sec"],
                    spec["end_sec"],
                )
                future_map[future] = (spec, chunk_dir)

            pending = set(future_map.keys())
            completed = 0
            while pending:
                done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                if not done:
                    log.info(
                        f"  [청크 대기 중] completed={completed}/{len(specs)}, "
                        f"running={len(pending)}"
                    )
                    continue

                for future in done:
                    spec, _ = future_map[future]
                    manifest_path = Path(future.result())
                    manifest_paths[spec["chunk_index"]] = manifest_path
                    completed += 1
                    try:
                        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                        log.info(
                            f"  [청크 완료 {completed}/{len(specs)} | idx={spec['chunk_index'] + 1}] "
                            f"backend={payload.get('decode_backend')} "
                            f"frames={payload.get('processed_frames')} "
                            f"range={spec['start_sec']:.1f}s~{spec['end_sec']:.1f}s"
                        )
                    except Exception:
                        log.info(
                            f"  [청크 완료 {completed}/{len(specs)} | idx={spec['chunk_index'] + 1}] "
                            f"range={spec['start_sec']:.1f}s~{spec['end_sec']:.1f}s"
                        )

        ordered_manifests = [manifest_paths[idx] for idx in sorted(manifest_paths)]
        total_sampled_frames = 0
        for manifest_path in ordered_manifests:
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                total_sampled_frames += int(payload.get("processed_frames", 0) or 0)
            except Exception:
                pass
        log.info(f"전역 판정 시작: ordered_chunks={len(ordered_manifests)}, sampled_frames={total_sampled_frames}")
        frame_iter = _ordered_sampled_frames(ordered_manifests)
        metadata = _run_slide_decision_pass(
            frame_iter=frame_iter,
            cfg=cfg,
            fps=fps,
            duration=duration,
            debug=debug,
        )

        _materialize_metadata_frames(input_path, out_path, metadata)
        metadata = add_slide_time_ranges(metadata, duration)
        metadata = mark_clean_final_frames(metadata)
        metadata = mark_visual_duplicates(metadata, out_path, cfg)
        metadata = maybe_run_local_vlm_review(metadata, out_path)
        metadata = finalize_scene_slide_metadata(metadata)
        log_scene_slide_summary(metadata)

        meta_path = out_path / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        scene_slide_map_path = out_path / "scene_slide_map.json"
        with open(scene_slide_map_path, "w", encoding="utf-8") as f:
            json.dump(build_scene_slide_map(metadata), f, ensure_ascii=False, indent=2)

        canonical_slide_annotations_path = out_path / "canonical_slide_annotations.json"
        with open(canonical_slide_annotations_path, "w", encoding="utf-8") as f:
            json.dump(build_canonical_slide_annotations(metadata), f, ensure_ascii=False, indent=2)

        log.info(f"\n완료: scene {len({m['scene_index'] for m in metadata})}개, 총 {len(metadata)}개 프레임 저장 → {output_dir}")
        return metadata


def _extract_slides_staged(
    input_path: str,
    output_dir: str,
    debug: bool = False,
) -> list[dict]:
    try:
        from .annotation_from_cache import detect_annotations
        from .materialize_from_cache import materialize_frames
        from .sample_cache import create_sample_cache
        from .scene_transition_from_cache import run_cache_probe
        from .scene_transition_probe import ProbeConfig
        from .timeline_region_classifier import classify_regions
    except ImportError:  # pragma: no cover - direct script execution fallback
        from annotation_from_cache import detect_annotations
        from materialize_from_cache import materialize_frames
        from sample_cache import create_sample_cache
        from scene_transition_from_cache import run_cache_probe
        from scene_transition_probe import ProbeConfig
        from timeline_region_classifier import classify_regions

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    work_dir = out_path.parent / f"{out_path.name}_staged"
    cache_dir = work_dir / "sample_cache"
    regions_dir = work_dir / "regions"
    scenes_dir = work_dir / "scenes"
    annotations_dir = work_dir / "annotations"
    review_dir = work_dir / "review_slides"
    for path in (cache_dir, regions_dir, scenes_dir, annotations_dir, review_dir):
        path.mkdir(parents=True, exist_ok=True)

    log.info("슬라이드 추출 staged pipeline 실행")
    log.info("  Step 0: sample cache 생성")
    create_sample_cache(input_path, str(cache_dir))

    log.info("  Step 1: timeline region 분류")
    region_payload = classify_regions(str(cache_dir), str(regions_dir))
    regions_path = regions_dir / "timeline_segments.json"

    log.info("  Step 2: slide region scene/base 추출")
    run_cache_probe(
        str(cache_dir),
        str(scenes_dir),
        ProbeConfig(),
        regions_path=str(regions_path),
    )
    scenes_path = scenes_dir / "scene_transitions.json"

    log.info("  Step 3: scene 내부 annotation frame 추출")
    detect_annotations(str(cache_dir), str(scenes_path), str(annotations_dir))
    annotations_path = annotations_dir / "scene_annotations.json"

    log.info("  Step 4A: LocalVLM review용 임시 frame materialize")
    materialize_frames(
        input_path,
        str(scenes_path),
        str(annotations_path),
        str(review_dir),
        regions_path=str(regions_path),
    )

    review_metadata_path = review_dir / "metadata.json"
    with open(review_metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    cfg = Config()
    metadata = mark_clean_final_frames(metadata)
    metadata = mark_visual_duplicates(metadata, review_dir, cfg)
    add_transition_review_candidates(review_dir, scenes_path, metadata)
    metadata = maybe_run_local_vlm_review(metadata, review_dir)
    metadata = remap_metadata_for_final_materialize(metadata)
    fps, total_frames, _, _ = _video_metadata(input_path)
    duration = total_frames / fps if fps > 0 and total_frames > 0 else 0.0
    metadata = refresh_scene_time_ranges(metadata, duration)
    metadata = mark_clean_final_frames(metadata)
    metadata = finalize_scene_slide_metadata(metadata)

    log.info("  Step 4B: VLM 판정 반영 후 최종 frame materialize")
    _materialize_metadata_frames(input_path, out_path, metadata)
    copy_local_vlm_review_artifacts(review_dir, out_path)
    log_scene_slide_summary(metadata)

    metadata_path = out_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    scene_slide_map_path = out_path / "scene_slide_map.json"
    with open(scene_slide_map_path, "w", encoding="utf-8") as f:
        json.dump(build_scene_slide_map(metadata), f, ensure_ascii=False, indent=2)

    canonical_slide_annotations_path = out_path / "canonical_slide_annotations.json"
    with open(canonical_slide_annotations_path, "w", encoding="utf-8") as f:
        json.dump(build_canonical_slide_annotations(metadata), f, ensure_ascii=False, indent=2)

    video_count = sum(1 for item in metadata if item.get("scene_type") == "video" and item.get("capture_type") == "base")
    scene_count = len({m["scene_index"] for m in metadata if m.get("scene_index") is not None})
    log.info(
        "\n완료: scene %s개(video %s개 포함), 총 %s개 프레임 저장 → %s",
        scene_count,
        video_count,
        len(metadata),
        output_dir,
    )
    if debug:
        log.info("  staged work dir=%s", work_dir)
        log.info("  Step 1 summary=%s", region_payload.get("summary"))
    return metadata


def extract_slides(
    input_path: str,
    output_dir: str,
    debug: bool = False,
    decode_backend: str | None = None,
    extract_workers: int | None = None,
    use_staged: bool = True,
):
    if use_staged:
        if decode_backend:
            log.info("staged slide_extractor에서는 decode_backend=%s 설정을 사용하지 않습니다.", decode_backend)
        if extract_workers is not None:
            log.info("staged slide_extractor에서는 extract_workers=%s 설정을 사용하지 않습니다.", extract_workers)
        return _extract_slides_staged(input_path, output_dir, debug=debug)

    return _extract_slides_legacy(
        input_path=input_path,
        output_dir=output_dir,
        debug=debug,
        decode_backend=decode_backend,
        extract_workers=extract_workers,
    )


def _save(frame: np.ndarray, path: Path):
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _read_frame_by_number(cap: cv2.VideoCapture, frame_no: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no - 1))
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return frame


def _read_frame_by_timestamp_opencv(input_path: str, timestamp_sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return frame
    finally:
        cap.release()


def _read_frame_by_timestamp_ffmpeg(input_path: str, timestamp_sec: float) -> np.ndarray | None:
    if shutil.which("ffmpeg") is None:
        return None

    timestamp_sec = max(0.0, float(timestamp_sec))
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp_sec:.6f}",
                "-i",
                input_path,
                "-frames:v",
                "1",
                "-q:v",
                "2",
                "-y",
                str(tmp_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size <= 0:
            return None
        frame = cv2.imread(str(tmp_path), cv2.IMREAD_COLOR)
        return frame if frame is not None else None
    except Exception:
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


def _materialize_frame_with_fallback(
    input_path: str,
    cap: cv2.VideoCapture,
    frame_no: int,
    timestamp_sec: float | None,
) -> np.ndarray | None:
    frame = _read_frame_by_number(cap, frame_no)
    if frame is not None:
        return frame

    if timestamp_sec is None:
        return None

    # Some VFR or slightly damaged MP4s report a frame count that OpenCV cannot
    # seek to near the tail. In that case, timestamp-based extraction is safer.
    offsets = (0.0, -0.05, 0.05, -0.2, 0.2, -0.5, 0.5, -1.0)
    for offset in offsets:
        ts = max(0.0, timestamp_sec + offset)
        frame = _read_frame_by_timestamp_ffmpeg(input_path, ts)
        if frame is not None:
            if abs(offset) > 0.0:
                log.warning(
                    "frame_no=%s 원본 추출을 timestamp %.3fs 보정값으로 복구했습니다.",
                    frame_no,
                    ts,
                )
            return frame

    for offset in offsets:
        ts = max(0.0, timestamp_sec + offset)
        frame = _read_frame_by_timestamp_opencv(input_path, ts)
        if frame is not None:
            if abs(offset) > 0.0:
                log.warning(
                    "frame_no=%s 원본 추출을 OpenCV timestamp %.3fs 보정값으로 복구했습니다.",
                    frame_no,
                    ts,
                )
            return frame

    return None


def _materialize_metadata_frames(input_path: str, out_path: Path, metadata: list[dict]):
    for stale in out_path.glob("slide_*.jpg"):
        stale.unlink(missing_ok=True)
    for stale in out_path.glob("scene_*.jpg"):
        stale.unlink(missing_ok=True)

    frame_targets: dict[int, list[dict]] = {}
    for item in metadata:
        frame_no = int(item.get("frame_no", 0) or 0)
        if frame_no <= 0:
            raise ValueError(f"frame_no가 없는 metadata 항목입니다: {item}")
        frame_targets.setdefault(frame_no, []).append(item)

    cfg = Config()
    fps, _, frame_width, frame_height = _video_metadata(input_path)

    saved_frame_nos: set[int] = set()
    frame_iter, active_backend = _frame_iterator(
        input_path,
        cfg,
        fps,
        frame_width,
        frame_height,
        cfg.DECODE_BACKEND,
    )
    for frame_no, _, frame in frame_iter:
        targets = frame_targets.get(int(frame_no))
        if not targets:
            continue
        for item in targets:
            _save(frame, out_path / item["filename"])
        saved_frame_nos.add(int(frame_no))
        if len(saved_frame_nos) == len(frame_targets):
            break

    missing_frame_nos = sorted(set(frame_targets) - saved_frame_nos)
    if not missing_frame_nos:
        log.info(f"  원본 프레임 materialize: sequential decode ({active_backend})")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {input_path}")

    try:
        fallback_count = 0
        for frame_no in missing_frame_nos:
            targets = frame_targets[frame_no]
            timestamp_sec = None
            timestamps = [
                float(item["timestamp_sec"])
                for item in targets
                if item.get("timestamp_sec") is not None
            ]
            if timestamps:
                timestamp_sec = min(timestamps)

            frame = _materialize_frame_with_fallback(input_path, cap, frame_no, timestamp_sec)
            if frame is not None:
                fallback_count += 1
                for item in targets:
                    _save(frame, out_path / item["filename"])
                continue

            detail = (
                f"timestamp_sec={timestamp_sec:.3f}"
                if timestamp_sec is not None
                else "timestamp_sec 없음"
            )
            raise RuntimeError(f"frame_no={frame_no} 원본 프레임을 추출하지 못했습니다. ({detail})")

        if fallback_count:
            log.warning("원본 프레임 fallback materialize: %s개 frame_no", fallback_count)
    finally:
        cap.release()


def _meta(
    fname: str,
    scene_idx: int,
    timestamp: float,
    capture_type: str,
    annot_index: int = 0,
    frame_no: int | None = None,
) -> dict:
    return {
        "filename":      fname,
        "scene_number":  scene_idx,
        "scene_index":   scene_idx,
        "timestamp_sec": round(timestamp, 2),
        "annot_index":   int(annot_index),
        "frame_no":      int(frame_no) if frame_no is not None else None,
        "capture_type":  capture_type,  # base | annotation
    }


# ──────────────────────────────────────────────
# 후처리: scene 단위 시간 구간 추가
# ──────────────────────────────────────────────
def add_slide_time_ranges(metadata: list, video_duration: float) -> list:
    """
    각 프레임 레코드에 scene/slide 시간 필드 추가.

    - scene_start_sec : 해당 scene_index의 base 프레임 타임스탬프
    - scene_end_sec   : 다음 scene_index의 시작 시각 (마지막 scene은 영상 길이)
    오디오 침묵 구간과 교차 분석할 때 이 구간을 기준으로 사용한다.
    """
    # scene_index → base 타임스탬프 수집
    scene_starts: dict[int, float] = {}
    for m in metadata:
        idx = m["scene_index"]
        if m["capture_type"] == "base" and idx not in scene_starts:
            scene_starts[idx] = m["timestamp_sec"]

    sorted_indices = sorted(scene_starts.keys())

    # 각 scene의 종료 시각 = 다음 scene 시작 시각
    scene_ends: dict[int, float] = {}
    for i, idx in enumerate(sorted_indices):
        if i + 1 < len(sorted_indices):
            scene_ends[idx] = scene_starts[sorted_indices[i + 1]]
        else:
            scene_ends[idx] = round(video_duration, 2)

    for m in metadata:
        idx = m["scene_index"]
        m["scene_start_sec"] = scene_starts.get(idx)
        m["scene_end_sec"]   = scene_ends.get(idx)

    return metadata


def mark_clean_final_frames(metadata: list[dict]) -> list[dict]:
    """scene별 base 프레임을 clean final로 표시한다."""
    from collections import defaultdict

    by_scene: dict[int, list[dict]] = defaultdict(list)
    for item in metadata:
        by_scene[int(item.get("scene_index", 0) or 0)].append(item)

    for _, items in by_scene.items():
        base = next((x for x in items if x.get("capture_type") == "base"), items[0])
        for item in items:
            item["clean_final_filename"] = base.get("filename")
            item["clean_final_capture_type"] = "base"
            item["is_clean_final"] = item is base

    return metadata


def maybe_run_local_vlm_review(metadata: list[dict], out_path: Path) -> list[dict]:
    """Optionally run LocalVLM review after candidate generation.

    GRAPHLEC_VLM_ENABLED=1 writes llm_review_results.json.
    GRAPHLEC_VLM_APPLY=1 additionally applies confident decisions to metadata.
    """
    try:
        from .local_vlm import (
            apply_vlm_slide_decisions,
            local_vlm_apply_enabled,
            local_vlm_enabled,
            run_local_vlm_review,
        )
    except ImportError:  # pragma: no cover - direct script execution fallback
        from local_vlm import (
            apply_vlm_slide_decisions,
            local_vlm_apply_enabled,
            local_vlm_enabled,
            run_local_vlm_review,
        )

    if not local_vlm_enabled():
        return metadata

    log.info("LocalVLM 후보 판정 실행: %s", out_path / "llm_review_candidates.json")
    review_payload = run_local_vlm_review(out_path)
    log.info(
        "LocalVLM 후보 판정 완료: processed=%s errors=%s apply=%s",
        review_payload.get("processed_count", 0),
        review_payload.get("error_count", 0),
        local_vlm_apply_enabled(),
    )
    if local_vlm_apply_enabled():
        return apply_vlm_slide_decisions(metadata, review_payload)
    return metadata


def _remap_optional_scene_index(value, index_map: dict[int, int]):
    if value is None:
        return None
    try:
        return index_map.get(int(value))
    except (TypeError, ValueError):
        return value


def _remap_scene_index_list(values, index_map: dict[int, int]) -> list[int]:
    remapped: list[int] = []
    for value in values or []:
        try:
            mapped = index_map.get(int(value))
        except (TypeError, ValueError):
            continue
        if mapped is not None and mapped not in remapped:
            remapped.append(mapped)
    return remapped


def _filename_for_final_scene(item: dict, scene_index: int) -> str:
    scene_type = item.get("scene_type", "slide")
    capture_type = item.get("capture_type", "base")
    if scene_type == "video":
        return f"scene_{scene_index:03d}_video.jpg"
    if capture_type == "annotation":
        annot_index = int(item.get("annot_index", item.get("scene_annot_index", 0)) or 0)
        return f"scene_{scene_index:03d}_annot_{annot_index:02d}.jpg"
    return f"scene_{scene_index:03d}_base.jpg"


def refresh_slide_group_relations(metadata: list[dict]) -> list[dict]:
    """Recompute same-slide relation fields after scene IDs are compacted."""
    from collections import defaultdict

    scenes = sorted({int(item["scene_index"]) for item in metadata if item.get("scene_index") is not None})
    parent = {idx: idx for idx in scenes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        if a not in parent or b not in parent:
            return
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for item in metadata:
        idx = int(item["scene_index"])
        for field in ("same_slide_group", "slide_group", "scene_group", "duplicate_of"):
            for other in item.get(field) or []:
                try:
                    union(idx, int(other))
                except (TypeError, ValueError):
                    continue

    groups: dict[int, set[int]] = defaultdict(set)
    for idx in scenes:
        groups[find(idx)].add(idx)

    preferred_representatives: dict[int, set[int]] = defaultdict(set)
    base_presence_ratio: dict[int, float] = {}
    for item in metadata:
        idx = int(item["scene_index"])
        root = find(idx)
        if item.get("capture_type") == "base":
            try:
                base_presence_ratio[idx] = float(item.get("person_presence_ratio", 0.0) or 0.0)
            except (TypeError, ValueError):
                base_presence_ratio[idx] = 0.0
        if item.get("vlm_preferred_representative"):
            preferred_representatives[root].add(idx)

    group_by_scene = {idx: sorted(members) for members in groups.values() for idx in members}
    canonical_by_root: dict[int, int] = {}
    for root, members in groups.items():
        preferred = sorted(preferred_representatives.get(root, set()) & members)
        if preferred:
            canonical_by_root[root] = preferred[0]
        else:
            canonical_by_root[root] = min(
                members,
                key=lambda idx: (base_presence_ratio.get(idx, 0.0), idx),
            )

    visit_order: dict[int, int] = {}
    prev_visit: dict[int, int | None] = {}
    next_visit: dict[int, int | None] = {}
    for members in groups.values():
        ordered = sorted(members)
        for pos, idx in enumerate(ordered, start=1):
            visit_order[idx] = pos
            prev_visit[idx] = ordered[pos - 2] if pos > 1 else None
            next_visit[idx] = ordered[pos] if pos < len(ordered) else None

    for item in metadata:
        idx = int(item["scene_index"])
        members = group_by_scene.get(idx, [idx])
        canonical = canonical_by_root.get(find(idx), members[0])
        others = [x for x in members if x != idx]
        item["duplicate_of"] = others
        item["scene_group"] = members
        item["scene_canonical"] = canonical
        item["scene_group_size"] = len(members)
        item["same_slide_group"] = members
        item["same_slide_canonical"] = canonical
        item["same_slide_group_size"] = len(members)
        item["same_slide_visit_order"] = visit_order.get(idx, 1)
        item["same_slide_is_revisit"] = visit_order.get(idx, 1) > 1
        item["same_slide_previous"] = prev_visit.get(idx)
        item["same_slide_next"] = next_visit.get(idx)
        item["slide_group"] = members
        item["slide_canonical_index"] = canonical
        item["slide_group_size"] = len(members)
        item["slide_visit_order"] = visit_order.get(idx, 1)
        item["slide_is_revisit"] = visit_order.get(idx, 1) > 1
        item["previous_scene_index"] = prev_visit.get(idx)
        item["next_scene_index"] = next_visit.get(idx)

    return metadata


def remap_metadata_for_final_materialize(metadata: list[dict]) -> list[dict]:
    """Compact surviving timeline scenes and rewrite filenames before final materialize.

    LocalVLM review runs against provisional images. After its decisions are
    applied, dropped transition scenes can leave gaps such as scene 1, 3. The
    final artifact should instead contain continuous scene indices and matching
    filenames, so this remaps metadata before reading final frames from the
    original video again.
    """
    if not metadata:
        return metadata

    old_scene_indices = sorted({int(item["scene_index"]) for item in metadata if item.get("scene_index") is not None})
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(old_scene_indices, start=1)}

    scalar_scene_fields = (
        "scene_index",
        "scene_number",
        "slide_index",
        "slide_number",
        "scene_canonical",
        "same_slide_canonical",
        "slide_canonical_index",
        "same_slide_previous",
        "same_slide_next",
        "previous_scene_index",
        "next_scene_index",
    )
    list_scene_fields = (
        "duplicate_of",
        "scene_group",
        "same_slide_group",
        "slide_group",
    )

    remapped: list[dict] = []
    for raw_item in metadata:
        item = dict(raw_item)
        old_idx = int(item["scene_index"])
        new_idx = index_map[old_idx]
        item["pre_vlm_scene_index"] = old_idx
        item["provisional_filename"] = item.get("filename")
        item["scene_index"] = new_idx
        item["scene_number"] = new_idx
        item["slide_index"] = new_idx
        item["slide_number"] = new_idx

        for field in scalar_scene_fields:
            if field in ("scene_index", "scene_number", "slide_index", "slide_number"):
                continue
            if field not in item:
                continue
            mapped = _remap_optional_scene_index(item.get(field), index_map)
            item[field] = mapped

        for field in list_scene_fields:
            if field in item:
                item[field] = _remap_scene_index_list(item.get(field), index_map)

        decisions = item.get("vlm_review_decisions")
        if isinstance(decisions, list):
            for decision in decisions:
                if not isinstance(decision, dict):
                    continue
                for field in ("scene_indices", "middle_scene_indices"):
                    if field in decision:
                        decision[field] = _remap_scene_index_list(decision.get(field), index_map)
                if "representative_scene_index" in decision:
                    decision["representative_scene_index"] = _remap_optional_scene_index(
                        decision.get("representative_scene_index"),
                        index_map,
                    )

        item["filename"] = _filename_for_final_scene(item, new_idx)
        remapped.append(item)

    remapped.sort(key=lambda x: (int(x["scene_index"]), int(x.get("annot_index", 0) or 0), int(x.get("frame_no", 0) or 0)))
    return refresh_slide_group_relations(remapped)


def refresh_scene_time_ranges(metadata: list[dict], video_duration: float) -> list[dict]:
    """Refresh scene/slide time ranges after transition scenes are dropped."""
    from collections import defaultdict

    by_scene: dict[int, list[dict]] = defaultdict(list)
    for item in metadata:
        by_scene[int(item["scene_index"])].append(item)

    scene_starts: dict[int, float] = {}
    scene_ends: dict[int, float] = {}
    for scene_idx, items in by_scene.items():
        base = next((x for x in items if x.get("capture_type") == "base"), items[0])
        if base.get("scene_type") == "video":
            scene_starts[scene_idx] = float(base.get("video_start_sec", base.get("timestamp_sec", 0.0)) or 0.0)
        else:
            scene_starts[scene_idx] = float(base.get("timestamp_sec", 0.0) or 0.0)

    ordered = sorted(scene_starts)
    for pos, scene_idx in enumerate(ordered):
        base = next((x for x in by_scene[scene_idx] if x.get("capture_type") == "base"), by_scene[scene_idx][0])
        if base.get("scene_type") == "video":
            scene_ends[scene_idx] = float(base.get("video_end_sec", scene_starts[scene_idx]) or scene_starts[scene_idx])
        elif pos + 1 < len(ordered):
            scene_ends[scene_idx] = scene_starts[ordered[pos + 1]]
        else:
            scene_ends[scene_idx] = float(video_duration or scene_starts[scene_idx])

    for item in metadata:
        idx = int(item["scene_index"])
        start = round(scene_starts.get(idx, float(item.get("timestamp_sec", 0.0) or 0.0)), 3)
        end = round(scene_ends.get(idx, start), 3)
        item["scene_start_sec"] = start
        item["scene_end_sec"] = end
        item["slide_start_sec"] = start
        item["slide_end_sec"] = end

    return metadata


def copy_local_vlm_review_artifacts(review_dir: Path, out_path: Path) -> None:
    """Keep LocalVLM debug artifacts next to the final slides output."""
    for filename in ("llm_review_candidates.json", "llm_review_results.json"):
        src = review_dir / filename
        dst = out_path / filename
        if src.exists():
            shutil.copy2(src, dst)
        else:
            dst.unlink(missing_ok=True)


def add_transition_review_candidates(out_path: Path, scenes_path: Path, metadata: list[dict]) -> None:
    """Merge Step 2 rapid-transition clusters into LocalVLM review candidates.

    Step 2 keeps all rapid cluster bases now. This converts its source
    scene_index values into the final materialized scene_index values so the
    VLM can judge [previous, middle, next] base images before anything is
    dropped from metadata.
    """
    candidates_path = out_path / "llm_review_candidates.json"
    if not candidates_path.exists() or not scenes_path.exists():
        return

    try:
        with open(candidates_path, "r", encoding="utf-8") as f:
            candidates_payload = json.load(f)
        with open(scenes_path, "r", encoding="utf-8") as f:
            scenes_payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("transition 후보 병합 실패: %s", exc)
        return

    transition_candidates = (
        scenes_payload.get("postprocess", {}).get("review_candidates", [])
    )
    if not transition_candidates:
        return

    source_to_scene: dict[int, int] = {}
    filename_by_scene: dict[int, str] = {}
    for item in metadata:
        if item.get("capture_type") != "base" or item.get("scene_type", "slide") != "slide":
            continue
        scene_index = int(item["scene_index"])
        source_scene_index = int(item.get("source_scene_index") or scene_index)
        source_to_scene[source_scene_index] = scene_index
        filename_by_scene[scene_index] = item["filename"]

    candidates = candidates_payload.setdefault("candidates", [])
    existing_keys = {
        (
            candidate.get("candidate_type"),
            tuple(candidate.get("scene_indices") or []),
            tuple(candidate.get("middle_scene_indices") or []),
        )
        for candidate in candidates
    }

    added = 0
    for candidate in transition_candidates:
        source_cluster = [int(x) for x in candidate.get("cluster_scene_indices", [])]
        if len(source_cluster) < 3:
            continue

        source_middle = [int(x) for x in candidate.get("middle_scene_indices", [])]
        for source_mid in source_middle:
            if source_mid not in source_cluster:
                continue
            pos = source_cluster.index(source_mid)
            if pos <= 0 or pos >= len(source_cluster) - 1:
                continue
            source_triplet = [source_cluster[pos - 1], source_mid, source_cluster[pos + 1]]
            if any(source not in source_to_scene for source in source_triplet):
                continue

            scene_indices = [source_to_scene[source] for source in source_triplet]
            middle_scene_indices = [source_to_scene[source_mid]]
            context_scene_indices = [scene_indices[0], scene_indices[2]]
            filenames = [filename_by_scene.get(idx) for idx in scene_indices]
            if any(not filename for filename in filenames):
                continue

            key = ("transition_noise", tuple(scene_indices), tuple(middle_scene_indices))
            if key in existing_keys:
                continue

            candidates.append(limit_vlm_review_candidate_images({
                "candidate_type": "transition_noise",
                "source": "rapid_transition_cluster_postprocess",
                "proposed_decision": "needs_vlm_transition_check",
                "scene_indices": scene_indices,
                "context_scene_indices": context_scene_indices,
                "middle_scene_indices": middle_scene_indices,
                "filenames": filenames,
                "reason": candidate.get("reason", "transition_cluster"),
                "metrics": {
                    "source_cluster_scene_indices": source_cluster,
                    "source_triplet_scene_indices": source_triplet,
                    "source_context_scene_indices": [source_triplet[0], source_triplet[2]],
                    "source_middle_scene_indices": [source_mid],
                    "cluster_start_sec": candidate.get("cluster_start_sec"),
                    "cluster_end_sec": candidate.get("cluster_end_sec"),
                    "max_adjacent_gap_sec": candidate.get("max_adjacent_gap_sec"),
                },
            }))
            existing_keys.add(key)
            added += 1

    if not added:
        return

    candidates_payload["candidate_count"] = len(candidates)
    candidates_payload["description"] = (
        str(candidates_payload.get("description", ""))
        + " transition_noise는 빠른 base cluster의 중간 scene이 전환 중 캡처인지 검증하는 후보이다."
    ).strip()
    with open(candidates_path, "w", encoding="utf-8") as f:
        json.dump(candidates_payload, f, ensure_ascii=False, indent=2)
    log.info("LocalLLM/VLM transition 후보 병합: %s개 추가", added)


def limit_vlm_review_candidate_images(candidate: dict) -> dict:
    """Keep LocalVLM visual inputs small and type-specific.

    - transition_noise: exactly the local [previous, middle, next] context.
    - same_slide_duplicate: at most 3 images, preserving broad context if a
      future grouped candidate contains more than pairwise inputs.
    - same_slide_build: previous and completed/base candidate only.
    """
    candidate = dict(candidate)
    candidate_type = candidate.get("candidate_type")
    scene_indices = list(candidate.get("scene_indices") or [])
    if not scene_indices:
        return candidate

    if candidate_type == "transition_noise":
        max_images = 3
        middle_indices = list(candidate.get("middle_scene_indices") or [])
        positions: list[int] = []
        for middle in middle_indices[:1]:
            if middle in scene_indices:
                mid_pos = scene_indices.index(middle)
                positions = [
                    max(0, mid_pos - 1),
                    mid_pos,
                    min(len(scene_indices) - 1, mid_pos + 1),
                ]
                break
        if not positions:
            positions = list(range(min(max_images, len(scene_indices))))
    elif candidate_type == "same_slide_build":
        max_images = 2
        positions = [0, len(scene_indices) - 1] if len(scene_indices) > 1 else [0]
    elif candidate_type == "same_slide_duplicate":
        max_images = 3
        if len(scene_indices) <= max_images:
            positions = list(range(len(scene_indices)))
        else:
            positions = sorted({0, len(scene_indices) // 2, len(scene_indices) - 1})
    else:
        return candidate

    positions = sorted(dict.fromkeys(pos for pos in positions if 0 <= pos < len(scene_indices)))
    if len(positions) >= len(scene_indices):
        candidate["vlm_image_policy"] = {
            "max_images": max_images,
            "selected_count": len(scene_indices),
        }
        return candidate

    for field in ("scene_indices", "filenames", "labels"):
        values = candidate.get(field)
        if isinstance(values, list) and len(values) == len(scene_indices):
            candidate[field] = [values[pos] for pos in positions]

    if candidate_type == "transition_noise":
        scenes = candidate.get("scene_indices") or []
        middle = [idx for idx in candidate.get("middle_scene_indices", []) if idx in scenes]
        context = [idx for idx in scenes if idx not in middle]
        candidate["middle_scene_indices"] = middle
        candidate["context_scene_indices"] = context

    candidate["vlm_image_policy"] = {
        "max_images": max_images,
        "selected_count": len(candidate.get("scene_indices") or []),
        "selected_positions": positions,
    }
    return candidate


# ──────────────────────────────────────────────
# 후처리: 같은 slide(재등장) 그룹 표시
# ──────────────────────────────────────────────
def mark_visual_duplicates(metadata: list, out_path: Path, cfg: Config) -> list:
    """
    모든 슬라이드의 대표 프레임(base + clean_final + last_annot)을 풀에 쌓고
    전체 쌍(all-pairs)을 비교하여 같은 슬라이드 그룹을 표시한다.

    프레임 풀 구성:
      - 각 scene_index 별로 base 프레임 + last_annot 프레임(있으면) 수집
      - 레이블: "base{idx}" / "annot{idx}"

    비교:
      - 풀 내 모든 쌍을 full-frame + content-region 복합 지표로 비교
      - 동일 scene_index 간 쌍은 건너뜀
      - strict / near-identical / content match 중 하나를 만족하면 같은 슬라이드로 간주

    여기서 "같은 slide"는 재등장(revisit)을 포함한 같은 원본 장표 계열을 의미한다.
    """
    from collections import defaultdict

    cache_dir = out_path.parent / "sample_cache"

    def _load_metadata_person_mask(item: dict) -> np.ndarray | None:
        filename = item.get("person_mask_filename")
        if not filename:
            return None
        path = cache_dir / str(filename)
        if not path.exists():
            return None
        try:
            return np.load(path, allow_pickle=False).astype(bool)
        except Exception:
            log.warning("  [중복 감지] person mask 로드 실패: %s", path, exc_info=True)
            return None

    def _metadata_presence_ratio(item: dict | None) -> float:
        if not item:
            return 0.0
        try:
            ratio = float(item.get("person_presence_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            ratio = 0.0
        if ratio > 0.0:
            return ratio
        filename = item.get("person_presence_mask_filename")
        if not filename:
            return 0.0
        path = cache_dir / str(filename)
        if not path.exists():
            return 0.0
        try:
            mask = np.load(path, allow_pickle=False).astype(bool)
            return float(np.mean(mask))
        except Exception:
            log.warning("  [중복 감지] person presence mask 로드 실패: %s", path, exc_info=True)
            return 0.0

    # scene_index별 프레임 그룹화
    groups: dict[int, list] = defaultdict(list)
    for m in metadata:
        groups[m["scene_index"]].append(m)

    # 프레임 풀 구성: label → (scene_index, filename, metadata item)
    pool: dict[str, tuple[int, str, dict]] = {}
    base_pool: dict[int, str] = {}
    base_items: dict[int, dict] = {}
    for idx in sorted(groups.keys()):
        frames     = groups[idx]
        base_list  = [f for f in frames if f["capture_type"] == "base"]
        annot_list = [f for f in frames if f["capture_type"] == "annotation"]

        if base_list:
            pool[f"base{idx}"] = (idx, base_list[0]["filename"], base_list[0])
            base_pool[idx] = base_list[0]["filename"]
            base_items[idx] = base_list[0]
        if annot_list:
            pool[f"annot{idx}"] = (idx, annot_list[-1]["filename"], annot_list[-1])

    # full frame은 보조로, content region은 실제 장표 본문 identity 판정에 사용한다.
    representatives: dict[str, dict] = {}
    for label, (_, fname, item) in pool.items():
        img = cv2.imread(str(out_path / fname))
        if img is not None:
            representatives[label] = duplicate_frame_features(img, cfg, mask=_load_metadata_person_mask(item))
        else:
            log.warning(f"  [중복 감지] 이미지 로드 실패: {fname}")

    # 전체 쌍 비교 — scene_index별 같은 슬라이드 관계 수집
    labels = sorted(representatives.keys())
    # duplicate_map[idx] = 이 scene과 같은 슬라이드로 판정된 다른 scene_index 집합
    duplicate_map: dict[int, set[int]] = defaultdict(set)
    auto_confirmed_scene_pairs: set[tuple[int, int]] = set()
    review_candidates_by_key: dict[tuple[str, int, int], dict] = {}

    def _candidate_score(candidate: dict) -> tuple:
        metrics = candidate.get("metrics", {})
        if candidate.get("candidate_type") == "same_slide_duplicate":
            return (
                int(metrics.get("content_phash", 9999)),
                int(metrics.get("phash", 9999)),
                float(metrics.get("content_changed", 1.0)),
            )
        return (
            -float(metrics.get("prev_edge_preserve", 0.0)),
            float(metrics.get("content_changed", 1.0)),
            int(metrics.get("content_phash", 9999)),
        )

    def _add_review_candidate(candidate: dict, allow_auto_confirmed: bool = False):
        scene_a, scene_b = sorted(candidate["scene_indices"])
        if not allow_auto_confirmed and (scene_a, scene_b) in auto_confirmed_scene_pairs:
            return
        key = (candidate["candidate_type"], scene_a, scene_b)
        previous = review_candidates_by_key.get(key)
        if previous is None or _candidate_score(candidate) < _candidate_score(previous):
            review_candidates_by_key[key] = candidate

    def _is_base_label(label: str) -> bool:
        return label.startswith("base")

    def _is_auto_confirmed_duplicate(label_a: str, label_b: str, metrics: dict) -> bool:
        """Reflect strong base-base matches into metadata.

        Annot-derived matches and looser content matches stay in the LocalVLM
        queue. Base-base strict matches are allowed a little more motion/noise
        because lecturer regions may be present before representative selection
        chooses the least-occluded base.
        """
        if not (_is_base_label(label_a) and _is_base_label(label_b)):
            return False
        if metrics.get("reason") not in {"strict", "near-identical"}:
            return False
        return (
            metrics["phash"] <= 22
            and metrics["content_phash"] <= 20
            and metrics["changed"] <= 0.065
            and metrics["content_changed"] <= 0.08
            and metrics["mse"] <= 0.007
            and metrics["content_mse"] <= 0.009
            and metrics["edge"] >= 0.87
            and metrics["content_edge"] >= 0.87
            and metrics["hist"] >= 0.995
            and metrics["content_hist"] >= 0.995
        )

    log.info("\n──────── 슬라이드 간 복합 비교 (같은 슬라이드 판정용) ────────")
    log.info(
        "  thresholds: phash<=%s, content_phash<=%s, content_edge>=%.2f",
        cfg.DUPLICATE_HASH_THRESHOLD,
        cfg.DUPLICATE_CONTENT_HASH_THRESHOLD,
        cfg.DUPLICATE_CONTENT_EDGE_OVERLAP_MIN,
    )
    log.info(
        f"  {'프레임 쌍':<30} {'ph':>4} {'dh':>4} {'cph':>4} "
        f"{'chg':>5} {'cedge':>6} {'hist':>5}  {'판정'}"
    )
    log.info(f"  {'-'*30}  {'-'*4} {'-'*4} {'-'*4} {'-'*5} {'-'*6} {'-'*5}  {'-'*12}")

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb  = labels[i], labels[j]
            idx_a   = pool[la][0]
            idx_b   = pool[lb][0]

            # 같은 슬라이드 내 base↔annot 쌍은 건너뜀
            if idx_a == idx_b:
                continue

            is_dup, metrics = duplicate_pair_decision(
                representatives[la],
                representatives[lb],
                cfg,
            )
            flag = f"★ 같은 슬라이드({metrics['reason']})" if is_dup else ""

            log.info(
                f"  {la:<14} ↔ {lb:<14}  "
                f"{metrics['phash']:>4} {metrics['dhash']:>4} {metrics['content_phash']:>4} "
                f"{metrics['content_changed']:>5.3f} {metrics['content_edge']:>6.3f} "
                f"{metrics['hist']:>5.3f}  {flag}"
            )

            if is_dup and _is_auto_confirmed_duplicate(la, lb, metrics):
                scene_pair = tuple(sorted((idx_a, idx_b)))
                auto_confirmed_scene_pairs.add(scene_pair)
                review_candidates_by_key.pop(("same_slide_duplicate", *scene_pair), None)
                review_candidates_by_key.pop(("same_slide_build", *scene_pair), None)
                duplicate_map[idx_a].add(idx_b)
                duplicate_map[idx_b].add(idx_a)
            elif is_dup:
                _add_review_candidate({
                    "candidate_type": "same_slide_duplicate",
                    "source": "visual_duplicate_postprocess",
                    "proposed_decision": "needs_vlm_same_slide_check",
                    "scene_indices": [idx_a, idx_b],
                    "labels": [la, lb],
                    "filenames": [pool[la][1], pool[lb][1]],
                    "reason": metrics["reason"],
                    "metrics": metrics,
                })

    log.info("──────────────────────────────────────────────────────────────\n")

    base_representatives: dict[int, dict] = {}
    for idx, fname in base_pool.items():
        label = f"base{idx}"
        if label in representatives:
            base_representatives[idx] = representatives[label]

    ordered_base_indices = sorted(base_representatives)
    for pos in range(len(ordered_base_indices) - 1):
        idx_a = ordered_base_indices[pos]
        idx_b = ordered_base_indices[pos + 1]
        if idx_b <= idx_a:
            continue
        is_build, build_metrics = build_pair_decision(
            base_representatives[idx_a],
            base_representatives[idx_b],
            cfg,
        )
        if not is_build:
            continue
        if (idx_a, idx_b) in auto_confirmed_scene_pairs:
            continue
        review_candidates_by_key.pop(("same_slide_duplicate", idx_a, idx_b), None)
        _add_review_candidate({
            "candidate_type": "same_slide_build",
            "source": "adjacent_base_build_postprocess",
            "proposed_decision": "same_slide_build",
            "scene_indices": [idx_a, idx_b],
            "labels": [f"base{idx_a}", f"base{idx_b}"],
            "filenames": [base_pool[idx_a], base_pool[idx_b]],
            "reason": build_metrics["reason"],
            "metrics": build_metrics,
        })

    provisional_parent = {idx: idx for idx in groups.keys()}

    def provisional_find(x: int) -> int:
        while provisional_parent[x] != x:
            provisional_parent[x] = provisional_parent[provisional_parent[x]]
            x = provisional_parent[x]
        return x

    def provisional_union(x: int, y: int) -> None:
        if x not in provisional_parent or y not in provisional_parent:
            return
        px, py = provisional_find(x), provisional_find(y)
        if px != py:
            provisional_parent[max(px, py)] = min(px, py)

    for idx_a, neighbors in duplicate_map.items():
        for idx_b in neighbors:
            provisional_union(idx_a, idx_b)

    provisional_groups: dict[int, set[int]] = defaultdict(set)
    for idx in groups.keys():
        provisional_groups[provisional_find(idx)].add(idx)

    for members in provisional_groups.values():
        ordered = sorted(idx for idx in members if idx in base_representatives and idx in base_pool)
        if len(ordered) < 3:
            continue
        for idx_a, idx_b in zip(ordered, ordered[1:]):
            _, metrics = duplicate_pair_decision(base_representatives[idx_a], base_representatives[idx_b], cfg)
            _add_review_candidate({
                "candidate_type": "same_slide_duplicate",
                "source": "auto_group_boundary_review",
                "proposed_decision": "needs_vlm_same_slide_check",
                "scene_indices": [idx_a, idx_b],
                "labels": [f"base{idx_a}", f"base{idx_b}"],
                "filenames": [base_pool[idx_a], base_pool[idx_b]],
                "reason": metrics.get("reason") or "auto_group_boundary",
                "metrics": metrics,
            }, allow_auto_confirmed=True)

    review_candidates = [
        limit_vlm_review_candidate_images(candidate)
        for candidate in sorted(
            review_candidates_by_key.values(),
            key=lambda item: (item["scene_indices"][0], item["scene_indices"][1], item["candidate_type"]),
        )
    ]
    review_payload = {
        "version": 1,
        "description": (
            "LocalLLM/VLM 검증용 후보. same_slide_duplicate는 규칙 기반 같은 슬라이드 후보이고, "
            "same_slide_build는 연속 base가 같은 강의자료 슬라이드의 build 단계일 가능성이 있는 후보이다."
        ),
        "candidate_count": len(review_candidates),
        "candidates": review_candidates,
    }
    review_path = out_path / "llm_review_candidates.json"
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review_payload, f, ensure_ascii=False, indent=2)
    log.info(
        "LocalLLM/VLM 검증 후보 저장: %s (count=%s)",
        review_path,
        len(review_candidates),
    )

    # ── union-find로 전이적 같은 슬라이드 그룹 확정 ────────────────────── #
    all_indices = list(groups.keys())
    parent = {idx: idx for idx in all_indices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for idx_a, neighbors in duplicate_map.items():
        for idx_b in neighbors:
            union(idx_a, idx_b)

    # 루트별로 그룹 구성
    from collections import defaultdict as _defaultdict
    dup_groups: dict[int, set[int]] = _defaultdict(set)
    for idx in all_indices:
        dup_groups[find(idx)].add(idx)

    # 각 scene의 slide family 산출
    group_of: dict[int, set[int]] = {}
    for members in dup_groups.values():
        for idx in members:
            group_of[idx] = members
    canonical_by_scene: dict[int, int] = {}
    for members in dup_groups.values():
        canonical = min(
            members,
            key=lambda idx: (_metadata_presence_ratio(base_items.get(idx)), idx),
        )
        for idx in members:
            canonical_by_scene[idx] = canonical

    # 같은 slide family 내 scene 방문 순서도 함께 기록한다.
    family_visit_order: dict[int, int] = {}
    family_prev_visit: dict[int, int | None] = {}
    family_next_visit: dict[int, int | None] = {}
    for members in dup_groups.values():
        ordered = sorted(members)
        for pos, idx in enumerate(ordered, start=1):
            family_visit_order[idx] = pos
            family_prev_visit[idx] = ordered[pos - 2] if pos > 1 else None
            family_next_visit[idx] = ordered[pos] if pos < len(ordered) else None

    # duplicate_of는 "같은 슬라이드 계열의 다른 scene_index"를 뜻한다.
    for m in metadata:
        idx = m["scene_index"]
        members = sorted(group_of.get(idx, {idx}))
        others = [x for x in members if x != idx]
        m["duplicate_of"] = others
        m["scene_group"] = members
        canonical = canonical_by_scene.get(idx, members[0])
        m["scene_canonical"] = canonical
        m["scene_group_size"] = len(members)
        m["same_slide_group"] = members
        m["same_slide_canonical"] = canonical
        m["same_slide_group_size"] = len(members)
        m["same_slide_visit_order"] = family_visit_order.get(idx, 1)
        m["same_slide_is_revisit"] = family_visit_order.get(idx, 1) > 1
        m["same_slide_previous"] = family_prev_visit.get(idx)
        m["same_slide_next"] = family_next_visit.get(idx)
        m["slide_group"] = members
        m["slide_canonical_index"] = canonical
        m["slide_group_size"] = len(members)
        m["slide_visit_order"] = family_visit_order.get(idx, 1)
        m["slide_is_revisit"] = family_visit_order.get(idx, 1) > 1
        m["previous_scene_index"] = family_prev_visit.get(idx)
        m["next_scene_index"] = family_next_visit.get(idx)

    return metadata


def finalize_scene_slide_metadata(metadata: list[dict]) -> list[dict]:
    """
    scene 기준 로컬 annot 번호와 slide 기준 누적 annot 번호를 함께 기록한다.

    - annot_index: scene 내부 로컬 번호 (기존 유지)
    - slide_annot_index: 같은 canonical slide 전체에서의 누적 번호
    - scene_annotation_count: 해당 scene의 annot 개수
    - slide_annotation_count_total: 해당 canonical slide 전체 annot 개수
    - scene_annotation_start_index / end_index:
        해당 scene이 canonical slide 누적 annot에서 차지하는 범위
    """
    from collections import defaultdict

    by_scene: dict[int, list[dict]] = defaultdict(list)
    for item in metadata:
        by_scene[int(item.get("scene_index", 0) or 0)].append(item)

    by_logical_slide: dict[int, list[tuple[int, list[dict]]]] = defaultdict(list)
    for scene_idx, items in by_scene.items():
        slide_idx = int(
            items[0].get("slide_canonical_index")
            or items[0].get("same_slide_canonical")
            or scene_idx
        )
        by_logical_slide[slide_idx].append((scene_idx, items))

    scene_ranges: dict[int, tuple[int, int]] = {}
    slide_totals: dict[int, int] = {}

    for slide_idx, scene_groups in by_logical_slide.items():
        scene_groups.sort(key=lambda pair: min(float(x.get("timestamp_sec", 0.0) or 0.0) for x in pair[1]))
        cumulative = 0
        for scene_idx, items in scene_groups:
            annots = sorted(
                [x for x in items if x.get("capture_type") == "annotation"],
                key=lambda x: (
                    int(x.get("annot_index", 0) or 0),
                    float(x.get("timestamp_sec", 0.0) or 0.0),
                ),
            )
            start = cumulative + 1 if annots else 0
            for offset, annot in enumerate(annots, start=1):
                annot["scene_annot_index"] = int(annot.get("annot_index", 0) or 0)
                annot["slide_annot_index"] = cumulative + offset
            cumulative += len(annots)
            end = cumulative if annots else 0
            scene_ranges[scene_idx] = (start, end)
        slide_totals[slide_idx] = cumulative

    slide_number_lookup = _build_slide_number_lookup(metadata)

    for scene_idx, items in by_scene.items():
        slide_idx = int(
            items[0].get("slide_canonical_index")
            or items[0].get("same_slide_canonical")
            or scene_idx
        )
        scene_annots = [x for x in items if x.get("capture_type") == "annotation"]
        start, end = scene_ranges.get(scene_idx, (0, 0))
        for item in items:
            item["slide_number"] = slide_number_lookup.get(slide_idx, slide_idx)
            item["scene_annotation_count"] = len(scene_annots)
            item["slide_annotation_count_total"] = slide_totals.get(slide_idx, 0)
            item["scene_annotation_start_index"] = start
            item["scene_annotation_end_index"] = end
            item["scene_local_annot_index"] = int(item.get("annot_index", 0) or 0)
            if item.get("capture_type") != "annotation":
                item["slide_annot_index"] = 0
                item["scene_annot_index"] = 0

    return metadata


def _build_slide_number_lookup(metadata: list[dict]) -> dict[int, int]:
    from collections import defaultdict

    by_scene: dict[int, list[dict]] = defaultdict(list)
    for item in metadata:
        by_scene[int(item.get("scene_index", 0) or 0)].append(item)

    ordered_pairs: list[tuple[float, int]] = []
    for scene_idx in sorted(by_scene):
        items = by_scene[scene_idx]
        base = next((x for x in items if x.get("capture_type") == "base"), items[0])
        slide_idx = int(base.get("slide_canonical_index") or base.get("same_slide_canonical") or scene_idx)
        ts = float(base.get("scene_start_sec", base.get("slide_start_sec", base.get("timestamp_sec", 0.0))) or 0.0)
        ordered_pairs.append((ts, slide_idx))

    lookup: dict[int, int] = {}
    for _, slide_idx in sorted(ordered_pairs, key=lambda x: x[0]):
        if slide_idx not in lookup:
            lookup[slide_idx] = len(lookup) + 1
    return lookup


def log_scene_slide_summary(metadata: list[dict]):
    scene_slide_map = build_scene_slide_map(metadata)
    mappings = scene_slide_map.get("mappings", [])
    if not mappings:
        return

    log.info("\n──────── slide ↔ scene 타임라인 ────────")
    timeline = [
        f"slide {row['slide_number']:03d}: scene {row['scene_index']:03d}"
        for row in mappings
    ]
    for i in range(0, len(timeline), 4):
        log.info("  " + " / ".join(timeline[i:i + 4]))

    log.info("──────── 상세 매핑 ────────")
    for row in mappings:
        scene_range = (
            f"{row['scene_annotation_start_index']}~{row['scene_annotation_end_index']}"
            if row["scene_annotation_start_index"] and row["scene_annotation_end_index"]
            else "-"
        )
        log.info(
            f"  [{row['scene_start_formatted']} ~ {row['scene_end_formatted']}] "
            f"scene {row['scene_index']:03d} -> slide {row['slide_number']:03d} "
            f"(canonical {row['slide_canonical_index']:03d}, "
            f"visit {row['slide_visit_order']}/{row['slide_group_size']}, "
            f"scene_annots={row['scene_annotation_count']}, "
            f"slide_annots={scene_range}, slide_total={row['slide_annotation_count_total']})"
        )
    log.info("────────────────────────────────────\n")


def _fmt_hms(sec: float) -> str:
    total = max(0, int(round(float(sec or 0.0))))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_scene_slide_map(metadata: list[dict]) -> dict:
    from collections import defaultdict

    by_scene: dict[int, list[dict]] = defaultdict(list)
    for item in metadata:
        by_scene[int(item.get("scene_index", 0) or 0)].append(item)

    slide_number_lookup = _build_slide_number_lookup(metadata)

    mappings: list[dict] = []
    for scene_idx in sorted(by_scene):
        items = by_scene[scene_idx]
        base = next((x for x in items if x.get("capture_type") == "base"), items[0])
        slide_idx = int(base.get("slide_canonical_index") or base.get("same_slide_canonical") or scene_idx)
        slide_number = int(base.get("slide_number", slide_number_lookup.get(slide_idx, slide_idx)) or slide_idx)
        scene_start = float(base.get("scene_start_sec", base.get("slide_start_sec", base.get("timestamp_sec", 0.0))) or 0.0)
        scene_end = float(base.get("scene_end_sec", base.get("slide_end_sec", scene_start)) or scene_start)
        scene_type = base.get("scene_type", "slide")
        mappings.append({
            "scene_index": scene_idx,
            "scene_type": scene_type,
            "slide_number": slide_number,
            "slide_canonical_index": slide_idx,
            "slide_group": list(base.get("slide_group", base.get("same_slide_group", [slide_idx]))),
            "slide_group_size": int(base.get("slide_group_size", base.get("same_slide_group_size", 1)) or 1),
            "slide_visit_order": int(base.get("slide_visit_order", base.get("same_slide_visit_order", 1)) or 1),
            "slide_is_revisit": bool(base.get("slide_is_revisit", base.get("same_slide_is_revisit", False))),
            "previous_scene_index": base.get("previous_scene_index", base.get("same_slide_previous")),
            "next_scene_index": base.get("next_scene_index", base.get("same_slide_next")),
            "scene_start_sec": scene_start,
            "scene_end_sec": scene_end,
            "scene_start_formatted": _fmt_hms(scene_start),
            "scene_end_formatted": _fmt_hms(scene_end),
            "scene_annotation_count": int(base.get("scene_annotation_count", 0) or 0),
            "slide_annotation_count_total": int(base.get("slide_annotation_count_total", 0) or 0),
            "scene_annotation_start_index": int(base.get("scene_annotation_start_index", 0) or 0),
            "scene_annotation_end_index": int(base.get("scene_annotation_end_index", 0) or 0),
            "base_filename": base.get("filename"),
            "clean_final_filename": base.get("clean_final_filename", base.get("filename")),
            "clean_final_capture_type": base.get("clean_final_capture_type", "base"),
        })
        if scene_type == "video":
            mappings[-1]["video_start_sec"] = float(base.get("video_start_sec", scene_start) or scene_start)
            mappings[-1]["video_end_sec"] = float(base.get("video_end_sec", scene_end) or scene_end)

    unique_slides = sorted({row["slide_canonical_index"] for row in mappings})
    return {
        "summary": {
            "total_scenes": len(mappings),
            "total_slides": len(unique_slides),
            "total_video_scenes": sum(1 for row in mappings if row.get("scene_type") == "video"),
        },
        "timeline": [
            {
                "order": i + 1,
                "scene_index": row["scene_index"],
                "scene_type": row.get("scene_type", "slide"),
                "slide_number": row["slide_number"],
                "slide_canonical_index": row["slide_canonical_index"],
            }
            for i, row in enumerate(mappings)
        ],
        "mappings": mappings,
    }


def build_canonical_slide_annotations(metadata: list[dict]) -> dict:
    from collections import defaultdict

    by_scene: dict[int, list[dict]] = defaultdict(list)
    for item in metadata:
        by_scene[int(item.get("scene_index", 0) or 0)].append(item)

    slide_number_lookup = _build_slide_number_lookup(metadata)

    by_logical_slide: dict[int, list[dict]] = defaultdict(list)
    for scene_idx, items in by_scene.items():
        base = next((x for x in items if x.get("capture_type") == "base"), items[0])
        slide_idx = int(base.get("slide_canonical_index") or base.get("same_slide_canonical") or scene_idx)
        by_logical_slide[slide_idx].append(items)

    slides_payload: list[dict] = []
    total_annotations = 0

    for slide_idx in sorted(by_logical_slide):
        scene_groups = sorted(
            by_logical_slide[slide_idx],
            key=lambda items: min(float(x.get("timestamp_sec", 0.0) or 0.0) for x in items),
        )
        slide_number = slide_number_lookup.get(slide_idx, slide_idx)

        visits: list[dict] = []
        all_annotations: list[dict] = []

        for items in scene_groups:
            base = next((x for x in items if x.get("capture_type") == "base"), items[0])
            scene_idx = int(base.get("scene_index", 0) or 0)
            scene_type = base.get("scene_type", "slide")
            annots = sorted(
                [x for x in items if x.get("capture_type") == "annotation"],
                key=lambda x: (
                    int(x.get("annot_index", 0) or 0),
                    float(x.get("timestamp_sec", 0.0) or 0.0),
                ),
            )
            visit_entry = {
                "scene_index": scene_idx,
                "scene_type": scene_type,
                "slide_number": slide_number,
                "visit_order": int(base.get("slide_visit_order", base.get("same_slide_visit_order", 1)) or 1),
                "is_revisit": bool(base.get("slide_is_revisit", base.get("same_slide_is_revisit", False))),
                "scene_start_sec": float(base.get("scene_start_sec", base.get("slide_start_sec", base.get("timestamp_sec", 0.0))) or 0.0),
                "scene_end_sec": float(base.get("scene_end_sec", base.get("slide_end_sec", base.get("timestamp_sec", 0.0))) or 0.0),
                "scene_start_formatted": _fmt_hms(base.get("scene_start_sec", base.get("slide_start_sec", base.get("timestamp_sec", 0.0)))),
                "scene_end_formatted": _fmt_hms(base.get("scene_end_sec", base.get("slide_end_sec", base.get("timestamp_sec", 0.0)))),
                "base_filename": base.get("filename"),
                "clean_final_filename": base.get("clean_final_filename", base.get("filename")),
                "clean_final_capture_type": base.get("clean_final_capture_type", "base"),
                "scene_annotation_count": len(annots),
                "scene_annotation_start_index": int(base.get("scene_annotation_start_index", 0) or 0),
                "scene_annotation_end_index": int(base.get("scene_annotation_end_index", 0) or 0),
                "annotations": [],
            }
            if scene_type == "video":
                visit_entry["video_start_sec"] = float(base.get("video_start_sec", visit_entry["scene_start_sec"]) or visit_entry["scene_start_sec"])
                visit_entry["video_end_sec"] = float(base.get("video_end_sec", visit_entry["scene_end_sec"]) or visit_entry["scene_end_sec"])

            for annot in annots:
                annot_entry = {
                    "filename": annot.get("filename"),
                    "scene_index": scene_idx,
                    "slide_number": slide_number,
                    "slide_canonical_index": slide_idx,
                    "timestamp_sec": float(annot.get("timestamp_sec", 0.0) or 0.0),
                    "timestamp_formatted": _fmt_hms(annot.get("timestamp_sec", 0.0)),
                    "annot_index": int(annot.get("annot_index", 0) or 0),
                    "scene_annot_index": int(annot.get("scene_annot_index", annot.get("annot_index", 0)) or 0),
                    "slide_annot_index": int(annot.get("slide_annot_index", 0) or 0),
                }
                visit_entry["annotations"].append(annot_entry)
                all_annotations.append(annot_entry)
                total_annotations += 1

            visits.append(visit_entry)

        slides_payload.append({
            "slide_number": slide_number,
            "slide_canonical_index": slide_idx,
            "scene_types": sorted({visit.get("scene_type", "slide") for visit in visits}),
            "contains_video": any(visit.get("scene_type") == "video" for visit in visits),
            "scene_indices": [visit["scene_index"] for visit in visits],
            "visit_count": len(visits),
            "total_annotation_count": len(all_annotations),
            "visits": visits,
            "all_annotations": all_annotations,
        })

    return {
        "summary": {
            "total_slides": len(slides_payload),
            "total_annotations": total_annotations,
            "total_video_scenes": sum(
                1
                for slide in slides_payload
                for visit in slide["visits"]
                if visit.get("scene_type") == "video"
            ),
        },
        "slides": slides_payload,
    }


# ──────────────────────────────────────────────
# 임계값 튜닝 도우미 (--tune 모드)
# ──────────────────────────────────────────────
def tune_thresholds(input_path: str, sample_sec: float = 30.0):
    """
    영상 앞부분 sample_sec초를 분석하여 diff 분포를 출력.
    튜닝 기준:
      ANNOT_INSTANT_RATIO:    p50 ~ p95 사이
      ANNOT_CUMULATIVE_RATIO: p95 ~ p99 사이
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(sample_sec * fps)

    instant_ratios    = []
    cumulative_ratios = []
    prev_small        = None
    base_small        = None

    cfg = Config()
    frame_no = 0
    while frame_no < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        if frame_no % cfg.PROCESS_EVERY_N_FRAMES != 0:
            continue
        decision_width = max(160, min(cfg.DECISION_RESIZE_WIDTH, cfg.RESIZE_WIDTH))
        small = resize_frame(frame, decision_width)
        if base_small is None:
            base_small = small.copy()
        if prev_small is not None:
            instant_ratios.append(count_changed_pixels(prev_small, small, cfg.ANNOT_DIFF_THRESHOLD))
            cumulative_ratios.append(count_changed_pixels(base_small, small, cfg.ANNOT_DIFF_THRESHOLD))
        prev_small = small

    cap.release()

    ir = np.array(instant_ratios)
    cr = np.array(cumulative_ratios)

    print("\n===== 임계값 튜닝 가이드 =====")
    print(f"[순간 diff]   p50={np.percentile(ir,50):.5f}  p95={np.percentile(ir,95):.5f}  max={ir.max():.5f}")
    print(f"  → ANNOT_INSTANT_RATIO 권장: p50~p95 (현재: {cfg.ANNOT_INSTANT_RATIO})")
    print()
    print(f"[누적 diff]   p50={np.percentile(cr,50):.5f}  p95={np.percentile(cr,95):.5f}"
          f"  p99={np.percentile(cr,99):.5f}  max={cr.max():.5f}")
    print(f"  → ANNOT_CUMULATIVE_RATIO 권장: p95~p99  (현재: {cfg.ANNOT_CUMULATIVE_RATIO})")
    print("================================\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPT 강의 영상 슬라이드 추출기")
    parser.add_argument("--input",  "-i", default="input/lecture.mp4",    help="입력 영상 경로")
    parser.add_argument("--output", "-o", default="output_slides/", help="출력 디렉토리")
    parser.add_argument("--debug",  action="store_true",             help="디버그 로그 출력")
    parser.add_argument("--tune",   action="store_true",             help="임계값 튜닝 모드")
    parser.add_argument(
        "--decode-backend",
        choices=["opencv", "ffmpeg-cuda", "ffmpeg-videotoolbox", "auto"],
        default=Config.DECODE_BACKEND,
        help="프레임 디코드 백엔드 선택 (default: 환경변수 GRAPHLEC_SLIDE_DECODE_BACKEND 또는 opencv)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="기존 slide_extractor 단일/청크 추출 로직 사용",
    )
    args = parser.parse_args()

    if args.tune:
        tune_thresholds(args.input)
    else:
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        extract_slides(
            args.input,
            args.output,
            debug=args.debug,
            decode_backend=args.decode_backend,
            use_staged=not args.legacy,
        )
