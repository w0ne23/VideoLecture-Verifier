"""
샘플 캐시 타임라인 구간 분류

단계별 슬라이드 추출 파이프라인의 Step 1:

  sampled_frames.avi + sampled_manifest.json
    -> timeline_segments.json
    -> 구간 미리보기 이미지

이 분류기는 의도적으로 거칠게(coarse) 판정, 이후 슬라이드 전용 처리 단계가
어떤 큰 구간을 처리해야 할지만 결정:

  - slide: 대체로 정지된 발표 화면, 작은 판서 포함
  - video: 지속적인 시각적 움직임, 삽입 영상/데모 포함
  - unknown: 모호해서 별도 검토나 신중한 처리가 필요한 구간

사용법:
    python -m pipeline.timeline_region_classifier \
        --cache sample_cache_dir \
        --output region_segments_dir
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Iterator

import cv2
import imagehash
import numpy as np
from PIL import Image

try:
    from .sample_cache import iter_sample_cache, iter_sample_cache_range as _cache_range, load_sample_cache
    from .person_masks import load_person_mask, masked_pair
except ImportError:  # pragma: no cover - allows direct script execution
    from sample_cache import iter_sample_cache, iter_sample_cache_range as _cache_range, load_sample_cache
    from person_masks import load_person_mask, masked_pair


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


SEGMENTS_FILENAME = "timeline_segments.json"


# 환경변수를 int로 파싱, 실패/미설정 시 default (파싱 실패는 경고 로그)
def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        log.warning("invalid integer env %s=%r; using default=%s", name, value, default)
        return default


# 타임라인 구간 분류 임계값 모음
@dataclass
class RegionClassifierConfig:
    window_sec: float = 4.0
    min_segment_sec: float = 12.0

    # Step 1 병렬 motion-metric 설정
    #
    # motion metric 계산이 이 단계에서 가장 비용이 큼: 각 샘플 프레임을
    # sampled_frames.avi에서 디코딩하고, 마스킹하고, 판정용 프레임으로 변환한 뒤
    # 이전 샘플 프레임과 비교
    #
    # 의존성이 바로 앞 샘플 1개뿐이라 청크로 나눈 worker들이 작은 guard 겹침만으로도
    # 안전하게 동작, 기본값은 sampled_frames.avi 임의 접근을 과도하게 만들지 않으면서
    # 대형 CPU 머신에서 긴 강의 영상을 처리하도록 맞춰짐
    motion_workers: int = _env_int("VLVERIFIER_REGION_WORKERS", 8)
    motion_chunk_samples: int = _env_int("VLVERIFIER_REGION_CHUNK_SAMPLES", 2000)
    motion_guard_samples: int = _env_int("VLVERIFIER_REGION_GUARD_SAMPLES", 1)

    # 샘플 단위 대략적 임계값, sampled-cache의 prev_* 지표 기준
    active_mse: float = 90.0
    very_active_mse: float = 220.0
    active_hash: int = 4
    cut_mse: float = 500.0
    cut_hash: int = 10

    # sampled_frames.avi 기준 픽셀 단위 움직임 임계값
    # 삽입 영상은 작은 사각형 영역 안에서만 움직이는 경우가 많아 MSE/pHash만으로는
    # 과소 감지될 수 있음, 시간에 걸쳐 지속되는 낮은 변화 픽셀 비율이 몇 번의 급격한
    # 슬라이드 전환보다 더 강한 신호
    diff_threshold: int = 12
    motion_ratio: float = 0.006
    strong_motion_ratio: float = 0.025

    # 윈도우 분류 임계값, 슬라이드 자료도 급격한 페이지 전환이 있을 수 있어
    # cut처럼 튀는 스파이크는 지속 움직임 비율 계산에서 제외
    slide_motion_ratio: float = 0.18
    slide_median_changed_ratio: float = 0.003
    video_motion_ratio: float = 0.42
    video_strong_motion_ratio: float = 0.18
    video_median_changed_ratio: float = 0.006
    unknown_margin: float = 0.08

    crop_left: float = 0.02
    crop_top: float = 0.02
    crop_right: float = 0.98
    crop_bottom: float = 0.98


# None을 제외한 값들만 float 리스트로 변환
def _safe_values(values: Iterable[float | int | None]) -> list[float]:
    return [float(v) for v in values if v is not None]


# 정렬된 값 목록에서 지정 백분위수 값을 선형 보간으로 계산
def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


# 프레임에서 여백을 제외한 콘텐츠 영역만 크롭
def _content_region(frame: np.ndarray, cfg: RegionClassifierConfig) -> np.ndarray:
    h, w = frame.shape[:2]
    x0 = max(0, min(w - 1, int(w * cfg.crop_left)))
    y0 = max(0, min(h - 1, int(h * cfg.crop_top)))
    x1 = max(x0 + 1, min(w, int(w * cfg.crop_right)))
    y1 = max(y0 + 1, min(h, int(h * cfg.crop_bottom)))
    return frame[y0:y1, x0:x1]


# 판정용 크롭+그레이스케일+블러 프레임 생성
def _decision_frame(frame: np.ndarray, cfg: RegionClassifierConfig) -> np.ndarray:
    cropped = _content_region(frame, cfg)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


# 마스크도 판정용 콘텐츠 영역으로 크롭
def _decision_mask(mask: np.ndarray | None, cfg: RegionClassifierConfig) -> np.ndarray | None:
    if mask is None:
        return None
    return _content_region(mask.astype(np.uint8), cfg).astype(bool)


# 임계값 초과 픽셀 비율 계산
def _changed_ratio(frame_a: np.ndarray, frame_b: np.ndarray, threshold: int) -> float:
    diff = cv2.absdiff(frame_a, frame_b)
    return float(np.sum(diff > threshold) / diff.size)


# 두 프레임 간 평균 제곱 오차(MSE) 계산
def _mse(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    a = frame_a.astype(np.float32)
    b = frame_b.astype(np.float32)
    return float(np.mean((a - b) ** 2))


# 프레임의 perceptual hash를 정수로 계산
def _phash_int(frame: np.ndarray) -> int:
    pil_img = Image.fromarray(frame)
    return int(str(imagehash.phash(pil_img)), 16)


# 두 hash 정수 간 해밍 거리(비트 차이 개수)
def _phash_distance(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


def iter_sample_cache_range(
    cache_dir: str | Path,
    start_pos: int,
    end_pos: int,
) -> Iterator[tuple[int, dict, np.ndarray]]:
    """0-based 비디오 프레임 위치 기준으로 샘플 캐시 프레임 yield

    항상 sampled_frames.avi를 처음부터 읽는 iter_sample_cache()와 의도적으로 분리,
    Step 1 병렬 worker는 여러 worker가 전체 캐시를 다 스캔하지 않도록 진짜 구간
    읽기가 필요함

    Parameters
    ----------
    cache_dir:
        sampled_manifest.json과 sampled_frames.avi가 있는 샘플 캐시 디렉터리
    start_pos:
        sampled_frames.avi/manifest frames 기준 0-based 시작 위치(포함)
    end_pos:
        0-based 종료 위치(미포함)
    """

    yield from _cache_range(cache_dir, start_pos, end_pos)


# 이전 샘플 대비 motion 지표(mse/hash 거리/변화 픽셀 비율) 계산, 첫 샘플이면 None 값들로 채움
def _motion_metric_for_sample(
    cache_dir: str | Path,
    cfg: RegionClassifierConfig,
    frame_info: dict,
    frame: np.ndarray,
    prev_decision: np.ndarray | None,
    prev_mask: np.ndarray | None,
) -> tuple[dict, np.ndarray, np.ndarray | None]:
    decision = _decision_frame(frame, cfg)
    mask = _decision_mask(load_person_mask(cache_dir, frame_info), cfg)

    if prev_decision is None:
        metric = {
            "changed_ratio": None,
            "motion_mse": None,
            "motion_hash_dist": None,
            "pixel_motion": False,
            "strong_pixel_motion": False,
        }
    else:
        masked_prev, masked_curr = masked_pair(prev_decision, prev_mask, decision, mask)
        changed = _changed_ratio(masked_prev, masked_curr, cfg.diff_threshold)
        motion_mse = _mse(masked_prev, masked_curr)
        motion_hash_dist = _phash_distance(_phash_int(masked_prev), _phash_int(masked_curr))
        metric = {
            "changed_ratio": round(changed, 6),
            "motion_mse": round(motion_mse, 6),
            "motion_hash_dist": motion_hash_dist,
            "pixel_motion": changed >= cfg.motion_ratio,
            "strong_pixel_motion": changed >= cfg.strong_motion_ratio,
        }

    return metric, decision, mask


# 전체 샘플을 순차 순회하며 motion 지표 계산 (단일 프로세스 경로)
def _sample_motion_metrics(
    cache_dir: str | Path,
    cfg: RegionClassifierConfig,
    total_samples: int | None = None,
) -> dict[int, dict]:
    metrics: dict[int, dict] = {}
    prev_decision = None
    prev_mask = None
    processed = 0
    for frame_info, frame in iter_sample_cache(cache_dir):
        sample_index = int(frame_info["sample_index"])
        processed += 1

        metric, prev_decision, prev_mask = _motion_metric_for_sample(
            cache_dir,
            cfg,
            frame_info,
            frame,
            prev_decision,
            prev_mask,
        )
        metrics[sample_index] = metric

        if processed == 1 or processed % 1000 == 0 or (total_samples is not None and processed == total_samples):
            timestamp = float(frame_info.get("timestamp_sec", 0.0) or 0.0)
            if total_samples:
                log.info(
                    "region motion metrics: processed=%s/%s %.1f%% ts=%.1fs",
                    processed,
                    total_samples,
                    (processed / total_samples) * 100.0,
                    timestamp,
                )
            else:
                log.info("region motion metrics: processed=%s ts=%.1fs", processed, timestamp)
    return metrics


def _sample_motion_metrics_range(
    cache_dir: str | Path,
    cfg: RegionClassifierConfig,
    read_start_pos: int,
    core_start_pos: int,
    core_end_pos: int,
) -> dict[int, dict]:
    """하나의 core 구간에 대해 motion 지표 계산

    read_start_pos는 core_start_pos보다 이를 수 있음, core_start_pos 이전 프레임은
    prev_decision/prev_mask를 초기화하는 데만 쓰이는 guard 프레임
    반환되는 지표는 [core_start_pos, core_end_pos) 범위의 위치만 포함
    """

    metrics: dict[int, dict] = {}
    prev_decision = None
    prev_mask = None

    for pos, frame_info, frame in iter_sample_cache_range(cache_dir, read_start_pos, core_end_pos):
        sample_index = int(frame_info["sample_index"])

        metric, prev_decision, prev_mask = _motion_metric_for_sample(
            cache_dir,
            cfg,
            frame_info,
            frame,
            prev_decision,
            prev_mask,
        )

        if core_start_pos <= pos < core_end_pos:
            metrics[sample_index] = metric

    return metrics


# 전체 샘플을 chunk_samples 단위로 나눠 (read_start, core_start, core_end) 구간 목록 생성,
# guard_samples만큼 앞쪽 여유를 둠
def _motion_metric_ranges(total_samples: int, chunk_samples: int, guard_samples: int) -> list[tuple[int, int, int]]:
    chunk_samples = max(1, int(chunk_samples))
    guard_samples = max(0, int(guard_samples))

    ranges: list[tuple[int, int, int]] = []
    for core_start in range(0, total_samples, chunk_samples):
        core_end = min(total_samples, core_start + chunk_samples)
        read_start = max(0, core_start - guard_samples)
        ranges.append((read_start, core_start, core_end))
    return ranges


# 샘플 수가 충분히 많으면 프로세스 풀로 motion 지표를 병렬 계산, 적으면 순차 계산으로 폴백
def _sample_motion_metrics_parallel(
    cache_dir: str | Path,
    cfg: RegionClassifierConfig,
    total_samples: int,
) -> dict[int, dict]:
    workers = max(1, int(cfg.motion_workers))
    chunk_samples = max(1, int(cfg.motion_chunk_samples))
    guard_samples = max(0, int(cfg.motion_guard_samples))

    if workers <= 1 or total_samples <= chunk_samples:
        log.info(
            "region motion metrics sequential path: workers=%s total_samples=%s chunk_samples=%s",
            workers,
            total_samples,
            chunk_samples,
        )
        return _sample_motion_metrics(cache_dir, cfg, total_samples=total_samples)

    ranges = _motion_metric_ranges(total_samples, chunk_samples, guard_samples)
    if len(ranges) <= 1:
        return _sample_motion_metrics(cache_dir, cfg, total_samples=total_samples)

    workers = min(workers, len(ranges))
    log.info(
        "region motion metrics parallel start: samples=%s chunk_samples=%s guard_samples=%s workers=%s chunks=%s",
        total_samples,
        chunk_samples,
        guard_samples,
        workers,
        len(ranges),
    )

    merged: dict[int, dict] = {}
    completed = 0

    # pHash, 마스킹, cv2 diff, 이미지 변환이 CPU 부하가 커서 프로세스 단위 병렬화의
    # 이득이 있어 ProcessPool 사용, 각 worker는 디코더 상태를 공유하지 않도록
    # 자체 VideoCapture를 염
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _sample_motion_metrics_range,
                str(cache_dir),
                cfg,
                read_start,
                core_start,
                core_end,
            ): (idx, read_start, core_start, core_end)
            for idx, (read_start, core_start, core_end) in enumerate(ranges, start=1)
        }

        for future in as_completed(future_map):
            idx, read_start, core_start, core_end = future_map[future]
            chunk_metrics = future.result()
            merged.update(chunk_metrics)
            completed += 1
            log.info(
                "region motion chunk done %s/%s | idx=%s read=%s:%s core=%s:%s metrics=%s",
                completed,
                len(ranges),
                idx,
                read_start,
                core_end,
                core_start,
                core_end,
                len(chunk_metrics),
            )

    expected = total_samples
    if len(merged) != expected:
        missing = expected - len(merged)
        log.warning(
            "region motion metrics count mismatch: expected=%s actual=%s missing_delta=%s",
            expected,
            len(merged),
            missing,
        )

    ordered = {sample_index: merged[sample_index] for sample_index in sorted(merged)}
    log.info("region motion metrics parallel done: metrics=%s", len(ordered))
    return ordered


# 윈도우(연속 샘플 묶음) 안의 motion 지표들을 요약 통계(비율/평균/중앙값/p90)로 집계
def _window_metrics(frames: list[dict], motion_metrics: dict[int, dict], cfg: RegionClassifierConfig) -> dict:
    mses = _safe_values(
        motion_metrics.get(int(f["sample_index"]), {}).get("motion_mse")
        for f in frames
    )
    hashes = _safe_values(
        motion_metrics.get(int(f["sample_index"]), {}).get("motion_hash_dist")
        for f in frames
    )
    changed_ratios = _safe_values(
        motion_metrics.get(int(f["sample_index"]), {}).get("changed_ratio")
        for f in frames
    )
    motion_mses = _safe_values(
        motion_metrics.get(int(f["sample_index"]), {}).get("motion_mse")
        for f in frames
    )
    sample_count = len(frames)
    metric_frames = [
        f for f in frames
        if f.get("prev_mse") is not None and f.get("prev_hash_dist") is not None
    ]
    metric_count = max(1, len(metric_frames))

    cut_flags = [
        (float(motion_metrics.get(int(f["sample_index"]), {}).get("motion_mse") or 0.0) >= cfg.cut_mse)
        and (int(motion_metrics.get(int(f["sample_index"]), {}).get("motion_hash_dist") or 0) >= cfg.cut_hash)
        for f in metric_frames
    ]
    active_flags = [
        (
            float(motion_metrics.get(int(f["sample_index"]), {}).get("motion_mse") or 0.0) >= cfg.active_mse
            or int(motion_metrics.get(int(f["sample_index"]), {}).get("motion_hash_dist") or 0) >= cfg.active_hash
        )
        and not cut
        for f, cut in zip(metric_frames, cut_flags)
    ]
    very_active_flags = [
        float(motion_metrics.get(int(f["sample_index"]), {}).get("motion_mse") or 0.0) >= cfg.very_active_mse and not cut
        for f, cut in zip(metric_frames, cut_flags)
    ]
    pixel_motion_flags = [
        bool(motion_metrics.get(int(f["sample_index"]), {}).get("pixel_motion"))
        and not cut
        for f, cut in zip(metric_frames, cut_flags)
    ]
    strong_pixel_motion_flags = [
        bool(motion_metrics.get(int(f["sample_index"]), {}).get("strong_pixel_motion"))
        and not cut
        for f, cut in zip(metric_frames, cut_flags)
    ]

    return {
        "sample_count": sample_count,
        "metric_count": metric_count,
        "active_ratio": round(sum(active_flags) / metric_count, 4),
        "very_active_ratio": round(sum(very_active_flags) / metric_count, 4),
        "pixel_motion_ratio": round(sum(pixel_motion_flags) / metric_count, 4),
        "strong_pixel_motion_ratio": round(sum(strong_pixel_motion_flags) / metric_count, 4),
        "cut_ratio": round(sum(cut_flags) / metric_count, 4),
        "mean_changed_ratio": round(mean(changed_ratios), 6) if changed_ratios else 0.0,
        "median_changed_ratio": round(median(changed_ratios), 6) if changed_ratios else 0.0,
        "p90_changed_ratio": round(_percentile(changed_ratios, 0.90), 6),
        "mean_motion_mse": round(mean(motion_mses), 4) if motion_mses else 0.0,
        "median_motion_mse": round(median(motion_mses), 4) if motion_mses else 0.0,
        "p90_motion_mse": round(_percentile(motion_mses, 0.90), 4),
        "mean_mse": round(mean(mses), 4) if mses else 0.0,
        "median_mse": round(median(mses), 4) if mses else 0.0,
        "p90_mse": round(_percentile(mses, 0.90), 4),
        "mean_hash_dist": round(mean(hashes), 4) if hashes else 0.0,
        "median_hash_dist": round(median(hashes), 4) if hashes else 0.0,
        "p90_hash_dist": round(_percentile(hashes, 0.90), 4),
    }


# 윈도우 요약 지표를 기준으로 slide/video/unknown과 신뢰도, 판단 근거를 결정
def _classify_window(metrics: dict, cfg: RegionClassifierConfig) -> tuple[str, float, str]:
    pixel_motion_ratio = metrics["pixel_motion_ratio"]
    strong_pixel_motion_ratio = metrics["strong_pixel_motion_ratio"]
    median_changed_ratio = metrics["median_changed_ratio"]
    active_ratio = metrics["active_ratio"]

    if (
        pixel_motion_ratio >= cfg.video_motion_ratio
        or strong_pixel_motion_ratio >= cfg.video_strong_motion_ratio
        or median_changed_ratio >= cfg.video_median_changed_ratio
    ):
        strength = max(
            pixel_motion_ratio - cfg.video_motion_ratio,
            strong_pixel_motion_ratio - cfg.video_strong_motion_ratio,
            (median_changed_ratio - cfg.video_median_changed_ratio) / max(0.001, cfg.video_median_changed_ratio),
        )
        return "video", round(max(0.62, min(0.98, 0.72 + strength * 0.35)), 3), "sustained_pixel_motion"

    if (
        pixel_motion_ratio <= cfg.slide_motion_ratio
        and median_changed_ratio <= cfg.slide_median_changed_ratio
    ):
        slack = min(
            cfg.slide_motion_ratio - pixel_motion_ratio,
            (cfg.slide_median_changed_ratio - median_changed_ratio) / max(0.001, cfg.slide_median_changed_ratio),
        )
        return "slide", round(max(0.58, min(0.98, 0.74 + slack * 0.45)), 3), "low_sustained_motion"

    if pixel_motion_ratio >= cfg.slide_motion_ratio + cfg.unknown_margin or active_ratio >= 0.25:
        return "video", 0.60, "moderate_sustained_motion"

    return "unknown", 0.45, "ambiguous_motion"


# manifest 프레임을 window_sec 단위로 나눠 각 윈도우를 분류
def _window_records(manifest: dict, motion_metrics: dict[int, dict], cfg: RegionClassifierConfig) -> list[dict]:
    frames = manifest.get("frames", [])
    sampled_fps = float(manifest.get("cache", {}).get("sampled_fps") or 1.0)
    window_size = max(1, int(round(cfg.window_sec * sampled_fps)))

    windows: list[dict] = []
    for window_index, start in enumerate(range(0, len(frames), window_size), start=1):
        chunk = frames[start:start + window_size]
        if not chunk:
            continue
        metrics = _window_metrics(chunk, motion_metrics, cfg)
        region_type, confidence, reason = _classify_window(metrics, cfg)
        windows.append({
            "window_index": window_index,
            "type": region_type,
            "confidence": confidence,
            "reason": reason,
            "start_sample_index": int(chunk[0]["sample_index"]),
            "end_sample_index": int(chunk[-1]["sample_index"]),
            "start_frame_no": int(chunk[0]["frame_no"]),
            "end_frame_no": int(chunk[-1]["frame_no"]),
            "start_sec": round(float(chunk[0]["timestamp_sec"]), 3),
            "end_sec": round(float(chunk[-1]["timestamp_sec"]), 3),
            "metrics": metrics,
        })
    return windows


# 연속된 같은 type의 윈도우를 하나의 그룹으로 병합
def _merge_windows(windows: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for window in windows:
        if not merged or merged[-1]["type"] != window["type"]:
            merged.append({
                "type": window["type"],
                "reasons": [window["reason"]],
                "windows": [window],
            })
            continue
        merged[-1]["windows"].append(window)
        if window["reason"] not in merged[-1]["reasons"]:
            merged[-1]["reasons"].append(window["reason"])
    return merged


# 그룹(윈도우 목록)을 하나의 segment 레코드로 평탄화, 지표는 평균
def _flatten_segment(segment: dict, index: int) -> dict:
    windows = segment["windows"]
    first = windows[0]
    last = windows[-1]
    confidences = [float(w["confidence"]) for w in windows]
    keys = [
        "active_ratio",
        "very_active_ratio",
        "cut_ratio",
        "pixel_motion_ratio",
        "strong_pixel_motion_ratio",
        "mean_changed_ratio",
        "median_changed_ratio",
        "p90_changed_ratio",
        "mean_motion_mse",
        "median_motion_mse",
        "p90_motion_mse",
        "mean_mse",
        "median_mse",
        "p90_mse",
        "mean_hash_dist",
        "median_hash_dist",
        "p90_hash_dist",
    ]
    metrics = {
        key: round(mean(float(w["metrics"][key]) for w in windows), 4)
        for key in keys
    }
    return {
        "segment_index": index,
        "type": segment["type"],
        "confidence": round(mean(confidences), 3),
        "reasons": segment["reasons"],
        "start_sample_index": first["start_sample_index"],
        "end_sample_index": last["end_sample_index"],
        "start_frame_no": first["start_frame_no"],
        "end_frame_no": last["end_frame_no"],
        "start_sec": first["start_sec"],
        "end_sec": last["end_sec"],
        "duration_sec": round(last["end_sec"] - first["start_sec"], 3),
        "window_count": len(windows),
        "metrics": metrics,
    }


# min_segment_sec보다 짧은 그룹을 양옆 타입이 같을 때 흡수 병합, 더 이상 합칠 게 없을 때까지 반복
def _merge_short_groups(groups: list[dict], cfg: RegionClassifierConfig) -> list[dict]:
    if len(groups) < 3:
        return groups

    merged = groups[:]
    changed = True
    while changed:
        changed = False
        next_groups: list[dict] = []
        i = 0
        while i < len(merged):
            current = merged[i]
            duration = (
                float(current["windows"][-1]["end_sec"])
                - float(current["windows"][0]["start_sec"])
            )
            if (
                i > 0
                and i + 1 < len(merged)
                and duration < cfg.min_segment_sec
                and merged[i - 1]["type"] == merged[i + 1]["type"]
            ):
                previous = next_groups.pop()
                combined = {
                    "type": previous["type"],
                    "reasons": sorted(set(previous["reasons"] + current["reasons"] + merged[i + 1]["reasons"])),
                    "windows": previous["windows"] + current["windows"] + merged[i + 1]["windows"],
                }
                next_groups.append(combined)
                i += 2
                changed = True
                continue
            next_groups.append(current)
            i += 1
        merged = next_groups
    return merged


# 각 segment의 중간 지점 프레임을 미리보기 이미지로 저장
def _save_segment_previews(cache_dir: str | Path, output_dir: Path, segments: list[dict]) -> None:
    midpoint_to_segment = {
        int(round((seg["start_sample_index"] + seg["end_sample_index"]) / 2)): seg
        for seg in segments
    }
    pending = set(midpoint_to_segment.keys())
    if not pending:
        return

    for frame_info, frame in iter_sample_cache(cache_dir):
        sample_index = int(frame_info["sample_index"])
        if sample_index not in pending:
            continue
        seg = midpoint_to_segment[sample_index]
        filename = f"segment_{seg['segment_index']:03d}_{seg['type']}.jpg"
        cv2.imwrite(str(output_dir / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        seg["preview_filename"] = filename
        pending.remove(sample_index)
        if not pending:
            break


# 샘플 캐시 전체에 대해 motion 지표 계산 -> 윈도우 분류 -> 병합 -> 미리보기 저장까지
# 전체 파이프라인 실행, timeline_segments.json 저장
def classify_regions(
    cache_dir: str,
    output_dir: str,
    cfg: RegionClassifierConfig | None = None,
) -> dict:
    import time

    cfg = cfg or RegionClassifierConfig()
    t0 = time.perf_counter()
    manifest = load_sample_cache(cache_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("segment_*.jpg"):
        stale.unlink(missing_ok=True)

    frame_count = len(manifest.get("frames", []))
    log.info(
        "region classification start: samples=%s window_sec=%.1f motion_workers=%s chunk_samples=%s guard_samples=%s",
        frame_count,
        cfg.window_sec,
        cfg.motion_workers,
        cfg.motion_chunk_samples,
        cfg.motion_guard_samples,
    )

    motion_t0 = time.perf_counter()
    motion_metrics = _sample_motion_metrics_parallel(cache_dir, cfg, total_samples=frame_count)
    log.info("region classification motion metrics done: elapsed=%.1fs", time.perf_counter() - motion_t0)

    windows_t0 = time.perf_counter()
    windows = _window_records(manifest, motion_metrics, cfg)
    log.info("region classification windows done: windows=%s elapsed=%.1fs", len(windows), time.perf_counter() - windows_t0)

    merge_t0 = time.perf_counter()
    groups = _merge_short_groups(_merge_windows(windows), cfg)
    segments = [_flatten_segment(seg, idx) for idx, seg in enumerate(groups, start=1)]
    log.info("region classification merge done: groups=%s segments=%s elapsed=%.1fs", len(groups), len(segments), time.perf_counter() - merge_t0)

    preview_t0 = time.perf_counter()
    _save_segment_previews(cache_dir, out_dir, segments)
    log.info("region classification previews done: elapsed=%.1fs", time.perf_counter() - preview_t0)

    summary: dict[str, dict] = {}
    for seg in segments:
        bucket = summary.setdefault(seg["type"], {"count": 0, "duration_sec": 0.0})
        bucket["count"] += 1
        bucket["duration_sec"] = round(bucket["duration_sec"] + float(seg["duration_sec"]), 3)

    result = {
        "schema_version": 1,
        "cache_dir": str(cache_dir),
        "config": asdict(cfg),
        "source": manifest.get("source", {}),
        "cache": manifest.get("cache", {}),
        "summary": summary,
        "segments": segments,
    }

    out_path = out_dir / SEGMENTS_FILENAME
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    log.info("region classification done: segments=%s output=%s elapsed=%.1fs", len(segments), out_path, time.perf_counter() - t0)
    for region_type, info in sorted(summary.items()):
        log.info("  %-8s count=%s duration=%.1fs", region_type, info["count"], info["duration_sec"])
    return result


# CLI 인자 파싱
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="샘플 캐시 타임라인 구간 분류")
    parser.add_argument("--cache", required=True, help="샘플 캐시 디렉터리")
    parser.add_argument("--output", "-o", required=True, help="출력 디렉터리")
    parser.add_argument("--window-sec", type=float, default=RegionClassifierConfig.window_sec)
    parser.add_argument("--min-segment-sec", type=float, default=RegionClassifierConfig.min_segment_sec)
    parser.add_argument("--motion-workers", type=int, default=RegionClassifierConfig.motion_workers)
    parser.add_argument("--motion-chunk-samples", type=int, default=RegionClassifierConfig.motion_chunk_samples)
    parser.add_argument("--motion-guard-samples", type=int, default=RegionClassifierConfig.motion_guard_samples)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


# CLI 진입점, 인자로 RegionClassifierConfig 구성 후 classify_regions 실행
def main() -> None:
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = RegionClassifierConfig(
        window_sec=args.window_sec,
        min_segment_sec=args.min_segment_sec,
        motion_workers=args.motion_workers,
        motion_chunk_samples=args.motion_chunk_samples,
        motion_guard_samples=args.motion_guard_samples,
    )
    classify_regions(args.cache, args.output, cfg)


if __name__ == "__main__":
    main()
