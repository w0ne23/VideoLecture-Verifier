"""
Classify sampled-cache timeline regions.

Step 1 of the staged slide extraction pipeline:

  sampled_frames.avi + sampled_manifest.json
    -> timeline_segments.json
    -> segment preview images

The classifier is intentionally coarse. It only decides which broad regions
should be processed by later slide-specific passes:

  - slide: mostly static presentation screen, including small annotations
  - video: continuous visual motion, including embedded videos and demos
  - unknown: ambiguous region that should be inspected or handled cautiously

Usage:
    python -m pipeline.timeline_region_classifier \
        --cache sample_cache_dir \
        --output region_segments_dir
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import cv2
import imagehash
import numpy as np
from PIL import Image

try:
    from .sample_cache import iter_sample_cache, load_sample_cache
    from .person_masks import load_person_mask, masked_pair
except ImportError:  # pragma: no cover - allows direct script execution
    from sample_cache import iter_sample_cache, load_sample_cache
    from person_masks import load_person_mask, masked_pair


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


SEGMENTS_FILENAME = "timeline_segments.json"


@dataclass
class RegionClassifierConfig:
    window_sec: float = 4.0
    min_segment_sec: float = 12.0

    # Per-sample coarse thresholds, based on sampled-cache prev_* metrics.
    active_mse: float = 90.0
    very_active_mse: float = 220.0
    active_hash: int = 4
    cut_mse: float = 500.0
    cut_hash: int = 10

    # Pixel-level motion thresholds from sampled_frames.avi.
    # Embedded videos often move only inside a small rectangle, so MSE/pHash
    # alone can under-detect them. A low changed-pixel ratio sustained over time
    # is a stronger signal than a few abrupt slide cuts.
    diff_threshold: int = 12
    motion_ratio: float = 0.006
    strong_motion_ratio: float = 0.025

    # Window classification thresholds. Cut-like spikes are ignored for the
    # continuous-motion ratios because slide decks can have abrupt page turns.
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


def _safe_values(values: Iterable[float | int | None]) -> list[float]:
    return [float(v) for v in values if v is not None]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _content_region(frame: np.ndarray, cfg: RegionClassifierConfig) -> np.ndarray:
    h, w = frame.shape[:2]
    x0 = max(0, min(w - 1, int(w * cfg.crop_left)))
    y0 = max(0, min(h - 1, int(h * cfg.crop_top)))
    x1 = max(x0 + 1, min(w, int(w * cfg.crop_right)))
    y1 = max(y0 + 1, min(h, int(h * cfg.crop_bottom)))
    return frame[y0:y1, x0:x1]


def _decision_frame(frame: np.ndarray, cfg: RegionClassifierConfig) -> np.ndarray:
    cropped = _content_region(frame, cfg)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _decision_mask(mask: np.ndarray | None, cfg: RegionClassifierConfig) -> np.ndarray | None:
    if mask is None:
        return None
    return _content_region(mask.astype(np.uint8), cfg).astype(bool)


def _changed_ratio(frame_a: np.ndarray, frame_b: np.ndarray, threshold: int) -> float:
    diff = cv2.absdiff(frame_a, frame_b)
    return float(np.sum(diff > threshold) / diff.size)


def _mse(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    a = frame_a.astype(np.float32)
    b = frame_b.astype(np.float32)
    return float(np.mean((a - b) ** 2))


def _phash_int(frame: np.ndarray) -> int:
    pil_img = Image.fromarray(frame)
    return int(str(imagehash.phash(pil_img)), 16)


def _phash_distance(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


def _sample_motion_metrics(cache_dir: str | Path, cfg: RegionClassifierConfig) -> dict[int, dict]:
    metrics: dict[int, dict] = {}
    prev_decision = None
    prev_mask = None
    for frame_info, frame in iter_sample_cache(cache_dir):
        sample_index = int(frame_info["sample_index"])
        decision = _decision_frame(frame, cfg)
        mask = _decision_mask(load_person_mask(cache_dir, frame_info), cfg)
        if prev_decision is None:
            metrics[sample_index] = {
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
            metrics[sample_index] = {
                "changed_ratio": round(changed, 6),
                "motion_mse": round(motion_mse, 6),
                "motion_hash_dist": motion_hash_dist,
                "pixel_motion": changed >= cfg.motion_ratio,
                "strong_pixel_motion": changed >= cfg.strong_motion_ratio,
            }
        prev_decision = decision
        prev_mask = mask
    return metrics


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


def classify_regions(
    cache_dir: str,
    output_dir: str,
    cfg: RegionClassifierConfig | None = None,
) -> dict:
    cfg = cfg or RegionClassifierConfig()
    manifest = load_sample_cache(cache_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("segment_*.jpg"):
        stale.unlink(missing_ok=True)

    motion_metrics = _sample_motion_metrics(cache_dir, cfg)
    windows = _window_records(manifest, motion_metrics, cfg)
    groups = _merge_short_groups(_merge_windows(windows), cfg)
    segments = [_flatten_segment(seg, idx) for idx, seg in enumerate(groups, start=1)]
    _save_segment_previews(cache_dir, out_dir, segments)

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

    log.info("region classification done: segments=%s output=%s", len(segments), out_path)
    for region_type, info in sorted(summary.items()):
        log.info("  %-8s count=%s duration=%.1fs", region_type, info["count"], info["duration_sec"])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify sampled-cache timeline regions.")
    parser.add_argument("--cache", required=True, help="Sample cache directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--window-sec", type=float, default=RegionClassifierConfig.window_sec)
    parser.add_argument("--min-segment-sec", type=float, default=RegionClassifierConfig.min_segment_sec)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = RegionClassifierConfig(
        window_sec=args.window_sec,
        min_segment_sec=args.min_segment_sec,
    )
    classify_regions(args.cache, args.output, cfg)


if __name__ == "__main__":
    main()
