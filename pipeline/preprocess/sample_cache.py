"""
Build a lightweight sampled-frame cache from an input video.

The cache is the shared coordinate system for later passes:
  1. practice/video/slide region classification
  2. scene/base detection
  3. annotation detection

It stores resized sampled frames plus a manifest that maps every sample back to
the original frame number.

Usage:
    python -m pipeline.sample_cache --input lecture.mp4 --output sample_cache/
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Iterator

import cv2
import imagehash
import numpy as np
from PIL import Image

try:
    from .person_masks import MASKS_DIRNAME, PRESENCE_MASKS_DIRNAME
    from .video_decode import iter_video_frames, read_video_metadata
except ImportError:  # pragma: no cover - allows direct script execution
    from person_masks import MASKS_DIRNAME, PRESENCE_MASKS_DIRNAME
    from video_decode import iter_video_frames, read_video_metadata


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


MANIFEST_FILENAME = "sampled_manifest.json"
VIDEO_FILENAME = "sampled_frames.avi"
SCHEMA_VERSION = 1


@dataclass
class SampleCacheConfig:
    sample_every: int = 2
    resize_width: int = 768
    jpeg_quality: int = 95
    decode_backend: str = os.getenv("GRAPHLEC_SLIDE_DECODE_BACKEND", "auto")
    person_mask_batch_size: int = 32
    person_masks: bool = True
    person_mask_model: str = "yolov8n-seg.pt"
    person_mask_conf: float = 0.25
    person_mask_dilate_px: int = 30
    person_mask_static_diff_threshold: float = 1.0
    person_mask_static_changed_ratio_threshold: float = 0.003
    person_mask_match_iou_threshold: float = 0.05
    person_mask_fill_gap_sec: float = 6.0
    save_person_mask_previews: bool = False
    person_mask_preview_limit: int = 30


def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def to_decision_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def compute_mse(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    a = frame_a.astype(np.float32) if frame_a.ndim == 2 else cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = frame_b.astype(np.float32) if frame_b.ndim == 2 else cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean((a - b) ** 2))


def compute_phash_int(frame: np.ndarray) -> int:
    if frame.ndim == 2:
        pil_img = Image.fromarray(frame)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return int(str(imagehash.phash(pil_img)), 16)


def phash_distance_int(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


def _load_person_model(model_name: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        log.warning("ultralytics is not installed; person masks disabled")
        return None
    try:
        return YOLO(model_name)
    except Exception as exc:
        log.warning("failed to load person mask model %s; person masks disabled: %s", model_name, exc)
        return None


def _detections_from_result(result, height: int, width: int) -> list[dict]:
    if result.masks is None:
        return []
    detections: list[dict] = []
    for box, polygon in zip(result.boxes, result.masks.xy):
        if len(polygon) < 3:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
        if not bool(mask.any()):
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "mask": mask,
        })
    return detections


def _person_detections_from_frames(model, frames: list[np.ndarray], conf: float) -> list[list[dict]]:
    if model is None or not frames:
        return [[] for _ in frames]
    height, width = frames[0].shape[:2]
    results = model(frames, classes=[0], conf=conf, verbose=False, stream=False)
    return [_detections_from_result(result, height, width) for result in results]


def _person_detections_from_frame(model, frame: np.ndarray, conf: float) -> list[dict]:
    if model is None:
        return []
    height, width = frame.shape[:2]
    result = model(frame, classes=[0], conf=conf, verbose=False)[0]
    return _detections_from_result(result, height, width)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = max(1.0, float((ax2 - ax1) * (ay2 - ay1)))
    area_b = max(1.0, float((bx2 - bx1) * (by2 - by1)))
    return inter / (area_a + area_b - inter)


def _bbox_motion_metrics(frame_a: np.ndarray, frame_b: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    h, w = frame_a.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    a = cv2.cvtColor(frame_a[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(frame_b[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(a - b)
    return float(np.mean(diff)), float(np.mean(diff >= 8.0))


def _moving_person_mask(
    frame: np.ndarray,
    detections: list[dict],
    next_frame: np.ndarray | None,
    next_detections: list[dict],
    static_diff_threshold: float,
    static_changed_ratio_threshold: float,
    match_iou_threshold: float,
    dilate_px: int,
) -> np.ndarray | None:
    if next_frame is None or not detections or not next_detections:
        return None

    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for det in detections:
        bbox = det["bbox"]
        best_iou = max((_bbox_iou(bbox, curr["bbox"]) for curr in next_detections), default=0.0)
        if best_iou < match_iou_threshold:
            continue
        mean_diff, changed_ratio = _bbox_motion_metrics(frame, next_frame, bbox)
        if mean_diff < static_diff_threshold and changed_ratio < static_changed_ratio_threshold:
            continue
        x1, y1, x2, y2 = bbox
        pad = max(0, int(dilate_px))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(width, x2 + pad), min(height, y2 + pad)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1

    if not bool(mask.any()):
        return None
    return (mask > 0).astype(np.uint8)


def _person_presence_mask(
    frame: np.ndarray,
    detections: list[dict],
    dilate_px: int,
) -> np.ndarray | None:
    if not detections:
        return None
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    pad = max(0, int(dilate_px))
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(width, x2 + pad), min(height, y2 + pad)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    if not bool(mask.any()):
        return None
    return (mask > 0).astype(np.uint8)


def _masked_preview(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    preview = frame.copy()
    preview[mask.astype(bool)] = 0
    return preview


def _fill_short_person_mask_gaps(frames: list[dict], max_gap: int) -> int:
    if max_gap <= 0:
        return 0
    mask_positions = [
        (idx, frame["person_mask_filename"])
        for idx, frame in enumerate(frames)
        if frame.get("person_mask_filename")
    ]
    if not mask_positions:
        return 0

    filled = 0
    for idx, frame in enumerate(frames):
        if frame.get("person_mask_filename"):
            continue
        best_filename = None
        best_dist = max_gap + 1
        for mask_idx, filename in mask_positions:
            dist = abs(mask_idx - idx)
            if dist < best_dist:
                best_dist = dist
                best_filename = filename
            if mask_idx > idx and dist > best_dist:
                break
        if best_filename and best_dist <= max_gap:
            frame["person_mask_filename"] = best_filename
            frame["person_mask_inherited"] = True
            frame["person_mask_inherited_distance"] = best_dist
            filled += 1
    return filled

def create_sample_cache(
    input_path: str,
    output_dir: str,
    cfg: SampleCacheConfig | None = None,
) -> dict:
    cfg = cfg or SampleCacheConfig()
    cfg.sample_every = max(1, int(cfg.sample_every))
    cfg.resize_width = max(160, int(cfg.resize_width))

    video_meta = read_video_metadata(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_path = output_path / VIDEO_FILENAME
    manifest_path = output_path / MANIFEST_FILENAME
    masks_path = output_path / MASKS_DIRNAME
    presence_masks_path = output_path / PRESENCE_MASKS_DIRNAME
    mask_previews_path = output_path / "person_mask_previews"
    video_path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    if masks_path.exists():
        for stale in masks_path.glob("person_mask_*.npy"):
            stale.unlink(missing_ok=True)
    masks_path.mkdir(parents=True, exist_ok=True)
    if presence_masks_path.exists():
        for stale in presence_masks_path.glob("person_presence_mask_*.npy"):
            stale.unlink(missing_ok=True)
    presence_masks_path.mkdir(parents=True, exist_ok=True)
    if mask_previews_path.exists():
        for stale in mask_previews_path.glob("person_mask_preview_*.jpg"):
            stale.unlink(missing_ok=True)
    if cfg.save_person_mask_previews:
        mask_previews_path.mkdir(parents=True, exist_ok=True)

    cached_height = int(video_meta["height"] * (cfg.resize_width / video_meta["width"]))
    sampled_fps = max(1.0, video_meta["fps"] / cfg.sample_every)

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        sampled_fps,
        (cfg.resize_width, cached_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open sample cache writer: {video_path}")

    frames: list[dict] = []
    prev_decision = None
    prev_phash = None
    sample_index = 0
    progress_interval = 2000
    person_model = _load_person_model(cfg.person_mask_model) if cfg.person_masks else None
    masks_enabled = person_model is not None
    preview_count = 0
    pending_sample = None
    batch_size = max(1, int(cfg.person_mask_batch_size))

    def finalize_sample(sample: dict, next_frame: np.ndarray | None, next_detections: list[dict]) -> None:
        nonlocal preview_count
        frame_record = dict(sample["record"])
        if masks_enabled:
            presence_mask = _person_presence_mask(
                sample["frame"],
                sample["detections"],
                dilate_px=max(0, int(cfg.person_mask_dilate_px)),
            )
            if presence_mask is not None:
                presence_mask_filename = f"{PRESENCE_MASKS_DIRNAME}/person_presence_mask_{int(sample['sample_index']):06d}.npy"
                np.save(output_path / presence_mask_filename, presence_mask, allow_pickle=False)
                frame_record["person_presence_mask_filename"] = presence_mask_filename
                frame_record["person_presence_ratio"] = round(float(np.mean(presence_mask.astype(bool))), 6)
            person_mask = _moving_person_mask(
                sample["frame"],
                sample["detections"],
                next_frame,
                next_detections,
                static_diff_threshold=max(0.0, float(cfg.person_mask_static_diff_threshold)),
                static_changed_ratio_threshold=max(0.0, float(cfg.person_mask_static_changed_ratio_threshold)),
                match_iou_threshold=max(0.0, float(cfg.person_mask_match_iou_threshold)),
                dilate_px=max(0, int(cfg.person_mask_dilate_px)),
            )
            if person_mask is not None:
                person_mask_filename = f"{MASKS_DIRNAME}/person_mask_{int(sample['sample_index']):06d}.npy"
                np.save(output_path / person_mask_filename, person_mask, allow_pickle=False)
                frame_record["person_mask_filename"] = person_mask_filename
                if cfg.save_person_mask_previews and preview_count < max(0, int(cfg.person_mask_preview_limit)):
                    preview_count += 1
                    preview_filename = f"person_mask_preview_{int(sample['sample_index']):06d}.jpg"
                    preview = _masked_preview(sample["frame"], person_mask)
                    cv2.imwrite(
                        str(mask_previews_path / preview_filename),
                        preview,
                        [cv2.IMWRITE_JPEG_QUALITY, int(cfg.jpeg_quality)],
                    )
        frames.append(frame_record)

    log.info(
        "sample cache start: input=%s fps=%.2f frames=%s sample_every=%s size=%sx%s",
        input_path,
        video_meta["fps"],
        video_meta["frame_count"],
        cfg.sample_every,
        cfg.resize_width,
        cached_height,
    )

    frame_iter, active_backend = iter_video_frames(
        input_path,
        fps=float(video_meta["fps"]),
        width=int(video_meta["width"]),
        height=int(video_meta["height"]),
        sample_every=cfg.sample_every,
        decode_backend=cfg.decode_backend,
        output_width=cfg.resize_width,
        output_height=cached_height,
    )
    log.info("sample cache decode backend: %s", active_backend)

    def flush_batch(batch_samples: list[dict]) -> None:
        nonlocal sample_index, pending_sample, prev_decision, prev_phash
        if not batch_samples:
            return
        if masks_enabled:
            detections_batch = _person_detections_from_frames(
                person_model,
                [sample["frame"] for sample in batch_samples],
                conf=max(0.0, float(cfg.person_mask_conf)),
            )
        else:
            detections_batch = [[] for _ in batch_samples]

        for sample, detections in zip(batch_samples, detections_batch):
            sample_index += 1
            small = sample["frame"]
            decision = to_decision_frame(small)
            phash_int = compute_phash_int(decision)
            prev_mse = compute_mse(prev_decision, decision) if prev_decision is not None else None
            prev_hash_dist = (
                phash_distance_int(prev_phash, phash_int)
                if prev_phash is not None
                else None
            )

            writer.write(small)
            frame_record = {
                "sample_index": sample_index,
                "frame_no": int(sample["frame_no"]),
                "timestamp_sec": round(float(sample["timestamp_sec"]), 6),
                "phash_int": phash_int,
                "prev_mse": round(prev_mse, 6) if prev_mse is not None else None,
                "prev_hash_dist": prev_hash_dist,
            }

            if pending_sample is not None:
                finalize_sample(pending_sample, small, detections)
            pending_sample = {
                "sample_index": sample_index,
                "frame": small.copy(),
                "detections": detections,
                "record": frame_record,
            }

            prev_decision = decision
            prev_phash = phash_int

            if sample_index % progress_interval == 0:
                pct = (sample["frame_no"] / video_meta["frame_count"] * 100.0) if video_meta["frame_count"] > 0 else 0.0
                log.info("sample cache progress: samples=%s frame=%s %.1f%%", sample_index, sample["frame_no"], pct)

    try:
        pending_batch: list[dict] = []
        for frame_no, timestamp_sec, frame in frame_iter:
            pending_batch.append({
                "frame_no": int(frame_no),
                "timestamp_sec": float(timestamp_sec),
                "frame": frame,
            })
            if len(pending_batch) >= batch_size:
                flush_batch(pending_batch)
                pending_batch = []
        if pending_batch:
            flush_batch(pending_batch)
    finally:
        writer.release()

    if pending_sample is not None:
        finalize_sample(pending_sample, None, [])

    mask_fill_gap_samples = max(0, int(round(float(cfg.person_mask_fill_gap_sec) * sampled_fps)))
    inherited_masks = _fill_short_person_mask_gaps(frames, max_gap=mask_fill_gap_samples)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input_path": input_path,
        "video_filename": VIDEO_FILENAME,
        "config": asdict(cfg),
        "person_masks": {
            "enabled": masks_enabled,
            "dirname": MASKS_DIRNAME,
            "presence_dirname": PRESENCE_MASKS_DIRNAME,
            "model": cfg.person_mask_model if masks_enabled else None,
            "coordinate_space": "sample_cache_frame",
            "mode": "moving_person_only",
            "static_diff_threshold": cfg.person_mask_static_diff_threshold,
            "static_changed_ratio_threshold": cfg.person_mask_static_changed_ratio_threshold,
            "match_iou_threshold": cfg.person_mask_match_iou_threshold,
            "fill_gap_sec": cfg.person_mask_fill_gap_sec,
            "fill_gap_samples": mask_fill_gap_samples,
            "inherited_count": inherited_masks,
            "preview_dirname": "person_mask_previews" if cfg.save_person_mask_previews else None,
            "preview_count": preview_count,
        },
        "source": video_meta,
        "cache": {
            "sampled_fps": sampled_fps,
            "width": cfg.resize_width,
            "height": cached_height,
            "sample_count": sample_index,
        },
        "frames": frames,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log.info("sample cache done: samples=%s output=%s", sample_index, output_path)
    return manifest


def load_sample_cache(cache_dir: str | Path) -> dict:
    manifest_path = Path(cache_dir) / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Sample cache manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_sample_cache(cache_dir: str | Path) -> Iterator[tuple[dict, np.ndarray]]:
    cache_path = Path(cache_dir)
    manifest = load_sample_cache(cache_path)
    video_path = cache_path / manifest["video_filename"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open sampled cache video: {video_path}")

    try:
        for frame_info in manifest.get("frames", []):
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError(f"Sample cache video ended early: {video_path}")
            yield frame_info, frame
    finally:
        cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sampled frame cache for video analysis passes.")
    parser.add_argument("--input", "-i", required=True, help="Input .mp4 path")
    parser.add_argument("--output", "-o", required=True, help="Output cache directory")
    parser.add_argument("--sample-every", type=int, default=SampleCacheConfig.sample_every)
    parser.add_argument("--resize-width", type=int, default=SampleCacheConfig.resize_width)
    parser.add_argument(
        "--decode-backend",
        choices=["opencv", "ffmpeg-cuda", "ffmpeg-videotoolbox", "auto"],
        default=SampleCacheConfig.decode_backend,
    )
    parser.add_argument("--person-mask-batch-size", type=int, default=SampleCacheConfig.person_mask_batch_size)
    parser.add_argument("--no-person-masks", action="store_true", help="Disable YOLO person mask generation")
    parser.add_argument("--person-mask-model", default=SampleCacheConfig.person_mask_model)
    parser.add_argument("--person-mask-dilate-px", type=int, default=SampleCacheConfig.person_mask_dilate_px)
    parser.add_argument("--person-mask-static-diff-threshold", type=float, default=SampleCacheConfig.person_mask_static_diff_threshold)
    parser.add_argument("--person-mask-static-changed-ratio-threshold", type=float, default=SampleCacheConfig.person_mask_static_changed_ratio_threshold)
    parser.add_argument("--person-mask-match-iou-threshold", type=float, default=SampleCacheConfig.person_mask_match_iou_threshold)
    parser.add_argument("--person-mask-fill-gap-sec", type=float, default=SampleCacheConfig.person_mask_fill_gap_sec)
    parser.add_argument("--save-person-mask-previews", action="store_true", help="Save a few masked sample preview JPGs for debugging")
    parser.add_argument("--person-mask-preview-limit", type=int, default=SampleCacheConfig.person_mask_preview_limit)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = SampleCacheConfig(
        sample_every=args.sample_every,
        resize_width=args.resize_width,
        decode_backend=args.decode_backend,
        person_mask_batch_size=args.person_mask_batch_size,
        person_masks=not args.no_person_masks,
        person_mask_model=args.person_mask_model,
        person_mask_dilate_px=args.person_mask_dilate_px,
        person_mask_static_diff_threshold=args.person_mask_static_diff_threshold,
        person_mask_static_changed_ratio_threshold=args.person_mask_static_changed_ratio_threshold,
        person_mask_match_iou_threshold=args.person_mask_match_iou_threshold,
        person_mask_fill_gap_sec=args.person_mask_fill_gap_sec,
        save_person_mask_previews=args.save_person_mask_previews,
        person_mask_preview_limit=args.person_mask_preview_limit,
    )
    create_sample_cache(args.input, args.output, cfg)


if __name__ == "__main__":
    main()
