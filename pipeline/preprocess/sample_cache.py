"""
Build a lightweight sampled-frame cache from an input video.

This replacement keeps the existing single-pass behavior and adds chunked cache
creation for long lecture videos. The chunked path creates overlap-trimmed
segments in parallel and records their exact local frame positions in one
downstream-compatible manifest.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import logging
import math
import os
import shutil
import subprocess
import gc
import tempfile
import time
import multiprocessing as mp
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass, field
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


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _system_memory_bytes() -> int | None:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None
    return pages * page_size if pages > 0 and page_size > 0 else None


def _sample_cache_temp_root(output_path: Path) -> tuple[Path, str]:
    mode = os.getenv("GRAPHLEC_SAMPLE_CACHE_STORAGE_MODE", "auto").strip().lower()
    if mode not in {"auto", "ram", "disk"}:
        mode = "auto"
    if mode == "disk":
        return output_path.parent, "disk"

    total_memory = _system_memory_bytes()
    ram_threshold = 16 * 1024**3
    ram_root = Path(os.getenv("GRAPHLEC_SAMPLE_CACHE_RAM_DIR", "/dev/shm"))
    if mode == "ram" or (mode == "auto" and total_memory is not None and total_memory > ram_threshold):
        try:
            ram_root.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(ram_root).free >= 2 * 1024**3:
                return ram_root, "ram"
        except OSError:
            pass
    return output_path.parent, "disk"


@dataclass
class SampleCacheConfig:
    sample_every: int = _env_int("GRAPHLEC_SAMPLE_CACHE_EVERY", 2)
    sample_fps: float = _env_float("GRAPHLEC_SAMPLE_CACHE_FPS", 10.0)
    resize_width: int = _env_int("GRAPHLEC_SAMPLE_CACHE_RESIZE_WIDTH", 768)
    jpeg_quality: int = 95
    decode_backend: str = os.getenv("GRAPHLEC_SLIDE_DECODE_BACKEND", "auto")
    person_mask_batch_size: int = _env_int("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_BATCH_SIZE", 32)
    person_masks: bool = _env_bool("GRAPHLEC_SAMPLE_CACHE_PERSON_MASKS", True)
    person_mask_model: str = os.getenv("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_MODEL", "/app/models/yolo26n.pt")
    person_mask_engine: str = os.getenv(
        "GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_ENGINE",
        "/app/storage/models/yolo26n-fp16.engine",
    )
    person_mask_workers: int = _env_int("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_WORKERS", 4)
    person_mask_task_sec: float = _env_float("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_TASK_SEC", 480.0)
    person_mask_task_overlap_sec: float = _env_float("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_TASK_OVERLAP_SEC", 5.0)
    person_mask_engine_batch_size: int = _env_int("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_ENGINE_BATCH_SIZE", 32)
    person_mask_roi_quantile: float = _env_float("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_ROI_QUANTILE", 0.95)
    person_mask_roi_recenter_px: int = _env_int("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_ROI_RECENTER_PX", 120)
    person_mask_roi_max_gap_sec: float = _env_float("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_ROI_MAX_GAP_SEC", 2.0)
    person_mask_gpu_min_free_mb: int = _env_int("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_MIN_FREE_MB", 0)
    person_mask_gpu_stagger_sec: float = _env_float("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_GPU_STAGGER_SEC", 0.0)
    person_mask_gpu_wait_sec: float = _env_float("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_GPU_WAIT_SEC", 0.5)
    person_mask_conf: float = _env_float("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_CONF", 0.70)
    person_mask_min_bbox_height_ratio: float = _env_float(
        "GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_MIN_BBOX_HEIGHT_RATIO", 0.08
    )
    person_mask_max_bbox_aspect: float = _env_float(
        "GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_MAX_BBOX_ASPECT", 1.50
    )
    # Detector boxes can miss a presenter's hair or outstretched hand.  The
    # fixed ROI must include this boundary motion or it leaks into annotation
    # detection as a tiny false write.
    person_mask_dilate_px: int = _env_int("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_DILATE_PX", 48)
    person_mask_static_diff_threshold: float = 1.0
    person_mask_static_changed_ratio_threshold: float = 0.003
    person_mask_match_iou_threshold: float = 0.05
    # A fixed ROI must only cover frames where movement was confirmed.  Filling
    # from a single nearby detection turns static false positives into long
    # masks, so this is disabled unless a deployment explicitly opts in.
    person_mask_fill_gap_sec: float = _env_float(
        "GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_FILL_GAP_SEC", 0.0
    )
    person_mask_fixed_motion_min_mean_diff: float = _env_float(
        "GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_FIXED_MOTION_MIN_MEAN_DIFF", 2.0
    )
    person_mask_fixed_motion_min_changed_ratio: float = _env_float(
        "GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_FIXED_MOTION_MIN_CHANGED_RATIO", 0.01
    )
    person_mask_fixed_min_center_shift_px: float = _env_float(
        "GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_FIXED_MIN_CENTER_SHIFT_PX", 12.0
    )
    person_mask_fixed_min_area_change_ratio: float = _env_float(
        "GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_FIXED_MIN_AREA_CHANGE_RATIO", 0.12
    )
    person_mask_active_ranges: list[tuple[float, float]] | None = field(default=None, repr=False)
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
        load_kwargs = {"task": "detect"} if str(model_name).lower().endswith(".engine") else {}
        return YOLO(model_name, **load_kwargs)
    except Exception as exc:
        log.warning("failed to load person mask model %s; person masks disabled: %s", model_name, exc)
        return None


def _detections_from_result(result, height: int, width: int) -> list[dict]:
    if result.boxes is None:
        return []
    detections: list[dict] = []
    polygons = list(result.masks.xy) if result.masks is not None else []
    for index, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
        detection = {
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "confidence": confidence,
        }
        if index < len(polygons):
            polygon = polygons[index]
            if len(polygon) >= 3:
                mask = np.zeros((height, width), dtype=np.uint8)
                pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 1)
                if bool(mask.any()):
                    detection["mask"] = mask
        detections.append(detection)
    return detections


def _cuda_oom_message(exc: Exception) -> bool:
    text = str(exc).lower()
    return "cuda" in text and "out of memory" in text


def _empty_torch_cuda_cache() -> None:
    try:
        import torch  # type: ignore
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _gpu_free_memory_mb() -> int | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None
    values: list[int] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(float(line)))
        except ValueError:
            continue
    if not values:
        return None
    return max(values)


def _wait_for_person_mask_gpu_gate(cfg: SampleCacheConfig, batch_size: int) -> None:
    min_free_mb = max(0, int(cfg.person_mask_gpu_min_free_mb))
    stagger_sec = max(0.0, float(cfg.person_mask_gpu_stagger_sec))
    wait_sec = max(0.05, float(cfg.person_mask_gpu_wait_sec))
    if min_free_mb <= 0 and stagger_sec <= 0.0:
        return

    lock_path = Path(os.getenv("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_GPU_GATE_LOCK", "/tmp/verilec_person_mask_gpu_gate.lock"))
    stamp_path = lock_path.with_suffix(".stamp")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    waited = False
    while True:
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            now = time.time()
            last_entered = 0.0
            try:
                last_entered = float(stamp_path.read_text(encoding="utf-8").strip() or "0")
            except Exception:
                last_entered = 0.0

            stagger_remaining = max(0.0, stagger_sec - (now - last_entered))
            free_mb = _gpu_free_memory_mb()
            enough_memory = min_free_mb <= 0 or free_mb is None or free_mb >= min_free_mb

            if stagger_remaining <= 0.0 and enough_memory:
                stamp_path.write_text(str(time.time()), encoding="utf-8")
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                if waited:
                    log.info(
                        "person mask GPU gate entered: free_mb=%s min_free_mb=%s batch_size=%s",
                        free_mb,
                        min_free_mb,
                        batch_size,
                    )
                return

            sleep_sec = max(wait_sec, stagger_remaining)
            if not enough_memory:
                log.info(
                    "person mask GPU gate waiting: free_mb=%s min_free_mb=%s batch_size=%s sleep=%.1fs",
                    free_mb,
                    min_free_mb,
                    batch_size,
                    sleep_sec,
                )
            waited = True
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        time.sleep(sleep_sec)


def _run_person_model_serialized(model, frames: list[np.ndarray], conf: float):
    """Run one person-model batch under a process-wide GPU execution lock.

    Chunk decoding remains parallel, but TensorRT execution contexts from
    separate spawned workers are not safe to enqueue concurrently with this
    dynamic engine.  Queueing only the short inference call avoids CUDA
    dispatch crashes without serializing video decode.
    """
    if not _env_bool("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_GPU_SERIALIZE", True):
        return model(frames, classes=[0], conf=conf, verbose=False, stream=False)

    lock_path = Path(
        os.getenv(
            "GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_GPU_EXEC_LOCK",
            "/tmp/verilec_person_mask_gpu_execute.lock",
        )
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    queued_at = time.perf_counter()
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        waited_sec = time.perf_counter() - queued_at
        if waited_sec >= 0.25:
            log.info(
                "person mask GPU inference queue acquired: waited=%.2fs batch_size=%s",
                waited_sec,
                len(frames),
            )
        try:
            return model(frames, classes=[0], conf=conf, verbose=False, stream=False)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _acquire_person_mask_trt_context_slot() -> tuple[object, int]:
    """Reserve one bounded TensorRT context slot for this chunk process.

    A YOLO TensorRT context consumes about 1.4 GiB for this dynamic engine.
    Keeping one context per decode chunk exhausts GPU memory even when batch
    execution is serialized, because context allocations remain resident.
    """
    limit = max(
        1,
        int(os.getenv("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_TRT_CONTEXTS", "2")),
    )
    root = Path(
        os.getenv(
            "GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_TRT_CONTEXT_DIR",
            "/tmp/verilec_person_mask_trt_context_slots",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    queued_at = time.perf_counter()
    while True:
        for slot_index in range(limit):
            handle = open(root / f"slot_{slot_index:02d}.lock", "a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            waited_sec = time.perf_counter() - queued_at
            if waited_sec >= 0.25:
                log.info(
                    "person mask TensorRT context slot acquired: slot=%s/%s waited=%.2fs",
                    slot_index + 1,
                    limit,
                    waited_sec,
                )
            return handle, slot_index
        time.sleep(0.05)


def _release_person_mask_trt_context_slot(slot: tuple[object, int] | None) -> None:
    if slot is None:
        return
    handle, _ = slot
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _person_bbox_allowed(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    cfg: SampleCacheConfig,
) -> bool:
    x1, y1, x2, y2 = bbox
    box_width = max(0, min(width, x2) - max(0, x1))
    box_height = max(0, min(height, y2) - max(0, y1))
    if box_width <= 0 or box_height <= 0:
        return False
    if box_height / max(1, height) < max(0.0, float(cfg.person_mask_min_bbox_height_ratio)):
        return False
    width_to_height = box_width / max(1, box_height)
    return width_to_height <= max(0.1, float(cfg.person_mask_max_bbox_aspect))


def _run_person_presence_gate(
    input_path: str,
    cfg: SampleCacheConfig,
    video_meta: dict,
    *,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    model=None,
) -> list[tuple[float, float]]:
    if not cfg.person_masks:
        return []

    interval_sec = max(
        1.0,
        float(os.getenv("GRAPHLEC_SAMPLE_CACHE_PERSON_GATE_INTERVAL_SEC", "20")),
    )
    # The gate is an activation guard, not the final person-mask decision.
    # Keep it conservative against false negatives: one positive seek sample
    # is enough to enable masking for the whole chunk.
    min_hits = max(
        1,
        int(os.getenv("GRAPHLEC_SAMPLE_CACHE_PERSON_GATE_MIN_HITS", "1")),
    )
    gate_conf = max(
        0.10,
        float(os.getenv("GRAPHLEC_SAMPLE_CACHE_PERSON_GATE_CONF", "0.45")),
    )
    video_duration = float(video_meta.get("duration_sec") or 0.0)
    start_sec = max(0.0, float(start_sec))
    end_sec = video_duration if end_sec is None else min(video_duration, float(end_sec))
    duration = max(0.0, end_sec - start_sec)
    if duration <= 0.0:
        return []

    width = int(video_meta["width"])
    height = int(video_meta["height"])
    output_width = int(os.getenv("GRAPHLEC_SAMPLE_CACHE_PERSON_GATE_WIDTH", "384"))
    output_height = int(height * (output_width / width))
    log.info(
        "person presence gate start: interval=%.1fs min_hits=%s conf=%.2f size=%sx%s mode=seek",
        interval_sec,
        min_hits,
        gate_conf,
        output_width,
        output_height,
    )

    model_source = cfg.person_mask_engine if Path(cfg.person_mask_engine).exists() else cfg.person_mask_model
    model_owned = model is None
    if model_owned:
        model = _load_person_model(model_source)
    if model is None:
        log.warning("person presence gate disabled: model load failed source=%s", model_source)
        return []

    sampled = 0
    hits = 0
    hit_timestamps: list[float] = []
    batch: list[tuple[float, np.ndarray]] = []
    fallback_used = False

    def check_batch(batch_items: list[tuple[float, np.ndarray]]) -> None:
        nonlocal sampled, hits, model, model_source, fallback_used
        if not batch_items:
            return
        batch_frames = [frame for _, frame in batch_items]
        try:
            gate_cfg = copy.deepcopy(cfg)
            gate_cfg.person_mask_conf = gate_conf
            # Do not apply the stricter final-mask geometry filters to the
            # presence gate. A partial or edge-of-frame person must activate
            # the chunk so the full-resolution mask pass can decide later.
            gate_cfg.person_mask_min_bbox_height_ratio = 0.0
            gate_cfg.person_mask_max_bbox_aspect = 10.0
            detections = _person_detections_from_frames(
                model,
                batch_frames,
                gate_conf,
                gate_cfg,
            )
        except AttributeError as exc:
            if not ("set_input_shape" in str(exc) and str(model_source).lower().endswith(".engine")):
                raise
            model_source = cfg.person_mask_model
            model = _load_person_model(model_source)
            fallback_used = True
            if model is None:
                detections = [[] for _ in batch_frames]
            else:
                detections = _person_detections_from_frames(
                    model,
                    batch_frames,
                    gate_conf,
                    gate_cfg,
                )
        sampled += len(batch_items)
        for (timestamp, _), items in zip(batch_items, detections):
            if any(float(item.get("confidence", 0.0)) >= gate_conf for item in items):
                hits += 1
                hit_timestamps.append(float(timestamp))

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video for person presence gate: {input_path}")
        timestamp = start_sec
        while timestamp <= end_sec + 1e-6:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                frame = resize_frame(frame, output_width)
                batch.append((timestamp, frame))
            if len(batch) >= max(1, int(cfg.person_mask_batch_size)):
                check_batch(batch)
                batch = []
            timestamp += interval_sec
        if batch:
            check_batch(batch)
        cap.release()
    except RuntimeError as exc:
        if _cuda_oom_message(exc):
            log.warning("person presence gate disabled after CUDA OOM")
            return []
        raise
    finally:
        if model_owned:
            del model
            _empty_torch_cuda_cache()
            gc.collect()

    ranges: list[tuple[float, float]] = []
    if hits >= min_hits:
        for timestamp in sorted(hit_timestamps):
            start = max(start_sec, timestamp - interval_sec / 2.0)
            end = min(end_sec, timestamp + interval_sec / 2.0)
            if ranges and start <= ranges[-1][1] + interval_sec:
                ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
            else:
                ranges.append((start, end))
    confirmed = bool(ranges)
    log.info(
        "person presence gate done: sampled=%s hits=%s min_hits=%s confirmed=%s active_ranges=%s backend=%s fallback=%s",
        sampled,
        hits,
        min_hits,
        confirmed,
        len(ranges),
        "opencv-seek",
        fallback_used,
    )
    return ranges


def _person_mask_active_at(timestamp_sec: float, ranges: list[tuple[float, float]] | None) -> bool:
    if ranges is None:
        return True
    timestamp = float(timestamp_sec)
    return any(start - 1e-6 <= timestamp <= end + 1e-6 for start, end in ranges)


def _person_detections_from_frames(
    model,
    frames: list[np.ndarray],
    conf: float,
    cfg: SampleCacheConfig,
) -> list[list[dict]]:
    if model is None or not frames:
        return [[] for _ in frames]
    _wait_for_person_mask_gpu_gate(cfg, len(frames))
    height, width = frames[0].shape[:2]
    try:
        results = _run_person_model_serialized(model, frames, conf)
    except RuntimeError as exc:
        if _cuda_oom_message(exc):
            log.warning(
                "person mask YOLO CUDA OOM after GPU gate; skipping person masks for this batch (batch_size=%s)",
                len(frames),
            )
            _empty_torch_cuda_cache()
            gc.collect()
            return [[] for _ in frames]
        raise
    detections_by_frame = []
    for result in results:
        detections = _detections_from_result(result, height, width)
        detections_by_frame.append([
            detection
            for detection in detections
            if _person_bbox_allowed(detection["bbox"], width, height, cfg)
        ])
    return detections_by_frame


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


def _person_presence_mask(frame: np.ndarray, detections: list[dict], dilate_px: int) -> np.ndarray | None:
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


def _prepare_output_dirs(output_path: Path, cfg: SampleCacheConfig) -> tuple[Path, Path, Path, Path, Path]:
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
    return video_path, manifest_path, masks_path, presence_masks_path, mask_previews_path


def _create_sample_cache_impl(
    input_path: str,
    output_dir: str,
    cfg: SampleCacheConfig | None = None,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
    chunk_index: int | None = None,
    core_start_sec: float | None = None,
    core_end_sec: float | None = None,
) -> dict:
    cfg = cfg or SampleCacheConfig()
    cfg.sample_every = max(1, int(cfg.sample_every))
    cfg.sample_fps = max(0.1, float(cfg.sample_fps))
    cfg.resize_width = max(160, int(cfg.resize_width))
    video_meta = read_video_metadata(input_path)
    output_path = Path(output_dir)
    video_path, manifest_path, _, _, mask_previews_path = _prepare_output_dirs(output_path, cfg)

    cached_height = int(video_meta["height"] * (cfg.resize_width / video_meta["width"]))
    sampled_fps = float(cfg.sample_fps)
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
    person_model_source = cfg.person_mask_model
    if cfg.person_masks and Path(cfg.person_mask_engine).exists():
        person_model_source = cfg.person_mask_engine
    person_model = None
    masks_enabled = bool(cfg.person_masks)
    async_person_model = None
    async_person_model_load_failed = False
    async_person_backend = "disabled"
    trt_context_slot = None
    person_model_warmed = False
    person_gate_interval_sec = max(
        1.0,
        float(os.getenv("GRAPHLEC_SAMPLE_CACHE_PERSON_GATE_INTERVAL_SEC", "20")),
    )
    person_gate_conf = max(
        0.10,
        float(os.getenv("GRAPHLEC_SAMPLE_CACHE_PERSON_GATE_CONF", "0.45")),
    )
    person_gate_next_timestamp = float(start_sec or 0.0)
    person_gate_active = False
    preview_count = 0
    pending_sample = None
    batch_size = max(1, int(cfg.person_mask_batch_size))
    total_frame_count = max(1, int(video_meta.get("frame_count") or 1))
    detection_executor = ThreadPoolExecutor(max_workers=1) if masks_enabled else None
    detection_queue: list[tuple[list[dict], list[int], object]] = []
    configured_inflight = os.getenv("GRAPHLEC_SAMPLE_CACHE_PERSON_MASK_INFLIGHT_BATCHES")
    if configured_inflight is None:
        total_memory = _system_memory_bytes()
        default_inflight = 2 if total_memory is None or total_memory > 16 * 1024**3 else 1
        max_inflight_batches = default_inflight
    else:
        max_inflight_batches = max(1, int(configured_inflight))
    if masks_enabled:
        log.info(
            "person mask async start: batch_size=%s inflight_batches=%s model=%s",
            batch_size,
            max_inflight_batches,
            person_model_source,
        )

    def detect_person_batch(
        batch_frames: list[np.ndarray],
        conf_override: float | None = None,
    ) -> list[list[dict]]:
        nonlocal async_person_model, async_person_model_load_failed, async_person_backend, trt_context_slot
        if not masks_enabled or async_person_model_load_failed:
            return [[] for _ in batch_frames]
        if async_person_model is None:
            if str(person_model_source).lower().endswith(".engine"):
                trt_context_slot = _acquire_person_mask_trt_context_slot()
            async_person_model = _load_person_model(person_model_source)
            if async_person_model is None:
                _release_person_mask_trt_context_slot(trt_context_slot)
                trt_context_slot = None
                async_person_model_load_failed = True
                return [[] for _ in batch_frames]
            async_person_backend = "tensorrt" if str(person_model_source).lower().endswith(".engine") else "pytorch"
        inference_conf = max(0.0, float(cfg.person_mask_conf if conf_override is None else conf_override))
        inference_cfg = cfg
        if conf_override is not None:
            inference_cfg = copy.deepcopy(cfg)
            inference_cfg.person_mask_conf = inference_conf
        try:
            return _person_detections_from_frames(
                async_person_model,
                batch_frames,
                inference_conf,
                inference_cfg,
            )
        except AttributeError as exc:
            is_trt_context_error = (
                "set_input_shape" in str(exc)
                and str(person_model_source).lower().endswith(".engine")
            )
            if not is_trt_context_error:
                raise
            fallback_model = cfg.person_mask_model
            log.warning(
                "TensorRT person mask context initialization failed for %s; "
                "falling back to GPU PyTorch model %s: %s",
                person_model_source,
                fallback_model,
                exc,
            )
            async_person_model = _load_person_model(fallback_model)
            if async_person_model is None:
                raise
            _release_person_mask_trt_context_slot(trt_context_slot)
            trt_context_slot = None
            async_person_backend = "pytorch"
            return _person_detections_from_frames(
                async_person_model,
                batch_frames,
                inference_conf,
                inference_cfg,
            )

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

    range_label = "full" if start_sec is None and end_sec is None else f"{start_sec:.2f}s~{end_sec:.2f}s"
    log.info(
        "sample cache start: input=%s range=%s fps=%.2f frames=%s sample_fps=%s size=%sx%s",
        input_path,
        range_label,
        video_meta["fps"],
        video_meta["frame_count"],
        cfg.sample_fps,
        cfg.resize_width,
        cached_height,
    )

    frame_iter, active_backend = iter_video_frames(
        input_path,
        fps=float(video_meta["fps"]),
        width=int(video_meta["width"]),
        height=int(video_meta["height"]),
        sample_every=1,
        decode_backend=cfg.decode_backend,
        output_width=cfg.resize_width,
        output_height=cached_height,
        sample_fps=cfg.sample_fps,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    log.info("sample cache decode backend: %s", active_backend)

    def process_batch(batch_samples: list[dict], detections_batch: list[list[dict]]) -> None:
        nonlocal sample_index, pending_sample, prev_decision, prev_phash
        if not batch_samples:
            return

        for sample, detections in zip(batch_samples, detections_batch):
            sample_index += 1
            small = sample["frame"]
            decision = to_decision_frame(small)
            phash_int = compute_phash_int(decision)
            prev_mse = compute_mse(prev_decision, decision) if prev_decision is not None else None
            prev_hash_dist = phash_distance_int(prev_phash, phash_int) if prev_phash is not None else None

            writer.write(small)
            frame_record = {
                "sample_index": sample_index,
                "frame_no": int(sample["frame_no"]),
                "timestamp_sec": round(float(sample["timestamp_sec"]), 6),
                "phash_int": phash_int,
                "prev_mse": round(prev_mse, 6) if prev_mse is not None else None,
                "prev_hash_dist": prev_hash_dist,
            }
            if masks_enabled:
                frame_record["person_boxes"] = [list(det["bbox"]) for det in detections]

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
                pct = (sample["frame_no"] / total_frame_count * 100.0) if total_frame_count > 0 else 0.0
                log.info("sample cache progress: samples=%s frame=%s %.1f%%", sample_index, sample["frame_no"], pct)

    def flush_batch(batch_samples: list[dict]) -> None:
        nonlocal person_gate_next_timestamp, person_gate_active, person_model_warmed
        if not batch_samples:
            return
        if detection_executor is None:
            process_batch(batch_samples, [[] for _ in batch_samples])
            return

        if cfg.person_mask_active_ranges is not None:
            active_indices = [
                index for index, sample in enumerate(batch_samples)
                if _person_mask_active_at(float(sample["timestamp_sec"]), cfg.person_mask_active_ranges)
            ]
        elif not person_gate_active:
            gate_indices = [
                index for index, sample in enumerate(batch_samples)
                if float(sample["timestamp_sec"]) + 1e-6 >= person_gate_next_timestamp
            ]
            if gate_indices:
                gate_index = gate_indices[0]
                gate_timestamp = float(batch_samples[gate_index]["timestamp_sec"])
                person_gate_next_timestamp = gate_timestamp + person_gate_interval_sec
                gate_future = detection_executor.submit(
                    detect_person_batch,
                    [batch_samples[gate_index]["frame"]],
                    person_gate_conf,
                )
                gate_detections = gate_future.result()
                if gate_detections and gate_detections[0]:
                    person_gate_active = True
                    log.info(
                        "person mask gate activated: chunk=%s timestamp=%.1fs",
                        chunk_index,
                        gate_timestamp,
                    )
                    active_indices = list(range(len(batch_samples)))
                else:
                    active_indices = []
            else:
                active_indices = []
        else:
            active_indices = list(range(len(batch_samples)))
        if not active_indices:
            process_batch(batch_samples, [[] for _ in batch_samples])
            return

        if not person_model_warmed:
            # Ultralytics creates the TensorRT execution context lazily at the
            # first prediction.  Initializing it from the chunk process main
            # thread is reliable; doing so first from the async worker thread
            # can leave its context as None and trigger the PyTorch fallback.
            warmup_detections = detect_person_batch(
                [batch_samples[index]["frame"] for index in active_indices],
            )
            person_model_warmed = True
            ready_detections = [[] for _ in batch_samples]
            for index, detections in zip(active_indices, warmup_detections):
                ready_detections[index] = detections
            log.info(
                "person mask warmup complete: chunk=%s frames=%s backend=%s model=%s",
                chunk_index,
                len(active_indices),
                async_person_backend,
                person_model_source,
            )
            process_batch(batch_samples, ready_detections)
            return

        future = detection_executor.submit(
            detect_person_batch,
            [batch_samples[index]["frame"] for index in active_indices],
        )
        detection_queue.append((batch_samples, active_indices, future))
        if len(detection_queue) >= max_inflight_batches:
            ready_samples, ready_indices, ready_future = detection_queue.pop(0)
            ready_detections = [[] for _ in ready_samples]
            for index, detections in zip(ready_indices, ready_future.result()):
                ready_detections[index] = detections
            process_batch(ready_samples, ready_detections)

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
        while detection_queue:
            ready_samples, ready_indices, ready_future = detection_queue.pop(0)
            ready_detections = [[] for _ in ready_samples]
            for index, detections in zip(ready_indices, ready_future.result()):
                ready_detections[index] = detections
            process_batch(ready_samples, ready_detections)
    finally:
        if detection_executor is not None:
            detection_executor.shutdown(wait=True, cancel_futures=False)
        _release_person_mask_trt_context_slot(trt_context_slot)
        writer.release()

    if pending_sample is not None:
        finalize_sample(pending_sample, None, [])

    mask_fill_gap_samples = max(0, int(round(float(cfg.person_mask_fill_gap_sec) * sampled_fps)))
    inherited_masks = _fill_short_person_mask_gaps(frames, max_gap=mask_fill_gap_samples)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input_path": input_path,
        "video_filename": VIDEO_FILENAME,
        "decode_backend": active_backend,
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
            "decode_backend": active_backend,
        },
        "range": {
            "chunk_index": chunk_index,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "core_start_sec": core_start_sec,
            "core_end_sec": core_end_sec,
        },
        "frames": frames,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log.info("sample cache done: samples=%s output=%s", sample_index, output_path)
    return manifest


def create_sample_cache(input_path: str, output_dir: str, cfg: SampleCacheConfig | None = None) -> dict:
    return _create_sample_cache_impl(input_path, output_dir, cfg)


def create_sample_cache_range(
    input_path: str,
    output_dir: str,
    cfg: SampleCacheConfig | None,
    start_sec: float,
    end_sec: float,
    *,
    chunk_index: int | None = None,
    core_start_sec: float | None = None,
    core_end_sec: float | None = None,
) -> dict:
    return _create_sample_cache_impl(
        input_path,
        output_dir,
        cfg,
        start_sec=start_sec,
        end_sec=end_sec,
        chunk_index=chunk_index,
        core_start_sec=core_start_sec,
        core_end_sec=core_end_sec,
    )


def _chunk_specs(duration: float, chunk_sec: float, overlap_sec: float) -> list[dict]:
    if duration <= chunk_sec:
        return []
    chunk_sec = max(30.0, float(chunk_sec))
    overlap_sec = max(0.0, min(float(overlap_sec), chunk_sec / 4.0))
    chunk_count = max(1, math.ceil(duration / chunk_sec))
    specs: list[dict] = []
    for idx in range(chunk_count):
        core_start = idx * chunk_sec
        core_end = min(duration, (idx + 1) * chunk_sec)
        start = 0.0 if idx == 0 else max(0.0, core_start - overlap_sec)
        end = duration if idx == chunk_count - 1 else min(duration, core_end + overlap_sec)
        specs.append({
            "chunk_index": idx,
            "start_sec": start,
            "end_sec": end,
            "core_start_sec": core_start,
            "core_end_sec": core_end,
            "is_last": idx == chunk_count - 1,
        })
    return specs


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float32), min(1.0, max(0.0, q))))


def _person_mask_task_specs(frames: list[dict], task_sec: float, overlap_sec: float) -> list[dict]:
    if not frames:
        return []
    duration = float(frames[-1].get("timestamp_sec", 0.0) or 0.0)
    task_sec = max(30.0, float(task_sec))
    overlap_sec = max(0.0, min(float(overlap_sec), task_sec / 4.0))
    count = max(1, math.ceil(max(duration, 0.01) / task_sec))
    specs: list[dict] = []
    for index in range(count):
        core_start = index * task_sec
        core_end = duration if index == count - 1 else min(duration, (index + 1) * task_sec)
        start = 0.0 if index == 0 else max(0.0, core_start - overlap_sec)
        end = duration if index == count - 1 else min(duration, core_end + overlap_sec)
        specs.append({
            "task_index": index,
            "start_sec": start,
            "end_sec": end,
            "core_start_sec": core_start,
            "core_end_sec": core_end,
            "is_last": index == count - 1,
        })
    return specs


def _ensure_person_mask_engine(cfg: SampleCacheConfig) -> Path:
    """Build the one shared TensorRT detector engine before worker processes start."""
    try:
        import tensorrt  # noqa: F401
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "TensorRT person mask mode requires the tensorrt-cu12 and ultralytics packages"
        ) from exc

    source_path = Path(cfg.person_mask_model)
    if not source_path.exists():
        raise FileNotFoundError(f"person detector model not found: {source_path}")
    engine_path = Path(cfg.person_mask_engine)
    if engine_path.exists() and engine_path.stat().st_size > 0:
        log.info("person mask TensorRT engine reuse: %s", engine_path)
        return engine_path

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "person mask TensorRT engine build: source=%s engine=%s batch=%s",
        source_path,
        engine_path,
        cfg.person_mask_engine_batch_size,
    )
    model = YOLO(str(source_path))
    exported = Path(
        model.export(
            format="engine",
            device=0,
            half=True,
            dynamic=True,
            batch=max(1, int(cfg.person_mask_engine_batch_size)),
            imgsz=640,
            workspace=2,
        )
    )
    if not exported.exists():
        raise RuntimeError(f"TensorRT export did not produce an engine: {exported}")
    if exported.resolve() != engine_path.resolve():
        shutil.copy2(exported, engine_path)
    return engine_path


def _detect_person_boxes(model, frames: list[np.ndarray], cfg: SampleCacheConfig) -> list[list[tuple[int, int, int, int]]]:
    if not frames:
        return []
    results = model(
        frames,
        classes=[0],
        conf=max(0.0, float(cfg.person_mask_conf)),
        verbose=False,
        stream=False,
        imgsz=640,
    )
    boxes_by_frame: list[list[tuple[int, int, int, int]]] = []
    for result in results:
        boxes: list[tuple[int, int, int, int]] = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2))
                if _person_bbox_allowed(bbox, int(frames[0].shape[1]), int(frames[0].shape[0]), cfg):
                    boxes.append(bbox)
        boxes_by_frame.append(boxes)
    return boxes_by_frame


def _presence_ratio_from_boxes(
    boxes: list[tuple[int, int, int, int]],
    height: int,
    width: int,
    pad: int,
) -> float:
    if not boxes:
        return 0.0
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(width, x2 + pad), min(height, y2 + pad)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return round(float(np.mean(mask)), 6)


def _fixed_box_epochs(
    records: list[dict],
    cfg: SampleCacheConfig,
    width: int,
    height: int,
) -> list[dict]:
    """Group detections into stable presenter epochs and derive one fixed box per epoch."""
    if not records:
        return []
    max_gap = max(0.1, float(cfg.person_mask_roi_max_gap_sec))
    recenter_px = max(1, int(cfg.person_mask_roi_recenter_px))
    groups: list[list[dict]] = []
    current: list[dict] = []
    last_seen_at: float | None = None
    last_center: tuple[float, float] | None = None

    for record in records:
        boxes = record.get("boxes") or []
        if not boxes:
            continue
        # One fixed ROI covers every detected person in this epoch.
        centers = [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2 in boxes]
        center_x = float(np.median([p[0] for p in centers]))
        center_y = float(np.median([p[1] for p in centers]))
        timestamp = float(record["timestamp_sec"])
        split = (
            last_seen_at is not None
            and (
                timestamp - last_seen_at > max_gap
                or (
                    last_center is not None
                    and max(abs(center_x - last_center[0]), abs(center_y - last_center[1])) > recenter_px
                )
            )
        )
        if split and current:
            groups.append(current)
            current = []
        current.append(record)
        last_seen_at = timestamp
        last_center = (center_x, center_y)
    if current:
        groups.append(current)

    quantile = min(0.999, max(0.5, float(cfg.person_mask_roi_quantile)))
    lower_q = 1.0 - quantile
    pad = max(0, int(cfg.person_mask_dilate_px))
    epochs: list[dict] = []
    for records_in_epoch in groups:
        boxes = [box for record in records_in_epoch for box in record.get("boxes") or []]
        if not boxes:
            continue
        x1 = max(0, int(math.floor(_quantile([box[0] for box in boxes], lower_q))) - pad)
        y1 = max(0, int(math.floor(_quantile([box[1] for box in boxes], lower_q))) - pad)
        x2 = min(width, int(math.ceil(_quantile([box[2] for box in boxes], quantile))) + pad)
        y2 = min(height, int(math.ceil(_quantile([box[3] for box in boxes], quantile))) + pad)
        if x2 <= x1 or y2 <= y1:
            continue
        epochs.append({
            "start_sec": float(records_in_epoch[0]["timestamp_sec"]),
            "end_sec": float(records_in_epoch[-1]["timestamp_sec"]),
            "box": (x1, y1, x2, y2),
        })
    return epochs


def _motion_filter_person_boxes(
    frames: list[np.ndarray],
    boxes_by_frame: list[list[tuple[int, int, int, int]]],
    cfg: SampleCacheConfig,
) -> tuple[list[list[tuple[int, int, int, int]]], int]:
    """Keep tracks only after their *box geometry* proves real movement.

    A fixed presenter, portrait, or false-positive character must not be
    masked merely because slide pixels behind the box change.  The former
    implementation compared pixels inside each box, which treated handwriting,
    slide transitions, and compression noise as person motion.
    """
    filtered: list[list[tuple[int, int, int, int]]] = []
    vetoed = 0
    min_center_shift = max(0.0, float(cfg.person_mask_fixed_min_center_shift_px))
    min_area_change = max(0.0, float(cfg.person_mask_fixed_min_area_change_ratio))
    match_iou = max(0.05, float(cfg.person_mask_match_iou_threshold))
    tracks: list[dict] = []
    max_missing_frames = max(
        1,
        int(round(max(0.1, float(cfg.person_mask_roi_max_gap_sec)) * float(cfg.sample_fps))),
    )

    def center(box: tuple[int, int, int, int]) -> tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def area(box: tuple[int, int, int, int]) -> float:
        return max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))

    for frame_index, boxes in enumerate(boxes_by_frame):
        tracks = [
            track
            for track in tracks
            if frame_index - int(track["last_seen_frame"]) <= max_missing_frames
        ]
        valid_boxes: list[tuple[int, int, int, int]] = []
        for bbox in boxes:
            best_track = None
            best_iou = 0.0
            for track in tracks:
                score = _bbox_iou(bbox, track["last"])
                if score > best_iou:
                    best_track = track
                    best_iou = score

            if best_track is None or best_iou < match_iou:
                # A newly seen box needs a second observation before it can
                # prove motion.  This avoids masking static artwork/portraits.
                tracks.append({
                    "anchor": bbox,
                    "last": bbox,
                    "last_seen_frame": frame_index,
                    "confirmed": False,
                })
                vetoed += 1
                continue

            anchor = best_track["anchor"]
            ax, ay = center(anchor)
            bx, by = center(bbox)
            center_shift = max(abs(bx - ax), abs(by - ay))
            area_change = abs(area(bbox) - area(anchor)) / area(anchor)
            best_track["last"] = bbox
            best_track["last_seen_frame"] = frame_index
            if not bool(best_track["confirmed"]) and (
                center_shift >= min_center_shift or area_change >= min_area_change
            ):
                best_track["confirmed"] = True
                # Keep a fresh baseline for diagnostics and a future track
                # split, but retain this confirmed presenter track.
                best_track["anchor"] = bbox

            if bool(best_track["confirmed"]):
                valid_boxes.append(bbox)
            else:
                vetoed += 1
        filtered.append(valid_boxes)
    return filtered, vetoed


def _run_person_mask_task(cache_dir: str, task_dir: str, spec: dict, cfg: SampleCacheConfig) -> str:
    """Run TensorRT person detection over an 8-minute cache task and emit metadata updates."""
    cache_path = Path(cache_dir)
    task_path = Path(task_dir)
    payload = json.loads((cache_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    all_frames = payload.get("frames", [])
    selected = [
        item for item in all_frames
        if float(spec["start_sec"]) - 1e-6 <= float(item["timestamp_sec"]) <= float(spec["end_sec"]) + 1e-6
    ]
    if not selected:
        task_path.write_text(json.dumps({"updates": [], "epochs": 0}), encoding="utf-8")
        return str(task_path)

    cached_boxes_available = all("person_boxes" in item for item in selected)
    height = int(payload.get("cache", {}).get("height") or 0)
    width = int(payload.get("cache", {}).get("width") or 0)
    frames: list[np.ndarray] = []
    if cached_boxes_available:
        boxes_by_frame = [
            [tuple(int(value) for value in box) for box in item.get("person_boxes") or []]
            for item in selected
        ]
        first_index = int(selected[0]["sample_index"]) - 1
        cap = cv2.VideoCapture(str(cache_path / payload["video_filename"]))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open sample cache video: {cache_path / payload['video_filename']}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_index)
        try:
            for _ in selected:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("sample cache video ended before motion veto completed")
                frames.append(frame)
        finally:
            cap.release()
    else:
        from ultralytics import YOLO

        first_index = int(selected[0]["sample_index"]) - 1
        cap = cv2.VideoCapture(str(cache_path / payload["video_filename"]))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open sample cache video: {cache_path / payload['video_filename']}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_index)
        frames: list[np.ndarray] = []
        try:
            for _ in selected:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("sample cache video ended before person mask task completed")
                frames.append(frame)
        finally:
            cap.release()

        model = YOLO(str(cfg.person_mask_engine), task="detect")
        boxes_by_frame = []
        batch_size = max(1, int(cfg.person_mask_batch_size))
        for offset in range(0, len(frames), batch_size):
            boxes_by_frame.extend(_detect_person_boxes(model, frames[offset:offset + batch_size], cfg))
        height, width = frames[0].shape[:2]

    presence_boxes_by_frame = boxes_by_frame
    boxes_by_frame, motion_vetoed = _motion_filter_person_boxes(frames, boxes_by_frame, cfg)
    detection_records = [
        {
            "sample_index": int(item["sample_index"]),
            "timestamp_sec": float(item["timestamp_sec"]),
            "boxes": boxes,
        }
        for item, boxes in zip(selected, boxes_by_frame)
    ]
    epochs = _fixed_box_epochs(detection_records, cfg, width, height)
    masks_dir = cache_path / MASKS_DIRNAME
    masks_dir.mkdir(parents=True, exist_ok=True)
    epoch_entries: list[dict] = []
    for epoch_index, epoch in enumerate(epochs):
        x1, y1, x2, y2 = epoch["box"]
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        filename = f"{MASKS_DIRNAME}/person_fixed_task_{int(spec['task_index']):03d}_epoch_{epoch_index:03d}.npy"
        np.save(cache_path / filename, mask, allow_pickle=False)
        epoch_entries.append({**epoch, "filename": filename})

    updates: list[dict] = []
    core_start = float(spec["core_start_sec"])
    core_end = float(spec["core_end_sec"])
    is_last = bool(spec.get("is_last"))
    for record, presence_boxes in zip(detection_records, presence_boxes_by_frame):
        timestamp = float(record["timestamp_sec"])
        in_core = timestamp >= core_start - 1e-6 and (timestamp <= core_end + 1e-6 if is_last else timestamp < core_end - 1e-6)
        if not in_core:
            continue
        epoch_match = next(
            (
                epoch for epoch in epoch_entries
                if float(epoch["start_sec"]) - max(0.1, float(cfg.person_mask_roi_max_gap_sec)) <= timestamp
                <= float(epoch["end_sec"]) + max(0.1, float(cfg.person_mask_roi_max_gap_sec))
            ),
            None,
        )
        update = {
            "sample_index": int(record["sample_index"]),
            "person_presence_ratio": _presence_ratio_from_boxes(
                presence_boxes, height, width, max(0, int(cfg.person_mask_dilate_px))
            ),
        }
        # An epoch gives the ROI geometry, but a static frame within that
        # epoch must remain unmasked.  Only a frame with confirmed box motion
        # receives the fixed ROI mask.
        if epoch_match is not None and record["boxes"]:
            update["person_mask_filename"] = epoch_match["filename"]
        updates.append(update)

    task_path.write_text(
        json.dumps(
            {"updates": updates, "epochs": len(epoch_entries), "motion_vetoed": motion_vetoed},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return str(task_path)


def materialize_fixed_person_masks(cache_dir: str | Path, cfg: SampleCacheConfig) -> dict:
    """Attach stable fixed-box person masks after chunked sample-cache merge."""
    cache_path = Path(cache_dir)
    manifest_path = cache_path / MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not cfg.person_masks:
        return {"enabled": False, "tasks": 0, "workers": 0, "epochs": 0}
    if not frames:
        return {"enabled": True, "tasks": 0, "workers": 0, "epochs": 0}

    cached_boxes_available = all("person_boxes" in frame for frame in frames)
    engine_path = Path(cfg.person_mask_engine) if cached_boxes_available else _ensure_person_mask_engine(cfg)
    cfg = copy.deepcopy(cfg)
    cfg.person_mask_engine = str(engine_path)
    specs = _person_mask_task_specs(
        frames,
        cfg.person_mask_task_sec,
        cfg.person_mask_task_overlap_sec,
    )
    workers = max(1, min(int(cfg.person_mask_workers), len(specs)))
    log.info(
        "person mask fixed-box start: tasks=%s task_sec=%.1fs overlap=%.1fs workers=%s engine=%s cached_boxes=%s",
        len(specs),
        cfg.person_mask_task_sec,
        cfg.person_mask_task_overlap_sec,
        workers,
        engine_path,
        cached_boxes_available,
    )

    masks_dir = cache_path / MASKS_DIRNAME
    presence_dir = cache_path / PRESENCE_MASKS_DIRNAME
    shutil.rmtree(masks_dir, ignore_errors=True)
    shutil.rmtree(presence_dir, ignore_errors=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    task_dir = cache_path / "person_mask_tasks"
    shutil.rmtree(task_dir, ignore_errors=True)
    task_dir.mkdir(parents=True, exist_ok=True)

    result_paths: list[Path] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
        futures = {
            executor.submit(_run_person_mask_task, str(cache_path), str(task_dir / f"task_{int(spec['task_index']):03d}.json"), spec, cfg): spec
            for spec in specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            result_path = Path(future.result())
            result_paths.append(result_path)
            task_payload = json.loads(result_path.read_text(encoding="utf-8"))
            log.info(
                "  [person mask task done] %s/%s idx=%s updates=%s epochs=%s motion_vetoed=%s",
                len(result_paths),
                len(specs),
                int(spec["task_index"]) + 1,
                len(task_payload.get("updates", [])),
                task_payload.get("epochs", 0),
                task_payload.get("motion_vetoed", 0),
            )

    updates_by_index: dict[int, dict] = {}
    epoch_count = 0
    for result_path in result_paths:
        task_payload = json.loads(result_path.read_text(encoding="utf-8"))
        epoch_count += int(task_payload.get("epochs", 0) or 0)
        for update in task_payload.get("updates", []):
            updates_by_index[int(update["sample_index"])] = update
    for frame in frames:
        frame.pop("person_mask_filename", None)
        frame.pop("person_presence_mask_filename", None)
        frame.pop("person_mask_inherited", None)
        frame.pop("person_mask_inherited_distance", None)
        update = updates_by_index.get(int(frame["sample_index"]))
        if not update:
            frame["person_presence_ratio"] = 0.0
            continue
        frame["person_presence_ratio"] = float(update.get("person_presence_ratio", 0.0) or 0.0)
        if update.get("person_mask_filename"):
            frame["person_mask_filename"] = str(update["person_mask_filename"])

    payload["person_masks"] = {
        "enabled": True,
        "dirname": MASKS_DIRNAME,
        "presence_dirname": None,
        "model": str(cfg.person_mask_model),
        "engine": str(engine_path),
        "mode": "fixed_box_epoch",
        "task_sec": float(cfg.person_mask_task_sec),
        "task_overlap_sec": float(cfg.person_mask_task_overlap_sec),
        "workers": workers,
        "roi_quantile": float(cfg.person_mask_roi_quantile),
        "roi_recenter_px": int(cfg.person_mask_roi_recenter_px),
        "roi_max_gap_sec": float(cfg.person_mask_roi_max_gap_sec),
        "epoch_count": epoch_count,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(
        "person mask fixed-box done: tasks=%s workers=%s epochs=%s updated_samples=%s",
        len(specs), workers, epoch_count, len(updates_by_index),
    )
    return {"enabled": True, "tasks": len(specs), "workers": workers, "epochs": epoch_count}


def _create_chunk_worker(input_path: str, chunk_dir: str, cfg: SampleCacheConfig, spec: dict) -> str:
    worker_cfg = copy.deepcopy(cfg)
    # The parent process performs the single-model presence gate. Workers only
    # receive the result and never load a gate model themselves.
    active_ranges = spec.get("person_mask_active_ranges")
    worker_cfg.person_mask_active_ranges = active_ranges
    worker_cfg.person_masks = bool(active_ranges) if cfg.person_masks else False
    log.info(
        "chunk person gate result: chunk=%s active=%s ranges=%s",
        int(spec["chunk_index"]),
        worker_cfg.person_masks,
        len(active_ranges or []),
    )
    create_sample_cache_range(
        input_path,
        chunk_dir,
        worker_cfg,
        float(spec["start_sec"]),
        float(spec["end_sec"]),
        chunk_index=int(spec["chunk_index"]),
        core_start_sec=float(spec["core_start_sec"]),
        core_end_sec=float(spec["core_end_sec"]),
    )
    return str(Path(chunk_dir) / MANIFEST_FILENAME)


def _copy_optional_mask(
    src_cache_dir: Path,
    dst_cache_dir: Path,
    src_filename: str | None,
    dst_rel_filename: str,
) -> str | None:
    if not src_filename:
        return None
    src = src_cache_dir / src_filename
    if not src.exists():
        return None
    dst = dst_cache_dir / dst_rel_filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    # The chunk cache lives under a TemporaryDirectory and disappears after the
    # merge, so final masks still need to be materialized into output_dir.
    # Prefer hardlinking when source/destination are on the same filesystem;
    # fall back to copy2 for cross-device filesystems, Docker bind mounts, etc.
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst_rel_filename


def _write_merge_part_worker(
    chunk_manifest_path: str,
    spec: dict,
    part_path: str,
    cfg: SampleCacheConfig,
    sampled_fps: float,
    resize_width: int,
    cached_height: int,
) -> dict:
    """Write one merge part AVI from one chunk cache.

    The worker reads the temporary chunk sampled_frames.avi and writes only the
    non-overlap core samples into part_XXX.avi. It also returns the selected
    manifest frame items in exactly the same order as frames written to the part
    file. Final sample_index re-numbering happens in the parent merge process.
    """

    import time

    started_at = time.perf_counter()
    chunk_manifest = Path(chunk_manifest_path)
    chunk_dir = chunk_manifest.parent
    part = Path(part_path)
    part.parent.mkdir(parents=True, exist_ok=True)
    part.unlink(missing_ok=True)

    payload = json.loads(chunk_manifest.read_text(encoding="utf-8"))
    chunk_video_path = chunk_dir / payload["video_filename"]

    cap = cv2.VideoCapture(str(chunk_video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open chunk sampled video: {chunk_video_path}")

    writer = cv2.VideoWriter(
        str(part),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(sampled_fps),
        (int(resize_width), int(cached_height)),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open merge part writer: {part}")

    selected_frames: list[dict] = []
    skipped_overlap = 0
    fallback_phash_count = 0
    video_read_elapsed = 0.0
    video_write_elapsed = 0.0
    metric_fallback_elapsed = 0.0

    try:
        core_start = float(spec["core_start_sec"])
        core_end = float(spec["core_end_sec"])
        is_last = bool(spec.get("is_last"))

        for item in payload.get("frames", []):
            read_t0 = time.perf_counter()
            ret, frame = cap.read()
            video_read_elapsed += time.perf_counter() - read_t0
            if not ret or frame is None:
                raise RuntimeError(f"Chunk sampled video ended early: {chunk_video_path}")

            timestamp = float(item["timestamp_sec"])
            if timestamp < core_start - 1e-6:
                skipped_overlap += 1
                continue
            if is_last:
                if timestamp > core_end + 1e-6:
                    skipped_overlap += 1
                    continue
            elif timestamp >= core_end - 1e-6:
                skipped_overlap += 1
                continue

            write_t0 = time.perf_counter()
            writer.write(frame)
            video_write_elapsed += time.perf_counter() - write_t0

            selected = dict(item)
            selected["frame_no"] = int(item["frame_no"])
            selected["timestamp_sec"] = round(float(item["timestamp_sec"]), 6)

            # In normal chunk caches this already exists. Keep a fallback so old
            # caches or partially written manifests still remain usable.
            if selected.get("phash_int") is None:
                metric_t0 = time.perf_counter()
                selected["phash_int"] = compute_phash_int(to_decision_frame(frame))
                metric_fallback_elapsed += time.perf_counter() - metric_t0
                fallback_phash_count += 1

            selected_frames.append(selected)
    finally:
        cap.release()
        writer.release()

    return {
        "chunk_index": int(spec["chunk_index"]),
        "chunk_dir": str(chunk_dir),
        "part_path": str(part),
        "selected_count": len(selected_frames),
        "skipped_overlap": skipped_overlap,
        "fallback_phash_count": fallback_phash_count,
        "video_read_elapsed": video_read_elapsed,
        "video_write_elapsed": video_write_elapsed,
        "metric_fallback_elapsed": metric_fallback_elapsed,
        "elapsed": time.perf_counter() - started_at,
        "frames": selected_frames,
    }


def _ffmpeg_concat_escape(path: Path) -> str:
    return str(path).replace("'", "'\\''")


def _concat_merge_parts_ffmpeg(
    part_paths: list[Path],
    list_path: Path,
    output_video_path: Path,
    *,
    reencode: bool = False,
) -> None:
    # ffmpeg concat resolves relative `file ...` entries relative to the list
    # file location. Use absolute paths to avoid duplicated path prefixes when
    # output_dir itself is relative, e.g. output_slides_staged/sample_cache.
    list_path = list_path.resolve()
    output_video_path = output_video_path.resolve()
    list_path.parent.mkdir(parents=True, exist_ok=True)

    with open(list_path, "w", encoding="utf-8") as f:
        for part in part_paths:
            f.write(f"file '{_ffmpeg_concat_escape(part.resolve())}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
    ]
    if reencode:
        # AVI/MJPEG stream copy can preserve discontinuous chunk timestamps.
        # Re-encoding normalizes the timeline while the expensive source decode
        # and per-chunk part generation remain fully parallel.
        cmd.extend(["-c:v", "mjpeg", "-q:v", "3", "-an"])
    else:
        cmd.extend(["-c", "copy"])
    cmd.append(str(output_video_path))
    subprocess.run(cmd, check=True)


def _concat_merge_parts_opencv(
    part_paths: list[Path],
    output_video_path: Path,
    sampled_fps: float,
    resize_width: int,
    cached_height: int,
) -> int:
    """Fallback concat path if ffmpeg stream-copy concat fails."""

    output_video_path.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(sampled_fps),
        (int(resize_width), int(cached_height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open fallback merged sample cache writer: {output_video_path}")

    written = 0
    try:
        for part in part_paths:
            cap = cv2.VideoCapture(str(part))
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open merge part video: {part}")
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    writer.write(frame)
                    written += 1
            finally:
                cap.release()
    finally:
        writer.release()
    return written


def _merge_chunk_caches(
    input_path: str,
    output_dir: str,
    cfg: SampleCacheConfig,
    manifest_paths: list[Path],
    specs: list[dict],
) -> dict:
    import time

    video_meta = read_video_metadata(input_path)
    output_path = Path(output_dir)
    # Chunk workers already write MJPEG parts in parallel.  Keep those parts as
    # the final cache layout instead of concatenating them into one AVI: ffmpeg
    # concat can alter/drop AVI frames at timestamp boundaries.
    _, merged_manifest_path, _, _, _ = _prepare_output_dirs(output_path, cfg)

    cached_height = int(video_meta["height"] * (cfg.resize_width / video_meta["width"]))
    sampled_fps = float(cfg.sample_fps)
    merge_started_at = time.perf_counter()

    parts_dir = output_path / "segments"
    if parts_dir.exists():
        shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    total_chunk_frames = 0
    for chunk_manifest_path in manifest_paths:
        try:
            payload = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
            total_chunk_frames += len(payload.get("frames", []))
        except Exception:
            pass

    requested_workers = int(
        os.getenv(
            "GRAPHLEC_SAMPLE_CACHE_MERGE_WORKERS",
            os.getenv("GRAPHLEC_SAMPLE_CACHE_CHUNK_WORKERS", "2"),
        )
    )
    merge_workers = max(1, min(requested_workers, len(specs) if specs else 1))

    log.info(
        "sample cache merge start: chunks=%s candidate_frames=%s output=%s merge_workers=%s",
        len(specs),
        total_chunk_frames,
        output_path,
        merge_workers,
    )

    part_started_at = time.perf_counter()
    part_by_index: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=merge_workers) as executor:
        future_map = {}
        for chunk_manifest_path, spec in zip(manifest_paths, specs):
            chunk_index = int(spec["chunk_index"])
            part_path = parts_dir / f"part_{chunk_index:03d}.avi"
            future = executor.submit(
                _write_merge_part_worker,
                str(chunk_manifest_path),
                dict(spec),
                str(part_path),
                cfg,
                sampled_fps,
                int(cfg.resize_width),
                int(cached_height),
            )
            future_map[future] = spec
            log.info(
                "  [sample merge part submit %s/%s] core=%.1fs~%.1fs",
                chunk_index + 1,
                len(specs),
                float(spec["core_start_sec"]),
                float(spec["core_end_sec"]),
            )

        completed = 0
        for future in as_completed(future_map):
            result = future.result()
            chunk_index = int(result["chunk_index"])
            part_by_index[chunk_index] = result
            completed += 1
            log.info(
                "  [sample merge part done %s/%s | idx=%s] selected=%s skipped_overlap=%s elapsed=%.1fs",
                completed,
                len(specs),
                chunk_index + 1,
                int(result["selected_count"]),
                int(result["skipped_overlap"]),
                float(result["elapsed"]),
            )

    ordered_indices = sorted(part_by_index)
    ordered_results = [part_by_index[idx] for idx in ordered_indices]
    part_paths = [Path(result["part_path"]) for result in ordered_results]
    part_wall_elapsed = time.perf_counter() - part_started_at

    if not part_paths:
        raise RuntimeError("No merge part videos were produced")

    # No final concat.  Each part is an independently valid, overlap-trimmed
    # cache segment.  The manifest below maps every global sample to a local
    # segment frame, so readers never infer identity from AVI timestamps.
    concat_mode = "none-segmented"
    concat_elapsed = 0.0

    manifest_started_at = time.perf_counter()
    frames: list[dict] = []
    seen_frame_nos: set[int] = set()
    sample_index = 0
    duplicate_frame_count = 0

    mask_workers_requested = int(
        os.getenv(
            "GRAPHLEC_SAMPLE_CACHE_MERGE_MASK_WORKERS",
            str(max(1, merge_workers)),
        )
    )
    mask_workers = max(1, mask_workers_requested)
    mask_copy_started_at = time.perf_counter()
    mask_future_map = {}

    def _apply_presence_result(frame_record: dict, item: dict, rel_filename: str | None) -> None:
        if not rel_filename:
            return
        frame_record["person_presence_mask_filename"] = rel_filename
        if item.get("person_presence_ratio") is not None:
            frame_record["person_presence_ratio"] = item.get("person_presence_ratio")

    def _apply_person_result(frame_record: dict, item: dict, rel_filename: str | None) -> None:
        # Fixed-box person masking intentionally does not emit a separate
        # per-frame presence-mask file. Its presence ratio therefore travels
        # with the person-mask record and must survive the chunk merge.
        if item.get("person_presence_ratio") is not None:
            frame_record["person_presence_ratio"] = item.get("person_presence_ratio")
        if not rel_filename:
            return
        frame_record["person_mask_filename"] = rel_filename
        if item.get("person_mask_inherited"):
            frame_record["person_mask_inherited"] = True
            frame_record["person_mask_inherited_distance"] = item.get("person_mask_inherited_distance")

    def _schedule_or_copy_mask(
        executor: ThreadPoolExecutor | None,
        frame_record: dict,
        item: dict,
        chunk_dir: Path,
        src_key: str,
        dst_rel_filename: str,
        kind: str,
    ) -> None:
        src_filename = item.get(src_key)
        if not src_filename:
            return
        if executor is None:
            rel = _copy_optional_mask(chunk_dir, output_path, src_filename, dst_rel_filename)
            if kind == "presence":
                _apply_presence_result(frame_record, item, rel)
            else:
                _apply_person_result(frame_record, item, rel)
            return

        future = executor.submit(
            _copy_optional_mask,
            chunk_dir,
            output_path,
            src_filename,
            dst_rel_filename,
        )
        mask_future_map[future] = (kind, frame_record, item)

    executor_ctx = ThreadPoolExecutor(max_workers=mask_workers) if mask_workers > 1 else None
    try:
        if executor_ctx is not None:
            executor_ctx.__enter__()

        for result in ordered_results:
            chunk_dir = Path(result["chunk_dir"])
            segment_index = int(result["chunk_index"])
            segment_rel = f"segments/part_{segment_index:03d}.avi"
            for segment_frame_index, item in enumerate(result.get("frames", [])):
                frame_no = int(item["frame_no"])
                if frame_no in seen_frame_nos:
                    duplicate_frame_count += 1
                    raise RuntimeError(
                        f"Duplicate frame_no during two-stage merge: frame_no={frame_no}. "
                        "This would desync sampled_frames.avi and sampled_manifest.json."
                    )
                seen_frame_nos.add(frame_no)

                sample_index += 1
                frame_record = {
                    "sample_index": sample_index,
                    "frame_no": frame_no,
                    "timestamp_sec": round(float(item["timestamp_sec"]), 6),
                    "cache_segment_index": segment_index,
                    "cache_segment_filename": segment_rel,
                    "cache_segment_frame_index": segment_frame_index,
                    "phash_int": item.get("phash_int"),
                    "prev_mse": item.get("prev_mse"),
                    "prev_hash_dist": item.get("prev_hash_dist"),
                }

                _schedule_or_copy_mask(
                    executor_ctx,
                    frame_record,
                    item,
                    chunk_dir,
                    "person_presence_mask_filename",
                    f"{PRESENCE_MASKS_DIRNAME}/person_presence_mask_{sample_index:06d}.npy",
                    "presence",
                )
                _schedule_or_copy_mask(
                    executor_ctx,
                    frame_record,
                    item,
                    chunk_dir,
                    "person_mask_filename",
                    f"{MASKS_DIRNAME}/person_mask_{sample_index:06d}.npy",
                    "person",
                )

                frames.append(frame_record)

        if executor_ctx is not None:
            completed = 0
            total = len(mask_future_map)
            for future in as_completed(mask_future_map):
                kind, frame_record, item = mask_future_map[future]
                rel = future.result()
                if kind == "presence":
                    _apply_presence_result(frame_record, item, rel)
                else:
                    _apply_person_result(frame_record, item, rel)
                completed += 1
                if completed % 2000 == 0 or completed == total:
                    log.info(
                        "  [sample merge mask materialize] completed=%s/%s",
                        completed,
                        total,
                    )
    finally:
        if executor_ctx is not None:
            executor_ctx.__exit__(None, None, None)

    mask_materialize_elapsed = time.perf_counter() - mask_copy_started_at

    post_started_at = time.perf_counter()
    # The frames are already ordered by part order, but keep an explicit stable
    # sort to preserve the previous merge behavior and guard against future
    # changes in chunk scheduling.
    frames.sort(key=lambda x: (float(x["timestamp_sec"]), int(x["frame_no"])))
    for idx, frame in enumerate(frames, start=1):
        frame["sample_index"] = idx

    mask_fill_gap_samples = max(0, int(round(float(cfg.person_mask_fill_gap_sec) * sampled_fps)))
    inherited_masks = _fill_short_person_mask_gaps(frames, max_gap=mask_fill_gap_samples)
    postprocess_elapsed = time.perf_counter() - post_started_at

    part_video_read_elapsed = sum(float(result.get("video_read_elapsed", 0.0)) for result in ordered_results)
    part_video_write_elapsed = sum(float(result.get("video_write_elapsed", 0.0)) for result in ordered_results)
    metric_fallback_elapsed = sum(float(result.get("metric_fallback_elapsed", 0.0)) for result in ordered_results)
    fallback_phash_count = sum(int(result.get("fallback_phash_count", 0)) for result in ordered_results)
    skipped_overlap_frames = sum(int(result.get("skipped_overlap", 0)) for result in ordered_results)

    person_masks_payload = {
        "enabled": bool(cfg.person_masks),
        "dirname": MASKS_DIRNAME,
        # Fixed-box mode stores the final mask only. Do not advertise a
        # presence directory when no frame references one.
        "presence_dirname": (
            PRESENCE_MASKS_DIRNAME
            if any(frame.get("person_presence_mask_filename") for frame in frames)
            else None
        ),
        "model": cfg.person_mask_model if cfg.person_masks else None,
        "coordinate_space": "sample_cache_frame",
        "mode": "moving_person_only",
        "static_diff_threshold": cfg.person_mask_static_diff_threshold,
        "static_changed_ratio_threshold": cfg.person_mask_static_changed_ratio_threshold,
        "match_iou_threshold": cfg.person_mask_match_iou_threshold,
        "fill_gap_sec": cfg.person_mask_fill_gap_sec,
        "fill_gap_samples": mask_fill_gap_samples,
        "inherited_count": inherited_masks,
        "preview_dirname": None,
        "preview_count": 0,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input_path": input_path,
        "video_filename": None,
        "config": asdict(cfg),
        "person_masks": person_masks_payload,
        "source": video_meta,
        "cache": {
            "sampled_fps": sampled_fps,
            "width": cfg.resize_width,
            "height": cached_height,
            "sample_count": len(frames),
            "chunked": True,
            "chunk_count": len(specs),
            "layout": "segmented",
            "merge_mode": "parallel_parts_no_concat",
            "merge_workers": merge_workers,
            "merge_mask_workers": mask_workers,
            "concat_mode": concat_mode,
            "segments_dirname": "segments",
            "segment_count": len(ordered_results),
        },
        "chunks": specs,
        "frames": frames,
    }

    manifest_write_started_at = time.perf_counter()
    with open(merged_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    manifest_write_elapsed = time.perf_counter() - manifest_write_started_at
    manifest_build_elapsed = time.perf_counter() - manifest_started_at

    log.info(
        "sample cache merge timings: part_wall=%.1fs part_video_read_sum=%.1fs "
        "part_video_write_sum=%.1fs concat=%.1fs mask_materialize=%.1fs "
        "metric_fallback=%.1fs postprocess=%.1fs manifest_build=%.1fs "
        "manifest_write=%.1fs skipped_overlap=%s skipped_duplicate=%s "
        "fallback_phash=%s concat_mode=%s manifest_frames=%s",
        part_wall_elapsed,
        part_video_read_elapsed,
        part_video_write_elapsed,
        concat_elapsed,
        mask_materialize_elapsed,
        metric_fallback_elapsed,
        postprocess_elapsed,
        manifest_build_elapsed,
        manifest_write_elapsed,
        skipped_overlap_frames,
        duplicate_frame_count,
        fallback_phash_count,
        concat_mode,
        len(frames),
    )
    log.info(
        "chunked sample cache merged: chunks=%s samples=%s output=%s elapsed=%.1fs",
        len(specs),
        len(frames),
        output_path,
        time.perf_counter() - merge_started_at,
    )
    return manifest


def _verify_sample_cache_alignment(cache_dir: str | Path, manifest: dict) -> dict:
    """Check that decoded cache frames still match their manifest entries.

    The manifest is produced before chunk merge. A damaged or timestamp-shifted
    merged AVI can otherwise cause every downstream stage to associate the
    right frame number with the wrong pixels.
    """
    if os.getenv("GRAPHLEC_SAMPLE_CACHE_VERIFY_ALIGNMENT", "1").strip().lower() in {
        "0", "false", "no", "off",
    }:
        return {"enabled": False, "ok": True, "checked": 0, "mismatches": []}

    frames = list(manifest.get("frames", []) or [])
    if not frames:
        return {"enabled": True, "ok": False, "checked": 0, "mismatches": ["empty_manifest"]}

    stride = max(1, int(os.getenv("GRAPHLEC_SAMPLE_CACHE_VERIFY_STRIDE", "25")))
    # Segment AVI is MJPEG/lossy. Near an end-of-stream keyframe its pHash can
    # drift by roughly 12 bits even though the local frame mapping is correct.
    # Keep this below the 26-bit mismatch observed for an actually shifted
    # merged cache, while accepting codec noise.
    max_hash_distance = max(0, int(os.getenv("GRAPHLEC_SAMPLE_CACHE_VERIFY_MAX_HASH_DISTANCE", "14")))
    target_positions = set(range(0, len(frames), stride))
    target_positions.add(len(frames) - 1)

    checked = 0
    mismatches: list[dict] = []
    segmented = is_segmented_sample_cache(manifest)
    verify_started_at = time.perf_counter()
    log.info(
        "sample cache alignment verify start: targets=%s samples=%s mode=%s",
        len(target_positions),
        len(frames),
        "segment-seek" if segmented else "legacy-sequential",
    )
    try:
        if segmented:
            frame_iter = iter_sample_cache_selected_positions(cache_dir, sorted(target_positions))
        else:
            frame_iter = iter_sample_cache_range(cache_dir, 0, len(frames))
        for position, frame_info, frame in frame_iter:
            if not segmented and position not in target_positions:
                continue
            checked += 1
            expected = frame_info.get("phash_int")
            if expected is None:
                continue
            actual = compute_phash_int(to_decision_frame(frame))
            distance = phash_distance_int(int(expected), actual)
            if distance > max_hash_distance:
                mismatches.append({
                    "sample_index": int(frame_info.get("sample_index", position + 1)),
                    "frame_no": int(frame_info.get("frame_no", 0) or 0),
                    "distance": int(distance),
                })
                # One mismatch is sufficient to reject the cache. Keep the
                # validation overhead bounded on long recordings.
                break
    except Exception as exc:
        mismatches.append({"reason": f"cache_read_failed:{exc}"})

    log.info(
        "sample cache alignment verify done: checked=%s mismatches=%s elapsed=%.1fs",
        checked,
        len(mismatches),
        time.perf_counter() - verify_started_at,
    )

    return {
        "enabled": True,
        "ok": not mismatches,
        "checked": checked,
        "stride": stride,
        "max_hash_distance": max_hash_distance,
        "mismatches": mismatches,
    }

def create_sample_cache_chunked(
    input_path: str,
    output_dir: str,
    cfg: SampleCacheConfig | None = None,
    *,
    chunk_sec: float | None = None,
    overlap_sec: float | None = None,
    workers: int | None = None,
) -> dict:
    cfg = copy.deepcopy(cfg or SampleCacheConfig())
    cfg.sample_every = max(1, int(cfg.sample_every))
    cfg.sample_fps = max(0.1, float(cfg.sample_fps))
    cfg.resize_width = max(160, int(cfg.resize_width))
    chunk_sec = float(chunk_sec if chunk_sec is not None else os.getenv("GRAPHLEC_SAMPLE_CACHE_CHUNK_SEC", "300"))
    overlap_sec = float(overlap_sec if overlap_sec is not None else os.getenv("GRAPHLEC_SAMPLE_CACHE_CHUNK_OVERLAP_SEC", "30"))
    requested_workers = int(workers if workers is not None else os.getenv("GRAPHLEC_SAMPLE_CACHE_CHUNK_WORKERS", "2"))

    video_meta = read_video_metadata(input_path)
    duration = float(video_meta.get("duration_sec") or 0.0)
    if duration <= 0.0 and video_meta.get("frame_count") and video_meta.get("fps"):
        duration = float(video_meta["frame_count"]) / float(video_meta["fps"])

    specs = _chunk_specs(duration, chunk_sec, overlap_sec)
    if not specs:
        log.info("sample cache chunking skipped: duration=%.1fs <= chunk_sec=%.1fs", duration, chunk_sec)
        return create_sample_cache(input_path, output_dir, cfg)

    workers = max(1, min(requested_workers, len(specs)))
    log.info(
        "sample cache chunked start: duration=%.1fs chunk_sec=%.1fs overlap=%.1fs workers=%s chunks=%s",
        duration,
        chunk_sec,
        overlap_sec,
        workers,
        len(specs),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    temp_root, storage_mode = _sample_cache_temp_root(output_path)
    log.info(
        "sample cache temporary storage: mode=%s root=%s total_memory_gb=%.1f",
        storage_mode,
        temp_root,
        ((_system_memory_bytes() or 0) / 1024**3),
    )

    # Run all seek-based gate checks in the parent with one YOLO instance.
    # This prevents every chunk worker from allocating its own TensorRT context
    # merely to discover that its chunk has no person-mask work.
    if cfg.person_masks:
        gate_model_source = cfg.person_mask_engine if Path(cfg.person_mask_engine).exists() else cfg.person_mask_model
        log.info(
            "person presence gate global start: chunks=%s model=%s one_model=true",
            len(specs),
            gate_model_source,
        )
        gate_model = _load_person_model(gate_model_source)
        if gate_model is None:
            log.warning("person presence gate global disabled: model load failed")
            for spec in specs:
                spec["person_mask_active_ranges"] = []
        else:
            try:
                for spec in specs:
                    active_ranges = _run_person_presence_gate(
                        input_path,
                        cfg,
                        video_meta,
                        start_sec=float(spec["start_sec"]),
                        end_sec=float(spec["end_sec"]),
                        model=gate_model,
                    )
                    spec["person_mask_active_ranges"] = active_ranges
                    log.info(
                        "chunk person gate: chunk=%s active=%s ranges=%s",
                        int(spec["chunk_index"]),
                        bool(active_ranges),
                        len(active_ranges),
                    )
            finally:
                del gate_model
                _empty_torch_cuda_cache()
                gc.collect()
        log.info(
            "person presence gate global done: active_chunks=%s/%s",
            sum(bool(spec.get("person_mask_active_ranges")) for spec in specs),
            len(specs),
        )
    else:
        for spec in specs:
            spec["person_mask_active_ranges"] = []
    with tempfile.TemporaryDirectory(prefix="verilec_sample_cache_chunks_", dir=str(temp_root)) as tmp_root:
        tmp_root_path = Path(tmp_root)
        manifest_by_index: dict[int, Path] = {}
        person_mask_futures = []
        person_executor = ThreadPoolExecutor(max_workers=1) if cfg.person_masks else None
        executor_kwargs = {"max_workers": workers}
        if cfg.person_masks:
            executor_kwargs["mp_context"] = mp.get_context("spawn")
        try:
            with ProcessPoolExecutor(**executor_kwargs) as executor:
                future_map = {}
                for spec in specs:
                    chunk_dir = tmp_root_path / f"chunk_{int(spec['chunk_index']):03d}"
                    chunk_dir.mkdir(parents=True, exist_ok=True)
                    log.info(
                        "  [sample chunk start %s/%s] range=%.1fs~%.1fs core=%.1fs~%.1fs",
                        int(spec["chunk_index"]) + 1,
                        len(specs),
                        spec["start_sec"],
                        spec["end_sec"],
                        spec["core_start_sec"],
                        spec["core_end_sec"],
                    )
                    chunk_cfg = copy.deepcopy(cfg)
                    chunk_cfg.person_masks = bool(cfg.person_masks)
                    future = executor.submit(_create_chunk_worker, input_path, str(chunk_dir), chunk_cfg, spec)
                    future_map[future] = spec

                pending = set(future_map.keys())
                completed = 0
                while pending:
                    done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                    if not done:
                        log.info("  [sample chunk waiting] completed=%s/%s running=%s", completed, len(specs), len(pending))
                        continue
                    for future in done:
                        spec = future_map[future]
                        manifest_path = Path(future.result())
                        manifest_by_index[int(spec["chunk_index"])] = manifest_path
                        completed += 1
                        chunk_person_masks_enabled = False
                        try:
                            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                            chunk_person_masks_enabled = bool(
                                payload.get("person_masks", {}).get("enabled")
                            )
                            log.info(
                                "  [sample chunk done %s/%s | idx=%s] samples=%s backend=%s",
                                completed,
                                len(specs),
                                int(spec["chunk_index"]) + 1,
                                payload.get("cache", {}).get("sample_count"),
                                payload.get("cache", {}).get("decode_backend") or payload.get("decode_backend"),
                            )
                        except Exception:
                            log.info("  [sample chunk done %s/%s | idx=%s]", completed, len(specs), int(spec["chunk_index"]) + 1)

                        if person_executor is not None and chunk_person_masks_enabled:
                            person_dir = str(Path(manifest_path).parent)
                            log.info(
                                "  [person mask submit chunk %s/%s | idx=%s] cache=%s",
                                completed,
                                len(specs),
                                int(spec["chunk_index"]) + 1,
                                person_dir,
                            )
                            person_mask_futures.append(
                                person_executor.submit(materialize_fixed_person_masks, person_dir, cfg)
                            )

                if person_mask_futures:
                    person_completed = 0
                    for future in as_completed(person_mask_futures):
                        result = future.result()
                        person_completed += 1
                        log.info(
                            "  [person mask done %s/%s] tasks=%s workers=%s epochs=%s",
                            person_completed,
                            len(person_mask_futures),
                            result.get("tasks", 0),
                            result.get("workers", 0),
                            result.get("epochs", 0),
                        )
        finally:
            if person_executor is not None:
                person_executor.shutdown(wait=True, cancel_futures=False)

        ordered_manifests = [manifest_by_index[idx] for idx in sorted(manifest_by_index)]
        ordered_specs = [specs[idx] for idx in sorted(manifest_by_index)]
        merged = _merge_chunk_caches(input_path, output_dir, cfg, ordered_manifests, ordered_specs)

        alignment = _verify_sample_cache_alignment(output_dir, merged)
        if alignment["ok"]:
            log.info(
                "sample cache alignment verified: checked=%s stride=%s",
                alignment["checked"],
                alignment.get("stride"),
            )
            return merged

        # A bad cache cannot be handed to scene/OCR/VLM stages. Do not silently
        # disable chunk parallelism here: the default concat path re-encodes
        # the already parallel-produced parts. A remaining mismatch means the
        # cache is unsafe and must fail visibly.
        raise RuntimeError(
            "sample cache alignment mismatch after parallel chunk merge: "
            f"checked={alignment['checked']} details={alignment['mismatches']}"
        )


def load_sample_cache(cache_dir: str | Path) -> dict:
    manifest_path = Path(cache_dir) / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Sample cache manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_segmented_sample_cache(manifest: dict) -> bool:
    return str(manifest.get("cache", {}).get("layout") or "").lower() == "segmented"


def iter_sample_cache_selected_positions(
    cache_dir: str | Path,
    positions: list[int],
) -> Iterator[tuple[int, dict, np.ndarray]]:
    """Read sparse global positions, grouping exact seeks by cache segment.

    This is for integrity checks and other sparse consumers.  Each segment is
    independently written MJPEG, so local frame seeking does not cross a
    concat boundary.  Normal analysis ranges should use the sequential range
    iterator below instead.
    """
    cache_path = Path(cache_dir)
    manifest = load_sample_cache(cache_path)
    frames = list(manifest.get("frames", []) or [])
    total = len(frames)
    selected = sorted({int(pos) for pos in positions if 0 <= int(pos) < total})
    if not selected:
        return

    if not is_segmented_sample_cache(manifest):
        for pos, frame_info, frame in iter_sample_cache_range(cache_path, selected[0], selected[-1] + 1):
            if pos in set(selected):
                yield pos, frame_info, frame
        return

    by_segment: dict[str, list[tuple[int, dict, int]]] = {}
    for pos in selected:
        frame_info = frames[pos]
        filename = frame_info.get("cache_segment_filename")
        local_index = frame_info.get("cache_segment_frame_index")
        if not filename or local_index is None:
            raise RuntimeError(
                f"Segmented cache frame has no segment mapping: sample_index={frame_info.get('sample_index')}"
            )
        by_segment.setdefault(str(filename), []).append((pos, frame_info, int(local_index)))

    for filename, entries in by_segment.items():
        segment_path = cache_path / filename
        cap = cv2.VideoCapture(str(segment_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open sample cache segment: {segment_path}")
        try:
            for pos, frame_info, local_index in entries:
                cap.set(cv2.CAP_PROP_POS_FRAMES, local_index)
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise RuntimeError(
                        f"Sample cache segment seek/read failed: {filename} local={local_index}"
                    )
                yield pos, frame_info, frame
        finally:
            cap.release()


def iter_sample_cache_range(
    cache_dir: str | Path,
    start_pos: int,
    end_pos: int,
) -> Iterator[tuple[int, dict, np.ndarray]]:
    """Yield global cache positions from legacy or segmented cache storage.

    A segmented cache stores overlap-trimmed chunk AVIs independently.  Every
    manifest frame carries its exact segment path and local frame index, so no
    reader relies on an AVI concat timestamp timeline.
    """
    cache_path = Path(cache_dir)
    manifest = load_sample_cache(cache_path)
    frames = list(manifest.get("frames", []) or [])
    total = len(frames)
    start_pos = max(0, min(int(start_pos), total))
    end_pos = max(start_pos, min(int(end_pos), total))
    if start_pos >= end_pos:
        return

    if not is_segmented_sample_cache(manifest):
        video_path = cache_path / str(manifest.get("video_filename") or VIDEO_FILENAME)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open sampled cache video: {video_path}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
            for pos in range(start_pos, end_pos):
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise RuntimeError(f"Sample cache video ended early at pos={pos}: {video_path}")
                yield pos, frames[pos], frame
        finally:
            cap.release()
        return

    cap: cv2.VideoCapture | None = None
    current_segment: str | None = None
    next_local_index = 0
    try:
        for pos in range(start_pos, end_pos):
            frame_info = frames[pos]
            segment_filename = frame_info.get("cache_segment_filename")
            local_index = frame_info.get("cache_segment_frame_index")
            if not segment_filename or local_index is None:
                raise RuntimeError(
                    f"Segmented cache frame has no segment mapping: sample_index={frame_info.get('sample_index')}"
                )
            segment_filename = str(segment_filename)
            local_index = int(local_index)
            if segment_filename != current_segment or local_index < next_local_index:
                if cap is not None:
                    cap.release()
                segment_path = cache_path / segment_filename
                cap = cv2.VideoCapture(str(segment_path))
                if not cap.isOpened():
                    raise FileNotFoundError(f"Cannot open sample cache segment: {segment_path}")
                current_segment = segment_filename
                next_local_index = 0

            # Decode forward from the segment start rather than relying on
            # random AVI seeks.  A segment is only one core chunk (~2400
            # samples), and normal range consumers remain sequential.
            while next_local_index <= local_index:
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise RuntimeError(
                        f"Sample cache segment ended early: {current_segment} local={next_local_index}"
                    )
                decoded_local_index = next_local_index
                next_local_index += 1
            if decoded_local_index != local_index:
                raise RuntimeError(f"Segment cache decode index mismatch: expected={local_index} got={decoded_local_index}")
            yield pos, frame_info, frame
    finally:
        if cap is not None:
            cap.release()


def iter_sample_cache(cache_dir: str | Path) -> Iterator[tuple[dict, np.ndarray]]:
    manifest = load_sample_cache(cache_dir)
    for _, frame_info, frame in iter_sample_cache_range(cache_dir, 0, len(manifest.get("frames", []))):
        yield frame_info, frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sampled frame cache for video analysis passes.")
    parser.add_argument("--input", "-i", required=True, help="Input .mp4 path")
    parser.add_argument("--output", "-o", required=True, help="Output cache directory")
    parser.add_argument("--sample-every", type=int, default=SampleCacheConfig.sample_every)
    parser.add_argument("--sample-fps", type=float, default=SampleCacheConfig.sample_fps)
    parser.add_argument("--resize-width", type=int, default=SampleCacheConfig.resize_width)
    parser.add_argument(
        "--decode-backend",
        choices=["opencv", "ffmpeg-cuda", "ffmpeg-videotoolbox", "auto"],
        default=SampleCacheConfig.decode_backend,
    )
    parser.add_argument("--person-mask-batch-size", type=int, default=SampleCacheConfig.person_mask_batch_size)
    parser.add_argument("--person-masks", action="store_true", default=None, help="Enable YOLO person mask generation")
    parser.add_argument("--no-person-masks", action="store_false", dest="person_masks", help="Disable YOLO person mask generation")
    parser.add_argument("--person-mask-model", default=SampleCacheConfig.person_mask_model)
    parser.add_argument("--person-mask-dilate-px", type=int, default=SampleCacheConfig.person_mask_dilate_px)
    parser.add_argument("--person-mask-static-diff-threshold", type=float, default=SampleCacheConfig.person_mask_static_diff_threshold)
    parser.add_argument("--person-mask-static-changed-ratio-threshold", type=float, default=SampleCacheConfig.person_mask_static_changed_ratio_threshold)
    parser.add_argument("--person-mask-match-iou-threshold", type=float, default=SampleCacheConfig.person_mask_match_iou_threshold)
    parser.add_argument("--person-mask-fill-gap-sec", type=float, default=SampleCacheConfig.person_mask_fill_gap_sec)
    parser.add_argument("--save-person-mask-previews", action="store_true", help="Save a few masked sample preview JPGs for debugging")
    parser.add_argument("--person-mask-preview-limit", type=int, default=SampleCacheConfig.person_mask_preview_limit)
    parser.add_argument("--chunked", action="store_true", help="Build sample cache by 5-minute chunks and merge it back")
    parser.add_argument("--chunk-sec", type=float, default=float(os.getenv("GRAPHLEC_SAMPLE_CACHE_CHUNK_SEC", "300")))
    parser.add_argument("--chunk-overlap-sec", type=float, default=float(os.getenv("GRAPHLEC_SAMPLE_CACHE_CHUNK_OVERLAP_SEC", "30")))
    parser.add_argument("--chunk-workers", type=int, default=int(os.getenv("GRAPHLEC_SAMPLE_CACHE_CHUNK_WORKERS", "2")))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    cfg = SampleCacheConfig(
        sample_every=args.sample_every,
        sample_fps=args.sample_fps,
        resize_width=args.resize_width,
        decode_backend=args.decode_backend,
        person_mask_batch_size=args.person_mask_batch_size,
        person_masks=SampleCacheConfig.person_masks if args.person_masks is None else bool(args.person_masks),
        person_mask_model=args.person_mask_model,
        person_mask_dilate_px=args.person_mask_dilate_px,
        person_mask_static_diff_threshold=args.person_mask_static_diff_threshold,
        person_mask_static_changed_ratio_threshold=args.person_mask_static_changed_ratio_threshold,
        person_mask_match_iou_threshold=args.person_mask_match_iou_threshold,
        person_mask_fill_gap_sec=args.person_mask_fill_gap_sec,
        save_person_mask_previews=args.save_person_mask_previews,
        person_mask_preview_limit=args.person_mask_preview_limit,
    )
    if args.chunked:
        create_sample_cache_chunked(
            args.input,
            args.output,
            cfg,
            chunk_sec=args.chunk_sec,
            overlap_sec=args.chunk_overlap_sec,
            workers=args.chunk_workers,
        )
    else:
        create_sample_cache(args.input, args.output, cfg)


if __name__ == "__main__":
    main()
