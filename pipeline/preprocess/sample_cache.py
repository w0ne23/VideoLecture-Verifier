"""
입력 영상에서 경량 샘플 프레임 캐시 생성

기존 단일 패스 동작은 유지하면서 긴 강의 영상을 위한 청크 단위 캐시 생성을 추가.
청크 방식은 독립적으로 기록된 청크 영상을 그대로 보존하고, 각 core 프레임의
정확한 로컬 위치를 후속 단계와 호환되는 하나의 manifest에 기록. overlap 트리밍은
가상으로만 처리되어, 병렬 생성 이후 어떤 청크 영상도 디코딩/자르기/재인코딩되지 않음
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


# 환경변수를 float로 읽음
def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


# 환경변수를 int로 읽음
def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


# 환경변수를 bool로 읽음
def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


# 시스템 물리 메모리 전체 크기 조회
def _system_memory_bytes() -> int | None:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None
    return pages * page_size if pages > 0 and page_size > 0 else None


def _available_memory_bytes() -> int | None:
    """이 컨테이너에서 보이는 보수적인 사용 가능 메모리 예산 반환"""
    candidates: list[int] = []
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass

    # Docker Desktop과 Kubernetes는 보통 cgroup v2 메모리 제한을 노출함,
    # 호스트 전체 /proc 메모리 정보보다 그 남은 예산을 우선 사용
    for root in (Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")):
        try:
            limit_raw = (root / "memory.max").read_text(encoding="utf-8").strip()
            current_raw = (root / "memory.current").read_text(encoding="utf-8").strip()
            if limit_raw != "max":
                remaining = int(limit_raw) - int(current_raw)
                if remaining > 0:
                    candidates.append(remaining)
                break
        except (OSError, ValueError):
            continue
    return min(candidates) if candidates else _system_memory_bytes()


# 리소스 admission 판단에만 쓰이는 보수적인 MJPEG 캐시 크기 추정
def _estimate_sample_cache_bytes(
    video_meta: dict,
    duration_sec: float,
    sample_fps: float,
    *,
    resize_width: int,
) -> int:
    """리소스 admission 판단에만 사용하는 보수적인 MJPEG 캐시 크기 추정치"""
    source_width = max(1, int(video_meta.get("width") or 1))
    source_height = max(1, int(video_meta.get("height") or 1))
    width = max(1, int(resize_width))
    height = max(1, int(source_height * (width / source_width)))
    samples = max(1, int(math.ceil(max(0.0, duration_sec) * max(0.1, sample_fps))))
    # 강의 이미지 특성상 raw BGR 크기의 35%로 의도적으로 보수적으로 추정
    bytes_per_frame = max(96 * 1024, int(width * height * 3 * 0.35))
    return samples * bytes_per_frame


# 지정 경로의 여유 공간이 필요량 이상인지 확인, 부족하면 예외 발생
def _require_free_space(path: Path, required_bytes: int, *, purpose: str) -> int:
    free_bytes = shutil.disk_usage(path).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"insufficient {purpose} space at {path}: "
            f"free={free_bytes / 1024**3:.1f}GiB required={required_bytes / 1024**3:.1f}GiB"
        )
    return free_bytes


# 샘플 캐시 작업용 임시 루트 경로 결정, 설정/가용 메모리에 따라 RAM(tmpfs) 또는 디스크 선택
def _sample_cache_temp_root(
    output_path: Path,
    *,
    estimated_cache_bytes: int,
) -> tuple[Path, str, int]:
    mode = os.getenv("VLVERIFIER_SAMPLE_CACHE_STORAGE_MODE", "auto").strip().lower()
    if mode not in {"auto", "ram", "disk"}:
        mode = "auto"

    # Step 0 동안 청크 파일과 병합 캐시가 공존함, 병합 도중 ENOSPC를 만나지 않도록
    # 둘 다와 manifest를 위한 공간을 미리 충분히 확보
    disk_required = max(2 * 1024**3, int(estimated_cache_bytes * 2.25 + 512 * 1024**2))
    output_root = output_path.parent
    output_root.mkdir(parents=True, exist_ok=True)
    disk_free = _require_free_space(output_root, disk_required, purpose="sample-cache disk")
    if mode == "disk":
        return output_root, "disk", disk_free

    total_memory = _available_memory_bytes()
    ram_threshold = 16 * 1024**3
    ram_root = Path(os.getenv("VLVERIFIER_SAMPLE_CACHE_RAM_DIR", "/dev/shm"))
    if mode == "ram" or (mode == "auto" and total_memory is not None and total_memory > ram_threshold):
        try:
            ram_root.mkdir(parents=True, exist_ok=True)
            ram_required = max(2 * 1024**3, int(estimated_cache_bytes * 1.15 + 512 * 1024**2))
            ram_free = shutil.disk_usage(ram_root).free
            if ram_free >= ram_required:
                return ram_root, "ram", ram_free
            if mode == "ram":
                raise RuntimeError(
                    f"insufficient sample-cache RAM space at {ram_root}: "
                    f"free={ram_free / 1024**3:.1f}GiB required={ram_required / 1024**3:.1f}GiB"
                )
        except OSError:
            if mode == "ram":
                raise
    return output_root, "disk", disk_free


# 샘플 캐시 생성 설정, 대부분 환경변수로 오버라이드 가능
@dataclass
class SampleCacheConfig:
    sample_every: int = _env_int("VLVERIFIER_SAMPLE_CACHE_EVERY", 2)
    sample_fps: float = _env_float("VLVERIFIER_SAMPLE_CACHE_FPS", 10.0)
    resize_width: int = _env_int("VLVERIFIER_SAMPLE_CACHE_RESIZE_WIDTH", 768)
    jpeg_quality: int = 95
    decode_backend: str = os.getenv("VLVERIFIER_SLIDE_DECODE_BACKEND", "auto")
    person_mask_batch_size: int = _env_int("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_BATCH_SIZE", 32)
    person_masks: bool = _env_bool("VLVERIFIER_SAMPLE_CACHE_PERSON_MASKS", True)
    person_mask_model: str = os.getenv("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_MODEL", "/app/models/yolo26n.pt")
    person_mask_engine: str = os.getenv(
        "VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_ENGINE",
        "/app/storage/models/yolo26n-fp16.engine",
    )
    person_mask_workers: int = _env_int("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_WORKERS", 4)
    person_mask_task_sec: float = _env_float("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_TASK_SEC", 480.0)
    person_mask_task_overlap_sec: float = _env_float("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_TASK_OVERLAP_SEC", 5.0)
    person_mask_engine_batch_size: int = _env_int("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_ENGINE_BATCH_SIZE", 32)
    person_mask_roi_quantile: float = _env_float("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_ROI_QUANTILE", 0.95)
    person_mask_roi_recenter_px: int = _env_int("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_ROI_RECENTER_PX", 120)
    person_mask_roi_max_gap_sec: float = _env_float("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_ROI_MAX_GAP_SEC", 2.0)
    person_mask_gpu_min_free_mb: int = _env_int("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_MIN_FREE_MB", 0)
    person_mask_gpu_stagger_sec: float = _env_float("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_GPU_STAGGER_SEC", 0.0)
    person_mask_gpu_wait_sec: float = _env_float("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_GPU_WAIT_SEC", 0.5)
    person_mask_conf: float = _env_float("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_CONF", 0.70)
    person_mask_min_bbox_height_ratio: float = _env_float(
        "VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_MIN_BBOX_HEIGHT_RATIO", 0.08
    )
    person_mask_max_bbox_aspect: float = _env_float(
        "VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_MAX_BBOX_ASPECT", 1.50
    )
    # detector 박스가 발표자의 머리카락이나 뻗은 손을 놓칠 수 있음, 고정 ROI가 이
    # 경계 움직임을 포함하지 못하면 작은 오탐 필기로 annotation 감지에 새어 들어감
    person_mask_dilate_px: int = _env_int("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_DILATE_PX", 48)
    person_mask_static_diff_threshold: float = 1.0
    person_mask_static_changed_ratio_threshold: float = 0.003
    person_mask_match_iou_threshold: float = 0.05
    # 고정 ROI는 움직임이 확인된 프레임만 커버해야 함, 인접한 단일 detection으로
    # 채우면 정적인 오탐이 긴 마스크로 바뀌므로 배포에서 명시적으로 켜지 않는 한 비활성화
    person_mask_fill_gap_sec: float = _env_float(
        "VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_FILL_GAP_SEC", 0.0
    )
    person_mask_fixed_motion_min_mean_diff: float = _env_float(
        "VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_FIXED_MOTION_MIN_MEAN_DIFF", 2.0
    )
    person_mask_fixed_motion_min_changed_ratio: float = _env_float(
        "VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_FIXED_MOTION_MIN_CHANGED_RATIO", 0.01
    )
    person_mask_fixed_min_center_shift_px: float = _env_float(
        "VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_FIXED_MIN_CENTER_SHIFT_PX", 4.0
    )
    person_mask_fixed_min_area_change_ratio: float = _env_float(
        "VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_FIXED_MIN_AREA_CHANGE_RATIO", 0.08
    )
    person_mask_active_ranges: list[tuple[float, float]] | None = field(default=None, repr=False)
    save_person_mask_previews: bool = False
    person_mask_preview_limit: int = 30


# 프레임을 지정 너비로 비율 유지 리사이즈
def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


# 프레임을 그레이스케일+가우시안 블러로 변환해 비교용 프레임 생성
def to_decision_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


# 두 프레임의 평균제곱오차(MSE) 계산
def compute_mse(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    a = frame_a.astype(np.float32) if frame_a.ndim == 2 else cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = frame_b.astype(np.float32) if frame_b.ndim == 2 else cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean((a - b) ** 2))


# 프레임의 perceptual hash를 정수로 계산
def compute_phash_int(frame: np.ndarray) -> int:
    if frame.ndim == 2:
        pil_img = Image.fromarray(frame)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return int(str(imagehash.phash(pil_img)), 16)


# 두 phash 정수 간 해밍 거리 계산
def phash_distance_int(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


# YOLO 인물 탐지 모델 로드, ultralytics 미설치나 로드 실패 시 None 반환하고 person mask 비활성화
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


# 설정된 TensorRT engine 파일 경로 조회, 빈 값을 '.'으로 취급하지 않음
def _existing_person_mask_engine(path_value: str | Path | None) -> Path | None:
    """설정된 TensorRT engine 반환, 빈 값을 '.'으로 취급하지 않음"""
    raw = str(path_value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


# YOLO 추론 결과에서 bbox/confidence/mask 목록 추출
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


# 예외 메시지가 CUDA OOM(메모리 부족)인지 확인
def _cuda_oom_message(exc: Exception) -> bool:
    text = str(exc).lower()
    return "cuda" in text and "out of memory" in text


# torch CUDA 캐시 비우기, torch 미설치/미사용 시 조용히 무시
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


# nvidia-smi로 GPU 여유 메모리(MB) 조회
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


# person mask 추론 시작 전 GPU 여유 메모리/stagger 조건을 만족할 때까지 파일 락으로 대기
def _wait_for_person_mask_gpu_gate(cfg: SampleCacheConfig, batch_size: int) -> None:
    min_free_mb = max(0, int(cfg.person_mask_gpu_min_free_mb))
    stagger_sec = max(0.0, float(cfg.person_mask_gpu_stagger_sec))
    wait_sec = max(0.05, float(cfg.person_mask_gpu_wait_sec))
    if min_free_mb <= 0 and stagger_sec <= 0.0:
        return

    lock_path = Path(os.getenv("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_GPU_GATE_LOCK", "/tmp/vlverifier_person_mask_gpu_gate.lock"))
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


# person 모델 배치 1개를 프로세스 전역 GPU 실행 락 아래에서 실행
def _run_person_model_serialized(model, frames: list[np.ndarray], conf: float):
    """프로세스 전역 GPU 실행 락 아래에서 person 모델 배치 1개 실행

    청크 디코딩은 계속 병렬로 진행하지만, 별도 spawn된 worker의 TensorRT 실행
    컨텍스트를 이 동적 engine에 동시에 enqueue하는 것은 안전하지 않다. 짧은
    추론 호출만 큐잉하면 비디오 디코드를 직렬화하지 않고도 CUDA dispatch
    충돌을 피할 수 있다
    """
    if not _env_bool("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_GPU_SERIALIZE", True):
        return model(frames, classes=[0], conf=conf, verbose=False, stream=False)

    lock_path = Path(
        os.getenv(
            "VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_GPU_EXEC_LOCK",
            "/tmp/vlverifier_person_mask_gpu_execute.lock",
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


# 이 청크 프로세스를 위해 제한된 TensorRT context slot 1개 예약
def _acquire_person_mask_trt_context_slot() -> tuple[object, int]:
    """이 청크 프로세스를 위해 제한된 TensorRT context slot 1개 예약

    YOLO TensorRT context는 이 동적 engine 기준 약 1.4 GiB를 소비한다. 디코드
    청크마다 context를 하나씩 유지하면, 배치 실행을 직렬화해도 context 할당이
    계속 상주하기 때문에 GPU 메모리가 고갈된다
    """
    limit = max(
        1,
        int(os.getenv("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_TRT_CONTEXTS", "2")),
    )
    root = Path(
        os.getenv(
            "VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_TRT_CONTEXT_DIR",
            "/tmp/vlverifier_person_mask_trt_context_slots",
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


# 예약했던 TensorRT context slot 해제
def _release_person_mask_trt_context_slot(slot: tuple[object, int] | None) -> None:
    if slot is None:
        return
    handle, _ = slot
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


# 탐지된 인물 bbox가 최소 높이 비율/최대 종횡비 조건을 만족하는지 확인
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


# 청크 구간을 저해상도로 seek 샘플링해 인물이 등장하는지 빠르게 확인,
# 등장 시각 주변 구간만 person mask 계산 대상으로 반환
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
        float(os.getenv("VLVERIFIER_SAMPLE_CACHE_PERSON_GATE_INTERVAL_SEC", "20")),
    )
    # 이 gate는 최종 person-mask 판정이 아니라 활성화 가드, 오탐(미검출) 방지를 위해
    # 보수적으로 설정 — seek 샘플 1개만 양성이어도 청크 전체 마스킹을 활성화
    min_hits = max(
        1,
        int(os.getenv("VLVERIFIER_SAMPLE_CACHE_PERSON_GATE_MIN_HITS", "1")),
    )
    gate_conf = max(
        0.10,
        float(os.getenv("VLVERIFIER_SAMPLE_CACHE_PERSON_GATE_CONF", "0.45")),
    )
    video_duration = float(video_meta.get("duration_sec") or 0.0)
    start_sec = max(0.0, float(start_sec))
    end_sec = video_duration if end_sec is None else min(video_duration, float(end_sec))
    duration = max(0.0, end_sec - start_sec)
    if duration <= 0.0:
        return []

    width = int(video_meta["width"])
    height = int(video_meta["height"])
    output_width = int(os.getenv("VLVERIFIER_SAMPLE_CACHE_PERSON_GATE_WIDTH", "384"))
    output_height = int(height * (output_width / width))
    log.info(
        "person presence gate start: interval=%.1fs min_hits=%s conf=%.2f size=%sx%s mode=seek",
        interval_sec,
        min_hits,
        gate_conf,
        output_width,
        output_height,
    )

    engine_path = _existing_person_mask_engine(cfg.person_mask_engine)
    model_source = str(engine_path) if engine_path else cfg.person_mask_model
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

    # 배치의 각 프레임에 대해 인물 탐지 실행, engine 로드 실패 시 기본 모델로 폴백
    def check_batch(batch_items: list[tuple[float, np.ndarray]]) -> None:
        nonlocal sampled, hits, model, model_source, fallback_used
        if not batch_items:
            return
        batch_frames = [frame for _, frame in batch_items]
        try:
            gate_cfg = copy.deepcopy(cfg)
            gate_cfg.person_mask_conf = gate_conf
            # presence gate에는 더 엄격한 최종 마스크 geometry 필터를 적용하지 않음,
            # 프레임 일부만 걸쳐 있거나 가장자리에 있는 인물도 청크를 활성화시켜야
            # 이후 원해상도 마스크 패스에서 최종 판단 가능
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


# 주어진 시각이 person mask 활성 구간에 속하는지 확인, ranges가 None이면 항상 활성
def _person_mask_active_at(timestamp_sec: float, ranges: list[tuple[float, float]] | None) -> bool:
    if ranges is None:
        return True
    timestamp = float(timestamp_sec)
    return any(start - 1e-6 <= timestamp <= end + 1e-6 for start, end in ranges)


# 프레임 배치에 대해 YOLO 인물 탐지 실행, GPU gate 대기와 CUDA OOM 폴백 포함
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


# 두 bbox 간 IoU(교집합/합집합 비율) 계산
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


# bbox 영역 내 두 프레임 간 평균 밝기 차/변경 픽셀 비율 계산
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


def _bbox_ring_changed_ratio(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    """인물 박스를 제외한 주변 배경 영역의 움직임 비율 측정"""
    height, width = frame_a.shape[:2]
    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    pad = max(16, int(round(min(box_width, box_height) * 0.15)))
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(width, x2 + pad), min(height, y2 + pad)
    if rx2 <= rx1 or ry2 <= ry1:
        return 0.0

    a = cv2.cvtColor(frame_a[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(frame_b[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    changed = np.abs(a - b) >= 8.0
    ring = np.ones(changed.shape, dtype=bool)
    inner_x1, inner_y1 = max(0, x1 - rx1), max(0, y1 - ry1)
    inner_x2, inner_y2 = min(ring.shape[1], x2 - rx1), min(ring.shape[0], y2 - ry1)
    if inner_x2 > inner_x1 and inner_y2 > inner_y1:
        ring[inner_y1:inner_y2, inner_x1:inner_x2] = False
    ring_pixels = int(np.count_nonzero(ring))
    if ring_pixels == 0:
        return 0.0
    return float(np.count_nonzero(changed & ring) / ring_pixels)


# 다음 프레임과 비교해 실제로 움직인(정적이지 않은) 인물 bbox만 마스크로 생성
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


# 움직임 여부와 무관하게 탐지된 모든 인물 bbox로 존재 마스크 생성
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


# 마스크 영역을 검게 칠한 미리보기 이미지 생성(디버그용)
def _masked_preview(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    preview = frame.copy()
    preview[mask.astype(bool)] = 0
    return preview


# 마스크 없는 짧은 구간을 가장 가까운 마스크로 채움(max_gap 이내만)
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


# 출력 디렉터리 및 하위 경로 준비, 기존 산출물 정리 후 재생성
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


# 샘플 캐시 생성 핵심 구현, 단일 패스 또는 청크 일부 구간 처리에 공용으로 사용
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
    engine_path = _existing_person_mask_engine(cfg.person_mask_engine)
    if cfg.person_masks and engine_path:
        person_model_source = str(engine_path)
    person_model = None
    masks_enabled = bool(cfg.person_masks)
    async_person_model = None
    async_person_model_load_failed = False
    async_person_backend = "disabled"
    trt_context_slot = None
    person_model_warmed = False
    person_gate_interval_sec = max(
        1.0,
        float(os.getenv("VLVERIFIER_SAMPLE_CACHE_PERSON_GATE_INTERVAL_SEC", "20")),
    )
    person_gate_conf = max(
        0.10,
        float(os.getenv("VLVERIFIER_SAMPLE_CACHE_PERSON_GATE_CONF", "0.45")),
    )
    person_gate_next_timestamp = float(start_sec or 0.0)
    person_gate_active = False
    preview_count = 0
    pending_sample = None
    batch_size = max(1, int(cfg.person_mask_batch_size))
    total_frame_count = max(1, int(video_meta.get("frame_count") or 1))
    detection_executor = ThreadPoolExecutor(max_workers=1) if masks_enabled else None
    detection_queue: list[tuple[list[dict], list[int], object]] = []
    configured_inflight = os.getenv("VLVERIFIER_SAMPLE_CACHE_PERSON_MASK_INFLIGHT_BATCHES")
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

    # 배치의 인물 탐지 실행, 모델을 lazy 로드하고 TensorRT context 오류 시 PyTorch로 폴백
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

    # 샘플 1개에 대해 presence/moving person mask를 계산해 저장하고 최종 frame 목록에 추가
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

    # 디코딩된 배치를 프레임별로 처리, phash/mse 계산 후 캐시 영상에 기록하고 직전 샘플을 finalize
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

    # 큐에서 가장 오래된 탐지 배치의 결과를 기다려 process_batch로 전달
    def process_oldest_detection_batch() -> None:
        ready_samples, ready_indices, ready_future = detection_queue.pop(0)
        ready_detections = [[] for _ in ready_samples]
        for index, detections in zip(ready_indices, ready_future.result()):
            ready_detections[index] = detections
        process_batch(ready_samples, ready_detections)

    # 탐지 큐에 남은 배치를 모두 순서대로 처리
    def drain_detection_queue() -> None:
        while detection_queue:
            process_oldest_detection_batch()

    # 배치를 person mask 활성 구간 판단 후 비동기 탐지에 제출하거나 즉시 처리
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
            # 앞선 활성 구간 배치가 아직 GPU에서 실행 중일 수 있음, 이 뒤의 비활성
            # 배치를 기록하기 전에 먼저 flush해야 함 — 그렇지 않으면 활성 구간
            # 경계 근처 프레임이 순서를 벗어나 기록됨
            drain_detection_queue()
            process_batch(batch_samples, [[] for _ in batch_samples])
            return

        if not person_model_warmed:
            # Ultralytics는 첫 예측 시점에 TensorRT 실행 컨텍스트를 지연 생성함,
            # 청크 프로세스의 메인 스레드에서 초기화하면 안정적이지만, 비동기 worker
            # 스레드에서 먼저 초기화하면 컨텍스트가 None으로 남아 PyTorch 폴백을 유발할 수 있음
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
            process_oldest_detection_batch()

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
        drain_detection_queue()
    finally:
        if detection_executor is not None:
            detection_executor.shutdown(wait=True, cancel_futures=False)
        _release_person_mask_trt_context_slot(trt_context_slot)
        writer.release()

    if pending_sample is not None:
        finalize_sample(pending_sample, None, [])

    mask_fill_gap_samples = max(0, int(round(float(cfg.person_mask_fill_gap_sec) * sampled_fps)))
    inherited_masks = _fill_short_person_mask_gaps(frames, max_gap=mask_fill_gap_samples)

    for previous, current in zip(frames, frames[1:]):
        if (
            float(current["timestamp_sec"]),
            int(current["frame_no"]),
        ) < (
            float(previous["timestamp_sec"]),
            int(previous["frame_no"]),
        ):
            raise RuntimeError(
                "sample cache chunk frame order violation: "
                f"previous={previous.get('frame_no')}@{previous.get('timestamp_sec')} "
                f"current={current.get('frame_no')}@{current.get('timestamp_sec')}"
            )

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


# 영상 전체에 대해 단일 패스로 샘플 캐시 생성
def create_sample_cache(input_path: str, output_dir: str, cfg: SampleCacheConfig | None = None) -> dict:
    return _create_sample_cache_impl(input_path, output_dir, cfg)


# 영상의 특정 구간(청크)에 대해 샘플 캐시 생성
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


# 전체 길이를 겹치는 청크 구간들로 분할, 각 청크의 core(비겹침) 구간도 함께 계산
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


# 값 목록의 분위수 계산
def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float32), min(1.0, max(0.0, q))))


# person mask 후처리를 병렬 task로 나누기 위해 프레임 타임라인을 겹치는 구간들로 분할
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


# worker 프로세스가 시작되기 전에 공유 TensorRT detector engine 1개를 미리 빌드
def _ensure_person_mask_engine(cfg: SampleCacheConfig) -> Path:
    """worker 프로세스 시작 전에 공유 TensorRT detector engine 1개 빌드"""
    source_path = Path(cfg.person_mask_model)
    configured_engine = str(cfg.person_mask_engine or "").strip()
    if not configured_engine:
        # CPU 전용 배포는 의도적으로 TensorRT engine이 없음, 호출자가 YOLO CPU로
        # 계속할 수 있도록 이식 가능한 .pt detector를 반환
        log.info("person mask TensorRT disabled; using portable model: %s", source_path)
        return source_path
    try:
        import tensorrt  # noqa: F401
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "TensorRT person mask mode requires the tensorrt-cu12 and ultralytics packages"
        ) from exc

    if not source_path.exists():
        raise FileNotFoundError(f"person detector model not found: {source_path}")
    engine_path = Path(configured_engine)
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


# 고정 ROI 계산용 인물 bbox 탐지(비동기 파이프라인과 별도의 동기 경로)
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


# bbox 목록으로부터 프레임 대비 인물 점유 비율 계산
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


# 탐지 결과를 안정적인 발표자 구간(epoch)으로 묶어 구간마다 고정 박스 하나를 도출
def _fixed_box_epochs(
    records: list[dict],
    cfg: SampleCacheConfig,
    width: int,
    height: int,
) -> list[dict]:
    """탐지 결과를 안정적인 발표자 구간으로 묶고 구간별 고정 박스 도출"""
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
        # 고정 ROI 하나가 이 구간에서 탐지된 모든 인물을 커버
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


# 트래킹으로 geometry 변화 또는 박스 내 국소적 움직임이 확인된 탐지만 유지, 정적 오탐 제거
def _motion_filter_person_boxes(
    frames: list[np.ndarray],
    boxes_by_frame: list[list[tuple[int, int, int, int]]],
    cfg: SampleCacheConfig,
) -> tuple[list[list[tuple[int, int, int, int]]], int]:
    """geometry 또는 박스 내 국소적 움직임으로 이동이 증명된 트랙만 유지"""
    filtered: list[list[tuple[int, int, int, int]]] = []
    vetoed = 0
    min_center_shift = max(0.0, float(cfg.person_mask_fixed_min_center_shift_px))
    min_area_change = max(0.0, float(cfg.person_mask_fixed_min_area_change_ratio))
    min_mean_diff = max(0.0, float(cfg.person_mask_fixed_motion_min_mean_diff))
    min_changed_ratio = max(0.0, float(cfg.person_mask_fixed_motion_min_changed_ratio))
    match_iou = max(0.05, float(cfg.person_mask_match_iou_threshold))
    tracks: list[dict] = []
    max_missing_frames = max(
        1,
        int(round(max(0.1, float(cfg.person_mask_roi_max_gap_sec)) * float(cfg.sample_fps))),
    )

    # bbox 중심 좌표 계산
    def center(box: tuple[int, int, int, int]) -> tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    # bbox 면적 계산
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
                # 새로 관측된 박스는 움직임을 증명하려면 두 번째 관측이 필요함,
                # 이렇게 하면 정적인 그림/인물 사진을 마스킹하는 것을 방지
                tracks.append({
                    "anchor": bbox,
                    "last": bbox,
                    "last_seen_frame": frame_index,
                    "confirmed": False,
                })
                vetoed += 1
                continue

            anchor = best_track["anchor"]
            previous_box = best_track["last"]
            previous_frame_index = int(best_track["last_seen_frame"])
            ax, ay = center(anchor)
            bx, by = center(bbox)
            center_shift = max(abs(bx - ax), abs(by - ay))
            area_change = abs(area(bbox) - area(anchor)) / area(anchor)
            motion_box = (
                min(previous_box[0], bbox[0]),
                min(previous_box[1], bbox[1]),
                max(previous_box[2], bbox[2]),
                max(previous_box[3], bbox[3]),
            )
            mean_diff, changed_ratio = _bbox_motion_metrics(
                frames[previous_frame_index], frames[frame_index], motion_box
            )
            ring_changed_ratio = _bbox_ring_changed_ratio(
                frames[previous_frame_index], frames[frame_index], motion_box
            )
            localized_pixel_motion = (
                mean_diff >= min_mean_diff
                and changed_ratio >= min_changed_ratio
                and ring_changed_ratio < max(min_changed_ratio, changed_ratio * 0.65)
            )
            best_track["last"] = bbox
            best_track["last_seen_frame"] = frame_index
            if not bool(best_track["confirmed"]) and (
                center_shift >= min_center_shift
                or area_change >= min_area_change
                or localized_pixel_motion
            ):
                best_track["confirmed"] = True
                # 진단과 향후 트랙 분할을 위해 새 기준점을 유지하되, 이 확정된
                # 발표자 트랙 자체는 그대로 보존
                best_track["anchor"] = bbox

            if bool(best_track["confirmed"]):
                valid_boxes.append(bbox)
            else:
                vetoed += 1
        filtered.append(valid_boxes)
    return filtered, vetoed


# 캐시 task 구간(기본 8분)에 대해 TensorRT 인물 탐지를 실행하고 metadata 업데이트 파일 생성
def _run_person_mask_task(cache_dir: str, task_dir: str, spec: dict, cfg: SampleCacheConfig) -> str:
    """캐시 task 구간에 대해 TensorRT 인물 탐지를 실행하고 metadata 업데이트 생성"""
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
        # epoch는 ROI geometry를 제공하지만, 그 epoch 내 정적인 프레임은
        # 마스킹되지 않은 채로 남아야 함, 박스 움직임이 확정된 프레임만
        # 고정 ROI 마스크를 받음
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


# 청크 병합 완료 후, task들을 병렬 실행해 안정적인 고정 박스 person mask를 캐시에 부착
def materialize_fixed_person_masks(cache_dir: str | Path, cfg: SampleCacheConfig) -> dict:
    """청크 단위 sample-cache 병합 이후 안정적인 고정 박스 person mask 부착"""
    cache_path = Path(cache_dir)
    manifest_path = cache_path / MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not cfg.person_masks:
        return {"enabled": False, "tasks": 0, "workers": 0, "epochs": 0}
    if not frames:
        return {"enabled": True, "tasks": 0, "workers": 0, "epochs": 0}

    cached_boxes_available = all("person_boxes" in frame for frame in frames)
    engine_path = _existing_person_mask_engine(cfg.person_mask_engine)
    model_path = engine_path or Path(cfg.person_mask_model)
    if not cached_boxes_available:
        model_path = _ensure_person_mask_engine(cfg)
    cfg = copy.deepcopy(cfg)
    cfg.person_mask_engine = str(model_path)
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
        model_path,
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


# 청크 1개를 별도 프로세스에서 생성하는 worker 함수, 부모가 계산한 person gate 결과를 그대로 사용
def _create_chunk_worker(input_path: str, chunk_dir: str, cfg: SampleCacheConfig, spec: dict) -> str:
    worker_cfg = copy.deepcopy(cfg)
    # 부모 프로세스가 단일 모델로 presence gate를 수행함, worker는 그 결과만
    # 전달받을 뿐 gate 모델을 직접 로드하지 않음
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


# 선택적 마스크 파일을 원본 청크 캐시에서 병합 캐시로 복사, 없으면 None
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
    _materialize_file(src, dst)
    return dst_rel_filename


# 임시 캐시 파일을 내용 재작성 없이 하드링크(우선) 또는 복사로 영구 저장
def _materialize_file(src: Path, dst: Path) -> str:
    """임시 캐시 파일을 내용 재작성 없이 영구 저장"""
    if not src.exists() or src.stat().st_size <= 0:
        raise FileNotFoundError(f"Cannot materialize missing or empty cache artifact: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


# 원본 청크 AVI 1개를 영구 저장하고, 그 core(비겹침) 프레임을 가상으로(디코딩 없이) 선택
def _materialize_chunk_segment_worker(
    chunk_manifest_path: str,
    spec: dict,
    segment_path: str,
) -> dict:
    """원본 청크 AVI 1개를 영구 저장하고 core 프레임을 가상으로 선택"""

    started_at = time.perf_counter()
    chunk_manifest = Path(chunk_manifest_path)
    chunk_dir = chunk_manifest.parent
    payload = json.loads(chunk_manifest.read_text(encoding="utf-8"))
    video_filename = payload.get("video_filename")
    if not video_filename:
        raise ValueError(f"Chunk cache manifest has no video_filename: {chunk_manifest}")
    chunk_video_path = chunk_dir / str(video_filename)

    core_start = float(spec["core_start_sec"])
    core_end = float(spec["core_end_sec"])
    is_last = bool(spec.get("is_last"))

    # 프레임 timestamp가 core(비겹침) 구간에 속하는지 확인
    def in_core(item: dict) -> bool:
        timestamp = float(item["timestamp_sec"])
        if timestamp < core_start - 1e-6:
            return False
        if is_last:
            return timestamp <= core_end + 1e-6
        return timestamp < core_end - 1e-6

    all_frames = list(payload.get("frames", []))
    selected_indices = [index for index, item in enumerate(all_frames) if in_core(item)]
    if not selected_indices:
        raise RuntimeError(
            f"Chunk has no samples in its core range: chunk={spec['chunk_index']} "
            f"core={core_start:.6f}~{core_end:.6f}"
        )
    expected_indices = list(range(selected_indices[0], selected_indices[-1] + 1))
    if selected_indices != expected_indices:
        raise RuntimeError(
            f"Chunk core sample indices are not contiguous: chunk={spec['chunk_index']}"
        )

    selected_frames = []
    for local_index in selected_indices:
        selected = dict(all_frames[local_index])
        selected["cache_segment_frame_index"] = int(local_index)
        selected["frame_no"] = int(selected["frame_no"])
        selected["timestamp_sec"] = round(float(selected["timestamp_sec"]), 6)
        selected_frames.append(selected)
    skipped_overlap = len(all_frames) - len(selected_frames)
    segment = Path(segment_path)
    materialize_mode = _materialize_file(chunk_video_path, segment)
    return {
        "chunk_index": int(spec["chunk_index"]),
        "chunk_dir": str(chunk_dir),
        "segment_path": str(segment),
        "segment_filename": f"segments/{segment.name}",
        "materialize_mode": materialize_mode,
        "selected_count": len(selected_frames),
        "skipped_overlap": skipped_overlap,
        "elapsed": time.perf_counter() - started_at,
        "frames": selected_frames,
    }


# 청크별로 생성된 캐시들을 병합해 하나의 논리적 샘플 캐시(가상 조립)로 만듦, 실제
# 영상 재인코딩 없이 각 청크의 core 프레임만 선택해 전체 manifest 구성
def _assemble_chunk_caches(
    input_path: str,
    output_dir: str,
    cfg: SampleCacheConfig,
    manifest_paths: list[Path],
    specs: list[dict],
) -> dict:
    import time

    video_meta = read_video_metadata(input_path)
    output_path = Path(output_dir)
    # 청크 worker들이 이미 최종 MJPEG 바이트를 병렬로 기록해둠, 그 파일들을 그대로
    # 영구 저장하고 overlap 제거는 전역 manifest에서만 처리 — 여기서는 디코딩,
    # 패킷 트리밍, 연결, 재인코딩을 하지 않음
    _, merged_manifest_path, _, _, _ = _prepare_output_dirs(output_path, cfg)

    cached_height = int(video_meta["height"] * (cfg.resize_width / video_meta["width"]))
    sampled_fps = float(cfg.sample_fps)
    assemble_started_at = time.perf_counter()

    segments_dir = output_path / "segments"
    if segments_dir.exists():
        shutil.rmtree(segments_dir, ignore_errors=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    total_chunk_frames = 0
    for chunk_manifest_path in manifest_paths:
        try:
            payload = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
            total_chunk_frames += len(payload.get("frames", []))
        except Exception:
            pass

    requested_workers = int(
        os.getenv(
            "VLVERIFIER_SAMPLE_CACHE_MERGE_WORKERS",
            os.getenv("VLVERIFIER_SAMPLE_CACHE_CHUNK_WORKERS", "2"),
        )
    )
    materialize_workers = max(1, min(requested_workers, len(specs) if specs else 1))

    log.info(
        "sample cache virtual assembly start: chunks=%s candidate_frames=%s output=%s workers=%s",
        len(specs),
        total_chunk_frames,
        output_path,
        materialize_workers,
    )

    materialize_started_at = time.perf_counter()
    segment_by_index: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=materialize_workers) as executor:
        future_map = {}
        for chunk_manifest_path, spec in zip(manifest_paths, specs):
            chunk_index = int(spec["chunk_index"])
            segment_path = segments_dir / f"chunk_{chunk_index:03d}.avi"
            future = executor.submit(
                _materialize_chunk_segment_worker,
                str(chunk_manifest_path),
                dict(spec),
                str(segment_path),
            )
            future_map[future] = spec
            log.info(
                "  [sample segment materialize submit %s/%s] core=%.1fs~%.1fs",
                chunk_index + 1,
                len(specs),
                float(spec["core_start_sec"]),
                float(spec["core_end_sec"]),
            )

        completed = 0
        for future in as_completed(future_map):
            result = future.result()
            chunk_index = int(result["chunk_index"])
            segment_by_index[chunk_index] = result
            completed += 1
            log.info(
                "  [sample segment materialized %s/%s | idx=%s] selected=%s skipped_overlap=%s mode=%s elapsed=%.1fs",
                completed,
                len(specs),
                chunk_index + 1,
                int(result["selected_count"]),
                int(result["skipped_overlap"]),
                str(result.get("materialize_mode") or "unknown"),
                float(result["elapsed"]),
            )

    ordered_indices = sorted(segment_by_index)
    ordered_results = [segment_by_index[idx] for idx in ordered_indices]
    materialize_elapsed = time.perf_counter() - materialize_started_at

    if not ordered_results:
        raise RuntimeError("No chunk cache segments were materialized")

    # 각 segment는 overlap 프레임을 포함한 원본 청크 캐시임, manifest는 core 프레임만
    # 노출하고 그 원본 로컬 인덱스를 그대로 보존
    concat_mode = "none-direct-chunks"

    manifest_started_at = time.perf_counter()
    frames: list[dict] = []
    seen_frame_nos: set[int] = set()
    sample_index = 0
    duplicate_frame_count = 0

    mask_workers_requested = int(
        os.getenv(
            "VLVERIFIER_SAMPLE_CACHE_MERGE_MASK_WORKERS",
            str(max(1, materialize_workers)),
        )
    )
    mask_workers = max(1, mask_workers_requested)
    mask_copy_started_at = time.perf_counter()
    mask_future_map = {}

    # presence mask 파일명/비율을 frame_record에 반영
    def _apply_presence_result(frame_record: dict, item: dict, rel_filename: str | None) -> None:
        if not rel_filename:
            return
        frame_record["person_presence_mask_filename"] = rel_filename
        if item.get("person_presence_ratio") is not None:
            frame_record["person_presence_ratio"] = item.get("person_presence_ratio")

    # person mask 파일명/비율/승계 정보를 frame_record에 반영
    def _apply_person_result(frame_record: dict, item: dict, rel_filename: str | None) -> None:
        # 고정 박스 person masking은 의도적으로 프레임별 별도 presence-mask 파일을
        # 만들지 않음, 그래서 그 presence 비율은 person-mask 레코드와 함께 이동하며
        # 청크 병합 이후에도 유지되어야 함
        if item.get("person_presence_ratio") is not None:
            frame_record["person_presence_ratio"] = item.get("person_presence_ratio")
        if not rel_filename:
            return
        frame_record["person_mask_filename"] = rel_filename
        if item.get("person_mask_inherited"):
            frame_record["person_mask_inherited"] = True
            frame_record["person_mask_inherited_distance"] = item.get("person_mask_inherited_distance")

    # 마스크 파일 복사를 executor에 제출하거나(병렬) 즉시 동기 복사(직렬)
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
            segment_rel = str(result["segment_filename"])
            for item in result.get("frames", []):
                frame_no = int(item["frame_no"])
                if frame_no in seen_frame_nos:
                    duplicate_frame_count += 1
                    raise RuntimeError(
                        f"Duplicate frame_no during virtual assembly: frame_no={frame_no}. "
                        "This would desync the segmented cache and its manifest."
                    )
                seen_frame_nos.add(frame_no)

                sample_index += 1
                frame_record = {
                    "sample_index": sample_index,
                    "frame_no": frame_no,
                    "timestamp_sec": round(float(item["timestamp_sec"]), 6),
                    "cache_segment_index": segment_index,
                    "cache_segment_filename": segment_rel,
                    "cache_segment_frame_index": int(item["cache_segment_frame_index"]),
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
    # segment-local 프레임 인덱스는 이 지점 이전에 이미 부여됨, manifest만 재정렬하면
    # 그 인덱스가 실제 비디오 프레임과 어긋나므로, timestamp 정렬로 숨기지 않고
    # 순서 버그를 예외로 명확히 드러냄
    for previous, current in zip(frames, frames[1:]):
        if (
            float(current["timestamp_sec"]),
            int(current["frame_no"]),
        ) < (
            float(previous["timestamp_sec"]),
            int(previous["frame_no"]),
        ):
            raise RuntimeError(
                "sample cache assembled frame order violation: "
                f"previous={previous.get('frame_no')}@{previous.get('timestamp_sec')} "
                f"current={current.get('frame_no')}@{current.get('timestamp_sec')}"
            )
    for idx, frame in enumerate(frames, start=1):
        frame["sample_index"] = idx

    mask_fill_gap_samples = max(0, int(round(float(cfg.person_mask_fill_gap_sec) * sampled_fps)))
    inherited_masks = _fill_short_person_mask_gaps(frames, max_gap=mask_fill_gap_samples)
    postprocess_elapsed = time.perf_counter() - post_started_at

    skipped_overlap_frames = sum(int(result.get("skipped_overlap", 0)) for result in ordered_results)

    person_masks_payload = {
        "enabled": bool(cfg.person_masks),
        "dirname": MASKS_DIRNAME,
        # 고정 박스 모드는 최종 마스크만 저장, 어떤 프레임도 참조하지 않으면
        # presence 디렉터리를 노출하지 않음
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
            "merge_mode": "virtual_manifest",
            "segment_storage": "original_chunk_cache",
            "merge_workers": materialize_workers,
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
        "sample cache assembly timings: segment_materialize=%.1fs mask_materialize=%.1fs "
        "postprocess=%.1fs manifest_build=%.1fs manifest_write=%.1fs "
        "skipped_overlap=%s skipped_duplicate=%s concat_mode=%s manifest_frames=%s",
        materialize_elapsed,
        mask_materialize_elapsed,
        postprocess_elapsed,
        manifest_build_elapsed,
        manifest_write_elapsed,
        skipped_overlap_frames,
        duplicate_frame_count,
        concat_mode,
        len(frames),
    )
    log.info(
        "chunked sample cache assembled virtually: chunks=%s samples=%s output=%s elapsed=%.1fs",
        len(specs),
        len(frames),
        output_path,
        time.perf_counter() - assemble_started_at,
    )
    return manifest


# 캐시 매핑을 검증하고 대표 프레임들이 실제로 디코딩되는지 확인
def _verify_sample_cache_alignment(cache_dir: str | Path, manifest: dict) -> dict:
    """캐시 매핑을 검증하고 대표 프레임의 디코딩 가능 여부 확인

    직접 청크 segment는 청크 manifest 옆에 기록된 AVI 바이트를 그대로 담고 있으므로,
    안전한 정렬 계약은 구조적으로 검증한다: 유효한 로컬 인덱스 매핑과 예상 크기로
    디코딩되는 프레임인지만 확인한다. 손실 MJPEG 인코딩 이전에 캡처한 해시를
    디코딩된 픽셀과 비교하면 오탐 불일치가 발생하며 가상 매핑을 검증하지도 못한다
    """
    if os.getenv("VLVERIFIER_SAMPLE_CACHE_VERIFY_ALIGNMENT", "1").strip().lower() in {
        "0", "false", "no", "off",
    }:
        return {"enabled": False, "ok": True, "checked": 0, "mismatches": []}

    frames = list(manifest.get("frames", []) or [])
    if not frames:
        return {"enabled": True, "ok": False, "checked": 0, "mismatches": ["empty_manifest"]}

    stride = max(1, int(os.getenv("VLVERIFIER_SAMPLE_CACHE_VERIFY_STRIDE", "25")))
    max_hash_distance = max(0, int(os.getenv("VLVERIFIER_SAMPLE_CACHE_VERIFY_MAX_HASH_DISTANCE", "14")))
    target_positions = set(range(0, len(frames), stride))
    target_positions.add(len(frames) - 1)

    checked = 0
    mismatches: list[dict] = []
    segmented = is_segmented_sample_cache(manifest)
    if segmented:
        current_segment: str | None = None
        previous_local_index: int | None = None
        completed_segments: set[str] = set()
        for frame_info in frames:
            segment = frame_info.get("cache_segment_filename")
            local_index = frame_info.get("cache_segment_frame_index")
            if segment is None or local_index is None:
                mismatches.append({
                    "reason": "missing_segment_mapping",
                    "sample_index": int(frame_info.get("sample_index", 0) or 0),
                })
                break
            segment = str(segment)
            local_index = int(local_index)
            if local_index < 0:
                mismatches.append({
                    "reason": "negative_segment_frame_index",
                    "sample_index": int(frame_info.get("sample_index", 0) or 0),
                    "segment": segment,
                    "local_index": local_index,
                })
                break

            if segment != current_segment:
                if segment in completed_segments:
                    mismatches.append({
                        "reason": "non_contiguous_segment_reuse",
                        "sample_index": int(frame_info.get("sample_index", 0) or 0),
                        "segment": segment,
                    })
                    break
                if current_segment is not None:
                    completed_segments.add(current_segment)
                current_segment = segment
                previous_local_index = local_index
                continue

            expected_local_index = int(previous_local_index) + 1
            if local_index != expected_local_index:
                mismatches.append({
                    "reason": "non_contiguous_segment_mapping",
                    "sample_index": int(frame_info.get("sample_index", 0) or 0),
                    "segment": segment,
                    "local_index": local_index,
                    "expected_local_index": expected_local_index,
                })
                break
            previous_local_index = local_index
        if mismatches:
            return {
                "enabled": True,
                "ok": False,
                "checked": 0,
                "stride": stride,
                "max_hash_distance": max_hash_distance,
                "mismatches": mismatches,
            }
    verify_started_at = time.perf_counter()
    log.info(
        "sample cache alignment verify start: targets=%s samples=%s mode=%s",
        len(target_positions),
        len(frames),
        "direct-segment-mapping" if segmented else "legacy-sequential",
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
            if segmented:
                expected_width = int(manifest.get("cache", {}).get("width") or 0)
                expected_height = int(manifest.get("cache", {}).get("height") or 0)
                actual_height, actual_width = frame.shape[:2]
                if (
                    (expected_width > 0 and actual_width != expected_width)
                    or (expected_height > 0 and actual_height != expected_height)
                ):
                    mismatches.append({
                        "reason": "segment_frame_size_mismatch",
                        "sample_index": int(frame_info.get("sample_index", position + 1)),
                        "expected": [expected_width, expected_height],
                        "actual": [actual_width, actual_height],
                    })
                    break
                continue

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
                # 불일치 하나만으로도 캐시를 거부하기에 충분함, 긴 녹화에서
                # 검증 오버헤드가 무한정 늘지 않도록 제한
                break
        if not mismatches and checked != len(target_positions):
            mismatches.append({
                "reason": "verification_target_count_mismatch",
                "expected": len(target_positions),
                "actual": checked,
            })
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
        "mode": "direct-segment-mapping" if segmented else "legacy-phash",
        "mismatches": mismatches,
    }

# 긴 영상을 청크 단위로 병렬 처리해 샘플 캐시 생성, 청크가 필요 없으면 단일 패스로 폴백
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
    chunk_sec = float(chunk_sec if chunk_sec is not None else os.getenv("VLVERIFIER_SAMPLE_CACHE_CHUNK_SEC", "300"))
    overlap_sec = float(overlap_sec if overlap_sec is not None else os.getenv("VLVERIFIER_SAMPLE_CACHE_CHUNK_OVERLAP_SEC", "30"))
    requested_workers = int(workers if workers is not None else os.getenv("VLVERIFIER_SAMPLE_CACHE_CHUNK_WORKERS", "2"))

    video_meta = read_video_metadata(input_path)
    duration = float(video_meta.get("duration_sec") or 0.0)
    if duration <= 0.0 and video_meta.get("frame_count") and video_meta.get("fps"):
        duration = float(video_meta["frame_count"]) / float(video_meta["fps"])

    specs = _chunk_specs(duration, chunk_sec, overlap_sec)
    if not specs:
        log.info("sample cache chunking skipped: duration=%.1fs <= chunk_sec=%.1fs", duration, chunk_sec)
        return create_sample_cache(input_path, output_dir, cfg)

    workers = max(1, min(requested_workers, len(specs)))
    available_memory = _available_memory_bytes()
    if available_memory is not None:
        reserve_bytes = int(float(os.getenv("VLVERIFIER_SAMPLE_CACHE_WORKER_MEMORY_RESERVE_GB", "1.5")) * 1024**3)
        default_worker_mb = 2560 if cfg.person_masks else 1024
        worker_bytes = max(
            256 * 1024**2,
            int(os.getenv("VLVERIFIER_SAMPLE_CACHE_WORKER_MEMORY_MB", str(default_worker_mb))) * 1024**2,
        )
        memory_workers = max(1, int(max(0, available_memory - reserve_bytes) // worker_bytes))
        if workers > memory_workers:
            log.info(
                "sample cache worker cap: requested=%s capped=%s available_memory_gb=%.1f reserve_gb=%.1f per_worker_mb=%s",
                workers,
                memory_workers,
                available_memory / 1024**3,
                reserve_bytes / 1024**3,
                worker_bytes // 1024**2,
            )
            workers = memory_workers
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
    estimated_cache_bytes = _estimate_sample_cache_bytes(
        video_meta,
        duration,
        cfg.sample_fps,
        resize_width=cfg.resize_width,
    )
    temp_root, storage_mode, temp_free_bytes = _sample_cache_temp_root(
        output_path,
        estimated_cache_bytes=estimated_cache_bytes,
    )
    log.info(
        "sample cache temporary storage: mode=%s root=%s estimated_cache_gb=%.1f free_gb=%.1f available_memory_gb=%.1f",
        storage_mode,
        temp_root,
        estimated_cache_bytes / 1024**3,
        temp_free_bytes / 1024**3,
        ((available_memory or 0) / 1024**3),
    )

    # seek 기반 gate 검사는 모두 부모 프로세스에서 YOLO 인스턴스 1개로 실행,
    # 이렇게 하면 모든 청크 worker가 person-mask 작업이 없다는 것을 알아내기 위해
    # 각자 TensorRT context를 할당하는 낭비를 방지
    if cfg.person_masks:
        engine_path = _existing_person_mask_engine(cfg.person_mask_engine)
        gate_model_source = str(engine_path) if engine_path else cfg.person_mask_model
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
    with tempfile.TemporaryDirectory(prefix="vlverifier_sample_cache_chunks_", dir=str(temp_root)) as tmp_root:
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
        merged = _assemble_chunk_caches(input_path, output_dir, cfg, ordered_manifests, ordered_specs)

        alignment = _verify_sample_cache_alignment(output_dir, merged)
        if alignment["ok"]:
            log.info(
                "sample cache alignment verified: checked=%s stride=%s",
                alignment["checked"],
                alignment.get("stride"),
            )
            return merged

        # 잘못된 캐시는 scene/OCR/VLM 단계로 넘길 수 없음, 청크 병렬 처리는 그대로
        # 유지하되 직접 segment 매핑이나 대표 디코딩 검사가 실패하면 명확하게 실패시킴
        raise RuntimeError(
            "sample cache alignment mismatch after parallel chunk assembly: "
            f"checked={alignment['checked']} details={alignment['mismatches']}"
        )


# 샘플 캐시 manifest.json 로드
def load_sample_cache(cache_dir: str | Path) -> dict:
    manifest_path = Path(cache_dir) / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Sample cache manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


# manifest가 청크 segment 방식(virtual assembly)으로 저장된 캐시인지 확인
def is_segmented_sample_cache(manifest: dict) -> bool:
    return str(manifest.get("cache", {}).get("layout") or "").lower() == "segmented"


# 흩어진 전역 위치들을 읽음, 정확한 seek을 캐시 segment별로 묶어서 처리
def iter_sample_cache_selected_positions(
    cache_dir: str | Path,
    positions: list[int],
) -> Iterator[tuple[int, dict, np.ndarray]]:
    """흩어진 전역 위치를 읽음, 정확한 seek을 캐시 segment별로 그룹화

    무결성 검사 등 sparse consumer를 위한 함수다. 각 segment는 독립적으로 기록된
    MJPEG이라 로컬 프레임 seek이 concat 경계를 넘지 않는다. 일반적인 분석 구간은
    아래의 순차 range iterator를 대신 사용해야 한다
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


# legacy 또는 segment 방식 캐시 저장소에서 전역 위치 범위를 순차적으로 읽어 반환
def iter_sample_cache_range(
    cache_dir: str | Path,
    start_pos: int,
    end_pos: int,
) -> Iterator[tuple[int, dict, np.ndarray]]:
    """legacy 또는 segment 방식 캐시 저장소에서 전역 위치 범위를 순차적으로 읽어 반환

    segment 방식 캐시는 원본 청크 AVI를 독립적으로 저장한다. overlap 프레임은
    그 파일들 안에 남아 있지만 전역 manifest에서는 제외된다. 모든 manifest 프레임은
    정확한 segment 경로와 로컬 프레임 인덱스를 갖고 있어, 어떤 reader도 AVI concat
    타임스탬프 타임라인에 의존하지 않는다
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

            # 임의 AVI seek에 의존하지 않고 segment 시작부터 순방향으로 디코딩,
            # segment는 core 청크 하나(~2400 샘플)뿐이고 일반 range consumer는
            # 순차적으로 읽으므로 이 방식으로 충분
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


# 캐시 전체를 순회하며 (frame_info, frame) 쌍 생성
def iter_sample_cache(cache_dir: str | Path) -> Iterator[tuple[dict, np.ndarray]]:
    manifest = load_sample_cache(cache_dir)
    for _, frame_info, frame in iter_sample_cache_range(cache_dir, 0, len(manifest.get("frames", []))):
        yield frame_info, frame


# CLI 인자 파서 구성
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="영상 분석 단계용 샘플 프레임 캐시 생성")
    parser.add_argument("--input", "-i", required=True, help="입력 .mp4 경로")
    parser.add_argument("--output", "-o", required=True, help="출력 캐시 디렉터리")
    parser.add_argument("--sample-every", type=int, default=SampleCacheConfig.sample_every)
    parser.add_argument("--sample-fps", type=float, default=SampleCacheConfig.sample_fps)
    parser.add_argument("--resize-width", type=int, default=SampleCacheConfig.resize_width)
    parser.add_argument(
        "--decode-backend",
        choices=["opencv", "ffmpeg-cuda", "ffmpeg-videotoolbox", "auto"],
        default=SampleCacheConfig.decode_backend,
    )
    parser.add_argument("--person-mask-batch-size", type=int, default=SampleCacheConfig.person_mask_batch_size)
    parser.add_argument("--person-masks", action="store_true", default=None, help="YOLO person mask 생성 활성화")
    parser.add_argument("--no-person-masks", action="store_false", dest="person_masks", help="YOLO person mask 생성 비활성화")
    parser.add_argument("--person-mask-model", default=SampleCacheConfig.person_mask_model)
    parser.add_argument("--person-mask-dilate-px", type=int, default=SampleCacheConfig.person_mask_dilate_px)
    parser.add_argument("--person-mask-static-diff-threshold", type=float, default=SampleCacheConfig.person_mask_static_diff_threshold)
    parser.add_argument("--person-mask-static-changed-ratio-threshold", type=float, default=SampleCacheConfig.person_mask_static_changed_ratio_threshold)
    parser.add_argument("--person-mask-match-iou-threshold", type=float, default=SampleCacheConfig.person_mask_match_iou_threshold)
    parser.add_argument("--person-mask-fill-gap-sec", type=float, default=SampleCacheConfig.person_mask_fill_gap_sec)
    parser.add_argument("--save-person-mask-previews", action="store_true", help="디버깅용으로 마스킹된 샘플 미리보기 JPG 일부 저장")
    parser.add_argument("--person-mask-preview-limit", type=int, default=SampleCacheConfig.person_mask_preview_limit)
    parser.add_argument("--chunked", action="store_true", help="5분 단위 청크로 샘플 캐시를 생성한 뒤 다시 병합")
    parser.add_argument("--chunk-sec", type=float, default=float(os.getenv("VLVERIFIER_SAMPLE_CACHE_CHUNK_SEC", "300")))
    parser.add_argument("--chunk-overlap-sec", type=float, default=float(os.getenv("VLVERIFIER_SAMPLE_CACHE_CHUNK_OVERLAP_SEC", "30")))
    parser.add_argument("--chunk-workers", type=int, default=int(os.getenv("VLVERIFIER_SAMPLE_CACHE_CHUNK_WORKERS", "2")))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


# CLI 진입점, 청크 모드 여부에 따라 단일 패스 또는 청크 병렬 캐시 생성 실행
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
