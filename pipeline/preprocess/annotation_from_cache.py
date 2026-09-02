"""
Step 2 씬 구간 안에서 판서(annotation) 캡처 프레임 감지

단계별 슬라이드 추출 파이프라인의 Step 3:

  샘플 프레임 캐시 + scene_transitions.json
    -> scene_annotations.json
    -> 작은 미리보기 annotation 이미지

이 단계는 원본 해상도 프레임을 실체화하지 않음, Step 4가 원본 영상에서 추출해야 할
원본 frame_no 값만 기록

사용법:
    python -m pipeline.annotation_from_cache \
        --cache sample_cache_dir \
        --scenes scene_probe_step2/scene_transitions.json \
        --output annotation_probe_dir
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

try:
    from .sample_cache import iter_sample_cache, iter_sample_cache_range as _cache_range, load_sample_cache
    from .person_masks import load_person_mask, masked_pair
except ImportError:  # pragma: no cover - allows direct script execution
    from sample_cache import iter_sample_cache, iter_sample_cache_range as _cache_range, load_sample_cache
    from person_masks import load_person_mask, masked_pair


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


ANNOTATIONS_FILENAME = "scene_annotations.json"


# annotation 감지 임계값 모음
@dataclass
class AnnotationConfig:
    diff_threshold: int = 15
    cumulative_ratio: float = 0.001
    instant_ratio: float = 0.0001
    stable_sec: float = 0.7
    min_annot_sec: float = 0.2
    min_gap_sec: float = 1.5
    # 마지막으로 유지된 annotation 대비 눈에 보이는 변화가 커서/압축 노이즈뿐이면
    # 새 annotation으로 기록하지 않음
    capture_dedupe_ratio: float = 0.0005
    scene_start_guard_sec: float = 0.5
    scene_end_guard_sec: float = 1.0
    reject_large_change_ratio: float = 0.16
    crop_left: float = 0.12
    crop_top: float = 0.06
    crop_right: float = 0.94
    crop_bottom: float = 0.90


# scene_transitions.json 로드 및 scene_index 순 정렬
def _load_scenes(scene_path: str | Path) -> dict:
    path = Path(scene_path)
    if not path.exists():
        raise FileNotFoundError(f"Scene transition file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    scenes = sorted(payload.get("scenes", []), key=lambda item: int(item["scene_index"]))
    if not scenes:
        raise RuntimeError(f"No scenes found in: {path}")
    payload["scenes"] = scenes
    return payload


# 프레임에서 여백을 제외한 콘텐츠 영역만 크롭
def _content_region(frame: np.ndarray, cfg: AnnotationConfig) -> np.ndarray:
    h, w = frame.shape[:2]
    x0 = max(0, min(w - 1, int(w * cfg.crop_left)))
    y0 = max(0, min(h - 1, int(h * cfg.crop_top)))
    x1 = max(x0 + 1, min(w, int(w * cfg.crop_right)))
    y1 = max(y0 + 1, min(h, int(h * cfg.crop_bottom)))
    return frame[y0:y1, x0:x1]


# 판정용 크롭+그레이스케일+블러 프레임 생성
def _decision_frame(frame: np.ndarray, cfg: AnnotationConfig) -> np.ndarray:
    cropped = _content_region(frame, cfg)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


# 마스크도 판정용 콘텐츠 영역으로 크롭
def _decision_mask(mask: np.ndarray | None, cfg: AnnotationConfig) -> np.ndarray | None:
    if mask is None:
        return None
    return _content_region(mask.astype(np.uint8), cfg).astype(bool)


# 임계값 초과 픽셀 비율 계산
def _changed_ratio(frame_a: np.ndarray, frame_b: np.ndarray, threshold: int) -> float:
    diff = cv2.absdiff(frame_a, frame_b)
    return float(np.sum(diff > threshold) / diff.size)


# 사람 마스크를 제외한 두 프레임의 변화 픽셀 비율 계산
def _masked_changed_ratio(
    frame_a: np.ndarray,
    mask_a: np.ndarray | None,
    frame_b: np.ndarray,
    mask_b: np.ndarray | None,
    threshold: int,
) -> float:
    masked_a, masked_b = masked_pair(frame_a, mask_a, frame_b, mask_b)
    return _changed_ratio(masked_a, masked_b, threshold)


# frame_no에 대응하는 sample_index 조회, 캐시에 없으면 sample_every로 근사
def _sample_index_for_frame(frame_no: int, frame_to_sample: dict[int, int], sample_every: int) -> int:
    if frame_no in frame_to_sample:
        return frame_to_sample[frame_no]
    return max(1, int(round(frame_no / max(1, sample_every))))


def _iter_sample_cache_range(
    cache_dir: str | Path,
    manifest: dict,
    start_sample_index: int,
    end_sample_index: int,
):
    """샘플링된 MJPG 캐시에서 지정 구간만 읽기

    임의 탐색(seek)은 원본 강의 영상이 아니라 생성된 샘플 캐시로만 의도적으로 제한,
    캐시가 분석의 좌표계 역할을 하고 임의의 원본 인코딩보다 훨씬 안전하게 탐색 가능
    """
    frames = manifest.get("frames", [])
    if not frames:
        return

    start_sample_index = max(1, int(start_sample_index))
    end_sample_index = min(int(end_sample_index), len(frames))
    if end_sample_index < start_sample_index:
        return

    for _, frame_info, frame in _cache_range(cache_dir, start_sample_index - 1, end_sample_index):
        yield frame_info, frame


# scene 목록을 기준으로 annotation을 감지할 sample_index 구간(start/detect_start/end) 계산,
# 다음 scene 시작 전 end_guard, 씬 시작 후 start_guard를 적용
def _build_scene_intervals(scene_payload: dict, manifest: dict, cfg: AnnotationConfig) -> list[dict]:
    scenes = scene_payload["scenes"]
    frames = manifest.get("frames", [])
    frame_to_sample = {int(f["frame_no"]): int(f["sample_index"]) for f in frames}
    sample_every = int(manifest.get("config", {}).get("sample_every") or 2)
    sample_count = int(manifest.get("cache", {}).get("sample_count") or len(frames))
    sampled_fps = float(manifest.get("cache", {}).get("sampled_fps") or 1.0)
    start_guard_samples = max(0, int(round(cfg.scene_start_guard_sec * sampled_fps)))
    end_guard_samples = max(0, int(round(cfg.scene_end_guard_sec * sampled_fps)))
    region_end_by_index = {
        int(region["segment_index"]): int(region["end_sample_index"])
        for region in scene_payload.get("slide_regions", [])
    }

    intervals: list[dict] = []
    for i, scene in enumerate(scenes):
        scene_index = int(scene["scene_index"])
        base_sample = int(scene.get("sample_index") or _sample_index_for_frame(
            int(scene.get("base_frame_no", scene["frame_no"])),
            frame_to_sample,
            sample_every,
        ))
        region_index = scene.get("region_segment_index")
        region_index = int(region_index) if region_index is not None else None
        default_end = region_end_by_index.get(region_index, sample_count)

        end_sample = default_end
        if i + 1 < len(scenes):
            next_scene = scenes[i + 1]
            next_region = next_scene.get("region_segment_index")
            next_region = int(next_region) if next_region is not None else None
            if region_index is None or next_region == region_index:
                next_start_frame = int(next_scene.get("scene_start_frame_no", next_scene.get("frame_no")))
                end_sample = min(end_sample, _sample_index_for_frame(next_start_frame, frame_to_sample, sample_every) - 1)

        end_sample = max(base_sample, end_sample - end_guard_samples)
        detect_start_sample = min(end_sample, base_sample + start_guard_samples)

        if end_sample <= base_sample:
            continue

        intervals.append({
            "scene": scene,
            "scene_index": scene_index,
            "region_segment_index": region_index,
            "start_sample_index": base_sample,
            "detect_start_sample_index": detect_start_sample,
            "end_sample_index": end_sample,
            "base_frame_no": int(scene.get("base_frame_no", scene["frame_no"])),
            "base_timestamp_sec": float(scene.get("base_timestamp_sec", scene["timestamp_sec"])),
        })
    return intervals


# annotation 미리보기 프레임을 JPEG로 저장
def _save_annotation_preview(
    output_dir: Path,
    scene_index: int,
    annot_index: int,
    frame: np.ndarray,
) -> str:
    filename = f"scene_{scene_index:03d}_annot_{annot_index:02d}.jpg"
    cv2.imwrite(str(output_dir / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return filename


# scene 구간에 대한 결과 딕셔너리 초기값 생성
def _new_scene_result(interval: dict) -> dict:
    scene = interval["scene"]
    return {
        "scene_index": int(scene["scene_index"]),
        "base_frame_no": int(interval["base_frame_no"]),
        "base_timestamp_sec": float(interval["base_timestamp_sec"]),
        "start_sample_index": int(interval["start_sample_index"]),
        "detect_start_sample_index": int(interval["detect_start_sample_index"]),
        "end_sample_index": int(interval["end_sample_index"]),
        "region_segment_index": interval.get("region_segment_index"),
        "annotations": [],
    }


# 씬 구간 안에서 STABLE <-> WRITING 상태를 오가며 판서 캡처 시점을 판정하는 상태 머신
#
# - STABLE: 변화가 거의 없는 상태, 누적 변화가 임계값을 넘으면 WRITING으로 전환
# - WRITING: 판서가 진행 중인 상태, 연속 안정 프레임 수가 stable_required에 도달하고
#   최소 작성 시간(min_writing)을 채우면 그 시점의(가장 변화가 컸던) 프레임을 캡처
# - history는 지금까지 캡처된 안정 상태들을 모두 보관, 현재 프레임이 이전 안정 상태와
#   다시 일치하면(예: 지우개로 지워서 이전 판서로 되돌아간 경우) 그 시점을 별도로 캡처하고
#   history를 그 지점까지 되돌림(erase-as-return 감지)
class AnnotationState:
    def __init__(
        self,
        scene: dict,
        base_decision: np.ndarray,
        base_mask: np.ndarray | None,
        base_frame: np.ndarray,
        cfg: AnnotationConfig,
        sampled_fps: float,
    ):
        self.scene = scene
        self.base_decision = base_decision.copy()
        self.base_mask = base_mask.copy() if base_mask is not None else None
        self.prev_decision = base_decision.copy()
        self.prev_mask = base_mask.copy() if base_mask is not None else None
        self.base_frame = base_frame.copy()
        self.cfg = cfg
        self.stable_required = max(2, int(round(cfg.stable_sec * sampled_fps)))
        self.min_writing = max(1, int(round(cfg.min_annot_sec * sampled_fps)))
        self.min_gap_samples = max(1, int(round(cfg.min_gap_sec * sampled_fps)))
        self.state = "STABLE"
        self.stable_count = 0
        self.writing_count = 0
        self.last_active: dict | None = None
        # 가장 최근 변화뿐 아니라 가장 컸던(peak) 변화도 함께 보관
        # 지우는 도중에는 가장 최근 active 프레임이 이미 거의 지워진 상태일 수 있는데,
        # 실제로 남겨야 할 판서는 이 peak 스냅샷임
        self.peak_active: dict | None = None
        # 안정된 annotation은 다음 비교 기준(baseline)이 됨, 모든 baseline을 보관해두면
        # 지우기를 '깨끗한 프레임을 새 annotation으로 기록'하는 대신 '이전 상태로의
        # 복귀'로 인식할 수 있음
        self.history: list[dict] = [{
            "decision": base_decision.copy(),
            "mask": base_mask.copy() if base_mask is not None else None,
        }]
        self.last_recorded_decision: np.ndarray | None = None
        self.last_recorded_mask: np.ndarray | None = None
        self.last_capture_sample_index = -10**9
        self.annotations: list[dict] = []

    # 프레임 1장을 처리해 상태를 갱신, 캡처가 발생하면 해당 캡처 dict(중복이면 None) 반환
    def process(
        self,
        frame_info: dict,
        frame: np.ndarray,
        decision: np.ndarray,
        mask: np.ndarray | None,
    ) -> dict | None:
        cumulative = _masked_changed_ratio(self.base_decision, self.base_mask, decision, mask, self.cfg.diff_threshold)
        instant = _masked_changed_ratio(self.prev_decision, self.prev_mask, decision, mask, self.cfg.diff_threshold)
        active = instant >= self.cfg.instant_ratio
        sample_index = int(frame_info["sample_index"])
        capture: dict | None = None

        history_match = self._matching_history_index(decision, mask)
        if history_match is not None and history_match != len(self.history) - 1:
            capture = self._capture_peak_before_clear(frame_info)
            self._restore_history(history_match, decision, mask)
            return self._emit_distinct_capture(capture)

        if cumulative >= self.cfg.reject_large_change_ratio:
            self._reset_to(decision, mask)
            return None

        if self.state == "STABLE":
            if cumulative >= self.cfg.cumulative_ratio:
                self.state = "WRITING"
                self.stable_count = 0
                self.writing_count = 1
                self.last_active = {
                    "frame_info": dict(frame_info),
                    "frame": frame.copy(),
                    "decision": decision.copy(),
                    "mask": mask.copy() if mask is not None else None,
                    "cumulative_ratio": cumulative,
                    "instant_ratio": instant,
                }
                self.peak_active = self.last_active
        elif self.state == "WRITING":
            if cumulative < self.cfg.cumulative_ratio:
                capture = self._capture_peak_before_clear(frame_info)
                self._reset_to(decision, mask)
            else:
                self.writing_count += 1
                if active:
                    self.stable_count = 0
                    self.last_active = {
                        "frame_info": dict(frame_info),
                        "frame": frame.copy(),
                        "decision": decision.copy(),
                        "mask": mask.copy() if mask is not None else None,
                        "cumulative_ratio": cumulative,
                        "instant_ratio": instant,
                    }
                    if (
                        self.peak_active is None
                        or cumulative > float(self.peak_active["cumulative_ratio"])
                    ):
                        self.peak_active = self.last_active
                else:
                    self.stable_count += 1

                if (
                    self.stable_count >= self.stable_required
                    and self.writing_count >= self.min_writing
                    and self.last_active is not None
                    and sample_index - self.last_capture_sample_index >= self.min_gap_samples
                ):
                    capture = self.last_active
                    capture["stable_sample_index"] = sample_index
                    capture["stable_frame_no"] = int(frame_info["frame_no"])
                    capture["stable_timestamp_sec"] = float(frame_info["timestamp_sec"])
                    self.base_decision = decision.copy()
                    self.base_mask = mask.copy() if mask is not None else None
                    self.history.append({
                        "decision": self.base_decision.copy(),
                        "mask": self.base_mask.copy() if self.base_mask is not None else None,
                    })
                    self.last_capture_sample_index = sample_index
                    self.state = "STABLE"
                    self.stable_count = 0
                    self.writing_count = 0
                    self.last_active = None
                    self.peak_active = None

        self.prev_decision = decision.copy()
        self.prev_mask = mask.copy() if mask is not None else None
        return self._emit_distinct_capture(capture)

    # 씬 구간이 끝날 때 아직 캡처되지 않은 진행 중(WRITING) 판서가 있으면 마지막으로 캡처
    def flush(self) -> dict | None:
        if (
            self.state == "WRITING"
            and self.writing_count >= self.min_writing
            and self.last_active is not None
        ):
            capture = self.last_active
            capture["stable_sample_index"] = int(capture["frame_info"]["sample_index"])
            capture["stable_frame_no"] = int(capture["frame_info"]["frame_no"])
            capture["stable_timestamp_sec"] = float(capture["frame_info"]["timestamp_sec"])
            return self._emit_distinct_capture(capture)
        return None

    # 상태를 STABLE로 리셋하고 비교 기준 프레임 갱신
    def _reset_to(self, decision: np.ndarray, mask: np.ndarray | None = None) -> None:
        self.state = "STABLE"
        self.stable_count = 0
        self.writing_count = 0
        self.last_active = None
        self.peak_active = None
        self.prev_decision = decision.copy()
        self.prev_mask = mask.copy() if mask is not None else None

    def _matching_history_index(
        self,
        decision: np.ndarray,
        mask: np.ndarray | None,
    ) -> int | None:
        """이 프레임이 이전 안정 상태로 돌아간(annotation erase) 경우 그 history 인덱스 반환"""
        for index in range(len(self.history) - 2, -1, -1):
            state = self.history[index]
            ratio = _masked_changed_ratio(
                state["decision"],
                state["mask"],
                decision,
                mask,
                self.cfg.diff_threshold,
            )
            if ratio < self.cfg.cumulative_ratio:
                return index
        return None

    def _capture_peak_before_clear(self, frame_info: dict) -> dict | None:
        """지우기로 상태가 리셋되기 전, 진행 중이던 annotation의 peak 프레임을 캡처로 확정"""
        if (
            self.state != "WRITING"
            or self.writing_count < self.min_writing
            or self.peak_active is None
        ):
            return None
        sample_index = int(frame_info["sample_index"])
        if sample_index - self.last_capture_sample_index < self.min_gap_samples:
            return None

        capture = dict(self.peak_active)
        capture["capture_reason"] = "annotation_before_clear"
        capture["stable_sample_index"] = sample_index
        capture["stable_frame_no"] = int(frame_info["frame_no"])
        capture["stable_timestamp_sec"] = float(frame_info["timestamp_sec"])
        self.last_capture_sample_index = sample_index
        return capture

    # 지정 history 시점으로 base 상태를 되돌리고, 그 이후 history는 버림
    def _restore_history(
        self,
        history_index: int,
        decision: np.ndarray,
        mask: np.ndarray | None,
    ) -> None:
        state = self.history[history_index]
        self.base_decision = state["decision"].copy()
        self.base_mask = state["mask"].copy() if state["mask"] is not None else None
        self.history = self.history[:history_index + 1]
        self._reset_to(decision, mask)

    def _emit_distinct_capture(self, capture: dict | None) -> dict | None:
        """직전 기록과 눈에 보이는 변화가 거의 없는 반복 캡처는 억제(중복 제거)"""
        if capture is None:
            return None
        decision = capture["decision"]
        mask = capture.get("mask")
        if self.last_recorded_decision is not None:
            changed = _masked_changed_ratio(
                self.last_recorded_decision,
                self.last_recorded_mask,
                decision,
                mask,
                self.cfg.diff_threshold,
            )
            if changed < self.cfg.capture_dedupe_ratio:
                log.debug(
                    "annotation capture suppressed as near-duplicate: changed_ratio=%.6f threshold=%.6f",
                    changed,
                    self.cfg.capture_dedupe_ratio,
                )
                return None
        self.last_recorded_decision = decision.copy()
        self.last_recorded_mask = mask.copy() if mask is not None else None
        return capture


# 하나의 씬 구간에서 샘플 캐시를 순회하며 AnnotationState로 판서 캡처를 감지
def _detect_interval_annotations(
    cache_dir: str,
    manifest: dict,
    interval: dict,
    cfg: AnnotationConfig,
    sampled_fps: float,
    output_dir: str,
) -> dict:
    out_dir = Path(output_dir)
    scene_result = _new_scene_result(interval)
    active_state: AnnotationState | None = None

    for frame_info, frame in _iter_sample_cache_range(
        cache_dir,
        manifest,
        int(interval["start_sample_index"]),
        int(interval["end_sample_index"]),
    ):
        sample_index = int(frame_info["sample_index"])
        decision = _decision_frame(frame, cfg)
        person_mask = _decision_mask(load_person_mask(cache_dir, frame_info), cfg)

        if active_state is None:
            active_state = AnnotationState(interval["scene"], decision, person_mask, frame, cfg, sampled_fps)
            continue

        if sample_index < int(interval["detect_start_sample_index"]):
            continue

        capture = active_state.process(frame_info, frame, decision, person_mask)
        if capture is not None:
            _record_capture(out_dir, interval, capture, 0, scene_result)

    if active_state is not None:
        capture = active_state.flush()
        if capture is not None:
            _record_capture(out_dir, interval, capture, 0, scene_result)

    return scene_result


# 프로세스 풀 worker용 함수, 여러 구간을 순차 처리해 결과 리스트 반환
def _detect_annotation_chunk_worker(args: tuple) -> list[dict]:
    cache_dir, manifest, intervals, cfg_dict, sampled_fps, output_dir = args
    cfg = AnnotationConfig(**cfg_dict)
    results = []
    for interval in intervals:
        results.append(
            _detect_interval_annotations(
                cache_dir,
                manifest,
                interval,
                cfg,
                sampled_fps,
                output_dir,
            )
        )
    return results


# annotation 감지에 쓸 프로세스 수 결정, 환경변수 우선, 없으면 CPU 코어 기반 추정
def _annotation_worker_count(interval_count: int) -> int:
    requested = os.getenv("VLVERIFIER_ANNOT_WORKERS", "0").strip()
    try:
        workers = int(requested)
    except ValueError:
        workers = 0
    if interval_count <= 1:
        return 1
    if workers <= 0:
        cpu_count = os.cpu_count() or 2
        workers = max(1, min(4, cpu_count // 2))
    return max(1, min(workers, interval_count))


# 구간 목록을 worker 수에 맞춰 청크로 분할
def _chunk_intervals(intervals: list[dict], worker_count: int) -> list[list[dict]]:
    if worker_count <= 1:
        return [intervals]
    chunk_size = max(1, (len(intervals) + worker_count - 1) // worker_count)
    return [intervals[i:i + chunk_size] for i in range(0, len(intervals), chunk_size)]


# 전체 씬 구간에 대해(단일 프로세스 또는 병렬로) 판서 캡처를 감지하고 scene_annotations.json 저장
def detect_annotations(
    cache_dir: str,
    scene_path: str,
    output_dir: str,
    cfg: AnnotationConfig | None = None,
) -> dict:
    cfg = cfg or AnnotationConfig()
    manifest = load_sample_cache(cache_dir)
    scene_payload = _load_scenes(scene_path)
    intervals = _build_scene_intervals(scene_payload, manifest, cfg)
    sampled_fps = float(manifest.get("cache", {}).get("sampled_fps") or 1.0)
    worker_count = _annotation_worker_count(len(intervals))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("scene_*_annot_*.jpg"):
        stale.unlink(missing_ok=True)

    log.info(
        "annotation detection start: cache=%s scenes=%s intervals=%s workers=%s",
        cache_dir,
        len(scene_payload["scenes"]),
        len(intervals),
        worker_count,
    )

    scene_results: list[dict] = []
    if worker_count <= 1:
        for interval in intervals:
            scene_results.append(
                _detect_interval_annotations(
                    cache_dir,
                    manifest,
                    interval,
                    cfg,
                    sampled_fps,
                    str(out_dir),
                )
            )
    else:
        chunks = _chunk_intervals(intervals, worker_count)
        cfg_dict = asdict(cfg)
        tasks = [
            (cache_dir, manifest, chunk, cfg_dict, sampled_fps, str(out_dir))
            for chunk in chunks
            if chunk
        ]
        with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
            futures = [executor.submit(_detect_annotation_chunk_worker, task) for task in tasks]
            for future in as_completed(futures):
                scene_results.extend(future.result())

    scene_results.sort(key=lambda item: int(item["scene_index"]))
    total_annotations = 0
    for scene_result in scene_results:
        scene_result["annotations"].sort(
            key=lambda item: (float(item["timestamp_sec"]), int(item["annot_index"]))
        )
        for annotation in scene_result["annotations"]:
            total_annotations += 1
            annotation["global_annot_index"] = total_annotations

    payload = {
        "schema_version": 1,
        "cache_dir": str(cache_dir),
        "scene_path": str(scene_path),
        "config": asdict(cfg),
        "worker_count": worker_count,
        "cache": manifest.get("cache"),
        "source": manifest.get("source"),
        "scene_count": len(scene_results),
        "annotation_count": total_annotations,
        "scenes": scene_results,
    }
    out_path = out_dir / ANNOTATIONS_FILENAME
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log.info("annotation detection done: annotations=%s output=%s", total_annotations, out_path)
    return payload


# 캡처된 판서 프레임을 이미지로 저장하고 scene_result에 레코드 추가
def _record_capture(
    out_dir: Path,
    interval: dict,
    capture: dict,
    global_annot_index: int,
    scene_result: dict,
) -> None:
    scene_index = int(interval["scene_index"])
    annot_index = len(scene_result["annotations"]) + 1
    frame_info = capture["frame_info"]
    filename = _save_annotation_preview(out_dir, scene_index, annot_index, capture["frame"])
    record = {
        "filename": filename,
        "scene_index": scene_index,
        "annot_index": annot_index,
        "global_annot_index": global_annot_index,
        "sample_index": int(frame_info["sample_index"]),
        "frame_no": int(frame_info["frame_no"]),
        "timestamp_sec": round(float(frame_info["timestamp_sec"]), 3),
        "stable_sample_index": int(capture["stable_sample_index"]),
        "stable_frame_no": int(capture["stable_frame_no"]),
        "stable_timestamp_sec": round(float(capture["stable_timestamp_sec"]), 3),
        "base_frame_no": int(interval["base_frame_no"]),
        "base_timestamp_sec": round(float(interval["base_timestamp_sec"]), 3),
        "person_mask_filename": frame_info.get("person_mask_filename"),
        "person_mask_inherited": bool(frame_info.get("person_mask_inherited", False)),
        "person_mask_inherited_distance": frame_info.get("person_mask_inherited_distance"),
        "person_presence_mask_filename": frame_info.get("person_presence_mask_filename"),
        "person_presence_ratio": float(frame_info.get("person_presence_ratio", 0.0) or 0.0),
        "details": {
            "cumulative_ratio": round(float(capture["cumulative_ratio"]), 6),
            "instant_ratio": round(float(capture["instant_ratio"]), 6),
            "person_masked": bool(frame_info.get("person_mask_filename")),
            "capture_reason": str(capture.get("capture_reason", "stable_annotation")),
        },
    }
    scene_result["annotations"].append(record)
    log.debug(
        "[scene %03d] annot_%02d @ %.3fs frame=%s",
        scene_index,
        annot_index,
        float(record["timestamp_sec"]),
        int(record["frame_no"]),
    )


# CLI 인자 파싱
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="샘플 캐시 씬 구간에서 annotation 프레임 감지")
    parser.add_argument("--cache", required=True, help="샘플 캐시 디렉터리")
    parser.add_argument("--scenes", required=True, help="Step 2에서 생성된 scene_transitions.json")
    parser.add_argument("--output", "-o", required=True, help="출력 annotation probe 디렉터리")
    parser.add_argument("--cumulative-ratio", type=float, default=AnnotationConfig.cumulative_ratio)
    parser.add_argument("--instant-ratio", type=float, default=AnnotationConfig.instant_ratio)
    parser.add_argument("--stable-sec", type=float, default=AnnotationConfig.stable_sec)
    parser.add_argument("--min-annot-sec", type=float, default=AnnotationConfig.min_annot_sec)
    parser.add_argument("--min-gap-sec", type=float, default=AnnotationConfig.min_gap_sec)
    parser.add_argument("--scene-start-guard-sec", type=float, default=AnnotationConfig.scene_start_guard_sec)
    parser.add_argument("--scene-end-guard-sec", type=float, default=AnnotationConfig.scene_end_guard_sec)
    parser.add_argument("--workers", type=int, help="이번 실행에 한해 VLVERIFIER_ANNOT_WORKERS 값을 덮어씀")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


# CLI 진입점, 인자로 AnnotationConfig 구성 후 detect_annotations 실행
def main() -> None:
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = AnnotationConfig(
        cumulative_ratio=max(0.0, args.cumulative_ratio),
        instant_ratio=max(0.0, args.instant_ratio),
        stable_sec=max(0.0, args.stable_sec),
        min_annot_sec=max(0.0, args.min_annot_sec),
        min_gap_sec=max(0.0, args.min_gap_sec),
        scene_start_guard_sec=max(0.0, args.scene_start_guard_sec),
        scene_end_guard_sec=max(0.0, args.scene_end_guard_sec),
    )
    if args.workers is not None:
        os.environ["VLVERIFIER_ANNOT_WORKERS"] = str(max(1, args.workers))
    detect_annotations(args.cache, args.scenes, args.output, cfg)


if __name__ == "__main__":
    main()
