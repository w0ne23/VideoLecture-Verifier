"""
Detect annotation capture frames inside Step 2 scene intervals.

Step 3 of the staged slide extraction pipeline:

  sampled frame cache + scene_transitions.json
    -> scene_annotations.json
    -> small preview annotation images

This pass does not materialize original-resolution frames. It only records the
original frame_no values that Step 4 should extract from the source video.

Usage:
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
    from .sample_cache import iter_sample_cache, load_sample_cache
    from .person_masks import load_person_mask, masked_pair
except ImportError:  # pragma: no cover - allows direct script execution
    from sample_cache import iter_sample_cache, load_sample_cache
    from person_masks import load_person_mask, masked_pair


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


ANNOTATIONS_FILENAME = "scene_annotations.json"


@dataclass
class AnnotationConfig:
    diff_threshold: int = 15
    cumulative_ratio: float = 0.001
    instant_ratio: float = 0.0001
    stable_sec: float = 0.7
    min_annot_sec: float = 0.2
    min_gap_sec: float = 1.5
    scene_start_guard_sec: float = 0.5
    scene_end_guard_sec: float = 1.0
    reject_large_change_ratio: float = 0.16
    crop_left: float = 0.12
    crop_top: float = 0.06
    crop_right: float = 0.94
    crop_bottom: float = 0.90


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


def _content_region(frame: np.ndarray, cfg: AnnotationConfig) -> np.ndarray:
    h, w = frame.shape[:2]
    x0 = max(0, min(w - 1, int(w * cfg.crop_left)))
    y0 = max(0, min(h - 1, int(h * cfg.crop_top)))
    x1 = max(x0 + 1, min(w, int(w * cfg.crop_right)))
    y1 = max(y0 + 1, min(h, int(h * cfg.crop_bottom)))
    return frame[y0:y1, x0:x1]


def _decision_frame(frame: np.ndarray, cfg: AnnotationConfig) -> np.ndarray:
    cropped = _content_region(frame, cfg)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _decision_mask(mask: np.ndarray | None, cfg: AnnotationConfig) -> np.ndarray | None:
    if mask is None:
        return None
    return _content_region(mask.astype(np.uint8), cfg).astype(bool)


def _changed_ratio(frame_a: np.ndarray, frame_b: np.ndarray, threshold: int) -> float:
    diff = cv2.absdiff(frame_a, frame_b)
    return float(np.sum(diff > threshold) / diff.size)


def _masked_changed_ratio(
    frame_a: np.ndarray,
    mask_a: np.ndarray | None,
    frame_b: np.ndarray,
    mask_b: np.ndarray | None,
    threshold: int,
) -> float:
    masked_a, masked_b = masked_pair(frame_a, mask_a, frame_b, mask_b)
    return _changed_ratio(masked_a, masked_b, threshold)


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
    """Read a bounded interval from the sampled MJPG cache.

    Random seeking is intentionally limited to the generated sample cache, not
    the original lecture video. The cache is our analysis coordinate system and
    is much safer to seek than arbitrary source encodings.
    """
    cache_path = Path(cache_dir)
    video_path = cache_path / manifest["video_filename"]
    frames = manifest.get("frames", [])
    if not frames:
        return

    start_sample_index = max(1, int(start_sample_index))
    end_sample_index = min(int(end_sample_index), len(frames))
    if end_sample_index < start_sample_index:
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open sampled cache video: {video_path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_sample_index - 1)
        for offset, sample_index in enumerate(range(start_sample_index, end_sample_index + 1)):
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError(
                    f"Sample cache video ended early: {video_path} sample={sample_index}"
                )
            frame_info = frames[start_sample_index - 1 + offset]
            yield frame_info, frame
    finally:
        cap.release()


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


def _save_annotation_preview(
    output_dir: Path,
    scene_index: int,
    annot_index: int,
    frame: np.ndarray,
) -> str:
    filename = f"scene_{scene_index:03d}_annot_{annot_index:02d}.jpg"
    cv2.imwrite(str(output_dir / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return filename


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
        self.last_capture_sample_index = -10**9
        self.annotations: list[dict] = []

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
        elif self.state == "WRITING":
            if cumulative < self.cfg.cumulative_ratio:
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
                    self.last_capture_sample_index = sample_index
                    self.state = "STABLE"
                    self.stable_count = 0
                    self.writing_count = 0
                    self.last_active = None

        self.prev_decision = decision.copy()
        self.prev_mask = mask.copy() if mask is not None else None
        return capture

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
            return capture
        return None

    def _reset_to(self, decision: np.ndarray, mask: np.ndarray | None = None) -> None:
        self.state = "STABLE"
        self.stable_count = 0
        self.writing_count = 0
        self.last_active = None
        self.prev_decision = decision.copy()
        self.prev_mask = mask.copy() if mask is not None else None


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


def _annotation_worker_count(interval_count: int) -> int:
    requested = os.getenv("VERILEC_ANNOT_WORKERS", "0").strip()
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


def _chunk_intervals(intervals: list[dict], worker_count: int) -> list[list[dict]]:
    if worker_count <= 1:
        return [intervals]
    chunk_size = max(1, (len(intervals) + worker_count - 1) // worker_count)
    return [intervals[i:i + chunk_size] for i in range(0, len(intervals), chunk_size)]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect annotation frames from sampled cache scene intervals.")
    parser.add_argument("--cache", required=True, help="Sample cache directory")
    parser.add_argument("--scenes", required=True, help="scene_transitions.json from Step 2")
    parser.add_argument("--output", "-o", required=True, help="Output annotation probe directory")
    parser.add_argument("--cumulative-ratio", type=float, default=AnnotationConfig.cumulative_ratio)
    parser.add_argument("--instant-ratio", type=float, default=AnnotationConfig.instant_ratio)
    parser.add_argument("--stable-sec", type=float, default=AnnotationConfig.stable_sec)
    parser.add_argument("--min-annot-sec", type=float, default=AnnotationConfig.min_annot_sec)
    parser.add_argument("--min-gap-sec", type=float, default=AnnotationConfig.min_gap_sec)
    parser.add_argument("--scene-start-guard-sec", type=float, default=AnnotationConfig.scene_start_guard_sec)
    parser.add_argument("--scene-end-guard-sec", type=float, default=AnnotationConfig.scene_end_guard_sec)
    parser.add_argument("--workers", type=int, help="Override VERILEC_ANNOT_WORKERS for this run")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


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
        os.environ["VERILEC_ANNOT_WORKERS"] = str(max(1, args.workers))
    detect_annotations(args.cache, args.scenes, args.output, cfg)


if __name__ == "__main__":
    main()
