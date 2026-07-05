"""
Materialize selected base/annotation frame numbers from the original video.

Step 4 of the staged slide extraction pipeline:

  input.mp4 + scene_transitions.json + scene_annotations.json
    -> scene_###_base.jpg
    -> scene_###_annot_##.jpg
    -> scene_###_video.jpg
    -> metadata.json

The important rule is that frames are read sequentially from the original
video. We avoid random OpenCV seeks for normal extraction because they can
return visually different frames on some encoded lecture videos.

Usage:
    python -m pipeline.materialize_from_cache \
        --input lecture.mp4 \
        --scenes scene_transitions.json \
        --annotations scene_annotations.json \
        --regions timeline_segments.json \
        --output slides/
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import cv2


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


METADATA_FILENAME = "metadata.json"
MATERIALIZED_FILENAME = "materialized_frames.json"


def _load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _timestamp_from_frame(frame_no: int, fps: float) -> float:
    return frame_no / fps if fps > 0 else 0.0


def _base_record(scene: dict, fps: float, output_scene_index: int | None = None) -> dict:
    source_scene_index = int(scene["scene_index"])
    scene_index = int(output_scene_index or source_scene_index)
    frame_no = int(scene.get("base_frame_no", scene.get("frame_no")))
    timestamp = float(scene.get("base_timestamp_sec", scene.get("timestamp_sec", _timestamp_from_frame(frame_no, fps))))
    return {
        "filename": f"scene_{scene_index:03d}_base.jpg",
        "scene_number": scene_index,
        "scene_index": scene_index,
        "source_scene_index": source_scene_index,
        "slide_index": scene_index,
        "slide_number": scene_index,
        "timestamp_sec": round(timestamp, 3),
        "frame_no": frame_no,
        "capture_type": "base",
        "scene_type": "slide",
        "annot_index": 0,
        "scene_annot_index": 0,
        "source": "step2_scene_base",
        "region_segment_index": scene.get("region_segment_index"),
        "sample_index": scene.get("sample_index"),
        "person_mask_filename": scene.get("person_mask_filename"),
        "person_mask_inherited": bool(scene.get("person_mask_inherited", False)),
        "person_mask_inherited_distance": scene.get("person_mask_inherited_distance"),
        "person_presence_mask_filename": scene.get("person_presence_mask_filename"),
        "person_presence_ratio": float(scene.get("person_presence_ratio", 0.0) or 0.0),
        "duplicate_of": [],
    }


def _annotation_records(
    annotation_payload: dict,
    scene_index_map: dict[int, int] | None = None,
    scene_lookup: dict[int, dict] | None = None,
) -> list[dict]:
    scene_index_map = scene_index_map or {}
    scene_lookup = scene_lookup or {}
    records: list[dict] = []
    for scene in annotation_payload.get("scenes", []):
        source_scene_index = int(scene["scene_index"])
        scene_index = int(scene_index_map.get(source_scene_index, source_scene_index))
        base_scene = scene_lookup.get(source_scene_index, {})
        for annot in scene.get("annotations", []):
            annot_index = int(annot["annot_index"])
            records.append({
                "filename": f"scene_{scene_index:03d}_annot_{annot_index:02d}.jpg",
                "scene_number": scene_index,
                "scene_index": scene_index,
                "source_scene_index": source_scene_index,
                "slide_index": scene_index,
                "slide_number": scene_index,
                "timestamp_sec": round(float(annot["timestamp_sec"]), 3),
                "frame_no": int(annot["frame_no"]),
                "capture_type": "annotation",
                "scene_type": "slide",
                "annot_index": annot_index,
                "scene_annot_index": annot_index,
                "global_annot_index": int(annot.get("global_annot_index", 0) or 0),
                "base_frame_no": int(annot.get("base_frame_no", scene.get("base_frame_no", 0)) or 0),
                "base_timestamp_sec": round(float(annot.get("base_timestamp_sec", scene.get("base_timestamp_sec", 0.0)) or 0.0), 3),
                "sample_index": annot.get("sample_index", base_scene.get("sample_index")),
                "person_mask_filename": annot.get("person_mask_filename", base_scene.get("person_mask_filename")),
                "person_mask_inherited": bool(annot.get("person_mask_inherited", base_scene.get("person_mask_inherited", False))),
                "person_mask_inherited_distance": annot.get("person_mask_inherited_distance", base_scene.get("person_mask_inherited_distance")),
                "person_presence_mask_filename": annot.get("person_presence_mask_filename", base_scene.get("person_presence_mask_filename")),
                "person_presence_ratio": float(annot.get("person_presence_ratio", base_scene.get("person_presence_ratio", 0.0)) or 0.0),
                "source": "step3_annotation",
                "duplicate_of": [],
            })
    return records


def _load_video_segments(regions_path: str | None, fps: float) -> list[dict]:
    if not regions_path:
        return []
    payload = _load_json(regions_path)
    videos: list[dict] = []
    for seg in payload.get("segments", []):
        if seg.get("type") != "video":
            continue
        start_frame = int(seg["start_frame_no"])
        end_frame = int(seg["end_frame_no"])
        frame_no = max(1, int(round((start_frame + end_frame) / 2)))
        timestamp = float(seg.get("start_sec", _timestamp_from_frame(start_frame, fps)))
        videos.append({
            "kind": "video",
            "source_segment_index": int(seg["segment_index"]),
            "timestamp_sec": timestamp,
            "frame_no": frame_no,
            "video_start_sec": float(seg["start_sec"]),
            "video_end_sec": float(seg["end_sec"]),
            "video_start_frame_no": start_frame,
            "video_end_frame_no": end_frame,
        })
    return sorted(videos, key=lambda item: item["timestamp_sec"])


def _timeline_units(scene_payload: dict, video_segments: list[dict], fps: float) -> tuple[list[dict], dict[int, int]]:
    units: list[dict] = []
    for scene in scene_payload.get("scenes", []):
        frame_no = int(scene.get("base_frame_no", scene.get("frame_no")))
        timestamp = float(scene.get("base_timestamp_sec", scene.get("timestamp_sec", _timestamp_from_frame(frame_no, fps))))
        units.append({
            "kind": "slide",
            "timestamp_sec": timestamp,
            "source_scene_index": int(scene["scene_index"]),
            "scene": scene,
        })
    units.extend(video_segments)
    units.sort(key=lambda item: (float(item["timestamp_sec"]), 0 if item["kind"] == "slide" else 1))

    scene_index_map: dict[int, int] = {}
    for output_index, unit in enumerate(units, start=1):
        unit["scene_index"] = output_index
        if unit["kind"] == "slide":
            scene_index_map[int(unit["source_scene_index"])] = output_index
    return units, scene_index_map


def _video_record(unit: dict) -> dict:
    scene_index = int(unit["scene_index"])
    return {
        "filename": f"scene_{scene_index:03d}_video.jpg",
        "scene_number": scene_index,
        "scene_index": scene_index,
        "slide_index": scene_index,
        "slide_number": scene_index,
        "timestamp_sec": round(float(unit["timestamp_sec"]), 3),
        "frame_no": int(unit["frame_no"]),
        "capture_type": "base",
        "scene_type": "video",
        "annot_index": 0,
        "scene_annot_index": 0,
        "source": "step1_video_segment",
        "region_segment_index": int(unit["source_segment_index"]),
        "video_start_sec": round(float(unit["video_start_sec"]), 3),
        "video_end_sec": round(float(unit["video_end_sec"]), 3),
        "video_start_frame_no": int(unit["video_start_frame_no"]),
        "video_end_frame_no": int(unit["video_end_frame_no"]),
        "duplicate_of": [],
    }


def build_metadata(
    scene_payload: dict,
    annotation_payload: dict,
    fps: float,
    regions_path: str | None = None,
) -> list[dict]:
    video_segments = _load_video_segments(regions_path, fps)
    units, scene_index_map = _timeline_units(scene_payload, video_segments, fps)
    scene_lookup = {int(scene["scene_index"]): scene for scene in scene_payload.get("scenes", [])}
    bases = [
        _base_record(unit["scene"], fps, output_scene_index=int(unit["scene_index"]))
        for unit in units
        if unit["kind"] == "slide"
    ]
    videos = [_video_record(unit) for unit in units if unit["kind"] == "video"]
    annotations = _annotation_records(annotation_payload, scene_index_map, scene_lookup)
    metadata = bases + annotations
    metadata.extend(videos)
    metadata.sort(key=lambda item: (int(item["scene_index"]), int(item["annot_index"]), int(item["frame_no"])))

    by_scene: dict[int, list[dict]] = {}
    for item in metadata:
        by_scene.setdefault(int(item["scene_index"]), []).append(item)

    for scene_index, items in by_scene.items():
        annots = [item for item in items if item["capture_type"] == "annotation"]
        for item in items:
            item["scene_annotation_count"] = len(annots)
            item["scene_annotation_start_index"] = 1 if annots else 0
            item["scene_annotation_end_index"] = len(annots)
            item["slide_annotation_count_total"] = len(annots)
            if item["capture_type"] != "annotation":
                item["slide_annot_index"] = 0
            else:
                item["slide_annot_index"] = int(item["annot_index"])

    unit_starts = {
        int(unit["scene_index"]): float(unit["timestamp_sec"])
        for unit in units
    }
    source_duration = float(scene_payload.get("source", {}).get("duration_sec") or 0.0)
    sorted_indices = sorted(unit_starts)
    unit_ends: dict[int, float] = {}
    for i, scene_index in enumerate(sorted_indices):
        unit = next(u for u in units if int(u["scene_index"]) == scene_index)
        if unit["kind"] == "video":
            unit_ends[scene_index] = float(unit["video_end_sec"])
        elif i + 1 < len(sorted_indices):
            unit_ends[scene_index] = unit_starts[sorted_indices[i + 1]]
        else:
            if unit["kind"] == "video":
                unit_ends[scene_index] = float(unit["video_end_sec"])
            else:
                unit_ends[scene_index] = source_duration

    for item in metadata:
        scene_index = int(item["scene_index"])
        start_sec = round(unit_starts.get(scene_index, float(item["timestamp_sec"])), 3)
        end_sec = round(unit_ends.get(scene_index, start_sec), 3)
        # Step 4 materializes timeline scenes only. True slide identity is assigned
        # later by duplicate/canonical slide grouping. Keep slide_* as a temporary
        # compatibility alias for downstream code that has not migrated to scene_*.
        item["slide_start_sec"] = start_sec
        item["slide_end_sec"] = end_sec
        item["scene_start_sec"] = start_sec
        item["scene_end_sec"] = end_sec
    return metadata


def _save_frame(frame, output_path: Path) -> None:
    ok = cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"Failed to write image: {output_path}")


def _read_frame_by_timestamp_ffmpeg(input_path: str, timestamp_sec: float):
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
                f"{max(0.0, timestamp_sec):.6f}",
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
        return cv2.imread(str(tmp_path), cv2.IMREAD_COLOR)
    except Exception:
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


def _materialize_sequential(input_path: str, output_dir: Path, metadata: list[dict]) -> set[int]:
    targets_by_frame: dict[int, list[dict]] = {}
    for item in metadata:
        frame_no = int(item["frame_no"])
        if frame_no <= 0:
            raise ValueError(f"Invalid frame_no in metadata: {item}")
        targets_by_frame.setdefault(frame_no, []).append(item)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    saved: set[int] = set()
    wanted = set(targets_by_frame)
    frame_no = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            targets = targets_by_frame.get(frame_no)
            if not targets:
                continue
            for item in targets:
                _save_frame(frame, output_dir / item["filename"])
            saved.add(frame_no)
            if saved == wanted:
                break
    finally:
        cap.release()

    return saved


def _materialize_missing_with_timestamp(input_path: str, output_dir: Path, metadata: list[dict], saved: set[int]) -> set[int]:
    targets_by_frame: dict[int, list[dict]] = {}
    for item in metadata:
        targets_by_frame.setdefault(int(item["frame_no"]), []).append(item)

    recovered: set[int] = set()
    for frame_no in sorted(set(targets_by_frame) - saved):
        targets = targets_by_frame[frame_no]
        timestamp = min(float(item["timestamp_sec"]) for item in targets)
        frame = None
        for offset in (0.0, -0.05, 0.05, -0.2, 0.2, -0.5, 0.5):
            frame = _read_frame_by_timestamp_ffmpeg(input_path, timestamp + offset)
            if frame is not None:
                if offset:
                    log.warning("frame_no=%s recovered by timestamp %.3fs", frame_no, timestamp + offset)
                break
        if frame is None:
            continue
        for item in targets:
            _save_frame(frame, output_dir / item["filename"])
        recovered.add(frame_no)
    return recovered


def materialize_frames(
    input_path: str,
    scenes_path: str,
    annotations_path: str,
    output_dir: str,
    regions_path: str | None = None,
) -> dict:
    scene_payload = _load_json(scenes_path)
    annotation_payload = _load_json(annotations_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or scene_payload.get("source", {}).get("fps") or 0.0)
    finally:
        cap.release()
    if fps <= 0:
        raise RuntimeError(f"Cannot read FPS from video: {input_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("scene_*.jpg"):
        stale.unlink(missing_ok=True)
    for stale in out_dir.glob("slide_*.jpg"):
        stale.unlink(missing_ok=True)

    metadata = build_metadata(scene_payload, annotation_payload, fps, regions_path=regions_path)
    target_frame_count = len({int(item["frame_no"]) for item in metadata})
    log.info(
        "materialize start: input=%s targets=%s records=%s output=%s",
        input_path,
        target_frame_count,
        len(metadata),
        output_dir,
    )

    saved = _materialize_sequential(input_path, out_dir, metadata)
    recovered = _materialize_missing_with_timestamp(input_path, out_dir, metadata, saved)
    all_saved = saved | recovered
    missing = sorted({int(item["frame_no"]) for item in metadata} - all_saved)
    if missing:
        raise RuntimeError(f"Failed to materialize frame_no values: {missing[:20]}{'...' if len(missing) > 20 else ''}")

    metadata_path = out_dir / METADATA_FILENAME
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    payload = {
        "schema_version": 1,
        "input_path": str(input_path),
        "scenes_path": str(scenes_path),
        "annotations_path": str(annotations_path),
        "regions_path": str(regions_path) if regions_path else None,
        "output_dir": str(output_dir),
        "fps": fps,
        "record_count": len(metadata),
        "target_frame_count": target_frame_count,
        "sequential_saved_frame_count": len(saved),
        "fallback_saved_frame_count": len(recovered),
        "metadata_path": str(metadata_path),
    }
    with open(out_dir / MATERIALIZED_FILENAME, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log.info(
        "materialize done: records=%s frames=%s output=%s",
        len(metadata),
        len(all_saved),
        output_dir,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize Step 2/3 frame numbers from the original video.")
    parser.add_argument("--input", "-i", required=True, help="Input .mp4 path")
    parser.add_argument("--scenes", required=True, help="scene_transitions.json from Step 2")
    parser.add_argument("--annotations", required=True, help="scene_annotations.json from Step 3")
    parser.add_argument("--regions", help="timeline_segments.json from Step 1; video segments are materialized as scenes")
    parser.add_argument("--output", "-o", required=True, help="Output slides directory")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    materialize_frames(args.input, args.scenes, args.annotations, args.output, regions_path=args.regions)


if __name__ == "__main__":
    main()
