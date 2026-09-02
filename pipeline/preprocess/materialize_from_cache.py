"""
선택된 base/annotation frame_no를 원본 영상에서 실체화(materialize)

단계별 슬라이드 추출 파이프라인의 Step 4:

  input.mp4 + scene_transitions.json + scene_annotations.json
    -> scene_###_base.jpg
    -> scene_###_annot_##.jpg
    -> scene_###_video.jpg
    -> metadata.json

중요한 원칙은 원본 영상에서 프레임을 순차적으로 읽는 것, 일반 추출에서는
OpenCV 임의 seek을 피하는데, 일부 인코딩된 강의 영상에서 시각적으로 다른
프레임을 반환할 수 있기 때문

사용법:
    python -m pipeline.materialize_from_cache \
        --input lecture.mp4 \
        --scenes scene_transitions.json \
        --annotations scene_annotations.json \
        --regions timeline_segments.json \
        --output slides/
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import os
from pathlib import Path

import cv2

try:
    from .sample_cache import iter_sample_cache_range, iter_sample_cache_selected_positions
    from .video_decode import iter_video_frames, read_frame_at_timestamp, read_video_metadata
except ImportError:  # pragma: no cover - direct script execution fallback
    from sample_cache import iter_sample_cache_range, iter_sample_cache_selected_positions
    from video_decode import iter_video_frames, read_frame_at_timestamp, read_video_metadata


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


METADATA_FILENAME = "metadata.json"
MATERIALIZED_FILENAME = "materialized_frames.json"
DEFAULT_DECODE_BACKEND = os.getenv("VLVERIFIER_SLIDE_DECODE_BACKEND", "auto")


# JSON 파일 로드, 없으면 예외
def _load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# frame_no를 fps 기준 타임스탬프(초)로 변환
def _timestamp_from_frame(frame_no: int, fps: float) -> float:
    return frame_no / fps if fps > 0 else 0.0


# metadata에 sample_index가 없는 항목에 대해 캐시 manifest에서 가장 가까운 frame_no의
# sample_index를 채움, 채운 개수 반환
def _attach_missing_sample_indices_from_manifest(metadata: list[dict], cache_manifest: dict) -> int:
    frames = cache_manifest.get("frames", []) or []
    frame_to_sample: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []
    for frame in frames:
        try:
            frame_no = int(frame["frame_no"])
            sample_index = int(frame["sample_index"])
        except (KeyError, TypeError, ValueError):
            continue
        frame_to_sample[frame_no] = sample_index
        pairs.append((frame_no, sample_index))
    if not pairs:
        return 0

    pairs.sort()
    frame_nos = [frame_no for frame_no, _ in pairs]
    filled = 0
    for item in metadata:
        if int(item.get("sample_index") or 0) > 0:
            continue
        try:
            frame_no = int(item.get("frame_no") or 0)
        except (TypeError, ValueError):
            continue
        if frame_no <= 0:
            continue

        sample_index = frame_to_sample.get(frame_no)
        if sample_index is None:
            pos = bisect.bisect_left(frame_nos, frame_no)
            candidates = []
            if pos < len(pairs):
                candidates.append(pairs[pos])
            if pos > 0:
                candidates.append(pairs[pos - 1])
            if not candidates:
                continue
            _, sample_index = min(candidates, key=lambda pair: abs(pair[0] - frame_no))
        item["sample_index"] = int(sample_index)
        item["sample_index_inferred"] = True
        filled += 1
    return filled


# scene(슬라이드 base)을 metadata 레코드 형식으로 변환
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


# annotation 목록을 metadata 레코드 형식으로 변환, scene_index_map으로 재정렬된 scene 번호에 맞춤
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
                # Step 3가 이 annotation 프레임을 선택한 이유를 보존, 특히 지우기 직전에
                # 유지된 annotation은 후속 metadata에서 일반적인 안정 annotation과
                # 구분 가능해야 함
                "annotation_capture_reason": str(
                    (annot.get("details") or {}).get("capture_reason", "stable_annotation")
                ),
                "annotation_stable_frame_no": int(annot.get("stable_frame_no", annot["frame_no"])),
                "annotation_stable_timestamp_sec": round(
                    float(annot.get("stable_timestamp_sec", annot["timestamp_sec"])), 3
                ),
                "source": "step3_annotation",
                "duplicate_of": [],
            })
    return records


# timeline_segments.json에서 video 타입 구간만 골라 중간 프레임 기준 레코드로 변환
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


# 슬라이드 scene과 video 구간을 타임스탬프 순으로 통합 정렬, 통합된 순서로 새 scene_index 부여
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


# video 구간을 metadata 레코드 형식으로 변환
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


# scene/annotation/video 구간을 통합해 metadata.json에 들어갈 전체 레코드 목록 생성
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
    annotations = _annotation_records(annotation_payload, scene_index_map, scene_lookup)
    metadata = bases + annotations
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
        # Step 4는 타임라인 scene만 실체화, 실제 슬라이드 정체성은 이후 중복/정규 슬라이드
        # 그룹화 단계에서 부여됨, scene_*로 아직 이관되지 않은 후속 코드를 위해 slide_*를
        # 임시 호환 별칭으로 유지
        item["slide_start_sec"] = start_sec
        item["slide_end_sec"] = end_sec
        item["scene_start_sec"] = start_sec
        item["scene_end_sec"] = end_sec
    return metadata


# 프레임을 JPEG로 저장, 실패 시 예외
def _save_frame(frame, output_path: Path) -> None:
    ok = cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"Failed to write image: {output_path}")


# 원본 영상을 처음부터 순차 디코딩하며 필요한 frame_no가 나올 때마다 저장, 모든 대상을
# 저장하면 조기 종료
def _materialize_sequential(
    input_path: str,
    output_dir: Path,
    metadata: list[dict],
    *,
    fps: float,
    width: int,
    height: int,
    decode_backend: str,
) -> set[int]:
    targets_by_frame: dict[int, list[dict]] = {}
    for item in metadata:
        frame_no = int(item["frame_no"])
        if frame_no <= 0:
            raise ValueError(f"Invalid frame_no in metadata: {item}")
        targets_by_frame.setdefault(frame_no, []).append(item)

    saved: set[int] = set()
    wanted = set(targets_by_frame)
    frame_iter, active_backend = iter_video_frames(
        input_path,
        fps=fps,
        width=width,
        height=height,
        sample_every=1,
        decode_backend=decode_backend,
    )
    log.info("materialize decode backend: %s", active_backend)
    for frame_no, _, frame in frame_iter:
        targets = targets_by_frame.get(int(frame_no))
        if not targets:
            continue
        for item in targets:
            _save_frame(frame, output_dir / item["filename"])
        saved.add(int(frame_no))
        if saved == wanted:
            break
    return saved


# 순차 디코딩으로 못 찾은 frame_no를 타임스탬프 기준 seek로 재시도 (약간의 오프셋을 순서대로 시도)
def _materialize_missing_with_timestamp(
    input_path: str,
    output_dir: Path,
    metadata: list[dict],
    saved: set[int],
    *,
    width: int,
    height: int,
    decode_backend: str,
) -> set[int]:
    targets_by_frame: dict[int, list[dict]] = {}
    for item in metadata:
        targets_by_frame.setdefault(int(item["frame_no"]), []).append(item)

    recovered: set[int] = set()
    for frame_no in sorted(set(targets_by_frame) - saved):
        targets = targets_by_frame[frame_no]
        timestamp = min(float(item["timestamp_sec"]) for item in targets)
        frame = None
        for offset in (0.0, -0.05, 0.05, -0.2, 0.2, -0.5, 0.5):
            frame = read_frame_at_timestamp(
                input_path,
                timestamp + offset,
                width=width,
                height=height,
                decode_backend=decode_backend,
            )
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


# metadata를 sample_index 기준으로 그룹화, sample_index가 없는 레코드가 있으면 예외
def _targets_by_sample_index(metadata: list[dict]) -> dict[int, list[dict]]:
    targets_by_sample: dict[int, list[dict]] = {}
    missing: list[dict] = []
    for item in metadata:
        sample_index = int(item.get("sample_index") or 0)
        if sample_index <= 0:
            missing.append(item)
            continue
        targets_by_sample.setdefault(sample_index, []).append(item)
    if missing:
        examples = [
            {
                "filename": item.get("filename"),
                "scene_index": item.get("scene_index"),
                "capture_type": item.get("capture_type"),
                "frame_no": item.get("frame_no"),
            }
            for item in missing[:5]
        ]
        raise RuntimeError(
            "Cannot materialize from sample cache because some metadata records "
            f"do not have sample_index. examples={examples}"
        )
    return targets_by_sample


# 필요한 sample_index 위치만 임의 접근으로 읽어 저장 (전체 스캔 없이 소수의 review 이미지만 빠르게 생성)
def _materialize_from_sample_cache_random(
    cache_dir: str | Path,
    output_dir: Path,
    metadata: list[dict],
) -> set[int]:
    cache_path = Path(cache_dir)
    manifest_path = cache_path / "sampled_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Sample cache manifest not found: {manifest_path}")

    manifest = _load_json(manifest_path)

    targets_by_sample = _targets_by_sample_index(metadata)
    wanted = set(targets_by_sample)
    saved: set[int] = set()

    # review 대상은 드문드문 흩어져 있음, manifest상의 위치만(독립적으로 쓰인 캐시
    # 세그먼트별로 그룹화해) 읽어서, 수백 장의 review 이미지를 만들려고 2만 7천 프레임
    # 전체를 스캔하지 않도록 함
    target_positions = [sample_index - 1 for sample_index in wanted]
    for _, frame_info, frame in iter_sample_cache_selected_positions(cache_path, target_positions):
        sample_index = int(frame_info.get("sample_index") or 0)
        if sample_index not in wanted:
            continue
        for item in targets_by_sample[sample_index]:
            _save_frame(frame, output_dir / item["filename"])
        saved.add(sample_index)
        if saved == wanted:
            break

    return saved


def _materialize_from_sample_cache_sequential(
    cache_dir: str | Path,
    output_dir: Path,
    metadata: list[dict],
    already_saved: set[int],
) -> set[int]:
    # 아직 못 찾은 대상을 위해 manifest에 매핑된 캐시를 순차적으로 폴백 스캔

    cache_path = Path(cache_dir)
    manifest = _load_json(cache_path / "sampled_manifest.json")
    frames = manifest.get("frames", [])
    targets_by_sample = _targets_by_sample_index(metadata)
    missing = set(targets_by_sample) - set(already_saved)
    recovered: set[int] = set()

    if not missing:
        return recovered

    for _, frame_info, frame in iter_sample_cache_range(cache_path, 0, len(frames)):
        sample_index = int(frame_info["sample_index"])
        if sample_index not in missing:
            continue
        for item in targets_by_sample[sample_index]:
            _save_frame(frame, output_dir / item["filename"])
        recovered.add(sample_index)
        if recovered == missing:
            break

    return recovered


def materialize_frames_from_sample_cache(
    cache_dir: str,
    scenes_path: str,
    annotations_path: str,
    output_dir: str,
    regions_path: str | None = None,
) -> dict:
    # Step 0 샘플 캐시로부터 Step 4A review 이미지를 실체화
    #
    # build_metadata()와 동일한 metadata를 유지: frame_no/timestamp_sec는 원본 영상
    # 좌표계 그대로, 임시 review 이미지 픽셀만 sample_index와 그 세그먼트 매핑을 이용해
    # sample_cache에서 로드
    #
    # 이 함수는 Step 4A review 전용, Step 4B 최종 출력은 원본 영상 기반
    # materialize_frames()를 계속 사용해야 함

    scene_payload = _load_json(scenes_path)
    annotation_payload = _load_json(annotations_path)
    cache_manifest = _load_json(Path(cache_dir) / "sampled_manifest.json")

    source_meta = cache_manifest.get("source", {}) or scene_payload.get("source", {})
    fps = float(source_meta.get("fps") or scene_payload.get("source", {}).get("fps") or 0.0)
    if fps <= 0:
        raise RuntimeError(f"Cannot read source FPS from sample cache or scene payload: {cache_dir}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("scene_*.jpg"):
        stale.unlink(missing_ok=True)
    for stale in out_dir.glob("slide_*.jpg"):
        stale.unlink(missing_ok=True)

    metadata = build_metadata(scene_payload, annotation_payload, fps, regions_path=regions_path)
    inferred_samples = _attach_missing_sample_indices_from_manifest(metadata, cache_manifest)
    if inferred_samples:
        log.info("sample-cache materialize inferred sample_index for %s metadata records", inferred_samples)
    targets_by_sample = _targets_by_sample_index(metadata)
    target_sample_count = len(targets_by_sample)

    log.info(
        "sample-cache materialize start: cache=%s targets=%s records=%s output=%s",
        cache_dir,
        target_sample_count,
        len(metadata),
        output_dir,
    )

    saved = _materialize_from_sample_cache_random(cache_dir, out_dir, metadata)
    recovered = _materialize_from_sample_cache_sequential(cache_dir, out_dir, metadata, saved)
    all_saved = saved | recovered

    missing = sorted(set(targets_by_sample) - all_saved)
    if missing:
        raise RuntimeError(
            f"Failed to materialize sample_index values from sample cache: "
            f"{missing[:20]}{'...' if len(missing) > 20 else ''}"
        )

    metadata_path = out_dir / METADATA_FILENAME
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    payload = {
        "schema_version": 1,
        "input_path": str(cache_manifest.get("input_path") or scene_payload.get("source_input") or ""),
        "cache_dir": str(cache_dir),
        "scenes_path": str(scenes_path),
        "annotations_path": str(annotations_path),
        "regions_path": str(regions_path) if regions_path else None,
        "output_dir": str(output_dir),
        "fps": fps,
        "decode_backend": "sample_cache",
        "source": "sample_cache",
        "record_count": len(metadata),
        "target_sample_count": target_sample_count,
        "random_saved_sample_count": len(saved),
        "fallback_saved_sample_count": len(recovered),
        "metadata_path": str(metadata_path),
    }
    with open(out_dir / MATERIALIZED_FILENAME, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log.info(
        "sample-cache materialize done: records=%s samples=%s output=%s",
        len(metadata),
        len(all_saved),
        output_dir,
    )
    return payload

# 원본 영상에서 순차 디코딩으로 frame들을 실체화(Step 4B 최종 출력용)
def materialize_frames(
    input_path: str,
    scenes_path: str,
    annotations_path: str,
    output_dir: str,
    regions_path: str | None = None,
    decode_backend: str = DEFAULT_DECODE_BACKEND,
) -> dict:
    scene_payload = _load_json(scenes_path)
    annotation_payload = _load_json(annotations_path)
    video_meta = read_video_metadata(input_path)
    fps = float(video_meta.get("fps") or scene_payload.get("source", {}).get("fps") or 0.0)
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

    saved = _materialize_sequential(
        input_path,
        out_dir,
        metadata,
        fps=fps,
        width=int(video_meta["width"]),
        height=int(video_meta["height"]),
        decode_backend=decode_backend,
    )
    recovered = _materialize_missing_with_timestamp(
        input_path,
        out_dir,
        metadata,
        saved,
        width=int(video_meta["width"]),
        height=int(video_meta["height"]),
        decode_backend=decode_backend,
    )
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
        "decode_backend": decode_backend,
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


# CLI 인자 파싱
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="원본 영상에서 Step 2/3의 frame_no를 실체화")
    parser.add_argument("--input", "-i", required=True, help="입력 .mp4 경로")
    parser.add_argument("--scenes", required=True, help="Step 2에서 생성된 scene_transitions.json")
    parser.add_argument("--annotations", required=True, help="Step 3에서 생성된 scene_annotations.json")
    parser.add_argument("--regions", help="Step 1에서 생성된 timeline_segments.json, video 구간도 scene으로 실체화됨")
    parser.add_argument("--output", "-o", required=True, help="출력 슬라이드 디렉터리")
    parser.add_argument(
        "--decode-backend",
        choices=["opencv", "ffmpeg-cuda", "ffmpeg-videotoolbox", "auto"],
        default=DEFAULT_DECODE_BACKEND,
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


# CLI 진입점, 원본 영상 기반 materialize_frames 실행
def main() -> None:
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    materialize_frames(
        args.input,
        args.scenes,
        args.annotations,
        args.output,
        regions_path=args.regions,
        decode_backend=args.decode_backend,
    )


if __name__ == "__main__":
    main()
