#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.preprocess.local_vlm import apply_vlm_slide_decisions, run_local_vlm_review
from pipeline.preprocess.slide_extractor import (
    _materialize_metadata_frames,
    _video_metadata,
    build_canonical_slide_annotations,
    build_scene_slide_map,
    collapse_contiguous_same_slide_scenes,
    copy_local_vlm_review_artifacts,
    finalize_scene_slide_metadata,
    mark_clean_final_frames,
    refresh_scene_time_ranges,
    remap_metadata_for_final_materialize,
)


def _find_stem(run_dir: Path) -> str:
    matches = sorted(
        path.name[:-len("_metadata.json")]
        for path in run_dir.glob("*_metadata.json")
        if path.is_file() and path.name != "metadata.json"
    )
    if not matches:
        raise FileNotFoundError(
            f"run_dir에서 '*_metadata.json'을 찾지 못했습니다: {run_dir}"
        )
    return matches[0]


def _resolve_input_path(run_dir: Path, explicit_input: str | None) -> Path:
    if explicit_input:
        path = Path(explicit_input)
        if path.exists():
            return path
        raise FileNotFoundError(f"--input 파일이 없습니다: {path}")

    stem = _find_stem(run_dir)
    candidates = [
        run_dir.parent / "inputs" / f"{stem}.mp4",
        run_dir.parent.parent / "input" / f"{stem}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"입력 영상을 자동으로 찾지 못했습니다. --input을 직접 지정하세요.\n{searched}"
    )


def _load_review_metadata(review_dir: Path) -> list[dict]:
    metadata_path = review_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"review metadata가 없습니다: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"review metadata 형식이 예상과 다릅니다: {metadata_path}")
    return payload


def rerun_local_vlm_from_review(run_dir: Path, input_path: Path) -> dict[str, object]:
    slides_dir = run_dir / "slides"
    staged_dir = run_dir / "slides_staged"
    review_dir = staged_dir / "review_slides"

    if not review_dir.exists():
        raise FileNotFoundError(f"review_slides 디렉터리가 없습니다: {review_dir}")

    metadata = _load_review_metadata(review_dir)
    metadata = mark_clean_final_frames(metadata)

    review_payload = run_local_vlm_review(review_dir)
    metadata = apply_vlm_slide_decisions(metadata, review_payload)
    metadata = collapse_contiguous_same_slide_scenes(metadata, review_dir=review_dir)
    metadata = remap_metadata_for_final_materialize(metadata)

    fps, total_frames, _, _ = _video_metadata(str(input_path))
    duration = total_frames / fps if fps > 0 and total_frames > 0 else 0.0

    metadata = refresh_scene_time_ranges(metadata, duration)
    metadata = mark_clean_final_frames(metadata)
    metadata = finalize_scene_slide_metadata(metadata)

    slides_dir.mkdir(parents=True, exist_ok=True)
    _materialize_metadata_frames(str(input_path), slides_dir, metadata)
    copy_local_vlm_review_artifacts(review_dir, slides_dir)

    with open(slides_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(slides_dir / "scene_slide_map.json", "w", encoding="utf-8") as f:
        json.dump(build_scene_slide_map(metadata), f, ensure_ascii=False, indent=2)
    with open(slides_dir / "canonical_slide_annotations.json", "w", encoding="utf-8") as f:
        json.dump(build_canonical_slide_annotations(metadata), f, ensure_ascii=False, indent=2)

    return {
        "run_dir": str(run_dir),
        "input_path": str(input_path),
        "review_dir": str(review_dir),
        "slides_dir": str(slides_dir),
        "scene_count": len({item["scene_index"] for item in metadata if item.get("scene_index") is not None}),
        "frame_count": len(metadata),
        "vlm_status": review_payload.get("status"),
        "processed_count": review_payload.get("processed_count", 0),
        "error_count": review_payload.get("error_count", 0),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="기존 slides_staged/review_slides에서 LocalVLM 이후 단계를 다시 실행한다."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="예: /app/storage/test_qwen35_base_group_fix",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="입력 영상 경로. 미지정 시 run_dir 기준으로 자동 탐색",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"run_dir가 없습니다: {run_dir}", file=sys.stderr)
        return 1

    input_path = _resolve_input_path(run_dir, args.input)
    summary = rerun_local_vlm_from_review(run_dir, input_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
