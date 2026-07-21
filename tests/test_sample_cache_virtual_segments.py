import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.preprocess import sample_cache


def _write_chunk(chunk_dir: Path, colors: list[int], frames: list[dict]) -> Path:
    chunk_dir.mkdir(parents=True)
    video_path = chunk_dir / sample_cache.VIDEO_FILENAME
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        2.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for blue in colors:
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :, 0] = blue
            writer.write(frame)
    finally:
        writer.release()

    manifest_path = chunk_dir / sample_cache.MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps({"video_filename": video_path.name, "frames": frames}),
        encoding="utf-8",
    )
    return manifest_path


def _frame(frame_no: int, timestamp: float) -> dict:
    return {
        "frame_no": frame_no,
        "timestamp_sec": timestamp,
        # Deliberately unrelated to the encoded pixels. Direct-segment alignment
        # must validate the mapping and decoding, not a pre-encode lossy hash.
        "phash_int": (1 << 64) - 1,
        "prev_mse": None,
        "prev_hash_dist": None,
    }


def test_virtual_assembly_preserves_chunk_bytes_and_local_overlap_offsets(tmp_path, monkeypatch):
    chunk_0 = _write_chunk(
        tmp_path / "chunks" / "chunk_000",
        [10, 20, 30, 40],
        [_frame(1, 0.5), _frame(2, 1.0), _frame(3, 1.5), _frame(4, 2.0)],
    )
    chunk_1 = _write_chunk(
        tmp_path / "chunks" / "chunk_001",
        [50, 60, 70, 80],
        [_frame(3, 1.5), _frame(4, 2.0), _frame(5, 2.5), _frame(6, 3.0)],
    )
    specs = [
        {
            "chunk_index": 0,
            "core_start_sec": 0.0,
            "core_end_sec": 2.0,
            "is_last": False,
        },
        {
            "chunk_index": 1,
            "core_start_sec": 2.0,
            "core_end_sec": 3.0,
            "is_last": True,
        },
    ]
    monkeypatch.setattr(
        sample_cache,
        "read_video_metadata",
        lambda _: {
            "fps": 30.0,
            "frame_count": 90,
            "width": 64,
            "height": 48,
            "duration_sec": 3.0,
        },
    )

    output_dir = tmp_path / "output" / "sample_cache"
    cfg = sample_cache.SampleCacheConfig(
        sample_fps=2.0,
        resize_width=64,
        person_masks=False,
    )
    manifest = sample_cache._assemble_chunk_caches(
        "lecture.mp4",
        str(output_dir),
        cfg,
        [chunk_0, chunk_1],
        specs,
    )

    assert manifest["cache"]["merge_mode"] == "virtual_manifest"
    assert manifest["cache"]["segment_storage"] == "original_chunk_cache"
    assert [frame["frame_no"] for frame in manifest["frames"]] == [1, 2, 3, 4, 5, 6]
    assert [frame["cache_segment_frame_index"] for frame in manifest["frames"]] == [0, 1, 2, 1, 2, 3]

    segment_0 = output_dir / "segments" / "chunk_000.avi"
    segment_1 = output_dir / "segments" / "chunk_001.avi"
    assert segment_0.read_bytes() == (chunk_0.parent / sample_cache.VIDEO_FILENAME).read_bytes()
    assert segment_1.read_bytes() == (chunk_1.parent / sample_cache.VIDEO_FILENAME).read_bytes()

    decoded = list(sample_cache.iter_sample_cache_range(output_dir, 0, 6))
    assert [position for position, _, _ in decoded] == list(range(6))
    assert [info["frame_no"] for _, info, _ in decoded] == [1, 2, 3, 4, 5, 6]
    decoded_blue = [float(frame[:, :, 0].mean()) for _, _, frame in decoded]
    assert all(
        abs(actual - expected) <= 5
        for actual, expected in zip(decoded_blue, [10, 20, 30, 60, 70, 80])
    )

    alignment = sample_cache._verify_sample_cache_alignment(output_dir, manifest)
    assert alignment["ok"] is True
    assert alignment["mode"] == "direct-segment-mapping"


def test_segment_mapping_allows_nonzero_start_but_rejects_internal_gap(tmp_path):
    manifest = {
        "cache": {"layout": "segmented", "width": 64, "height": 48},
        "frames": [
            {
                "sample_index": 1,
                "cache_segment_filename": "segments/chunk_000.avi",
                "cache_segment_frame_index": 2,
            },
            {
                "sample_index": 2,
                "cache_segment_filename": "segments/chunk_000.avi",
                "cache_segment_frame_index": 4,
            },
        ],
    }

    alignment = sample_cache._verify_sample_cache_alignment(tmp_path, manifest)

    assert alignment["ok"] is False
    assert alignment["mismatches"][0]["reason"] == "non_contiguous_segment_mapping"
    assert alignment["mismatches"][0]["expected_local_index"] == 3
