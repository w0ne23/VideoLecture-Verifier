"""
Scene transition probe.

This script intentionally ignores annotations. It scans an input video in order,
detects slide/screen scene transitions, waits for the new screen to stabilize,
and saves only scene base frames.

Usage:
    python -m pipeline.scene_transition_probe --input lecture.mp4 --output scene_probe/
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

try:
    from .person_masks import masked_pair
except ImportError:  # pragma: no cover - allows direct script execution
    from person_masks import masked_pair


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class ProbeConfig:
    sample_every: int = 2
    resize_width: int = 768
    delay_sec: float = 0.8
    max_pending_sec: float = 2.0
    stable_mse: float = 80.0
    stable_hash: int = 4
    stable_prev_mse: float = 40.0
    stable_prev_hash: int = 2
    cut_mse: float = 500.0
    cut_hash: int = 10
    base_mse: float = 350.0
    base_changed_ratio: float = 0.045
    base_hash: int = 8
    strong_changed_ratio: float = 0.10
    edge_break_ratio: float = 0.45
    same_scene_edge_preserve: float = 0.64
    same_scene_changed_ratio_max: float = 0.32
    fine_diff_threshold: int = 5
    subtle_changed_ratio: float = 0.012
    duplicate_hash: int = 6
    duplicate_edge_preserve: float = 0.95
    duplicate_changed_ratio: float = 0.18
    diff_threshold: int = 15
    crop_left: float = 0.12
    crop_top: float = 0.06
    crop_right: float = 0.94
    crop_bottom: float = 0.90


def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def to_decision_frame(frame: np.ndarray, width: int) -> np.ndarray:
    small = resize_frame(frame, width)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def compute_mse(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    a = frame_a.astype(np.float32) if frame_a.ndim == 2 else cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = frame_b.astype(np.float32) if frame_b.ndim == 2 else cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean((a - b) ** 2))


def compute_phash(frame: np.ndarray) -> imagehash.ImageHash:
    if frame.ndim == 2:
        pil_img = Image.fromarray(frame)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img)


def count_changed_pixels(frame_a: np.ndarray, frame_b: np.ndarray, threshold: int) -> float:
    diff = cv2.absdiff(frame_a, frame_b)
    max_diff = diff if diff.ndim == 2 else np.max(diff, axis=2)
    return float(np.sum(max_diff > threshold) / max_diff.size)


def edge_mask(frame: np.ndarray) -> np.ndarray:
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1) > 0


def edge_preservation_ratio(reference: np.ndarray, frame: np.ndarray) -> float:
    ref_edges = edge_mask(reference)
    ref_count = int(ref_edges.sum())
    if ref_count <= 0:
        return 0.0
    frame_edges = edge_mask(frame)
    kernel = np.ones((5, 5), np.uint8)
    frame_edges_dilated = cv2.dilate(frame_edges.astype(np.uint8), kernel, iterations=1) > 0
    return float(np.logical_and(ref_edges, frame_edges_dilated).sum() / ref_count)


def symmetric_edge_overlap(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    return min(edge_preservation_ratio(frame_a, frame_b), edge_preservation_ratio(frame_b, frame_a))


def content_region(frame: np.ndarray, cfg: ProbeConfig) -> np.ndarray:
    h, w = frame.shape[:2]
    x0 = max(0, min(w - 1, int(w * cfg.crop_left)))
    y0 = max(0, min(h - 1, int(h * cfg.crop_top)))
    x1 = max(x0 + 1, min(w, int(w * cfg.crop_right)))
    y1 = max(y0 + 1, min(h, int(h * cfg.crop_bottom)))
    return frame[y0:y1, x0:x1]


def mask_content_region(mask: np.ndarray | None, cfg: ProbeConfig) -> np.ndarray | None:
    if mask is None:
        return None
    return content_region(mask.astype(np.uint8), cfg).astype(bool)


def scene_metrics(
    reference: np.ndarray,
    frame: np.ndarray,
    cfg: ProbeConfig,
    reference_mask: np.ndarray | None = None,
    frame_mask: np.ndarray | None = None,
) -> dict:
    ref = content_region(reference, cfg)
    cur = content_region(frame, cfg)
    ref, cur = masked_pair(ref, mask_content_region(reference_mask, cfg), cur, mask_content_region(frame_mask, cfg))
    return {
        "mse": compute_mse(ref, cur),
        "changed_ratio": count_changed_pixels(ref, cur, cfg.diff_threshold),
        "fine_changed_ratio": count_changed_pixels(ref, cur, cfg.fine_diff_threshold),
        "edge_preserve": edge_preservation_ratio(ref, cur),
        "symmetric_edge": symmetric_edge_overlap(ref, cur),
        "hash_dist": int(compute_phash(ref) - compute_phash(cur)),
    }


def is_duplicate_scene(
    reference: np.ndarray,
    frame: np.ndarray,
    cfg: ProbeConfig,
    reference_mask: np.ndarray | None = None,
    frame_mask: np.ndarray | None = None,
) -> bool:
    metrics = scene_metrics(reference, frame, cfg, reference_mask, frame_mask)
    return is_same_scene_content(metrics, cfg)


def is_same_scene_content(metrics: dict, cfg: ProbeConfig) -> bool:
    if metrics["changed_ratio"] > cfg.same_scene_changed_ratio_max:
        return False
    if metrics.get("fine_changed_ratio", 0.0) >= cfg.subtle_changed_ratio:
        return False
    return metrics["edge_preserve"] >= cfg.same_scene_edge_preserve


def transition_reason(
    base_frame: np.ndarray,
    prev_frame: np.ndarray,
    current_frame: np.ndarray,
    prev_hash: imagehash.ImageHash,
    current_hash: imagehash.ImageHash,
    cfg: ProbeConfig,
    base_mask: np.ndarray | None = None,
    prev_mask: np.ndarray | None = None,
    current_mask: np.ndarray | None = None,
) -> tuple[str | None, dict]:
    masked_prev, masked_current = masked_pair(prev_frame, prev_mask, current_frame, current_mask)
    prev_mse = compute_mse(masked_prev, masked_current)
    if prev_mask is not None or current_mask is not None:
        prev_hash_dist = int(compute_phash(masked_prev) - compute_phash(masked_current))
    else:
        prev_hash_dist = int(prev_hash - current_hash)
    metrics = scene_metrics(base_frame, current_frame, cfg, base_mask, current_mask)
    same_content = is_same_scene_content(metrics, cfg)

    details = {
        "prev_mse": prev_mse,
        "prev_hash_dist": prev_hash_dist,
        "same_content": same_content,
        **metrics,
    }

    if not same_content and prev_mse >= cfg.cut_mse and prev_hash_dist >= cfg.cut_hash:
        return "cut", details

    if (
        not same_content
        and metrics["mse"] >= cfg.base_mse
        and metrics["changed_ratio"] >= cfg.base_changed_ratio
    ):
        return "base_structure", details

    if (
        not same_content
        and metrics["changed_ratio"] >= cfg.strong_changed_ratio
        and metrics["hash_dist"] >= max(4, cfg.base_hash // 2)
    ):
        return "base_strong_change", details

    if (
        not same_content
        and metrics["fine_changed_ratio"] >= cfg.subtle_changed_ratio
        and (
            metrics["mse"] >= 10.0
            or metrics["changed_ratio"] >= cfg.base_changed_ratio * 0.12
        )
        and (
            metrics["edge_preserve"] < 0.995
            or metrics["changed_ratio"] >= 0.02
            or metrics["mse"] >= 50.0
        )
        and (
            metrics["mse"] >= cfg.base_mse * 0.35
            or metrics["changed_ratio"] >= cfg.base_changed_ratio * 0.5
            or metrics["hash_dist"] >= max(2, cfg.base_hash // 4)
        )
    ):
        return "subtle_text_change", details

    if (
        not same_content
        and metrics["mse"] >= cfg.base_mse
        and metrics["changed_ratio"] >= cfg.base_changed_ratio
        and metrics["hash_dist"] >= cfg.base_hash
    ):
        return "base_diff", details

    if (
        not same_content
        and metrics["mse"] >= cfg.base_mse
        and metrics["changed_ratio"] >= cfg.base_changed_ratio
        and metrics["symmetric_edge"] <= cfg.edge_break_ratio
    ):
        return "edge_break", details

    return None, details


def save_scene(out_dir: Path, scene_index: int, frame: np.ndarray, frame_no: int, timestamp: float, reason: str, details: dict) -> dict:
    filename = f"scene_{scene_index:03d}_base.jpg"
    cv2.imwrite(str(out_dir / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    record = {
        "filename": filename,
        "scene_index": scene_index,
        "frame_no": int(frame_no),
        "timestamp_sec": round(float(timestamp), 3),
        "reason": reason,
        "details": details,
    }
    log.info("[scene %03d] %s @ %.3fs frame=%s", scene_index, reason, timestamp, frame_no)
    return record


def run_probe(input_path: str, output_dir: str, cfg: ProbeConfig) -> list[dict]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        raise RuntimeError(f"Cannot read FPS from video: {input_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("scene_*.jpg"):
        stale.unlink(missing_ok=True)

    stable_frames_required = max(2, int(cfg.delay_sec * fps / cfg.sample_every))
    pending_max_frames = max(stable_frames_required, int(cfg.max_pending_sec * fps / cfg.sample_every))

    records: list[dict] = []
    scene_index = 0
    frame_no = 0
    processed = 0
    base_decision = None
    prev_decision = None
    prev_hash = None
    pending = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            if frame_no % cfg.sample_every != 0:
                continue

            processed += 1
            timestamp = frame_no / fps
            decision = to_decision_frame(frame, cfg.resize_width)
            decision_hash = compute_phash(decision)

            if scene_index == 0:
                scene_index = 1
                base_decision = decision.copy()
                prev_decision = decision.copy()
                prev_hash = decision_hash
                records.append(save_scene(out_dir, scene_index, frame, frame_no, timestamp, "first_frame", {}))
                continue

            if pending is not None:
                anchor_mse = compute_mse(pending["anchor_decision"], decision)
                anchor_hash_dist = int(pending["anchor_hash"] - decision_hash)
                prev_pending_mse = compute_mse(pending["last_decision"], decision)
                prev_pending_hash_dist = int(pending["last_hash"] - decision_hash)
                pending["observed"] += 1
                if (
                    anchor_mse <= cfg.stable_mse
                    and anchor_hash_dist <= cfg.stable_hash
                    and prev_pending_mse <= cfg.stable_prev_mse
                    and prev_pending_hash_dist <= cfg.stable_prev_hash
                ):
                    pending["stable"] += 1
                    pending.update({
                        "frame": frame.copy(),
                        "decision": decision.copy(),
                        "frame_no": frame_no,
                        "timestamp": timestamp,
                        "hash": decision_hash,
                    })
                else:
                    pending.update({
                        "anchor_decision": decision.copy(),
                        "anchor_hash": decision_hash,
                        "frame": frame.copy(),
                        "decision": decision.copy(),
                        "frame_no": frame_no,
                        "timestamp": timestamp,
                        "hash": decision_hash,
                        "stable": 1,
                    })
                pending["last_decision"] = decision.copy()
                pending["last_hash"] = decision_hash

                if pending["stable"] >= stable_frames_required or pending["observed"] >= pending_max_frames:
                    if base_decision is not None and is_duplicate_scene(base_decision, pending["decision"], cfg):
                        log.info("[suppress] duplicate pending scene @ %.3fs frame=%s", pending["timestamp"], pending["frame_no"])
                    else:
                        scene_index += 1
                        records.append(save_scene(
                            out_dir,
                            scene_index,
                            pending["frame"],
                            pending["frame_no"],
                            pending["start_timestamp"],
                            pending["reason"] + "_stabilized",
                            pending["details"],
                        ))
                        base_decision = pending["decision"].copy()
                    prev_decision = decision.copy()
                    prev_hash = decision_hash
                    pending = None
                continue

            assert base_decision is not None and prev_decision is not None and prev_hash is not None
            reason, details = transition_reason(base_decision, prev_decision, decision, prev_hash, decision_hash, cfg)
            if reason is not None:
                pending = {
                    "start_frame_no": frame_no,
                    "start_timestamp": timestamp,
                    "anchor_decision": decision.copy(),
                    "anchor_hash": decision_hash,
                    "last_decision": decision.copy(),
                    "last_hash": decision_hash,
                    "frame": frame.copy(),
                    "decision": decision.copy(),
                    "frame_no": frame_no,
                    "timestamp": timestamp,
                    "hash": decision_hash,
                    "stable": 1,
                    "observed": 1,
                    "reason": reason,
                    "details": details,
                }
                log.info("[pending] %s @ %.3fs frame=%s", reason, timestamp, frame_no)
                continue

            prev_decision = decision.copy()
            prev_hash = decision_hash

            if processed % 1000 == 0:
                pct = (frame_no / total_frames * 100.0) if total_frames > 0 else 0.0
                log.info("processed=%s frame=%s/%s %.1f%%", processed, frame_no, total_frames, pct)
    finally:
        cap.release()

    if pending is not None:
        if base_decision is None or not is_duplicate_scene(base_decision, pending["decision"], cfg):
            scene_index += 1
            records.append(save_scene(
                out_dir,
                scene_index,
                pending["frame"],
                pending["frame_no"],
                pending["start_timestamp"],
                pending["reason"] + "_flush",
                pending["details"],
            ))

    payload = {
        "input": input_path,
        "config": asdict(cfg),
        "scene_count": len(records),
        "scenes": records,
    }
    with open(out_dir / "scene_transitions.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log.info("done: scenes=%s output=%s", len(records), out_dir)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture only stabilized scene transition frames from an MP4.")
    parser.add_argument("--input", "-i", required=True, help="Input .mp4 path")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--sample-every", type=int, default=ProbeConfig.sample_every)
    parser.add_argument("--resize-width", type=int, default=ProbeConfig.resize_width)
    parser.add_argument("--delay-sec", type=float, default=ProbeConfig.delay_sec)
    parser.add_argument("--max-pending-sec", type=float, default=ProbeConfig.max_pending_sec)
    parser.add_argument("--stable-mse", type=float, default=ProbeConfig.stable_mse)
    parser.add_argument("--stable-prev-mse", type=float, default=ProbeConfig.stable_prev_mse)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    cfg = ProbeConfig(
        sample_every=max(1, args.sample_every),
        resize_width=max(160, args.resize_width),
        delay_sec=max(0.0, args.delay_sec),
        max_pending_sec=max(args.delay_sec, args.max_pending_sec),
        stable_mse=max(0.0, args.stable_mse),
        stable_prev_mse=max(0.0, args.stable_prev_mse),
    )
    run_probe(args.input, args.output, cfg)


if __name__ == "__main__":
    main()
