"""
Local VLM review for slide duplicate/build candidates.

The VLM stage is intentionally optional. By default it writes review results
without changing metadata, so CPU-only environments can test model quality
before enabling automatic application.
"""

from __future__ import annotations

import base64
from collections import Counter
import io
import json
import os
import re
import logging
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from PIL import Image

try:
    from .ocr_hint import (
        compare_slide_ocr,
        format_ocr_hint_block,
        ocr_enabled,
        ocr_similarity_threshold,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from ocr_hint import (
        compare_slide_ocr,
        format_ocr_hint_block,
        ocr_enabled,
        ocr_similarity_threshold,
    )


DECISIONS = {
    "same_slide_duplicate",
    "same_slide_build",
    "same_slide_annotation",
    "transition_noise",
    "different_slide",
    "uncertain",
}

_SLIDE_DECISION_LABELS = """Decision labels:
- same_slide_duplicate: nearly identical completed slide; only crop, toolbar, compression noise, cursor, or lecturer pose may differ.
- same_slide_build: a staged reveal where the earlier printed body remains visibly present and related printed content is added around it.
- same_slide_annotation: printed slide is unchanged; only handwriting, highlighting, underline, pointer, or other overlay changed or disappeared.
- transition_noise: movement/animation capture that should be dropped.
- different_slide: different printed lecture-material slide.
- uncertain: evidence is insufficient.

"""

_SLIDE_DECISION_POLICY = """Decision policy:
1. Judge printed slide content, not title/template similarity. The same title, topic, section, colors, logo, layout, lecturer, toolbar, or visual style alone never permits a merge.
2. different_slide: choose this when the central printed content is replaced with a new body/topic: previous distinctive bullets, paragraph, equation, definition, diagram, chart, table, screenshot, panel, image, or example disappear and are replaced by different main content. This includes a slide that keeps the same title and headings but replaces question prompts with their answers or replaces short bullets with new explanatory sentences. A different section number, title, agenda item, or slide number is always different_slide.
3. same_slide_build: choose this only when every earlier printed body item is still visibly present in the later image and related new printed material is added around it. Reflow, expansion, or movement of the SAME text/content is allowed, but replacement is never a build. Example: "Causation: what causes X?" changing into "Causation: seasons and hormones cause X" is different_slide, not a build, even if the title and all section headings are identical. Title, shared headings, or a shared conceptual topic alone are never enough.
4. same_slide_annotation: choose this when printed content is unchanged and only temporary overlay ink changed. If handwriting/highlighting/underlining disappears but the printed slide is the same, this is same_slide_annotation, not different_slide.
5. transition_noise: if the first and last images are the same printed slide but a different-looking whiteboard/editor/transition frame appears only between them, choose transition_noise, set should_drop_scene=true for the middle scene, and keep the outer slide states in one group.
6. Ignore tiny non-semantic differences: crop edge, player toolbar, page chrome, compression noise, cursor/laser position, and minor OCR mistakes. Do not ignore replacement of the central printed content.
7. For uncertainty, choose uncertain and should_merge_slide_group=false.

"""

_MERGE_OUTPUT_POLICY = """Merge output rules:
- same_slide_duplicate, same_slide_build, and same_slide_annotation: set should_merge_slide_group=true only when confident.
- different_slide, transition_noise, and uncertain: set should_merge_slide_group=false.

"""

log = logging.getLogger(__name__)
_OLLAMA_BASE_URL_LOCK = threading.Lock()
_OLLAMA_BASE_URL_INDEX = 0


class LocalVLMResponseError(ValueError):
    def __init__(self, message: str, *, raw_response: dict[str, Any] | None = None, content: str = ""):
        super().__init__(message)
        self.raw_response = raw_response or {}
        self.content = content


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def local_vlm_enabled() -> bool:
    return env_bool("GRAPHLEC_VLM_ENABLED", False)


def local_vlm_apply_enabled() -> bool:
    return env_bool("GRAPHLEC_VLM_APPLY", False)


def local_vlm_worker_count(candidate_count: int) -> int:
    if candidate_count <= 1:
        return 1
    workers = env_int("GRAPHLEC_VLM_WORKERS", 2)
    return max(1, min(workers, candidate_count))


def local_vlm_batch_image_limit() -> int:
    timeline_images = local_vlm_timeline_context_images()
    return max(1, env_int("GRAPHLEC_VLM_BATCH_IMAGES", 10), timeline_images)


def local_vlm_batch_overlap_images() -> int:
    return max(0, env_int("GRAPHLEC_VLM_BATCH_OVERLAP_IMAGES", 0))


def local_vlm_batch_candidate_limit() -> int:
    return max(1, env_int("GRAPHLEC_VLM_BATCH_CANDIDATES", 5))


def local_vlm_timeline_context_images() -> int:
    return max(0, env_int("GRAPHLEC_VLM_TIMELINE_CONTEXT_IMAGES", 10))


def local_vlm_auto_build_candidates_enabled() -> bool:
    return env_bool("GRAPHLEC_VLM_AUTO_BUILD_CANDIDATES", True)


def _image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _vlm_image_width() -> int:
    return max(0, env_int("GRAPHLEC_VLM_IMAGE_WIDTH", 768))


def _image_b64_for_vlm(path: Path) -> str:
    width = _vlm_image_width()
    if width <= 0:
        return _image_b64(path)
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.width > width:
            height = max(1, round(img.height * (width / img.width)))
            img = img.resize((width, height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _response_content(body: dict[str, Any]) -> str:
    message = body.get("message", {})
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        content = "".join(parts)
    if content:
        return str(content)
    for key in ("response", "content"):
        value = body.get(key)
        if value:
            return str(value)
    if isinstance(message, dict):
        thinking = message.get("thinking")
        if thinking:
            return str(thinking)
    return ""


def _candidate_scene_indices(candidate: dict[str, Any]) -> list[int]:
    indices = []
    for value in candidate.get("scene_indices") or []:
        try:
            indices.append(int(value))
        except (TypeError, ValueError):
            continue
    return indices


def _is_adjacent_scene_pair(candidate: dict[str, Any]) -> bool:
    scenes = sorted(set(_candidate_scene_indices(candidate)))
    return len(scenes) == 2 and scenes[1] - scenes[0] == 1


def _should_defer_same_slide_candidate(candidate: dict[str, Any]) -> bool:
    candidate_type = str(candidate.get("candidate_type") or "")
    if candidate_type not in {"same_slide_build", "same_slide_duplicate"}:
        return False
    scenes = sorted(set(_candidate_scene_indices(candidate)))
    if len(scenes) < 2:
        return False
    return not _is_adjacent_scene_pair(candidate)


def _deferred_candidate_record(candidate_index: int, candidate: dict[str, Any]) -> dict[str, Any]:
    scenes = sorted(set(_candidate_scene_indices(candidate)))
    return {
        "candidate_index": candidate_index,
        "candidate_type": candidate.get("candidate_type"),
        "source": candidate.get("source"),
        "scene_indices": scenes,
        "filenames": candidate.get("filenames", []),
        "previous_final_filename": candidate.get("previous_final_filename"),
        "next_final_filename": candidate.get("next_final_filename"),
        "reason": "deferred_non_adjacent_same_slide_candidate",
        "original_reason": candidate.get("reason"),
    }


def adjacent_ocr_similarity_threshold() -> float:
    return max(
        0.0,
        min(
            1.0,
            env_float("GRAPHLEC_SLIDE_OCR_ADJACENT_SIMILARITY_THRESHOLD", 0.83),
        ),
    )


def _apply_ocr_threshold(comparison: dict[str, Any], threshold: float) -> dict[str, Any]:
    updated = dict(comparison)
    try:
        similarity = float(updated.get("similarity", 0.0) or 0.0)
    except (TypeError, ValueError):
        similarity = 0.0

    left_norm = str(updated.get("left_normalized") or "")
    right_norm = str(updated.get("right_normalized") or "")
    if not left_norm or not right_norm:
        decision = "unavailable"
    elif left_norm == right_norm or similarity >= threshold:
        decision = "merge"
    elif similarity <= threshold - 0.05:
        decision = "reject"
    else:
        decision = "uncertain"

    updated["threshold"] = threshold
    updated["decision"] = decision
    return updated


def _limited_candidate_filenames(candidate: dict[str, Any]) -> list[str]:
    filenames = list(candidate.get("filenames") or [])
    scene_indices = list(candidate.get("scene_indices") or [])
    candidate_type = str(candidate.get("candidate_type") or "")

    # Build decisions are endpoint-only boundary decisions.  Supplying the
    # base and annotation context for both scenes lets a VLM confuse a prior
    # title/base image with the actual next base.  Always show exactly the
    # prior terminal state and the next scene base (or the two supplied
    # filenames when explicit endpoints are unavailable).
    if candidate_type == "same_slide_build":
        boundary_filenames = [
            candidate.get("previous_final_filename") or (filenames[0] if filenames else None),
            candidate.get("next_final_filename") or (filenames[-1] if filenames else None),
        ]
        endpoints = []
        for filename in boundary_filenames:
            if filename and filename not in endpoints:
                endpoints.append(str(filename))
        if len(endpoints) >= 2:
            return endpoints[:2]
        for filename in filenames:
            if filename and filename not in endpoints:
                endpoints.append(str(filename))
            if len(endpoints) >= 2:
                break
        return endpoints

    if len(filenames) <= 2 or len(filenames) != len(scene_indices):
        return filenames

    if candidate_type == "transition_noise":
        middle_indices = list(candidate.get("middle_scene_indices") or [])
        positions = []
        for middle in middle_indices[:1]:
            if middle in scene_indices:
                mid_pos = scene_indices.index(middle)
                positions = [
                    max(0, mid_pos - 1),
                    mid_pos,
                    min(len(scene_indices) - 1, mid_pos + 1),
                ]
                break
        if not positions:
            positions = list(range(min(3, len(filenames))))
    elif candidate_type == "same_slide_build":
        if len(filenames) <= 4:
            return filenames
        positions = [0, len(filenames) - 1]
    elif candidate_type == "same_slide_duplicate":
        if len(filenames) <= 3:
            return filenames
        positions = sorted({0, len(filenames) // 2, len(filenames) - 1})
    else:
        return filenames

    positions = sorted(dict.fromkeys(pos for pos in positions if 0 <= pos < len(filenames)))
    return [filenames[pos] for pos in positions]


def _load_timeline_context(slides_dir: Path) -> list[dict[str, Any]]:
    metadata_path = slides_dir / "metadata.json"
    if not metadata_path.exists():
        return []
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    rows = []
    for item in metadata:
        if item.get("capture_type") != "base":
            continue
        filename = item.get("filename")
        if not filename or not (slides_dir / filename).exists():
            continue
        try:
            scene_index = int(item.get("scene_index", 0) or 0)
        except (TypeError, ValueError):
            continue
        rows.append({
            "scene_index": scene_index,
            "timestamp_sec": item.get("timestamp_sec"),
            "scene_type": item.get("scene_type"),
            "filename": filename,
        })
    rows.sort(key=lambda row: (int(row.get("scene_index", 0) or 0), str(row.get("filename") or "")))
    return rows


def _timeline_rows_for_candidate(
    candidate: dict[str, Any],
    timeline_rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or not timeline_rows:
        return []
    scene_indices = _candidate_scene_indices(candidate)
    if not scene_indices:
        return []

    positions_by_scene: dict[int, list[int]] = {}
    for pos, row in enumerate(timeline_rows):
        try:
            scene_index = int(row.get("scene_index", 0) or 0)
        except (TypeError, ValueError):
            continue
        positions_by_scene.setdefault(scene_index, []).append(pos)

    anchors = []
    for scene_index in scene_indices:
        anchors.extend(positions_by_scene.get(scene_index, []))
    anchors = sorted(set(anchors))
    if not anchors:
        return []

    selected: set[int] = set()
    radius = 0
    while len(selected) < min(limit, len(timeline_rows)) and radius <= len(timeline_rows):
        for anchor in anchors:
            offsets = [0] if radius == 0 else [-radius, radius]
            for offset in offsets:
                pos = anchor + offset
                if 0 <= pos < len(timeline_rows):
                    selected.add(pos)
                if len(selected) >= limit:
                    break
            if len(selected) >= limit:
                break
        radius += 1
    return [timeline_rows[pos] for pos in sorted(selected)]


def _timeline_context_by_candidate(
    indexed_candidates: list[tuple[int, dict[str, Any]]],
    slides_dir: Path,
) -> dict[int, list[dict[str, Any]]]:
    limit = local_vlm_timeline_context_images()
    if limit <= 0:
        return {}
    timeline_rows = _load_timeline_context(slides_dir)
    return {
        candidate_index: _timeline_rows_for_candidate(candidate, timeline_rows, limit)
        for candidate_index, candidate in indexed_candidates
    }


def _load_local_vlm_metadata(slides_dir: Path) -> list[dict[str, Any]]:
    metadata_path = slides_dir / "metadata.json"
    if not metadata_path.exists():
        return []
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return metadata if isinstance(metadata, list) else []


def _generate_adjacent_build_candidates(slides_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not local_vlm_auto_build_candidates_enabled():
        return [], {"enabled": False}

    metadata = _load_local_vlm_metadata(slides_dir)
    if not metadata:
        return [], {"enabled": True, "reason": "metadata missing"}

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        try:
            from .slide_extractor import Config, build_pair_decision, duplicate_frame_features
            from .slide_extractor import agenda_text_guard_metrics
        except ImportError:
            from slide_extractor import Config, build_pair_decision, duplicate_frame_features
            from slide_extractor import agenda_text_guard_metrics
    except Exception as exc:
        return [], {
            "enabled": True,
            "reason": "opencv build candidate generation unavailable",
            "error": str(exc),
        }

    cfg = Config()
    cache_dir = slides_dir.parent / "sample_cache"
    groups: dict[int, list[dict[str, Any]]] = {}
    for item in metadata:
        try:
            scene_index = int(item.get("scene_index") or 0)
        except (TypeError, ValueError):
            continue
        groups.setdefault(scene_index, []).append(item)

    def load_person_mask(item: dict[str, Any]):
        filename = item.get("person_mask_filename")
        if not filename:
            return None
        path = cache_dir / str(filename)
        if not path.exists():
            return None
        try:
            return np.load(path, allow_pickle=False).astype(bool)
        except Exception:
            return None

    base_pool: dict[int, str] = {}
    final_annot_pool: dict[int, str] = {}
    base_representatives: dict[int, dict[str, Any]] = {}
    for scene_index in sorted(groups):
        base_items = [
            item for item in groups[scene_index]
            if item.get("capture_type") == "base"
            and item.get("filename")
            and (slides_dir / str(item.get("filename"))).exists()
        ]
        if not base_items:
            continue
        annot_items = [
            item for item in groups[scene_index]
            if item.get("capture_type") in {"annotation", "build"}
            and item.get("filename")
            and (slides_dir / str(item.get("filename"))).exists()
        ]
        base_item = base_items[0]
        base_filename = str(base_item["filename"])
        base_pool[scene_index] = base_filename
        if annot_items:
            final_annot_pool[scene_index] = str(annot_items[-1]["filename"])
        image = cv2.imread(str(slides_dir / base_filename))
        if image is None:
            continue
        base_representatives[scene_index] = duplicate_frame_features(
            image,
            cfg,
            mask=load_person_mask(base_item),
        )

    def candidate_filenames(scene_a: int, scene_b: int) -> list[str]:
        filenames: list[str] = []
        for scene_index in (scene_a, scene_b):
            for filename in (base_pool.get(scene_index), final_annot_pool.get(scene_index)):
                if filename and filename not in filenames:
                    filenames.append(filename)
        return filenames

    def title_band_metrics(scene_a: int, scene_b: int) -> dict[str, Any]:
        path_a = slides_dir / str(base_pool.get(scene_a) or "")
        path_b = slides_dir / str(base_pool.get(scene_b) or "")
        if not path_a.exists() or not path_b.exists():
            return {}
        image_a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
        image_b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
        if image_a is None or image_b is None:
            return {}
        height = min(image_a.shape[0], image_b.shape[0])
        width = min(image_a.shape[1], image_b.shape[1])
        if height <= 0 or width <= 0:
            return {}
        image_a = image_a[:height, :width]
        image_b = image_b[:height, :width]
        x0 = min(width - 1, max(0, int(width * 0.12)))
        x1 = min(width, max(x0 + 1, int(width * 0.95)))
        y0 = min(height - 1, max(0, int(height * 0.02)))
        y1 = min(height, max(y0 + 1, int(height * 0.22)))
        band_a = image_a[y0:y1, x0:x1]
        band_b = image_b[y0:y1, x0:x1]
        if band_a.size == 0 or band_b.size == 0:
            return {}
        diff = float(np.mean(cv2.absdiff(band_a, band_b)) / 255.0)
        _, ink_a = cv2.threshold(band_a, 220, 255, cv2.THRESH_BINARY_INV)
        _, ink_b = cv2.threshold(band_b, 220, 255, cv2.THRESH_BINARY_INV)
        ink_mask_a = ink_a > 0
        ink_mask_b = ink_b > 0
        union = int(np.logical_or(ink_mask_a, ink_mask_b).sum())
        intersection = int(np.logical_and(ink_mask_a, ink_mask_b).sum())
        iou = float(intersection / union) if union else 1.0
        return {
            "title_band_diff": diff,
            "title_band_iou": iou,
            "title_band_union_pixels": union,
        }

    candidates: list[dict[str, Any]] = []
    rejected = 0
    ordered = sorted(base_representatives)
    for scene_a, scene_b in zip(ordered, ordered[1:]):
        is_build, metrics = build_pair_decision(
            base_representatives[scene_a],
            base_representatives[scene_b],
            cfg,
        )
        metrics.update(title_band_metrics(scene_a, scene_b))
        if not is_build:
            rejected += 1
            candidates.append({
                "candidate_type": "same_slide_build",
                "source": "local_vlm_adjacent_sequence_review",
                "proposed_decision": "needs_vlm_same_slide_build_check",
                "scene_indices": [scene_a, scene_b],
                "labels": [],
                "filenames": candidate_filenames(scene_a, scene_b),
                "previous_final_filename": final_annot_pool.get(scene_a) or base_pool.get(scene_a),
                # The chronological boundary is compared to the next scene's
                # base.  Its annotations belong to a later state and must not
                # hide a clear/reset at the boundary.
                "next_final_filename": base_pool.get(scene_b) or final_annot_pool.get(scene_b),
                "reason": metrics.get("reason") or "adjacent_sequence_review",
                "metrics": metrics,
            })
            continue
        agenda_metrics = agenda_text_guard_metrics(
            base_representatives[scene_a],
            base_representatives[scene_b],
        )
        metrics.update(agenda_metrics)
        if (
            agenda_metrics.get("agenda_like")
            and (
                agenda_metrics.get("agenda_text_mismatch", 0.0) > getattr(cfg, "AGENDA_TEXT_MISMATCH_MAX", 0.18)
                or agenda_metrics.get("agenda_text_xor", 0.0) > getattr(cfg, "AGENDA_TEXT_XOR_MAX", 0.045)
            )
        ):
            rejected += 1
            candidates.append({
                "candidate_type": "same_slide_build",
                "source": "local_vlm_adjacent_sequence_review",
                "proposed_decision": "needs_vlm_same_slide_build_check",
                "scene_indices": [scene_a, scene_b],
                "labels": [],
                "filenames": candidate_filenames(scene_a, scene_b),
                "previous_final_filename": final_annot_pool.get(scene_a) or base_pool.get(scene_a),
                "next_final_filename": base_pool.get(scene_b) or final_annot_pool.get(scene_b),
                "reason": metrics.get("reason") or "adjacent_sequence_review_agenda_guard",
                "metrics": metrics,
            })
            continue
        candidates.append({
            "candidate_type": "same_slide_build",
            "source": "local_vlm_adjacent_opencv_build",
            "proposed_decision": "same_slide_build",
            "scene_indices": [scene_a, scene_b],
            "labels": [],
            "filenames": candidate_filenames(scene_a, scene_b),
            "previous_final_filename": final_annot_pool.get(scene_a) or base_pool.get(scene_a),
            "next_final_filename": base_pool.get(scene_b) or final_annot_pool.get(scene_b),
            "reason": metrics.get("reason") or "opencv_adjacent_build_candidate",
            "metrics": metrics,
        })

    return candidates, {
        "enabled": True,
        "generated_count": len(candidates),
        "rejected_count": rejected,
        "changed_ratio_min": getattr(cfg, "BUILD_CANDIDATE_CHANGED_RATIO_MIN", None),
        "prev_edge_preserve_min": getattr(cfg, "BUILD_CANDIDATE_PREV_EDGE_PRESERVE_MIN", None),
    }


def _prepare_review_candidates(
    candidates_payload: dict[str, Any],
    slides_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_candidates = list(candidates_payload.get("candidates") or [])
    if not local_vlm_auto_build_candidates_enabled():
        return raw_candidates, {
            "auto_build_candidates": {"enabled": False},
            "original_candidate_count": len(raw_candidates),
            "prepared_candidate_count": len(raw_candidates),
        }

    non_build_candidates = [
        candidate for candidate in raw_candidates
        if candidate.get("candidate_type") != "same_slide_build"
    ]
    generated_build_candidates, build_meta = _generate_adjacent_build_candidates(slides_dir)
    prepared = non_build_candidates + generated_build_candidates
    return prepared, {
        "auto_build_candidates": build_meta,
        "original_candidate_count": len(raw_candidates),
        "original_build_candidate_count": len(raw_candidates) - len(non_build_candidates),
        "prepared_candidate_count": len(prepared),
        "prepared_build_candidate_count": len(generated_build_candidates),
    }


def _candidate_prompt_filenames(
    candidate_index: int,
    candidate: dict[str, Any],
    timeline_context: dict[int, list[dict[str, Any]]] | None = None,
) -> list[str]:
    if str(candidate.get("candidate_type") or "") == "same_slide_build":
        return _limited_candidate_filenames(candidate)

    filenames = []
    for row in (timeline_context or {}).get(candidate_index, []):
        filename = row.get("filename")
        if filename and filename not in filenames:
            filenames.append(filename)
    for filename in _limited_candidate_filenames(candidate):
        if filename and filename not in filenames:
            filenames.append(filename)
    return filenames


def _normalize_result(candidate: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    scene_indices = _candidate_scene_indices(candidate)
    context_scene_indices = _candidate_scene_indices({"scene_indices": candidate.get("context_scene_indices")})
    middle_scene_indices = _candidate_scene_indices({"scene_indices": candidate.get("middle_scene_indices")})
    decision = str(raw.get("decision", "uncertain")).strip()
    if decision not in DECISIONS:
        decision = "uncertain"
    candidate_type = str(candidate.get("candidate_type") or "").strip()
    reason = str(raw.get("reason", "")).strip()
    reason_lc = reason.lower()
    is_adjacent_candidate = (
        len(scene_indices) == 2
        and abs(int(scene_indices[0]) - int(scene_indices[1])) == 1
    )
    if candidate_type == "same_slide_duplicate" and decision == "same_slide_build" and not is_adjacent_candidate:
        decision = "different_slide"
    if candidate_type == "same_slide_duplicate" and decision == "different_slide":
        duplicate_false_negative = (
            ("scene indices are different" in reason_lc and "identical" in reason_lc)
            or (
                "speaker" in reason_lc
                and ("hand position" in reason_lc or "subtle visual" in reason_lc)
                and "content differs" not in reason_lc
                and "main content" not in reason_lc
            )
        )
        if duplicate_false_negative:
            decision = "same_slide_duplicate"
            raw = dict(raw)
            raw["local_vlm_override"] = "duplicate_false_negative_scene_index_or_speaker_pose"
            raw["should_merge_slide_group"] = True
            reason = (
                f"{reason} "
                "[local_vlm_override: scene index or lecturer pose is not a substantive slide-content difference]"
            ).strip()
    if candidate_type == "same_slide_build" and is_adjacent_candidate and decision in {
        "different_slide",
        "uncertain",
    }:
        # OCR can hallucinate changes in book-cover text, handwriting, or
        # screen overlays. When the actual pixels are effectively identical,
        # do not let that OCR/VLM false negative split a chronological slide.
        metrics = candidate.get("metrics") or {}
        try:
            edge_preserve = min(
                float(metrics.get("prev_edge_preserve", 0.0) or 0.0),
                float(metrics.get("curr_edge_preserve", 0.0) or 0.0),
            )
            exact_content = (
                int(metrics.get("content_phash", 9999)) <= 10
                and int(metrics.get("content_dhash", 9999)) <= 12
                and float(metrics.get("content_mse", 1.0) or 1.0) <= 0.003
                and float(metrics.get("content_changed", 1.0) or 1.0) <= 0.02
                and edge_preserve >= 0.95
                and float(metrics.get("content_hist", 0.0) or 0.0) >= 0.998
            )
        except (TypeError, ValueError):
            exact_content = False
        if exact_content:
            raw = dict(raw)
            raw["local_vlm_override"] = "strong_visual_identity_overrides_ocr_false_negative"
            raw["should_merge_slide_group"] = True
            decision = "same_slide_annotation"
            reason = (
                f"{reason} [local_vlm_override: strong visual identity; OCR text variance is treated as overlay/noise]"
            ).strip()
    if decision in {"same_slide_build", "same_slide_annotation"}:
        fallback_representative = scene_indices[-1] if scene_indices else None
    else:
        fallback_representative = scene_indices[0] if scene_indices else None

    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    representative = raw.get("representative_scene_index", fallback_representative)
    try:
        representative = int(representative) if representative is not None else fallback_representative
    except (TypeError, ValueError):
        representative = fallback_representative
    if representative not in scene_indices:
        representative = fallback_representative

    if (
        candidate_type == "same_slide_duplicate"
        and decision != "same_slide_duplicate"
        and not (is_adjacent_candidate and decision in {"same_slide_build", "same_slide_annotation"})
    ):
        should_merge_slide_group = False
        should_drop_scene = False
    elif decision in {"same_slide_duplicate", "same_slide_build", "same_slide_annotation"}:
        should_merge_slide_group = True
        should_drop_scene = False
    elif decision == "transition_noise":
        should_merge_slide_group = False
        should_drop_scene = True
    elif decision in {"different_slide", "uncertain"}:
        should_merge_slide_group = False
        should_drop_scene = bool(raw.get("should_drop_scene", False))
    else:
        should_merge_slide_group = bool(raw.get("should_merge_slide_group", False))
        should_drop_scene = bool(raw.get("should_drop_scene", False))

    return {
        "candidate_type": candidate_type or candidate.get("candidate_type"),
        "scene_indices": scene_indices,
        "context_scene_indices": context_scene_indices,
        "middle_scene_indices": middle_scene_indices,
        "filenames": candidate.get("filenames", []),
        "decision": decision,
        "confidence": confidence,
        "representative_scene_index": representative,
        "should_merge_slide_group": should_merge_slide_group,
        "should_drop_scene": should_drop_scene,
        "reason": reason,
        "raw_response": raw,
    }


def _compact_candidate_for_prompt(
    candidate_index: int,
    candidate: dict[str, Any],
    image_labels: dict[str, str],
    timeline_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    filenames = _limited_candidate_filenames(candidate)

    def image_role(filename: str) -> str:
        if "_annot_" in filename:
            return "final_annotation"
        if "_base" in filename:
            return "base"
        return "context"

    timeline = []
    for row in timeline_rows or []:
        filename = row.get("filename")
        timeline.append({
            "image_label": image_labels.get(filename, filename),
            "scene_index": row.get("scene_index"),
            "timestamp_sec": row.get("timestamp_sec"),
            "scene_type": row.get("scene_type"),
            "filename": filename,
        })
    return {
        "candidate_index": candidate_index,
        "candidate_type": candidate.get("candidate_type"),
        "proposed_decision": candidate.get("proposed_decision"),
        "source": candidate.get("source"),
        "scene_indices": candidate.get("scene_indices"),
        "previous_final_image": image_labels.get(candidate.get("previous_final_filename"), candidate.get("previous_final_filename")),
        "next_final_image": image_labels.get(candidate.get("next_final_filename"), candidate.get("next_final_filename")),
        "candidate_images": [
            {
                "label": image_labels.get(filename, filename),
                "filename": filename,
                "role": image_role(filename),
            }
            for filename in filenames
        ],
        "timeline_context": timeline,
    }


def _ocr_hint_block(image_filenames: list[str], slides_dir: Path) -> str:
    if not ocr_enabled():
        return ""
    blocks: list[str] = []
    for filename in image_filenames:
        path = slides_dir / filename
        if not path.exists():
            continue
        block = format_ocr_hint_block(path, label=filename)
        if block:
            blocks.append(block.strip())
    return "\n\n".join(blocks).strip()


def _scene_boundary_ocr_filenames(
    candidate: dict[str, Any],
    slides_dir: Path,
    provisional_bounds_for_scene: Callable[[int], tuple[int, int] | None] | None = None,
) -> tuple[str | None, str | None]:
    scene_indices = _candidate_scene_indices(candidate)
    if len(scene_indices) < 2:
        return None, None

    metadata = _load_local_vlm_metadata(slides_dir)
    if not metadata:
        return None, None

    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in metadata:
        try:
            scene_index = int(item.get("scene_index") or 0)
        except (TypeError, ValueError):
            continue
        grouped.setdefault(scene_index, []).append(item)

    prev_scene = min(scene_indices)
    next_scene = max(scene_indices)
    if provisional_bounds_for_scene is not None:
        prev_bounds = provisional_bounds_for_scene(prev_scene)
        if prev_bounds is not None:
            prev_scene = int(prev_bounds[1])
        next_bounds = provisional_bounds_for_scene(next_scene)
        if next_bounds is not None:
            next_scene = int(next_bounds[0])
    if next_scene - prev_scene != 1:
        return None, None

    def _base_filename(scene_index: int) -> str | None:
        for item in grouped.get(scene_index, []):
            if item.get("capture_type") == "base" and item.get("filename"):
                filename = str(item.get("filename"))
                if (slides_dir / filename).exists():
                    return filename
        return None

    def _last_annot_filename(scene_index: int) -> str | None:
        annots = [
            item for item in grouped.get(scene_index, [])
            if item.get("capture_type") in {"annotation", "build"} and item.get("filename")
        ]
        annots = [item for item in annots if (slides_dir / str(item.get("filename"))).exists()]
        if annots:
            return str(annots[-1].get("filename"))
        return _base_filename(scene_index)

    return _last_annot_filename(prev_scene), _base_filename(next_scene)


def _ocr_prefilter_same_slide_candidate(
    candidate: dict[str, Any],
    slides_dir: Path,
    provisional_bounds_for_scene: Callable[[int], tuple[int, int] | None] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    candidate_type = str(candidate.get("candidate_type") or "")
    if candidate_type not in {"same_slide_build", "same_slide_duplicate"}:
        return None, None
    if not ocr_enabled():
        return None, None

    filenames = list(candidate.get("filenames") or [])
    if len(filenames) < 2:
        return None, {"candidate_type": candidate_type, "status": "skipped", "reason": "not_enough_filenames"}

    prev_filename, next_filename = _scene_boundary_ocr_filenames(
        candidate,
        slides_dir,
        provisional_bounds_for_scene=provisional_bounds_for_scene,
    )
    if not prev_filename or not next_filename:
        prev_filename = candidate.get("previous_final_filename") or filenames[0]
        next_filename = candidate.get("next_final_filename") or filenames[-1]
    if not prev_filename or not next_filename:
        return None, {"candidate_type": candidate_type, "status": "skipped", "reason": "missing_endpoint_filename"}

    prev_path = slides_dir / str(prev_filename)
    next_path = slides_dir / str(next_filename)
    if not prev_path.exists() or not next_path.exists():
        return None, {
            "candidate_type": candidate_type,
            "status": "skipped",
            "reason": "endpoint_file_missing",
            "previous_final_filename": str(prev_filename),
            "next_final_filename": str(next_filename),
        }

    try:
        comparison = compare_slide_ocr(prev_path, next_path)
        if _is_adjacent_scene_pair(candidate):
            comparison = _apply_ocr_threshold(comparison, adjacent_ocr_similarity_threshold())
    except Exception as exc:
        return None, {
            "candidate_type": candidate_type,
            "status": "error",
            "reason": str(exc),
            "previous_final_filename": str(prev_filename),
            "next_final_filename": str(next_filename),
        }
    similarity = float(comparison.get("similarity", 0.0) or 0.0)
    threshold = float(comparison.get("threshold", ocr_similarity_threshold()) or ocr_similarity_threshold())
    diagnostic = {
        "candidate_type": candidate_type,
        "status": "checked",
        "decision": comparison.get("decision"),
        "similarity": similarity,
        "threshold": threshold,
        "previous_final_filename": str(prev_filename),
        "next_final_filename": str(next_filename),
        "ocr_comparison": comparison,
    }
    if comparison.get("decision") != "merge" or similarity < threshold:
        diagnostic["status"] = "not_prefiltered"
        return None, diagnostic

    # OCR similarity is dominated by shared titles, section headings, and
    # boilerplate.  It must not auto-merge a slide whose body was replaced by
    # another slide with the same title/topic.  Large visual changes remain a
    # LocalVLM decision, while ordinary handwriting-only boundaries still
    # complete in the OCR fast path.
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    content_changed = float(metrics.get("content_changed", 0.0) or 0.0)
    content_mse = float(metrics.get("content_mse", 0.0) or 0.0)
    edge_preserve = float(metrics.get("prev_edge_preserve", 1.0) or 0.0)
    substantive_visual_change = (
        content_changed >= 0.08
        or content_mse >= 0.012
        or edge_preserve < 0.88
    )
    diagnostic["visual_guard"] = {
        "content_changed": content_changed,
        "content_mse": content_mse,
        "prev_edge_preserve": edge_preserve,
        "substantive_visual_change": substantive_visual_change,
    }
    if substantive_visual_change:
        diagnostic["status"] = "review_required"
        diagnostic["reason"] = "high OCR similarity but substantive visual body change requires LocalVLM"
        return None, diagnostic

    # OCR can establish text similarity but cannot distinguish a real printed
    # build, temporary ink, and a duplicate base captured while a lecturer
    # moves.  Auto-labeling a build candidate as an annotation promoted later
    # base frames (including presenter-only differences) into fake annot JPGs.
    # Keep OCR as the cheap candidate signal, then let the VLM choose the
    # capture type for every same_slide_build boundary.
    if candidate_type == "same_slide_build":
        diagnostic["status"] = "review_required"
        diagnostic["reason"] = "OCR merge candidate requires LocalVLM build/annotation/duplicate labeling"
        return None, diagnostic

    scene_indices = _candidate_scene_indices(candidate)
    representative = scene_indices[-1] if scene_indices else None
    prefilter_decision = "same_slide_duplicate"
    decision = {
        "candidate_type": candidate.get("candidate_type"),
        "source": f"ocr_prefilter_{candidate_type}",
        "decision": prefilter_decision,
        "confidence": min(0.99, max(0.95, similarity)),
        "representative_scene_index": representative,
        "should_merge_slide_group": True,
        "should_drop_scene": False,
        "reason": (
            f"ocr similarity {similarity:.4f} >= threshold {threshold:.2f}; "
            "same-slide group merge without OCR build/annotation relabeling"
        ),
        "scene_indices": scene_indices,
        "raw_response": {
            "ocr_prefilter": comparison,
        },
        "ocr_prefilter": comparison,
        "candidate_index": candidate.get("candidate_index"),
    }
    diagnostic["status"] = "prefiltered"
    return decision, diagnostic


def _batch_prompt(
    batch_items: list[tuple[int, dict[str, Any]]],
    image_filenames: list[str],
    timeline_context: dict[int, list[dict[str, Any]]] | None = None,
    ocr_hint_text: str = "",
) -> str:
    image_labels = {filename: f"image_{idx}" for idx, filename in enumerate(image_filenames, start=1)}
    tasks = [
        _compact_candidate_for_prompt(
            candidate_index,
            candidate,
            image_labels,
            (timeline_context or {}).get(candidate_index, []),
        )
        for candidate_index, candidate in batch_items
    ]
    image_manifest = [
        {
            "label": image_labels[filename],
            "filename": filename,
            "role": "final_annotation" if "_annot_" in filename else "base" if "_base" in filename else "context",
        }
        for filename in image_filenames
    ]
    ocr_block = ""
    if str(ocr_hint_text or "").strip():
        ocr_block = (
            "[Pre-extracted OCR hint - use only as a hint, trust the image if they conflict]\n"
            f"{ocr_hint_text.strip()}\n\n"
        )
    return (
        "/no_think\n"
        "no_think\n"
        "NO THINKING. no_think. Return final JSON directly. "
        "Do not write thinking, analysis, or explanations outside JSON.\n\n"
        "You are reviewing lecture slide extraction candidates.\n"
        "The attached images are provided in the same order as this image manifest:\n"
        f"{json.dumps(image_manifest, ensure_ascii=False)}\n\n"
        "For each candidate task, use timeline_context only as chronological context. "
        "Make the final decision only for candidate_images and scene_indices. "
        "The candidate list is suspicious input, not a claim that the images are duplicates.\n\n"
        "For candidate_type=same_slide_build, compare previous_final_image against next_final_image first. "
        "Base images and annotation images from the same scene are context for that scene's evolution, "
        "but the merge decision is whether the earlier scene's final visible state and the later scene's final visible state are the same slide/build.\n\n"
        f"{_SLIDE_DECISION_LABELS}"
        f"{_SLIDE_DECISION_POLICY}"
        f"{_MERGE_OUTPUT_POLICY}"
        f"{ocr_block}"
        "Confidence rules:\n"
        "- confidence is your actual certainty from 0.0 to 1.0, not a fixed placeholder.\n"
        "- Use 0.9-1.0 for clear different_slide cases with different section numbers, titles, or key text.\n"
        "- Use low confidence only when the text is unreadable or the evidence is genuinely ambiguous.\n\n"
        "Candidate tasks:\n"
        f"{json.dumps(tasks, ensure_ascii=False)}\n\n"
        "Return one valid JSON object only. No markdown, no prose. Use exactly this schema:\n"
        "{\n"
        '  "results": [\n'
        "    {\n"
        '      "candidate_index": 1,\n'
        '      "decision": "same_slide_duplicate|same_slide_build|same_slide_annotation|transition_noise|different_slide|uncertain",\n'
        '      "confidence": 0.0,\n'
        '      "representative_scene_index": 0,\n'
        '      "should_merge_slide_group": false,\n'
        '      "should_drop_scene": false,\n'
        '      "reason": "short reason"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def _pack_candidate_batches(
    indexed_candidates: list[tuple[int, dict[str, Any]]],
    *,
    max_images: int,
    overlap_images: int,
    max_candidates: int,
    timeline_context: dict[int, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    current_items: list[tuple[int, dict[str, Any]]] = []
    current_images: list[str] = []

    def add_image(filename: str) -> None:
        if filename and filename not in current_images:
            current_images.append(filename)

    def candidate_new_images(candidate_index: int, candidate: dict[str, Any]) -> list[str]:
        return [
            filename
            for filename in _candidate_prompt_filenames(candidate_index, candidate, timeline_context)
            if filename not in current_images
        ]

    def flush() -> None:
        if current_items:
            batches.append({
                "batch_index": len(batches) + 1,
                "items": list(current_items),
                "image_filenames": list(current_images),
            })

    for candidate_index, candidate in indexed_candidates:
        filenames = _candidate_prompt_filenames(candidate_index, candidate, timeline_context)
        if not filenames:
            continue
        if candidate.get("candidate_type") == "same_slide_build":
            flush()
            current_items = [(candidate_index, candidate)]
            current_images = []
            for filename in filenames:
                add_image(filename)
            flush()
            current_items = []
            current_images = []
            continue
        new_images = candidate_new_images(candidate_index, candidate)
        if current_items and (
            len(current_items) >= max_candidates
            or len(current_images) + len(new_images) > max_images
        ):
            previous_images = list(current_images)
            flush()
            current_items = []
            current_images = previous_images[-min(overlap_images, len(previous_images)):] if overlap_images else []
            new_images = candidate_new_images(candidate_index, candidate)
        current_items.append((candidate_index, candidate))
        for filename in filenames:
            add_image(filename)

    flush()
    return batches


def _prompt(candidate: dict[str, Any], ocr_hint_text: str = "") -> str:
    ocr_block = ""
    if str(ocr_hint_text or "").strip():
        ocr_block = (
            "[Pre-extracted OCR hint - use only as a hint, trust the image if they conflict]\n"
            f"{ocr_hint_text.strip()}\n\n"
        )
    return (
        "/no_think\n"
        "no_think\n"
        "NO THINKING. no_think. Return final JSON directly. "
        "Do not write thinking, analysis, or explanations outside JSON.\n\n"
        "You are reviewing lecture slide extraction results.\n"
        "Compare the provided images in order. Decide whether they are the same lecture slide, "
        "a build/animation step of the same slide, a transition/noisy intermediate frame, "
        "or genuinely different slides.\n\n"
        f"{_SLIDE_DECISION_LABELS}"
        f"{_SLIDE_DECISION_POLICY}"
        f"Candidate type: {candidate.get('candidate_type')}\n\n"
        f"{ocr_block}"
        f"{_MERGE_OUTPUT_POLICY}"
        "Confidence rules:\n"
        "- confidence is your actual certainty from 0.0 to 1.0, not a fixed placeholder.\n"
        "- Use 0.9-1.0 for clear different_slide cases with different section numbers, titles, or key text.\n"
        "- Use low confidence only when the text is unreadable or the evidence is genuinely ambiguous.\n\n"
        f"Scene indices: {candidate.get('scene_indices')}\n"
        f"Candidate type: {candidate.get('candidate_type')}\n"
        f"Proposed decision: {candidate.get('proposed_decision')}\n"
        f"Context scene indices: {candidate.get('context_scene_indices')}\n"
        f"Middle scene indices: {candidate.get('middle_scene_indices')}\n"
        f"Filenames: {candidate.get('filenames')}\n"
        "This candidate is suspicious input only; do not assume it is a duplicate.\n\n"
        "For transition_noise candidates, images are ordered as surrounding context and middle candidates; "
        "set should_drop_scene=true only when the middle image(s) are transitional/noisy captures.\n\n"
        "Return one valid JSON object only. Do not include markdown, code fences, explanations, or prose.\n"
        "Use exactly this schema:\n"
        "{\n"
        '  "decision": "same_slide_duplicate|same_slide_build|same_slide_annotation|transition_noise|different_slide|uncertain",\n'
        '  "confidence": 0.0,\n'
        '  "representative_scene_index": 0,\n'
        '  "should_merge_slide_group": false,\n'
        '  "should_drop_scene": false,\n'
        '  "reason": "short reason"\n'
        "}\n"
    )


class OllamaVLMProvider:
    def __init__(self):
        self.base_url = self._next_base_url()
        self.model = os.getenv("GRAPHLEC_OLLAMA_MODEL", "qwen3.5:9b")

    @staticmethod
    def _base_urls() -> list[str]:
        raw = os.getenv("GRAPHLEC_OLLAMA_BASE_URLS") or os.getenv(
            "GRAPHLEC_OLLAMA_BASE_URL",
            "http://host.docker.internal:11434",
        )
        urls = [url.strip().rstrip("/") for url in raw.split(",") if url.strip()]
        return urls or ["http://host.docker.internal:11434"]

    @classmethod
    def _next_base_url(cls) -> str:
        global _OLLAMA_BASE_URL_INDEX
        urls = cls._base_urls()
        if len(urls) == 1:
            return urls[0]
        with _OLLAMA_BASE_URL_LOCK:
            url = urls[_OLLAMA_BASE_URL_INDEX % len(urls)]
            _OLLAMA_BASE_URL_INDEX += 1
        return url

    def review(self, candidate: dict[str, Any], slides_dir: Path) -> dict[str, Any]:
        images = []
        for filename in _limited_candidate_filenames(candidate):
            path = slides_dir / filename
            if path.exists():
                images.append(_image_b64_for_vlm(path))
        if not images:
            raise FileNotFoundError(f"candidate images not found: {candidate.get('filenames')}")
        ocr_hint_text = _ocr_hint_block(_limited_candidate_filenames(candidate), slides_dir)

        payload = {
            "model": self.model,
            "think": False,
            "messages": [
                {
                    "role": "user",
                    "content": _prompt(candidate, ocr_hint_text=ocr_hint_text),
                    "images": images,
                }
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": env_float("GRAPHLEC_VLM_TEMPERATURE", 0.0),
                "num_predict": env_int("GRAPHLEC_VLM_NUM_PREDICT", 1024),
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = env_float("GRAPHLEC_VLM_TIMEOUT_SEC", 180.0)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = _response_content(body)
        if not content.strip():
            raise LocalVLMResponseError("empty VLM response content", raw_response=body, content=content)
        try:
            raw = _extract_json_object(content)
        except json.JSONDecodeError as exc:
            raise LocalVLMResponseError(
                f"invalid VLM JSON response: {exc}",
                raw_response=body,
                content=content,
            ) from exc
        return _normalize_result(candidate, raw)

    def _chat_json(self, prompt: str, image_filenames: list[str], slides_dir: Path) -> dict[str, Any]:
        images = []
        for filename in image_filenames:
            path = slides_dir / filename
            if path.exists():
                images.append(_image_b64_for_vlm(path))
        if not images:
            raise FileNotFoundError(f"batch images not found: {image_filenames}")

        payload = {
            "model": self.model,
            "think": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": images,
                }
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": env_float("GRAPHLEC_VLM_TEMPERATURE", 0.0),
                "num_predict": env_int("GRAPHLEC_VLM_NUM_PREDICT", 1024),
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = env_float("GRAPHLEC_VLM_TIMEOUT_SEC", 180.0)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = _response_content(body)
        if not content.strip():
            raise LocalVLMResponseError("empty VLM response content", raw_response=body, content=content)
        try:
            return _extract_json_object(content)
        except json.JSONDecodeError as exc:
            raise LocalVLMResponseError(
                f"invalid VLM JSON response: {exc}",
                raw_response=body,
                content=content,
            ) from exc

    def review_batch(
        self,
        batch_items: list[tuple[int, dict[str, Any]]],
        image_filenames: list[str],
        slides_dir: Path,
        timeline_context: dict[int, list[dict[str, Any]]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ocr_hint_text = _ocr_hint_block(image_filenames, slides_dir)
        raw = self._chat_json(
            _batch_prompt(batch_items, image_filenames, timeline_context, ocr_hint_text=ocr_hint_text),
            image_filenames,
            slides_dir,
        )
        raw_results = raw.get("results", [])
        if not isinstance(raw_results, list):
            raise LocalVLMResponseError("VLM batch response missing results array", raw_response=raw)

        candidates_by_index = {candidate_index: candidate for candidate_index, candidate in batch_items}
        seen: set[int] = set()
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for raw_item in raw_results:
            if not isinstance(raw_item, dict):
                continue
            try:
                candidate_index = int(raw_item.get("candidate_index"))
            except (TypeError, ValueError):
                continue
            candidate = candidates_by_index.get(candidate_index)
            if candidate is None:
                continue
            normalized = _normalize_result(candidate, raw_item)
            normalized["candidate_index"] = candidate_index
            results.append(normalized)
            seen.add(candidate_index)

        for candidate_index, candidate in batch_items:
            if candidate_index not in seen:
                errors.append({
                    "candidate_index": candidate_index,
                    "scene_indices": candidate.get("scene_indices"),
                    "filenames": candidate.get("filenames"),
                    "error": "VLM batch response omitted candidate",
                })
        return results, errors



class OpenAICompatibleVLMProvider:
    def __init__(self):
        self.base_url = os.getenv("GRAPHLEC_OPENAI_BASE_URL", "http://host.docker.internal:8000/v1").rstrip("/")
        self.api_key = os.getenv("GRAPHLEC_OPENAI_API_KEY", "none")
        self.model = os.getenv(
            "GRAPHLEC_OPENAI_MODEL",
            os.getenv("GRAPHLEC_OLLAMA_MODEL", "google/diffusiongemma-26B-A4B-it"),
        )

    @staticmethod
    def _image_content(path: Path) -> dict[str, Any]:
        # OpenAI-compatible vision servers generally accept base64 data URLs.
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{_image_b64_for_vlm(path)}"
            },
        }

    def _chat_json(self, prompt: str, image_filenames: list[str], slides_dir: Path) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for filename in image_filenames:
            path = slides_dir / filename
            if path.exists():
                content.append(self._image_content(path))
        if not content:
            raise FileNotFoundError(f"batch images not found: {image_filenames}")

        # Put images before text. DiffusionGemma's model card recommends image
        # content before text for multimodal inputs.
        content.append({"type": "text", "text": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": env_int("GRAPHLEC_VLM_NUM_PREDICT", 1024),
        }
        # GPT-5.6 models accept only their server-side default temperature.
        # Sending the inherited LocalVLM default (0.0) makes every request,
        # including the single-candidate fallback, fail with HTTP 400.
        if not self.model.strip().lower().startswith("gpt-5.6"):
            payload["temperature"] = env_float("GRAPHLEC_VLM_TEMPERATURE", 0.0)
        if env_bool("GRAPHLEC_OPENAI_RESPONSE_FORMAT", True):
            payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        timeout = env_float("GRAPHLEC_VLM_TIMEOUT_SEC", 600.0)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = ""
            raise LocalVLMResponseError(
                f"OpenAI-compatible VLM HTTP {exc.code}: {error_body[:2000]}",
                raw_response={"status_code": exc.code, "body": error_body[:2000]},
            ) from exc

        choices = body.get("choices") or []
        if not choices:
            raise LocalVLMResponseError("OpenAI-compatible VLM response missing choices", raw_response=body)

        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content_text = message.get("content", "") if isinstance(message, dict) else ""
        if isinstance(content_text, list):
            parts = []
            for item in content_text:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            content_text = "".join(parts)
        content_text = str(content_text or "")
        if not content_text.strip():
            raise LocalVLMResponseError("empty OpenAI-compatible VLM response content", raw_response=body, content=content_text)

        try:
            return _extract_json_object(content_text)
        except json.JSONDecodeError as exc:
            raise LocalVLMResponseError(
                f"invalid OpenAI-compatible VLM JSON response: {exc}",
                raw_response=body,
                content=content_text,
            ) from exc

    def review(self, candidate: dict[str, Any], slides_dir: Path) -> dict[str, Any]:
        image_filenames = []
        for filename in _limited_candidate_filenames(candidate):
            if filename and (slides_dir / filename).exists():
                image_filenames.append(filename)
        if not image_filenames:
            raise FileNotFoundError(f"candidate images not found: {candidate.get('filenames')}")
        ocr_hint_text = _ocr_hint_block(image_filenames, slides_dir)
        raw = self._chat_json(_prompt(candidate, ocr_hint_text=ocr_hint_text), image_filenames, slides_dir)
        return _normalize_result(candidate, raw)

    def review_batch(
        self,
        batch_items: list[tuple[int, dict[str, Any]]],
        image_filenames: list[str],
        slides_dir: Path,
        timeline_context: dict[int, list[dict[str, Any]]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ocr_hint_text = _ocr_hint_block(image_filenames, slides_dir)
        raw = self._chat_json(
            _batch_prompt(batch_items, image_filenames, timeline_context, ocr_hint_text=ocr_hint_text),
            image_filenames,
            slides_dir,
        )
        raw_results = raw.get("results", [])
        if not isinstance(raw_results, list):
            raise LocalVLMResponseError("OpenAI-compatible VLM batch response missing results array", raw_response=raw)

        candidates_by_index = {candidate_index: candidate for candidate_index, candidate in batch_items}
        seen: set[int] = set()
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for raw_item in raw_results:
            if not isinstance(raw_item, dict):
                continue
            try:
                candidate_index = int(raw_item.get("candidate_index"))
            except (TypeError, ValueError):
                continue
            candidate = candidates_by_index.get(candidate_index)
            if candidate is None:
                continue
            normalized = _normalize_result(candidate, raw_item)
            normalized["candidate_index"] = candidate_index
            results.append(normalized)
            seen.add(candidate_index)

        for candidate_index, candidate in batch_items:
            if candidate_index not in seen:
                errors.append({
                    "candidate_index": candidate_index,
                    "scene_indices": candidate.get("scene_indices"),
                    "filenames": candidate.get("filenames"),
                    "error": "OpenAI-compatible VLM batch response omitted candidate",
                })
        return results, errors



def load_provider():
    provider = os.getenv("GRAPHLEC_VLM_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        return OllamaVLMProvider()
    if provider in {"openai", "openai-compatible", "openai_compatible", "vllm", "sglang"}:
        return OpenAICompatibleVLMProvider()
    raise ValueError(
        f"unsupported GRAPHLEC_VLM_PROVIDER={provider!r}; "
        "supported providers: 'ollama', 'openai'"
    )


def _pair_keys(scene_indices: Any) -> set[tuple[int, int]]:
    scenes = _candidate_scene_indices({"scene_indices": scene_indices})
    pairs: set[tuple[int, int]] = set()
    for idx, scene_a in enumerate(scenes):
        for scene_b in scenes[idx + 1:]:
            pairs.add(tuple(sorted((scene_a, scene_b))))
    return pairs


def _resolve_cross_candidate_conflicts(results: list[dict[str, Any]]) -> None:
    different_pairs: set[tuple[int, int]] = set()

    for result in results:
        decision = result.get("decision")
        confidence = float(result.get("confidence", 0.0) or 0.0)
        candidate_type = result.get("candidate_type")
        pairs = _pair_keys(result.get("scene_indices"))
        if not pairs:
            continue
        if candidate_type == "same_slide_duplicate" and decision == "different_slide" and confidence >= 0.95:
            reason_lc = str(result.get("reason", "")).lower()
            strong_different = (
                "not a duplicate or incremental build" in reason_lc
                or "main content has changed" in reason_lc
                or "main content is not identical" in reason_lc
                or "content differs significantly" in reason_lc
                or "fundamentally different" in reason_lc
            )
            if strong_different:
                different_pairs.update(pairs)

    veto_pairs = different_pairs
    if not veto_pairs:
        return

    for result in results:
        if result.get("candidate_type") != "same_slide_build":
            continue
        if result.get("decision") != "same_slide_build" and not result.get("should_merge_slide_group"):
            continue
        pairs = _pair_keys(result.get("scene_indices"))
        conflicts = sorted(pairs & veto_pairs)
        if not conflicts:
            continue
        raw = dict(result.get("raw_response") or {})
        raw["local_vlm_veto"] = "cross_candidate_substantive_different_not_build"
        raw["conflicting_pairs"] = conflicts
        result["raw_response"] = raw
        result["decision"] = "different_slide"
        result["should_merge_slide_group"] = False
        result["representative_scene_index"] = (result.get("scene_indices") or [None])[0]
        result["reason"] = (
            f"{result.get('reason', '')} "
            "[local_vlm_veto: another high-confidence duplicate candidate marked this pair as substantive different_slide]"
        ).strip()


def _result_preference_key(result: dict[str, Any]) -> tuple[int, float]:
    decision = str(result.get("decision") or "")
    candidate_type = str(result.get("candidate_type") or "")
    confidence = float(result.get("confidence", 0.0) or 0.0)
    merge = bool(result.get("should_merge_slide_group"))

    if candidate_type == "same_slide_build" and decision in {"same_slide_build", "same_slide_annotation"} and merge:
        rank = 6
    elif candidate_type == "same_slide_duplicate" and decision == "same_slide_duplicate" and merge:
        rank = 5
    elif decision in {"same_slide_build", "same_slide_annotation"} and merge:
        rank = 4
    elif decision == "same_slide_duplicate" and merge:
        rank = 3
    elif decision == "different_slide":
        rank = 2
    elif decision == "uncertain":
        rank = 1
    else:
        rank = 0
    return (rank, confidence)


def _consolidate_pair_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    passthrough: list[dict[str, Any]] = []

    for result in results:
        pair = tuple(sorted(_candidate_scene_indices({"scene_indices": result.get("scene_indices")})))
        if len(pair) == 2 and abs(pair[0] - pair[1]) == 1:
            grouped.setdefault(pair, []).append(result)
        else:
            passthrough.append(result)

    consolidated = list(passthrough)
    for pair, rows in grouped.items():
        best = max(rows, key=_result_preference_key)
        if len(rows) > 1:
            raw = dict(best.get("raw_response") or {})
            raw["pair_consolidation"] = {
                "pair": list(pair),
                "candidate_count": len(rows),
                "candidates": [
                    {
                        "candidate_index": row.get("candidate_index"),
                        "candidate_type": row.get("candidate_type"),
                        "decision": row.get("decision"),
                        "confidence": row.get("confidence"),
                        "should_merge_slide_group": row.get("should_merge_slide_group"),
                    }
                    for row in rows
                ],
            }
            best = dict(best)
            best["raw_response"] = raw
        consolidated.append(best)

    consolidated.sort(key=lambda item: int(item.get("candidate_index", 0) or 0))
    return consolidated


def _scene_chain_components_from_results(results: list[dict[str, Any]]) -> list[list[int]]:
    parent: dict[int, int] = {}

    def find(scene: int) -> int:
        parent.setdefault(scene, scene)
        while parent[scene] != scene:
            parent[scene] = parent[parent[scene]]
            scene = parent[scene]
        return scene

    def union(scene_a: int, scene_b: int) -> None:
        root_a = find(scene_a)
        root_b = find(scene_b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    for result in results:
        if not result.get("should_merge_slide_group"):
            continue
        scenes = sorted(set(_candidate_scene_indices({"scene_indices": result.get("scene_indices")})))
        if len(scenes) < 2:
            continue
        for scene in scenes:
            parent.setdefault(scene, scene)
        for left, right in zip(scenes, scenes[1:]):
            if right - left == 1:
                union(left, right)

    groups: dict[int, list[int]] = {}
    for scene in sorted(parent):
        groups.setdefault(find(scene), []).append(scene)
    return [members for members in sorted(groups.values(), key=lambda members: (members[0], len(members)))]


def _format_scene_chain(scene_indices: list[int]) -> str:
    return " -> ".join(f"{scene:03d}" for scene in scene_indices)


def _assign_resolved_scene_groups(
    metadata: list[dict[str, Any]],
    *,
    group_of: dict[int, list[int]],
    canonical_by_scene: dict[int, int],
    build_representatives: set[int],
    result_by_scene: dict[int, list[dict[str, Any]]],
    min_confidence: float,
    dropped_scenes: set[int],
) -> None:
    visit_order_by_scene: dict[int, int] = {}
    prev_by_scene: dict[int, int | None] = {}
    next_by_scene: dict[int, int | None] = {}

    seen_groups: set[tuple[int, ...]] = set()
    for members in group_of.values():
        key = tuple(members)
        if key in seen_groups:
            continue
        seen_groups.add(key)
        for pos, idx in enumerate(members, start=1):
            visit_order_by_scene[idx] = pos
            prev_by_scene[idx] = members[pos - 2] if pos > 1 else None
            next_by_scene[idx] = members[pos] if pos < len(members) else None

    for item in metadata:
        idx = int(item.get("scene_index", 0) or 0)
        members = group_of.get(idx, [idx])
        canonical = canonical_by_scene.get(idx, members[0])
        visit_order = visit_order_by_scene.get(idx, 1)
        previous_scene = prev_by_scene.get(idx)
        next_scene = next_by_scene.get(idx)
        scene_results = result_by_scene.get(idx, [])

        item["duplicate_of"] = [x for x in members if x != idx]
        item["scene_group"] = members
        item["scene_canonical"] = canonical
        item["scene_group_size"] = len(members)
        item["same_slide_group"] = members
        item["same_slide_canonical"] = canonical
        item["same_slide_group_size"] = len(members)
        item["same_slide_visit_order"] = visit_order
        item["same_slide_is_revisit"] = visit_order > 1
        item["same_slide_previous"] = previous_scene
        item["same_slide_next"] = next_scene
        item["slide_group"] = members
        item["slide_canonical_index"] = canonical
        item["slide_group_size"] = len(members)
        item["slide_visit_order"] = visit_order
        item["slide_is_revisit"] = visit_order > 1
        item["previous_scene_index"] = previous_scene
        item["next_scene_index"] = next_scene

        if scene_results:
            item["vlm_review_decisions"] = [
                {
                    "decision": r.get("decision"),
                    "confidence": r.get("confidence"),
                    "scene_indices": r.get("scene_indices"),
                    "middle_scene_indices": r.get("middle_scene_indices"),
                    "representative_scene_index": r.get("representative_scene_index"),
                    "should_merge_slide_group": r.get("should_merge_slide_group"),
                    "reason": r.get("reason"),
                }
                for r in scene_results
            ]
        else:
            item.pop("vlm_review_decisions", None)

        if any(
            r.get("decision") == "different_slide"
            and float(r.get("confidence", 0.0) or 0.0) >= min_confidence
            for r in scene_results
        ):
            item["vlm_different_slide_veto"] = True
        else:
            item.pop("vlm_different_slide_veto", None)

        preferred = idx == canonical and idx in build_representatives
        if preferred:
            item["vlm_preferred_representative"] = True
        else:
            item["vlm_preferred_representative"] = False

        if idx in dropped_scenes:
            item["is_transition_noise"] = True
            item["manual_review"] = False
        else:
            item.pop("is_transition_noise", None)
            if any(r.get("decision") == "uncertain" for r in scene_results):
                item["manual_review"] = True
            else:
                item.pop("manual_review", None)


def run_local_vlm_review(slides_dir: str | Path) -> dict[str, Any]:
    slides_dir = Path(slides_dir)
    candidates_path = slides_dir / "llm_review_candidates.json"
    results_path = slides_dir / "llm_review_results.json"

    if not candidates_path.exists():
        payload = {"status": "skipped", "reason": "candidate file missing", "results": []}
        results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    candidates_payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates, candidate_preparation = _prepare_review_candidates(candidates_payload, slides_dir)
    all_indexed_candidates = list(enumerate(candidates, start=1))
    indexed_candidates = all_indexed_candidates

    prefiltered_results: list[dict[str, Any]] = []
    ocr_prefilter_diagnostics: list[dict[str, Any]] = []
    reviewable_candidates: list[tuple[int, dict[str, Any]]] = []
    indexed_lookup: dict[int, dict[str, Any]] = {}
    total_candidates = len(indexed_candidates)
    original_candidate_count = len(all_indexed_candidates)

    provisional_scenes = sorted({
        scene
        for _, candidate in indexed_candidates
        for scene in _candidate_scene_indices(candidate)
    })
    provisional_parent = {scene: scene for scene in provisional_scenes}

    def provisional_find(scene: int) -> int:
        while provisional_parent.get(scene, scene) != scene:
            provisional_parent[scene] = provisional_parent[provisional_parent[scene]]
            scene = provisional_parent[scene]
        return scene

    def provisional_union(scene_a: int, scene_b: int) -> None:
        if scene_a not in provisional_parent or scene_b not in provisional_parent:
            return
        root_a = provisional_find(scene_a)
        root_b = provisional_find(scene_b)
        if root_a != root_b:
            provisional_parent[max(root_a, root_b)] = min(root_a, root_b)

    def scenes_in_same_provisional_group(candidate: dict[str, Any]) -> bool:
        scenes = sorted(set(_candidate_scene_indices(candidate)))
        if len(scenes) < 2:
            return False
        if any(scene not in provisional_parent for scene in scenes):
            return False
        return len({provisional_find(scene) for scene in scenes}) == 1

    ocr_attempted_count = 0
    ocr_prefiltered_count = 0
    ocr_not_prefiltered_count = 0
    ocr_error_count = 0
    ocr_skipped_count = 0
    ocr_build_review_required_count = 0
    log.info(
        "OCR prefilter 시작: total_candidates=%s original_candidates=%s",
        total_candidates,
        original_candidate_count,
    )
    for candidate_index, candidate in indexed_candidates:
        indexed_lookup[candidate_index] = candidate
        ocr_result, ocr_diagnostic = _ocr_prefilter_same_slide_candidate(
            candidate,
            slides_dir,
        )
        ocr_attempted_count += 1
        if ocr_diagnostic is not None:
            ocr_diagnostic["candidate_index"] = candidate_index
            ocr_prefilter_diagnostics.append(ocr_diagnostic)
            status = str(ocr_diagnostic.get("status") or "")
            if status == "prefiltered":
                ocr_prefiltered_count += 1
            elif status == "not_prefiltered":
                ocr_not_prefiltered_count += 1
            elif status == "error":
                ocr_error_count += 1
            elif status == "review_required":
                ocr_build_review_required_count += 1
            else:
                ocr_skipped_count += 1
        if ocr_result is not None:
            ocr_result["candidate_index"] = candidate_index
            ocr_result["prefiltered"] = True
            prefiltered_results.append(ocr_result)
            scenes = _candidate_scene_indices(candidate)
            if len(scenes) == 2:
                provisional_union(scenes[0], scenes[1])
        else:
            reviewable_candidates.append((candidate_index, candidate))
        if ocr_attempted_count % 10 == 0 or ocr_attempted_count == total_candidates:
            log.info(
                "OCR prefilter 진행: %s/%s checked, prefiltered=%s, review_required=%s, reviewable=%s, not_prefiltered=%s, errors=%s, skipped=%s",
                ocr_attempted_count,
                total_candidates,
                ocr_prefiltered_count,
                ocr_build_review_required_count,
                len(reviewable_candidates),
                ocr_not_prefiltered_count,
                ocr_error_count,
                ocr_skipped_count,
            )

    ocr_diagnostics_path = slides_dir / "ocr_prefilter_diagnostics.json"
    ocr_diagnostics_payload = {
        "enabled": ocr_enabled(),
        "threshold": ocr_similarity_threshold(),
        "adjacent_threshold": adjacent_ocr_similarity_threshold(),
        "candidate_count": len(candidates),
        "reviewed_candidate_count": total_candidates,
        "attempted_count": len(ocr_prefilter_diagnostics),
        "prefiltered_count": len(prefiltered_results),
        "review_required_count": sum(1 for item in ocr_prefilter_diagnostics if item.get("status") == "review_required"),
        "not_prefiltered_count": sum(1 for item in ocr_prefilter_diagnostics if item.get("status") == "not_prefiltered"),
        "error_count": sum(1 for item in ocr_prefilter_diagnostics if item.get("status") == "error"),
        "skipped_count": sum(1 for item in ocr_prefilter_diagnostics if item.get("status") == "skipped"),
        "phase": "after_ocr_prefilter",
        "diagnostics": ocr_prefilter_diagnostics,
    }
    ocr_diagnostics_path.write_text(
        json.dumps(ocr_diagnostics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("OCR prefilter diagnostics 저장: %s", ocr_diagnostics_path)
    ocr_chain_groups = _scene_chain_components_from_results(prefiltered_results)
    remaining_candidates_report = [
        {
            "candidate_index": candidate_index,
            "candidate_type": candidate.get("candidate_type"),
            "source": candidate.get("source"),
            "proposed_decision": candidate.get("proposed_decision"),
            "scene_indices": candidate.get("scene_indices"),
            "filenames": candidate.get("filenames"),
            "reason": candidate.get("reason"),
            "labels": candidate.get("labels"),
            "vlm_image_policy": candidate.get("vlm_image_policy"),
        }
        for candidate_index, candidate in reviewable_candidates
    ]
    ocr_report_path = slides_dir / "ocr_prefilter_report.json"
    ocr_report_payload = {
        "enabled": ocr_enabled(),
        "threshold": ocr_similarity_threshold(),
        "adjacent_threshold": adjacent_ocr_similarity_threshold(),
        "candidate_count": len(candidates),
        "reviewed_candidate_count": total_candidates,
        "prefiltered_count": len(prefiltered_results),
        "remaining_candidate_count": len(reviewable_candidates),
        "merged_scene_chains": [
            {
                "scene_indices": group,
                "chain": _format_scene_chain(group),
            }
            for group in ocr_chain_groups
        ],
        "remaining_candidates": remaining_candidates_report,
    }
    ocr_report_path.write_text(
        json.dumps(ocr_report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(
        "OCR prefilter 후 남은 후보: %s/%s, 병합 체인=%s, 보고서=%s",
        len(reviewable_candidates),
        total_candidates,
        len(ocr_chain_groups),
        ocr_report_path,
    )
    log.info(
        "OCR -> LocalVLM 전달: adjacent_build_review_required=%s total_reviewable=%s",
        ocr_build_review_required_count,
        len(reviewable_candidates),
    )

    def _run_vlm_phase(
        phase_name: str,
        phase_candidates: list[tuple[int, dict[str, Any]]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        if not phase_candidates:
            return [], [], {"phase": phase_name, "candidate_count": 0, "batch_count": 0, "worker_count": 0}

        timeline_context = _timeline_context_by_candidate(phase_candidates, slides_dir)
        batch_image_limit = local_vlm_batch_image_limit()
        batch_overlap_images = min(local_vlm_batch_overlap_images(), max(0, batch_image_limit - 1))
        batch_candidate_limit = local_vlm_batch_candidate_limit()
        batches = _pack_candidate_batches(
            phase_candidates,
            max_images=batch_image_limit,
            overlap_images=batch_overlap_images,
            max_candidates=batch_candidate_limit,
            timeline_context=timeline_context,
        )
        worker_count = local_vlm_worker_count(len(batches))
        phase_started = time.time()
        log.info(
            "LocalVLM phase start: phase=%s candidates=%s batches=%s workers=%s image_limit=%s candidate_limit=%s",
            phase_name,
            len(phase_candidates),
            len(batches),
            worker_count,
            batch_image_limit,
            batch_candidate_limit,
        )
        phase_results: list[dict[str, Any]] = []
        phase_errors: list[dict[str, Any]] = []

        def review_batch(batch: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            batch_index = int(batch.get("batch_index", 0) or 0)
            batch_items = batch.get("items", [])
            image_filenames = batch.get("image_filenames", [])
            try:
                provider = load_provider()
                batch_results, batch_errors = provider.review_batch(
                    batch_items,
                    image_filenames,
                    slides_dir,
                    timeline_context,
                )
                for item in batch_results:
                    item["batch_index"] = batch_index
                    item["vlm_phase"] = phase_name
                for item in batch_errors:
                    item["batch_index"] = batch_index
                    item["vlm_phase"] = phase_name
                return batch_results, batch_errors
            except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                log.warning(
                    "LocalVLM batch fallback: phase=%s batch=%s candidates=%s error=%s",
                    phase_name,
                    batch_index,
                    len(batch_items),
                    exc,
                )
                fallback_results: list[dict[str, Any]] = []
                fallback_errors: list[dict[str, Any]] = []
                for candidate_index, candidate in batch_items:
                    try:
                        provider = load_provider()
                        result = provider.review(candidate, slides_dir)
                        result["candidate_index"] = candidate_index
                        result["batch_index"] = batch_index
                        result["batch_fallback"] = True
                        result["vlm_phase"] = phase_name
                        fallback_results.append(result)
                    except Exception as fallback_exc:
                        fallback_errors.append({
                            "candidate_index": candidate_index,
                            "scene_indices": candidate.get("scene_indices"),
                            "filenames": candidate.get("filenames"),
                            "batch_index": batch_index,
                            "vlm_phase": phase_name,
                            "batch_error": str(exc),
                            "error": str(fallback_exc),
                        })
                return fallback_results, fallback_errors

        completed_batches = 0
        if worker_count <= 1:
            for batch in batches:
                batch_results, batch_errors = review_batch(batch)
                phase_results.extend(batch_results)
                phase_errors.extend(batch_errors)
                completed_batches += 1
                log.info(
                    "LocalVLM phase progress: phase=%s batches=%s/%s results=%s errors=%s",
                    phase_name,
                    completed_batches,
                    len(batches),
                    len(phase_results),
                    len(phase_errors),
                )
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(review_batch, batch) for batch in batches]
                for future in as_completed(futures):
                    batch_results, batch_errors = future.result()
                    phase_results.extend(batch_results)
                    phase_errors.extend(batch_errors)
                    completed_batches += 1
                    if completed_batches % 5 == 0 or completed_batches == len(batches):
                        log.info(
                            "LocalVLM phase progress: phase=%s batches=%s/%s results=%s errors=%s",
                            phase_name,
                            completed_batches,
                            len(batches),
                            len(phase_results),
                            len(phase_errors),
                        )

        decision_counts = dict(sorted(Counter(str(item.get("decision") or "unknown") for item in phase_results).items()))
        elapsed_sec = round(time.time() - phase_started, 3)
        log.info(
            "LocalVLM phase done: phase=%s results=%s errors=%s decisions=%s elapsed=%.1fs",
            phase_name,
            len(phase_results),
            len(phase_errors),
            decision_counts,
            elapsed_sec,
        )
        return phase_results, phase_errors, {
            "phase": phase_name,
            "candidate_count": len(phase_candidates),
            "batch_count": len(batches),
            "worker_count": worker_count,
            "result_count": len(phase_results),
            "error_count": len(phase_errors),
            "decision_counts": decision_counts,
            "elapsed_sec": elapsed_sec,
        }

    started = time.time()
    batch_image_limit = local_vlm_batch_image_limit()
    batch_overlap_images = min(local_vlm_batch_overlap_images(), max(0, batch_image_limit - 1))
    batch_candidate_limit = local_vlm_batch_candidate_limit()

    adjacent_reviewable = [
        (candidate_index, candidate)
        for candidate_index, candidate in reviewable_candidates
        if _is_adjacent_scene_pair(candidate) or str(candidate.get("candidate_type") or "") == "transition_noise"
    ]
    post_adjacent_reviewable = [
        (candidate_index, candidate)
        for candidate_index, candidate in reviewable_candidates
        if (candidate_index, candidate) not in adjacent_reviewable
    ]

    log.info("LocalVLM 1-pass 시작: adjacent_candidates=%s", len(adjacent_reviewable))
    adjacent_results, adjacent_errors, adjacent_meta = _run_vlm_phase("adjacent", adjacent_reviewable)

    # Apply adjacent VLM approvals to the provisional chain before inspecting distant candidates.
    adjacent_vlm_merges = 0
    for result in adjacent_results:
        scenes = _candidate_scene_indices(result)
        decision = result.get("decision")
        confidence = float(result.get("confidence", 0.0) or 0.0)
        if (
            len(scenes) == 2
            and _is_adjacent_scene_pair({"scene_indices": scenes})
            and decision in {"same_slide_build", "same_slide_annotation", "same_slide_duplicate"}
            and result.get("should_merge_slide_group")
            and confidence >= env_float("GRAPHLEC_VLM_MERGE_MIN_CONFIDENCE", 0.95)
        ):
            provisional_union(scenes[0], scenes[1])
            adjacent_vlm_merges += 1

    log.info(
        "LocalVLM 1-pass applied: adjacent_merges=%s results=%s errors=%s",
        adjacent_vlm_merges,
        len(adjacent_results),
        len(adjacent_errors),
    )

    post_group_results: list[dict[str, Any]] = []
    post_candidates_for_vlm: list[tuple[int, dict[str, Any]]] = []
    for candidate_index, candidate in post_adjacent_reviewable:
        if scenes_in_same_provisional_group(candidate):
            post_group_results.append({
                "candidate_type": candidate.get("candidate_type"),
                "source": "vlm_prefilter_provisional_adjacent_group",
                "decision": "same_slide_duplicate",
                "confidence": 0.99,
                "representative_scene_index": (_candidate_scene_indices(candidate) or [None])[-1],
                "should_merge_slide_group": True,
                "should_drop_scene": False,
                "reason": "candidate is already connected by adjacent OCR/VLM approvals",
                "scene_indices": _candidate_scene_indices(candidate),
                "raw_response": {"provisional_adjacent_group": True},
                "candidate_index": candidate_index,
                "prefiltered": True,
                "vlm_phase": "post_adjacent_skipped",
            })
        else:
            post_candidates_for_vlm.append((candidate_index, candidate))

    log.info(
        "LocalVLM 2-pass 시작: post_adjacent_candidates=%s skipped_by_group=%s",
        len(post_candidates_for_vlm),
        len(post_group_results),
    )
    post_results, post_errors, post_meta = _run_vlm_phase("post_adjacent", post_candidates_for_vlm)

    results = list(prefiltered_results) + adjacent_results + post_group_results + post_results
    errors = adjacent_errors + post_errors
    batches = [adjacent_meta.get("batch_count", 0), post_meta.get("batch_count", 0)]
    batch_count = sum(batches)
    worker_count = max(adjacent_meta.get("worker_count", 0), post_meta.get("worker_count", 0))
    indexed_candidates = adjacent_reviewable + post_candidates_for_vlm
    omitted_retry_count = 0
    results.sort(key=lambda item: int(item.get("candidate_index", 0) or 0))
    errors.sort(key=lambda item: int(item.get("candidate_index", 0) or 0))
    ocr_diagnostics_path = slides_dir / "ocr_prefilter_diagnostics.json"
    ocr_diagnostics_payload = {
        "enabled": ocr_enabled(),
        "threshold": ocr_similarity_threshold(),
        "adjacent_threshold": adjacent_ocr_similarity_threshold(),
        "candidate_count": len(candidates),
        "reviewed_candidate_count": total_candidates,
        "attempted_count": len(ocr_prefilter_diagnostics),
        "prefiltered_count": len(prefiltered_results),
        "review_required_count": sum(1 for item in ocr_prefilter_diagnostics if item.get("status") == "review_required"),
        "not_prefiltered_count": sum(1 for item in ocr_prefilter_diagnostics if item.get("status") == "not_prefiltered"),
        "error_count": sum(1 for item in ocr_prefilter_diagnostics if item.get("status") == "error"),
        "skipped_count": sum(1 for item in ocr_prefilter_diagnostics if item.get("status") == "skipped"),
        "phase": "final",
        "diagnostics": ocr_prefilter_diagnostics,
    }
    ocr_diagnostics_path.write_text(
        json.dumps(ocr_diagnostics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("OCR prefilter diagnostics 저장: %s", ocr_diagnostics_path)
    _resolve_cross_candidate_conflicts(results)
    results = _consolidate_pair_results(results)

    payload = {
        "status": "ok" if not errors else "partial",
        "provider": os.getenv("GRAPHLEC_VLM_PROVIDER", "ollama"),
        "model": os.getenv("GRAPHLEC_OLLAMA_MODEL", "qwen3.5:9b"),
        "candidate_count": len(candidates),
        "reviewed_candidate_count": total_candidates,
        "prefiltered_count": len(prefiltered_results),
        "ocr_prefilter": {
            "enabled": ocr_enabled(),
            "attempted_count": len(ocr_prefilter_diagnostics),
            "reviewed_candidate_count": total_candidates,
            "prefiltered_count": len(prefiltered_results),
            "remaining_candidate_count": len(reviewable_candidates),
            "review_required_count": sum(1 for item in ocr_prefilter_diagnostics if item.get("status") == "review_required"),
            "not_prefiltered_count": sum(1 for item in ocr_prefilter_diagnostics if item.get("status") == "not_prefiltered"),
            "error_count": sum(1 for item in ocr_prefilter_diagnostics if item.get("status") == "error"),
            "skipped_count": sum(1 for item in ocr_prefilter_diagnostics if item.get("status") == "skipped"),
            "threshold": ocr_similarity_threshold(),
            "adjacent_threshold": adjacent_ocr_similarity_threshold(),
            "diagnostics_path": str(ocr_diagnostics_path),
            "report_path": str(ocr_report_path),
            "diagnostics": ocr_prefilter_diagnostics[:200],
        },
        "candidate_preparation": candidate_preparation,
        "batch_count": batch_count,
        "phase_counts": {"adjacent": adjacent_meta, "post_adjacent": post_meta},
        "batch_image_limit": batch_image_limit,
        "batch_overlap_images": batch_overlap_images,
        "batch_candidate_limit": batch_candidate_limit,
        "timeline_context_images": local_vlm_timeline_context_images(),
        "omitted_retry_count": omitted_retry_count,
        "worker_count": worker_count,
        "processed_count": len(results),
        "error_count": len(errors),
        "elapsed_sec": round(time.time() - started, 3),
        "apply_enabled": local_vlm_apply_enabled(),
        "results": results,
        "errors": errors,
    }
    final_decision_counts = dict(sorted(Counter(str(item.get("decision") or "unknown") for item in results).items()))
    log.info(
        "LocalVLM review done: candidates=%s ocr_prefiltered=%s adjacent=%s post_adjacent=%s post_skipped=%s results=%s errors=%s decisions=%s elapsed=%.1fs",
        len(candidates),
        len(prefiltered_results),
        adjacent_meta.get("candidate_count", 0),
        post_meta.get("candidate_count", 0),
        len(post_group_results),
        len(results),
        len(errors),
        final_decision_counts,
        payload["elapsed_sec"],
    )
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def apply_vlm_slide_decisions(metadata: list[dict[str, Any]], review_payload: dict[str, Any]) -> list[dict[str, Any]]:
    min_confidence = env_float("GRAPHLEC_VLM_APPLY_MIN_CONFIDENCE", 0.65)
    merge_min_confidence = max(min_confidence, env_float("GRAPHLEC_VLM_MERGE_MIN_CONFIDENCE", 0.95))
    results = review_payload.get("results", [])

    scenes = sorted({int(item.get("scene_index")) for item in metadata if item.get("scene_index") is not None})
    parent = {scene: scene for scene in scenes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    def known_scene_indices(values: Any) -> list[int]:
        indices = []
        for value in values or []:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            if idx in parent:
                indices.append(idx)
        return indices

    base_presence_ratio: dict[int, float] = {}
    for item in metadata:
        if item.get("capture_type") != "base":
            continue
        try:
            idx = int(item.get("scene_index", 0) or 0)
            base_presence_ratio[idx] = float(item.get("person_presence_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue

    for item in metadata:
        idx = int(item.get("scene_index", 0) or 0)
        for other in item.get("same_slide_group") or []:
            other_idx = known_scene_indices([other])
            if idx in parent and other_idx:
                union(idx, other_idx[0])

    result_by_scene: dict[int, list[dict[str, Any]]] = {scene: [] for scene in scenes}
    dropped_scenes: set[int] = set()
    build_representatives: set[int] = set()
    veto_pairs: set[tuple[int, int]] = set()
    approved_pairs: set[tuple[int, int]] = set()
    reviewed_pairs: set[tuple[int, int]] = set()

    def is_adjacent_pair(scene_a: int, scene_b: int) -> bool:
        return abs(int(scene_a) - int(scene_b)) == 1

    for result in results:
        scene_indices = known_scene_indices(result.get("scene_indices", []))
        decision = result.get("decision")
        confidence = float(result.get("confidence", 0.0) or 0.0)
        for scene in scene_indices:
            result_by_scene.setdefault(scene, []).append(result)
        if len(scene_indices) >= 2:
            for i, scene_a in enumerate(scene_indices):
                for scene_b in scene_indices[i + 1:]:
                    reviewed_pairs.add(tuple(sorted((scene_a, scene_b))))

        if confidence < min_confidence:
            continue
        if decision in {"same_slide_duplicate", "same_slide_build", "same_slide_annotation"}:
            if not result.get("should_merge_slide_group", False):
                continue
            if confidence < merge_min_confidence:
                continue
            if len(scene_indices) >= 2:
                first = scene_indices[0]
                for other in scene_indices[1:]:
                    if is_adjacent_pair(first, other):
                        union(first, other)
                        approved_pairs.add(tuple(sorted((first, other))))
            if decision == "same_slide_build":
                representative = known_scene_indices([result.get("representative_scene_index")])
                if not representative and scene_indices:
                    representative = [scene_indices[-1]]
                if representative:
                    build_representatives.add(representative[0])
        elif decision == "transition_noise" and result.get("should_drop_scene", True):
            drop_indices = known_scene_indices(result.get("middle_scene_indices") or scene_indices)
            dropped_scenes.update(drop_indices)
            # A verified [same slide, transient middle, same slide] sandwich
            # makes the two outer states adjacent after the middle is dropped.
            # This is the sole intentional non-adjacent union in this stage.
            middle_set = set(drop_indices)
            outer_scenes = [scene for scene in scene_indices if scene not in middle_set]
            if len(outer_scenes) == 2:
                union(outer_scenes[0], outer_scenes[1])
                approved_pairs.add(tuple(sorted((outer_scenes[0], outer_scenes[1]))))
        elif decision == "different_slide" and len(scene_indices) >= 2:
            for i, scene_a in enumerate(scene_indices):
                for scene_b in scene_indices[i + 1:]:
                    veto_pairs.add(tuple(sorted((scene_a, scene_b))))

    constrained_parent = {scene: scene for scene in scenes}

    def constrained_find(x: int) -> int:
        while constrained_parent[x] != x:
            constrained_parent[x] = constrained_parent[constrained_parent[x]]
            x = constrained_parent[x]
        return x

    def constrained_union(a: int, b: int) -> None:
        if a not in constrained_parent or b not in constrained_parent:
            return
        if tuple(sorted((a, b))) in veto_pairs:
            return
        ra, rb = constrained_find(a), constrained_find(b)
        if ra != rb:
            constrained_parent[max(ra, rb)] = min(ra, rb)

    existing_adjacent_pairs: set[tuple[int, int]] = set()
    for item in metadata:
        members = known_scene_indices(item.get("same_slide_group") or [])
        for scene_a, scene_b in zip(sorted(set(members)), sorted(set(members))[1:]):
            pair = tuple(sorted((scene_a, scene_b)))
            if is_adjacent_pair(scene_a, scene_b) and pair not in reviewed_pairs:
                existing_adjacent_pairs.add(pair)

    for scene_a, scene_b in sorted(existing_adjacent_pairs | approved_pairs):
        constrained_union(scene_a, scene_b)

    split_group_by_root: dict[int, set[int]] = {}
    for scene in scenes:
        split_group_by_root.setdefault(constrained_find(scene), set()).add(scene)
    split_groups = list(split_group_by_root.values())

    group_of = {scene: sorted(members) for members in split_groups for scene in members}
    canonical_by_scene: dict[int, int] = {}
    for members in split_groups:
        preferred = sorted(scene for scene in members if scene in build_representatives)
        canonical = preferred[0] if preferred else min(
            members,
            key=lambda scene: (base_presence_ratio.get(scene, 0.0), scene),
        )
        for scene in members:
            canonical_by_scene[scene] = canonical

    _assign_resolved_scene_groups(
        metadata,
        group_of=group_of,
        canonical_by_scene=canonical_by_scene,
        build_representatives=build_representatives,
        result_by_scene=result_by_scene,
        min_confidence=min_confidence,
        dropped_scenes=dropped_scenes,
    )

    if dropped_scenes:
        return [
            item
            for item in metadata
            if int(item.get("scene_index", 0) or 0) not in dropped_scenes
        ]

    return metadata
