"""
Local VLM review for slide duplicate/build candidates.

The VLM stage is intentionally optional. By default it writes review results
without changing metadata, so CPU-only environments can test model quality
before enabling automatic application.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image


DECISIONS = {
    "same_slide_duplicate",
    "same_slide_build",
    "transition_noise",
    "different_slide",
    "uncertain",
}

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


def _limited_candidate_filenames(candidate: dict[str, Any]) -> list[str]:
    filenames = list(candidate.get("filenames") or [])
    scene_indices = list(candidate.get("scene_indices") or [])
    if len(filenames) <= 2 or len(filenames) != len(scene_indices):
        return filenames

    candidate_type = candidate.get("candidate_type")
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
            if item.get("capture_type") == "annotation"
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

    candidates: list[dict[str, Any]] = []
    rejected = 0
    ordered = sorted(base_representatives)
    for scene_a, scene_b in zip(ordered, ordered[1:]):
        is_build, metrics = build_pair_decision(
            base_representatives[scene_a],
            base_representatives[scene_b],
            cfg,
        )
        if not is_build:
            rejected += 1
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
            continue
        candidates.append({
            "candidate_type": "same_slide_build",
            "source": "local_vlm_adjacent_opencv_build",
            "proposed_decision": "same_slide_build",
            "scene_indices": [scene_a, scene_b],
            "labels": [],
            "filenames": candidate_filenames(scene_a, scene_b),
            "previous_final_filename": final_annot_pool.get(scene_a) or base_pool.get(scene_a),
            "next_final_filename": final_annot_pool.get(scene_b) or base_pool.get(scene_b),
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
    if candidate_type == "same_slide_duplicate" and decision == "same_slide_build":
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
    if candidate_type == "same_slide_build" and decision == "same_slide_build":
        agenda_progression_terms = (
            "agenda item",
            "agenda/items",
            "agenda progression",
            "table-of-contents",
            "table of contents",
            "numbered item",
            "new item",
            "item 01",
            "item 02",
            "item 03",
            "item 04",
            "item 05",
            "section 01",
            "section 02",
            "section 03",
            "section 04",
            "section 05",
            "new section",
            "summary",
            "학습정리",
            "확습정리",
            "목차",
            "항목",
        )
        core_visual_addition_terms = (
            "adds a chart",
            "adds chart",
            "new chart",
            "adds a diagram",
            "adds diagram",
            "new diagram",
            "adds a screenshot",
            "adds screenshot",
            "new screenshot",
            "adds a word cloud",
            "adds word cloud",
            "new word cloud",
            "donut chart",
            "adds a table",
            "adds table",
            "new table",
            "adds a panel",
            "adds panel",
            "new panel",
            "adds a webpage",
            "adds webpage",
            "new webpage",
            "adds a score bar",
            "adds score bar",
            "score bar",
            "new score bar",
            "adds a visual diagram",
            "adds a visual element",
            "mind map",
            "adds a highlighted example",
            "adds an example",
            "example text block",
            "adds a 'tones' section",
            "adds a tones section",
            "adds a 'tones'",
            "'tones' section",
            "tones section",
            "tones analysis panel",
            "detailed 'tones'",
            "adds a 'sentence-level' section",
            "adds a sentence-level section",
            "sentence-level section",
            "sentence-level analysis",
            "document-level section",
            "adds a 'document-level' panel",
            "adds a document-level panel",
            "document-level panel",
            "show button",
            "new example",
            "adds bullet",
            "adds two bullet",
            "bullet points under",
            "new bullet",
            "bullet points",
            "main content area is added",
            "main content blocks",
            "adds two main content",
            "previous image shows only",
            "only the title",
            "title and a placeholder",
            "title-only",
            "mostly blank",
            "evolves from",
            "more detailed",
        )
        if any(term in reason_lc for term in agenda_progression_terms):
            decision = "different_slide"
            raw = dict(raw)
            raw["local_vlm_veto"] = "agenda_item_progression_not_build"
            reason = (
                f"{reason} "
                "[local_vlm_veto: agenda/summary item progression is treated as different_slide]"
            ).strip()
        elif any(term in reason_lc for term in core_visual_addition_terms):
            decision = "different_slide"
            raw = dict(raw)
            raw["local_vlm_veto"] = "core_visual_or_content_addition_not_build"
            reason = (
                f"{reason} "
                "[local_vlm_veto: new core visual/text content is treated as different_slide]"
            ).strip()

    if decision == "same_slide_build":
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

    if candidate_type == "same_slide_duplicate" and decision != "same_slide_duplicate":
        should_merge_slide_group = False
        should_drop_scene = False
    elif decision in {"same_slide_duplicate", "same_slide_build"}:
        should_merge_slide_group = bool(raw.get("should_merge_slide_group", False))
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


def _batch_prompt(
    batch_items: list[tuple[int, dict[str, Any]]],
    image_filenames: list[str],
    timeline_context: dict[int, list[dict[str, Any]]] | None = None,
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
        "Decision labels:\n"
        "- same_slide_duplicate: nearly identical completed slide or nearly identical revisited slide only.\n"
        "- same_slide_build: later image keeps the same topic, preserves the previous readable text/content, "
        "and only adds non-core annotations, highlights, pointer/animation marks, or small explanatory text that does not change the slide's main content.\n"
        "- transition_noise: middle image is slide movement/animation/noisy capture and should be dropped.\n"
        "- different_slide: different lecture-material slides.\n"
        "- uncertain: not enough evidence.\n\n"
        "Absolute non-negotiable rules. You must always follow these rules:\n"
        "- If readable section numbers differ, choose different_slide.\n"
        "- If readable titles differ, choose different_slide.\n"
        "- If key Korean/English text differs in meaning, choose different_slide.\n"
        "- If agenda/item numbers or item sets change, choose different_slide even when the title/section is the same.\n"
        "- If agenda/item numbers or item sets switch to a different title/section/topic, choose different_slide.\n"
        "- If previous substantive text/content disappears, is covered, is replaced, or the main content area is redrawn with different content, choose different_slide.\n"
        "- If a new screenshot, chart, diagram, word cloud, table, panel, webpage, example, score bar, or other main visual/content block appears, choose different_slide.\n"
        "- Never merge based only on template, layout, colors, logos, speaker position, charts, or visual similarity.\n\n"
        "Strict rules:\n"
        "- Default to different_slide unless readable slide content is clearly the same.\n"
        "- For candidate_type=same_slide_duplicate, use a near-exact identity standard: substantive text, title, "
        "section number, bullets, tables, charts, screenshots, item sets, and main content must be almost exactly the same.\n"
        "- For candidate_type=same_slide_duplicate, do not use same_slide_build. If content is added, removed, "
        "revealed, hidden, replaced, reordered, or semantically changed, choose different_slide.\n"
        "- For candidate_type=same_slide_duplicate, same template, same title, same section, same layout, or "
        "similar visual appearance is not enough. Any meaningful content difference means different_slide.\n"
        "- For candidate_type=same_slide_build, additive reveal is allowed only for adjacent/chronological build candidates.\n"
        "- For candidate_type=same_slide_build, do not ignore previous_final_image. If previous_final_image is a mostly blank/title-only slide and next_final_image adds a main UI, panel, screenshot, chart, diagram, word cloud, table, example, or content block, choose different_slide.\n"
        "- For candidate_type=same_slide_build, first compare readable OCR text conceptually: previous readable text must remain visible and unchanged.\n"
        "- For candidate_type=same_slide_build, added text is allowed only when it is non-core annotation/explanation/highlight text. "
        "If added text is a new main bullet, agenda item, section item, table row, chart label, example, or substantive body content, choose different_slide.\n"
        "- For candidate_type=same_slide_build, if any previous readable text is removed, covered, replaced, reordered, or semantically changed, choose different_slide.\n"
        "- Similar template, colors, logos, speaker position, layout, or visual metrics are not enough for a merge.\n"
        "- If titles, section numbers, agenda numbers, table headings, bullet labels, key Korean/English text, "
        "or item sets switch to a different topic, choose different_slide.\n"
        "- For title/agenda/table-of-contents/summary slides, adding, removing, revealing, hiding, or changing "
        "numbered items is different_slide, not same_slide_build.\n"
        "- Choose same_slide_build only for additive reveal: prior visible text/items/figures must remain visible "
        "and unchanged, and the later image may add only non-core annotation/highlight/animation content without replacing the earlier content.\n"
        "- same_slide_build is forbidden when a previously visible screenshot, diagram, chart, panel, table, "
        "highlight box, or large background/content block disappears in the later image. Choose different_slide.\n"
        "- same_slide_build is also forbidden when the later image adds a new screenshot, diagram, chart, word cloud, "
        "table, panel, webpage, example, score bar, or other main visual/content block. Choose different_slide.\n"
        "- For agenda/summary slides, same_slide_build is not allowed for item-number progression. "
        "If numbered items differ in any meaningful way, choose different_slide.\n"
        "- If the later image changes a selected tab/category/filter, swaps examples, replaces a chart/table, "
        "overwrites a screenshot region, or otherwise hides/removes the previous main content, choose different_slide.\n"
        "- If a slide disappears and later a similar-looking slide appears, treat it as different_slide unless "
        "the substantive text is still the same.\n"
        "- For candidate_type=same_slide_duplicate, set should_merge_slide_group=true only when decision is same_slide_duplicate.\n"
        "- For other candidate types, if decision is same_slide_duplicate or same_slide_build, set should_merge_slide_group=true.\n"
        "- If decision is different_slide, transition_noise, or uncertain, set should_merge_slide_group=false.\n"
        "- Set should_merge_slide_group=true only for high-confidence exact duplicates or true incremental builds. "
        "For any uncertainty, keep should_merge_slide_group=false.\n\n"
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
        '      "decision": "same_slide_duplicate|same_slide_build|transition_noise|different_slide|uncertain",\n'
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


def _prompt(candidate: dict[str, Any]) -> str:
    return (
        "/no_think\n"
        "no_think\n"
        "NO THINKING. no_think. Return final JSON directly. "
        "Do not write thinking, analysis, or explanations outside JSON.\n\n"
        "You are reviewing lecture slide extraction results.\n"
        "Compare the provided images in order. Decide whether they are the same lecture slide, "
        "a build/animation step of the same slide, a transition/noisy intermediate frame, "
        "or genuinely different slides.\n\n"
        "Definitions:\n"
        "- same_slide_duplicate: nearly identical completed slide or nearly identical revisited slide only; minor crop/noise/toolbars may differ.\n"
        "- same_slide_build: second image preserves the first slide structure and prior readable text/content, "
        "and only adds non-core annotations, highlights, pointer/animation marks, or small explanatory text that does not change the slide's main content.\n"
        "- transition_noise: image is captured during slide movement/animation and should not be used as a representative scene.\n"
        "- different_slide: images are different lecture-material slides.\n"
        "- uncertain: not enough evidence.\n\n"
        "Absolute non-negotiable rules. You must always follow these rules:\n"
        "- If readable section numbers differ, choose different_slide.\n"
        "- If readable titles differ, choose different_slide.\n"
        "- If key Korean/English text differs in meaning, choose different_slide.\n"
        "- If agenda/item numbers or item sets change, choose different_slide even when the title/section is the same.\n"
        "- If agenda/item numbers or item sets switch to a different title/section/topic, choose different_slide.\n"
        "- If previous substantive text/content disappears, is covered, is replaced, or the main content area is redrawn with different content, choose different_slide.\n"
        "- If a new screenshot, chart, diagram, word cloud, table, panel, webpage, example, score bar, or other main visual/content block appears, choose different_slide.\n"
        "- Never merge based only on template, layout, colors, logos, speaker position, charts, or visual similarity.\n\n"
        "Critical rule for lecture slides:\n"
        "- Default to different_slide unless readable slide content is clearly the same.\n"
        "- Candidate type: "
        f"{candidate.get('candidate_type')}\n"
        "- If candidate_type is same_slide_duplicate, use a near-exact identity standard: substantive text, title, "
        "section number, bullets, tables, charts, screenshots, item sets, and main content must be almost exactly the same.\n"
        "- If candidate_type is same_slide_duplicate, do not use same_slide_build. If content is added, removed, "
        "revealed, hidden, replaced, reordered, or semantically changed, choose different_slide.\n"
        "- If candidate_type is same_slide_duplicate, same template, same title, same section, same layout, or "
        "similar visual appearance is not enough. Any meaningful content difference means different_slide.\n"
        "- If candidate_type is same_slide_build, additive reveal is allowed only for adjacent/chronological build candidates.\n"
        "- If candidate_type is same_slide_build, do not ignore the earlier scene's final visible state. If it is mostly blank/title-only and the later scene adds a main UI, panel, screenshot, chart, diagram, word cloud, table, example, or content block, choose different_slide.\n"
        "- If candidate_type is same_slide_build, first compare readable OCR text conceptually: previous readable text must remain visible and unchanged.\n"
        "- If candidate_type is same_slide_build, added text is allowed only when it is non-core annotation/explanation/highlight text. "
        "If added text is a new main bullet, agenda item, section item, table row, chart label, example, or substantive body content, choose different_slide.\n"
        "- If candidate_type is same_slide_build, if any previous readable text is removed, covered, replaced, reordered, or semantically changed, choose different_slide.\n"
        "- Prioritize readable slide content over layout similarity. If titles, section numbers, agenda numbers, "
        "bullet labels, table headings, or key Korean/English text switch to a different topic, choose different_slide even "
        "when the template, colors, circles, logos, speaker position, or overall layout are nearly identical.\n"
        "- For title/agenda/table-of-contents/summary slides, adding, removing, revealing, hiding, or changing "
        "numbered items is different_slide, not same_slide_build.\n"
        "- same_slide_build requires additive reveal: prior visible text/items/figures must remain visible "
        "and unchanged, and the later image may add only non-core annotation/highlight/animation content without replacing the earlier content.\n"
        "- same_slide_build is forbidden when a previously visible screenshot, diagram, chart, panel, table, "
        "highlight box, or large background/content block disappears in the later image. Choose different_slide.\n"
        "- same_slide_build is also forbidden when the later image adds a new screenshot, diagram, chart, word cloud, "
        "table, panel, webpage, example, score bar, or other main visual/content block. Choose different_slide.\n"
        "- For agenda/summary slides, same_slide_build is not allowed for item-number progression. "
        "If numbered items differ in any meaningful way, choose different_slide.\n"
        "- If the later image changes a selected tab/category/filter, swaps examples, replaces a chart/table, "
        "overwrites a screenshot region, or otherwise hides/removes the previous main content, choose different_slide.\n"
        "- Choose same_slide_duplicate only when the substantive text and semantic content are the same. "
        "Allowed differences are lecturer pose, masking, crop/toolbars, compression noise, pointer position, "
        "or tiny non-semantic visual changes.\n"
        "- Choose same_slide_build only when the later image keeps the same slide topic and merely adds/reveals "
        "non-core annotation/highlight/animation content from that same topic; do not use it for new main bullets, "
        "examples, table rows, chart labels, agenda/table-of-contents item progression, a different agenda page, "
        "different title, or different section.\n\n"
        "Merge output rules:\n"
        "- If candidate_type is same_slide_duplicate, set should_merge_slide_group=true only when decision is same_slide_duplicate.\n"
        "- For other candidate types, if decision is same_slide_duplicate or same_slide_build, set should_merge_slide_group=true.\n"
        "- If decision is different_slide, transition_noise, or uncertain, set should_merge_slide_group=false.\n\n"
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
        '  "decision": "same_slide_duplicate|same_slide_build|transition_noise|different_slide|uncertain",\n'
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
        self.model = os.getenv("GRAPHLEC_OLLAMA_MODEL", "gemma3:4b")

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

        payload = {
            "model": self.model,
            "think": False,
            "messages": [
                {
                    "role": "user",
                    "content": _prompt(candidate),
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
        raw = self._chat_json(
            _batch_prompt(batch_items, image_filenames, timeline_context),
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


def load_provider():
    provider = os.getenv("GRAPHLEC_VLM_PROVIDER", "ollama").strip().lower()
    if provider != "ollama":
        raise ValueError(f"unsupported GRAPHLEC_VLM_PROVIDER={provider!r}; only 'ollama' is implemented")
    return OllamaVLMProvider()


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
    indexed_candidates = list(enumerate(candidates, start=1))
    timeline_context = _timeline_context_by_candidate(indexed_candidates, slides_dir)
    batch_image_limit = local_vlm_batch_image_limit()
    batch_overlap_images = min(local_vlm_batch_overlap_images(), max(0, batch_image_limit - 1))
    batch_candidate_limit = local_vlm_batch_candidate_limit()
    batches = _pack_candidate_batches(
        indexed_candidates,
        max_images=batch_image_limit,
        overlap_images=batch_overlap_images,
        max_candidates=batch_candidate_limit,
        timeline_context=timeline_context,
    )
    worker_count = local_vlm_worker_count(len(batches))

    results = []
    errors = []
    started = time.time()

    def review_one(idx: int, candidate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        item_started = time.time()
        try:
            provider = load_provider()
            result = provider.review(candidate, slides_dir)
            result["candidate_index"] = idx
            result["elapsed_sec"] = round(time.time() - item_started, 3)
            return "result", result
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            error_payload = {
                "candidate_index": idx,
                "scene_indices": candidate.get("scene_indices"),
                "filenames": candidate.get("filenames"),
                "elapsed_sec": round(time.time() - item_started, 3),
                "error": str(exc),
            }
            if isinstance(exc, LocalVLMResponseError):
                if exc.content:
                    error_payload["raw_content_preview"] = exc.content[:500]
                if exc.raw_response:
                    error_payload["raw_response_keys"] = sorted(exc.raw_response.keys())
                    message = exc.raw_response.get("message")
                    if isinstance(message, dict):
                        error_payload["raw_message_keys"] = sorted(message.keys())
                    if exc.raw_response.get("done_reason"):
                        error_payload["done_reason"] = exc.raw_response.get("done_reason")
            return "error", error_payload

    def review_batch(batch: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        item_started = time.time()
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
            elapsed = round(time.time() - item_started, 3)
            for item in batch_results:
                item["batch_index"] = batch_index
                item["elapsed_sec"] = elapsed
            for item in batch_errors:
                item["batch_index"] = batch_index
                item["elapsed_sec"] = elapsed
            return batch_results, batch_errors
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            batch_errors = []
            fallback_results = []

            def retry_one(
                candidate_index: int,
                candidate: dict[str, Any],
                parent_error: Exception,
                fallback_level: str,
            ) -> None:
                one_started = time.time()
                try:
                    provider = load_provider()
                    result = provider.review(candidate, slides_dir)
                    result["candidate_index"] = candidate_index
                    result["batch_index"] = batch_index
                    result["elapsed_sec"] = round(time.time() - one_started, 3)
                    result["batch_fallback"] = True
                    result["batch_fallback_level"] = fallback_level
                    fallback_results.append(result)
                except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as fallback_exc:
                    item_exc = fallback_exc

                    error_payload = {
                        "candidate_index": candidate_index,
                        "batch_index": batch_index,
                        "scene_indices": candidate.get("scene_indices"),
                        "filenames": candidate.get("filenames"),
                        "elapsed_sec": round(time.time() - one_started, 3),
                        "batch_error": str(parent_error),
                        "error": str(item_exc),
                    }
                    if isinstance(item_exc, LocalVLMResponseError):
                        if item_exc.content:
                            error_payload["raw_content_preview"] = item_exc.content[:500]
                        if item_exc.raw_response:
                            error_payload["raw_response_keys"] = sorted(item_exc.raw_response.keys())
                            message = item_exc.raw_response.get("message")
                            if isinstance(message, dict):
                                error_payload["raw_message_keys"] = sorted(message.keys())
                            if item_exc.raw_response.get("done_reason"):
                                error_payload["done_reason"] = item_exc.raw_response.get("done_reason")
                    batch_errors.append(error_payload)

            def retry_pair(pair_items: list[tuple[int, dict[str, Any]]]) -> None:
                if len(pair_items) <= 1:
                    candidate_index, candidate = pair_items[0]
                    retry_one(candidate_index, candidate, exc, "single")
                    return

                pair_started = time.time()
                pair_images = []
                for candidate_index, candidate in pair_items:
                    for filename in _candidate_prompt_filenames(candidate_index, candidate, timeline_context):
                        if filename and filename not in pair_images:
                            pair_images.append(filename)
                try:
                    provider = load_provider()
                    pair_results, pair_errors = provider.review_batch(
                        pair_items,
                        pair_images,
                        slides_dir,
                        timeline_context,
                    )
                    elapsed = round(time.time() - pair_started, 3)
                    for item in pair_results:
                        item["batch_index"] = batch_index
                        item["elapsed_sec"] = elapsed
                        item["batch_fallback"] = True
                        item["batch_fallback_level"] = "pair"
                        fallback_results.append(item)

                    omitted = {
                        int(item.get("candidate_index", 0) or 0)
                        for item in pair_errors
                        if item.get("error") == "VLM batch response omitted candidate"
                    }
                    for candidate_index, candidate in pair_items:
                        if candidate_index in omitted:
                            retry_one(candidate_index, candidate, exc, "single")
                    for item in pair_errors:
                        candidate_index = int(item.get("candidate_index", 0) or 0)
                        if candidate_index not in omitted:
                            item["batch_index"] = batch_index
                            item["elapsed_sec"] = elapsed
                            item["batch_error"] = str(exc)
                            batch_errors.append(item)
                except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as pair_exc:
                    for candidate_index, candidate in pair_items:
                        retry_one(candidate_index, candidate, pair_exc, "single")

            for start in range(0, len(batch_items), 2):
                retry_pair(batch_items[start:start + 2])
            return fallback_results, batch_errors

    if batches:
        if worker_count <= 1:
            for batch in batches:
                batch_results, batch_errors = review_batch(batch)
                results.extend(batch_results)
                errors.extend(batch_errors)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(review_batch, batch): int(batch.get("batch_index", 0) or 0)
                    for batch in batches
                }
                for future in as_completed(futures):
                    batch_results, batch_errors = future.result()
                    results.extend(batch_results)
                    errors.extend(batch_errors)
    elif worker_count <= 1:
        iterator = (
            review_one(idx, candidate)
            for idx, candidate in enumerate(candidates, start=1)
        )
        for kind, payload_item in iterator:
            if kind == "result":
                results.append(payload_item)
            else:
                errors.append(payload_item)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(review_one, idx, candidate): idx
                for idx, candidate in enumerate(candidates, start=1)
            }
            for future in as_completed(futures):
                kind, payload_item = future.result()
                if kind == "result":
                    results.append(payload_item)
                else:
                    errors.append(payload_item)

    omitted_retry_count = 0
    if batches and errors:
        retry_indexes = sorted({
            int(item.get("candidate_index", 0) or 0)
            for item in errors
            if item.get("error") == "VLM batch response omitted candidate"
        })
        retry_indexes = [idx for idx in retry_indexes if 1 <= idx <= len(candidates)]
        if retry_indexes:
            omitted_retry_count = len(retry_indexes)
            retry_set = set(retry_indexes)
            errors = [
                item
                for item in errors
                if int(item.get("candidate_index", 0) or 0) not in retry_set
            ]
            retry_worker_count = local_vlm_worker_count(len(retry_indexes))
            if retry_worker_count <= 1:
                iterator = (review_one(idx, candidates[idx - 1]) for idx in retry_indexes)
                for kind, payload_item in iterator:
                    payload_item["batch_retry"] = True
                    if kind == "result":
                        results.append(payload_item)
                    else:
                        errors.append(payload_item)
            else:
                with ThreadPoolExecutor(max_workers=retry_worker_count) as executor:
                    futures = {
                        executor.submit(review_one, idx, candidates[idx - 1]): idx
                        for idx in retry_indexes
                    }
                    for future in as_completed(futures):
                        kind, payload_item = future.result()
                        payload_item["batch_retry"] = True
                        if kind == "result":
                            results.append(payload_item)
                        else:
                            errors.append(payload_item)

    results.sort(key=lambda item: int(item.get("candidate_index", 0) or 0))
    errors.sort(key=lambda item: int(item.get("candidate_index", 0) or 0))
    _resolve_cross_candidate_conflicts(results)

    payload = {
        "status": "ok" if not errors else "partial",
        "provider": os.getenv("GRAPHLEC_VLM_PROVIDER", "ollama"),
        "model": os.getenv("GRAPHLEC_OLLAMA_MODEL", "gemma3:4b"),
        "candidate_count": len(candidates),
        "candidate_preparation": candidate_preparation,
        "batch_count": len(batches),
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
        if decision in {"same_slide_duplicate", "same_slide_build"}:
            if not result.get("should_merge_slide_group", False):
                continue
            if confidence < merge_min_confidence:
                continue
            if len(scene_indices) >= 2:
                first = scene_indices[0]
                for other in scene_indices[1:]:
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
            if pair not in reviewed_pairs:
                existing_adjacent_pairs.add(pair)

    for scene_a, scene_b in sorted(existing_adjacent_pairs | approved_pairs):
        constrained_union(scene_a, scene_b)

    split_group_by_root: dict[int, set[int]] = {}
    for scene in scenes:
        split_group_by_root.setdefault(constrained_find(scene), set()).add(scene)
    split_groups = list(split_group_by_root.values())

    group_of = {scene: sorted(members) for members in split_groups for scene in members}
    canonical_by_root: dict[int, int] = {}
    canonical_by_scene: dict[int, int] = {}
    for members in split_groups:
        preferred = sorted(scene for scene in members if scene in build_representatives)
        canonical = preferred[0] if preferred else min(
            members,
            key=lambda scene: (base_presence_ratio.get(scene, 0.0), scene),
        )
        for scene in members:
            canonical_by_scene[scene] = canonical

    for item in metadata:
        idx = int(item.get("scene_index", 0) or 0)
        members = group_of.get(idx, [idx])
        canonical = canonical_by_scene.get(idx, members[0])
        item["duplicate_of"] = [x for x in members if x != idx]
        item["same_slide_group"] = members
        item["same_slide_canonical"] = canonical
        item["same_slide_group_size"] = len(members)
        item["slide_group"] = members
        item["slide_canonical_index"] = canonical
        item["slide_group_size"] = len(members)
        scene_results = result_by_scene.get(idx, [])
        if scene_results:
            item["vlm_review_decisions"] = [
                {
                    "decision": r.get("decision"),
                    "confidence": r.get("confidence"),
                    "scene_indices": r.get("scene_indices"),
                    "middle_scene_indices": r.get("middle_scene_indices"),
                    "representative_scene_index": r.get("representative_scene_index"),
                    "reason": r.get("reason"),
                }
                for r in scene_results
            ]
        if any(r.get("decision") == "different_slide" and float(r.get("confidence", 0.0) or 0.0) >= min_confidence for r in scene_results):
            item["vlm_different_slide_veto"] = True
        if idx == canonical:
            item["vlm_preferred_representative"] = idx in build_representatives
        if idx in dropped_scenes:
            item["is_transition_noise"] = True
            item["manual_review"] = False
        elif any(r.get("decision") == "uncertain" for r in scene_results):
            item["manual_review"] = True

    if dropped_scenes:
        return [
            item
            for item in metadata
            if int(item.get("scene_index", 0) or 0) not in dropped_scenes
        ]

    return metadata
