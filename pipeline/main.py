"""
main.py
강의 영상 분석 통합 파이프라인

실행 흐름:
  [병렬] P1A extract_slides            — 슬라이드 프레임 추출
         P1B analyze_audio_quality     — 오디오 품질 분석
  [병렬] P2A textualize_slides         — 슬라이드 텍스트 + 강조 추출
         P2B transcribe_audio          — 전체 전사 (scene 매핑용)
  [직렬] P3  process_audio             — 오디오 후처리 (P2 완료 후)
                     text_processor    — 3-pass 교정
                     segment_grouper   — scene/context 그룹화 (verifier 입력용)
  [직렬] V1  build_analyzer_input      — 검증 입력 데이터 구성
  [직렬] V2  run_verifier              — claim 추출 → issue 판단/분류 → 멀티 LLM 검증

Usage:
    python main.py --input lecture.mp4
    python main.py --input lecture.mp4 --output output/ --slides output/slides/
    python main.py --input lecture.mp4 --skip-extract
    python main.py --input lecture.mp4 --debug
    python main.py --input lecture.mp4 --force
"""

import json
import os
import sys
import time
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

from .logging_utils import pipeline_log_context
from .utils import resolve_pipeline_package_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


JOB_TYPE_LEGACY_FULL = "legacy_full"
JOB_TYPE_VERIFY = "verify"
JOB_TYPE_VERIFY_ONLY = "verify_only"
JOB_TYPE_UPLOAD = "upload"
JOB_TYPE_PUBLISH = "publish"
JOB_TYPE_DIRECT_UPLOAD = "direct_upload"
JOB_TYPE_VERIFIED_UPLOAD = "verified_upload"
JOB_TYPE_GRAPH_UPLOAD = "graph_upload"
PIPELINE_JOB_TYPES = {
    JOB_TYPE_LEGACY_FULL,
    JOB_TYPE_VERIFY,
    JOB_TYPE_VERIFY_ONLY,
    JOB_TYPE_PUBLISH,
    JOB_TYPE_DIRECT_UPLOAD,
    JOB_TYPE_VERIFIED_UPLOAD,
    JOB_TYPE_GRAPH_UPLOAD,
}
PIPELINE_JOB_TYPE_ALIASES = {
    "legacy": JOB_TYPE_LEGACY_FULL,
    JOB_TYPE_LEGACY_FULL: JOB_TYPE_LEGACY_FULL,
    JOB_TYPE_VERIFY: JOB_TYPE_VERIFY,
    "verified": JOB_TYPE_VERIFY,
    JOB_TYPE_VERIFY_ONLY: JOB_TYPE_VERIFY_ONLY,
    "run_verify": JOB_TYPE_VERIFY_ONLY,
    JOB_TYPE_VERIFIED_UPLOAD: JOB_TYPE_VERIFIED_UPLOAD,
    JOB_TYPE_PUBLISH: JOB_TYPE_PUBLISH,
    "publication": JOB_TYPE_PUBLISH,
    JOB_TYPE_UPLOAD: JOB_TYPE_PUBLISH,
    "direct": JOB_TYPE_DIRECT_UPLOAD,
    JOB_TYPE_DIRECT_UPLOAD: JOB_TYPE_DIRECT_UPLOAD,
    "graph": JOB_TYPE_GRAPH_UPLOAD,
    JOB_TYPE_GRAPH_UPLOAD: JOB_TYPE_GRAPH_UPLOAD,
}
VERIFIER_DETAIL_STAGE_KEYS = [
    "verifier_claim_extraction",
    "verifier_issue_judge",
    "verifier_issue_classification",
    "verifier_final_verification",
    "verifier_web_grounding",
    "verify_slide_inspect",
    "verify_slide_syntax",
]
# 외부 라이브러리 노이즈 로그 억제
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.ai.generativelanguage").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)

# ──────────────────────────────────────────────────────────────
# 출력 헬퍼
# ──────────────────────────────────────────────────────────────

def _banner(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def _normalize_pipeline_job_type(value: str | None) -> str:
    job_type = (value or JOB_TYPE_LEGACY_FULL).strip().lower().replace("-", "_")
    return PIPELINE_JOB_TYPE_ALIASES.get(job_type, JOB_TYPE_LEGACY_FULL)


def _done(label: str, elapsed: float):
    print(f"\n  ✓ {label}  ({elapsed:.1f}초)")
    print("─" * 70)


def _save_json(path: Path, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _fmt_ts(sec: float) -> str:
    m, s = int(sec) // 60, int(sec) % 60
    return f"{m:02d}:{s:02d}"


def _is_done(path: Path, label: str, force: bool) -> bool:
    """출력 파일이 이미 존재하면 True 반환 (force=True면 항상 False)."""
    if not force and path.exists() and path.stat().st_size > 0:
        print(f"\n  ⏭  {label} — 출력 파일 존재, 스킵")
        print(f"     {path}")
        print("─" * 70)
        return True
    return False


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


# ──────────────────────────────────────────────────────────────
# 전사 헬퍼
# ──────────────────────────────────────────────────────────────

def _transcribe_by_scene(
    video_path: str,
    duration: float,
    meta_path: Optional[str],
    slide_ranges: list[dict],
    output_dir: Path,
    progress_callback=None,
) -> dict:
    """
    전역 전사 후 scene 시간축에 매핑하기 위한 세그먼트를 생성한다.

    Returns:
        {
            "segments": [{"start","end","text","words","scene_index"?}, ...],
            "silences": [{"start","end","duration"}, ...]   # 영상 절대 시간
        }
    """
    from .transcriber import transcribe_video

    if not meta_path or not slide_ranges:
        print("  ℹ️ metadata 없음 → 전체 전사 방식 사용")
        return transcribe_video(video_path, duration, output_dir=output_dir, progress_callback=progress_callback)

    unique_scenes = len(slide_ranges)
    print(f"  ℹ️ scene {unique_scenes}개 기준 시간 매핑용 전체 전사 사용")
    return transcribe_video(video_path, duration, output_dir=output_dir, progress_callback=progress_callback)


# ──────────────────────────────────────────────────────────────
# 파이프라인 스테이지
# ──────────────────────────────────────────────────────────────

def extract_slides(args, slides_dir: Path, output_dir: Path, notify_stage=None) -> dict:
    from .slide_extractor import (
        build_canonical_slide_annotations,
        build_scene_slide_map,
        extract_slides,
    )

    stem = Path(args.input).stem
    meta_path = output_dir / f"{stem}_metadata.json"
    scene_slide_map_path = output_dir / f"{stem}_scene_slide_map.json"
    canonical_slide_annotations_path = output_dir / f"{stem}_canonical_slide_annotations.json"

    if _is_done(meta_path, "P1A extract_slides — 슬라이드 추출", args.force):
        return {
            "meta_path": str(meta_path),
            "scene_slide_map_path": str(scene_slide_map_path),
            "canonical_slide_annotations_path": str(canonical_slide_annotations_path),
            "elapsed": 0.0,
        }

    _banner("P1A extract_slides — 슬라이드 추출  (slide_extractor)")
    t0 = time.time()
    progress_callback = None
    if notify_stage:
        progress_callback = lambda done, total: notify_stage("preprocess_slide_extract", "run", (done, total))
    metadata = extract_slides(
        input_path=args.input,
        output_dir=str(slides_dir),
        debug=args.debug,
        decode_backend=getattr(args, "slide_decode_backend", None),
        extract_workers=getattr(args, "slide_extract_workers", None),
        progress_callback=progress_callback,
    )
    elapsed = time.time() - t0

    _save_json(meta_path, metadata)
    _save_json(scene_slide_map_path, build_scene_slide_map(metadata))
    _save_json(canonical_slide_annotations_path, build_canonical_slide_annotations(metadata))

    scene_count = len({
        idx for idx in (
            m.get("scene_index") if m.get("scene_index") is not None else m.get("slide_index")
            for m in metadata
        )
        if idx is not None
    })
    _done(f"scene {scene_count}개, 프레임 {len(metadata)}개 추출", elapsed)
    return {
        "meta_path": str(meta_path),
        "scene_slide_map_path": str(scene_slide_map_path),
        "canonical_slide_annotations_path": str(canonical_slide_annotations_path),
        "elapsed": elapsed,
    }


def analyze_audio_quality(args, output_dir: Path, notify_stage=None) -> dict:
    """P1B analyze_audio_quality: 오디오 품질 분석 (slide_extractor와 병렬).

    내부에 배치/아이템 단위가 없는 단일 파이프라인(오디오 추출 → 특성 분석 → 품질
    채점)이라, 이 세 경계를 진행도 1/3씩으로 보고한다 — 각 경계가 실제 작업
    구간이라 시간 기반으로 흉내 내는 게 아니라 진짜 진행 상태다.
    """
    from .audio_analyzer import extract_audio_from_video, analyze_audio_features, evaluate_audio_quality
    from .utils import get_video_duration

    stem = Path(args.input).stem
    audio_quality_path = output_dir / f"{stem}_audio_quality.json"

    if _is_done(audio_quality_path, "P1B analyze_audio_quality — 오디오 품질 분석", args.force):
        # duration은 파일에서 복원
        duration = 0.0
        try:
            with open(audio_quality_path) as f:
                q = json.load(f)
            duration = _safe_float(q.get("duration_sec"), 0.0)
        except Exception:
            pass
        if duration == 0.0:
            duration = get_video_duration(args.input)
        return {"duration": duration, "elapsed": 0.0}

    video_path = args.input

    def _tick(step: int) -> None:
        if notify_stage:
            notify_stage("preprocess_audio_quality", "run", (step, 3))

    _banner("P1B analyze_audio_quality — 오디오 품질 분석  (audio_analyzer)")
    t0 = time.time()
    duration = get_video_duration(video_path)
    audio_path = str(output_dir / "temp_full_audio.wav")
    extract_audio_from_video(video_path, audio_path)
    _tick(1)
    try:
        audio_features = analyze_audio_features(audio_path)
        _tick(2)
        audio_quality = evaluate_audio_quality(audio_features)
        _tick(3)
        _save_json(output_dir / f"{stem}_audio_features.json", audio_features)
        _save_json(audio_quality_path, audio_quality)
        print(f"  ✓ 품질: {audio_quality['overall_score']}/100 ({audio_quality['overall_grade']})")
    finally:
        Path(audio_path).unlink(missing_ok=True)
    elapsed = time.time() - t0
    _done("오디오 품질 분석", elapsed)
    return {"duration": duration, "elapsed": elapsed}


def textualize_slides(args, slides_dir: Path, output_dir: Path, notify_stage=None) -> dict:
    from .slide_textualizer import TextualizationPipeline, Config as TextConfig

    stem = Path(args.input).stem
    textualized_path = output_dir / f"{stem}_slide_textualized.json"

    if _is_done(textualized_path, "P2A textualize_slides — 슬라이드 텍스트화", args.force):
        return {"textualized_path": str(textualized_path), "elapsed": 0.0}

    _banner("P2A textualize_slides — 슬라이드 텍스트화  (slide_textualizer)")
    t0 = time.time()
    text_config = TextConfig(
        slides_dir=slides_dir,
        output_dir=output_dir,
        output_filename=textualized_path.name,
        max_retries=args.retries,
    )
    progress_callback = None
    if notify_stage:
        progress_callback = lambda done, total: notify_stage("preprocess_slide_analyze", "run", (done, total))
    text_result = TextualizationPipeline(text_config, progress_callback=progress_callback).run()
    elapsed = time.time() - t0

    meta = text_result["metadata"]
    _done(f"scene {meta['total_scenes']}개 / slide {meta['total_slides']}개 텍스트화", elapsed)
    return {"textualized_path": str(textualized_path), "elapsed": elapsed}


def transcribe_audio(args, meta_path: str, duration: float, output_dir: Path, band_progress=None) -> dict:
    from .segment_grouper import load_slide_ranges

    stem = Path(args.input).stem
    transcript_raw_path = output_dir / f"{stem}_transcript_raw.json"

    if _is_done(transcript_raw_path, "P2B transcribe_audio — 전체 전사", args.force):
        if band_progress is not None:
            band_progress.report("transcribe", 1, 1)
        return {"transcript_raw_path": str(transcript_raw_path), "elapsed": 0.0}

    _banner("P2B transcribe_audio — 전체 전사  (Groq Whisper)")
    t0 = time.time()
    slide_ranges = load_slide_ranges(meta_path, duration) if meta_path and Path(meta_path).is_file() else []
    progress_callback = None
    if band_progress is not None:
        progress_callback = lambda done, total: band_progress.report("transcribe", done, total)
    transcribe_result = _transcribe_by_scene(
        args.input, duration, meta_path, slide_ranges, output_dir, progress_callback=progress_callback
    )
    payload = {
        "video_path": args.input,
        "segment_count": len(transcribe_result.get("segments", [])),
        "silence_count": len(transcribe_result.get("silences", [])),
        "segments": transcribe_result.get("segments", []),
        "silences": transcribe_result.get("silences", []),
    }
    _save_json(transcript_raw_path, payload)
    elapsed = time.time() - t0
    _done(
        f"전체 전사 {payload['segment_count']}개 세그먼트, 무음 {payload['silence_count']}개",
        elapsed,
    )
    return {"transcript_raw_path": str(transcript_raw_path), "elapsed": elapsed}


def process_audio(
    args,
    meta_path: str,
    textualized_path: str,
    duration: float,
    output_dir: Path,
    transcript_raw_path: Optional[str] = None,
    on_contexts_ready: Optional[Callable[[dict], None]] = None,
    band_progress=None,
) -> dict:
    from .text_processor import correct_segments_three_pass
    from .segment_grouper import (
        load_slide_ranges,
        group_segments_by_scene_and_context,
    )

    stem = Path(args.input).stem
    segments_path = output_dir / f"{stem}_segments.json"

    video_path = args.input

    _banner("P3 process_audio — 오디오 파이프라인")
    stage_started_at = time.time()

    # 슬라이드 텍스트화 데이터 로드
    textualized_data: dict = {"scenes": []}
    if textualized_path and Path(textualized_path).is_file():
        with open(textualized_path, "r", encoding="utf-8") as f:
            textualized_data = json.load(f)

    # metadata 로드 (슬라이드 occurrence 정보)
    metadata: list[dict] = []
    if meta_path and Path(meta_path).is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    slide_ranges = load_slide_ranges(meta_path, duration) if meta_path and Path(meta_path).is_file() else []
    if transcript_raw_path and Path(transcript_raw_path).is_file():
        print("  [3B-1] 사전 전사 로드...")
        t0 = time.time()
        with open(transcript_raw_path, "r", encoding="utf-8") as f:
            transcribe_payload = json.load(f)
        segments_raw = transcribe_payload.get("segments", [])
        print(f"    ✓ {len(segments_raw)}개 세그먼트 로드  ({time.time()-t0:.1f}초)")
    else:
        print("  [3B-1] 전체 전사 폴백 실행...")
        t0 = time.time()
        transcribe_result = _transcribe_by_scene(video_path, duration, meta_path, slide_ranges, output_dir)
        segments_raw = transcribe_result.get("segments", [])
        print(f"    ✓ {len(segments_raw)}개 세그먼트  ({time.time()-t0:.1f}초)")

    # [3B-2] 3-pass 텍스트 교정 (Gemini 후보 → GPT 보강 → GPT 적용 판정)
    print("  [3B-2] 텍스트 교정 (3-pass)...")
    t0 = time.time()
    textualized_dir = Path(textualized_path).parent if textualized_path else output_dir
    correction_progress_callback = None
    if band_progress is not None:
        correction_progress_callback = lambda band, done, total: band_progress.report(band, done, total)
    segments = correct_segments_three_pass(
        segments=segments_raw,
        metadata=metadata,
        textualized_data=textualized_data,
        textualized_dir=textualized_dir,
        progress_callback=correction_progress_callback,
    )
    segments_clean = [{k: v for k, v in s.items() if k != "words"} for s in segments]
    _save_json(segments_path, {
        "video_path": video_path,
        "segment_count": len(segments_clean),
        "segments": segments_clean,
    })
    print(f"    ✓ 교정 완료  ({time.time()-t0:.1f}초)")

    # [3B-3] scene/context 그룹화 — verifier 입력(merged_clean.json)의 slides[].contexts로
    # 그대로 이어지므로 유지한다. slide_ranges가 없으면(메타데이터 없이 폴백 전사한 경우)
    # scene 단위로 묶을 기준이 없어 scenes_structure는 None으로 남는다.
    print("  [3B-3] scene/context 그룹화...")
    t0 = time.time()
    scenes_structure = None
    if slide_ranges:
        group_progress_callback = None
        if band_progress is not None:
            group_progress_callback = lambda done, total: band_progress.report("context_group", done, total)
        _, scenes_structure = group_segments_by_scene_and_context(
            segments_clean, slide_ranges, duration, use_pause_sentence=False, use_llm_merge=True,
            progress_callback=group_progress_callback,
        )
    elif band_progress is not None:
        # slide_ranges가 없으면 그룹화 자체를 안 하니 band를 즉시 완료 처리한다.
        band_progress.report("context_group", 0, 0)
    print(f"    ✓ context {sum(len(s.get('contexts', [])) for s in scenes_structure or [])}개  ({time.time()-t0:.1f}초)")

    if on_contexts_ready and scenes_structure:
        try:
            on_contexts_ready({
                "segments_path": str(segments_path),
                "slides_structure": scenes_structure,
                "slide_ranges": slide_ranges,
                "duration": duration,
            })
        except Exception as exc:
            log.warning(f"analyzer 조기 시작 실패(context 사용): {exc}")

    elapsed = time.time() - stage_started_at
    _done("오디오 파이프라인", elapsed)
    return {
        "segments_path": str(segments_path),
        "scenes_structure": scenes_structure,
        "slide_ranges": slide_ranges,
        "duration": duration,
        "elapsed": elapsed,
    }


def build_analyzer_input(
    args,
    meta_path: str,
    textualized_path: str,
    segments_path: str,
    output_dir: Path,
    duration: float,
    slides_structure: Optional[list[dict]] = None,
) -> dict:
    from .segment_grouper import load_slide_ranges
    from .text_processor import classify_lecture_domain

    stem = Path(args.input).stem
    analyzer_dir = output_dir / f"{stem}_analyzer"
    analyzer_dir.mkdir(parents=True, exist_ok=True)
    merged_clean_path = analyzer_dir / f"{stem}_merged_clean.json"

    if not args.force and merged_clean_path.exists() and merged_clean_path.stat().st_size > 0:
        try:
            with open(merged_clean_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            has_existing_contexts = any(slide.get("contexts") for slide in existing.get("slides", []))
            has_structured_contexts = any(scene.get("contexts") for scene in slides_structure or [])
            existing_context_ids = [
                str(ctx.get("context_id", "") or "")
                for slide in existing.get("slides", []) or []
                for ctx in slide.get("contexts", []) or []
            ]
            fallback_only = bool(existing_context_ids) and all("-V01-" in cid for cid in existing_context_ids)
            if has_existing_contexts and not (has_structured_contexts and fallback_only):
                print(f"\n  ⏭  V1 build_analyzer_input — verifier 입력 context 파일 존재, 스킵")
                print(f"     {merged_clean_path}")
                print("─" * 70)
                return {"merged_clean_path": str(merged_clean_path), "elapsed": 0.0}
        except Exception:
            pass

    _banner("V1 build_analyzer_input — verifier 입력용 merged_clean 생성")
    t0 = time.time()

    with open(textualized_path, "r", encoding="utf-8") as f:
        textualized = json.load(f)
    with open(segments_path, "r", encoding="utf-8") as f:
        segment_payload = json.load(f)
    segments = segment_payload.get("segments", [])
    slide_ranges = load_slide_ranges(meta_path, duration) if meta_path and Path(meta_path).is_file() else []
    metadata = []
    if meta_path and Path(meta_path).is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    scene_meta_by_index: dict[int, dict] = {}
    for entry in metadata:
        if entry.get("capture_type") != "base" and int(entry.get("annot_index", 0) or 0) != 0:
            continue
        scene_idx = entry.get("scene_index")
        if not isinstance(scene_idx, int) or scene_idx in scene_meta_by_index:
            continue
        scene_meta_by_index[scene_idx] = {
            "slide_number": entry.get("slide_number", entry.get("slide_canonical_index", scene_idx)),
            "slide_is_revisit": bool(
                entry.get("slide_is_revisit", entry.get("same_slide_is_revisit", False))
            ),
            "slide_visit_order": int(
                entry.get("slide_visit_order", entry.get("same_slide_visit_order", 1)) or 1
            ),
        }

    slide_meta_by_no = {}
    for slide in textualized.get("scenes", []):
        scene_no = slide.get("scene_number", slide.get("slide_number"))
        scene_info = scene_meta_by_index.get(scene_no if isinstance(scene_no, int) else -1, {})
        slide_no = scene_info.get(
            "slide_number",
            slide.get("slide_number"),
        )
        if isinstance(slide_no, int):
            text_parts = []
            if slide.get("t1"):
                text_parts.append(str(slide.get("t1")))
            if slide.get("t1_structure"):
                text_parts.append(str(slide.get("t1_structure")))
            candidate_meta = {
                "title": str(slide.get("title", "") or ""),
                "slide_text": "\n".join(part for part in text_parts if part),
                "slide_id": slide.get("slide_id", ""),
                "slide_type": slide.get("slide_type", ""),
                "text_source": slide.get("text_source", ""),
                "t1": slide.get("t1", ""),
                "t1_structure": slide.get("t1_structure", ""),
                "image_path": slide.get("image_path", ""),
            }
            current_meta = slide_meta_by_no.get(slide_no)
            if current_meta is None or len(candidate_meta["slide_text"]) > len(current_meta.get("slide_text", "")):
                slide_meta_by_no[slide_no] = candidate_meta

    segs_by_logical_slide: dict[int, list[dict]] = {}
    for seg in segments:
        slide_no = seg.get("slide_number")
        if isinstance(slide_no, int):
            seg_copy = {
                "start": float(seg.get("start", 0.0) or 0.0),
                "end": float(seg.get("end", seg.get("start", 0.0)) or 0.0),
                "text": str(seg.get("text", "") or "").strip(),
            }
            segs_by_logical_slide.setdefault(slide_no, []).append(seg_copy)

    contexts_by_slide: dict[int, list[dict]] = {}
    context_count = 0
    for slide_ctx in slides_structure or []:
        scene_idx = slide_ctx.get("slide_index", slide_ctx.get("scene_index"))
        scene_info = scene_meta_by_index.get(scene_idx if isinstance(scene_idx, int) else -1, {})
        slide_no = scene_info.get("slide_number", scene_idx)
        if not isinstance(slide_no, int):
            continue
        visit_order = int(scene_info.get("slide_visit_order", 1) or 1)
        for raw_ctx in slide_ctx.get("contexts", []) or []:
            text = str(raw_ctx.get("text", "") or "").strip()
            if not text:
                continue
            context_index = int(raw_ctx.get("context_index", 0) or 0)
            context_count += 1
            scene_token = int(scene_idx) if isinstance(scene_idx, int) else context_count
            context_id = f"S{slide_no:03d}-SC{scene_token:04d}-C{context_index + 1:03d}"
            context_payload = {
                "context_id": context_id,
                "slide_number": slide_no,
                "scene_index": scene_idx,
                "visit_order": visit_order,
                "context_index": context_index,
                "start_time": float(raw_ctx.get("start", 0.0) or 0.0),
                "end_time": float(raw_ctx.get("end", raw_ctx.get("start", 0.0)) or 0.0),
                "text": text,
                "source_segment_indices": raw_ctx.get("segment_indices", []),
            }
            contexts_by_slide.setdefault(slide_no, []).append(context_payload)

    if not contexts_by_slide:
        log.warning("V1 build_analyzer_input context 입력이 비어 있어 segment를 context 단위로 폴백합니다.")
        for slide_no, slide_segments in segs_by_logical_slide.items():
            for idx, seg in enumerate(sorted(slide_segments, key=lambda item: item.get("start", 0.0))):
                text = str(seg.get("text", "") or "").strip()
                if not text:
                    continue
                context_count += 1
                contexts_by_slide.setdefault(slide_no, []).append({
                    "context_id": f"S{slide_no:03d}-V01-C{idx + 1:03d}",
                    "slide_number": slide_no,
                    "scene_index": None,
                    "visit_order": 1,
                    "context_index": idx,
                    "start_time": float(seg.get("start", 0.0) or 0.0),
                    "end_time": float(seg.get("end", seg.get("start", 0.0)) or 0.0),
                    "text": text,
                    "source_segment_indices": [],
                })

    slide_titles = [slide_meta_by_no.get(slide_no, {}).get("title", "") for slide_no in sorted(slide_meta_by_no)]
    transcript_sample = " ".join(str(seg.get("text", "") or "") for seg in segments[:30])
    domain_info = classify_lecture_domain(slide_titles, transcript_sample)

    occurrences_by_logical_slide: dict[int, list[dict]] = {}
    for slide_range in slide_ranges:
        scene_idx = int(slide_range["scene_index"])
        scene_info = scene_meta_by_index.get(scene_idx, {})
        slide_no = scene_info.get("slide_number", scene_idx)
        if not isinstance(slide_no, int):
            continue
        start_sec = float(slide_range["start_sec"])
        end_sec = float(slide_range["end_sec"])
        slide_duration = round(end_sec - start_sec, 1)
        occurrences_by_logical_slide.setdefault(slide_no, []).append({
            "scene_index": scene_idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration": slide_duration,
            "is_dup": bool(scene_info.get("slide_is_revisit", False)),
            "visit_order": int(scene_info.get("slide_visit_order", 1) or 1),
        })

    all_slide_numbers = sorted(
        set(slide_meta_by_no)
        | set(segs_by_logical_slide)
        | set(occurrences_by_logical_slide)
        | set(contexts_by_slide)
    )
    slides = []
    for slide_no in all_slide_numbers:
        slide_meta = slide_meta_by_no.get(slide_no, {})
        transcript_segments = sorted(segs_by_logical_slide.get(slide_no, []), key=lambda item: item.get("start", 0.0))
        occurrences = sorted(
            occurrences_by_logical_slide.get(slide_no, []),
            key=lambda item: (item["start_sec"], item["scene_index"]),
        )
        if occurrences:
            start_sec = min(item["start_sec"] for item in occurrences)
            end_sec = max(item["end_sec"] for item in occurrences)
            total_duration_sec = sum(item["duration"] for item in occurrences)
        elif transcript_segments:
            start_sec = min(item["start"] for item in transcript_segments)
            end_sec = max(item["end"] for item in transcript_segments)
            total_duration_sec = round(end_sec - start_sec, 1)
        else:
            start_sec = 0.0
            end_sec = 0.0
            total_duration_sec = 0.0
        slide_payload = {
            "slide_number": slide_no,
            "title": slide_meta.get("title", ""),
            "time_range": f"{_fmt_ts(start_sec)} ~ {_fmt_ts(end_sec)}",
            "time_range_seconds": [start_sec, end_sec],
            "total_duration": round(total_duration_sec, 1),
            "occurrences": occurrences,
            "slide_text": slide_meta.get("slide_text", ""),
            "contexts": sorted(contexts_by_slide.get(slide_no, []), key=lambda item: item.get("start_time", 0.0)),
            "context_count": len(contexts_by_slide.get(slide_no, [])),
        }
        for key in (
            "slide_id",
            "slide_type",
            "text_source",
            "t1",
            "t1_structure",
            "image_path",
        ):
            value = slide_meta.get(key)
            if value not in (None, "", []):
                slide_payload[key] = value
        slides.append(slide_payload)

    total_duration = 0.0
    if slides:
        total_duration = max(float(slide["time_range_seconds"][1]) for slide in slides)
    elif segments:
        total_duration = max(float(seg.get("end", 0.0) or 0.0) for seg in segments)
    else:
        total_duration = duration

    result = {
        "description": "슬라이드+전사 통합 JSON (교정 완료, 검증용)",
        "domain": domain_info.get("domain", ""),
        "subdomain": domain_info.get("subdomain", ""),
        "total_slides": len(slides),
        "total_contexts": context_count,
        "total_duration_formatted": _fmt_ts(total_duration),
        "slides": slides,
    }
    _save_json(merged_clean_path, result)

    elapsed = time.time() - t0
    _done("analyzer 입력용 merged_clean 생성", elapsed)
    return {"merged_clean_path": str(merged_clean_path), "elapsed": elapsed}


def _claim_output_is_final_verification(claim_output_path: Path) -> bool:
    if not claim_output_path.exists():
        return False
    try:
        with open(claim_output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False
    return payload.get("mode") == "classified_issue_verifier"


def extract_claims(args, merged_clean_path: str, output_dir: Path) -> dict:
    from .analyzer.claim_extractor import (
        _claim_extract_batch_mode,
        _claim_extract_context_window,
        extract_claims_only,
    )
    from .analyzer.claim_pipeline import prepare_verification
    from .analyzer.verifier_utils import _write_claims_jsonl

    def _claim_cache_matches(path: Path, batch_mode: str, context_window: tuple[int, int]) -> bool:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return (
            payload.get("claim_batch_mode") == batch_mode
            and tuple(payload.get("claim_context_window", [])) == context_window
        )

    def _claim_output_payload(claim: dict) -> dict:
        context_id = str(claim.get("context_id") or "").strip()
        payload = {
            "claim_id": claim.get("claim_id", ""),
            "context_id": context_id,
            "claim_text": claim.get("claim_text", ""),
            "resolved_claim": claim.get("resolved_claim", ""),
            "claim_type": claim.get("claim_type", ""),
        }
        return {key: value for key, value in payload.items() if value not in ("", [], None)}

    stem = Path(args.input).stem
    analyzer_dir = output_dir / f"{stem}_analyzer"
    analyzer_dir.mkdir(parents=True, exist_ok=True)
    claim_stub_path = analyzer_dir / f"{stem}_verification_final.json"
    claims_jsonl_path = analyzer_dir / f"{stem}_claims.jsonl"
    claims_json_path = analyzer_dir / f"{stem}_claims.json"
    merged_file = Path(merged_clean_path)
    claim_batch_mode = _claim_extract_batch_mode()
    claim_context_window = _claim_extract_context_window()

    if (
        not args.force
        and claims_jsonl_path.exists()
        and _claim_cache_matches(claims_json_path, claim_batch_mode, claim_context_window)
        and claims_jsonl_path.stat().st_mtime >= merged_file.stat().st_mtime
    ):
        print(f"\n  ⏭  V2A extract_claims — claim 추출 출력 파일 존재, 스킵")
        print(f"     {claims_jsonl_path}")
        print("─" * 70)
        return {
            "claims_jsonl": str(claims_jsonl_path),
            "claims_json": str(claims_json_path),
            "claim_count": 0,
            "elapsed": 0.0,
            "skipped": True,
        }

    _banner("V2A extract_claims — claim 추출")
    t0 = time.time()
    ctx = prepare_verification(str(merged_file))
    claims_by_batch, api_calls, token_usage = extract_claims_only(
        ctx["contexts"],
        ctx["current_date"],
        ctx["hint"],
        ctx["slide_ctx"],
    )
    claims: list[dict] = []
    for _, batch_claims in claims_by_batch:
        claims.extend(_claim_output_payload(claim) for claim in batch_claims)

    claims_log_path = _write_claims_jsonl(claims, claim_stub_path)
    claims_json_path.write_text(
        json.dumps(
            {
                "mode": "claim_extraction",
                "claim_batch_mode": claim_batch_mode,
                "claim_context_window": list(claim_context_window),
                "merged_path": str(merged_file),
                "claims_log_path": claims_log_path,
                "claim_count": len(claims),
                "api_calls": api_calls,
                "token_usage": token_usage,
                "claims": claims,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    elapsed = time.time() - t0
    _done(f"claim {len(claims)}개 추출", elapsed)
    return {
        "claims_jsonl": str(claims_jsonl_path),
        "claims_json": str(claims_json_path),
        "claim_count": len(claims),
        "api_calls": api_calls,
        "elapsed": elapsed,
    }


def judge_issues(args, merged_clean_path: str, output_dir: Path, claims_jsonl: str) -> dict:
    from .analyzer.run_all import run_issue_judge_only

    stem = Path(args.input).stem
    analyzer_dir = output_dir / f"{stem}_analyzer"
    analyzer_dir.mkdir(parents=True, exist_ok=True)

    _banner("V2B judge_issues — 1차 issue 판단")
    t0 = time.time()
    result = run_issue_judge_only(
        merged_clean_path,
        output_dir=str(analyzer_dir),
        claims_jsonl=claims_jsonl,
    )
    elapsed = time.time() - t0
    _done("1차 issue judge", elapsed)
    return {"elapsed": elapsed, **result}


def start_verifier_background(args, merged_clean_path: str, output_dir: Path) -> dict:
    stem = Path(args.input).stem
    analyzer_dir = output_dir / f"{stem}_analyzer"
    analyzer_dir.mkdir(parents=True, exist_ok=True)
    claim_output_path = analyzer_dir / f"{stem}_verification_final.json"
    claim_report_path = analyzer_dir / f"{stem}_report.txt"
    analyzer_log_path = analyzer_dir / f"{stem}_analyzer.log"

    if (
        not args.force
        and _claim_output_is_final_verification(claim_output_path)
        and claim_output_path.stat().st_mtime >= Path(merged_clean_path).stat().st_mtime
    ):
        print(f"\n  ⏭  V2 start_verifier_background — verifier 출력 파일 존재, 스킵")
        print(f"     {claim_output_path}")
        print("─" * 70)
        return {
            "claim_output": str(claim_output_path),
            "claim_report": "",
            "log_path": str(analyzer_log_path),
            "elapsed": 0.0,
            "pid": None,
            "spawned": False,
        }

    _banner("V2 start_verifier_background — verifier 백그라운드 실행")
    t0 = time.time()
    pkg_root = resolve_pipeline_package_root()
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "pipeline.analyzer.run_all",
        merged_clean_path,
        "--output-dir",
        str(analyzer_dir),
    ]

    with open(analyzer_log_path, "a", encoding="utf-8") as log_fp:
        log_fp.write(
            f"\n=== verifier launch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
        )
        log_fp.write(f"cwd       : {pkg_root}\n")
        log_fp.write(f"cmd       : {' '.join(cmd)}\n")
        log_fp.write("mode      : classified_issue_pipeline\n")
        log_fp.flush()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(pkg_root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    elapsed = time.time() - t0
    print(f"\n  ✓ verifier 백그라운드 시작  ({elapsed:.1f}초)")
    print(f"     PID : {proc.pid}")
    print("     mode: classified_issue_pipeline")
    print(f"     로그: {analyzer_log_path}")
    print("─" * 70)
    return {
        "claim_output": str(claim_output_path),
        "claim_report": "",
        "log_path": str(analyzer_log_path),
        "elapsed": elapsed,
        "pid": proc.pid,
        "spawned": True,
    }


def run_verifier(
    args,
    merged_clean_path: str,
    output_dir: Path,
    notify_stage: Callable[..., None] | None = None,
) -> dict:
    """Run the verifier synchronously for approval-gated uploads."""
    from .analyzer.run_all import run_classified_issue_pipeline

    stem = Path(args.input).stem
    analyzer_dir = output_dir / f"{stem}_analyzer"
    analyzer_dir.mkdir(parents=True, exist_ok=True)
    claim_output_path = analyzer_dir / f"{stem}_verification_final.json"
    claim_report_path = analyzer_dir / f"{stem}_report.txt"

    if (
        not args.force
        and _claim_output_is_final_verification(claim_output_path)
        and claim_output_path.stat().st_mtime >= Path(merged_clean_path).stat().st_mtime
    ):
        print(f"\n  ⏭  V2 run_verifier — verifier 출력 파일 존재, 스킵")
        print(f"     {claim_output_path}")
        print("─" * 70)
        if notify_stage:
            for stage_key in VERIFIER_DETAIL_STAGE_KEYS:
                notify_stage(stage_key, "done")
        return {
            "claim_output": str(claim_output_path),
            "claim_report": str(claim_report_path) if claim_report_path.exists() else "",
            "log_path": "",
            "elapsed": 0.0,
            "pid": None,
            "spawned": False,
            "skipped": True,
        }

    _banner("V2 run_verifier — verifier 실행")
    t0 = time.time()
    result = run_classified_issue_pipeline(
        merged_clean_path,
        output_dir=str(analyzer_dir),
        stage_notify=notify_stage,
    )
    elapsed = time.time() - t0
    _done("verifier 실행", elapsed)
    return {
        "claim_output": result.get("claim_output", str(claim_output_path)),
        "claim_report": result.get("claim_report", str(claim_report_path)),
        "log_path": result.get("log_path", ""),
        "elapsed": elapsed,
        "pid": None,
        "spawned": False,
        **result,
    }


# ──────────────────────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────────────────────

def run_preprocess_pipeline(
    args,
    *,
    stem: str,
    output_dir: Path,
    slides_dir: Path,
    paths: dict,
    timings: dict[str, float],
    notify_stage,
) -> dict:
    """Run shared preprocessing stages used by verifier and graph workflows."""
    from .orchestration.preprocess import run_preprocess_pipeline as _run_preprocess_pipeline

    return _run_preprocess_pipeline(
        args,
        stem=stem,
        output_dir=output_dir,
        slides_dir=slides_dir,
        paths=paths,
        timings=timings,
        notify_stage=notify_stage,
        helpers=sys.modules[__name__],
    )

def save_preprocess_manifest(stem: str, output_dir: Path, preprocess_result: dict) -> Path:
    """Persist the in-memory preprocess payload so graph_upload can resume later."""
    from .orchestration.preprocess import save_preprocess_manifest as _save_preprocess_manifest

    return _save_preprocess_manifest(
        stem,
        output_dir,
        preprocess_result,
        helpers=sys.modules[__name__],
    )

def load_preprocess_result_from_outputs(stem: str, output_dir: Path, paths: dict) -> dict:
    """Restore preprocess_result from the manifest written by run_preprocess_pipeline."""
    from .orchestration.preprocess import load_preprocess_result_from_outputs as _load_preprocess_result_from_outputs

    return _load_preprocess_result_from_outputs(stem, output_dir, paths)

def run_verifier_pipeline(
    args,
    *,
    preprocess_result: dict,
    output_dir: Path,
    paths: dict,
    timings: dict[str, float],
    background: bool = True,
    notify_stage=lambda _stage, _status, _progress=None: None,
) -> dict:
    """Build verifier input and run the verifier path."""
    from .orchestration.verifier import run_verifier_pipeline as _run_verifier_pipeline

    return _run_verifier_pipeline(
        args,
        preprocess_result=preprocess_result,
        output_dir=output_dir,
        paths=paths,
        timings=timings,
        background=background,
        notify_stage=notify_stage,
        helpers=sys.modules[__name__],
    )

def _print_generated_files(output_files: list[str]) -> None:
    print("\n  생성된 파일:")
    for path_str in output_files:
        if not path_str:
            continue
        p = Path(path_str)
        print(f"    {'✓' if p.exists() else '✗'}  {p}")


def run_pipeline(args, progress_callback=None):
    from .orchestration.workflows import run_pipeline as _run_pipeline

    return _run_pipeline(args, progress_callback, helpers=sys.modules[__name__])

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def get_parser():
    
    from .config import DEFAULT_SLIDES_DIR, DEFAULT_OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="강의 영상 분석 통합 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py --input input/lecture.mp4
  python main.py --input input/lecture.mp4 --output output/ --slides output_slides/
  python main.py --input input/lecture.mp4 --skip-extract
  python main.py --input input/lecture.mp4 --debug --masks
  python main.py --input input/lecture.mp4 --force
        """,
    )
    parser.add_argument("--input",  "-i", default="input/lecture.mp4", help="입력 강의 영상 경로 (.mp4)")
    parser.add_argument("--slides", "-s", default=str(DEFAULT_SLIDES_DIR),
                        help=f"슬라이드 프레임 저장 디렉토리 (default: {DEFAULT_SLIDES_DIR})")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT_DIR),
                        help=f"분석 결과 저장 디렉토리 (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--skip-extract", action="store_true",
                        help="P1A extract_slides 건너뜀 (이미 슬라이드가 추출된 경우)")
    parser.add_argument("--force", action="store_true",
                        help="출력 파일이 있어도 모든 단계 강제 재실행")
    parser.add_argument("--retries", type=int, default=3,
                        help="Gemini API 재시도 횟수 (default: 3)")
    parser.add_argument("--debug", action="store_true", help="P1 extract_media 디버그 로그 출력")
    parser.add_argument(
        "--slide-decode-backend",
        choices=["opencv", "ffmpeg-cuda", "ffmpeg-videotoolbox", "auto"],
        default=os.getenv("VLVERIFIER_SLIDE_DECODE_BACKEND", "auto"),
        help="P1A extract_slides 프레임 디코드 백엔드 (default: auto)",
    )
    parser.add_argument(
        "--slide-extract-workers",
        type=int,
        default=int(os.getenv("VLVERIFIER_SLIDE_EXTRACT_WORKERS", "0")),
        help="P1A extract_slides 시간 청크 병렬 추출 worker 수 (기본: 0, chunk 개수만큼 자동)",
    )
    parser.add_argument("--skip-analyzer", action="store_true",
                        help="V2 verifier 실행 스킵")
    parser.add_argument(
        "--stop-after-claim-extract",
        action="store_true",
        help="P3 process_audio context 기반 analyzer 입력 생성 및 claim 추출 후 종료",
    )
    parser.add_argument(
        "--stop-after-issue-judge",
        action="store_true",
        help="P3 process_audio context 기반 analyzer 입력 생성, claim 추출, 1차 issue judge 후 종료",
    )
    parser.add_argument("--title",      default="", help="강의명 (미입력 시 Gemini 자동 생성)")
    parser.add_argument("--lecture-id", dest="lecture_id", default=None,
                        help="lecture_metadata DB 저장 대상 lectures.id")
    parser.add_argument("--uploaded-at", dest="uploaded_at", default=None,
                        help="강의 업로드 시각 ISO 문자열")
    parser.add_argument("--job-type", dest="job_type", default=JOB_TYPE_LEGACY_FULL,
                        choices=sorted(PIPELINE_JOB_TYPES),
                        help=f"실행 workflow 타입 (default: {JOB_TYPE_LEGACY_FULL})")
    
    return parser

def main():
    args = get_parser().parse_args()

    if not args.skip_extract and not Path(args.input).exists():
        print(f"❌ 입력 영상 없음: {args.input}")
        sys.exit(1)

    with pipeline_log_context(args.output):
        run_pipeline(args)


if __name__ == "__main__":
    main()
