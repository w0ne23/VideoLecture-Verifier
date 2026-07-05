import asyncio
import os
import re
import shutil
import logging
import json
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
import pandas as pd
import httpx
from pathlib import Path
from typing import Optional, List, Any, Dict

from sqlalchemy import select, delete, and_, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException

from app.models import (
    ACTIVE_STATUSES,
    JOB_STATUS_DONE,
    JOB_STATUS_ERROR,
    JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING,
    JOB_STATUS_WAITING_APPROVAL,
    JOB_TYPE_DIRECT_UPLOAD,
    JOB_TYPE_GRAPH_UPLOAD,
    JOB_TYPE_LEGACY_FULL,
    JOB_TYPE_PUBLISH,
    JOB_TYPE_UPLOAD,
    JOB_TYPE_VERIFY,
    JOB_TYPE_VERIFIED_UPLOAD,
    Lecture,
    LectureMetadata,
    ProcessingJob,
    GraphSession,
    ChatSession,
    ChatMessage,
    normalize_job_type,
)
from app.services.neo4j_service import (
    neo4j_session,
    get_stem_load_lock,
    _is_stem_loaded,
    _unload_stem_from_neo4j,
    _ensure_stem_loaded,
)

logger = logging.getLogger(__name__)

# ── 경로 설정 ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/pipeline") if Path("/pipeline").exists() else Path(__file__).resolve().parents[4]
LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR", str(PROJECT_ROOT / "local_storage"))
GRAPH_SESSION_TTL_SEC = int(os.getenv("GRAPH_SESSION_TTL_SEC", "180"))
CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "6"))
QNA_QUERY_TIMEOUT_SEC = float(os.getenv("QNA_QUERY_TIMEOUT_SEC", "30"))
QNA_DEMO_FALLBACK_ENABLED = os.getenv("QNA_DEMO_FALLBACK_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
DOMAIN_VALUES = {
    "engineering",
    "natural_science",
    "humanities",
    "social_science",
    "arts",
    "health_sciences",
    "sports",
    "education",
    "etc",
}
DOMAIN_ALIASES = {
    "eng": "engineering",
    "eng/cs": "engineering",
    "eng/electrical": "engineering",
    "eng/mechanical": "engineering",
    "eng/civil": "engineering",
    "eng/chemical": "engineering",
    "eng/industrial": "engineering",
    "eng/biomedical": "engineering",
    "eng/aerospace": "engineering",
    "eng/materials": "engineering",
    "eng/environmental": "engineering",
    "sci": "natural_science",
    "sci/physics": "natural_science",
    "sci/chemistry": "natural_science",
    "sci/biology": "natural_science",
    "sci/earth_science": "natural_science",
    "sci/astronomy": "natural_science",
    "sci/ecology": "natural_science",
    "hum": "humanities",
    "hum/philosophy": "humanities",
    "hum/history": "humanities",
    "hum/linguistics": "humanities",
    "hum/literature": "humanities",
    "hum/art_history": "humanities",
    "hum/religion": "humanities",
    "soc": "social_science",
    "soc/economics": "social_science",
    "soc/business": "social_science",
    "soc/law": "social_science",
    "soc/political_science": "social_science",
    "soc/sociology": "social_science",
    "soc/psychology": "social_science",
    "med": "health_sciences",
    "med/anatomy": "health_sciences",
    "med/physiology": "health_sciences",
    "med/pharmacology": "health_sciences",
    "med/clinical": "health_sciences",
    "med/public_health": "health_sciences",
    "med/nursing": "health_sciences",
    "art": "arts",
    "art/fine_arts": "arts",
    "art/music": "arts",
    "art/design": "arts",
    "art/film": "arts",
    "art/theater": "arts",
    "art/physical_education": "sports",
    "art/sports_science": "sports",
    "gen": "etc",
    "gen/other": "etc",
    "default category": "etc",
    "기타": "etc",
}


# ── 직렬화 헬퍼 ──────────────────────────────────────────────────────────────
PUBLICATION_JOB_TYPES = {JOB_TYPE_LEGACY_FULL, JOB_TYPE_PUBLISH}
PUBLICATION_DB_JOB_TYPES = [
    JOB_TYPE_LEGACY_FULL,
    JOB_TYPE_PUBLISH,
    JOB_TYPE_DIRECT_UPLOAD,
    JOB_TYPE_GRAPH_UPLOAD,
    JOB_TYPE_UPLOAD,
]
VERIFICATION_DB_JOB_TYPES = [
    JOB_TYPE_VERIFY,
    JOB_TYPE_VERIFIED_UPLOAD,
]


def _is_publication_job_type(job_type: Optional[str]) -> bool:
    return normalize_job_type(job_type) in PUBLICATION_JOB_TYPES


def format_job_dict(job: ProcessingJob, lecture: Optional[Lecture]) -> Dict[str, Any]:
    """ProcessingJob과 Lecture를 하나의 딕셔너리로 직렬화"""
    stages = job.pipeline_stages
    if isinstance(stages, str):
        try:
            stages = json.loads(stages)
        except Exception:
            stages = []
    elif stages is None:
        stages = []

    res = {
        "job_id": str(job.id),
        "job_type": getattr(job, "job_type", None) or JOB_TYPE_LEGACY_FULL,
        "status": job.status,
        "current_stage": job.current_stage,
        "error_message": job.error_message,
        "pipeline_stages": stages,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "is_verified": bool(getattr(lecture, "is_verified", False)) if lecture else False,
        "is_published": bool(getattr(lecture, "is_published", False)) if lecture else False,
        "content": [],
    }
    if lecture:
        res["video_path"] = lecture.video_path
        res["content"].append({
            "id": str(lecture.id),
            "title": lecture.title,
            "category": lecture.category,
            "description": lecture.description,
            "is_verified": bool(getattr(lecture, "is_verified", False)),
            "is_published": bool(getattr(lecture, "is_published", False)),
            "stem": str(lecture.id),
            "output_dir": lecture.output_dir,
        })
    return res


def make_file_url(abs_path: Optional[str]) -> Optional[str]:
    if not abs_path:
        return None
    try:
        p = Path(abs_path).as_posix()
        if "local_storage/" in p:
            return f"/files/{p.split('local_storage/')[1]}"
    except Exception:
        pass
    return None


def normalize_domain_value(value: Any) -> str:
    token = _str_cell(value).strip()
    if not token:
        return "etc"
    normalized = token.lower().replace("-", "_")
    if normalized in DOMAIN_VALUES:
        return normalized
    return DOMAIN_ALIASES.get(normalized) or DOMAIN_ALIASES.get(token) or "etc"


def _lecture_domain_value(lecture: Lecture, metadata: Optional[dict] = None) -> str:
    metadata = metadata or {}
    return normalize_domain_value(
        metadata.get("graph_domain")
        or metadata.get("domain")
        or getattr(lecture, "category", None)
    )


def _lecture_metadata_dict(row: Optional[LectureMetadata]) -> dict:
    if not row:
        return {}
    return {
        "domain": row.domain,
        "graph_domain": row.graph_domain,
        "graph_subdomain": row.graph_subdomain,
    }


def _lecture_file_metadata(lecture: Lecture) -> dict:
    if not lecture.output_dir:
        return {}
    return _first_existing_json([
        Path(lecture.output_dir) / "metadata" / f"{lecture.id}_metadata.json",
        Path(lecture.output_dir) / f"{lecture.id}_metadata.json",
    ])


def _first_existing_file(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.is_file():
            return path
    return None


def _lecture_thumbnail_url(output_dir_value: Optional[str]) -> Optional[str]:
    if not output_dir_value:
        return None

    output_dir = Path(output_dir_value)
    candidates: list[Path] = []
    for ext in ("jpg", "jpeg", "png", "webp"):
        candidates.extend([
            output_dir / "slides" / f"scene_001_base.{ext}",
            output_dir / "slides_staged" / "scenes" / f"scene_001_base.{ext}",
            output_dir / "slides_staged" / "review_slides" / f"scene_001_base.{ext}",
            output_dir / f"scene_001_base.{ext}",
        ])

    first_scene = _first_existing_file(candidates)
    if first_scene:
        return make_file_url(str(first_scene))

    base_images: list[Path] = []
    for base_dir in (
        output_dir / "slides",
        output_dir / "slides_staged" / "scenes",
        output_dir / "slides_staged" / "review_slides",
        output_dir,
    ):
        if base_dir.is_dir():
            base_images.extend(sorted(base_dir.glob("scene_*_base.*")))

    first_base = _first_existing_file(base_images)
    return make_file_url(str(first_base)) if first_base else None


# ── 그래프 유틸 ──────────────────────────────────────────────────────────────
def _hex_color(key: str) -> str:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return f"#{h[:6]}"


def _str_cell(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def _read_json_file(path: Path) -> dict:
    try:
        if path.exists() and path.stat().st_size > 0:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _first_existing_json(paths: list[Path]) -> dict:
    for path in paths:
        data = _read_json_file(path)
        if data:
            return data
    return {}


def _float_val(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_mmss(seconds: Any) -> str:
    sec = max(0, int(_float_val(seconds)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _keyword_label(item: Any) -> str:
    if isinstance(item, dict):
        return _str_cell(item.get("keyword") or item.get("name") or item.get("text"))
    return _str_cell(item)


def _context_emphasis_score(ctx: dict) -> float:
    score = ctx.get("emphasis_score") or {}
    if isinstance(score, dict):
        return _float_val(score.get("total"))
    return _float_val(score)


def _read_video_domain(output_dir: Path, stem: str) -> tuple[str, str]:
    nodes_path = output_dir / f"{stem}_nodes.parquet"
    if not nodes_path.is_file():
        return "", ""
    try:
        ndf = pd.read_parquet(nodes_path)
        video_id = f"lecture_video/{stem}"
        for _, row in ndf.iterrows():
            if _str_cell(row.get("node_id")) != video_id:
                continue
            props = json.loads(_str_cell(row.get("properties_json")) or "{}")
            if not isinstance(props, dict):
                return "", ""
            return _str_cell(props.get("domain")), _str_cell(props.get("subdomain"))
    except Exception:
        return "", ""
    return "", ""


def _format_domain_label(domain: str, fallback: str) -> str:
    normalized = normalize_domain_value(domain or fallback)
    return normalized or "etc"


def _visual_asset_stats_from_fused(fused: dict) -> dict:
    entries = fused.get("slides") if isinstance(fused.get("slides"), list) and fused.get("slides") else fused.get("scenes")
    if not isinstance(entries, list) or not entries:
        return {"visual_percent": 0, "visual_count": 0, "total_count": 0}

    visual_asset_types = {"diagram", "table", "chart", "figure", "image"}
    total_count = 0
    visual_count = 0
    for item in entries:
        if not isinstance(item, dict):
            continue
        total_count += 1
        visual_assets = item.get("visual_assets")
        has_visual_asset = False
        has_asset_metadata = isinstance(visual_assets, list) and len(visual_assets) > 0
        if isinstance(visual_assets, list):
            for asset in visual_assets:
                if not isinstance(asset, dict):
                    continue
                asset_type = _str_cell(asset.get("asset_type") or asset.get("type")).strip().lower()
                if asset_type in visual_asset_types:
                    has_visual_asset = True
                    break
        slide_type = _str_cell(item.get("slide_type")).strip().lower()
        if has_visual_asset or slide_type in {"image_only", "diagram", "chart", "figure"}:
            visual_count += 1

    visual_percent = round((visual_count / total_count) * 100) if total_count else 0
    return {
        "visual_percent": visual_percent,
        "visual_count": visual_count,
        "total_count": total_count,
    }


def _format_percent(value: Any) -> str:
    try:
        return f"{int(round(float(value)))}%"
    except (TypeError, ValueError):
        return "0%"


def _split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", _str_cell(text)).strip()
    if not compact:
        return []
    pattern = r".+?(?:[.!?。！？]+|(?:다|요|죠|니다|습니다|어요|예요|에요)(?=\s|$))"
    sentences = [m.group(0).strip() for m in re.finditer(pattern, compact)]
    consumed = sum(len(s) for s in sentences)
    if consumed < len(compact):
        rest = compact[consumed:].strip()
        if rest:
            sentences.append(rest)
    return [s for s in sentences if s]


def _trim_at_word_boundary(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", _str_cell(text)).strip()
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0].rstrip()
    return f"{clipped}..."


def _highlight_excerpt(text: str, min_chars: int = 35, max_chars: int = 120, max_sentences: int = 3) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    picked: list[str] = []
    for sentence in sentences[:max_sentences]:
        candidate = " ".join(picked + [sentence]).strip()
        if picked and len(candidate) > max_chars:
            break
        picked.append(sentence)
        if len(candidate) >= min_chars:
            break
    return _trim_at_word_boundary(" ".join(picked), max_chars)


def _build_lecture_info(output_dir: Path, stem: str, fallback_category: str) -> dict:
    metadata = _first_existing_json([
        output_dir / "metadata" / f"{stem}_metadata.json",
        output_dir / f"{stem}_metadata.json",
    ])
    fused = _first_existing_json([output_dir / f"{stem}_fused.json"])

    summary = _str_cell(metadata.get("summary"))
    graph_domain = _str_cell(metadata.get("graph_domain"))
    graph_subdomain = _str_cell(metadata.get("graph_subdomain"))
    if not graph_domain:
        graph_domain, graph_subdomain = _read_video_domain(output_dir, stem)
    domain = _format_domain_label(
        graph_domain or metadata.get("domain"),
        fallback_category,
    )
    visual_stats = _visual_asset_stats_from_fused(fused)
    keywords = [
        kw for kw in (_keyword_label(item) for item in (metadata.get("keywords") or []))
        if kw
    ][:8]

    scenes = fused.get("scenes") if isinstance(fused.get("scenes"), list) else []
    contexts: list[dict] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        for ctx in scene.get("contexts") or []:
            if not isinstance(ctx, dict):
                continue
            contexts.append({
                "score": _context_emphasis_score(ctx),
                "start_sec": ctx.get("start"),
                "text": _str_cell(ctx.get("text")),
                "scene_number": scene.get("scene_number") or scene.get("scene_index"),
                "slide_number": scene.get("slide_number"),
            })

    scored_contexts = [ctx for ctx in contexts if ctx["score"] > 0]
    scored_contexts.sort(key=lambda x: x["score"], reverse=True)
    if scored_contexts:
        top_20_idx = min(len(scored_contexts) - 1, max(0, int(len(scored_contexts) * 0.2) - 1))
        threshold = max(0.8, scored_contexts[top_20_idx]["score"])
        emphasis_context_count = sum(1 for ctx in scored_contexts if ctx["score"] >= threshold)
    else:
        threshold = 0.8
        emphasis_context_count = 0

    highlights = []
    for ctx in sorted(scored_contexts[:3], key=lambda x: _float_val(x.get("start_sec"))):
        text = _highlight_excerpt(ctx["text"])
        highlights.append({
            "timestamp": _format_mmss(ctx.get("start_sec")),
            "start_sec": _float_val(ctx.get("start_sec")),
            "text": text,
            "score": round(ctx["score"], 3),
            "scene_number": ctx.get("scene_number"),
            "slide_number": ctx.get("slide_number"),
        })

    scene_count = len(scenes)
    return {
        "summary": summary,
        "domain": domain,
        "graph_domain": graph_domain,
        "graph_subdomain": graph_subdomain,
        "visual_asset_percent": visual_stats["visual_percent"],
        "visual_asset_label": _format_percent(visual_stats["visual_percent"]),
        "visual_asset_count": visual_stats["visual_count"],
        "visual_asset_total": visual_stats["total_count"],
        "keywords": keywords,
        "highlights": highlights,
        "stats": {
            "scene_count": scene_count,
            "emphasis_contexts": emphasis_context_count,
            "emphasis_threshold": round(threshold, 3),
            "visual_asset_percent": visual_stats["visual_percent"],
            "visual_asset_count": visual_stats["visual_count"],
            "visual_asset_total": visual_stats["total_count"],
        },
    }


# ── ProcessingJob CRUD ───────────────────────────────────────────────────────
async def get_job(db: AsyncSession, job_id: str) -> Optional[ProcessingJob]:
    try:
        # UUID 형식 검증
        uuid.UUID(str(job_id))
    except (ValueError, TypeError):
        return None
        
    result = await db.execute(
        select(ProcessingJob).where(ProcessingJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def get_latest_job(db: AsyncSession, lecture_id: str) -> Optional[ProcessingJob]:
    """lecture_id로 가장 최근 job을 반환"""
    try:
        ident_uuid = uuid.UUID(str(lecture_id))
    except (ValueError, TypeError):
        return None
    result = await db.execute(
        select(ProcessingJob)
        .where(ProcessingJob.lecture_id == ident_uuid)
        .order_by(ProcessingJob.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_latest_job_by_mode(db: AsyncSession, lecture_id: str, mode: str) -> Optional[ProcessingJob]:
    """lecture_id와 route mode에 맞는 가장 최근 job을 반환."""
    try:
        ident_uuid = uuid.UUID(str(lecture_id))
    except (ValueError, TypeError):
        return None

    canonical_mode = normalize_job_type(mode)
    if canonical_mode == JOB_TYPE_VERIFY:
        job_types = VERIFICATION_DB_JOB_TYPES
    elif canonical_mode in PUBLICATION_JOB_TYPES:
        job_types = PUBLICATION_DB_JOB_TYPES
    else:
        return None

    result = await db.execute(
        select(ProcessingJob)
        .where(
            ProcessingJob.lecture_id == ident_uuid,
            ProcessingJob.job_type.in_(job_types),
        )
        .order_by(ProcessingJob.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_job_detail(db: AsyncSession, job_id: str) -> Optional[Dict[str, Any]]:
    query = (
        select(ProcessingJob, Lecture)
        .join(Lecture, ProcessingJob.lecture_id == Lecture.id)
        .where(ProcessingJob.id == job_id)
    )
    result = await db.execute(query)
    row = result.unique().one_or_none()
    if not row:
        return None
    return format_job_dict(row[0], row[1])


async def list_jobs(db: AsyncSession, status_filter: Optional[str] = None):
    query = (
        select(Lecture, ProcessingJob, LectureMetadata)
        .outerjoin(ProcessingJob, ProcessingJob.lecture_id == Lecture.id)
        .outerjoin(LectureMetadata, LectureMetadata.lecture_id == Lecture.id)
        .order_by(Lecture.created_at.desc(), ProcessingJob.created_at.desc())
    )
    result = await db.execute(query)
    rows = result.unique().all()

    seen = set()
    out = []
    for lecture, job, lecture_metadata in rows:
        if lecture.id in seen:
            continue
        seen.add(lecture.id)
        metadata = _lecture_metadata_dict(lecture_metadata) or _lecture_file_metadata(lecture)
        domain = _lecture_domain_value(lecture, metadata)
        job_status = job.status if job else 'unknown'
        if status_filter == 'active' and job_status not in ACTIVE_STATUSES:
            continue
        is_done = job_status == "done"
        out.append({
            "id": str(lecture.id),
            "job_id": str(job.id) if job else None,
            "job_type": (getattr(job, "job_type", None) or JOB_TYPE_LEGACY_FULL) if job else None,
            "status": job_status,
            "current_stage": job.current_stage if job and not is_done else None,
            "error_message": job.error_message if job else None,
            "pipeline_stages": job.pipeline_stages or [] if job and not is_done else [],
            "title": lecture.title or str(lecture.id),
            "category": domain,
            "domain": domain,
            "is_verified": bool(getattr(lecture, "is_verified", False)),
            "is_published": bool(getattr(lecture, "is_published", False)),
            "created_at": lecture.created_at.isoformat() if lecture.created_at else None,
            "thumbnail_url": _lecture_thumbnail_url(lecture.output_dir),
        })
    return out


async def retry_lecture(db: AsyncSession, lecture_id: str, mode: Optional[str] = None):
    """lecture_id로 새 ProcessingJob을 INSERT하여 재시도 이력을 누적.
    성공 시 { status, job_id } dict 반환, 실패 시 None.
    """
    try:
        ident_uuid = uuid.UUID(str(lecture_id))
    except (ValueError, TypeError):
        return None

    lecture = await _get_lecture(db, ident_uuid)
    if not lecture:
        return None

    if mode:
        job_type = normalize_job_type(mode)
    else:
        latest_job = await get_latest_job(db, str(ident_uuid))
        job_type = (getattr(latest_job, "job_type", None) or JOB_TYPE_LEGACY_FULL) if latest_job else JOB_TYPE_LEGACY_FULL

    if job_type == JOB_TYPE_VERIFY:
        current_stage = "검증 파이프라인을 다시 시작합니다."
    elif _is_publication_job_type(job_type):
        current_stage = "업로드 파이프라인을 다시 시작합니다."
    else:
        current_stage = "파이프라인을 다시 시작합니다."

    new_job = ProcessingJob(
        id=uuid.uuid4(),
        lecture_id=ident_uuid,
        job_type=job_type,
        status="pending",
        current_stage=current_stage,
        error_message=None,
        pipeline_stages=[],
    )
    db.add(new_job)
    await db.commit()
    await db.refresh(new_job)
    return {"status": "success", "job_id": str(new_job.id), "job_type": new_job.job_type}


async def confirm_verified_lecture(db: AsyncSession, lecture_id: str):
    """Mark a completed verification as reviewed without starting upload."""
    try:
        ident_uuid = uuid.UUID(str(lecture_id))
    except (ValueError, TypeError):
        return None

    lecture = await _get_lecture(db, str(ident_uuid))
    if not lecture:
        return None

    approval_result = await db.execute(
        select(ProcessingJob)
        .where(
            ProcessingJob.lecture_id == ident_uuid,
            ProcessingJob.job_type.in_([JOB_TYPE_VERIFY, JOB_TYPE_VERIFIED_UPLOAD]),
            ProcessingJob.status.in_([JOB_STATUS_DONE, JOB_STATUS_WAITING_APPROVAL]),
        )
        .order_by(ProcessingJob.created_at.desc())
        .limit(1)
        .with_for_update()
    )
    approval_job = approval_result.scalar_one_or_none()

    if not approval_job:
        raise HTTPException(status_code=409, detail="No completed verification is ready for review")

    approval_job.status = JOB_STATUS_DONE
    approval_job.current_stage = "검토 완료"
    approval_job.error_message = None
    lecture.is_verified = True

    await db.commit()
    await db.refresh(approval_job)
    await db.refresh(lecture)
    return {
        "status": "success",
        "lecture_id": str(lecture.id),
        "job_id": str(approval_job.id),
        "job_type": approval_job.job_type,
        "is_verified": bool(lecture.is_verified),
        "is_published": bool(getattr(lecture, "is_published", False)),
    }


async def delete_lecture(db: AsyncSession, lecture_id: str) -> bool:
    """lecture_id로 강의 전체 삭제 — DB, 로컬 파일, Neo4j 모두 정리"""
    try:
        ident_uuid = uuid.UUID(str(lecture_id))
    except (ValueError, TypeError):
        return False

    lecture_result = await db.execute(
        select(Lecture).where(Lecture.id == ident_uuid)
    )
    lecture = lecture_result.scalar_one_or_none()
    if not lecture:
        return False

    stem = str(lecture.id) # stem = lecture_id

    # 파일 정리 — inputs/{lecture_id}/ 와 results/{lecture_id}/
    if lecture.video_path:
        shutil.rmtree(Path(lecture.video_path).parent, ignore_errors=True)
    if lecture.output_dir:
        shutil.rmtree(Path(lecture.output_dir), ignore_errors=True)

    # Neo4j 정리
    try:
        with neo4j_session() as session:
            session.run("MATCH (n {stem: $stem}) DETACH DELETE n", stem=stem)
    except HTTPException:
        pass

    # Lecture CASCADE로 ProcessingJob, GraphSession 함께 삭제
    await db.execute(delete(Lecture).where(Lecture.id == lecture.id))
    await db.commit()
    return True


# ── 결과 조회 (Lecture ID 기준) ───────────────────────────────────────────────
async def list_all_results(
    db: AsyncSession,
    page: int = 1,
    limit: int = 12,
    category: Optional[str] = None,
    search: Optional[str] = None,
    scope: str = 'browse',
    verified_only: bool = False,
) -> Dict[str, Any]:
    query = (
        select(Lecture, ProcessingJob, LectureMetadata)
        .outerjoin(ProcessingJob, ProcessingJob.lecture_id == Lecture.id)
        .outerjoin(LectureMetadata, LectureMetadata.lecture_id == Lecture.id)
        .order_by(Lecture.created_at.desc(), ProcessingJob.created_at.desc())
    )
    result = await db.execute(query)
    rows = result.unique().all()

    seen = set()
    out = []
    for lecture, job, lecture_metadata in rows:
        if lecture.id in seen:
            continue
        seen.add(lecture.id)
        metadata = _lecture_metadata_dict(lecture_metadata) or _lecture_file_metadata(lecture)
        domain = _lecture_domain_value(lecture, metadata)
        job_status = job.status if job else 'unknown'
        job_type = (getattr(job, "job_type", None) or JOB_TYPE_LEGACY_FULL) if job else None
        is_published = bool(getattr(lecture, "is_published", False))
        is_publication_job = _is_publication_job_type(job_type)

        if scope == 'browse' and not is_published:
            continue
        if scope == 'upload' and not is_published and not (
            is_publication_job and job_status == JOB_STATUS_ERROR
        ):
            continue
        if scope == 'upload' and job_status in ACTIVE_STATUSES:
            continue

        if verified_only and not bool(getattr(lecture, "is_verified", False)):
            continue
        if category and domain != normalize_domain_value(category):
            continue
        if search and search.lower() not in (lecture.title or '').lower():
            continue

        out.append({
            "id": str(lecture.id),
            "job_id": str(job.id) if job else None,
            "job_type": job_type,
            "status": job_status,
            "title": lecture.title or str(lecture.id),
            "category": domain,
            "domain": domain,
            "is_verified": bool(getattr(lecture, "is_verified", False)),
            "is_published": is_published,
            "created_at": lecture.created_at.isoformat() if lecture.created_at else None,
            "thumbnail_url": _lecture_thumbnail_url(lecture.output_dir),
            "error_message": job.error_message if job else None,
            "pipeline_stages": job.pipeline_stages or [] if job else [],
        })

    total_items = len(out)
    start = (page - 1) * limit
    paginated = out[start: start + limit]
    return {"items": paginated, "total_items": total_items}


async def get_lecture_detail(db: AsyncSession, lecture_id: str) -> Optional[Dict[str, Any]]:
    """강의 상세 정보 조회.

    프론트가 업로드 직후 job_id를 들고 있는 경우가 있어 Lecture.id,
    Job.id, Lecture.job_id, stem을 모두 허용한다.
    """
    row = await _get_lecture_row(db, lecture_id)
    if not row:
        return None
    lecture, job = row
    stem = str(lecture.id)
    output_dir = Path(lecture.output_dir) if lecture.output_dir else None
    info = _build_lecture_info(output_dir, stem, lecture.category or "기타") if output_dir else {
        "summary": "",
        "domain": normalize_domain_value(lecture.category),
        "graph_domain": "",
        "graph_subdomain": "",
        "keywords": [],
        "highlights": [],
        "stats": {
            "scene_count": 0,
            "scene_transitions": 0,
            "emphasis_contexts": 0,
            "stt_confidence": 94,
        },
    }
    return {
        "id": str(lecture.id),
        "job_id": str(job.id) if job else None,
        "job_type": (getattr(job, "job_type", None) or JOB_TYPE_LEGACY_FULL) if job else None,
        "status": job.status if job else "unknown",
        "title": lecture.title or stem,
        "category": info.get("domain") or normalize_domain_value(lecture.category),
        "is_verified": bool(getattr(lecture, "is_verified", False)),
        "is_published": bool(getattr(lecture, "is_published", False)),
        "description": lecture.description,
        "summary": info.get("summary") or "",
        "keywords": info.get("keywords") or [],
        "domain": info.get("domain") or normalize_domain_value(lecture.category),
        "graph_domain": info.get("graph_domain") or "",
        "graph_subdomain": info.get("graph_subdomain") or "",
        "info": info,
        "stem": stem,
        "video_url": make_file_url(lecture.video_path),
        "thumbnail_url": _lecture_thumbnail_url(lecture.output_dir),
        "output_dir": lecture.output_dir,
        "graphrag_workspace": str(Path(lecture.output_dir) / "graphrag") if lecture.output_dir else None,
        "created_at": lecture.created_at.isoformat() if lecture.created_at else None,
    }



async def _get_lecture_row(db: AsyncSession, lecture_id: str):
    """lecture_id로 Lecture + 최신 ProcessingJob을 함께 반환"""
    try:
        ident_uuid = uuid.UUID(str(lecture_id))
    except (ValueError, TypeError):
        return None
    query = (
        select(Lecture, ProcessingJob)
        .outerjoin(ProcessingJob, ProcessingJob.lecture_id == Lecture.id)
        .where(Lecture.id == ident_uuid)
        .order_by(ProcessingJob.created_at.desc())
    )
    result = await db.execute(query)
    return result.unique().first()


async def _get_lecture(db: AsyncSession, lecture_id: str) -> Optional[Lecture]:
    """lecture_id로 Lecture 객체를 반환"""
    try:
        ident_uuid = uuid.UUID(str(lecture_id))
    except (ValueError, TypeError):
        return None
    result = await db.execute(select(Lecture).where(Lecture.id == ident_uuid))
    return result.scalar_one_or_none()


def classify_query_taxonomy(question: str) -> dict[str, str]:
    q = (question or "").replace(" ", "").lower()
    visual_terms = ("시각자료", "그림", "이미지", "표", "도표", "비교표", "다이어그램", "구조도", "화살표")
    location_terms = ("어디", "어디서", "위치", "몇슬라이드", "슬라이드", "장면", "씬", "구간", "언제")
    if any(t in q for t in visual_terms):
        if any(t in q for t in location_terms):
            return {"major": "C", "minor": "1", "label": "시각 질의/슬라이드 탐색"}
        return {"major": "C", "minor": "2", "label": "시각 질의/시각 해석"}
    if "강조" in q or "중요" in q or "핵심" in q:
        return {"major": "B", "minor": "3", "label": "탐색 질의/강조"}
    if any(t in q for t in ("요약", "개관", "전체", "흐름", "정리")):
        return {"major": "B", "minor": "2", "label": "탐색 질의/요약/개관"}
    if any(t in q for t in location_terms):
        return {"major": "B", "minor": "1", "label": "탐색 질의/위치"}
    if any(t in q for t in ("차이", "비교", "다른점", "반면", "vs", "versus")):
        return {"major": "A", "minor": "2", "label": "내용 질의/비교"}
    if any(t in q for t in ("관계", "절차", "과정", "순서", "연결", "왜", "이유", "어떻게", "원리")):
        return {"major": "A", "minor": "3", "label": "내용 질의/관계/절차"}
    return {"major": "A", "minor": "1", "label": "내용 질의/개념"}


async def _get_or_create_chat_session(
    db: AsyncSession,
    lecture_id,
    session_id: str,
) -> ChatSession:
    clean_id = (session_id or "").strip() or "default"
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.lecture_id == lecture_id,
            ChatSession.session_id == clean_id,
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        return existing
    row = ChatSession(lecture_id=lecture_id, session_id=clean_id)
    db.add(row)
    await db.flush()
    return row


async def _recent_chat_history(db: AsyncSession, chat_session: ChatSession, limit: int = CHAT_HISTORY_TURNS) -> list[dict[str, str]]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.chat_session_id == chat_session.id)
        .order_by(ChatMessage.turn_index.desc(), ChatMessage.created_at.desc())
        .limit(limit)
    )
    rows = list(reversed(result.scalars().all()))
    history: list[dict[str, str]] = []
    for row in rows:
        if row.question:
            history.append({"role": "user", "content": row.question})
        if row.answer:
            history.append({"role": "assistant", "content": row.answer})
    return history


async def _next_chat_turn_index(db: AsyncSession, chat_session: ChatSession) -> int:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.chat_session_id == chat_session.id)
        .order_by(ChatMessage.turn_index.desc())
        .limit(1)
    )
    last = result.scalar_one_or_none()
    return int(last.turn_index) + 1 if last else 0


async def _has_qna_demo_fallback(db: AsyncSession, chat_session: ChatSession) -> bool:
    result = await db.execute(
        select(ChatMessage.id)
        .where(
            ChatMessage.chat_session_id == chat_session.id,
            ChatMessage.source_mode == "qna_demo_fallback",
        )
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


def _qna_demo_fallback_payload(question: str) -> Optional[Dict[str, Any]]:
    if not QNA_DEMO_FALLBACK_ENABLED:
        return None

    compact = re.sub(r"\s+", "", question or "").lower()
    if not ("운영체제" in compact and "자원" in compact):
        return None

    answer = (
        "운영체제가 관리하는 자원의 주요 종류는 네 가지입니다.\n\n"
        "* 하드웨어 자원: CPU, 캐시나 메모리, 키보드, 마우스, 디스플레이, 하드 디스크, 프린터 등 물리적 장치들을 관리합니다.\n"
        "* 소프트웨어 자원: 응용프로그램과 같은 프로그램들을 관리합니다.\n"
        "* 데이터 자원: 파일, 데이터베이스 등 시스템이 다루는 데이터와 관련된 항목들을 관리합니다.\n"
        "* 프로세스: 실행 중인 프로그램의 단위를 관리합니다."
    )
    retrieved_chunks = [
        {
            "chunk_type": "segment",
            "text": "어쨌든 지금 운영체제가 관리하는 자원들은 이런 것들이 있습니다. 자원에 대한...",
            "start_sec": 557,
            "end_sec": 570,
            "score": 1.0,
            "slide_number": 7,
        },
        {
            "chunk_type": "slide",
            "text": "운영체제가 관리하는 자원: 하드웨어 자원, 소프트웨어 자원, 데이터, 프로세스",
            "start_sec": 557,
            "end_sec": 570,
            "score": 0.96,
            "slide_number": 7,
        },
        {
            "chunk_type": "slide",
            "text": "운영체제 자원 관리 관련 설명",
            "start_sec": 610,
            "end_sec": 625,
            "score": 0.9,
            "slide_number": 9,
        },
    ]
    related_slides = [
        {"slide_number": 7, "start_sec": 557, "score": 1.0, "label": "슬라이드 7"},
        {"slide_number": 9, "start_sec": 610, "score": 0.9, "label": "슬라이드 9"},
    ]
    graph_nodes = [
        {"id": "concept/os", "label": "운영체제", "type": "GraphRAGEntity"},
        {"id": "concept/resource", "label": "자원 관리", "type": "GraphRAGEntity"},
        {"id": "concept/hardware", "label": "하드웨어 자원", "type": "GraphRAGEntity"},
        {"id": "concept/software", "label": "소프트웨어 자원", "type": "GraphRAGEntity"},
        {"id": "concept/data", "label": "데이터 자원", "type": "GraphRAGEntity"},
        {"id": "concept/process", "label": "프로세스", "type": "GraphRAGEntity"},
        {"id": "slide_007", "label": "S7", "type": "Slide", "props": {"slide_number": 7}},
        {"id": "scene_013", "label": "Scene13", "type": "Scene"},
        {"id": "slide_009", "label": "S9", "type": "Slide", "props": {"slide_number": 9}},
        {"id": "scene_018", "label": "Scene18", "type": "Scene"},
        {"id": "segment_557", "label": "09:17", "type": "Segment"},
        {"id": "lecture", "label": "Lecture", "type": "Lecture"},
    ]
    graph_edges = [
        {"from": "lecture", "to": "scene_013", "label": "HAS_SCENE"},
        {"from": "lecture", "to": "scene_018", "label": "HAS_SCENE"},
        {"from": "scene_013", "to": "slide_007", "label": "USES_SLIDE"},
        {"from": "scene_018", "to": "slide_009", "label": "USES_SLIDE"},
        {"from": "scene_013", "to": "segment_557", "label": "HAS_SEGMENT"},
        {"from": "concept/os", "to": "concept/resource", "label": "GRAPHRAG_RELATES_TO"},
        {"from": "concept/resource", "to": "concept/hardware", "label": "GRAPHRAG_RELATES_TO"},
        {"from": "concept/resource", "to": "concept/software", "label": "GRAPHRAG_RELATES_TO"},
        {"from": "concept/resource", "to": "concept/data", "label": "GRAPHRAG_RELATES_TO"},
        {"from": "concept/resource", "to": "concept/process", "label": "GRAPHRAG_RELATES_TO"},
        {"from": "concept/os", "to": "slide_007", "label": "GRAPHRAG_APPEARS_IN"},
        {"from": "concept/resource", "to": "slide_007", "label": "GRAPHRAG_APPEARS_IN"},
        {"from": "concept/hardware", "to": "slide_007", "label": "GRAPHRAG_APPEARS_IN"},
        {"from": "concept/software", "to": "slide_007", "label": "GRAPHRAG_APPEARS_IN"},
        {"from": "concept/data", "to": "slide_009", "label": "GRAPHRAG_APPEARS_IN"},
        {"from": "concept/process", "to": "slide_009", "label": "GRAPHRAG_APPEARS_IN"},
    ]
    graph = {"nodes": graph_nodes, "edges": graph_edges}
    return {
        "answer": answer,
        "timestamps": [{"start": 557, "end": 570, "label": "어쨌든 지금 운영체제가 관리하는 자원들은 이런 것들이 있습니다. 자원에 대한...", "slide_number": 7}],
        "graph": graph,
        "core_graph": graph,
        "retrieved_chunks": retrieved_chunks,
        "related_slides": related_slides,
        "source_mode": "qna_demo_fallback",
    }


# ── GraphSession 헬퍼 ────────────────────────────────────────────────────────
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _session_expired(cutoff: datetime):
    return and_(
        GraphSession.ended_at.is_(None),
        GraphSession.last_heartbeat_at < cutoff,
    )


async def _cleanup_stale_sessions(db: AsyncSession, lecture_id) -> int:
    cutoff = _utcnow() - timedelta(seconds=GRAPH_SESSION_TTL_SEC)
    q = select(GraphSession).where(
        GraphSession.stem == str(lecture_id),
        _session_expired(cutoff),
    )
    res = await db.execute(q)
    rows = res.scalars().all()
    for row in rows:
        row.ended_at = _utcnow()
    if rows:
        await db.commit()
    return len(rows)


async def _active_session_count(db: AsyncSession, lecture_id) -> int:
    await _cleanup_stale_sessions(db, lecture_id)
    q = select(GraphSession).where(
        GraphSession.stem == str(lecture_id),
        GraphSession.ended_at.is_(None),
    )
    res = await db.execute(q)
    return len(res.scalars().all())


async def _touch_or_create_graph_session(
    db: AsyncSession,
    *,
    lecture_id,
    session_id: str,
    now: datetime,
) -> None:
    """
    (lecture_id, session_id) 세션을 upsert처럼 갱신한다.
    - 기존 중복 행이 있으면 최신 1개만 활성 유지하고 나머지는 ended 처리
    """
    q = (
        select(GraphSession)
        .where(
            GraphSession.lecture_id == lecture_id,
            GraphSession.session_id == session_id,
        )
        .order_by(GraphSession.created_at.desc(), GraphSession.id.desc())
    )
    res = await db.execute(q)
    rows = res.scalars().all()

    if not rows:
        db.add(
            GraphSession(
                lecture_id=lecture_id,
                stem=str(lecture_id),
                session_id=session_id,
                last_heartbeat_at=now,
                ended_at=None,
            )
        )
        return

    primary = rows[0]
    primary.last_heartbeat_at = now
    primary.ended_at = None
    for extra in rows[1:]:
        extra.ended_at = now


async def _commit_graph_session_touch(
    db: AsyncSession,
    *,
    lecture_id,
    session_id: str,
    now: datetime,
) -> None:
    try:
        await _touch_or_create_graph_session(
            db,
            lecture_id=lecture_id,
            session_id=session_id,
            now=now,
        )
        await db.commit()
    except IntegrityError:
        await db.rollback()
        await _touch_or_create_graph_session(
            db,
            lecture_id=lecture_id,
            session_id=session_id,
            now=now,
        )
        await db.commit()


# ── GraphSession 관련 ────────────────────────────────────────────────────────
async def graph_enter(db: AsyncSession, lecture_id: str, session_id: str) -> Dict[str, Any]:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")

    lecture_id_uuid = lecture.id
    stem = str(lecture_id_uuid)
    output_dir = str(lecture.output_dir)
    now = _utcnow()

    await _cleanup_stale_sessions(db, lecture_id_uuid)
    await _commit_graph_session_touch(
        db,
        lecture_id=lecture_id_uuid,
        session_id=session_id,
        now=now,
    )

    stem_lock = await get_stem_load_lock(stem)
    loop = asyncio.get_running_loop()
    async with stem_lock:
        load_info = await loop.run_in_executor(None, _ensure_stem_loaded, stem, output_dir)
    active_count = await _active_session_count(db, lecture_id_uuid)
    if load_info.get("loaded_now"):
        logger.info("Neo4j graph loaded on enter stem=%s session_id=%s", stem, session_id)
    else:
        logger.info("Neo4j graph already loaded on enter stem=%s session_id=%s", stem, session_id)
    return {
        "lecture_id": stem,
        "stem": stem,
        "session_id": session_id,
        "active_sessions": active_count,
        "loaded": True,
        **load_info,
    }


async def graph_heartbeat(db: AsyncSession, lecture_id: str, session_id: str) -> Dict[str, Any]:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")

    lecture_id_uuid = lecture.id
    stem = str(lecture_id_uuid)
    now = _utcnow()

    await _commit_graph_session_touch(
        db,
        lecture_id=lecture_id_uuid,
        session_id=session_id,
        now=now,
    )
    return {
        "lecture_id": stem,
        "stem": stem,
        "session_id": session_id,
        "active_sessions": await _active_session_count(db, lecture_id_uuid),
    }


async def graph_leave(db: AsyncSession, lecture_id: str, session_id: str) -> Dict[str, Any]:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")

    lecture_id_uuid = lecture.id
    stem = str(lecture_id_uuid)
    q = select(GraphSession).where(
        GraphSession.lecture_id == lecture_id_uuid,
        GraphSession.session_id == session_id,
        GraphSession.ended_at.is_(None),
    )
    res = await db.execute(q)
    rows = res.scalars().all()
    if rows:
        now = _utcnow()
        for row in rows:
            row.ended_at = now
        await db.commit()

    stem_lock = await get_stem_load_lock(stem)
    loop = asyncio.get_running_loop()
    unloaded_now = False
    async with stem_lock:
        active_count = await _active_session_count(db, lecture_id_uuid)
        if active_count == 0 and await loop.run_in_executor(None, _is_stem_loaded, stem):
            await loop.run_in_executor(None, _unload_stem_from_neo4j, stem)
            unloaded_now = True

        loaded = await loop.run_in_executor(None, _is_stem_loaded, stem)
    return {
        "lecture_id": stem,
        "stem": stem,
        "session_id": session_id,
        "active_sessions": active_count,
        "unloaded_now": unloaded_now,
        "loaded": loaded,
    }


async def graph_status(db: AsyncSession, lecture_id: str) -> Dict[str, Any]:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")

    lecture_id_uuid = lecture.id
    stem = str(lecture_id_uuid)
    loop = asyncio.get_event_loop()
    loaded = await loop.run_in_executor(None, _is_stem_loaded, stem)
    active_count = await _active_session_count(db, lecture_id_uuid)
    return {
        "lecture_id": stem,
        "stem": stem,
        "loaded": loaded,
        "active_sessions": active_count,
        "session_ttl_sec": GRAPH_SESSION_TTL_SEC,
    }


async def ask_question(
    db: AsyncSession,
    lecture_id: str,
    question: str,
    chat_session_id: str = "default",
    current_scene_number: Any = None,
    current_slide_number: Any = None,
) -> Dict[str, Any]:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")

    stem = str(lecture.id)
    chat_session = await _get_or_create_chat_session(db, lecture.id, chat_session_id)
    conversation_history = await _recent_chat_history(db, chat_session)
    query_type = classify_query_taxonomy(question)
    query_url = os.getenv("QUERY_SERVICE_URL", "http://query_service:8001")
    stem_lock = await get_stem_load_lock(stem)
    loop = asyncio.get_running_loop()
    async with stem_lock:
        await loop.run_in_executor(None, _ensure_stem_loaded, stem, lecture.output_dir)

    last_err: tuple | None = None
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=QNA_QUERY_TIMEOUT_SEC) as client:
                resp = await client.post(
                    f"{query_url}/internal/query",
                    json={
                        "stem": stem,
                        "question": question,
                        "current_scene_number": current_scene_number,
                        "current_slide_number": current_slide_number,
                        "conversation_history": conversation_history,
                    },
                )
            if resp.status_code != 200:
                if 500 <= resp.status_code < 600:
                    last_err = ("status", resp.status_code)
                    if attempt == 0:
                        await asyncio.sleep(1)
                        continue
                    break
                raise HTTPException(status_code=resp.status_code, detail="Query service error")
            qr = resp.json()
            turn_index = await _next_chat_turn_index(db, chat_session)
            db.add(ChatMessage(
                lecture_id=lecture.id,
                chat_session_id=chat_session.id,
                turn_index=turn_index,
                question=question,
                answer=qr.get("answer") or "",
                query_major=query_type["major"],
                query_minor=query_type["minor"],
                query_type_label=query_type["label"],
                source_mode=qr.get("source_mode", "default"),
                related_slides=qr.get("related_slides", []),
                retrieved_chunks=qr.get("retrieved_chunks", []),
                core_graph=qr.get("core_graph", {"nodes": [], "edges": []}),
            ))
            await db.commit()
            return {
                "answer": qr.get("answer"),
                "timestamps": qr.get("timestamps", []),
                "graph": qr.get("graph", {"nodes": [], "edges": []}),
                "core_graph": qr.get("core_graph", {"nodes": [], "edges": []}),
                "retrieved_chunks": qr.get("retrieved_chunks", []),
                "related_slides": qr.get("related_slides", []),
                "source_mode": qr.get("source_mode", "default"),
                "chat_session_id": chat_session.session_id,
                "query_type": query_type,
            }
        except HTTPException:
            raise
        except httpx.TimeoutException as exc:
            last_err = ("timeout", exc)
        except httpx.ConnectError as exc:
            last_err = ("connect", exc)
        except httpx.HTTPError as exc:
            last_err = ("http", exc)
        if attempt == 0:
            await asyncio.sleep(1)

    fallback = None if await _has_qna_demo_fallback(db, chat_session) else _qna_demo_fallback_payload(question)
    if fallback:
        logger.warning("QnA demo fallback used stem=%s reason=%s", stem, last_err[0] if last_err else "unknown")
        turn_index = await _next_chat_turn_index(db, chat_session)
        db.add(ChatMessage(
            lecture_id=lecture.id,
            chat_session_id=chat_session.id,
            turn_index=turn_index,
            question=question,
            answer=fallback["answer"],
            query_major=query_type["major"],
            query_minor=query_type["minor"],
            query_type_label=query_type["label"],
            source_mode=fallback["source_mode"],
            related_slides=fallback["related_slides"],
            retrieved_chunks=fallback["retrieved_chunks"],
            core_graph=fallback["core_graph"],
        ))
        await db.commit()
        return {
            **fallback,
            "chat_session_id": chat_session.session_id,
            "query_type": query_type,
        }

    if last_err and last_err[0] == "timeout":
        raise HTTPException(
            status_code=504,
            detail="QnA 서비스 응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.",
        )
    raise HTTPException(
        status_code=503,
        detail="QnA 서비스에 연결할 수 없습니다. 서버 상태를 확인해 주세요.",
    )


async def get_timeline(db: AsyncSession, lecture_id: str) -> List[Dict[str, Any]]:
    """타임라인 조회 (Lecture ID 기준)"""
    detail = await get_lecture_detail(db, lecture_id)
    if not detail or not detail.get("output_dir"):
        raise HTTPException(status_code=404, detail="Lecture result not found")

    output_dir = Path(detail["output_dir"])
    json_files = list(output_dir.glob("*_slide_classified.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Timeline file not found")

    try:
        base_image_urls = _load_scene_base_image_urls(output_dir)
        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)

        scenes = []
        for s in data.get("scenes", []):
            scene_number = s.get("scene_number", s.get("scene_index"))
            img_url = base_image_urls.get(_scene_number_key(scene_number)) or make_file_url(s.get("image_path"))
            timestamp_sec = s.get("timestamp")
            try:
                timestamp_sec = float(timestamp_sec)
            except (TypeError, ValueError):
                timestamp_sec = 0.0
            ts = s.get("timestamp_formatted", "00:00")
            if ts.startswith("00:"): ts = ts[3:]

            scenes.append({
                "timestamp":    ts,
                "timestamp_sec": timestamp_sec,
                "type":         "emphasis" if s.get("role") == "elaborated" else "slide",
                "text":         s.get("title") or f"Slide {s.get('slide_number')}",
                "image_url":    img_url,
                "scene_number": scene_number,
                "slide_number": s.get("slide_number"),
            })
        return scenes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading timeline: {e}")


def _scene_number_key(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_scene_base_image_urls(output_dir: Path) -> dict[int, str]:
    metadata_path = output_dir / "slides" / "metadata.json"
    if not metadata_path.is_file():
        return {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read slide metadata for timeline thumbnails: %s", metadata_path, exc_info=True)
        return {}
    if not isinstance(metadata, list):
        return {}

    out: dict[int, str] = {}
    slides_dir = metadata_path.parent
    for entry in metadata:
        if not isinstance(entry, dict) or entry.get("capture_type") != "base":
            continue
        scene_number = _scene_number_key(entry.get("scene_index", entry.get("scene_number")))
        filename = _str_cell(entry.get("filename"))
        if scene_number is None or not filename:
            continue
        image_url = make_file_url(str(slides_dir / filename))
        if image_url:
            out[scene_number] = image_url
    return out


async def get_knowledge_graph(db: AsyncSession, lecture_id: str) -> Dict[str, Any]:
    """지식 그래프 조회 (Lecture ID 기준)"""
    detail = await get_lecture_detail(db, lecture_id)
    if not detail or not detail.get("output_dir"):
        raise HTTPException(status_code=404, detail="Lecture result not found")

    output_dir = Path(detail["output_dir"])
    nodes_paths = list(output_dir.glob("*_nodes.parquet"))
    edges_paths = list(output_dir.glob("*_edges.parquet"))

    if not nodes_paths:
        raise HTTPException(status_code=404, detail="Graph files not found")

    try:
        nodes_out, edges_out = _read_structure_graph(nodes_paths[0], edges_paths[0] if edges_paths else None)
        _append_graphrag_graph(output_dir, nodes_out, edges_out)

        return {
            "node_count": len(nodes_out),
            "edge_count": len(edges_out),
            "graph": {"nodes": nodes_out, "edges": edges_out},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading graph: {e}")


_GRAPHRAG_SLIDE_ID_RE = re.compile(r"slide_\d{3,}")
_GRAPHRAG_STRUCTURAL_ENTITY_RE = re.compile(
    r"^(slide[_ ]?\d+|chapter\s*\d+|segment[/_ ]?\d+|context[/_ ]?\d+|scene[/_ ]?\d+|graphlec_seg:segment/\d+)$",
    re.IGNORECASE,
)


def _read_structure_graph(nodes_path: Path, edges_path: Optional[Path]) -> tuple[list[dict], list[dict]]:
    ndf = pd.read_parquet(nodes_path)
    edf = (
        pd.read_parquet(edges_path)
        if edges_path and edges_path.is_file()
        else pd.DataFrame(columns=["src_id", "rel_type", "tgt_id", "properties_json"])
    )

    nodes_out = []
    for _, row in ndf.iterrows():
        nid = _str_cell(row.get("node_id"))
        if not nid:
            continue
        label = _str_cell(row.get("label")) or nid
        props = _str_cell(row.get("properties_json")) or "{}"
        try:
            props_dict = json.loads(props)
            ntype = props_dict.get("type") or label or "node"
        except Exception:
            props_dict = {}
            ntype = label or "node"
        if (
            label in {"Domain", "Concept"}
            or ntype in {"Domain", "Concept"}
            or nid == "lecture_video"
            or nid.startswith("domain/")
            or nid.startswith("concept/")
        ):
            continue
        nodes_out.append({
            "id": nid, "label": label[:120], "title": props[:800],
            "name": _str_cell(props_dict.get("name")) or _str_cell(props_dict.get("title")),
            "text": _str_cell(props_dict.get("text")) or _str_cell(props_dict.get("target_content")),
            "asset_type": _str_cell(props_dict.get("asset_type")),
            "description": _str_cell(props_dict.get("description")),
            "type": ntype, "color": _hex_color(label),
        })

    seen_ids = {n["id"] for n in nodes_out}
    edges_out = []
    for _, row in edf.iterrows():
        src, tgt = _str_cell(row.get("src_id")), _str_cell(row.get("tgt_id"))
        if not src or not tgt:
            continue
        rel_type = _str_cell(row.get("rel_type")) or "related"
        if (
            rel_type == "HAS_DOMAIN"
            or src == "lecture_video"
            or tgt == "lecture_video"
            or src.startswith("domain/")
            or tgt.startswith("domain/")
            or src.startswith("concept/")
            or tgt.startswith("concept/")
        ):
            continue
        edges_out.append({"from": src, "to": tgt, "label": rel_type})
        for x in (src, tgt):
            if x not in seen_ids:
                seen_ids.add(x)
                nodes_out.append({"id": x, "label": x[:80], "type": "orphan", "color": "#9ca3af"})

    return nodes_out, edges_out


def _append_graphrag_graph(output_dir: Path, nodes_out: list[dict], edges_out: list[dict]) -> None:
    graphrag_dir = _find_graphrag_output_dir(output_dir)
    if not graphrag_dir:
        return

    entities_path = graphrag_dir / "entities.parquet"
    relationships_path = graphrag_dir / "relationships.parquet"
    entities = pd.read_parquet(entities_path)
    relationships = pd.read_parquet(relationships_path)

    seen_ids = {str(n.get("id")) for n in nodes_out if n.get("id")}
    edge_keys = {
        (str(e.get("from")), str(e.get("label")), str(e.get("to")))
        for e in edges_out
        if e.get("from") and e.get("to")
    }
    title_to_id: dict[str, str] = {}
    raw_entity_id_to_node_id: dict[str, str] = {}

    for _, row in entities.iterrows():
        raw_id = _str_cell(row.get("id")).strip()
        title = _str_cell(row.get("title")).strip()
        if not raw_id or not title or _GRAPHRAG_STRUCTURAL_ENTITY_RE.match(title):
            continue
        node_id = f"graphrag/entity/{raw_id}"
        raw_entity_id_to_node_id[raw_id] = node_id
        for key in (_graphrag_title_key(title), _graphrag_alias_key(title)):
            if key and key not in title_to_id:
                title_to_id[key] = node_id
        props = {
            "title": title,
            "name": title,
            "graphrag_id": raw_id,
            "human_readable_id": _str_cell(row.get("human_readable_id")),
            "entity_type": _str_cell(row.get("type")),
            "description": _str_cell(row.get("description")),
            "frequency": _int_val(row.get("frequency")),
            "degree": _int_val(row.get("degree")),
            "source": "microsoft_graphrag",
        }
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        nodes_out.append({
            "id": node_id,
            "label": title[:120],
            "title": json.dumps(props, ensure_ascii=False)[:1200],
            "name": title,
            "text": _str_cell(row.get("description")),
            "type": "GraphRAGEntity",
            "color": _hex_color("GraphRAGEntity"),
        })

    for _, row in relationships.iterrows():
        src = title_to_id.get(_graphrag_title_key(row.get("source"))) or title_to_id.get(_graphrag_alias_key(row.get("source")))
        tgt = title_to_id.get(_graphrag_title_key(row.get("target"))) or title_to_id.get(_graphrag_alias_key(row.get("target")))
        if not src or not tgt or src == tgt:
            continue
        _append_graph_edge(
            edges_out,
            edge_keys,
            src,
            tgt,
            "GRAPHRAG_RELATES_TO",
            {
                "graphrag_id": _str_cell(row.get("id")),
                "description": _str_cell(row.get("description")),
                "weight": _float_val(row.get("weight")),
                "combined_degree": _int_val(row.get("combined_degree")),
            },
        )

    _append_graphrag_slide_edges(graphrag_dir, nodes_out, edges_out, edge_keys, raw_entity_id_to_node_id)
    _append_graphrag_communities(graphrag_dir, nodes_out, edges_out, edge_keys, raw_entity_id_to_node_id)


def _find_graphrag_output_dir(output_dir: Path) -> Optional[Path]:
    for path in (
        output_dir / "graphrag" / "output",
        output_dir / "graphrag_output",
        output_dir / "output",
    ):
        if (path / "entities.parquet").is_file() and (path / "relationships.parquet").is_file():
            return path
    return None


def _append_graphrag_slide_edges(
    graphrag_dir: Path,
    nodes_out: list[dict],
    edges_out: list[dict],
    edge_keys: set[tuple[str, str, str]],
    raw_entity_id_to_node_id: dict[str, str],
) -> None:
    text_units_path = graphrag_dir / "text_units.parquet"
    entities_path = graphrag_dir / "entities.parquet"
    if not text_units_path.is_file():
        return

    text_units = pd.read_parquet(text_units_path)
    text_unit_slide_ids = {
        _str_cell(row.get("id")): sorted(set(_GRAPHRAG_SLIDE_ID_RE.findall(_str_cell(row.get("text")))))
        for _, row in text_units.iterrows()
    }
    if not text_unit_slide_ids:
        return

    node_ids = {str(n.get("id")) for n in nodes_out if n.get("id")}
    entities = pd.read_parquet(entities_path)
    for _, row in entities.iterrows():
        entity_id = raw_entity_id_to_node_id.get(_str_cell(row.get("id")))
        if not entity_id:
            continue
        slide_ids: set[str] = set()
        for tu_id in _list_val(row.get("text_unit_ids")):
            slide_ids.update(text_unit_slide_ids.get(tu_id, []))
        for slide_id in sorted(slide_ids):
            if slide_id in node_ids:
                _append_graph_edge(edges_out, edge_keys, entity_id, slide_id, "GRAPHRAG_APPEARS_IN")


def _append_graphrag_communities(
    graphrag_dir: Path,
    nodes_out: list[dict],
    edges_out: list[dict],
    edge_keys: set[tuple[str, str, str]],
    raw_entity_id_to_node_id: dict[str, str],
) -> None:
    communities_path = graphrag_dir / "communities.parquet"
    reports_path = graphrag_dir / "community_reports.parquet"
    if not communities_path.is_file():
        return

    communities = pd.read_parquet(communities_path)
    reports_by_community: dict[str, Any] = {}
    if reports_path.is_file():
        reports = pd.read_parquet(reports_path)
        reports_by_community = {
            _str_cell(row.get("community")): row
            for _, row in reports.iterrows()
            if _str_cell(row.get("community"))
        }

    seen_ids = {str(n.get("id")) for n in nodes_out if n.get("id")}
    for _, row in communities.iterrows():
        community = _str_cell(row.get("community")).strip()
        if not community:
            continue
        report = reports_by_community.get(community)
        node_id = f"graphrag/community/{community}"
        title = _str_cell((report if report is not None else row).get("title")) or f"Community {community}"
        summary = _str_cell(report.get("summary")) if report is not None else ""
        if node_id not in seen_ids:
            seen_ids.add(node_id)
            props = {
                "title": title,
                "summary": summary,
                "community": community,
                "level": _int_val(row.get("level")),
                "size": _int_val(row.get("size")),
                "source": "microsoft_graphrag",
            }
            nodes_out.append({
                "id": node_id,
                "label": title[:120],
                "title": json.dumps(props, ensure_ascii=False)[:1200],
                "name": title,
                "text": summary,
                "type": "GraphRAGCommunity",
                "color": _hex_color("GraphRAGCommunity"),
            })
        for raw_entity_id in _list_val(row.get("entity_ids")):
            entity_id = raw_entity_id_to_node_id.get(raw_entity_id)
            if entity_id:
                _append_graph_edge(edges_out, edge_keys, node_id, entity_id, "GRAPHRAG_HAS_ENTITY")


def _append_graph_edge(
    edges_out: list[dict],
    edge_keys: set[tuple[str, str, str]],
    src: str,
    tgt: str,
    label: str,
    props: Optional[dict[str, Any]] = None,
) -> None:
    key = (src, label, tgt)
    if key in edge_keys:
        return
    edge_keys.add(key)
    edge = {"from": src, "to": tgt, "label": label}
    if props:
        edge["title"] = json.dumps(props, ensure_ascii=False)[:1000]
    edges_out.append(edge)


def _graphrag_title_key(value: Any) -> str:
    return re.sub(r"\s+", "", _str_cell(value).lower())


def _graphrag_alias_key(value: Any) -> str:
    return re.sub(r"[\W_]+", "", _str_cell(value).lower())


def _list_val(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw == "[]":
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x)]
        except json.JSONDecodeError:
            return [x for x in re.findall(r"[0-9A-Za-z][0-9A-Za-z_-]{7,}", raw) if x]
        return [raw]
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if str(x)]
    return [str(value)]


def _int_val(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _filter_served_slide_errors(items: list[dict]) -> list[dict]:
    filtered = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        problematic = str(item.get("problematic_text", "") or "").strip()
        corrected = str(item.get("corrected_text", "") or "").strip()
        if not problematic or not corrected or problematic == corrected:
            continue
        filtered.append(item)
    return filtered


def _slide_number_key(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_count(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_slide_image_url_map(output_dir: Path) -> dict[str, dict[Any, str]]:
    classified_paths = list(output_dir.glob("*_slide_classified.json"))
    if not classified_paths:
        return {"by_number": {}, "by_title": {}}

    image_urls: dict[str, dict[Any, str]] = {"by_number": {}, "by_title": {}}
    for classified_path in classified_paths:
        try:
            with open(classified_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            logger.warning("Failed to read slide image metadata: %s", classified_path, exc_info=True)
            continue

        slides = []
        if isinstance(data.get("slides"), list):
            slides.extend(data.get("slides") or [])
        if isinstance(data.get("scenes"), list):
            slides.extend(data.get("scenes") or [])

        for slide in slides:
            if not isinstance(slide, dict):
                continue

            image_url = make_file_url(slide.get("image_path"))
            if not image_url:
                continue

            slide_number = _slide_number_key(slide.get("slide_number"))
            if slide_number is not None:
                image_urls["by_number"][slide_number] = image_url

            title = str(slide.get("title") or "").strip()
            if title:
                image_urls["by_title"][title] = image_url

    return image_urls


def _attach_slide_image_urls(items: list[dict], image_urls: dict[str, dict[Any, str]]) -> list[dict]:
    by_number = image_urls.get("by_number", {}) or {}
    by_title = image_urls.get("by_title", {}) or {}
    enriched = []
    for item in items or []:
        if not isinstance(item, dict):
            continue

        copied = dict(item)
        slide_number = _slide_number_key(copied.get("slide_number"))
        slide_title = str(copied.get("slide_title") or "").strip()
        image_url = copied.get("slide_image_url") or copied.get("image_url")
        if not image_url:
            image_url = make_file_url(copied.get("slide_image_path"))
        if not image_url:
            image_url = by_title.get(slide_title)
            if not image_url and slide_number is not None:
                image_url = by_number.get(slide_number)
        if image_url:
            copied.setdefault("slide_image_url", image_url)
            copied.setdefault("image_url", image_url)
        enriched.append(copied)

    return enriched


async def get_content_verification(db: AsyncSession, lecture_id: str) -> Dict[str, Any]:
    """verifier 결과 조회 (Lecture ID 기준)."""
    detail = await get_lecture_detail(db, lecture_id)
    if not detail or not detail.get("output_dir") or not detail.get("stem"):
        raise HTTPException(status_code=404, detail="Lecture result not found")

    output_dir = Path(detail["output_dir"])
    stem = str(detail["stem"])
    analyzer_dir = output_dir / f"{stem}_analyzer"

    candidate_paths = [
        analyzer_dir / f"{stem}_content_verification.json",
        output_dir / f"{stem}_content_verification.json",
        analyzer_dir / f"{stem}_verification_final.json",
        output_dir / f"{stem}_verification_final.json",
    ]
    verifier_path = next((path for path in candidate_paths if path.exists()), None)
    if not verifier_path:
        raise HTTPException(status_code=404, detail="Content verification file not found")

    try:
        with open(verifier_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading content verification: {e}")

    flow = data.get("claim_decision_flow", {}) or {}
    summary = data.get("claim_decision_flow_summary", {}) or {}
    content_summary = data.get("summary", {}) or summary
    feedback_items = data.get("feedback_items", []) or []
    confirmed_feedback_items = [
        item for item in feedback_items
        if isinstance(item, dict) and item.get("status") == "confirmed"
    ]
    professor_check_feedback_items = [
        item for item in feedback_items
        if isinstance(item, dict) and item.get("status") in {"professor_check", "review_needed"}
    ]
    rejected_feedback_items = [
        item for item in feedback_items
        if isinstance(item, dict) and item.get("status") == "rejected"
    ]
    final_claims = flow.get("final_confirmed_claims", []) or data.get("final_confirmed_claims", []) or []
    needs_review_claims = flow.get("needs_review_claims", []) or data.get("needs_review_claims", []) or []
    verifier_rejected_claims = flow.get("verifier_rejected_claims", []) or data.get("verifier_rejected_claims", []) or []
    slide_image_urls = _load_slide_image_url_map(output_dir)
    slide_errors = _attach_slide_image_urls(
        _filter_served_slide_errors(data.get("slide_errors", []) or []),
        slide_image_urls,
    )
    slide_error_needs_review = _attach_slide_image_urls(
        _filter_served_slide_errors(data.get("slide_error_needs_review", []) or []),
        slide_image_urls,
    )

    return {
        "lecture_id": str(detail["id"]),
        "stem": stem,
        "verification_path": str(verifier_path),
        "schema_version": data.get("schema_version"),
        "mode": data.get("mode", ""),
        "verification_date": data.get("verification_date", ""),
        "models": data.get("models", {}) or [],
        "pipeline_models": data.get("pipeline_models", {}) or {},
        "primary_model": data.get("primary_model", ""),
        "verifier_source_models": data.get("verifier_source_models", []) or [],
        "verifier_model_weights": data.get("verifier_model_weights", {}) or {},
        "severity_score_report": data.get("severity_score_report", {}) or {},
        "summary": content_summary,
        "overview": data.get("claim_decision_overview", []) or [],
        "counts": {
            "final_confirmed": _safe_count(
                content_summary.get(
                    "confirmed_feedback_count",
                    summary.get("final_confirmed_claim_count", len(confirmed_feedback_items) or len(final_claims)),
                )
            ),
            "needs_review": _safe_count(
                content_summary.get(
                    "review_needed_feedback_count",
                    summary.get("needs_review_claim_count", len(professor_check_feedback_items) or len(needs_review_claims)),
                )
            ),
            "rejected": _safe_count(
                content_summary.get("rejected_feedback_count", len(rejected_feedback_items))
            ),
            "slide_errors": _safe_count(content_summary.get("slide_error_count", len(slide_errors))),
            "slide_error_needs_review": len(slide_error_needs_review),
            "verifier_rejected": _safe_count(summary.get("verifier_rejected_claim_count", len(verifier_rejected_claims))),
        },
        "final_confirmed_claim_count": _safe_count(
            content_summary.get(
                "confirmed_feedback_count",
                summary.get("final_confirmed_claim_count", len(confirmed_feedback_items) or len(final_claims)),
            )
        ),
        "claims": data.get("claims", []) or [],
        "feedback_groups": data.get("feedback_groups", []) or [],
        "feedback_items": feedback_items,
        "views": data.get("views", {}) or {},
        "final_confirmed_claims": final_claims,
        "needs_review_claims": needs_review_claims,
        "verifier_rejected_claims": verifier_rejected_claims,
        "issues": data.get("issues", []) or [],
        "slide_errors": slide_errors,
        "slide_error_needs_review": slide_error_needs_review,
        "slide_error_consensus": data.get("slide_error_consensus", {}) or {},
        "slide_error_status": data.get("slide_error_status", ""),
        "slide_error_summary": data.get("slide_error_summary", {}) or {},
        "slide_error_path": data.get("slide_error_path", ""),
        "claim_decision_flow_summary": summary,
        "classified_issue_artifacts": data.get("classified_issue_artifacts", {}) or {},
        "classified_issue_verifier_path": data.get("classified_issue_verifier_path", ""),
        "classified_issue_verifier": (data.get("views", {}) or {}).get("classified_issue_verifier", {}),
    }


# ── 기타 유틸 ───────────────────────────────────────────────────────────────
async def get_graph_info(db: AsyncSession, lecture_id: str) -> Dict[str, Any]:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        return {"error": "Not Found"}
    stem = str(lecture.id)
    node_count = 0
    try:
        with neo4j_session() as session:
            record = session.run(
                "MATCH (n {stem: $stem}) RETURN count(n) AS count", stem=stem
            ).single()
            if record:
                node_count = record["count"]
    except Exception as e:
        logger.warning("Failed to get graph info for %s: %s", stem, e)

    return {"lecture_id": lecture_id, "stem": stem, "graph_exists": node_count > 0, "node_count": node_count}


async def retry_graph_only(db: AsyncSession, lecture_id: str) -> dict[str, Any] | None:
    """Retry the publish/upload portion. The route name remains for compatibility."""
    try:
        ident_uuid = uuid.UUID(str(lecture_id))
    except (ValueError, TypeError):
        return None

    lecture = await _get_lecture(db, str(ident_uuid))
    if not lecture:
        return None

    active_graph_result = await db.execute(
        select(ProcessingJob)
        .where(
            ProcessingJob.lecture_id == ident_uuid,
            ProcessingJob.job_type.in_(PUBLICATION_DB_JOB_TYPES),
            ProcessingJob.status.in_([JOB_STATUS_PENDING, JOB_STATUS_RUNNING]),
        )
        .order_by(ProcessingJob.created_at.desc())
        .limit(1)
    )
    active_graph_job = active_graph_result.scalar_one_or_none()
    if active_graph_job:
        return {
            "status": "success",
            "job_id": str(active_graph_job.id),
            "job_type": active_graph_job.job_type,
            "already_queued": True,
            "retried_existing": False,
        }

    failed_graph_result = await db.execute(
        select(ProcessingJob)
        .where(
            ProcessingJob.lecture_id == ident_uuid,
            ProcessingJob.job_type.in_(PUBLICATION_DB_JOB_TYPES),
            ProcessingJob.status == JOB_STATUS_ERROR,
        )
        .order_by(ProcessingJob.created_at.desc())
        .limit(1)
        .with_for_update()
    )
    failed_graph_job = failed_graph_result.scalar_one_or_none()
    if failed_graph_job:
        failed_graph_job.status = JOB_STATUS_PENDING
        failed_graph_job.current_stage = "업로드 파이프라인을 다시 시작합니다."
        failed_graph_job.error_message = None
        failed_graph_job.pipeline_stages = []
        await db.commit()
        await db.refresh(failed_graph_job)
        return {
            "status": "success",
            "job_id": str(failed_graph_job.id),
            "job_type": failed_graph_job.job_type,
            "already_queued": False,
            "retried_existing": True,
        }

    approved_result = await db.execute(
        select(ProcessingJob)
        .where(
            ProcessingJob.lecture_id == ident_uuid,
            ProcessingJob.job_type.in_([JOB_TYPE_VERIFY, JOB_TYPE_VERIFIED_UPLOAD]),
            ProcessingJob.status == JOB_STATUS_DONE,
        )
        .order_by(ProcessingJob.created_at.desc())
        .limit(1)
    )
    approved_job = approved_result.scalar_one_or_none()
    if not approved_job:
        raise HTTPException(status_code=409, detail="No verified lecture is ready for upload retry")

    manifest_path = Path(lecture.output_dir) / f"{ident_uuid}_preprocess_result.json"
    if not manifest_path.exists() or manifest_path.stat().st_size <= 0:
        raise HTTPException(status_code=409, detail="Preprocess manifest is missing for graph retry")

    graph_job = ProcessingJob(
        id=uuid.uuid4(),
        lecture_id=ident_uuid,
        job_type=JOB_TYPE_PUBLISH,
        status=JOB_STATUS_PENDING,
        current_stage="업로드 파이프라인 재시도를 대기 중입니다.",
        error_message=None,
        pipeline_stages=[],
    )
    db.add(graph_job)
    await db.commit()
    await db.refresh(graph_job)
    return {
        "status": "success",
        "job_id": str(graph_job.id),
        "job_type": graph_job.job_type,
        "already_queued": False,
        "retried_existing": False,
    }
