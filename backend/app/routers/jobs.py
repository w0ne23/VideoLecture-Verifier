import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import json

from app.db import AsyncSessionLocal, get_db
from app.models import (
    JOB_STATUS_DONE,
    JOB_STATUS_ERROR,
    JOB_STATUS_REJECTED,
    JOB_STATUS_WAITING_APPROVAL,
    JOB_TYPE_LEGACY_FULL,
    JOB_TYPE_PUBLISH,
    JOB_TYPE_VERIFY,
    Lecture,
    ProcessingJob,
)
from app.services import lecture_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs")


def _normalize_upload_job_type(value: str) -> str:
    token = (value or JOB_TYPE_LEGACY_FULL).strip().lower().replace("-", "_")
    aliases = {
        "legacy": JOB_TYPE_LEGACY_FULL,
        "legacy_full": JOB_TYPE_LEGACY_FULL,
        "publish": JOB_TYPE_PUBLISH,
        "publication": JOB_TYPE_PUBLISH,
        "direct": JOB_TYPE_PUBLISH,
        "upload": JOB_TYPE_PUBLISH,
        "direct_upload": JOB_TYPE_PUBLISH,
        "graph": JOB_TYPE_PUBLISH,
        "graph_upload": JOB_TYPE_PUBLISH,
        "verified": JOB_TYPE_VERIFY,
        "verify": JOB_TYPE_VERIFY,
        "verified_upload": JOB_TYPE_VERIFY,
    }
    if token not in aliases:
        raise HTTPException(status_code=400, detail="Invalid workflow_mode")
    return aliases[token]


@router.get("")
async def list_jobs(
    db: AsyncSession = Depends(get_db),
    status: Optional[str] = Query(None),
):
    return await lecture_service.list_jobs(db, status_filter=status)


@router.get("/{lecture_id}/stream")
async def stream_job_status(
    lecture_id: str,
    request: Request,
    job_id: Optional[str] = Query(None),
    mode: Optional[str] = Query(None),
):
    """
    job_id 쿼리파라미터가 있으면 해당 job을 고정 추적.
    mode 쿼리파라미터가 있으면 lecture_id 기준 해당 mode의 최신 job을 추적.
    둘 다 없으면 lecture_id 기준 최신 job을 추적 (기존 동작).
    retry 후 새 job_id로 재연결하면 정확한 시도별 추적이 가능.
    """
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                async with AsyncSessionLocal() as db:
                    if job_id:
                        job = await lecture_service.get_job(db, job_id)
                    elif mode:
                        job = await lecture_service.get_latest_job_by_mode(db, lecture_id, mode)
                    else:
                        job = await lecture_service.get_latest_job(db, lecture_id)
                if not job:
                    yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                    break
                payload = {
                    "job_id": str(job.id),
                    "job_type": getattr(job, "job_type", None) or JOB_TYPE_LEGACY_FULL,
                    "lecture_status": job.status,
                    "current_stage": job.current_stage,
                    "error_message": job.error_message,
                    "pipeline_stages": job.pipeline_stages or [],
                }
                yield f"data: {json.dumps(payload)}\n\n"
                if job.status in {
                    JOB_STATUS_DONE,
                    JOB_STATUS_ERROR,
                    JOB_STATUS_WAITING_APPROVAL,
                    JOB_STATUS_REJECTED,
                }:
                    break
            except Exception as e:
                logger.error(f"SSE error for lecture {lecture_id}: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")




@router.post("")
async def create_job(
    video: UploadFile = File(...),
    title: str = Form(...),
    category: str = Form("etc"),
    description: str = Form(""),
    workflow_mode: str = Form(JOB_TYPE_LEGACY_FULL),
    db: AsyncSession = Depends(get_db),
):
    job_type = _normalize_upload_job_type(workflow_mode)
    normalized_category = lecture_service.normalize_domain_value(category)
    lecture_id = uuid.uuid4()
    base_dir = Path(lecture_service.LOCAL_STORAGE_DIR)

    input_dir = base_dir / "inputs" / str(lecture_id)
    output_dir = base_dir / "results" / str(lecture_id)

    original_stem = Path(video.filename).stem
    extension = Path(video.filename).suffix
    safe_filename = f"{lecture_id}{extension}"

    final_title = title.strip() if title and title.strip() else original_stem

    try:
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / safe_filename
        
        # 최초 시도 시 결과 디렉터리 초기화 (새로운 UUID이므로 보통 비어있음)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(input_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
    except Exception as e:
        shutil.rmtree(input_dir, ignore_errors=True)
        logger.error(f"File save failed for lecture {lecture_id}: {e}")
        raise HTTPException(status_code=500, detail="업로드 파일을 저장하지 못했습니다.")

    try:
        new_lecture = Lecture(
            id=lecture_id,
            title=final_title,
            category=normalized_category,
            description=description,
            video_path=str(input_path),
            output_dir=str(output_dir),
        )
        db.add(new_lecture)

        job_id = uuid.uuid4()
        new_job = ProcessingJob(
            id=job_id,
            lecture_id=lecture_id,
            job_type=job_type,
            status="pending",
        )
        db.add(new_job)
        await db.commit()
        await db.refresh(new_lecture)
        await db.refresh(new_job)

    except Exception as e:
        await db.rollback()
        shutil.rmtree(input_dir, ignore_errors=True)
        logger.error(f"DB commit failed for lecture {lecture_id}: {e}")
        raise HTTPException(status_code=500, detail="업로드 작업을 생성하지 못했습니다.")

    return {
        "id": str(lecture_id),
        "title": final_title,
        "category": normalized_category,
        "description": description,
        "job_id": str(new_job.id),
        "job_type": new_job.job_type,
        "status": "pending",
        "is_verified": False,
        "is_published": False,
        "created_at": new_lecture.created_at.isoformat() if new_lecture.created_at else None,
    }


@router.delete("/{lecture_id}")
async def delete_lecture(lecture_id: str, db: AsyncSession = Depends(get_db)):
    success = await lecture_service.delete_lecture(db, lecture_id)
    if not success:
        raise HTTPException(status_code=404, detail="Lecture not found")
    return {"status": "success"}


@router.post("/{lecture_id}/retry")
async def retry_lecture(
    lecture_id: str,
    mode: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    result = await lecture_service.retry_lecture(db, lecture_id, mode=mode)
    if not result:
        raise HTTPException(status_code=404, detail="Lecture not found")
    return result


@router.post("/{lecture_id}/approve")
async def confirm_verified_lecture_compat(lecture_id: str, db: AsyncSession = Depends(get_db)):
    result = await lecture_service.confirm_verified_lecture(db, lecture_id)
    if not result:
        raise HTTPException(status_code=404, detail="Lecture not found")
    return result


@router.post("/{lecture_id}/verify/confirm")
async def confirm_verified_lecture(lecture_id: str, db: AsyncSession = Depends(get_db)):
    result = await lecture_service.confirm_verified_lecture(db, lecture_id)
    if not result:
        raise HTTPException(status_code=404, detail="Lecture not found")
    return result


@router.post("/{lecture_id}/retry_graph")
async def retry_graph_ingestion(lecture_id: str, db: AsyncSession = Depends(get_db)):
    result = await lecture_service.retry_graph_only(db, lecture_id)
    if not result:
        raise HTTPException(status_code=404, detail="Lecture not found")
    return result
