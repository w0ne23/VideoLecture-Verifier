import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import AsyncSessionLocal, get_db
from app.models import (
    JOB_STATUS_DONE,
    JOB_STATUS_ERROR,
    JOB_STATUS_REJECTED,
    JOB_STATUS_WAITING_APPROVAL,
    JOB_TYPE_VERIFY,
    Lecture,
    ProcessingJob,
    normalize_job_type,
)
from app.services import lecture_service
from app.services.storage_service import save_upload

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/jobs')


@router.get('')
async def list_jobs(db: AsyncSession = Depends(get_db), status: Optional[str] = Query(None)):
    return await lecture_service.list_jobs(db, status_filter=status)


@router.post('')
async def create_job(
    video: UploadFile = File(...),
    title: str = Form(''),
    description: str = Form(''),
    workflow_mode: str = Form(JOB_TYPE_VERIFY),
    db: AsyncSession = Depends(get_db),
):
    job_type = normalize_job_type(workflow_mode, JOB_TYPE_VERIFY)
    if job_type != JOB_TYPE_VERIFY:
        raise HTTPException(status_code=400, detail='Only verify workflow is supported for uploaded videos')

    lecture_id = uuid.uuid4()
    original_stem = Path(video.filename or '').stem
    final_title = title.strip() if title and title.strip() else original_stem or str(lecture_id)

    try:
        input_path, output_dir = save_upload(video, lecture_id)
    except Exception as exc:
        logger.error('File save failed for lecture %s: %s', lecture_id, exc)
        raise HTTPException(status_code=500, detail='업로드 파일을 저장하지 못했습니다.') from exc

    lecture = Lecture(
        id=lecture_id,
        title=final_title,
        description=description,
        video_path=str(input_path),
        output_dir=str(output_dir),
    )
    job = ProcessingJob(id=uuid.uuid4(), lecture_id=lecture_id, job_type=job_type, status='pending')
    db.add(lecture)
    db.add(job)
    await db.commit()
    await db.refresh(lecture)
    await db.refresh(job)

    return {
        'id': str(lecture.id),
        'title': lecture.title,
        'description': lecture.description or '',
        'job_id': str(job.id),
        'job_type': job.job_type,
        'status': job.status,
        'is_verified': False,
        'created_at': lecture.created_at.isoformat() if lecture.created_at else None,
    }


@router.get('/{lecture_id}/stream')
async def stream_job_status(
    lecture_id: str,
    request: Request,
    job_id: Optional[str] = Query(None),
    mode: Optional[str] = Query(None),
):
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
                    'job_id': str(job.id),
                    'job_type': job.job_type,
                    'lecture_status': job.status,
                    'current_stage': job.current_stage,
                    'error_message': job.error_message,
                    'pipeline_stages': job.pipeline_stages or [],
                }
                yield f'data: {json.dumps(payload)}\n\n'
                if job.status in {JOB_STATUS_DONE, JOB_STATUS_ERROR, JOB_STATUS_WAITING_APPROVAL, JOB_STATUS_REJECTED}:
                    break
            except Exception as exc:
                logger.error('SSE error for lecture %s: %s', lecture_id, exc)
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type='text/event-stream')


@router.post('/{lecture_id}/retry')
async def retry_lecture(lecture_id: str, mode: Optional[str] = Query(None), db: AsyncSession = Depends(get_db)):
    result = await lecture_service.retry_lecture(db, lecture_id, mode=mode)
    if not result:
        raise HTTPException(status_code=404, detail='Lecture not found')
    return result


@router.delete('/{lecture_id}')
async def delete_lecture(lecture_id: str, db: AsyncSession = Depends(get_db)):
    success = await lecture_service.delete_lecture(db, lecture_id)
    if not success:
        raise HTTPException(status_code=404, detail='Lecture not found')
    return {'status': 'success'}
