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
    LECTURE_SOURCE_TAGS,
    Lecture,
    normalize_job_type,
)
from app.services import lecture_service
from app.services.storage_service import save_upload, storage_relpath

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/lectures')


@router.get('')
async def list_lectures(db: AsyncSession = Depends(get_db), status: Optional[str] = Query(None)):
    return await lecture_service.list_lectures(db, status_filter=status)


@router.post('')
async def create_lecture(
    video: UploadFile = File(...),
    title: str = Form(''),
    description: str = Form(''),
    source_tag: str = Form(...),
    workflow_mode: str = Form(JOB_TYPE_VERIFY),
    db: AsyncSession = Depends(get_db),
):
    """영상 업로드와 함께 Lecture·최초 Job을 생성하는 합성 연산.

    Lecture 행 삽입과 job 삽입을 같은 트랜잭션에서 수행하며, job 삽입은 재시도 경로와
    동일한 build_job을 호출해 'Job 없는 Lecture'가 존재하지 않도록 보장한다.
    """
    job_type = normalize_job_type(workflow_mode, JOB_TYPE_VERIFY)
    if job_type != JOB_TYPE_VERIFY:
        raise HTTPException(status_code=400, detail='Only verify workflow is supported for uploaded videos')

    tag = (source_tag or '').strip().lower()
    if tag not in LECTURE_SOURCE_TAGS:
        raise HTTPException(
            status_code=400,
            detail=f'source_tag must be one of: {", ".join(LECTURE_SOURCE_TAGS)}',
        )

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
        source_tag=tag,
        # 컨테이너/호스트 어디서든 해석 가능하도록 LOCAL_STORAGE_DIR 상대경로로 저장
        video_path=storage_relpath(input_path),
        output_dir=storage_relpath(output_dir),
    )
    job = lecture_service.build_job(lecture_id, workflow_mode)
    db.add(lecture)
    db.add(job)
    await db.commit()
    await db.refresh(lecture)
    await db.refresh(job)

    return {
        'id': str(lecture.id),
        'title': lecture.title,
        'description': lecture.description or '',
        'source_tag': lecture.source_tag,
        'job_id': str(job.id),
        'job_type': job.job_type,
        'status': job.status,
        'created_at': lecture.created_at.isoformat() if lecture.created_at else None,
    }


@router.get('/{lecture_id}')
async def get_lecture(lecture_id: str, db: AsyncSession = Depends(get_db)):
    detail = await lecture_service.get_lecture_detail(db, lecture_id)
    if not detail:
        raise HTTPException(status_code=404, detail='Lecture not found')
    return detail


@router.get('/{lecture_id}/stream')
async def stream_lecture_status(lecture_id: str, request: Request):
    """lecture의 현재 job 상태를 SSE로 스트리밍한다. lecture당 실행 중 job은 최대 하나이므로
    job_id 없이 lecture_id만으로 추적한다."""

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                async with AsyncSessionLocal() as db:
                    job = await lecture_service.get_current_job(db, lecture_id)
                if not job:
                    yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                    break
                payload = {
                    'job_id': str(job.id),
                    'job_type': job.job_type,
                    'status': job.status,
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


@router.get('/{lecture_id}/result')
async def get_lecture_result(lecture_id: str, db: AsyncSession = Depends(get_db)):
    return await lecture_service.get_verified_result(db, lecture_id)


@router.get('/{lecture_id}/artifacts/{stage}')
async def get_lecture_artifact(lecture_id: str, stage: str, db: AsyncSession = Depends(get_db)):
    return await lecture_service.get_lecture_artifact(db, lecture_id, stage)


@router.post('/{lecture_id}/jobs')
async def create_lecture_job(lecture_id: str, mode: Optional[str] = Query(None), db: AsyncSession = Depends(get_db)):
    result = await lecture_service.create_job(db, lecture_id, mode=mode)
    if not result:
        raise HTTPException(status_code=404, detail='Lecture not found')
    return result


@router.delete('/{lecture_id}')
async def delete_lecture(lecture_id: str, db: AsyncSession = Depends(get_db)):
    success = await lecture_service.delete_lecture(db, lecture_id)
    if not success:
        raise HTTPException(status_code=404, detail='Lecture not found')
    return {'status': 'success'}
