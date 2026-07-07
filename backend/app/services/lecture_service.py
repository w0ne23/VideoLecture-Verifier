import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import (
    JOB_STATUS_DONE,
    JOB_STATUS_ERROR,
    JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING,
    JOB_TYPE_VERIFY,
    JOB_TYPE_VERIFY_ONLY,
    Lecture,
    ProcessingJob,
    normalize_job_type,
)
from app.services.storage_service import lecture_output_dir, make_file_url, resolve_storage_path


def normalize_domain_value(value: str | None) -> str:
    return (value or 'etc').strip() or 'etc'


async def _get_lecture(db: AsyncSession, lecture_id: str | uuid.UUID) -> Lecture | None:
    try:
        ident = uuid.UUID(str(lecture_id))
    except (TypeError, ValueError):
        return None
    result = await db.execute(
        select(Lecture)
        .options(selectinload(Lecture.processing_jobs))
        .where(Lecture.id == ident)
    )
    return result.scalar_one_or_none()


async def get_job(db: AsyncSession, job_id: str) -> ProcessingJob | None:
    try:
        ident = uuid.UUID(str(job_id))
    except (TypeError, ValueError):
        return None
    result = await db.execute(select(ProcessingJob).where(ProcessingJob.id == ident))
    return result.scalar_one_or_none()


async def get_latest_job(db: AsyncSession, lecture_id: str) -> ProcessingJob | None:
    try:
        ident = uuid.UUID(str(lecture_id))
    except (TypeError, ValueError):
        return None
    result = await db.execute(
        select(ProcessingJob)
        .where(ProcessingJob.lecture_id == ident)
        .order_by(ProcessingJob.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_latest_job_by_mode(db: AsyncSession, lecture_id: str, mode: str) -> ProcessingJob | None:
    canonical = normalize_job_type(mode)
    try:
        ident = uuid.UUID(str(lecture_id))
    except (TypeError, ValueError):
        return None
    result = await db.execute(
        select(ProcessingJob)
        .where(ProcessingJob.lecture_id == ident, ProcessingJob.job_type == canonical)
        .order_by(ProcessingJob.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


def _job_dict(job: ProcessingJob | None) -> dict[str, Any] | None:
    if not job:
        return None
    return {
        'id': str(job.id),
        'job_type': job.job_type,
        'status': job.status,
        'current_stage': job.current_stage,
        'error_message': job.error_message,
        'pipeline_stages': job.pipeline_stages or [],
        'created_at': job.created_at.isoformat() if job.created_at else None,
    }


async def get_lecture_detail(db: AsyncSession, lecture_id: str) -> dict[str, Any] | None:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        return None
    stem = str(lecture.id)
    return {
        'id': str(lecture.id),
        'title': lecture.title or stem,
        'description': lecture.description or '',
        'video_path': lecture.video_path,
        'video_url': make_file_url(lecture.video_path),
        'output_dir': str(resolve_storage_path(lecture.output_dir) or ''),
        'stem': stem,
        'is_verified': bool(lecture.is_verified),
        'created_at': lecture.created_at.isoformat() if lecture.created_at else None,
        'job': _job_dict(lecture.last_job),
    }


async def list_jobs(db: AsyncSession, status_filter: str | None = None) -> list[dict[str, Any]]:
    result = await db.execute(
        select(Lecture)
        .options(selectinload(Lecture.processing_jobs))
        .order_by(Lecture.created_at.desc())
    )
    rows = []
    for lecture in result.scalars().unique().all():
        job = lecture.last_job
        if status_filter and (not job or job.status != status_filter):
            continue
        rows.append({
            'id': str(lecture.id),
            'title': lecture.title or str(lecture.id),
            'description': lecture.description or '',
            'status': job.status if job else 'unknown',
            'job_id': str(job.id) if job else None,
            'job_type': job.job_type if job else None,
            'current_stage': job.current_stage if job else None,
            'error_message': job.error_message if job else None,
            'pipeline_stages': job.pipeline_stages or [] if job else [],
            'is_verified': bool(lecture.is_verified),
            'created_at': lecture.created_at.isoformat() if lecture.created_at else None,
        })
    return rows


async def retry_lecture(db: AsyncSession, lecture_id: str, mode: str | None = None) -> dict[str, Any] | None:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        return None
    job_type = normalize_job_type(mode, JOB_TYPE_VERIFY)
    job = ProcessingJob(
        id=uuid.uuid4(),
        lecture_id=lecture.id,
        job_type=job_type,
        status=JOB_STATUS_PENDING,
        current_stage='검증 파이프라인을 다시 시작합니다.',
        error_message=None,
        pipeline_stages=[],
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return {'status': 'success', 'job_id': str(job.id), 'job_type': job.job_type}


async def confirm_verified_lecture(db: AsyncSession, lecture_id: str) -> dict[str, Any] | None:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        return None
    latest_job = lecture.last_job
    if not latest_job or latest_job.status not in {JOB_STATUS_DONE}:
        raise HTTPException(status_code=409, detail='No completed verification is ready for review')
    lecture.is_verified = True
    latest_job.current_stage = '검토 완료'
    await db.commit()
    return {
        'status': 'success',
        'lecture_id': str(lecture.id),
        'job_id': str(latest_job.id),
        'is_verified': True,
    }


async def delete_lecture(db: AsyncSession, lecture_id: str) -> bool:
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        return False
    video_path = resolve_storage_path(lecture.video_path)
    if video_path is not None:
        shutil.rmtree(video_path.parent, ignore_errors=True)
    output_dir = resolve_storage_path(lecture.output_dir)
    if output_dir is not None:
        shutil.rmtree(output_dir, ignore_errors=True)
    await db.delete(lecture)
    await db.commit()
    return True


def _safe_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def build_content_verification_response(lecture_id: str, stem: str, verifier_path: str, data: dict) -> dict[str, Any]:
    """검증 결과 원본 JSON → API 응답 매핑 (순수 함수).

    반환 dict의 키 구성은 프론트엔드와의 계약이다 — tests/test_verifier_contract.py가
    이 계약을 고정하므로, 키를 바꾸면 테스트와 프론트를 함께 수정해야 한다.
    """
    flow = data.get('claim_decision_flow', {}) or {}
    summary = data.get('claim_decision_flow_summary', {}) or {}
    content_summary = data.get('summary', {}) or summary
    feedback_items = data.get('feedback_items', []) or []
    final_claims = flow.get('final_confirmed_claims', []) or data.get('final_confirmed_claims', []) or []
    needs_review_claims = flow.get('needs_review_claims', []) or data.get('needs_review_claims', []) or []
    verifier_rejected_claims = flow.get('verifier_rejected_claims', []) or data.get('verifier_rejected_claims', []) or []
    slide_errors = data.get('slide_errors', []) or []

    return {
        'lecture_id': str(lecture_id),
        'stem': stem,
        'verification_path': str(verifier_path),
        'schema_version': data.get('schema_version'),
        'mode': data.get('mode', ''),
        'verification_date': data.get('verification_date', ''),
        'models': data.get('models', []) or [],
        'pipeline_models': data.get('pipeline_models', {}) or {},
        'primary_model': data.get('primary_model', ''),
        'verifier_source_models': data.get('verifier_source_models', []) or [],
        'verifier_model_weights': data.get('verifier_model_weights', {}) or {},
        'severity_score_report': data.get('severity_score_report', {}) or {},
        'summary': content_summary,
        'overview': data.get('claim_decision_overview', []) or [],
        'counts': {
            'final_confirmed': _safe_count(content_summary.get('confirmed_feedback_count', summary.get('final_confirmed_claim_count', len(final_claims)))),
            'needs_review': _safe_count(content_summary.get('review_needed_feedback_count', summary.get('needs_review_claim_count', len(needs_review_claims)))),
            'rejected': _safe_count(content_summary.get('rejected_feedback_count', len(verifier_rejected_claims))),
            'slide_errors': _safe_count(content_summary.get('slide_error_count', len(slide_errors))),
            'slide_error_needs_review': len(data.get('slide_error_needs_review', []) or []),
            'verifier_rejected': _safe_count(summary.get('verifier_rejected_claim_count', len(verifier_rejected_claims))),
        },
        'claims': data.get('claims', []) or [],
        'feedback_groups': data.get('feedback_groups', []) or [],
        'feedback_items': feedback_items,
        'views': data.get('views', {}) or {},
        'final_confirmed_claims': final_claims,
        'needs_review_claims': needs_review_claims,
        'verifier_rejected_claims': verifier_rejected_claims,
        'issues': data.get('issues', []) or [],
        'slide_errors': slide_errors,
        'slide_error_needs_review': data.get('slide_error_needs_review', []) or [],
        'slide_error_consensus': data.get('slide_error_consensus', {}) or {},
        'slide_error_status': data.get('slide_error_status', ''),
        'slide_error_summary': data.get('slide_error_summary', {}) or {},
        'slide_error_path': data.get('slide_error_path', ''),
        'claim_decision_flow_summary': summary,
        'classified_issue_artifacts': data.get('classified_issue_artifacts', {}) or {},
        'classified_issue_verifier_path': data.get('classified_issue_verifier_path', ''),
        'classified_issue_verifier': (data.get('views', {}) or {}).get('classified_issue_verifier', {}),
    }


async def get_content_verification(db: AsyncSession, lecture_id: str) -> dict[str, Any]:
    detail = await get_lecture_detail(db, lecture_id)
    if not detail:
        raise HTTPException(status_code=404, detail='Lecture result not found')

    output_dir = Path(detail['output_dir'])
    stem = str(detail['stem'])
    analyzer_dir = output_dir / f'{stem}_analyzer'
    candidates = [
        analyzer_dir / f'{stem}_content_verification.json',
        output_dir / f'{stem}_content_verification.json',
        analyzer_dir / f'{stem}_verification_final.json',
        output_dir / f'{stem}_verification_final.json',
    ]
    verifier_path = next((path for path in candidates if path.exists()), None)
    if not verifier_path:
        raise HTTPException(status_code=404, detail='Content verification file not found')

    try:
        data = json.loads(verifier_path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Error reading content verification: {exc}') from exc

    return build_content_verification_response(str(detail['id']), stem, str(verifier_path), data)
