import json
import re
import shutil
import uuid
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import (
    JOB_STATUS_DONE,
    JOB_STATUS_PENDING,
    JOB_TYPE_VERIFY,
    Lecture,
    ProcessingJob,
    normalize_job_type,
)
from app.services.storage_service import make_file_url, resolve_storage_path


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


async def get_current_job(db: AsyncSession, lecture_id: str) -> ProcessingJob | None:
    """lecture의 현재(최신) job을 반환한다. lecture당 실행 중 job은 최대 하나이므로
    job_id 없이 lecture_id만으로 스트리밍 추적이 가능하다."""
    lecture = await _get_lecture(db, lecture_id)
    return lecture.last_job if lecture else None


def build_job(lecture_id, mode: str | None = None) -> ProcessingJob:
    """업로드·재시도 공통 job 삽입 단위.

    'Job 없는 Lecture'가 존재하지 않는다는 불변조건을, 두 생성 경로(POST /lectures의
    합성 연산, POST /lectures/{id}/jobs의 원시 연산)가 이 함수 하나를 공유함으로써 구조로 보장한다.
    호출자는 반환된 job을 같은 트랜잭션에서 db.add 한다.

    job_type은 mode에 따라 정해진다 — verify가 기본이고 verify_only도 만들 수 있다.
    """
    return ProcessingJob(
        id=uuid.uuid4(),
        lecture_id=lecture_id,
        job_type=normalize_job_type(mode, JOB_TYPE_VERIFY),
        status=JOB_STATUS_PENDING,
        current_stage='검증 파이프라인을 시작합니다.',
        pipeline_stages=[],
    )


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
    job = lecture.last_job
    return {
        'id': str(lecture.id),
        'title': lecture.title or str(lecture.id),
        'description': lecture.description or '',
        'source_tag': lecture.source_tag,
        'video_url': make_file_url(lecture.video_path),
        'created_at': lecture.created_at.isoformat() if lecture.created_at else None,
        'job': _job_dict(job),
    }


def _lecture_thumbnail_url(lecture: Lecture) -> str:
    """전처리 단계에서 이미 추출해둔 슬라이드 프레임 중 첫 장을 썸네일로 재사용한다.
    별도로 영상에서 프레임을 다시 뽑을 필요가 없다 - 아직 전처리 전이면 빈 문자열."""
    output_dir = resolve_storage_path(lecture.output_dir)
    if not output_dir:
        return ''
    slides_dir = output_dir / 'slides'
    if not slides_dir.is_dir():
        return ''
    candidates = sorted(slides_dir.glob('scene_*_base.jpg'))
    if not candidates:
        return ''
    return make_file_url(candidates[0])


async def list_lectures(db: AsyncSession, status_filter: str | None = None) -> list[dict[str, Any]]:
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
            'source_tag': lecture.source_tag,
            'thumbnail_url': _lecture_thumbnail_url(lecture),
            'status': job.status if job else 'unknown',
            'job_id': str(job.id) if job else None,
            'job_type': job.job_type if job else None,
            'current_stage': job.current_stage if job else None,
            'error_message': job.error_message if job else None,
            'pipeline_stages': job.pipeline_stages or [] if job else [],
            'created_at': lecture.created_at.isoformat() if lecture.created_at else None,
        })
    return rows


async def create_job(db: AsyncSession, lecture_id: str, mode: str | None = None) -> dict[str, Any] | None:
    """기존 Lecture에 새 Job(재시도)을 생성하는 원시 연산. build_job을 공유한다."""
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        return None
    job = build_job(lecture.id, mode)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return {'status': 'success', 'job_id': str(job.id), 'job_type': job.job_type}


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


def _attach_slide_image_urls(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            enriched.append(item)
            continue
        row = dict(item)
        image_url = make_file_url(row.get('slide_image_path') or row.get('image_path'))
        if image_url:
            row['slide_image_url'] = image_url
        enriched.append(row)
    return enriched


def _filter_feedback_by_status(items: list[dict[str, Any]], status: str) -> list[dict[str, Any]]:
    return [
        item
        for item in items
        if isinstance(item, dict) and str(item.get('status') or '') == status
    ]


def _restore_feedback_display_aliases(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild legacy UI aliases removed from the stored verifier JSON."""
    restored: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            restored.append(item)
            continue
        row = dict(item)
        problem = dict(row.get('problem') or {})
        feedback = dict(row.get('professor_feedback') or {})
        evidence = dict(row.get('evidence') or {})

        summary = problem.get('summary') or feedback.get('summary') or ''
        why_wrong = problem.get('why_wrong') or feedback.get('why_wrong') or ''
        recommendation = (
            problem.get('recommendation')
            or problem.get('correct_info')
            or feedback.get('suggested_rephrase')
            or feedback.get('teaching_note')
            or ''
        )

        if recommendation and 'correct_info' not in problem:
            problem['correct_info'] = recommendation
        if why_wrong and 'evidence_in_context' not in evidence:
            evidence['evidence_in_context'] = why_wrong
        if not feedback and (summary or why_wrong or recommendation):
            row['professor_feedback'] = {
                'summary': summary,
                'why_wrong': why_wrong,
                'teaching_note': recommendation,
                'suggested_rephrase': recommendation,
            }
        elif feedback:
            row['professor_feedback'] = feedback
        row['problem'] = problem
        row['evidence'] = evidence
        restored.append(row)
    return restored


def build_content_verification_response(lecture_id: str, stem: str, verifier_path: str, data: dict) -> dict[str, Any]:
    """검증 결과 원본 JSON → API 응답 매핑 (순수 함수).

    반환 dict의 키 구성은 프론트엔드와의 계약이다.
    """
    flow = data.get('claim_decision_flow', {}) or {}
    summary = data.get('claim_decision_flow_summary', {}) or {}
    content_summary = data.get('summary', {}) or summary
    feedback_items = _restore_feedback_display_aliases(data.get('feedback_items', []) or [])
    final_claims = (
        flow.get('final_confirmed_claims', [])
        or data.get('final_confirmed_claims', [])
        or _filter_feedback_by_status(feedback_items, 'confirmed')
    )
    needs_review_claims = (
        flow.get('needs_review_claims', [])
        or data.get('needs_review_claims', [])
        or _filter_feedback_by_status(feedback_items, 'professor_check')
    )
    verifier_rejected_claims = (
        flow.get('verifier_rejected_claims', [])
        or data.get('verifier_rejected_claims', [])
        or _filter_feedback_by_status(feedback_items, 'rejected')
    )
    slide_errors = _attach_slide_image_urls(data.get('slide_errors', []) or [])
    slide_error_needs_review = _attach_slide_image_urls(data.get('slide_error_needs_review', []) or [])

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
            'slide_error_needs_review': len(slide_error_needs_review),
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
        'slide_error_needs_review': slide_error_needs_review,
        'slide_error_consensus': data.get('slide_error_consensus', {}) or {},
        'slide_error_status': data.get('slide_error_status', ''),
        'slide_error_summary': data.get('slide_error_summary', {}) or {},
        'slide_error_path': data.get('slide_error_path', ''),
        'claim_decision_flow_summary': summary,
        'classified_issue_artifacts': data.get('classified_issue_artifacts', {}) or {},
        'classified_issue_verifier_path': data.get('classified_issue_verifier_path', ''),
        'classified_issue_verifier': (data.get('views', {}) or {}).get('classified_issue_verifier', {}),
    }


def _is_complete_verification_result(path, stem: str) -> bool:
    """Return whether a benchmark verification artifact is safe to expose.

    Benchmark runs can leave a syntactically valid final JSON behind even when
    one of the model-specific verifier calls failed or produced unparseable
    rows.  Those artifacts must not take precedence over a later complete run.
    """
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return False
    if not isinstance(data, dict) or data.get('schema_version') != 'content_verification.v2':
        return False

    model_path = path.parent / f'{stem}_classified_issue_verifier.json'
    if not model_path.is_file():
        return True
    try:
        verifier = json.loads(model_path.read_text(encoding='utf-8'))
    except Exception:
        return False
    model_breakdown = (verifier.get('summary', {}) or {}).get('model_breakdown', {}) or {}
    for model_result in model_breakdown.values():
        if not isinstance(model_result, dict):
            return False
        if model_result.get('status') not in (None, 'ok'):
            return False
        if _safe_count(model_result.get('parse_failed_count')):
            return False
    return True


def _verification_result_candidates(analyzer_dir, output_dir, stem: str) -> list:
    """Order complete benchmark results before legacy result locations."""
    benchmark_results = []
    for benchmark_dir in analyzer_dir.glob('benchmark_*'):
        path = benchmark_dir / f'{stem}_verification_final.json'
        if not _is_complete_verification_result(path, stem):
            continue
        match = re.search(r'_v(\d+)$', benchmark_dir.name)
        version = int(match.group(1)) if match else -1
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0
        benchmark_results.append((version, mtime, path))
    benchmark_results.sort(key=lambda item: (item[0], item[1]), reverse=True)

    return [item[2] for item in benchmark_results] + [
        analyzer_dir / f'{stem}_content_verification.json',
        output_dir / f'{stem}_content_verification.json',
        analyzer_dir / f'{stem}_verification_final.json',
        output_dir / f'{stem}_verification_final.json',
    ]


ARTIFACT_STAGE_KEYS: dict[str, str] = {
    'claim_extraction': 'claims_json',
    'issue_judge': 'issue_judge',
    'issue_judge_summary': 'issue_judge_summary',
    'issue_classification': 'issue_types',
    'web_grounding': 'classified_issue_evidence',
    'final_verification': 'classified_issue_verifier',
    'slide_review': 'slide_errors',
}


async def get_lecture_artifact(db: AsyncSession, lecture_id: str, stage: str) -> dict[str, Any]:
    """GET /lectures/{id}/artifacts/{stage} 전용. 검증 파이프라인 중간단계 산출물 원본을 반환한다.

    /result와 달리 가공 없이 raw JSON을 그대로 내려준다 — 프론트(단계별 리포트 화면)에서
    필요한 필드만 추려서 렌더링한다.
    """
    artifact_key = ARTIFACT_STAGE_KEYS.get(stage)
    if not artifact_key:
        raise HTTPException(status_code=404, detail=f'Unknown stage: {stage}')

    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        raise HTTPException(status_code=404, detail='Lecture not found')
    job = lecture.last_job
    if not job or job.status != JOB_STATUS_DONE:
        raise HTTPException(status_code=404, detail='Verification result not ready')

    stem = str(lecture.id)
    output_dir = resolve_storage_path(lecture.output_dir)
    if not output_dir:
        raise HTTPException(status_code=404, detail='Content verification file not found')
    analyzer_dir = output_dir / f'{stem}_analyzer'
    candidates = _verification_result_candidates(analyzer_dir, output_dir, stem)
    verifier_path = next((path for path in candidates if path.exists()), None)
    if not verifier_path:
        raise HTTPException(status_code=404, detail='Content verification file not found')
    try:
        data = json.loads(verifier_path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Error reading content verification: {exc}') from exc

    artifacts = data.get('classified_issue_artifacts', {}) or {}
    artifact_path_value = artifacts.get(artifact_key)
    if not artifact_path_value:
        raise HTTPException(status_code=404, detail=f'Artifact not available for stage: {stage}')
    artifact_path = resolve_storage_path(artifact_path_value)
    if not artifact_path or not artifact_path.exists():
        raise HTTPException(status_code=404, detail=f'Artifact file not found for stage: {stage}')
    try:
        artifact_data = json.loads(artifact_path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Error reading artifact: {exc}') from exc

    if stage == 'slide_review' and isinstance(artifact_data, dict):
        artifact_data['slide_errors'] = _attach_slide_image_urls(artifact_data.get('slide_errors', []) or [])
    return artifact_data


async def get_verified_result(db: AsyncSession, lecture_id: str) -> dict[str, Any]:
    """GET /lectures/{id}/result 전용. job이 done인 lecture의 검증 결과를 반환한다.

    /lectures/{id}(job 상태 조회)와 독립된 라우트에서만 호출되므로, 실패를 그대로
    HTTPException으로 던져도 job 상태 조회에는 영향이 없다.
    """
    lecture = await _get_lecture(db, lecture_id)
    if not lecture:
        raise HTTPException(status_code=404, detail='Lecture not found')
    job = lecture.last_job
    if not job or job.status != JOB_STATUS_DONE:
        raise HTTPException(status_code=404, detail='Verification result not ready')

    stem = str(lecture.id)
    output_dir = resolve_storage_path(lecture.output_dir)
    if not output_dir:
        raise HTTPException(status_code=404, detail='Content verification file not found')
    analyzer_dir = output_dir / f'{stem}_analyzer'
    candidates = _verification_result_candidates(analyzer_dir, output_dir, stem)
    verifier_path = next((path for path in candidates if path.exists()), None)
    if not verifier_path:
        raise HTTPException(status_code=404, detail='Content verification file not found')
    try:
        data = json.loads(verifier_path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Error reading content verification: {exc}') from exc
    return build_content_verification_response(str(lecture.id), stem, str(verifier_path), data)
