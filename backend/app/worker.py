import asyncio
import concurrent.futures
import logging
import multiprocessing
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from sqlalchemy import text

from app.config import LOCAL_STORAGE_DIR, PIPELINE_ROOT
from app.db import AsyncSessionLocal
from app.models import JOB_STATUS_DONE, JOB_STATUS_ERROR, JOB_TYPE_VERIFY
from app.services.job_service import update_job_stage_sync

logger = logging.getLogger(__name__)

PIPELINE_STAGE_KEYS = [
    'preprocess_extract_media',
    'preprocess_textualize_transcribe',
    'preprocess_enrich_audio_annotation',
    'verifier_build_analyzer_input',
    'verifier_claim_extraction',
    'verifier_issue_judge',
    'verifier_issue_classification',
    'verifier_final_verification',
    'verify_slide_errors',
]

PIPELINE_STAGE_LABELS = {
    'preprocess_extract_media': '슬라이드 추출 및 오디오 품질 분석',
    'preprocess_textualize_transcribe': '슬라이드 텍스트화 및 전체 전사',
    'preprocess_enrich_audio_annotation': '필기 강조 및 오디오 후처리',
    'verifier_build_analyzer_input': '검증 입력 데이터 구성',
    'verifier_claim_extraction': '주장 후보 추출',
    'verifier_issue_judge': '이슈 후보 판단',
    'verifier_issue_classification': '이슈 유형 분류',
    'verifier_final_verification': '멀티 LLM 검증',
    'verify_slide_errors': '슬라이드 오류 검사',
}


def _stage_text(stage_key: str, status: str) -> str:
    label = PIPELINE_STAGE_LABELS.get(stage_key, stage_key)
    if status == 'run':
        return f'{label} 진행 중'
    if status == 'done':
        return f'{label} 완료'
    if status == 'error':
        return f'{label} 실패'
    return f'{label} 대기 중'


def pipeline_process(
    job_id: str,
    lecture_id: str,
    input_path: str,
    job_type: str = JOB_TYPE_VERIFY,
    uploaded_at: str | None = None,
    title: str = '',
    worker_index: int = 0,
    worker_count: int = 1,
):
    pipeline_root = Path(os.getenv('PIPELINE_ROOT', str(PIPELINE_ROOT))).resolve()
    if str(pipeline_root) not in sys.path:
        sys.path.insert(0, str(pipeline_root))
    os.chdir(pipeline_root)

    try:
        video_path = Path(input_path)
        if not video_path.is_absolute():
            video_path = pipeline_root / input_path

        output_dir = LOCAL_STORAGE_DIR / 'results' / lecture_id
        slides_dir = output_dir / 'slides'
        log_file_path = output_dir / 'pipeline.log'
        output_dir.mkdir(parents=True, exist_ok=True)
        slides_dir.mkdir(parents=True, exist_ok=True)

        stages_state = {key: 'wait' for key in PIPELINE_STAGE_KEYS}

        def on_progress(stage_key: str, status: str):
            if stage_key in stages_state:
                stages_state[stage_key] = status
            stages_array = [{'stage': key, 'status': value} for key, value in stages_state.items()]
            update_job_stage_sync(job_id, stages_array, _stage_text(stage_key, status))
            logger.info('[%s] Progress: %s -> %s', job_id, stage_key, status)

        with log_file_path.open('w', encoding='utf-8', buffering=1) as log_file:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                import pipeline.main as pipeline_main

                update_job_stage_sync(
                    job_id,
                    [{'stage': key, 'status': value} for key, value in stages_state.items()],
                    '파이프라인을 시작합니다.',
                )
                args = pipeline_main.get_parser().parse_args([
                    '--input', str(video_path),
                    '--output', str(output_dir),
                    '--slides', str(slides_dir),
                    '--lecture-id', lecture_id,
                    *(['--title', title] if title else []),
                    *(['--uploaded-at', uploaded_at] if uploaded_at else []),
                ])
                args.job_type = JOB_TYPE_VERIFY
                os.environ['PYTHONUNBUFFERED'] = '1'
                pipeline_main.run_pipeline(args, progress_callback=on_progress)

        return True, str(output_dir), None
    except BaseException as exc:
        details = traceback.format_exc()
        logger.error('--- [Child Process %s] FAILED: %s ---', job_id, exc)
        log_file_path = LOCAL_STORAGE_DIR / 'results' / lecture_id / 'pipeline.log'
        if log_file_path.parent.exists():
            with log_file_path.open('a', encoding='utf-8') as log_file:
                log_file.write(f'\n[{job_id}] Pipeline failed: {details}\n')
        return False, None, str(exc)


async def worker_loop(worker_index: int = 0, worker_count: int = 1):
    mp_context = multiprocessing.get_context('spawn')
    executor = ProcessPoolExecutor(max_workers=1, mp_context=mp_context)

    async with AsyncSessionLocal() as db:
        await db.execute(text(
            "UPDATE processing_jobs SET status = 'error', error_message = '서버 재시작으로 인해 분석이 중단되었습니다.' "
            "WHERE status = 'running'"
        ))
        await db.commit()

    while True:
        job_id_val = None
        lecture_id_val = None
        input_path = None
        uploaded_at = None
        title = ''

        async with AsyncSessionLocal() as db:
            result = await db.execute(text("""
                SELECT pj.id, pj.lecture_id, l.video_path, l.created_at, l.title
                FROM processing_jobs pj
                JOIN lectures l ON l.id = pj.lecture_id
                WHERE pj.status = 'pending'
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            """))
            job = result.mappings().first()
            if job:
                job_id_val = job['id']
                lecture_id_val = job['lecture_id']
                input_path = job['video_path']
                title = job['title'] or ''
                uploaded_at = job['created_at'].isoformat() if job['created_at'] else None
                await db.execute(text("""
                    UPDATE processing_jobs
                    SET status = 'running', current_stage = '파이프라인을 시작합니다.'
                    WHERE id = :id
                """), {'id': job_id_val})
                await db.commit()

        if not job_id_val:
            await asyncio.sleep(5)
            continue

        try:
            loop = asyncio.get_running_loop()
            success, output_dir, error = await loop.run_in_executor(
                executor,
                pipeline_process,
                str(job_id_val),
                str(lecture_id_val),
                str(input_path),
                JOB_TYPE_VERIFY,
                uploaded_at,
                title,
                worker_index,
                worker_count,
            )
        except concurrent.futures.process.BrokenProcessPool as exc:
            error = f'파이프라인 프로세스 강제 종료: {exc}'
            success = False
            executor.shutdown(wait=False)
            executor = ProcessPoolExecutor(max_workers=1, mp_context=mp_context)
        except Exception as exc:
            error = f'시스템/프로세스 오류: {exc}'
            success = False

        async with AsyncSessionLocal() as db:
            if success:
                await db.execute(text("""
                    UPDATE processing_jobs
                    SET status = :status, current_stage = :stage
                    WHERE id = :id
                """), {'id': job_id_val, 'status': JOB_STATUS_DONE, 'stage': '검증 결과가 준비되었습니다.'})
            else:
                await db.execute(text("""
                    UPDATE processing_jobs
                    SET status = :status, error_message = :err, current_stage = :stage
                    WHERE id = :id
                """), {'id': job_id_val, 'status': JOB_STATUS_ERROR, 'err': error, 'stage': '분석 중 오류가 발생했습니다.'})
            await db.commit()
