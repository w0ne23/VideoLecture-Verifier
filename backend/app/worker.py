import asyncio
import concurrent.futures
import logging
import multiprocessing
import os
import sys
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from sqlalchemy import text

from app.config import LOCAL_STORAGE_DIR, PIPELINE_ROOT
from app.db import AsyncSessionLocal
from app.models import JOB_STATUS_DONE, JOB_STATUS_ERROR, JOB_TYPE_VERIFY, JOB_TYPE_VERIFY_ONLY
from pipeline.logging_utils import pipeline_log_context
from app.services.job_service import update_job_stage_sync
from app.services.model_settings_service import STAGE_MODEL_ENV_KEYS, fetch_stage_models_sync

logger = logging.getLogger(__name__)

PIPELINE_STAGE_KEYS = [
    'preprocess_slide_extract',
    'preprocess_audio_quality',
    'preprocess_slide_analyze',
    'preprocess_audio_transcribe',
    'verifier_build_analyzer_input',
    'verifier_claim_extraction',
    'verifier_issue_judge',
    'verifier_issue_classification',
    'verifier_final_verification',
    'verifier_web_grounding',
    'verify_slide_inspect',
    'verify_slide_syntax',
]

PIPELINE_STAGE_LABELS = {
    'preprocess_slide_extract': '슬라이드 추출',
    'preprocess_audio_quality': '오디오 품질 분석',
    'preprocess_slide_analyze': '슬라이드 분석',
    # transcribe_audio(P2B)에 이어 P3 process_audio(오디오 맥락 후처리)까지 끝나야
    # done으로 보고된다 — 다이어그램의 "음성 전사" 노드에 맞춰 P3를 이 단계에 묶었다.
    'preprocess_audio_transcribe': '음성 전사',
    'verifier_build_analyzer_input': '검증 입력 데이터 구성',
    'verifier_claim_extraction': '주장 후보 추출',
    'verifier_issue_judge': '이슈 후보 판단',
    'verifier_issue_classification': '이슈 유형 분류',
    'verifier_final_verification': '멀티 LLM 검증',
    'verifier_web_grounding': '웹 근거 검증',
    'verify_slide_inspect': '슬라이드 검사',
    'verify_slide_syntax': '문법/코드 오류 점검',
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
        # 관리자가 설정 화면에서 지정한 모델이 있으면 그 값을, 없으면 지금까지처럼
        # .env 기본값을 쓴다. 이 프로세스 안에서만 os.environ을 갱신하므로 다른
        # 동시 작업이나 부모 프로세스에는 영향을 주지 않는다.
        stage_models = fetch_stage_models_sync()
        # 이전 잡에서 남은 stage env가 남지 않도록 허용 키를 먼저 비운다.
        for env_key in STAGE_MODEL_ENV_KEYS:
            os.environ.pop(env_key, None)
        for env_key, value in stage_models.items():
            os.environ[env_key] = value
        os.environ['VERIFIER_MODEL_MODE'] = 'generic' if stage_models else 'fixed'

        from app.services.storage_service import resolve_storage_path
        video_path = resolve_storage_path(input_path)

        output_dir = LOCAL_STORAGE_DIR / 'results' / lecture_id
        slides_dir = output_dir / 'slides'
        slides_dir.mkdir(parents=True, exist_ok=True)

        stages_state = {key: 'wait' for key in PIPELINE_STAGE_KEYS}
        if job_type == JOB_TYPE_VERIFY_ONLY:
            # verify_only는 이전 실행이 남긴 전처리 산출물을 그대로 재사용하는 것이
            # 전제라, 전처리 단계는 이번 실행에서 아예 호출되지 않는다. 'wait'로
            # 영원히 남겨두면 프론트에 전처리가 안 끝난 것처럼 보이므로 시작 시점에
            # 바로 완료 처리한다.
            for key in (
                'preprocess_slide_extract',
                'preprocess_audio_quality',
                'preprocess_slide_analyze',
                'preprocess_audio_transcribe',
            ):
                stages_state[key] = 'done'

        # 슬라이드 오류 검사(verify_slide_inspect/verify_slide_syntax) 단계가
        # verifier_claim_extraction 체인과 별도 스레드에서 동시에 진행되므로
        # (pipeline/verifier/run_all.py), 두 스레드가 동시에 이 콜백을 호출할 수 있다.
        # 락 없이 두면 스냅샷을 만들어 DB에 쓰는 순서가 뒤집혀 먼저 만든(더 오래된)
        # 스냅샷이 나중 것을 덮어쓸 수 있으므로 락으로 직렬화한다.
        on_progress_lock = threading.Lock()

        def on_progress(stage_key: str, status: str):
            with on_progress_lock:
                if stage_key in stages_state:
                    stages_state[stage_key] = status
                stages_array = [{'stage': key, 'status': value} for key, value in stages_state.items()]
                update_job_stage_sync(job_id, stages_array, _stage_text(stage_key, status))
            logger.info('[%s] Progress: %s -> %s', job_id, stage_key, status)

        with pipeline_log_context(output_dir):
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
            args.job_type = job_type
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
        job_type_val = JOB_TYPE_VERIFY

        async with AsyncSessionLocal() as db:
            result = await db.execute(text("""
                SELECT pj.id, pj.lecture_id, pj.job_type, l.video_path, l.created_at, l.title
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
                job_type_val = job['job_type'] or JOB_TYPE_VERIFY
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
                job_type_val,
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
