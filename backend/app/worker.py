import asyncio
import os
import sys
import multiprocessing
import logging
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import traceback
from contextlib import redirect_stdout, redirect_stderr

from sqlalchemy import text
from app.db import AsyncSessionLocal
from app.models import (
    JOB_STATUS_DONE,
    JOB_TYPE_DIRECT_UPLOAD,
    JOB_TYPE_LEGACY_FULL,
    JOB_TYPE_GRAPH_UPLOAD,
    JOB_TYPE_PUBLISH,
    JOB_TYPE_UPLOAD,
    JOB_TYPE_VERIFY,
    JOB_TYPE_VERIFIED_UPLOAD,
    normalize_job_type,
)
from app.services.job_service import update_job_stage_sync

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
root_env         = PROJECT_ROOT_DIR / ".env"

if root_env.exists():
    load_dotenv(dotenv_path=root_env)
else:
    load_dotenv()

PROJECT_ROOT      = Path("/pipeline") if Path("/pipeline").exists() else PROJECT_ROOT_DIR
LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR", str(PROJECT_ROOT / "local_storage"))

PIPELINE_STAGE_KEYS = [
    "preprocess_extract_media",
    "preprocess_textualize_transcribe",
    "preprocess_enrich_audio_annotation",
    "verifier_build_analyzer_input",
    "verifier_claim_extraction",
    "verifier_issue_judge",
    "verifier_issue_classification",
    "verifier_final_verification",
    "verify_slide_errors",
    "verifier_run",
    "graph_classify_scene",
    "graph_fusion",
    "graph_triples",
    "graph_lance_index",
    "graph_graphrag_index",
    "graph_metadata",
    "graph_recommender_index",
]

GRAPH_UPLOAD_PRECOMPLETED_STAGE_KEYS = {
    "preprocess_extract_media",
    "preprocess_textualize_transcribe",
    "preprocess_enrich_audio_annotation",
}

PIPELINE_JOB_TYPE_ALIASES = {
    "legacy": JOB_TYPE_LEGACY_FULL,
    JOB_TYPE_LEGACY_FULL: JOB_TYPE_LEGACY_FULL,
    "direct": JOB_TYPE_DIRECT_UPLOAD,
    JOB_TYPE_DIRECT_UPLOAD: JOB_TYPE_DIRECT_UPLOAD,
    JOB_TYPE_PUBLISH: JOB_TYPE_PUBLISH,
    "publication": JOB_TYPE_PUBLISH,
    JOB_TYPE_UPLOAD: JOB_TYPE_PUBLISH,
    "verified": JOB_TYPE_VERIFY,
    JOB_TYPE_VERIFIED_UPLOAD: JOB_TYPE_VERIFIED_UPLOAD,
    JOB_TYPE_VERIFY: JOB_TYPE_VERIFY,
    "graph": JOB_TYPE_GRAPH_UPLOAD,
    JOB_TYPE_GRAPH_UPLOAD: JOB_TYPE_GRAPH_UPLOAD,
}


def _pipeline_job_type(job_type: str | None) -> str:
    token = (job_type or JOB_TYPE_LEGACY_FULL).strip().lower().replace("-", "_")
    return PIPELINE_JOB_TYPE_ALIASES.get(token, JOB_TYPE_LEGACY_FULL)

PIPELINE_STAGE_LABELS = {
    "preprocess_extract_media": "슬라이드 추출 및 오디오 품질 분석",
    "preprocess_textualize_transcribe": "슬라이드 텍스트화 및 전체 전사",
    "preprocess_enrich_audio_annotation": "필기 강조 및 오디오 후처리",
    "verifier_build_analyzer_input": "검증 입력 데이터 구성",
    "verifier_claim_extraction": "주장 후보 추출",
    "verifier_issue_judge": "이슈 후보 판단",
    "verifier_issue_classification": "이슈 유형 분류",
    "verifier_final_verification": "멀티 LLM 검증",
    "verify_slide_errors": "최종 검증",
    "verifier_run": "강의 내용 검증 실행",
    "graph_classify_scene": "강의 구조 파악",
    "graph_fusion": "데이터 통합",
    "graph_triples": "그래프 데이터 생성",
    "graph_lance_index": "벡터 검색 인덱스 생성",
    "graph_graphrag_index": "GraphRAG 인덱스 생성",
    "graph_metadata": "강의 메타데이터 생성",
    "graph_recommender_index": "강의 추천 인덱스 생성",
}


def _parse_cpu_group(group: str) -> list[int]:
    cpus: set[int] = set()
    for part in group.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                start, end = end, start
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(token))
    return sorted(cpus)


def _available_cpus() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return list(range(os.cpu_count() or 1))


def _cpu_set_for_worker(worker_index: int, worker_count: int) -> list[int]:
    explicit = os.getenv("GRAPHLEC_WORKER_CPU_SETS", "").strip()
    if explicit:
        groups = [group.strip() for group in explicit.split(";") if group.strip()]
        if groups:
            return _parse_cpu_group(groups[worker_index % len(groups)])

    cpus = _available_cpus()
    worker_count = max(1, int(worker_count))
    worker_index = max(0, int(worker_index)) % worker_count
    chunk = max(1, (len(cpus) + worker_count - 1) // worker_count)
    start = worker_index * chunk
    assigned = cpus[start:start + chunk]
    return assigned or cpus


def _apply_pipeline_cpu_affinity(worker_index: int, worker_count: int, job_id: str) -> None:
    if os.getenv("GRAPHLEC_WORKER_PIN_CPUS", "1").lower() in {"0", "false", "no"}:
        return
    cpus = _cpu_set_for_worker(worker_index, worker_count)
    try:
        os.sched_setaffinity(0, set(cpus))
        logger.info(
            "--- [Child Process %s] CPU affinity set: worker=%s/%s cpus=%s ---",
            job_id,
            worker_index + 1,
            worker_count,
            ",".join(map(str, cpus)),
        )
    except AttributeError:
        logger.info("--- [Child Process %s] CPU affinity unsupported on this platform. ---", job_id)
    except Exception as exc:
        logger.warning("--- [Child Process %s] CPU affinity failed: %s ---", job_id, exc)


def _initial_stage_state(job_type: str) -> dict[str, str]:
    stages = {key: "wait" for key in PIPELINE_STAGE_KEYS}
    if job_type == JOB_TYPE_GRAPH_UPLOAD:
        for key in GRAPH_UPLOAD_PRECOMPLETED_STAGE_KEYS:
            stages[key] = "done"
    return stages


def _stage_text(stage_key: str, status: str) -> str:
    label = PIPELINE_STAGE_LABELS.get(stage_key, stage_key)
    if status == "run":
        return f"{label} 진행 중"
    if status == "done":
        return f"{label} 완료"
    if status == "error":
        return f"{label} 실패"
    return f"{label} 대기 중"


def pipeline_process(
    job_id: str,
    lecture_id: str,
    input_path: str,
    job_type: str,
    uploaded_at: str | None = None,
    title: str = "",
    worker_index: int = 0,
    worker_count: int = 1,
):
    pipeline_path = os.getenv("PIPELINE_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent))
    _apply_pipeline_cpu_affinity(worker_index, worker_count, job_id)

    # spawn된 자식 프로세스는 부모의 sys.path를 상속받지 않으므로 pipeline 패키지를 import하기 위해 명시적으로 경로를 추가한다.
    # os.chdir은 pipeline 내부의 상대 경로 참조를 위해 필요하다.
    logger.info(f"--- [Child Process {job_id}] Setting sys.path to: {pipeline_path} ---")
    if pipeline_path not in sys.path:
        sys.path.insert(0, pipeline_path)
    os.chdir(pipeline_path)

    try:
        video_path = Path(input_path)
        if not video_path.is_absolute():
            video_path = Path(pipeline_path) / input_path

        job_type = _pipeline_job_type(job_type)
        logger.info(f"--- [Child Process {job_id}] Target video: {video_path} ({job_type}) ---")

        output_dir    = Path(LOCAL_STORAGE_DIR) / "results" / lecture_id
        slides_dir    = output_dir / "slides"
        log_file_path = output_dir / "pipeline.log"

        output_dir.mkdir(parents=True, exist_ok=True)
        slides_dir.mkdir(parents=True, exist_ok=True)

        stages_state = _initial_stage_state(job_type)

        def on_progress(stage_key: str, status: str):
            if stage_key in stages_state:
                stages_state[stage_key] = status
            stages_array = [{"stage": k, "status": v} for k, v in stages_state.items()]
            stage_text = _stage_text(stage_key, status)
            update_job_stage_sync(job_id, stages_array, stage_text)
            logger.info(f"[{job_id}] Progress: {stage_key} -> {status}")

        with open(log_file_path, "w", encoding="utf-8", buffering=1) as log_file:
            try:
                log_file.reconfigure(line_buffering=True, write_through=True)
            except Exception:
                pass
            with redirect_stdout(log_file), redirect_stderr(log_file):
                try:
                    logger.info(f"[{job_id}] Importing pipeline...")
                    import pipeline.main as pipeline_main

                    init_array = [{"stage": k, "status": v} for k, v in stages_state.items()]
                    update_job_stage_sync(job_id, init_array, "파이프라인을 시작합니다.")

                    args = pipeline_main.get_parser().parse_args([
                        "--input",        str(video_path),
                        "--output",       str(output_dir),
                        "--slides",       str(slides_dir),
                        "--skip-neo4j",
                        "--metadata-dir", str(output_dir / "metadata"),
                        "--lance-root",   str(output_dir / "lancedb"),
                        "--lecture-id",   lecture_id,
                    ] + (["--title", title] if title else [])
                      + (["--uploaded-at", uploaded_at] if uploaded_at else []))
                    args.job_type = job_type
                    os.environ["PYTHONUNBUFFERED"] = "1"
                    logger.info(f"[{job_id}] Starting pipeline job_type={job_type}...")
                    pipeline_main.run_pipeline(args, progress_callback=on_progress)

                except ImportError as ie:
                    logger.error(f"[{job_id}] ImportError: {ie}. sys.path: {sys.path}")
                    raise

        return True, str(output_dir), None

    except BaseException as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"--- [Child Process {job_id}] FAILED: {e} ---")
        log_file_path = Path(LOCAL_STORAGE_DIR) / "results" / lecture_id / "pipeline.log"
        if log_file_path.parent.exists():
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"\n[{job_id}] Pipeline failed: {error_details}\n")
        return False, None, str(e)


async def worker_loop(worker_index: int = 0, worker_count: int = 1):
    try:
        mp_context = multiprocessing.get_context("spawn")
        executor   = ProcessPoolExecutor(max_workers=1, mp_context=mp_context)

        async with AsyncSessionLocal() as db:
            result = await db.execute(text(
                "UPDATE processing_jobs SET status = 'error', error_message = '서버 재시작으로 인해 분석이 중단되었습니다.' "
                "WHERE status = 'running'"
            ))
            count = result.rowcount
            if count > 0:
                logger.info(f"--- [Worker Recovery] Marked {count} orphaned jobs as error. ---")
            await db.commit()

        logger.info(
            "--- [Worker %s/%s] Started. Storage: %s CPU set: %s ---",
            worker_index + 1,
            worker_count,
            LOCAL_STORAGE_DIR,
            ",".join(map(str, _cpu_set_for_worker(worker_index, worker_count))),
        )

        while True:
            try:
                job_id_val = None
                job_lecture_id = None
                job_input_path = None
                job_uploaded_at = None
                job_title = ""
                job_type_val = None

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
                        job_id_val     = job["id"]
                        job_lecture_id = job["lecture_id"]
                        job_input_path = job["video_path"]
                        job_title      = job["title"] or ""
                        job_uploaded_at = (
                            job["created_at"].isoformat()
                            if job["created_at"]
                            else None
                        )
                        job_type_val   = job["job_type"] or JOB_TYPE_LEGACY_FULL
                        await db.execute(text("""
                            UPDATE processing_jobs
                            SET status = 'running', current_stage = '파이프라인을 시작합니다.'
                            WHERE id = :id
                        """), {"id": job_id_val})
                        await db.commit()

                if not job_id_val:
                    await asyncio.sleep(5)
                    continue

                job_id_str     = str(job_id_val)
                job_lecture_str = str(job_lecture_id)
                job_type_str = str(job_type_val or JOB_TYPE_LEGACY_FULL)
                pipeline_job_type = _pipeline_job_type(job_type_str)
                logger.info(
                    "--- [Worker] Starting pipeline: %s (lecture: %s, type: %s, pipeline_type: %s) ---",
                    job_id_str,
                    job_lecture_str,
                    job_type_str,
                    pipeline_job_type,
                )

                try:
                    loop = asyncio.get_running_loop()
                    success, output_dir, error = await loop.run_in_executor(
                        executor,
                        pipeline_process,
                        job_id_str,
                        job_lecture_str,
                        job_input_path,
                        pipeline_job_type,
                        job_uploaded_at,
                        job_title,
                        worker_index,
                        worker_count,
                    )
                except concurrent.futures.process.BrokenProcessPool as bp_err:
                    logger.error(f"--- [Worker EXECUTOR BROKEN] {job_id_str}: {bp_err} ---")
                    error = f"파이프라인 프로세스 강제 종료 (메모리 부족 등): {str(bp_err)}"
                    success = False
                    output_dir = None
                    # 손상된 Executor 재시작
                    executor.shutdown(wait=False)
                    executor = ProcessPoolExecutor(max_workers=1, mp_context=mp_context)
                    logger.info("--- [Worker] Executor restarted ---")
                except Exception as exec_err:
                    logger.error(f"--- [Worker EXECUTOR ERROR] {job_id_str}: {exec_err} ---")
                    error = f"시스템/프로세스 오류: {str(exec_err)}"
                    success = False
                    output_dir = None

                logger.info(f"--- [Worker] Pipeline done: {job_id_str} success={success} ---")

                async with AsyncSessionLocal() as db:
                    if success:
                        canonical_job_type = normalize_job_type(job_type_str)
                        final_status = JOB_STATUS_DONE
                        final_stage = (
                            "검증 결과가 준비되었습니다."
                            if canonical_job_type == JOB_TYPE_VERIFY
                            else "분석이 완료되었습니다."
                        )
                        await db.execute(text("""
                            UPDATE processing_jobs
                            SET status = :status, current_stage = :stage
                            WHERE id = :id
                        """), {"id": job_id_val, "status": final_status, "stage": final_stage})
                        if canonical_job_type in {JOB_TYPE_PUBLISH, JOB_TYPE_LEGACY_FULL}:
                            await db.execute(text("""
                                UPDATE lectures
                                SET is_published = TRUE
                                WHERE id = :lecture_id
                            """), {"lecture_id": job_lecture_id})
                    else:
                        logger.error(f"--- [Worker ERROR] {job_id_str}: {error} ---")
                        await db.execute(text("""
                            UPDATE processing_jobs
                            SET status = 'error', error_message = :err, current_stage = '분석 중 오류가 발생했습니다.'
                            WHERE id = :id
                        """), {"id": job_id_val, "err": error})
                    await db.commit()

            except asyncio.CancelledError:
                raise  # 바깥 try의 CancelledError 처리로 전달
            except Exception as e:
                logger.error(f"--- [Worker FATAL ERROR in loop]: {e} ---")
                # traceback.print_exc() 는 logger.error(..., exc_info=True)로 대체 가능
                logger.error(traceback.format_exc())
                await asyncio.sleep(5)

    except asyncio.CancelledError:
        logger.info("--- [Worker] Shutdown signal received. ---")
    except Exception as fatal_e:
        logger.error(f"--- [Worker FATAL ERROR ON STARTUP]: {fatal_e} ---")
        logger.error(traceback.format_exc())
    finally:
        try:
            executor.shutdown(wait=True)
        except:
            pass
        logger.info("--- [Worker] Executor shut down. ---")

if __name__ == "__main__":
    asyncio.run(worker_loop())
    
