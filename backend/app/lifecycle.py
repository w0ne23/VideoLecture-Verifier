# FastAPI 앱 시작/종료 시 DB 초기화와 백그라운드 워커 태스크를 관리
import asyncio
import logging
from contextlib import asynccontextmanager

from app.config import BACKEND_WORKERS, LOCAL_STORAGE_DIR
from app.db import init_db
from app.worker import worker_loop

logger = logging.getLogger(__name__)
worker_tasks = []


# 워커 태스크 생존 여부를 5초 간격으로 로깅하는 백그라운드 모니터
async def monitor_workers():
    while True:
        try:
            await asyncio.sleep(5)
            if worker_tasks:
                active = len([task for task in worker_tasks if not task.done()])
                logger.info('--- [Backend Heartbeat] Active Workers: %s/%s ---', active, len(worker_tasks))
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error('--- [Monitor Error]: %s ---', exc)


# 앱 시작 시 저장 경로 준비, DB 초기화, 워커 태스크 기동 / 종료 시 태스크 정리
@asynccontextmanager
async def lifespan(app):
    global worker_tasks
    logger.info('--- [FastAPI] Starting lifespan events... ---')
    # 업로드/결과 저장 디렉토리 사전 생성
    (LOCAL_STORAGE_DIR / 'inputs').mkdir(parents=True, exist_ok=True)
    (LOCAL_STORAGE_DIR / 'results').mkdir(parents=True, exist_ok=True)
    await init_db()

    # 설정된 워커 개수만큼 파이프라인 처리 루프 기동
    worker_tasks = [asyncio.create_task(worker_loop(index, BACKEND_WORKERS)) for index in range(BACKEND_WORKERS)]
    monitor_task = asyncio.create_task(monitor_workers())
    logger.info('--- [FastAPI] %s worker task(s) started. ---', len(worker_tasks))

    yield

    # 앱 종료 시 모니터와 워커 태스크를 모두 취소하고 정리 대기
    logger.info('--- [FastAPI] Shutting down... ---')
    monitor_task.cancel()
    for task in worker_tasks:
        task.cancel()
    await asyncio.gather(monitor_task, *worker_tasks, return_exceptions=True)
