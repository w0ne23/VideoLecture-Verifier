import asyncio
import logging
from contextlib import asynccontextmanager

from app.config import BACKEND_WORKERS, LOCAL_STORAGE_DIR
from app.db import init_db
from app.worker import worker_loop

logger = logging.getLogger(__name__)
worker_tasks = []


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


@asynccontextmanager
async def lifespan(app):
    global worker_tasks
    logger.info('--- [FastAPI] Starting lifespan events... ---')
    (LOCAL_STORAGE_DIR / 'inputs').mkdir(parents=True, exist_ok=True)
    (LOCAL_STORAGE_DIR / 'results').mkdir(parents=True, exist_ok=True)
    await init_db()

    worker_tasks = [asyncio.create_task(worker_loop(index, BACKEND_WORKERS)) for index in range(BACKEND_WORKERS)]
    monitor_task = asyncio.create_task(monitor_workers())
    logger.info('--- [FastAPI] %s worker task(s) started. ---', len(worker_tasks))

    yield

    logger.info('--- [FastAPI] Shutting down... ---')
    monitor_task.cancel()
    for task in worker_tasks:
        task.cancel()
    await asyncio.gather(monitor_task, *worker_tasks, return_exceptions=True)
