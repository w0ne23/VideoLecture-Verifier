import asyncio
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

from app.db import init_db
from app.worker import worker_loop
from app.services.neo4j_service import get_neo4j_driver, clear_runtime_lecture_graphs

logger = logging.getLogger(__name__)

# Cross-platform path handling for local_storage
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
PROJECT_ROOT = Path("/pipeline") if Path("/pipeline").exists() else PROJECT_ROOT_DIR
LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR", str(PROJECT_ROOT / "local_storage"))

# Global worker task registry for monitoring
worker_tasks = []

def _worker_count() -> int:
    try:
        return max(1, int(os.getenv("GRAPHLEC_BACKEND_WORKERS", "3")))
    except ValueError:
        return 3

async def monitor_workers():
    """Periodically logs the number of active workers."""
    while True:
        try:
            await asyncio.sleep(5)
            if worker_tasks:
                active = len([t for t in worker_tasks if not t.done()])
                total = len(worker_tasks)
                logger.info(f"--- [Backend Heartbeat] Active Workers: {active}/{total} ---")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"--- [Monitor Error]: {e} ---")

@asynccontextmanager
async def lifespan(app):
    global worker_tasks
    logger.info("--- [FastAPI] Starting lifespan events... ---")
    
    # Ensure local storage directory exists
    os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)
    logger.info(f"--- [FastAPI] Local storage directory ensured: {LOCAL_STORAGE_DIR} ---")
    
    try:
        logger.info("--- [FastAPI] Initializing DB... ---")
        await init_db()
        logger.info("--- [FastAPI] DB initialized successfully. ---")
    except Exception as e:
        logger.error(f"--- [FastAPI] ERROR initializing DB: {e} ---")
        
    try:
        logger.info("--- [FastAPI] Initializing Neo4j driver... ---")
        get_neo4j_driver()
        logger.info("--- [FastAPI] Neo4j driver initialized successfully. ---")
    except Exception as e:
        logger.error(f"--- [FastAPI] ERROR initializing Neo4j driver: {e} ---")

    if os.getenv("GRAPHLEC_CLEAR_NEO4J_ON_START", "0").lower() not in {"0", "false", "no"}:
        cleanup = clear_runtime_lecture_graphs()
        logger.info(
            f"--- [FastAPI] Neo4j runtime graph cleanup: before={cleanup['before']} after={cleanup['after']} ---"
        )
    
    logger.info("--- [FastAPI] Starting worker loops... ---")
    count = _worker_count()
    worker_tasks = [asyncio.create_task(worker_loop(i, count)) for i in range(count)]
    monitor_task = asyncio.create_task(monitor_workers())
    logger.info(f"--- [FastAPI] {len(worker_tasks)} worker tasks and monitor created. ---")
    
    yield
    
    logger.info("--- [FastAPI] Shutting down... Cancelling workers. ---")
    monitor_task.cancel()
    for task in worker_tasks:
        task.cancel()
    await asyncio.gather(monitor_task, *worker_tasks, return_exceptions=True)
    logger.info("--- [FastAPI] Shutdown complete. ---")
