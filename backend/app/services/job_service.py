"""
job_service.py

spawn된 자식 프로세스(ProcessPoolExecutor)에서 파이프라인 진행 상황을 DB에 기록하기 위한 모듈.
spawn된 자식 프로세스는 부모의 이벤트 루프를 상속받지 않아 asyncpg를 사용할 수 없으므로
파이프라인 진행 상황을 DB에 기록하기 위해 psycopg2(동기 드라이버)를 사용한다.

향후 job 생성/상태 변경 등 job 관련 DB 접근 로직이 추가될 예정.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

import psycopg2
from psycopg2.extras import Json

logger = logging.getLogger(__name__)

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
root_env = PROJECT_ROOT_DIR / ".env"

if root_env.exists():
    load_dotenv(dotenv_path=root_env)
else:
    load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# asyncpg URL을 psycopg2(동기) URL로 변환한다.
DATABASE_URL_SYNC = DATABASE_URL.replace("+asyncpg", "") if DATABASE_URL else ""


def update_job_stage_sync(job_id: str, current_stages: list, current_stage_text: str):
    """파이프라인 진행 단계를 동기적으로 DB에 기록한다.

    spawn된 자식 프로세스에서 호출되므로 psycopg2(동기 드라이버)를 사용한다.
    """
    if not DATABASE_URL_SYNC:
        return
    try:
        conn = psycopg2.connect(DATABASE_URL_SYNC)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE processing_jobs SET pipeline_stages = %s, current_stage = %s WHERE id = %s",
                    (Json(current_stages), current_stage_text, job_id),
                )
        conn.close()
    except Exception as e:
        logger.error(f"--- [Worker Sync DB Error] Failed to update stage: {e} ---")
