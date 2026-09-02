# 비동기 DB 엔진, 세션 팩토리, 초기화 로직
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from app.config import DATABASE_URL
from app.models import Base

logger = logging.getLogger(__name__)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# 기존 DB에는 create_all이 컬럼을 추가하지 않아 IF NOT EXISTS로 컬럼 보정
async def _ensure_lecture_source_tag(conn):
    await conn.execute(text(
        "ALTER TABLE lectures ADD COLUMN IF NOT EXISTS source_tag VARCHAR"
    ))


# config API 도입 이전에 생성된 DB를 위한 llm_config 컬럼 보정
async def _ensure_llm_config_columns(conn):
    await conn.execute(text(
        "ALTER TABLE model_settings ADD COLUMN IF NOT EXISTS llm_config JSONB NOT NULL DEFAULT '{}'::jsonb"
    ))
    await conn.execute(text(
        "ALTER TABLE model_setting_profiles ADD COLUMN IF NOT EXISTS llm_config JSONB NOT NULL DEFAULT '{}'::jsonb"
    ))


# 테이블 생성과 컬럼 보정을 포함한 DB 초기화
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_lecture_source_tag(conn)
        await _ensure_llm_config_columns(conn)
    logger.info('--- [DB] Database initialized. ---')


# 요청 단위 DB 세션을 생성하는 FastAPI 의존성
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
