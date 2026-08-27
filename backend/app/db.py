import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from app.config import DATABASE_URL
from app.models import Base

logger = logging.getLogger(__name__)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def _ensure_lecture_source_tag(conn):
    """기존 DB에는 create_all이 컬럼을 추가하지 않으므로 IF NOT EXISTS로 보정한다."""
    await conn.execute(text(
        "ALTER TABLE lectures ADD COLUMN IF NOT EXISTS source_tag VARCHAR"
    ))


async def _ensure_llm_config_columns(conn):
    """Add endpoint config columns for databases created before the config API."""
    await conn.execute(text(
        "ALTER TABLE model_settings ADD COLUMN IF NOT EXISTS llm_config JSONB NOT NULL DEFAULT '{}'::jsonb"
    ))
    await conn.execute(text(
        "ALTER TABLE model_setting_profiles ADD COLUMN IF NOT EXISTS llm_config JSONB NOT NULL DEFAULT '{}'::jsonb"
    ))


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_lecture_source_tag(conn)
        await _ensure_llm_config_columns(conn)
    logger.info('--- [DB] Database initialized. ---')


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
