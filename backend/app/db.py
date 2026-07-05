import os
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.models import Base, JOB_TYPE_LEGACY_FULL

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@db/graphlec")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    """DB 초기화: 모든 모델 테이블 생성"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text(
            "ALTER TABLE processing_jobs "
            "ADD COLUMN IF NOT EXISTS job_type VARCHAR NOT NULL "
            f"DEFAULT '{JOB_TYPE_LEGACY_FULL}'"
        ))
        await conn.execute(text(
            "ALTER TABLE lectures "
            "ADD COLUMN IF NOT EXISTS is_verified BOOLEAN NOT NULL DEFAULT FALSE"
        ))
        await conn.execute(text(
            "ALTER TABLE lectures "
            "ADD COLUMN IF NOT EXISTS is_published BOOLEAN NOT NULL DEFAULT FALSE"
        ))
        await conn.execute(text(
            "UPDATE lectures "
            "SET is_verified = TRUE "
            "WHERE id IN ("
            "  SELECT lecture_id FROM processing_jobs "
            "  WHERE ("
            "    job_type = 'legacy_full' AND status = 'done'"
            "  ) OR ("
            "    job_type = 'verified_upload' AND status IN ('done', 'waiting_approval')"
            "  )"
            ")"
        ))
        await conn.execute(text(
            "UPDATE lectures "
            "SET is_published = TRUE "
            "WHERE id IN ("
            "  SELECT lecture_id FROM processing_jobs "
            "  WHERE job_type IN ('legacy_full', 'direct_upload', 'graph_upload', 'upload', 'publish') "
            "  AND status = 'done'"
            ")"
        ))
    logger.info("--- [DB] Database initialized via SQLAlchemy models. ---")

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
