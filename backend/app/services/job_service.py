import asyncio
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import DATABASE_URL
from app.models import ProcessingJob


async def get_job(db: AsyncSession, job_id: str) -> ProcessingJob | None:
    try:
        ident = uuid.UUID(str(job_id))
    except (TypeError, ValueError):
        return None
    result = await db.execute(select(ProcessingJob).where(ProcessingJob.id == ident))
    return result.scalar_one_or_none()


async def update_job_stage(db: AsyncSession, job_id: str, stages: list[dict], current_stage: str) -> None:
    job = await get_job(db, job_id)
    if not job:
        return
    job.pipeline_stages = stages
    job.current_stage = current_stage
    await db.commit()


def update_job_stage_sync(job_id: str, stages: list[dict], current_stage: str) -> None:
    # 파이프라인 자식 프로세스에서 asyncio.run()으로 호출될 때마다 이벤트 루프가
    # 새로 생기므로, 루프에 묶이는 풀 커넥션을 재사용하면 안 된다 (asyncpg
    # InterfaceError). 매번 NullPool 엔진을 만들고 즉시 정리한다.
    async def _run():
        engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
        try:
            session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            async with session_factory() as db:
                await update_job_stage(db, job_id, stages, current_stage)
        finally:
            await engine.dispose()

    asyncio.run(_run())
