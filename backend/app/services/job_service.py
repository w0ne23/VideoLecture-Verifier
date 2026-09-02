# ProcessingJob 상태/스테이지 갱신 유틸
import asyncio
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import DATABASE_URL
from app.models import ProcessingJob


# job_id로 ProcessingJob 조회, 형식이 잘못되면 None
async def get_job(db: AsyncSession, job_id: str) -> ProcessingJob | None:
    try:
        ident = uuid.UUID(str(job_id))
    except (TypeError, ValueError):
        return None
    result = await db.execute(select(ProcessingJob).where(ProcessingJob.id == ident))
    return result.scalar_one_or_none()


# job의 pipeline_stages와 current_stage 갱신
async def update_job_stage(db: AsyncSession, job_id: str, stages: list[dict], current_stage: str) -> None:
    job = await get_job(db, job_id)
    if not job:
        return
    job.pipeline_stages = stages
    job.current_stage = current_stage
    await db.commit()


# update_job_stage의 동기 래퍼, 파이프라인 자식 프로세스에서 사용
def update_job_stage_sync(job_id: str, stages: list[dict], current_stage: str) -> None:
    # asyncio.run() 호출마다 이벤트 루프가 새로 생겨 루프에 묶인 풀 커넥션은 재사용 불가 (asyncpg
    # InterfaceError), 매번 NullPool 엔진을 새로 만들고 즉시 정리
    async def _run():
        engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
        try:
            session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            async with session_factory() as db:
                await update_job_stage(db, job_id, stages, current_stage)
        finally:
            await engine.dispose()

    asyncio.run(_run())
