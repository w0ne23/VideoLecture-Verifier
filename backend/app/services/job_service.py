import asyncio
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import AsyncSessionLocal
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
    async def _run():
        async with AsyncSessionLocal() as db:
            await update_job_stage(db, job_id, stages, current_stage)

    asyncio.run(_run())
