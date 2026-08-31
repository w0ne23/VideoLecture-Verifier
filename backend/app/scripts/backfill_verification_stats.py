"""기존에 완료된 verify 강의들을 verification_stats 에 소급 적재한다.

    docker compose exec backend python -m app.scripts.backfill_verification_stats

- job_type='verify' + status='done' 인 강의만 대상 (verify_only 제외).
- 결과 JSON 파일이 없으면 skip.
- 멱등: record_verification_stats 가 lecture 당 기존 행을 지우고 다시 넣는다.
"""

import asyncio
import logging

from sqlalchemy import select

from app.db import AsyncSessionLocal
from app.models import JOB_STATUS_DONE, JOB_TYPE_VERIFY, Lecture, ProcessingJob
from app.services import stats_service

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger('backfill')


async def main() -> None:
    inserted = skipped = 0
    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                select(Lecture.id, Lecture.title, ProcessingJob.id)
                .join(ProcessingJob, ProcessingJob.lecture_id == Lecture.id)
                .where(
                    ProcessingJob.job_type == JOB_TYPE_VERIFY,
                    ProcessingJob.status == JOB_STATUS_DONE,
                )
                .order_by(Lecture.id, ProcessingJob.created_at)
            )
        ).all()

        # 강의당 마지막 verify job 만 남긴다.
        latest_job: dict = {}
        titles: dict = {}
        for lecture_id, title, job_id in rows:
            latest_job[lecture_id] = job_id
            titles[lecture_id] = title

        for lecture_id, job_id in latest_job.items():
            ok = await stats_service.record_verification_stats(db, lecture_id, job_id)
            if ok:
                await db.commit()
                inserted += 1
                logger.info('OK   %s  (%s)', lecture_id, titles.get(lecture_id))
            else:
                skipped += 1
                logger.info('SKIP %s  (결과 파일 없음)', lecture_id)

    logger.info('완료: %d개 적재, %d개 skip', inserted, skipped)


if __name__ == '__main__':
    asyncio.run(main())
