from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services import stats_service

router = APIRouter(prefix='/stats')


@router.get('')
async def get_stats(db: AsyncSession = Depends(get_db)):
    """통계 페이지용 집계. verification_stats(완료된 verify 실행)만 반영한다.

    강의가 하나도 없으면 lecture_count=0 과 빈 배열들을 반환한다.
    """
    return await stats_service.aggregate(db)
