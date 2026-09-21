# 강의자가 검증 오류 항목별로 남기는 동의/중립/비동의 평가 CRUD
import uuid

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from app.models import REVIEW_RATINGS, InstructorReview, Lecture


def _lecture_uuid(lecture_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(lecture_id))
    except (TypeError, ValueError):
        raise HTTPException(status_code=404, detail='Lecture not found')


# lecture의 저장된 평가 전체를 { item_id: rating } 맵으로 반환
async def list_reviews(db: AsyncSession, lecture_id: str) -> dict[str, str]:
    ident = _lecture_uuid(lecture_id)
    result = await db.execute(
        select(InstructorReview.item_id, InstructorReview.rating).where(InstructorReview.lecture_id == ident)
    )
    return {item_id: rating for item_id, rating in result.all()}


# 항목 하나의 평가를 생성/갱신 (upsert), 이미 있으면 rating·updated_at만 갱신
async def upsert_review(db: AsyncSession, lecture_id: str, item_id: str, rating: str) -> dict:
    if rating not in REVIEW_RATINGS:
        raise HTTPException(status_code=400, detail=f'Invalid rating: {rating}')
    ident = _lecture_uuid(lecture_id)
    if not await db.get(Lecture, ident):
        raise HTTPException(status_code=404, detail='Lecture not found')

    stmt = pg_insert(InstructorReview).values(
        lecture_id=ident,
        item_id=item_id,
        rating=rating,
    )
    stmt = stmt.on_conflict_do_update(
        constraint='uq_instructor_review_lecture_item',
        set_={'rating': stmt.excluded.rating, 'updated_at': func.now()},
    )
    await db.execute(stmt)
    await db.commit()
    return {'item_id': item_id, 'rating': rating}
