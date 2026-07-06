from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services import lecture_service

router = APIRouter(prefix='/results')


@router.get('/{lecture_id}/verifier')
async def get_content_verification(lecture_id: str, db: AsyncSession = Depends(get_db)):
    return await lecture_service.get_content_verification(db, lecture_id)


@router.get('/{lecture_id}')
async def get_result_detail(lecture_id: str, db: AsyncSession = Depends(get_db)):
    detail = await lecture_service.get_lecture_detail(db, lecture_id)
    if not detail:
        raise HTTPException(status_code=404, detail='Lecture not found')
    return detail
