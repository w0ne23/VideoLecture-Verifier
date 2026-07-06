from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services import lecture_service

router = APIRouter(prefix='/lectures')


@router.get('/{lecture_id}')
async def get_lecture(lecture_id: str, db: AsyncSession = Depends(get_db)):
    detail = await lecture_service.get_lecture_detail(db, lecture_id)
    if not detail:
        raise HTTPException(status_code=404, detail='Lecture not found')
    return detail


@router.get('/{lecture_id}/verifier')
async def get_verifier_result(lecture_id: str, db: AsyncSession = Depends(get_db)):
    return await lecture_service.get_content_verification(db, lecture_id)


@router.post('/{lecture_id}/verify/confirm')
async def confirm_verified_lecture(lecture_id: str, db: AsyncSession = Depends(get_db)):
    result = await lecture_service.confirm_verified_lecture(db, lecture_id)
    if not result:
        raise HTTPException(status_code=404, detail='Lecture not found')
    return result
