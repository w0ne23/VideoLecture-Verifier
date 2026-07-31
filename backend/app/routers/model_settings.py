from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services import model_settings_service

router = APIRouter()


class ModelSettingsIn(BaseModel):
    stage_models: dict[str, str] = {}


@router.get('/admin/model-settings')
async def get_model_settings(db: AsyncSession = Depends(get_db)):
    stage_models = await model_settings_service.get_model_settings(db)
    return {'stage_models': stage_models, 'mode': 'generic' if stage_models else 'fixed'}


@router.put('/admin/model-settings')
async def update_model_settings(payload: ModelSettingsIn, db: AsyncSession = Depends(get_db)):
    stage_models = await model_settings_service.update_model_settings(db, payload.stage_models)
    return {'stage_models': stage_models, 'mode': 'generic' if stage_models else 'fixed'}
