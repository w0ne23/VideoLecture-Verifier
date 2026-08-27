from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services import model_settings_service

router = APIRouter()


class ModelSettingsIn(BaseModel):
    stage_models: dict[str, str] = {}
    llm_config: dict = {}


class ModelSettingProfileIn(BaseModel):
    name: str
    stage_models: dict[str, str] = {}
    llm_config: dict = {}
    editor_state: dict = {}


@router.get('/admin/model-settings')
async def get_model_settings(db: AsyncSession = Depends(get_db)):
    payload = await model_settings_service.get_runtime_model_settings(db)
    return {**payload, 'mode': 'generic' if payload['stage_models'] else 'fixed'}


@router.put('/admin/model-settings')
async def update_model_settings(payload: ModelSettingsIn, db: AsyncSession = Depends(get_db)):
    stage_models = await model_settings_service.update_model_settings(db, payload.stage_models, payload.llm_config)
    saved = await model_settings_service.get_runtime_model_settings(db)
    return {**saved, 'mode': 'generic' if stage_models else 'fixed'}


@router.get('/admin/model-settings/profiles')
async def list_profiles(db: AsyncSession = Depends(get_db)):
    return await model_settings_service.list_profiles(db)


@router.get('/admin/model-settings/profiles/{profile_id}')
async def get_profile(profile_id: str, db: AsyncSession = Depends(get_db)):
    profile = await model_settings_service.get_profile(db, profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail='프리셋을 찾을 수 없습니다.')
    return profile


@router.post('/admin/model-settings/profiles')
async def create_profile(payload: ModelSettingProfileIn, db: AsyncSession = Depends(get_db)):
    try:
        return await model_settings_service.create_profile(
            db,
            name=payload.name,
            stage_models=payload.stage_models,
            llm_config=payload.llm_config,
            editor_state=payload.editor_state,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put('/admin/model-settings/profiles/{profile_id}')
async def update_profile(profile_id: str, payload: ModelSettingProfileIn, db: AsyncSession = Depends(get_db)):
    try:
        return await model_settings_service.update_profile(
            db,
            profile_id,
            name=payload.name,
            stage_models=payload.stage_models,
            llm_config=payload.llm_config,
            editor_state=payload.editor_state,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete('/admin/model-settings/profiles/{profile_id}')
async def delete_profile(profile_id: str, db: AsyncSession = Depends(get_db)):
    try:
        await model_settings_service.delete_profile(db, profile_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {'ok': True}


@router.post('/admin/model-settings/profiles/{profile_id}/apply')
async def apply_profile(profile_id: str, db: AsyncSession = Depends(get_db)):
    try:
        return await model_settings_service.apply_profile(db, profile_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
