from __future__ import annotations

from pydantic import BaseModel, SecretStr
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services import credential_service

router = APIRouter()


class LlmCredentialIn(BaseModel):
    provider: str
    model: str = ""
    api_key: SecretStr


def _response(row, api_key: str) -> dict:
    return {
        'credential_ref': credential_service.credential_ref(row),
        'key_masked': credential_service.mask_secret(api_key),
    }


@router.post('/admin/llm-credentials')
async def create_llm_credential(
    payload: LlmCredentialIn,
    db: AsyncSession = Depends(get_db),
):
    try:
        row = await credential_service.save_credential(
            db,
            provider=payload.provider,
            model=payload.model,
            api_key=payload.api_key.get_secret_value(),
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _response(row, payload.api_key.get_secret_value())
