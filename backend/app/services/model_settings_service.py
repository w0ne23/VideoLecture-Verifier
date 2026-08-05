import asyncio
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import DATABASE_URL
from app.models import ModelSettingProfile, ModelSettings

SETTINGS_ID = 1

# 검증 파이프라인 각 스테이지가 모델을 고르는 env var. DB에서 온 값을 그대로
# os.environ에 주입하기 전에 이 목록으로 걸러서, 설정 테이블에 임의의 키가 들어와도
# 예상 밖의 env var가 파이프라인 프로세스에 주입되지 않게 한다.
STAGE_MODEL_ENV_KEYS = (
    'VERIFIER_CLAIM_EXTRACT_MODEL',
    'ISSUE_JUDGE_MODELS',
    'ISSUE_TYPE_CLASSIFIER_MODELS',
    'CLASSIFIED_ISSUE_VERIFIER_MODELS',
    'CLASSIFIED_ISSUE_GROUNDING_MODELS',
    'VERIFIER_SLIDE_ERROR_MODEL',
    'VERIFIER_SLIDE_ERROR_TRANSCRIBE_MODEL',
)


def _sanitize_stage_models(raw: dict | None) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    cleaned = {}
    for key in STAGE_MODEL_ENV_KEYS:
        value = str(raw.get(key) or '').strip()
        if value:
            cleaned[key] = value
    return cleaned


def _sanitize_editor_state(raw: dict | None) -> dict:
    return raw if isinstance(raw, dict) else {}


def _serialize_profile(row: ModelSettingProfile) -> dict:
    return {
        'id': str(row.id),
        'name': row.name,
        'stage_models': _sanitize_stage_models(row.stage_models),
        'editor_state': _sanitize_editor_state(row.editor_state),
        'is_active': bool(row.is_active),
        'last_used_at': row.last_used_at.isoformat() if row.last_used_at else None,
        'created_at': row.created_at.isoformat() if row.created_at else None,
        'updated_at': row.updated_at.isoformat() if row.updated_at else None,
    }


async def _get_active_profile_row(db: AsyncSession) -> ModelSettingProfile | None:
    result = await db.execute(
        select(ModelSettingProfile)
        .where(ModelSettingProfile.is_active.is_(True))
        .order_by(ModelSettingProfile.last_used_at.desc().nullslast(), ModelSettingProfile.updated_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _get_legacy_settings(db: AsyncSession) -> dict[str, str]:
    row = await db.get(ModelSettings, SETTINGS_ID)
    if not row:
        return {}
    return _sanitize_stage_models(row.stage_models)


async def _ensure_unique_name(db: AsyncSession, name: str, *, exclude_id: UUID | None = None):
    stmt = select(ModelSettingProfile).where(func.lower(ModelSettingProfile.name) == name.lower())
    if exclude_id is not None:
        stmt = stmt.where(ModelSettingProfile.id != exclude_id)
    existing = (await db.execute(stmt.limit(1))).scalar_one_or_none()
    if existing:
        raise ValueError('이미 사용 중인 프리셋 이름입니다.')


async def get_model_settings(db: AsyncSession) -> dict[str, str]:
    active = await _get_active_profile_row(db)
    if active:
        return _sanitize_stage_models(active.stage_models)
    return await _get_legacy_settings(db)


async def update_model_settings(db: AsyncSession, stage_models: dict) -> dict[str, str]:
    cleaned = _sanitize_stage_models(stage_models)
    active = await _get_active_profile_row(db)
    if active:
        active.stage_models = cleaned
    else:
        row = await db.get(ModelSettings, SETTINGS_ID)
        if row:
            row.stage_models = cleaned
        else:
            db.add(ModelSettings(id=SETTINGS_ID, stage_models=cleaned))
    await db.commit()
    return cleaned


async def list_profiles(db: AsyncSession) -> dict:
    rows = (
        await db.execute(
            select(ModelSettingProfile)
            .order_by(
                ModelSettingProfile.is_active.desc(),
                ModelSettingProfile.last_used_at.desc().nullslast(),
                ModelSettingProfile.updated_at.desc(),
            )
        )
    ).scalars().all()
    active = next((row for row in rows if row.is_active), None)
    return {
        'profiles': [_serialize_profile(row) for row in rows],
        'active_profile_id': str(active.id) if active else None,
    }


async def get_profile(db: AsyncSession, profile_id: UUID | str) -> dict | None:
    row = await db.get(ModelSettingProfile, profile_id)
    return _serialize_profile(row) if row else None


async def create_profile(db: AsyncSession, *, name: str, stage_models: dict, editor_state: dict | None = None) -> dict:
    clean_name = (name or '').strip()
    if not clean_name:
        raise ValueError('프리셋 이름을 입력해주세요.')
    await _ensure_unique_name(db, clean_name)

    row = ModelSettingProfile(
        name=clean_name,
        stage_models=_sanitize_stage_models(stage_models),
        editor_state=_sanitize_editor_state(editor_state),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _serialize_profile(row)


async def update_profile(
    db: AsyncSession,
    profile_id: UUID | str,
    *,
    name: str,
    stage_models: dict,
    editor_state: dict | None = None,
) -> dict:
    row = await db.get(ModelSettingProfile, profile_id)
    if not row:
        raise LookupError('프리셋을 찾을 수 없습니다.')

    clean_name = (name or '').strip()
    if not clean_name:
        raise ValueError('프리셋 이름을 입력해주세요.')
    await _ensure_unique_name(db, clean_name, exclude_id=row.id)

    row.name = clean_name
    row.stage_models = _sanitize_stage_models(stage_models)
    row.editor_state = _sanitize_editor_state(editor_state)
    await db.commit()
    await db.refresh(row)
    return _serialize_profile(row)


async def apply_profile(db: AsyncSession, profile_id: UUID | str) -> dict:
    rows = (await db.execute(select(ModelSettingProfile))).scalars().all()
    target = next((row for row in rows if str(row.id) == str(profile_id)), None)
    if not target:
        raise LookupError('프리셋을 찾을 수 없습니다.')

    now = datetime.now(timezone.utc)
    for row in rows:
        row.is_active = row.id == target.id
        if row.id == target.id:
            row.last_used_at = now
    await db.commit()
    await db.refresh(target)
    return _serialize_profile(target)


async def delete_profile(db: AsyncSession, profile_id: UUID | str):
    row = await db.get(ModelSettingProfile, profile_id)
    if not row:
        raise LookupError('프리셋을 찾을 수 없습니다.')

    was_active = bool(row.is_active)
    await db.delete(row)
    await db.commit()

    if was_active:
        rows = (
            await db.execute(
                select(ModelSettingProfile).order_by(
                    ModelSettingProfile.last_used_at.desc().nullslast(),
                    ModelSettingProfile.updated_at.desc(),
                )
            )
        ).scalars().all()
        if rows:
            rows[0].is_active = True
            await db.commit()


def fetch_stage_models_sync() -> dict[str, str]:
    # update_job_stage_sync와 동일한 이유로 NullPool을 쓴다: 파이프라인 자식
    # 프로세스에서 asyncio.run()을 쓸 때마다 이벤트 루프가 새로 생기므로, 루프에
    # 묶이는 풀 커넥션을 재사용하면 안 된다.
    async def _run() -> dict[str, str]:
        engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
        try:
            session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            async with session_factory() as db:
                return await get_model_settings(db)
        finally:
            await engine.dispose()

    return asyncio.run(_run())
