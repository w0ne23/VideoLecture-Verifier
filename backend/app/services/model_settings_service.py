# 모델 설정(현재 적용 값)과 프로필(저장된 프리셋) 관리
import asyncio
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import DATABASE_URL
from app.llm_config import normalize_llm_config
from app.models import ModelSettingProfile, ModelSettings
from app.services.credential_service import load_runtime_credentials

SETTINGS_ID = 1

# 검증 파이프라인 각 스테이지가 모델을 고르는 env var 목록
# DB에서 온 값을 os.environ에 주입하기 전에 이 목록으로 필터링, 설정 테이블에 임의의 키가 들어와도
# 예상 밖의 env var가 파이프라인 프로세스에 주입되지 않도록 차단
STAGE_MODEL_ENV_KEYS = (
    'VERIFIER_CLAIM_EXTRACT_MODEL',
    'ISSUE_JUDGE_MODELS',
    'ISSUE_TYPE_CLASSIFIER_MODELS',
    'CLASSIFIED_ISSUE_VERIFIER_MODELS',
    'CLASSIFIED_ISSUE_GROUNDING_MODELS',
    'CLASSIFIED_ISSUE_EVIDENCE_ENABLED',
    'VERIFIER_SLIDE_ERROR_MODEL',
    'VERIFIER_SLIDE_ERROR_TRANSCRIBE_MODEL',
)


# stage_models 원본에서 허용된 env var 키만 골라 값이 있는 항목만 통과
def _sanitize_stage_models(raw: dict | None) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    cleaned = {}
    for key in STAGE_MODEL_ENV_KEYS:
        value = str(raw.get(key) or '').strip()
        if value:
            cleaned[key] = value
    return cleaned


# editor_state는 dict 형태만 통과, 아니면 빈 dict
def _sanitize_editor_state(raw: dict | None) -> dict:
    return raw if isinstance(raw, dict) else {}


# 프로필 row를 API 응답용 dict로 직렬화
def _serialize_profile(row: ModelSettingProfile) -> dict:
    return {
        'id': str(row.id),
        'name': row.name,
        'stage_models': _sanitize_stage_models(row.stage_models),
        'llm_config': normalize_llm_config(row.llm_config),
        'editor_state': _sanitize_editor_state(row.editor_state),
        'is_active': bool(row.is_active),
        'last_used_at': row.last_used_at.isoformat() if row.last_used_at else None,
        'created_at': row.created_at.isoformat() if row.created_at else None,
        'updated_at': row.updated_at.isoformat() if row.updated_at else None,
    }


# 현재 활성화된 프로필 row 조회, 여러 개면 최근 사용/수정 순으로 하나 선택
async def _get_active_profile_row(db: AsyncSession) -> ModelSettingProfile | None:
    result = await db.execute(
        select(ModelSettingProfile)
        .where(ModelSettingProfile.is_active.is_(True))
        .order_by(ModelSettingProfile.last_used_at.desc().nullslast(), ModelSettingProfile.updated_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# 프로필 도입 이전 방식(단일 ModelSettings 행)의 stage_models 조회
async def _get_legacy_settings(db: AsyncSession) -> dict[str, str]:
    row = await db.get(ModelSettings, SETTINGS_ID)
    if not row:
        return {}
    return _sanitize_stage_models(row.stage_models)


# 프로필 도입 이전 방식의 llm_config 조회
async def _get_legacy_llm_config(db: AsyncSession) -> dict:
    row = await db.get(ModelSettings, SETTINGS_ID)
    return normalize_llm_config(row.llm_config if row else {})


# 활성 프로필이 있으면 그 값, 없으면 레거시 단일 설정 반환
async def get_runtime_model_settings(db: AsyncSession) -> dict:
    active = await _get_active_profile_row(db)
    if active:
        return {
            'stage_models': _sanitize_stage_models(active.stage_models),
            'llm_config': normalize_llm_config(active.llm_config),
        }
    return {
        'stage_models': await _get_legacy_settings(db),
        'llm_config': await _get_legacy_llm_config(db),
    }


# 설정에 복호화된 credential을 더해 반환, 워커 프로세스 내부 전용
async def get_runtime_pipeline_settings(db: AsyncSession) -> dict:
    payload = await get_runtime_model_settings(db)
    refs = {
        str(endpoint.get('credential_ref') or '').strip()
        for endpoint in payload.get('llm_config', {}).get('endpoints', [])
        if isinstance(endpoint, dict)
    }
    payload['credentials'] = await load_runtime_credentials(db, refs)
    return payload


# 프로필 이름 중복 검사, exclude_id는 자기 자신 제외용
async def _ensure_unique_name(db: AsyncSession, name: str, *, exclude_id: UUID | None = None):
    stmt = select(ModelSettingProfile).where(func.lower(ModelSettingProfile.name) == name.lower())
    if exclude_id is not None:
        stmt = stmt.where(ModelSettingProfile.id != exclude_id)
    existing = (await db.execute(stmt.limit(1))).scalar_one_or_none()
    if existing:
        raise ValueError('이미 사용 중인 프리셋 이름입니다.')


# 현재 적용 중인 stage_models 조회 (활성 프로필 우선)
async def get_model_settings(db: AsyncSession) -> dict[str, str]:
    active = await _get_active_profile_row(db)
    if active:
        return _sanitize_stage_models(active.stage_models)
    return await _get_legacy_settings(db)


# 현재 적용 중인 모델 설정 갱신, 활성 프로필이 있으면 프로필에 없으면 레거시 행에 반영
async def update_model_settings(db: AsyncSession, stage_models: dict, llm_config: dict | None = None) -> dict[str, str]:
    cleaned = _sanitize_stage_models(stage_models)
    cleaned_config = normalize_llm_config(llm_config)
    active = await _get_active_profile_row(db)
    if active:
        active.stage_models = cleaned
        active.llm_config = cleaned_config
    else:
        row = await db.get(ModelSettings, SETTINGS_ID)
        if row:
            row.stage_models = cleaned
            row.llm_config = cleaned_config
        else:
            db.add(ModelSettings(id=SETTINGS_ID, stage_models=cleaned, llm_config=cleaned_config))
    await db.commit()
    return cleaned


# 전체 프로필 목록과 활성 프로필 id 조회
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


# 프로필 단건 조회
async def get_profile(db: AsyncSession, profile_id: UUID | str) -> dict | None:
    row = await db.get(ModelSettingProfile, profile_id)
    return _serialize_profile(row) if row else None


# 새 프로필 생성, 이름 중복 시 예외
async def create_profile(
    db: AsyncSession,
    *,
    name: str,
    stage_models: dict,
    llm_config: dict | None = None,
    editor_state: dict | None = None,
) -> dict:
    clean_name = (name or '').strip()
    if not clean_name:
        raise ValueError('프리셋 이름을 입력해주세요.')
    await _ensure_unique_name(db, clean_name)

    row = ModelSettingProfile(
        name=clean_name,
        stage_models=_sanitize_stage_models(stage_models),
        llm_config=normalize_llm_config(llm_config),
        editor_state=_sanitize_editor_state(editor_state),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _serialize_profile(row)


# 프로필 수정, 이름 중복 시 예외
async def update_profile(
    db: AsyncSession,
    profile_id: UUID | str,
    *,
    name: str,
    stage_models: dict,
    llm_config: dict | None = None,
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
    row.llm_config = normalize_llm_config(llm_config)
    row.editor_state = _sanitize_editor_state(editor_state)
    await db.commit()
    await db.refresh(row)
    return _serialize_profile(row)


# 지정 프로필을 활성화하고 나머지는 비활성화, 대상 프로필의 last_used_at 갱신
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


# 프로필 삭제, 활성 프로필이었으면 최근 사용 순으로 다음 프로필을 자동 활성화
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


# get_model_settings의 동기 래퍼, 파이프라인 자식 프로세스에서 사용
def fetch_stage_models_sync() -> dict[str, str]:
    # update_job_stage_sync와 동일한 이유로 NullPool 사용: 파이프라인 자식 프로세스에서
    # asyncio.run() 호출마다 이벤트 루프가 새로 생겨 루프에 묶인 풀 커넥션 재사용 불가
    async def _run() -> dict[str, str]:
        engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
        try:
            session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            async with session_factory() as db:
                return await get_model_settings(db)
        finally:
            await engine.dispose()

    return asyncio.run(_run())


def fetch_runtime_model_settings_sync() -> dict:
    """워커 프로세스에서 동기적으로 전체 endpoint 설정을 조회하는 진입점"""
    async def _run() -> dict:
        engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
        try:
            session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            async with session_factory() as db:
                return await get_runtime_pipeline_settings(db)
        finally:
            await engine.dispose()

    return asyncio.run(_run())
