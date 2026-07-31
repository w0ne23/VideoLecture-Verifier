import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import DATABASE_URL
from app.models import ModelSettings

SETTINGS_ID = 1

# 검증 파이프라인 각 스테이지가 모델을 고르는 env var. DB에서 온 값을 그대로
# os.environ에 주입하기 전에 이 목록으로 걸러서, 설정 테이블에 임의의 키가 들어와도
# 예상 밖의 env var가 파이프라인 프로세스에 주입되지 않게 한다.
STAGE_MODEL_ENV_KEYS = (
    'VERIFIER_CLAIM_EXTRACT_MODEL',
    'ISSUE_JUDGE_MODELS',
    'ISSUE_TYPE_CLASSIFIER_MODELS',
    'CLASSIFIED_ISSUE_VERIFIER_MODELS',
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


async def get_model_settings(db: AsyncSession) -> dict[str, str]:
    row = await db.get(ModelSettings, SETTINGS_ID)
    if not row:
        return {}
    return _sanitize_stage_models(row.stage_models)


async def update_model_settings(db: AsyncSession, stage_models: dict) -> dict[str, str]:
    cleaned = _sanitize_stage_models(stage_models)
    row = await db.get(ModelSettings, SETTINGS_ID)
    if row:
        row.stage_models = cleaned
    else:
        db.add(ModelSettings(id=SETTINGS_ID, stage_models=cleaned))
    await db.commit()
    return cleaned


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
