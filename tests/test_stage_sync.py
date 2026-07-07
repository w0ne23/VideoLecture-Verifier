"""update_job_stage_sync 회귀 테스트 (DB 필요 — 없으면 skip).

과거 버그: asyncio.run()마다 새 이벤트 루프가 생기는데 전역 엔진의 풀 커넥션이
이전 루프에 묶여 있어 두 번째 호출부터 InterfaceError로 전부 실패했다.
연속 호출이 예외 없이 끝나는지가 회귀 포인트다.
"""
import asyncio

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool


def _db_reachable() -> bool:
    from app.config import DATABASE_URL

    async def ping():
        engine = create_async_engine(DATABASE_URL, poolclass=NullPool)
        try:
            async with engine.connect():
                pass
        finally:
            await engine.dispose()

    try:
        asyncio.run(ping())
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_reachable(), reason='PostgreSQL에 연결할 수 없음')


def test_repeated_sync_updates_across_event_loops():
    from app.services.job_service import update_job_stage_sync

    # 존재하지 않는 job id → SELECT 후 조기 반환. 데이터 변경 없이
    # '루프마다 새 커넥션' 경로만 반복 검증한다.
    for i in range(3):
        update_job_stage_sync(
            '00000000-0000-0000-0000-000000000000',
            [{'stage': 'preprocess_extract_media', 'status': 'run'}],
            f'stage sync 회귀 테스트 {i}',
        )
