"""실행 중인 백엔드에 대한 API 스모크 (백엔드 미기동 시 skip).

VERILEC_API_BASE 환경변수로 대상 주소를 바꿀 수 있다 (기본 http://localhost:8010).
"""
import json
import os
import urllib.error
import urllib.request

import pytest

BASE = os.getenv('VERILEC_API_BASE', 'http://localhost:8010')


def _get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=5) as res:
        return res.status, json.loads(res.read().decode('utf-8'))


def _backend_up() -> bool:
    try:
        status, _ = _get('/health')
        return status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _backend_up(), reason=f'백엔드({BASE})가 실행 중이 아님')


def test_health():
    status, body = _get('/health')
    assert status == 200
    assert body == {'status': 'ok'}


def test_jobs_returns_list():
    status, body = _get('/jobs')
    assert status == 200
    assert isinstance(body, list)


def test_unknown_lecture_is_404():
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get('/results/00000000-0000-0000-0000-000000000000')
    assert exc.value.code == 404
