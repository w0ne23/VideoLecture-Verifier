"""검증 결과 API 응답 스키마 계약 테스트.

프론트엔드(frontend/src/components/verifier/VerifierResults.jsx 등)가 소비하는
키 구성을 고정한다. 이 테스트가 깨지면 프론트도 함께 수정해야 한다는 신호다.
"""
import json
import os
import urllib.request

import pytest

from app.services.lecture_service import build_content_verification_response

# 프론트가 실제로 읽는 키들 (키 이름, 기대 타입)
CONTRACT = {
    'lecture_id': str,
    'stem': str,
    'schema_version': (str, type(None)),
    'verification_date': str,
    'models': list,
    'counts': dict,
    'final_confirmed_claims': list,
    'needs_review_claims': list,
    'verifier_rejected_claims': list,
    'slide_errors': list,
    'slide_error_status': str,
    'summary': dict,
}
COUNT_KEYS = {
    'final_confirmed', 'needs_review', 'rejected',
    'slide_errors', 'slide_error_needs_review', 'verifier_rejected',
}


def _assert_contract(payload: dict):
    for key, expected_type in CONTRACT.items():
        assert key in payload, f'계약 키 누락: {key}'
        assert isinstance(payload[key], expected_type), f'{key} 타입 불일치: {type(payload[key])}'
    assert COUNT_KEYS <= set(payload['counts']), f"counts 키 누락: {COUNT_KEYS - set(payload['counts'])}"
    for key in COUNT_KEYS:
        assert isinstance(payload['counts'][key], int)


def test_mapper_contract_with_full_fixture():
    raw = {
        'schema_version': 'content_verification.v2',
        'verification_date': '2026-07-07T00:00:00+09:00',
        'models': ['gpt', 'claude'],
        'claim_decision_flow': {
            'final_confirmed_claims': [{'claim': 'A'}],
            'needs_review_claims': [{'claim': 'B'}],
            'verifier_rejected_claims': [{'claim': 'C'}, {'claim': 'D'}],
        },
        'summary': {
            'confirmed_feedback_count': 1,
            'review_needed_feedback_count': 1,
            'rejected_feedback_count': 2,
            'slide_error_count': 0,
        },
        'slide_errors': [],
        'slide_error_status': 'ok',
    }
    payload = build_content_verification_response('lec-1', 'stem-1', '/tmp/x.json', raw)
    _assert_contract(payload)
    assert payload['counts']['final_confirmed'] == 1
    assert payload['counts']['rejected'] == 2
    assert len(payload['final_confirmed_claims']) == 1


def test_mapper_contract_with_minimal_fixture():
    """요약/플로우가 없는 결과 파일도 리스트 길이로 counts를 채워야 한다."""
    raw = {
        'final_confirmed_claims': [{'claim': 'A'}, {'claim': 'B'}],
        'needs_review_claims': [],
        'verifier_rejected_claims': [{'claim': 'C'}],
        'slide_errors': [{'slide': 1}],
    }
    payload = build_content_verification_response('lec-2', 'stem-2', '/tmp/y.json', raw)
    _assert_contract(payload)
    assert payload['counts']['final_confirmed'] == 2
    assert payload['counts']['verifier_rejected'] == 1
    assert payload['counts']['slide_errors'] == 1


def test_mapper_contract_with_empty_fixture():
    payload = build_content_verification_response('lec-3', 'stem-3', '/tmp/z.json', {})
    _assert_contract(payload)
    assert all(value == 0 for value in payload['counts'].values())


# ── 통합: 실행 중인 백엔드의 실제 응답도 같은 계약을 지키는지 ──────────────

BASE = os.getenv('VERILEC_API_BASE', 'http://localhost:8010')


def _live_verifier_payload():
    try:
        with urllib.request.urlopen(BASE + '/jobs', timeout=3) as res:
            lectures = json.loads(res.read().decode('utf-8'))
        done = next((row for row in lectures if row.get('status') == 'done'), None)
        if not done:
            return None
        with urllib.request.urlopen(f"{BASE}/results/{done['id']}/verifier", timeout=5) as res:
            return json.loads(res.read().decode('utf-8'))
    except Exception:
        return None


def test_live_endpoint_matches_contract():
    payload = _live_verifier_payload()
    if payload is None:
        pytest.skip('백엔드 미기동 또는 완료된 강의 없음')
    _assert_contract(payload)
