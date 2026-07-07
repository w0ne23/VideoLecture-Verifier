"""pipeline.config lazy 클라이언트 초기화 회귀 테스트.

import 시점에 죽지 않고, 필수 키 누락은 명확한 에러, 선택 키는 None을 유지해야 한다.
"""
import pytest

import pipeline.config as config

GEMINI_VARS = ['GOOGLE_API_KEY_1', 'GOOGLE_API_KEY', 'GEMINI_API_KEY', 'GOOGLE_API_KEY_2']
OPTIONAL_VARS = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'XAI_API_KEY', 'DEEPSEEK_API_KEY']


@pytest.fixture(autouse=True)
def clean_client_cache():
    config._client_cache.clear()
    yield
    config._client_cache.clear()


def test_missing_groq_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='GROQ_API_KEY'):
        config.get_groq_client()


def test_missing_gemini_key_raises_clear_error(monkeypatch):
    for var in GEMINI_VARS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match='GOOGLE_API_KEY'):
        config.get_gemini_client()
    with pytest.raises(RuntimeError, match='GOOGLE_API_KEY'):
        config.get_gemini_client_2()
    with pytest.raises(RuntimeError, match='GOOGLE_API_KEY'):
        config.get_gemini_client_sequence()


def test_optional_clients_are_none_without_keys(monkeypatch):
    for var in OPTIONAL_VARS:
        monkeypatch.delenv(var, raising=False)
    assert config.get_openai_client() is None
    assert config.get_anthropic_client() is None
    assert config.get_xai_client() is None
    assert config.get_deepseek_client() is None


def test_client_is_cached_per_key(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'gsk_test_key')
    first = config.get_groq_client()
    assert first is config.get_groq_client()
    # 키가 바뀌면 재생성
    monkeypatch.setenv('GROQ_API_KEY', 'gsk_other_key')
    assert config.get_groq_client() is not first


def test_module_getattr_compat(monkeypatch):
    """`from config import groq_client` 스타일 접근이 lazy 생성으로 이어져야 한다."""
    monkeypatch.setenv('GROQ_API_KEY', 'gsk_test_key')
    assert config.groq_client is config.get_groq_client()


def test_module_getattr_unknown_name():
    with pytest.raises(AttributeError):
        config.no_such_attribute


def test_preprocess_shim_delegates(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'gsk_test_key')
    from pipeline.preprocess import config as shim
    assert shim.groq_client is config.get_groq_client()
    assert shim.GEMINI_GENERATIVE_MODEL == config.GEMINI_GENERATIVE_MODEL
