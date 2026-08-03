"""
config.py
=========
API 클라이언트 초기화 및 경로 상수 정의

경로 구조:
    input/
        lecture.mp4
    output_slides/
        metadata.json
        scene_0001_base.jpg
        ...
    output/
        {stem}_slide_textualized.json
        {stem}_annotation.json
        {stem}_segments.json
        {stem}_silences.json
        {stem}_audio_features.json
        {stem}_audio_quality.json
        {stem}_emphasis.json
        {stem}_by_scene.json
        {stem}_slide_classified.json
        {stem}_fused.json
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from google import genai

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

load_dotenv()

# ──────────────────────────────────────────────────────────────
# API 키
# ──────────────────────────────────────────────────────────────

# 비디오 파이프라인용 (slide_textualizer, annotation_analyzer)
GEMINI_API_KEY_1 = os.getenv("GOOGLE_API_KEY_1") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
# 오디오 파이프라인용 (text_processor, segment_grouper, emphasis_keyword)
GEMINI_API_KEY_2 = os.getenv("GOOGLE_API_KEY_2") or GEMINI_API_KEY_1  # 키 1개만 있을 때 fallback

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# Ollama는 인증이 필요 없지만 OpenAI SDK가 빈 문자열 키를 거부하므로 더미 값을 쓴다.
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")

# ──────────────────────────────────────────────────────────────
# API 클라이언트 (lazy init)
#
# import 시점에는 아무 클라이언트도 만들지 않는다. 실제 사용 시점에 생성하고,
# 필수 키(Gemini/Groq)가 없으면 그때 어떤 환경변수가 왜 필요한지 명시한
# RuntimeError를 낸다. 기존 코드의 `from .config import gemini_client` 같은
# 접근은 아래 모듈 __getattr__(PEP 562)이 받아서 처리한다.
# ──────────────────────────────────────────────────────────────

# name -> (생성에 쓴 키 조합, 클라이언트). 키가 바뀌면 재생성한다.
_client_cache: dict[str, tuple[str, object | None]] = {}


def _cached_client(name: str, cache_key: str, factory):
    entry = _client_cache.get(name)
    if entry is not None and entry[0] == cache_key:
        return entry[1]
    client = factory()
    _client_cache[name] = (cache_key, client)
    return client


def _gemini_env_keys() -> tuple[str, str]:
    key_1 = os.getenv("GOOGLE_API_KEY_1") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    key_2 = os.getenv("GOOGLE_API_KEY_2") or key_1
    return key_1, key_2


def get_gemini_client():
    """비디오 파이프라인용 Gemini 클라이언트. 키가 없으면 명확한 에러."""
    key_1, _ = _gemini_env_keys()
    if not key_1:
        raise RuntimeError(
            "Gemini API 키가 없습니다. GOOGLE_API_KEY_1 (또는 GOOGLE_API_KEY) 환경변수를 "
            "설정하세요 — 슬라이드 텍스트화/필기 분석 단계에 필요합니다."
        )
    return _cached_client("gemini_1", key_1, lambda: genai.Client(api_key=key_1))


def get_gemini_client_2():
    """오디오 파이프라인용 Gemini 클라이언트. 키가 없으면 명확한 에러."""
    _, key_2 = _gemini_env_keys()
    if not key_2:
        raise RuntimeError(
            "Gemini API 키가 없습니다. GOOGLE_API_KEY_1 (또는 GOOGLE_API_KEY, 보조로 "
            "GOOGLE_API_KEY_2) 환경변수를 설정하세요 — 텍스트 교정/세그먼트 그룹핑 단계에 필요합니다."
        )
    return _cached_client("gemini_2", key_2, lambda: genai.Client(api_key=key_2))


def get_groq_client():
    """Groq(Whisper 전사) 클라이언트. 키가 없으면 명확한 에러."""
    key = os.getenv("GROQ_API_KEY") or ""
    if not key:
        raise RuntimeError(
            "Groq API 키가 없습니다. GROQ_API_KEY 환경변수를 설정하세요 — 전체 전사 단계에 필요합니다."
        )
    return _cached_client("groq", key, lambda: Groq(api_key=key))

# ──────────────────────────────────────────────────────────────
# 기본 Gemini 모델 상수
# ──────────────────────────────────────────────────────────────

GEMINI_GENERATIVE_MODEL = "gemini-2.5-flash"
GEMINI_GENERATIVE_MODEL_WITH_PREFIX = f"models/{GEMINI_GENERATIVE_MODEL}"

# ──────────────────────────────────────────────────────────────
# 기본 경로 상수 (CLI 인자로 override 가능)
# ──────────────────────────────────────────────────────────────

DEFAULT_INPUT_DIR  = Path("input")
DEFAULT_SLIDES_DIR = Path("output_slides")   # metadata.json + 슬라이드 이미지
DEFAULT_OUTPUT_DIR = Path("output")          # 모든 분석 결과

# ──────────────────────────────────────────────────────────────
# 파일명 헬퍼
# ──────────────────────────────────────────────────────────────

def output_paths(stem: str, output_dir: Path, slides_dir: Path) -> dict[str, Path]:
    """
    영상 stem과 디렉토리로 모든 출력 경로를 한 번에 반환.

    사용 예:
        paths = output_paths("lecture", Path("output"), Path("output_slides"))
        paths["segments"]   # output/lecture_segments.json
        paths["metadata"]   # output_slides/metadata.json
    """
    return {
        # slides_dir
        "metadata":            slides_dir / "metadata.json",
        # output_dir
        "textualized":         output_dir / f"{stem}_slide_textualized.json",
        "annotation":          output_dir / f"{stem}_annotation.json",
        "segments":            output_dir / f"{stem}_segments.json",
        "silences":            output_dir / f"{stem}_silences.json",
        "audio_features":      output_dir / f"{stem}_audio_features.json",
        "audio_quality":       output_dir / f"{stem}_audio_quality.json",
        "emphasis":            output_dir / f"{stem}_emphasis.json",
        "by_scene":            output_dir / f"{stem}_by_scene.json",
        "classified":          output_dir / f"{stem}_slide_classified.json",
        "fused":               output_dir / f"{stem}_fused.json",
    }


def get_openai_client():
    current_key = os.getenv("OPENAI_API_KEY") or ""
    return _cached_client(
        "openai", current_key,
        lambda: OpenAI(api_key=current_key) if current_key and OpenAI is not None else None,
    )


def get_xai_client():
    current_key = os.getenv("XAI_API_KEY") or ""
    current_base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
    return _cached_client(
        "xai", f"{current_key}|{current_base_url}",
        lambda: (
            OpenAI(api_key=current_key, base_url=current_base_url)
            if current_key and OpenAI is not None
            else None
        ),
    )


def get_deepseek_client():
    current_key = os.getenv("DEEPSEEK_API_KEY") or ""
    current_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return _cached_client(
        "deepseek", f"{current_key}|{current_base_url}",
        lambda: (
            OpenAI(api_key=current_key, base_url=current_base_url)
            if current_key and OpenAI is not None
            else None
        ),
    )


def get_ollama_client():
    current_key = os.getenv("OLLAMA_API_KEY", "ollama") or "ollama"
    current_base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")
    return _cached_client(
        "ollama", f"{current_key}|{current_base_url}",
        lambda: (
            OpenAI(api_key=current_key, base_url=current_base_url)
            if OpenAI is not None
            else None
        ),
    )


def get_anthropic_client():
    current_key = os.getenv("ANTHROPIC_API_KEY") or ""
    current_base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    return _cached_client(
        "anthropic", f"{current_key}|{current_base_url}",
        lambda: (
            Anthropic(api_key=current_key, base_url=current_base_url)
            if current_key and Anthropic is not None
            else None
        ),
    )


def get_gemini_client_sequence():
    key_1, key_2 = _gemini_env_keys()
    seq = []
    if key_2:
        seq.append(("gemini_client_2", get_gemini_client_2()))
    if key_1 and key_1 != key_2:
        seq.append(("gemini_client", get_gemini_client()))
    if not seq:
        raise RuntimeError(
            "Gemini API 키가 없습니다. GOOGLE_API_KEY_1 (또는 GOOGLE_API_KEY) 환경변수를 설정하세요."
        )
    return seq


# 기존 모듈 전역 클라이언트 이름 호환 (PEP 562).
# `from .config import gemini_client` / `config.groq_client` 접근 시 이 함수가 호출되어
# 그 시점에 클라이언트를 생성한다. 값을 모듈에 저장하지 않으므로 매 접근마다 캐시를 거친다.
_LAZY_CLIENT_GETTERS = {
    "gemini_client": get_gemini_client,
    "gemini_client_2": get_gemini_client_2,
    "groq_client": get_groq_client,
    "openai_client": get_openai_client,
    "xai_client": get_xai_client,
    "deepseek_client": get_deepseek_client,
    "anthropic_client": get_anthropic_client,
}


def __getattr__(name: str):
    getter = _LAZY_CLIENT_GETTERS.get(name)
    if getter is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getter()


def resolve_anthropic_model(model_name: str) -> str:
    aliases = {
        "haiku-4.5": "claude-haiku-4-5-20251001",
        "claude-haiku-4.5": "claude-haiku-4-5-20251001",
        "claude-haiku-4-5": "claude-haiku-4-5-20251001",
        "sonnet-4.5": "claude-sonnet-4-5-20250929",
        "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
        "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
        "opus-4.5": "claude-3-opus-latest",
        "claude-opus-4.5": "claude-3-opus-latest",
        "claude-opus-4-5": "claude-3-opus-latest",
    }
    return aliases.get(str(model_name or "").strip(), str(model_name or "").strip())
