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
import sys
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
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

missing_keys: list[str] = []
if not GEMINI_API_KEY_1:
    missing_keys.append("GOOGLE_API_KEY_1 (또는 GOOGLE_API_KEY)")
if not GROQ_API_KEY:
    missing_keys.append("GROQ_API_KEY")

if missing_keys:
    print("❌ 필요한 API 키를 환경 변수로 설정해주세요:")
    for k in missing_keys:
        print(f"   - {k}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
# API 클라이언트
# ──────────────────────────────────────────────────────────────

gemini_client   = genai.Client(api_key=GEMINI_API_KEY_1)  # 비디오 파이프라인용
gemini_client_2 = genai.Client(api_key=GEMINI_API_KEY_2)  # 오디오 파이프라인용
groq_client     = Groq(api_key=GROQ_API_KEY)
openai_client   = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY and OpenAI is not None else None
xai_client      = OpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL) if XAI_API_KEY and OpenAI is not None else None
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL) if DEEPSEEK_API_KEY and OpenAI is not None else None
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY and Anthropic is not None else None

_openai_client_key = OPENAI_API_KEY or ""
_xai_client_key = XAI_API_KEY or ""
_xai_base_url = XAI_BASE_URL or ""
_deepseek_client_key = DEEPSEEK_API_KEY or ""
_deepseek_base_url = DEEPSEEK_BASE_URL or ""
_anthropic_client_key = ANTHROPIC_API_KEY or ""
_gemini_client_key_1 = GEMINI_API_KEY_1 or ""
_gemini_client_key_2 = GEMINI_API_KEY_2 or ""

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
    global openai_client, _openai_client_key
    current_key = os.getenv("OPENAI_API_KEY") or ""
    if current_key != _openai_client_key:
        _openai_client_key = current_key
        openai_client = OpenAI(api_key=current_key) if current_key and OpenAI is not None else None
    return openai_client


def get_xai_client():
    global xai_client, _xai_client_key, _xai_base_url
    current_key = os.getenv("XAI_API_KEY") or ""
    current_base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
    if current_key != _xai_client_key or current_base_url != _xai_base_url:
        _xai_client_key = current_key
        _xai_base_url = current_base_url
        xai_client = (
            OpenAI(api_key=current_key, base_url=current_base_url)
            if current_key and OpenAI is not None
            else None
        )
    return xai_client


def get_deepseek_client():
    global deepseek_client, _deepseek_client_key, _deepseek_base_url
    current_key = os.getenv("DEEPSEEK_API_KEY") or ""
    current_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if current_key != _deepseek_client_key or current_base_url != _deepseek_base_url:
        _deepseek_client_key = current_key
        _deepseek_base_url = current_base_url
        deepseek_client = (
            OpenAI(api_key=current_key, base_url=current_base_url)
            if current_key and OpenAI is not None
            else None
        )
    return deepseek_client


def get_anthropic_client():
    global anthropic_client, _anthropic_client_key
    current_key = os.getenv("ANTHROPIC_API_KEY") or ""
    if current_key != _anthropic_client_key:
        _anthropic_client_key = current_key
        anthropic_client = Anthropic(api_key=current_key) if current_key and Anthropic is not None else None
    return anthropic_client


def get_gemini_client_sequence():
    global gemini_client, gemini_client_2, _gemini_client_key_1, _gemini_client_key_2
    current_key_1 = os.getenv("GOOGLE_API_KEY_1") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    current_key_2 = os.getenv("GOOGLE_API_KEY_2") or current_key_1
    if current_key_1 != _gemini_client_key_1:
        _gemini_client_key_1 = current_key_1
        gemini_client = genai.Client(api_key=current_key_1) if current_key_1 else None
    if current_key_2 != _gemini_client_key_2:
        _gemini_client_key_2 = current_key_2
        gemini_client_2 = genai.Client(api_key=current_key_2) if current_key_2 else None
    seq = []
    if gemini_client_2 is not None:
        seq.append(("gemini_client_2", gemini_client_2))
    if gemini_client is not None and gemini_client is not gemini_client_2:
        seq.append(("gemini_client", gemini_client))
    return seq


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
