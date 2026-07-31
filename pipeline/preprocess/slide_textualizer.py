"""
슬라이드 텍스트화 파이프라인 (Stage 1)

슬라이드 이미지 내의 텍스트, 다이어그램, 표 등 시각 요소를
OpenAI Vision을 통해 텍스트로 변환합니다.

Input:
  - output_slides/ 폴더: slide_extractor.py 출력 디렉토리
    ├── scene_001_base.jpg           ← 항상 텍스트 추출 대상
    ├── scene_001_build_01.jpg       ← 애니메이션 진행으로 콘텐츠가 바뀐 상태 (있으면 함께 분석)
    ├── scene_001_build_00_annot_01.jpg  ← 교수 필기 스냅샷 (텍스트 추출 대상 아님)
    ├── metadata.json
    └── ...

  base + 모든 distinct build 프레임을 함께 모델에 보내 텍스트를 추출한다. build 단계마다
  콘텐츠가 점진적으로 바뀌거나 대체될 수 있어, 마지막 build 프레임 하나만 보면 중간 단계에만
  존재했던 내용을 놓칠 수 있기 때문이다. annotation(교수 필기) 프레임은 텍스트 추출 대상에서
  완전히 제외한다 — 필기는 새 콘텐츠 상태가 아니라 안정화 스냅샷일 뿐이다.

Output:
  - slide_textualized.json: 텍스트화 결과

추출 항목:
  - t1          : 슬라이드 원본 텍스트 (OpenAI Vision)
  - t1_structure: 다이어그램/표/화살표 관계 (Stage 3 관계 추출 힌트)

※ 오디오 정제는 다음 단계에서 t1을 사용하여 진행
"""

import os
import re
import json
import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from PIL import Image

from .emphasis_keyword import (
    get_topic_keyword_count_map,
    summarize_topic_keyword_counts_for_text,
    topic_keyword_count_items,
)
try:
    from .ocr_hint import get_slide_ocr_hint, ocr_enabled
except ImportError:  # pragma: no cover - direct script execution fallback
    from ocr_hint import get_slide_ocr_hint, ocr_enabled

try:
    from json_repair import repair_json
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 외부 라이브러리 노이즈 로그 억제
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ============================================================================ #
#  설정                                                                         #
# ============================================================================ #

@dataclass
class Config:
    slides_dir: Path = Path("output_slides")     # slide_extractor.py 출력 디렉토리
    output_dir: Path = Path("output")
    output_filename: str = "slide_textualized.json"  # 저장 파일명 ({stem}_slide_textualized.json)
    # T1 textualization is OpenAI-only. Kept for config compatibility.
    provider: str = "openai"
    model: str = os.getenv("GRAPHLEC_SLIDE_TEXTUALIZER_MODEL", "gpt-5.4-mini")
    max_retries: int = 3
    retry_delay: float = 5.0
    workers: int = int(os.getenv("GRAPHLEC_SLIDE_TEXTUALIZER_WORKERS", "12"))
    ocr_provider: str = os.getenv("GRAPHLEC_SLIDE_OCR_PROVIDER", "none")
    ocr_model_dir: str = os.getenv("GRAPHLEC_SLIDE_OCR_MODEL_DIR", "")
    ocr_lang: str = os.getenv("GRAPHLEC_SLIDE_OCR_LANG", "multilingual")

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Post-video verifier input is intentionally base-only. Teacher annotations are not textualized.
T1_BASE_ONLY_EXTRACTION_PROMPT = """
이 슬라이드 BASE 이미지에서 인쇄된 슬라이드 원본 텍스트와 구조만 추출하라.
교수 필기, 손글씨, 밑줄, 화살표, 강조 표시는 모두 무시한다. 설명 없이 JSON만 출력.

출력 형식:
{
  "slide_type": "text" | "image_only" | "mixed",
  "title": "슬라이드 제목 (없으면 빈 문자열)",
  "raw_text": "인쇄된 텍스트 전체 (위→아래, 좌→우 순서, 줄바꿈은 \n)",
  "structure": "표/다이어그램/화살표 관계를 텍스트로 기술",
  "visual_assets": [
    {
      "asset_type": "table" | "diagram" | "figure" | "list" | "chart" | "image" | "other",
      "title": "시각자료 제목/캡션",
      "description": "시각자료가 전달하는 내용",
      "raw_text": "시각자료 내부의 인쇄된 텍스트",
      "visual_elements": [],
      "visual_relations": [],
      "layout": {},
      "bbox": null
    }
  ]
}

규칙:
- 제목, 본문, 불릿, 표 셀, 다이어그램 레이블을 포함한다.
- 교수 필기/손글씨/강조 표시는 절대 추출하지 않는다.
- structure와 visual_assets는 인쇄된 슬라이드 구조만 설명한다.
- 시각자료가 없으면 visual_assets는 []로 둔다.
"""

# base + build 프레임이 있는 슬라이드 (이미지 2장 이상):
# Image 1 (BASE)        → slide_emphasis 추출 (애니메이션 진행 전 원본)
# Image 2..N (BUILD 1..N-1) → t1 텍스트 추출, BASE와 합쳐 놓치는 내용 없이 종합
T1_EXTRACTION_PROMPT_WITH_BUILDS = """
여러 장의 이미지가 순서대로 제공됩니다:
  - Image 1        : BASE, 이 슬라이드의 최초 상태
  - Image 2..N     : 이후 애니메이션/빌드 단계에서 순서대로 드러난 상태 (번호가 클수록 나중 단계)

설명 없이 JSON만 출력.

출력 형식:
{
  "slide_type": "text" | "image_only" | "mixed",
  "title": "슬라이드 제목 (없으면 빈 문자열)",
  "raw_text": "슬라이드에 보이는 모든 텍스트 (위→아래, 좌→우 순서, 줄바꿈은 \\n)",
  "structure": "다이어그램/표/화살표 관계를 텍스트로 기술",
  "visual_assets": [
    {
      "asset_type": "table" | "diagram" | "figure" | "list" | "chart" | "image" | "other",
      "title": "시각자료 제목/캡션 (없으면 빈 문자열)",
      "description": "이 시각자료 하나가 전달하는 내용",
      "raw_text": "시각자료 내부의 셀/레이블/캡션 텍스트",
      "visual_elements": [
        {
          "type": "arrow" | "box" | "icon" | "icon_group" | "line" | "layer" | "callout" | "label" | "text_block" | "table_cell" | "axis" | "marker" | "other",
          "label": "요소에 보이는 텍스트/이름",
          "role": "요소가 시각자료 안에서 맡는 역할",
          "meaning": "요소가 전달하는 의미",
          "bbox": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        }
      ],
      "visual_relations": [
        {
          "source": "관계 시작 요소",
          "target": "관계 대상 요소",
          "relation": "관계 의미",
          "visual_cue": "arrow" | "line" | "position" | "containment" | "color" | "alignment" | "other",
          "direction": "bidirectional" | "source_to_target" | "target_to_source" | "none",
          "meaning": "이 시각적 관계가 설명하는 내용"
        }
      ],
      "layout": {
        "top": ["상단 요소"],
        "middle": ["중앙 요소"],
        "bottom": ["하단 요소"],
        "left": ["왼쪽 요소"],
        "right": ["오른쪽 요소"]
      },
      "bbox": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    }
  ],
  "slide_emphasis": [
    {
      "text": "강조된 텍스트 원문",
      "type": "color" | "bold" | "underline" | "box" | "highlight" | "italic" | "callout" | "other",
      "color": "red",
      "bbox": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    }
  ]
}

추출 규칙:
[raw_text / structure / slide_type]
  - 제공된 모든 이미지(Image 1..N)에 나타나는 내용을 빠짐없이 합쳐서 작성할 것.
  - 애니메이션이 진행되며 특정 단계에서만 보였다가 이후 단계에서 사라지거나 다른 내용으로
    대체된 것이 있다면, 그 시점에 존재했던 내용도 빠뜨리지 말고 전부 포함할 것.
  - 나중 단계에서 같은 항목이 반복해서 나온다고 중복으로 기재하지 말고, 실제로 서로 다른
    내용일 때만 각각 반영할 것.
  - 인쇄된 원본 텍스트만 추출하고, 특정 이미지에서만 손으로 쓴 필기가 보이더라도 그 필기
    자체는 포함하지 말 것.
  - visual_assets도 모든 이미지에서 나타나는 시각자료를 종합해서 구성하되, 손글씨/필기로
    추가된 선과 도형은 포함하지 말 것.
  - 한 슬라이드에 표/다이어그램/그림/차트/목록이 여러 개 있으면 visual_assets 배열에 각각 분리할 것.
  - 시각자료가 없으면 빈 배열 [].
  - visual_elements는 화살표, 박스, 아이콘, 레이어, 말풍선, 라벨, 선, 축 등 질문 대상이 될 수 있는 시각 요소를 분리할 것.
  - visual_relations는 화살표/선/위치/포함 관계가 무엇을 연결하고 무엇을 의미하는지 분리할 것.
  - layout은 상단/중앙/하단/좌/우 등 공간 배치가 의미를 갖는 경우에만 작성할 것.

[slide_emphasis]
  - 반드시 Image 1 (BASE) 만을 기준으로 판단할 것. 이후 이미지들은 slide_emphasis 판단에
    절대 사용하지 말 것.
  - 슬라이드 제작 시 의도적으로 삽입된 시각적 강조만 수집:
    - 슬라이드 최상단 제목과 페이지 번호는 제외하되, 본문 영역 안에서 하위 내용을 묶는
      섹션 헤더/소제목은 포함할 것. 특히 주변 본문보다 명확히 크거나 굵거나 색이 다르거나
      계층 제목처럼 배치된 텍스트는 "bold" 또는 적절한 type으로 수집할 것.
    "color"     : Image 1에서 이미 다른 텍스트와 색상이 다른 텍스트 (빨간색, 주황색 등)
    "bold"      : 굵게 처리된 텍스트
    "underline" : 슬라이드 디자인의 일부인 밑줄 (Image 1에 이미 존재하는 것만)
    "box"       : 강조 박스/테두리로 감싸진 텍스트
    "highlight" : 형광펜 효과
    "italic"    : 기울임 처리된 텍스트
    "callout"   : 말풍선/풍선 도형(callout shape) 안에 담긴 텍스트.
                  특정 요소를 화살표로 가리키며 설명하는 줄글 형태도 포함.
                  (단순 강조 목적이 아닌 설명 대체 용도로 보이는 경우에도 "callout"으로 분류)
    "other"     : 위에 해당하지 않는 기타 강조
  - 불릿 포인트/리스트 마커(■ ▪ □ • ▶ 등)의 색상은 텍스트 강조로 보지 말 것.
    마커 색이 주황·빨강이더라도, 그 옆 텍스트 자체가 같은 색이 아니라면 "color" 강조로 포함하지 말 것.
  - 강조 요소가 없으면 빈 배열 []
  - bbox: 정규화 좌표 (0.0~1.0, 좌상단 기준) {"x": float, "y": float, "w": float, "h": float}
  - color: 텍스트 색상이 강조 이유일 때만 기재 (예: "red", "orange"), 아니면 null
"""


# ============================================================================ #
#  슬라이드 로더                                                                 #
# ============================================================================ #

class SlideLoader:
    """
    slide_extractor.py 출력 디렉토리에서 텍스트 추출 대상 이미지 로드.

    선택 전략:
      - base + 모든 distinct build 프레임(있으면)을 함께 텍스트 추출 대상으로 사용
      - annotation(교수 필기) 프레임은 텍스트 추출 대상에서 완전히 제외

    타임스탬프는 metadata.json에서 읽고, 없으면 파일명 패턴으로 폴백.
    """

    def __init__(self, slides_dir: Path):
        self.slides_dir = Path(slides_dir)

    def load(self) -> List[Dict]:
        metadata_path = self.slides_dir / "metadata.json"

        if metadata_path.exists():
            return self._load_from_metadata(metadata_path)
        else:
            logger.warning("metadata.json 없음 - 파일명 패턴으로 폴백")
            return self._load_from_filenames()

    def _load_from_metadata(self, metadata_path: Path) -> List[Dict]:
        """
        metadata.json 기준으로 scene별 텍스트 추출 대상을 결정한다.
        실제 OpenAI Vision 호출은 slide_canonical_number 기준으로 캐시 가능하도록
        representative scene 정보를 함께 싣는다.
        """
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        slide_number_lookup = self._build_slide_number_lookup(metadata)

        # scene_index 기준으로 base / build 분류. annotation(교수 필기) 프레임은
        # 텍스트 추출 후보에서 아예 배제하고, "있었다"는 사실만 참고용으로 기록한다.
        base_entries: Dict[int, dict] = {}
        build_entries: Dict[int, List[dict]] = {}
        annotation_present: Dict[int, bool] = {}

        for entry in metadata:
            idx = entry.get("scene_index")
            if idx is None:
                continue
            capture_type = entry.get("capture_type")
            if capture_type == "base":
                base_entries[idx] = entry
            elif capture_type == "build":
                build_entries.setdefault(idx, []).append(entry)
            elif capture_type in ("annot", "annotation"):
                annotation_present[idx] = True

        scene_records = []
        for scene_num in sorted(base_entries.keys()):
            base = base_entries[scene_num]
            builds = sorted(
                build_entries.get(scene_num, []),
                key=lambda entry: (
                    int(entry.get("build_index", 0) or 0),
                    float(entry.get("timestamp_sec", 0.0) or 0.0),
                ),
            )
            scene_type = base.get("scene_type", "slide")

            canonical = (
                base.get("slide_canonical_index")
                or base.get("same_slide_canonical")
                or scene_num
            )
            slide_number = base.get("slide_number")
            if not isinstance(slide_number, int):
                slide_number = slide_number_lookup.get(int(canonical), int(canonical))
            scene_records.append({
                "scene_number": scene_num,
                "scene_index": scene_num,
                "slide_number": slide_number,
                "slide_canonical_number": canonical,
                "slide_visit_order": base.get("slide_visit_order", base.get("same_slide_visit_order", 1)),
                "slide_is_revisit": bool(base.get("slide_is_revisit", base.get("same_slide_is_revisit", False))),
                "timestamp": base.get("timestamp_sec", 0.0),
                "scene_type": scene_type,
                "base_entry": base,
                "build_entries": builds,
                "has_teacher_annotation": bool(annotation_present.get(scene_num)),
                "text_source": "base+build" if builds else "base",
            })

        # 같은 slide family는 가장 정보가 많은 representative scene 하나만 LLM 입력으로 사용
        canonical_representatives: Dict[int, dict] = {}
        for record in scene_records:
            if record.get("scene_type") == "video":
                continue
            canonical = record["slide_canonical_number"]
            current = canonical_representatives.get(canonical)
            score = (
                float(record["timestamp"]),
                int(record["scene_number"]),
            )
            if current is None or score > current["_rep_score"]:
                canonical_representatives[canonical] = {
                    "_rep_score": score,
                    "scene_number": record["scene_number"],
                    "base_entry": record["base_entry"],
                    "build_entries": record["build_entries"],
                    "has_teacher_annotation": record["has_teacher_annotation"],
                    "text_source": record["text_source"],
                }

        slides = []
        for record in scene_records:
            if record.get("scene_type") == "video":
                image_path = self.slides_dir / record["base_entry"]["filename"]
                if not image_path.exists():
                    logger.warning(f"video thumbnail 없음: {image_path}, 스킵")
                    continue
                slides.append({
                    "scene_number":  record["scene_number"],
                    "scene_index":   record["scene_index"],
                    "scene_type":    "video",
                    "slide_number":  record["slide_number"],
                    "slide_canonical_number": record["slide_canonical_number"],
                    "slide_visit_order": record["slide_visit_order"],
                    "slide_is_revisit": record["slide_is_revisit"],
                    "representative_scene_number": record["scene_number"],
                    "timestamp":      record["timestamp"],
                    "image_path":     str(image_path),
                    "base_image":     None,
                    "build_images":   [],
                    "has_teacher_annotation": False,
                    "text_source":    "video",
                    "title":          "영상 구간",
                    "t1":             "",
                    "t1_structure":   "",
                    "slide_type":     "video",
                    "slide_emphasis": [],
                })
                continue

            canonical = record["slide_canonical_number"]
            rep = canonical_representatives[canonical]

            base_image_path = self.slides_dir / rep["base_entry"]["filename"]
            if not base_image_path.exists():
                logger.warning(f"base 이미지 없음: {base_image_path}, 스킵")
                continue

            build_image_paths: List[str] = []
            build_images: List[Image.Image] = []
            for build_entry in rep["build_entries"]:
                build_path = self.slides_dir / build_entry["filename"]
                if not build_path.exists():
                    logger.warning(f"build 이미지 없음: {build_path}, 건너뜀")
                    continue
                build_image_paths.append(str(build_path))
                build_images.append(Image.open(build_path).convert("RGB"))

            image_path = build_image_paths[-1] if build_image_paths else str(base_image_path)

            slides.append({
                "scene_number":  record["scene_number"],
                "scene_index":   record["scene_index"],
                "scene_type":    record.get("scene_type", "slide"),
                "slide_number":  record["slide_number"],
                "slide_canonical_number": canonical,
                "slide_visit_order": record["slide_visit_order"],
                "slide_is_revisit": record["slide_is_revisit"],
                "representative_scene_number": rep["scene_number"],
                "timestamp":      record["timestamp"],  # 타임스탬프는 현재 scene 기준
                "image_path":     image_path,
                "base_image":     Image.open(base_image_path).convert("RGB"),
                "build_images":   build_images,
                "has_teacher_annotation": rep["has_teacher_annotation"],
                "text_source":    rep["text_source"],
            })

        slides.sort(key=lambda x: x["scene_number"])
        build_present_count = sum(1 for s in slides if s.get("build_images"))
        logger.info(
            f"✓ Loaded {len(slides)} scenes from metadata.json "
            f"(unique slides: {len(canonical_representatives)}, "
            f"base_input: {len(slides)}, build_present: {build_present_count})"
        )
        return slides

    @staticmethod
    def _build_slide_number_lookup(metadata: list[dict]) -> dict[int, int]:
        from collections import defaultdict

        by_scene: dict[int, list[dict]] = defaultdict(list)
        for item in metadata:
            scene_idx = item.get("scene_index")
            if isinstance(scene_idx, int):
                by_scene[scene_idx].append(item)

        ordered_pairs: list[tuple[float, int]] = []
        for scene_idx in sorted(by_scene):
            items = by_scene[scene_idx]
            base = next((x for x in items if x.get("capture_type") == "base"), items[0])
            canonical = int(base.get("slide_canonical_index") or base.get("same_slide_canonical") or scene_idx)
            ts = float(base.get("scene_start_sec", base.get("slide_start_sec", base.get("timestamp_sec", 0.0))) or 0.0)
            ordered_pairs.append((ts, canonical))

        lookup: dict[int, int] = {}
        for _, canonical in sorted(ordered_pairs, key=lambda x: x[0]):
            if canonical not in lookup:
                lookup[canonical] = len(lookup) + 1
        return lookup

    def _load_from_filenames(self) -> List[Dict]:
        """
        metadata 없을 때 파일명 패턴으로 폴백. base + build 프레임만 사용하고,
        annotation(교수 필기)은 텍스트 추출 대상에서 제외한다.
        """
        base_pattern  = re.compile(r'slide_(\d+)_base\.(jpg|png)', re.IGNORECASE)
        build_pattern = re.compile(r'slide_(\d+)_build_(\d+)\.(jpg|png)', re.IGNORECASE)
        annot_pattern = re.compile(r'slide_(\d+)_annot_(\d+)\.(jpg|png)', re.IGNORECASE)

        base_files: Dict[int, Path] = {}
        build_files: Dict[int, List[tuple]] = {}  # {slide_num: [(build_idx, path), ...]}
        annot_present: Dict[int, bool] = {}

        for file in self.slides_dir.iterdir():
            m = base_pattern.match(file.name)
            if m:
                base_files[int(m.group(1))] = file
                continue
            m = build_pattern.match(file.name)
            if m:
                num, idx = int(m.group(1)), int(m.group(2))
                build_files.setdefault(num, []).append((idx, file))
                continue
            m = annot_pattern.match(file.name)
            if m:
                annot_present[int(m.group(1))] = True

        slides = []
        for slide_num in sorted(base_files.keys()):
            base_path = base_files[slide_num]
            builds = sorted(build_files.get(slide_num, []), key=lambda pair: pair[0])
            build_images = [Image.open(path).convert("RGB") for _, path in builds]

            slides.append({
                "scene_number": slide_num,
                "scene_index": slide_num,
                "scene_type": "slide",
                "slide_number": slide_num,
                "slide_canonical_number": slide_num,
                "slide_visit_order": 1,
                "slide_is_revisit": False,
                "representative_scene_number": slide_num,
                "timestamp":    0.0,
                "image_path":   str(builds[-1][1] if builds else base_path),
                "base_image":   Image.open(base_path).convert("RGB"),
                "build_images": build_images,
                "has_teacher_annotation": bool(annot_present.get(slide_num)),
                "text_source":  "base+build" if builds else "base",
            })

        build_present_count = sum(1 for s in slides if s.get("build_images"))
        logger.info(
            f"✓ Loaded {len(slides)} slides from filenames "
            f"(build_present: {build_present_count}, base_only: {len(slides) - build_present_count})"
        )
        return slides


# ============================================================================ #
#  t1 추출기 (OpenAI Vision)                                                     #
# ============================================================================ #

class T1Extractor:
    """슬라이드 base 이미지 → t1 (원본 텍스트) + t1_structure 추출"""

    _ASSET_TYPES = {"table", "diagram", "figure", "list", "chart", "image", "other"}

    def __init__(self, config: Config):
        self.config = config
        from .config import get_openai_client
        self.client = get_openai_client()
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
        logger.info(f"✓ OpenAI initialized for t1 extraction: {self.config.model}")

    @staticmethod
    def _compose_prompt_with_ocr(prompt: str, ocr_hint: str) -> str:
        hint = str(ocr_hint or "").strip()
        if not hint:
            return prompt
        return (
            f"{prompt}\n\n"
            "[Pre-extracted OCR hint - use only as a hint, trust the image if they conflict]\n"
            f"{hint}\n"
        )

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _call_openai(
        self,
        base_image: Image.Image,
        build_images: List[Image.Image] = None,
        ocr_hint: str = "",
    ) -> str:
        build_images = build_images or []
        template = T1_EXTRACTION_PROMPT_WITH_BUILDS if build_images else T1_BASE_ONLY_EXTRACTION_PROMPT
        prompt = self._compose_prompt_with_ocr(template, ocr_hint)
        content = [{"type": "text", "text": prompt}]
        content.append({"type": "image_url", "image_url": {"url": self._image_to_data_url(base_image), "detail": "high"}})
        for build_image in build_images:
            content.append({"type": "image_url", "image_url": {"url": self._image_to_data_url(build_image), "detail": "high"}})

        last_exc = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": content}],
                    response_format={"type": "json_object"},
                    max_completion_tokens=4096,
                    temperature=0,
                )
                try:
                    from .cost_report import record_model_call

                    record_model_call(
                        stage="stage2a_slide_textualizer",
                        provider="openai",
                        model=self.config.model,
                        response=response,
                        image_count=1 + len(build_images),
                        prompt_chars=len(prompt),
                    )
                except Exception:
                    pass
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                last_exc = e
                logger.warning(
                    f"  ⚠ OpenAI call failed "
                    f"(attempt {attempt+1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        raise last_exc

    @classmethod
    def _normalize_visual_elements(cls, elements: Any) -> List[Dict]:
        if not isinstance(elements, list):
            return []

        normalized: List[Dict] = []
        for item in elements:
            if isinstance(item, str):
                item = {"label": item}
            if not isinstance(item, dict):
                continue

            bbox = item.get("bbox")
            if not isinstance(bbox, dict):
                bbox = None

            entry = {
                "type": str(item.get("type") or "other").strip().lower(),
                "label": str(item.get("label") or item.get("text") or item.get("name") or "").strip(),
                "role": str(item.get("role") or "").strip(),
                "meaning": str(item.get("meaning") or item.get("description") or "").strip(),
                "bbox": bbox,
            }
            if any(entry.get(k) for k in ("label", "role", "meaning")):
                normalized.append(entry)
        return normalized

    @classmethod
    def _normalize_visual_relations(cls, relations: Any) -> List[Dict]:
        if not isinstance(relations, list):
            return []

        normalized: List[Dict] = []
        for item in relations:
            if isinstance(item, str):
                item = {"meaning": item}
            if not isinstance(item, dict):
                continue

            entry = {
                "source": str(item.get("source") or item.get("from") or "").strip(),
                "target": str(item.get("target") or item.get("to") or "").strip(),
                "relation": str(item.get("relation") or item.get("type") or "").strip(),
                "visual_cue": str(item.get("visual_cue") or item.get("cue") or "").strip().lower(),
                "direction": str(item.get("direction") or "").strip().lower(),
                "meaning": str(item.get("meaning") or item.get("description") or "").strip(),
            }
            if any(entry.values()):
                normalized.append(entry)
        return normalized

    @staticmethod
    def _normalize_visual_layout(layout: Any) -> Dict:
        if not isinstance(layout, dict):
            return {}
        normalized: Dict[str, Any] = {}
        for key, value in layout.items():
            key_s = str(key).strip()
            if not key_s:
                continue
            if isinstance(value, list):
                vals = [str(v).strip() for v in value if str(v).strip()]
                if vals:
                    normalized[key_s] = vals
            elif isinstance(value, (str, int, float)):
                value_s = str(value).strip()
                if value_s:
                    normalized[key_s] = value_s
        return normalized

    @staticmethod
    def _visual_elements_text(elements: List[Dict]) -> str:
        lines = []
        for el in elements:
            parts = [
                str(el.get("type") or "").strip(),
                str(el.get("label") or "").strip(),
                str(el.get("role") or "").strip(),
                str(el.get("meaning") or "").strip(),
            ]
            line = " | ".join(part for part in parts if part)
            if line:
                lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _visual_relations_text(relations: List[Dict]) -> str:
        lines = []
        for rel in relations:
            endpoints = " -> ".join(
                part for part in [str(rel.get("source") or "").strip(), str(rel.get("target") or "").strip()]
                if part
            )
            parts = [
                endpoints,
                str(rel.get("relation") or "").strip(),
                str(rel.get("visual_cue") or "").strip(),
                str(rel.get("direction") or "").strip(),
                str(rel.get("meaning") or "").strip(),
            ]
            line = " | ".join(part for part in parts if part)
            if line:
                lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _visual_layout_text(layout: Dict) -> str:
        lines = []
        for key, value in layout.items():
            if isinstance(value, list):
                value_s = ", ".join(str(v) for v in value if str(v).strip())
            else:
                value_s = str(value).strip()
            if value_s:
                lines.append(f"{key}: {value_s}")
        return "\n".join(lines)

    @classmethod
    def _normalize_visual_assets(cls, assets: Any) -> List[Dict]:
        if not isinstance(assets, list):
            return []

        normalized: List[Dict] = []
        for idx, item in enumerate(assets, start=1):
            if isinstance(item, str):
                item = {"description": item}
            if not isinstance(item, dict):
                continue

            asset_type = str(item.get("asset_type") or item.get("type") or "other").strip().lower()
            if asset_type not in cls._ASSET_TYPES:
                asset_type = "other"

            title = str(item.get("title") or item.get("caption") or "").strip()
            description = str(item.get("description") or item.get("summary") or "").strip()
            raw_text = str(item.get("raw_text") or item.get("text") or "").strip()
            if not (title or description or raw_text):
                continue

            bbox = item.get("bbox")
            if not isinstance(bbox, dict):
                bbox = None
            visual_elements = cls._normalize_visual_elements(item.get("visual_elements") or item.get("elements"))
            visual_relations = cls._normalize_visual_relations(item.get("visual_relations") or item.get("relations"))
            layout = cls._normalize_visual_layout(item.get("layout"))

            normalized.append(
                {
                    "asset_index": idx,
                    "asset_type": asset_type,
                    "title": title,
                    "description": description,
                    "raw_text": raw_text,
                    "visual_elements": visual_elements,
                    "visual_relations": visual_relations,
                    "layout": layout,
                    "visual_elements_text": cls._visual_elements_text(visual_elements),
                    "visual_relations_text": cls._visual_relations_text(visual_relations),
                    "layout_text": cls._visual_layout_text(layout),
                    "bbox": bbox,
                }
            )
        return normalized

    def _call_gemini(
        self,
        base_image: Image.Image,
        build_images: List[Image.Image] = None,
        ocr_hint: str = "",
    ) -> str:
        """재시도 로직을 포함한 base(+build) 이미지 Gemini Vision 호출."""
        build_images = build_images or []
        if self.provider == "openai":
            return self._call_openai(base_image, build_images=build_images, ocr_hint=ocr_hint)

        template = T1_EXTRACTION_PROMPT_WITH_BUILDS if build_images else T1_BASE_ONLY_EXTRACTION_PROMPT
        contents = [self._compose_prompt_with_ocr(template, ocr_hint), base_image]
        contents.extend(build_images)

        last_exc = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.config.gemini_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                try:
                    from .cost_report import record_model_call

                    record_model_call(
                        stage="stage2a_slide_textualizer",
                        provider="google",
                        model=self.config.gemini_model,
                        response=response,
                        image_count=1 + len(build_images),
                        prompt_chars=len(template),
                    )
                except Exception:
                    pass
                return response.text
            except Exception as e:
                last_exc = e
                logger.warning(
                    f"  ⚠ Gemini call failed "
                    f"(attempt {attempt+1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        raise last_exc

    def extract(self, slide: Dict) -> Dict:
        """단일 base 슬라이드에서 t1과 t1_structure를 추출한다."""
        slide.setdefault("title", f"Slide {slide['slide_number']}")
        slide.setdefault("t1", "")
        slide.setdefault("t1_structure", "")
        slide.setdefault("visual_assets", [])
        slide.setdefault("slide_emphasis", [])
        slide.setdefault("ocr_text", "")
        if slide.get("scene_type") == "video":
            slide["title"] = slide.get("title") or "영상 구간"
            slide["slide_type"] = "video"
            slide["has_teacher_annotation"] = False
            slide["text_source"] = "video"
            return slide

        # text_source/has_teacher_annotation은 SlideLoader가 이미 정확히 계산해
        # 뒀으므로 여기서 덮어쓰지 않는다.
        base_image = slide.get("base_image")
        build_images = slide.get("build_images") or []
        ocr_hint = ""
        if ocr_enabled():
            try:
                ocr_hint = get_slide_ocr_hint(slide.get("image_path") or "")
            except Exception as exc:
                logger.warning("  ⚠ OCR hint failed for slide %s: %s", slide.get("slide_number"), exc)
                ocr_hint = ""
        slide["ocr_text"] = ocr_hint

        try:
            raw_text = self._call_gemini(base_image, build_images=build_images, ocr_hint=ocr_hint)

            # 코드펜스 제거
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0]
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0]
            raw_text = raw_text.strip()

            # 문자열 값 내부의 제어문자 정제
            def clean_string_value(m):
                inner = m.group(1)
                inner = re.sub(r'[\n\r\t]', ' ', inner)
                inner = re.sub(r' {2,}', ' ', inner).strip()
                return f'"{inner}"'
            raw_text = re.sub(r'"((?:[^"\\]|\\.)*)"', clean_string_value, raw_text)

            # 파싱 → json_repair fallback
            try:
                result = json.loads(raw_text)
            except json.JSONDecodeError:
                if JSON_REPAIR_AVAILABLE:
                    result = json.loads(repair_json(raw_text))
                else:
                    raise

            slide["title"]          = result.get("title", f"Slide {slide['slide_number']}")
            slide["t1"]             = result.get("raw_text", "")
            slide["t1_structure"]   = result.get("structure", "")
            slide["slide_type"]     = result.get("slide_type", "text")
            slide["visual_assets"]  = self._normalize_visual_assets(
                result.get("visual_assets", [])
            )
            slide["slide_emphasis"] = []

        except Exception as e:
            logger.error(
                f"  ✗ t1 extraction failed for slide {slide['slide_number']}: {e}"
            )

        return slide

    def extract_batch(self, slides: List[Dict]) -> List[Dict]:
        unique_slides = len({s.get("slide_canonical_number", s["slide_number"]) for s in slides})
        workers = max(1, min(self.config.workers, unique_slides))
        logger.info(
            f"Extracting t1 from {len(slides)} scenes "
            f"(unique slides: {unique_slides}, workers: {workers})..."
        )

        # One base-image request per canonical slide. Repeated scenes share its result.
        canonical_representatives: Dict[int, Dict] = {}
        for slide in slides:
            if slide.get("scene_type") == "video":
                continue
            cache_key = int(slide.get("slide_canonical_number", slide["slide_number"]))
            canonical_representatives.setdefault(cache_key, slide)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="t1") as executor:
            futures = {
                executor.submit(self.extract, slide): cache_key
                for cache_key, slide in canonical_representatives.items()
            }
            for future in as_completed(futures):
                cache_key = futures[future]
                future.result()
                slide = canonical_representatives[cache_key]
                logger.info(
                    "  [t1 complete] Scene %s (slide %s): t1=%s chars, structure=%s chars, visual_assets=%s개",
                    slide["scene_number"],
                    cache_key,
                    len(slide["t1"]),
                    len(slide["t1_structure"]),
                    len(slide.get("visual_assets", [])),
                )

        cache: Dict[int, Dict] = {
            cache_key: {
                "title": slide["title"],
                "t1": slide["t1"],
                "t1_structure": slide["t1_structure"],
                "slide_type": slide.get("slide_type", "text"),
                "visual_assets": list(slide.get("visual_assets", [])),
                "slide_emphasis": [],
                "ocr_text": slide.get("ocr_text", ""),
                "representative_scene_number": slide.get("representative_scene_number", slide.get("scene_number")),
            }
            for cache_key, slide in canonical_representatives.items()
        }

        for i, slide in enumerate(slides):
            if slide.get("scene_type") == "video":
                self.extract(slide)
                logger.info(
                    f"  [{i+1}/{len(slides)}] Scene {slide['scene_number']} "
                    "(video, textualization skipped)"
                )
                continue

            cache_key = int(slide.get("slide_canonical_number", slide["slide_number"]))
            cached = cache[cache_key]
            slide["title"] = cached["title"]
            slide["t1"] = cached["t1"]
            slide["t1_structure"] = cached["t1_structure"]
            slide["slide_type"] = cached["slide_type"]
            slide["visual_assets"] = list(cached.get("visual_assets", []))
            slide["slide_emphasis"] = []
            slide["ocr_text"] = cached.get("ocr_text", "")
            if slide is not canonical_representatives[cache_key]:
                logger.info(
                    f"  [{i+1}/{len(slides)}] Scene {slide['scene_number']} "
                    f"(slide {cache_key}, cached from scene {cached['representative_scene_number']})"
                )

        logger.info("✓ t1 extraction complete")
        return slides


# ============================================================================ #
#  파이프라인                                                                    #
# ============================================================================ #

class TextualizationPipeline:
    """슬라이드 텍스트화 파이프라인"""

    def __init__(self, config: Config = None):
        self.config = config or Config()

    def run(self) -> Dict:
        start_time = time.time()

        print("\n" + "="*70)
        print("🎓 슬라이드 텍스트화 파이프라인 (Stage 1)")
        print("="*70)
        print(f"📁 Slides : {self.config.slides_dir}")
        print(f"📂 Output : {self.config.output_dir}")

        # Stage 1: base 슬라이드 로드
        print("\n" + "-"*70)
        print("Stage 1: 텍스트 추출 대상 슬라이드 로드 (annot 우선, base fallback)")
        print("-"*70)

        slides = SlideLoader(self.config.slides_dir).load()

        # Stage 2: t1 추출 (텍스트 + 구조)
        print("\n" + "-"*70)
        print("Stage 2: 텍스트화 (OpenAI Vision)")
        print("-"*70)

        slides = T1Extractor(self.config).extract_batch(slides)

        # Stage 3: 결과 저장
        print("\n" + "-"*70)
        print("Stage 3: 결과 저장")
        print("-"*70)

        for slide in slides:
            ts = slide["timestamp"]
            mins, secs = divmod(ts, 60)
            hrs, mins = divmod(mins, 60)
            slide["timestamp_formatted"] = f"{int(hrs):02d}:{int(mins):02d}:{secs:05.2f}"
            slide["slide_id"] = f"slide_{slide['slide_number']:03d}"
            slide["scene_id"] = f"scene/{int(slide.get('scene_number', slide['slide_number'])):04d}"

        slide_keyword_units = [
            {
                "text": s.get("t1") or "",
                "slide_id": s["slide_id"],
                "slide_number": s["slide_number"],
            }
            for s in slides
            if (s.get("t1") or "").strip()
        ]
        slide_topic_keyword_counts = get_topic_keyword_count_map(
            slide_keyword_units,
            min_freq=2,
            max_keywords=20,
            max_segment_ratio=1.0,
            min_keyword_len=2,
            candidate_pool_size=80,
            use_llm_filter=True,
        )
        for slide in slides:
            summary = summarize_topic_keyword_counts_for_text(
                slide.get("t1") or "",
                slide_topic_keyword_counts,
                min_keyword_len=2,
            )
            slide["slide_topic_keywords"] = summary["keywords"]
            slide["slide_topic_total_count_sum"] = summary["total_count_sum"]
            slide["slide_topic_keyword_scores"] = summary["keyword_scores"]
            slide["slide_topic_keyword_score"] = summary["score_sum"]

        total_time = time.time() - start_time

        result = {
            "metadata": {
                "slides_dir":      str(self.config.slides_dir),
                "processing_time": total_time,
                "total_scenes":    len(slides),
                "total_slides":    len({s.get("slide_canonical_number", s["slide_number"]) for s in slides}),
            },
            "slide_keyword_report": {
                "description": "slide_textualized t1 전체에서 추출한 반복 주제 키워드와 전체 등장 횟수",
                "min_freq": 2,
                "slide_topic_keywords": topic_keyword_count_items(slide_topic_keyword_counts),
            },
            "scenes": [
                {
                    "slide_id":            s["slide_id"],
                    "scene_id":            s.get("scene_id"),
                    "scene_type":          s.get("scene_type", "slide"),
                    "scene_number":        s.get("scene_number", s["slide_number"]),
                    "scene_index":         s.get("scene_index", s.get("scene_number", s["slide_number"])),
                    "slide_number":        s["slide_number"],
                    "slide_canonical_number": s.get("slide_canonical_number", s["slide_number"]),
                    "slide_visit_order":   s.get("slide_visit_order", 1),
                    "slide_is_revisit":    s.get("slide_is_revisit", False),
                    "representative_scene_number": s.get("representative_scene_number", s["slide_number"]),
                    "timestamp":           s["timestamp"],
                    "timestamp_formatted": s["timestamp_formatted"],
                    "image_path":          s["image_path"],
                    "title":               s["title"],
                    "t1":                  s["t1"],
                    "t1_structure":        s["t1_structure"],
                    "slide_type":          s.get("slide_type", "text"),
                    "visual_assets":       s.get("visual_assets", []),
                    "slide_emphasis":      [],
                    "ocr_text":            s.get("ocr_text", ""),
                    "slide_topic_keywords": s.get("slide_topic_keywords", []),
                    "slide_topic_total_count_sum": s.get("slide_topic_total_count_sum", 0),
                    "slide_topic_keyword_scores": s.get("slide_topic_keyword_scores", {}),
                    "slide_topic_keyword_score": s.get("slide_topic_keyword_score", 0),
                    "text_source":         s.get("text_source", "base"),
                    "has_teacher_annotation": s.get("has_teacher_annotation", False),
                }
                for s in slides
            ]
        }

        output_path = self.config.output_dir / self.config.output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved: {output_path}")

        structure_count = sum(1 for s in slides if s.get("t1_structure"))

        print("\n" + "="*70)
        print("✅ 텍스트화 완료!")
        print("="*70)
        print(f"\n📊 결과:")
        print(f"  • 슬라이드:      {len(slides)}개")
        image_only_count = sum(1 for s in slides if s.get("slide_type") == "image_only")
        mixed_count      = sum(1 for s in slides if s.get("slide_type") == "mixed")
        print(f"  • t1 추출:       {sum(1 for s in slides if s['t1'])}개")
        print(f"  • t1_structure:  {structure_count}개")
        print(f"  • image_only:    {image_only_count}개")
        print(f"  • mixed:         {mixed_count}개")
        print(f"\n📁 생성된 파일:")
        print(f"  • {output_path}")
        print(f"\n⏱️  처리 시간: {total_time:.2f}초")

        return result


# ============================================================================ #
#  메인                                                                         #
# ============================================================================ #

def main():
    import argparse
    from .config import DEFAULT_SLIDES_DIR, DEFAULT_OUTPUT_DIR

    parser = argparse.ArgumentParser(description="슬라이드 시각 정보 텍스트화")
    parser.add_argument("-s", "--slides", default=str(DEFAULT_SLIDES_DIR),
                        help=f"slide_extractor.py 출력 디렉토리 (default: {DEFAULT_SLIDES_DIR})")
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT_DIR),
                        help=f"결과 저장 디렉토리 (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--retries", type=int, default=3,
                        help="OpenAI API 재시도 횟수 (default: 3)")

    args = parser.parse_args()

    config = Config(
        slides_dir=Path(args.slides),
        output_dir=Path(args.output),
        max_retries=args.retries,
    )

    if not config.slides_dir.exists():
        print(f"❌ Slides folder not found: {config.slides_dir}")
        return

    TextualizationPipeline(config).run()


if __name__ == "__main__":
    main()
