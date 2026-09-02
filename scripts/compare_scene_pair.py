#!/usr/bin/env python3
# 인접 슬라이드 전환 구간의 OCR 유사도를 비교하는 디버그 스크립트
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import urllib.error

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.preprocess.ocr_hint import compare_slide_ocr, ocr_similarity_threshold


# 기본 검사 대상 씬 범위와 슬라이드 디렉터리
DEFAULT_SCENES = list(range(18, 32))
DEFAULT_SLIDES_DIR = REPO_ROOT / "storage" / "형법_full_run" / "slides_staged" / "review_slides"


# 씬의 마지막 주석(annot) 프레임 경로 조회, 없으면 예외
def last_annot_path(slides_dir: Path, scene: int) -> Path:
    annots = sorted(slides_dir.glob(f"scene_{scene:03d}_annot_*.jpg"))
    if not annots:
        raise FileNotFoundError(f"no annot frames for scene {scene:03d}")
    return annots[-1]


# 이전 씬의 마지막 주석 프레임과 다음 씬의 base 프레임 OCR 결과를 비교해 출력
def compare_transition(base_url: str, slides_dir: Path, prev_scene: int, next_scene: int) -> int:
    prev_base = slides_dir / f"scene_{prev_scene:03d}_base.jpg"
    next_base = slides_dir / f"scene_{next_scene:03d}_base.jpg"
    if not prev_base.exists():
        print(f"scene {prev_scene:03d}: base not found: {prev_base}", file=sys.stderr)
        return 2
    if not next_base.exists():
        print(f"scene {next_scene:03d}: base not found: {next_base}", file=sys.stderr)
        return 2

    try:
        annot = last_annot_path(slides_dir, prev_scene)
    except FileNotFoundError:
        annot = prev_base

    try:
        comparison = compare_slide_ocr(annot, next_base, base_url=base_url)
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"scene {prev_scene:03d}->{next_scene:03d}: OCR request failed: {exc}", file=sys.stderr)
        return 1

    text_a = str(comparison.get("left_text", ""))
    text_b = str(comparison.get("right_text", ""))
    ratio = float(comparison.get("similarity", 0.0) or 0.0)
    threshold = float(comparison.get("threshold", ocr_similarity_threshold()) or ocr_similarity_threshold())

    print(f"SCENE {prev_scene:03d} -> {next_scene:03d}")
    print(f"PREV  : {annot.name}")
    print(text_a)
    print(f"NEXT  : {next_base.name}")
    print(text_b)
    print(f"normalized_similarity={ratio:.4f}")
    print(f"threshold={threshold:.2f}")
    print(
        f"prev_len={len(str(comparison.get('left_normalized', '')))} "
        f"next_len={len(str(comparison.get('right_normalized', '')))}"
    )
    decision = str(comparison.get("decision", "uncertain"))
    print(f"decision={decision}")
    print()
    return 0


# 인접한 씬 쌍을 순회하며 전환 판정을 비교, 실패한 씬이 있으면 0이 아닌 코드 반환
def main() -> int:
    parser = argparse.ArgumentParser(description="이전 씬의 마지막 annot과 다음 씬의 base 프레임 OCR 텍스트 비교")
    parser.add_argument("scenes", nargs="*", type=int, default=DEFAULT_SCENES)
    parser.add_argument("--slides-dir", default=str(DEFAULT_SLIDES_DIR))
    parser.add_argument("--base-url", default="http://localhost:8010")
    args = parser.parse_args()

    slides_dir = Path(args.slides_dir)
    if not slides_dir.exists():
        print(f"slides dir not found: {slides_dir}", file=sys.stderr)
        return 2

    exit_code = 0
    for prev_scene, next_scene in zip(args.scenes, args.scenes[1:]):
        code = compare_transition(args.base_url, slides_dir, prev_scene, next_scene)
        if code != 0:
            exit_code = code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
