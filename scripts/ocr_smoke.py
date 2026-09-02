#!/usr/bin/env python3
# OCR 서비스에 이미지 1장을 보내 정상 동작을 확인하는 스모크 테스트 스크립트
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
# 기본 테스트 이미지 경로
DEFAULT_IMAGE = REPO_ROOT / "storage" / "형법_full_run" / "slides_staged" / "review_slides" / "scene_007_base.jpg"


# 호스트 경로를 OCR 컨테이너 내부 경로(/app 기준)로 변환, 레포 밖 경로면 그대로 반환
def to_container_path(path: Path) -> str:
    path = path.expanduser().resolve()
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        return str(path)
    return str(Path("/app") / rel)


# 이미지를 OCR 서비스에 POST 요청으로 전송해 응답을 출력
def main() -> int:
    parser = argparse.ArgumentParser(description="이미지 1장으로 OCR 서비스 동작 확인")
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE), help="호스트 기준 이미지 파일 경로")
    parser.add_argument("--base-url", default=os.getenv("VLVERIFIER_SLIDE_OCR_BASE_URL", "http://localhost:8010"))
    parser.add_argument("--lang", default=os.getenv("VLVERIFIER_SLIDE_OCR_LANG", "multilingual"))
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"image not found: {image_path}", file=sys.stderr)
        return 2

    payload = {
        "image_path": to_container_path(image_path),
        "lang": args.lang,
    }

    req = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/ocr",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"OCR request failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(body, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
