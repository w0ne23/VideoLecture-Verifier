#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / "storage" / "형법_full_run" / "slides_staged" / "review_slides" / "scene_007_base.jpg"


def to_container_path(path: Path) -> str:
    path = path.expanduser().resolve()
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        return str(path)
    return str(Path("/app") / rel)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the OCR service with one image.")
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE), help="Host path to an image file")
    parser.add_argument("--base-url", default=os.getenv("GRAPHLEC_SLIDE_OCR_BASE_URL", "http://localhost:8010"))
    parser.add_argument("--lang", default=os.getenv("GRAPHLEC_SLIDE_OCR_LANG", "multilingual"))
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
