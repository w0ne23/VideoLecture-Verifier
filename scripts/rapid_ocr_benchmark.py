#!/usr/bin/env python3
"""설정된 RapidOCR 서비스를 review-slide 이미지로 벤치마크"""

from __future__ import annotations

import argparse
import math
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


# 로컬 이미지 경로를 OCR 서비스 컨테이너 내부 경로(/app 기준)로 변환
def service_path(image: Path, repo_root: Path) -> str:
    return str(Path("/app") / image.resolve().relative_to(repo_root))


# OCR 서비스에 이미지 경로를 보내 인식 결과 조회
def request_ocr(base_url: str, image_path: str, timeout: float) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/ocr",
        data=json.dumps({"image_path": image_path, "lang": "korean"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


# /health 엔드포인트가 응답할 때까지 폴링, timeout 초과 시 예외
def wait_for_service(base_url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=2.0):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise SystemExit(f"RapidOCR service did not become ready within {timeout:.0f}s: {base_url}")


# 슬라이드 이미지 여러 장을 순차로 OCR 요청해 처리 시간/성공률 통계 산출
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("slides_dir", type=Path)
    parser.add_argument("--base-url", default="http://localhost:8003")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    slides_dir = args.slides_dir.resolve()
    images = sorted(path for path in slides_dir.glob("*.jpg") if path.is_file())[: max(0, args.limit)]
    if not images:
        raise SystemExit(f"no JPG files found: {slides_dir}")

    wait_for_service(args.base_url)

    wall_times: list[float] = []
    inference_times: list[float] = []
    nonempty = 0
    failures = 0
    started = time.perf_counter()
    for index, image in enumerate(images, start=1):
        try:
            before = time.perf_counter()
            body = request_ocr(args.base_url, service_path(image, repo_root), args.timeout)
            wall_times.append(time.perf_counter() - before)
            inference_times.append(float(body.get("elapsed_sec") or 0.0))
            nonempty += int(bool(str(body.get("text") or "").strip()))
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            failures += 1
            print(f"{index}/{len(images)} failed {image.name}: {exc}")
            continue
        if index % 10 == 0 or index == len(images):
            print(f"{index}/{len(images)} completed")

    elapsed = time.perf_counter() - started
    print(json.dumps({
        "images": len(images),
        "success": len(wall_times),
        "failures": failures,
        "nonempty_text": nonempty,
        "total_wall_sec": round(elapsed, 3),
        "wall_sec_mean": round(statistics.mean(wall_times), 4) if wall_times else None,
        "wall_sec_p95": round(sorted(wall_times)[max(0, math.ceil(len(wall_times) * 0.95) - 1)], 4) if wall_times else None,
        "engine_sec_mean": round(statistics.mean(inference_times), 4) if inference_times else None,
    }, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
