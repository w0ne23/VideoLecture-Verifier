# 사람이 촬영된 영역을 마스킹해 슬라이드 비교/전환 판정에서 제외하는 유틸
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# 사람 마스크 / 사람 존재 마스크 캐시 디렉터리 이름
MASKS_DIRNAME = "person_masks"
PRESENCE_MASKS_DIRNAME = "person_presence_masks"


# 프레임 정보에 저장된 사람 마스크 파일 로드, 없으면 None
def load_person_mask(cache_dir: str | Path, frame_info: dict) -> np.ndarray | None:
    filename = frame_info.get("person_mask_filename")
    if not filename:
        return None
    path = Path(cache_dir) / str(filename)
    if not path.exists():
        return None
    mask = np.load(path, allow_pickle=False)
    return mask.astype(bool)


# 마스크를 목표 크기로 리사이즈, 이미 같은 크기면 그대로 반환
def resize_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    height, width = shape
    if mask.shape[:2] == (height, width):
        return mask.astype(bool)
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


# 두 마스크를 같은 크기로 맞춘 뒤 합집합(OR) 계산, 둘 다 없으면 None
def mask_union(mask_a: np.ndarray | None, mask_b: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    a = resize_mask(mask_a, shape)
    b = resize_mask(mask_b, shape)
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return np.logical_or(a, b)


# 두 프레임에서 사람 영역(마스크 합집합)을 0으로 채워 비교용으로 반환
def masked_pair(
    frame_a: np.ndarray,
    mask_a: np.ndarray | None,
    frame_b: np.ndarray,
    mask_b: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    ignore = mask_union(mask_a, mask_b, frame_a.shape[:2])
    if ignore is None or not bool(ignore.any()):
        return frame_a, frame_b
    a = frame_a.copy()
    b = frame_b.copy()
    a[ignore] = 0
    b[ignore] = 0
    return a, b
