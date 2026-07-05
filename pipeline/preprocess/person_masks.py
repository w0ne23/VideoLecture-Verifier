from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


MASKS_DIRNAME = "person_masks"
PRESENCE_MASKS_DIRNAME = "person_presence_masks"


def load_person_mask(cache_dir: str | Path, frame_info: dict) -> np.ndarray | None:
    filename = frame_info.get("person_mask_filename")
    if not filename:
        return None
    path = Path(cache_dir) / str(filename)
    if not path.exists():
        return None
    mask = np.load(path, allow_pickle=False)
    return mask.astype(bool)


def resize_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    height, width = shape
    if mask.shape[:2] == (height, width):
        return mask.astype(bool)
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


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
