from types import SimpleNamespace

import numpy as np

from pipeline.preprocess.sample_cache import _motion_filter_person_boxes


def _config():
    return SimpleNamespace(
        person_mask_fixed_min_center_shift_px=4.0,
        person_mask_fixed_min_area_change_ratio=0.08,
        person_mask_fixed_motion_min_mean_diff=2.0,
        person_mask_fixed_motion_min_changed_ratio=0.01,
        person_mask_match_iou_threshold=0.05,
        person_mask_roi_max_gap_sec=2.0,
        sample_fps=2.0,
    )


def test_fixed_presenter_is_confirmed_by_localized_internal_motion():
    first = np.zeros((120, 120, 3), dtype=np.uint8)
    second = first.copy()
    second[35:75, 30:50] = 255
    box = (20, 20, 60, 90)

    filtered, _ = _motion_filter_person_boxes(
        [first, second], [[box], [box]], _config()
    )

    assert filtered == [[], [box]]


def test_full_slide_transition_does_not_confirm_static_box():
    first = np.zeros((120, 120, 3), dtype=np.uint8)
    second = np.full((120, 120, 3), 255, dtype=np.uint8)
    box = (20, 20, 60, 90)

    filtered, _ = _motion_filter_person_boxes(
        [first, second], [[box], [box]], _config()
    )

    assert filtered == [[], []]


def test_static_portrait_remains_vetoed():
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    box = (20, 20, 60, 90)

    filtered, _ = _motion_filter_person_boxes(
        [frame, frame.copy()], [[box], [box]], _config()
    )

    assert filtered == [[], []]
