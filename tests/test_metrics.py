from __future__ import annotations

import numpy as np

from grn.inference.metrics import compute_dice_single, compute_iou_single


def test_iou_and_dice_perfect_match() -> None:
    pred = np.array([[0, 1], [1, 2]])
    true = np.array([[0, 1], [1, 2]])

    assert compute_iou_single(pred, true, num_classes=3) == [1.0, 1.0, 1.0]
    assert compute_dice_single(pred, true, num_classes=3) == [1.0, 1.0, 1.0]


def test_iou_and_dice_partial_match() -> None:
    pred = np.array([[0, 1], [0, 2]])
    true = np.array([[0, 1], [1, 2]])

    iou = compute_iou_single(pred, true, num_classes=3)
    dice = compute_dice_single(pred, true, num_classes=3)

    assert iou[1] == 0.5
    assert round(dice[1], 4) == 0.6667
