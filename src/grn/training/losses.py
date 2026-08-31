from __future__ import annotations

from monai.losses import DiceLoss


def build_segmentation_loss(include_background: bool = True) -> DiceLoss:
    return DiceLoss(to_onehot_y=True, softmax=True, include_background=include_background)
