from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def build_transforms(image_size: int = 256, augment: bool = False) -> A.Compose:
    """Build paired image/mask transforms.

    The original scripts kept image and mask transforms separate. For supervised
    segmentation, paired transforms are safer because the same geometric random
    state is applied to both image and mask.
    """
    transforms: list[A.BasicTransform] = [A.Resize(image_size, image_size)]
    if augment:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.5),
            ]
        )

    transforms.extend(
        [
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)
