from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from torch.utils.data import Dataset

from grn.data.transforms import build_transforms

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".nii", ".nii.gz")


def load_image_file(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


def load_sitk_file(path: str | Path) -> np.ndarray:
    sitk_image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(sitk_image)


def resolve_existing_dir(*candidates: str | Path | None) -> Path:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these directories exist: {candidates}")


class SegmentationDataset(Dataset):
    """2D image segmentation dataset for grayscale images and NIfTI masks."""

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path | None = None,
        augment: bool = False,
        image_size: int = 256,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if self.mask_dir and not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.images = sorted(
            f for f in os.listdir(self.image_dir) if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        self.transform = build_transforms(image_size=image_size, augment=augment)

    def __len__(self) -> int:
        return len(self.images)

    def _mask_path_for(self, image_name: str) -> Path:
        if self.mask_dir is None:
            raise ValueError("mask_dir is not configured")
        stem = image_name.removesuffix(".nii.gz")
        if stem == image_name:
            stem = Path(image_name).stem
        candidates = [self.mask_dir / f"{stem}.nii", self.mask_dir / f"{stem}.nii.gz"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No mask found for image {image_name} in {self.mask_dir}")

    def __getitem__(self, idx: int):
        image_name = self.images[idx]
        image = load_image_file(self.image_dir / image_name)

        if self.mask_dir:
            mask_array = np.squeeze(load_sitk_file(self._mask_path_for(image_name))).astype(np.uint8)
            transformed = self.transform(image=np.array(image), mask=mask_array)
            image_tensor = transformed["image"]
            mask_tensor = transformed["mask"].long().unsqueeze(0)
        else:
            transformed = self.transform(image=np.array(image))
            image_tensor = transformed["image"]
            mask_tensor = None

        return image_tensor, mask_tensor, image_name


def collate_unlabeled(batch):
    images, names = [], []
    for image, _mask, name in batch:
        images.append(image)
        names.append(name)
    return torch.stack(images, dim=0), names
