from __future__ import annotations

import numpy as np
import torch
from PIL import Image

COLORS = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def tensor_to_rgba_pil(img_tensor: torch.Tensor) -> Image.Image:
    img = np.squeeze(img_tensor.detach().cpu().numpy())
    img_min, img_max = float(img.min()), float(img.max())
    if img_max - img_min > 0:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = img - img_min
    return Image.fromarray((img * 255).astype(np.uint8)).convert("RGBA")


def colorize_mask(mask_2d: np.ndarray, colors: list[tuple[int, int, int]] | None = None) -> Image.Image:
    palette = colors or COLORS
    h, w = mask_2d.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(palette):
        color_mask[mask_2d == cls_idx] = color
    return Image.fromarray(color_mask).convert("RGBA")


def blend_overlay(base: Image.Image, mask_2d: np.ndarray, alpha: float = 0.3) -> Image.Image:
    return Image.blend(base.convert("RGBA"), colorize_mask(mask_2d), alpha=alpha)
