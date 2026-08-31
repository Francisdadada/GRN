from __future__ import annotations

import torch

from grn.models.discriminator import PatchGANDiscriminator
from grn.models.generator import UNetGenerator
from grn.models.segmenter import build_segmenter


def test_generator_output_shape() -> None:
    model = UNetGenerator()
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    assert y.shape == x.shape


def test_discriminator_forward_shape() -> None:
    model = PatchGANDiscriminator()
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    assert y.shape[0] == 1
    assert y.shape[1] == 1


def test_segmenter_output_classes() -> None:
    model = build_segmenter(out_channels=7)
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    assert y.shape == (1, 7, 256, 256)
