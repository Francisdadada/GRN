from __future__ import annotations

from grn.models.discriminator import PatchGANDiscriminator
from grn.models.generator import UNetGenerator
from grn.models.segmenter import build_segmenter


def build_grn_models(num_classes: int = 7):
    generator = UNetGenerator(input_channels=1, output_channels=1)
    discriminator = PatchGANDiscriminator(input_channels=1)
    segmenter = build_segmenter(in_channels=1, out_channels=num_classes)
    return generator, discriminator, segmenter
