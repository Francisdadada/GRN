from __future__ import annotations

from monai.networks.nets import UNet


def build_segmenter(in_channels: int = 1, out_channels: int = 7) -> UNet:
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    )
