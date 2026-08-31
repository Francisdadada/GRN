from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UNetEncoder(nn.Module):
    def __init__(self, input_channels: int = 1, ngf: int = 64) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder_layers = nn.ModuleList()
        in_features = ngf
        for _ in range(4):
            out_features = min(in_features * 2, 512)
            self.encoder_layers.append(
                nn.Sequential(
                    ResidualBlock(in_features),
                    nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_features),
                    nn.ReLU(inplace=True),
                )
            )
            in_features = out_features

        self.bottleneck = nn.Sequential(
            ResidualBlock(in_features),
            ResidualBlock(in_features),
            ResidualBlock(in_features),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.initial(x)
        skips = []
        for layer in self.encoder_layers:
            skips.append(x)
            x = layer(x)
        return self.bottleneck(x), skips


class UNetDecoder(nn.Module):
    def __init__(self, output_channels: int = 1) -> None:
        super().__init__()
        decoder_channels = [512, 256, 128, 64]
        skip_channels = [512, 256, 128, 64]
        in_channels = 512
        self.decoder_layers = nn.ModuleList()
        for out_channels, skip_channels_i in zip(decoder_channels, skip_channels):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = out_channels + skip_channels_i

        self.final = nn.Sequential(
            nn.ConvTranspose2d(in_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if i < len(skips):
                x = torch.cat([x, skips[-(i + 1)]], dim=1)
        return self.final(x)


class UNetGenerator(nn.Module):
    def __init__(self, input_channels: int = 1, output_channels: int = 1, ngf: int = 64) -> None:
        super().__init__()
        self.encoder = UNetEncoder(input_channels=input_channels, ngf=ngf)
        self.decoder = UNetDecoder(output_channels=output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, skips = self.encoder(x)
        return self.decoder(encoded, skips)
