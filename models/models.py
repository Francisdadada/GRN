import torch
import torch.nn as nn
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

# Define the UNet Encoder
class UNetEncoder(nn.Module):
    def __init__(self, input_channels=1, ngf=64):
        super(UNetEncoder, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        in_features = ngf
        for i in range(4):
            out_features = min(in_features * 2, 512)
            self.encoder_layers.append(nn.Sequential(
                ResidualBlock(in_features),
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU(inplace=True)
            ))
            in_features = out_features

        # Bottleneck layers
        self.bottleneck = nn.Sequential(
            ResidualBlock(in_features),
            ResidualBlock(in_features),
            ResidualBlock(in_features)
        )

    def forward(self, x):
        x = self.initial(x)
        skips = []
        for layer in self.encoder_layers:
            skips.append(x)
            x = layer(x)
        x = self.bottleneck(x)
        return x, skips

# Define the UNet Decoder
class UNetDecoder(nn.Module):
    def __init__(self, output_channels=1, ngf=64):
        super(UNetDecoder, self).__init__()

        self.decoder_layers = nn.ModuleList()

        # Starting from the bottleneck
        # The number of channels after each decoder layer (before concatenation)
        decoder_channels = [512, 256, 128, 64]

        # The number of channels in the skip connections (from encoder)
        skip_channels = [512, 256, 128, 64]  # From deepest to shallowest

        # Initialize in_channels for the first decoder layer
        in_channels = 512  # Output channels from the bottleneck

        for idx in range(len(decoder_channels)):
            out_channels = decoder_channels[idx]
            self.decoder_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            # After concatenation, in_channels for the next layer increases
            in_channels = out_channels + skip_channels[idx]

        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(in_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x, skips):
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if i < len(skips):
                x = torch.cat([x, skips[-(i + 1)]], dim=1)  # Concatenate skip connection
        x = self.final(x)
        return x

# Define the Generator (G)
class UNetGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, ngf=64):
        super(UNetGenerator, self).__init__()
        self.encoder = UNetEncoder(input_channels, ngf)
        self.decoder = UNetDecoder(output_channels, ngf)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        return x

# Define the Discriminator (D) - PatchGANDiscriminator
class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=1, ndf=64, n_layers=3):
        super(PatchGANDiscriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                    kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                kernel_size=4, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(ndf * nf_mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)