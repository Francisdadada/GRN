from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from grn.data.transforms import build_transforms
from grn.models.generator import UNetGenerator
from grn.models.segmenter import build_segmenter


class GRNPredictor:
    def __init__(
        self,
        generator_weights: str | Path,
        segmenter_weights: str | Path,
        num_classes: int = 7,
        device: str | torch.device | None = None,
        image_size: int = 256,
    ) -> None:
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.transform = build_transforms(image_size=image_size, augment=False)
        self.generator = UNetGenerator(input_channels=1, output_channels=1).to(self.device)
        self.segmenter = build_segmenter(in_channels=1, out_channels=num_classes).to(self.device)
        self.generator.load_state_dict(torch.load(generator_weights, map_location=self.device))
        self.segmenter.load_state_dict(torch.load(segmenter_weights, map_location=self.device))
        self.generator.eval()
        self.segmenter.eval()

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        transformed = self.transform(image=np.array(image.convert("L")))
        return transformed["image"].unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_image(self, image: Image.Image) -> np.ndarray:
        input_tensor = self.preprocess(image)
        generated = self.generator(input_tensor)
        logits = self.segmenter(generated)
        return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    def predict_path(self, image_path: str | Path) -> np.ndarray:
        return self.predict_image(Image.open(image_path))
