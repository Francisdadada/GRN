from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from grn.inference.overlays import blend_overlay, colorize_mask
from grn.inference.predictor import GRNPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRN inference on a single image.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--generator", required=True, help="Generator checkpoint path.")
    parser.add_argument("--segmenter", required=True, help="Segmenter checkpoint path.")
    parser.add_argument("--output", default="outputs/prediction.png", help="Output PNG path.")
    parser.add_argument("--overlay", action="store_true", help="Save color overlay instead of mask only.")
    parser.add_argument("--num-classes", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = GRNPredictor(args.generator, args.segmenter, num_classes=args.num_classes)
    image = Image.open(args.image).convert("L")
    mask = predictor.predict_image(image)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overlay:
        result = blend_overlay(image.convert("RGBA").resize((mask.shape[1], mask.shape[0])), mask)
    else:
        result = colorize_mask(mask)
    result.save(output_path)
    print(f"Saved prediction to {output_path}")


if __name__ == "__main__":
    main()
