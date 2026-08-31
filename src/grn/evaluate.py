from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from grn.data.dataset import SegmentationDataset
from grn.inference.metrics import (
    ci_mean_bounds,
    compute_asd_hd95_per_class,
    compute_dice_single,
    compute_iou_single,
)
from grn.inference.overlays import blend_overlay, tensor_to_rgba_pil
from grn.models.generator import UNetGenerator
from grn.models.segmenter import build_segmenter


def evaluate_from_config(config: dict) -> None:
    device = torch.device(config.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model_cfg = config.get("model", {})
    data_cfg = config["data"]
    weights_cfg = config["weights"]
    output_cfg = config.get("outputs", {})
    num_classes = int(model_cfg.get("num_classes", 7))

    dataset = SegmentationDataset(
        data_cfg["images"],
        data_cfg["masks"],
        augment=False,
        image_size=int(data_cfg.get("image_size", 256)),
    )
    loader = DataLoader(dataset, batch_size=int(data_cfg.get("batch_size", 1)), shuffle=False, num_workers=0)

    generator = UNetGenerator(input_channels=1, output_channels=1).to(device)
    segmenter = build_segmenter(in_channels=1, out_channels=num_classes).to(device)
    generator.load_state_dict(torch.load(weights_cfg["generator"], map_location=device))
    segmenter.load_state_dict(torch.load(weights_cfg["segmenter"], map_location=device))
    generator.eval()
    segmenter.eval()

    pred_dir = Path(output_cfg.get("prediction_overlay_dir", "outputs/predictions"))
    gt_dir = Path(output_cfg.get("ground_truth_overlay_dir", "outputs/ground_truth"))
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    ious_list, dices_list, asds_list, hd95_list = [], [], [], []
    overall_iou, overall_dice, overall_asd, overall_hd95 = [], [], [], []

    with torch.no_grad():
        for images, masks, names in loader:
            images = images.to(device)
            masks = masks.to(device)
            pred = torch.argmax(segmenter(generator(images)), dim=1).cpu().numpy()
            true = masks.cpu().numpy()[:, 0, :, :]

            for b in range(pred.shape[0]):
                p, g = pred[b], true[b]
                ious = compute_iou_single(p, g, num_classes)
                dices = compute_dice_single(p, g, num_classes)
                asds, hd95s = compute_asd_hd95_per_class(p, g, num_classes)
                ious_list.append(ious)
                dices_list.append(dices)
                asds_list.append(asds)
                hd95_list.append(hd95s)
                overall_iou.append(np.nanmean(ious[1:]))
                overall_dice.append(np.nanmean(dices[1:]))
                overall_asd.append(np.nanmean(asds[1:]))
                overall_hd95.append(np.nanmean(hd95s[1:]))

                stem = Path(names[b]).stem
                base = tensor_to_rgba_pil(images[b])
                blend_overlay(base, p).save(pred_dir / f"pred_{stem}.png")
                blend_overlay(base, g).save(gt_dir / f"gt_{stem}.png")

    _print_metrics("IoU", np.asarray(ious_list, dtype=float), num_classes)
    _print_metrics("Dice", np.asarray(dices_list, dtype=float), num_classes)
    _print_metrics("ASD", np.asarray(asds_list, dtype=float), num_classes)
    _print_metrics("HD95", np.asarray(hd95_list, dtype=float), num_classes)
    for label, values in [
        ("Overall IoU", overall_iou),
        ("Overall Dice", overall_dice),
        ("Overall ASD", overall_asd),
        ("Overall HD95", overall_hd95),
    ]:
        mean, _margin, lo, hi = ci_mean_bounds(values)
        print(f"{label}: mean={mean:.4f} CI=({lo:.4f}, {hi:.4f})")


def _print_metrics(name: str, arr: np.ndarray, num_classes: int) -> None:
    print(f"\n--- {name} per class with 95% CI ---")
    for c in range(num_classes):
        mean, _margin, lo, hi = ci_mean_bounds(arr[:, c])
        print(f"Class {c}: mean={mean:.4f} CI=({lo:.4f}, {hi:.4f})")
