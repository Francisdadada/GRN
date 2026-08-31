from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.stats import t


def compute_iou_single(pred_mask: np.ndarray, true_mask: np.ndarray, num_classes: int) -> list[float]:
    ious = []
    for cls in range(num_classes):
        pred_inds = pred_mask == cls
        target_inds = true_mask == cls
        union = np.logical_or(pred_inds, target_inds).sum()
        inter = np.logical_and(pred_inds, target_inds).sum()
        ious.append(np.nan if union == 0 else float(inter / union))
    return ious


def compute_dice_single(pred_mask: np.ndarray, true_mask: np.ndarray, num_classes: int) -> list[float]:
    dices = []
    for cls in range(num_classes):
        pred_inds = pred_mask == cls
        target_inds = true_mask == cls
        total = pred_inds.sum() + target_inds.sum()
        inter = np.logical_and(pred_inds, target_inds).sum()
        dices.append(np.nan if total == 0 else float((2.0 * inter) / total))
    return dices


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return mask ^ eroded


def compute_asd_hd95_for_binary(
    pred_bin: np.ndarray, gt_bin: np.ndarray, spacing: tuple[float, float] = (1.0, 1.0)
) -> tuple[float, float]:
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return np.nan, np.nan

    surf_pred = _surface(pred_bin)
    surf_gt = _surface(gt_bin)
    if surf_pred.sum() == 0 or surf_gt.sum() == 0:
        return np.nan, np.nan

    dt_to_pred = distance_transform_edt(~surf_pred, sampling=spacing)
    dt_to_gt = distance_transform_edt(~surf_gt, sampling=spacing)
    distances = np.concatenate([dt_to_gt[surf_pred], dt_to_pred[surf_gt]]).astype(np.float64)
    if distances.size == 0:
        return np.nan, np.nan
    return float(np.mean(distances)), float(np.percentile(distances, 95))


def compute_asd_hd95_per_class(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    num_classes: int,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> tuple[list[float], list[float]]:
    asds, hd95s = [], []
    for cls in range(num_classes):
        asd, hd95 = compute_asd_hd95_for_binary(pred_mask == cls, true_mask == cls, spacing)
        asds.append(asd)
        hd95s.append(hd95)
    return asds, hd95s


def ci_mean_bounds(data, alpha: float = 0.05) -> tuple[float, float, float, float]:
    values = np.asarray(data, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) <= 1:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    margin = float(t.ppf(1 - alpha / 2, df=len(values) - 1) * se)
    return mean, margin, mean - margin, mean + margin
