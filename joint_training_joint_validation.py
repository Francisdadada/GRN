import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from monai.networks.nets import UNet
from monai.metrics import DiceMetric  # (not used below, but you can keep)

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.stats import t

from PIL import Image

from utils.utils import SegmentationDataset, get_transforms

# ---------------- Residual + UNet blocks (unchanged) ----------------
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

class UNetEncoder(nn.Module):
    def __init__(self, input_channels=1, ngf=64):
        super(UNetEncoder, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
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

class UNetDecoder(nn.Module):
    def __init__(self, output_channels=1, ngf=64):
        super(UNetDecoder, self).__init__()
        self.decoder_layers = nn.ModuleList()
        decoder_channels = [512, 256, 128, 64]
        skip_channels = [512, 256, 128, 64]
        in_channels = 512
        for idx in range(len(decoder_channels)):
            out_channels = decoder_channels[idx]
            self.decoder_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            in_channels = out_channels + skip_channels[idx]
        self.final = nn.Sequential(
            nn.ConvTranspose2d(in_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x, skips):
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if i < len(skips):
                x = torch.cat([x, skips[-(i + 1)]], dim=1)
        x = self.final(x)
        return x

class UNetGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, ngf=64):
        super(UNetGenerator, self).__init__()
        self.encoder = UNetEncoder(input_channels, ngf)
        self.decoder = UNetDecoder(output_channels, ngf)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        return x

# ---------------------- Metrics helpers ----------------------
PIXEL_SPACING = (1.0, 1.0)  # set (row_spacing, col_spacing) if known

def compute_iou_single(pred_mask, true_mask, num_classes):
    """pred_mask, true_mask: [H,W] integer labels"""
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred_mask == cls)
        target_inds = (true_mask == cls)
        inter = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        ious.append(np.nan if union == 0 else inter / union)
    return ious

def compute_dice_single(pred_mask, true_mask, num_classes):
    dices = []
    for cls in range(num_classes):
        pred_inds = (pred_mask == cls)
        target_inds = (true_mask == cls)
        inter = np.logical_and(pred_inds, target_inds).sum()
        total = pred_inds.sum() + target_inds.sum()
        dices.append(np.nan if total == 0 else (2.0 * inter) / total)
    return dices

def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return mask ^ eroded

def compute_asd_hd95_for_binary(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing=(1.0, 1.0)):
    """Symmetric ASD (mean surface distance) and HD95 (95th percentile)."""
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return np.nan, np.nan

    surf_pred = _surface(pred_bin)
    surf_gt   = _surface(gt_bin)
    if surf_pred.sum() == 0 or surf_gt.sum() == 0:
        return np.nan, np.nan

    dt_to_pred = distance_transform_edt(~surf_pred, sampling=spacing)
    dt_to_gt   = distance_transform_edt(~surf_gt,   sampling=spacing)

    d_pred_to_gt = dt_to_gt[surf_pred]
    d_gt_to_pred = dt_to_pred[surf_gt]
    all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred]).astype(np.float64)
    if all_dists.size == 0:
        return np.nan, np.nan

    asd  = float(np.mean(all_dists))
    hd95 = float(np.percentile(all_dists, 95))
    return asd, hd95

def compute_asd_hd95_per_class(pred_mask: np.ndarray, true_mask: np.ndarray, num_classes: int, spacing=(1.0,1.0)):
    asds, hd95s = [], []
    for cls in range(num_classes):
        asd, hd95 = compute_asd_hd95_for_binary(pred_mask == cls, true_mask == cls, spacing=spacing)
        asds.append(asd)
        hd95s.append(hd95)
    return asds, hd95s

def ci_mean_bounds(data, alpha=0.05):
    """Mean ± 95% CI (Student t). Returns (mean, margin, lower, upper). NaNs ignored."""
    x = np.asarray(data, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n <= 1:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    se   = float(np.std(x, ddof=1) / np.sqrt(n))
    tcrit = float(t.ppf(1 - alpha/2, df=n-1))
    margin = tcrit * se
    return mean, margin, mean - margin, mean + margin

# ---------------------- Overlay helpers (same as your first code) ----------------------
colors = [
    (0, 0, 0),        # Class 0: Background - Black
    (255, 0, 0),      # Class 1: Red
    (0, 255, 0),      # Class 2: Green
    (0, 0, 255),      # Class 3: Blue
    (255, 255, 0),    # Class 4: Yellow
    (255, 0, 255),    # Class 5: Magenta
    (0, 255, 255)     # Class 6: Cyan
]
alpha = 0.3

def tensor_to_rgba_pil(img_tensor: torch.Tensor) -> Image.Image:
    """
    img_tensor: [1,H,W] or [H,W]
    normalize to 0..255 uint8, convert to RGBA PIL (same behavior as your first code)
    """
    img = img_tensor.detach().cpu().numpy()
    img = np.squeeze(img)  # [H,W]
    img_min, img_max = float(img.min()), float(img.max())
    if img_max - img_min > 0:
        img_norm = (img - img_min) / (img_max - img_min)
    else:
        img_norm = img - img_min
    img_u8 = (img_norm * 255).astype(np.uint8)
    return Image.fromarray(img_u8).convert("RGB").convert("RGBA")

def colorize_mask(mask_2d: np.ndarray) -> Image.Image:
    """mask_2d: [H,W] integer labels -> RGBA colored mask PIL"""
    h, w = mask_2d.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(len(colors)):
        color_mask[mask_2d == cls_idx] = colors[cls_idx]
    return Image.fromarray(color_mask).convert("RGBA")

# --------------------------- Main ---------------------------
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Output dirs (requested)
    pred_output_dir = "DatasetA/val/prediction_GRN_SEL_SGE"
    gt_output_dir   = "DatasetA/val/ground_truth_GRN_SEL_SGE"
    os.makedirs(pred_output_dir, exist_ok=True)
    os.makedirs(gt_output_dir, exist_ok=True)

    val_dataset = SegmentationDataset(
        image_dir='dataset/new_test_img',
        mask_dir='dataset/new_test_msk',
        augment=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Models
    G = UNetGenerator(input_channels=1, output_channels=1).to(device)
    S = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=7,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2)
    ).to(device)

    G.load_state_dict(torch.load('model_weights/0.05/0.05joingtraining_aug_noenhance/best_G.pth', map_location=device))
    S.load_state_dict(torch.load('model_weights/0.05/0.05joingtraining_aug_noenhance/best_S.pth', map_location=device))
    G.eval(); S.eval()

    num_classes = 7

    # Collect per-image, per-class metrics
    ious_list, dices_list = [], []
    asds_list, hd95_list  = [], []

    # Also collect per-image overall means (classes 1–6) for each metric
    overall_iou_means, overall_dice_means = [], []
    overall_asd_means, overall_hd95_means = [], []

    global_image_counter = 0

    with torch.no_grad():
        for val_batch in val_loader:
            val_images, val_masks, meta = val_batch
            val_images = val_images.to(device)
            val_masks  = val_masks.to(device)

            # Generate augmented images and segment
            val_I_prime = G(val_images)
            val_M_prime = S(val_I_prime)

            # Convert to label maps
            pred_mask = torch.argmax(val_M_prime, dim=1).cpu().numpy()        # [B,H,W]
            true_mask = val_masks.cpu().numpy()[:, 0, :, :]                   # [B,H,W]

            # Per-image calculation (batch size may be >1)
            for b in range(pred_mask.shape[0]):
                p = pred_mask[b]
                g = true_mask[b]

                # Per-class metrics
                ious  = compute_iou_single(p, g, num_classes)
                dices = compute_dice_single(p, g, num_classes)
                asds, hd95s = compute_asd_hd95_per_class(p, g, num_classes, spacing=PIXEL_SPACING)

                ious_list.append(ious)
                dices_list.append(dices)
                asds_list.append(asds)
                hd95_list.append(hd95s)

                # Per-image overall means (exclude background class 0)
                overall_iou_means.append(np.nanmean(ious[1:]))
                overall_dice_means.append(np.nanmean(dices[1:]))
                overall_asd_means.append(np.nanmean(asds[1:]))
                overall_hd95_means.append(np.nanmean(hd95s[1:]))

                # ---------------- Overlay saving (same as your first code) ----------------
                # IMPORTANT: overlay should be on the ORIGINAL input image (val_images), not val_I_prime
                base_img = tensor_to_rgba_pil(val_images[b])

                pred_mask_pil = colorize_mask(p)
                blended_pred  = Image.blend(base_img, pred_mask_pil, alpha=alpha)

                gt_mask_pil   = colorize_mask(g)
                blended_gt    = Image.blend(base_img, gt_mask_pil, alpha=alpha)

                # Try to use meta as filename stem if available; otherwise fallback to counter
                stem = None
                try:
                    if isinstance(meta, (list, tuple)) and len(meta) == pred_mask.shape[0]:
                        m = meta[b]
                        if isinstance(m, str):
                            stem = os.path.splitext(os.path.basename(m))[0]
                    elif isinstance(meta, str):
                        stem = os.path.splitext(os.path.basename(meta))[0]
                except Exception:
                    stem = None

                if stem is None or stem == "":
                    stem = f"{global_image_counter}"

                blended_pred.save(os.path.join(pred_output_dir, f"pred_{stem}.png"))
                blended_gt.save(os.path.join(gt_output_dir,   f"gt_{stem}.png"))

                global_image_counter += 1

    # Convert to arrays: [N_images, num_classes]
    ious_arr  = np.asarray(ious_list, dtype=float)
    dices_arr = np.asarray(dices_list, dtype=float)
    asds_arr  = np.asarray(asds_list, dtype=float)
    hd95_arr  = np.asarray(hd95_list, dtype=float)

    # Means per class
    mean_iou_per_class  = np.nanmean(ious_arr,  axis=0)
    mean_dice_per_class = np.nanmean(dices_arr, axis=0)
    mean_asd_per_class  = np.nanmean(asds_arr,  axis=0)
    mean_hd95_per_class = np.nanmean(hd95_arr,  axis=0)

    print("\n=== Mean metrics per class (across images) ===")
    print("IoU  :", mean_iou_per_class)
    print("Dice :", mean_dice_per_class)
    print("ASD  :", mean_asd_per_class,  "(lower is better)")
    print("HD95 :", mean_hd95_per_class, "(lower is better)")

    # Per-class 95% CI
    def print_ci(title, arr):
        print(f"\n--- {title} per class with 95% CI ---")
        for c in range(num_classes):
            mean, margin, lo, hi = ci_mean_bounds(arr[:, c])
            if np.isnan(mean):
                print(f"Class {c}: Not enough data")
            else:
                print(f"Class {c}: mean={mean:.4f}  CI=({lo:.4f}, {hi:.4f})")

    print_ci("IoU",  ious_arr)
    print_ci("Dice", dices_arr)
    print_ci("ASD",  asds_arr)
    print_ci("HD95", hd95_arr)

    # Overall (classes 1–6) CI based on per-image means
    def print_overall(name, values):
        mean, margin, lo, hi = ci_mean_bounds(values)
        if np.isnan(mean):
            print(f"{name}: Not enough data")
        else:
            print(f"{name}: mean={mean:.4f}  CI=({lo:.4f}, {hi:.4f})")

    print("\n=== Overall (classes 1–6) with 95% CI ===")
    print_overall("Overall IoU",  overall_iou_means)
    print_overall("Overall Dice", overall_dice_means)
    print_overall("Overall ASD",  overall_asd_means)
    print_overall("Overall HD95", overall_hd95_means)

    print("\nMean Dice over classes 1–6 (simple average across class means):",
          float(np.nanmean(mean_dice_per_class[1:])))

    print(f"\nSaved overlays to:\n  {pred_output_dir}\n  {gt_output_dir}")

if __name__ == "__main__":
    main()
