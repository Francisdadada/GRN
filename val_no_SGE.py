import os
import yaml
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import SimpleITK as sitk
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from monai.networks.nets import UNet
from scipy.stats import t
import torch.nn.functional as F
from utils.load_dataset_SEL import SegmentationDataset
from utils.function import compute_iou, compute_dice

# ---------------------------------- #
# Confidence Interval Computation
# ---------------------------------- #

def compute_confidence_interval(data, confidence=0.95):
    """
    Compute the confidence interval for a given dataset.

    Args:
        data (list or numpy.ndarray): Data points.
        confidence (float): Confidence level.

    Returns:
        tuple: (mean, margin of error)
    """
    data = np.array(data)
    data = data[~np.isnan(data)]  # Exclude NaN values
    n = len(data)
    if n > 1:
        mean = np.mean(data)
        std_err = np.std(data, ddof=1) / np.sqrt(n)  # Sample standard error
        t_crit = t.ppf((1 + confidence) / 2., n-1)  # t-critical value
        margin = t_crit * std_err
        return mean, margin
    else:
        return np.nan, np.nan

# ----------------------------- #
# Configuration Loader
# ----------------------------- #

def load_config(config_path):
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# ----------------------------- #
# Main Prediction and Evaluation
# ----------------------------- #

def main():
    # ----------------------- #
    # Load Configuration
    # ----------------------- #
    config_path = 'validate_no_SGE.yaml'  # Path to your configuration file
    config = load_config(config_path)

    # Extract Configuration Parameters
    csv_file = config['paths']['csv_file']
    model_weights_path = config['paths']['model_weights']
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    batch_size = config['dataloader']['batch_size']
    num_workers = config['dataloader']['num_workers']
    pin_memory = config['dataloader']['pin_memory']

    spatial_dims = config['model']['spatial_dims']
    in_channels = config['model']['in_channels']
    out_channels = config['model']['out_channels']
    channels = tuple(config['model']['channels'])
    strides = tuple(config['model']['strides'])

    num_classes = config['metrics']['num_classes']

    confidence_level = config.get('confidence_level', 0.95)  # Default to 95% if not specified

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------- #
    # Initialize Dataset and DataLoader
    # ----------------------- #
    val_dataset = SegmentationDataset(
        csv_file=csv_file,
        augment=False,  # No augmentation for validation
        supervised=True  # Assuming supervised dataset with masks
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    print(f"Dataset loaded with {len(val_dataset)} samples.")
    print(f"DataLoader initialized with batch size {batch_size} and {num_workers} workers.")

    # ----------------------- #
    # Initialize the Model
    # ----------------------- #
    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides
    ).to(device)

    # Load the Saved Model Weights
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded and set to evaluation mode.")

    # ----------------------- #
    # Metrics Initialization
    # ----------------------- #
    ious = []
    dices = []
    per_image_dice_means = []
    per_image_iou_means = []

    # ----------------------- #
    # Iterate Over the Validation DataLoader
    # ----------------------- #
    with torch.no_grad():
        for batch_idx, (images, masks, image_names) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)  # [B, C, H, W]
            masks = masks.to(device, non_blocking=True)    # [B, H, W]

            # Forward Pass
            outputs = model(images)  # [B, num_classes, H, W]
            # Apply Softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)
            # Get the predicted classes
            pred_masks = torch.argmax(probabilities, dim=1)  # [B, H, W]

            # Move tensors to CPU for metric computation
            pred_masks_cpu = pred_masks.cpu().numpy()
            true_masks_cpu = masks.cpu().numpy()

            for i in range(pred_masks_cpu.shape[0]):
                pred_mask = pred_masks_cpu[i]
                true_mask = true_masks_cpu[i]

                # Compute IoU and Dice for each class
                iou_scores = compute_iou(pred_mask, true_mask, num_classes=num_classes)
                dice_scores = compute_dice(pred_mask, true_mask, num_classes=num_classes)

                ious.append(iou_scores)
                dices.append(dice_scores)

                # Compute per-image mean over classes 1-6 (excluding background)
                per_image_dice_mean = np.nanmean(dice_scores[1:])
                per_image_iou_mean = np.nanmean(iou_scores[1:])

                per_image_dice_means.append(per_image_dice_mean)
                per_image_iou_means.append(per_image_iou_mean)

                print(f"Processed Image {batch_idx * batch_size + i + 1}/{len(val_dataset)}")

    # ----------------------- #
    # Save Per-Image Average Dice and IoU Scores to Text Files
    # ----------------------- #
    dice_output_file = os.path.join(output_dir, 'average_dice_scores.txt')
    with open(dice_output_file, 'w') as f:
        for idx, dice_mean in enumerate(per_image_dice_means):
            f.write(f"Image {idx + 1}: Average Dice = {dice_mean:.4f}\n")
    print(f"Average Dice scores saved to {dice_output_file}")

    iou_output_file = os.path.join(output_dir, 'average_iou_scores.txt')
    with open(iou_output_file, 'w') as f:
        for idx, iou_mean in enumerate(per_image_iou_means):
            f.write(f"Image {idx + 1}: Average IoU = {iou_mean:.4f}\n")
    print(f"Average IoU scores saved to {iou_output_file}")

    # ----------------------- #
    # Convert Lists to NumPy Arrays for Easier Manipulation
    # ----------------------- #
    ious_array = np.array(ious)    # Shape: [num_images, num_classes]
    dices_array = np.array(dices)  # Shape: [num_images, num_classes]

    # ----------------------- #
    # Compute Mean IoU and Dice per Class Across All Images
    # ----------------------- #
    mean_ious = np.nanmean(ious_array, axis=0)
    mean_dices = np.nanmean(dices_array, axis=0)
    overall_mean_iou = np.nanmean(mean_ious[1:])  # Excluding background
    overall_mean_dice = np.nanmean(mean_dices[1:])  # Excluding background

    print("\nMean IoU per class:")
    for cls in range(num_classes):
        print(f"Class {cls}: {mean_ious[cls]:.4f}")

    print("\nMean Dice per class:")
    for cls in range(num_classes):
        print(f"Class {cls}: {mean_dices[cls]:.4f}")

    print(f"\nOverall Mean IoU (Classes 1-6): {overall_mean_iou:.4f}")
    print(f"Overall Mean Dice (Classes 1-6): {overall_mean_dice:.4f}")

    # ----------------------- #
    # Compute Confidence Intervals for Dice per Class
    # ----------------------- #
    print("\nDice Coefficient per Class with 95% Confidence Intervals:")
    for cls in range(num_classes):
        cls_dices = dices_array[:, cls]
        mean, margin = compute_confidence_interval(cls_dices, confidence=confidence_level)
        if not np.isnan(mean):
            print(f"Class {cls}: Mean Dice = {mean:.4f} ± {margin:.4f}")
        else:
            print(f"Class {cls}: Not enough data to compute confidence interval")

    # ----------------------- #
    # Compute Confidence Intervals for IoU per Class
    # ----------------------- #
    print("\nIoU per Class with 95% Confidence Intervals:")
    for cls in range(num_classes):
        cls_ious = ious_array[:, cls]
        mean, margin = compute_confidence_interval(cls_ious, confidence=confidence_level)
        if not np.isnan(mean):
            print(f"Class {cls}: Mean IoU = {mean:.4f} ± {margin:.4f}")
        else:
            print(f"Class {cls}: Not enough data to compute confidence interval")

    # ----------------------- #
    # Compute Confidence Intervals for Overall Metrics over Classes 1-6
    # ----------------------- #
    print("\nOverall Metrics over Classes 1-6 with 95% Confidence Intervals:")

    # Dice
    mean, margin = compute_confidence_interval(per_image_dice_means, confidence=confidence_level)
    if not np.isnan(mean):
        print(f"Overall Mean Dice = {mean:.4f} ± {margin:.4f}")
    else:
        print("Not enough data to compute overall confidence interval for Dice")

    # IoU
    mean, margin = compute_confidence_interval(per_image_iou_means, confidence=confidence_level)
    if not np.isnan(mean):
        print(f"Overall Mean IoU = {mean:.4f} ± {margin:.4f}")
    else:
        print("Not enough data to compute overall confidence interval for IoU")

if __name__ == "__main__":
    main()

