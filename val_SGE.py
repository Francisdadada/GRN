import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from monai.networks.nets import UNet
import numpy as np
from PIL import Image  # Import PIL for image saving

# Import the updated utility functions
from utils.load_dataset_SSL import SegmentationDataset
from scipy.stats import bootstrap

import argparse
import yaml

from models.models import UNetEncoder, UNetDecoder, UNetGenerator, PatchGANDiscriminator
from utils.function import compute_iou, compute_dice

# Define the Generator (G) (reuse from your training script)


def compute_confidence_interval(data, alpha=0.05):
    # Bootstrap method for confidence interval
    boot_result = bootstrap((data,), np.nanmean, confidence_level=1-alpha, n_resamples=1000, method='basic')
    return boot_result.confidence_interval


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Inference with config file')
    parser.add_argument('--config', type=str, default='validate_SGE.yaml', help='Path to the config file')
    args = parser.parse_args()

    # Load hyperparameters from config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device configuration
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    # Initialize validation dataset and dataloader
    val_dataset = SegmentationDataset(
        csv_file=config['dataset']['test_csv'],  # Path to your CSV file for the test dataset
        augment=False,
        supervised=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory']
    )

    # Initialize models
    G = UNetGenerator(
        input_channels=config['model']['generator']['input_channels'],
        output_channels=config['model']['generator']['output_channels'],
        ngf=config['model']['generator']['ngf']
    ).to(device)
    S = UNet(
        spatial_dims=config['model']['segmenter']['spatial_dims'],
        in_channels=config['model']['segmenter']['in_channels'],
        out_channels=config['model']['segmenter']['out_channels'],  # Adjust based on your number of classes
        channels=tuple(config['model']['segmenter']['channels']),
        strides=tuple(config['model']['segmenter']['strides'])
    ).to(device)

    # Load the saved model weights
    G.load_state_dict(torch.load(config['model']['generator']['weights'], map_location=device))
    S.load_state_dict(torch.load(config['model']['segmenter']['weights'], map_location=device))

    G.eval()
    S.eval()

    num_classes = config['model']['num_classes']

    ious = []
    dices = []


    with torch.no_grad():
        for batch_idx, val_batch in enumerate(val_loader):
            val_images, val_masks, image_names = val_batch
            val_images = val_images.to(device)
            val_masks = val_masks.to(device)

            # Generate augmented images
            val_I_prime = G(val_images)

            # Segmentation prediction
            val_M_prime = S(val_I_prime)

            # Get predicted and true masks
            pred_mask = torch.argmax(val_M_prime, dim=1).cpu().numpy()  # Shape: [batch_size, height, width]
            true_mask = val_masks.cpu().numpy()[:, 0, :, :]  # Remove channel dimension: [batch_size, height, width]

            # Compute IoU and Dice for each class
            iou_scores = compute_iou(pred_mask, true_mask, num_classes=num_classes)
            dice_scores = compute_dice(pred_mask, true_mask, num_classes=num_classes)

            ious.append(iou_scores)
            dices.append(dice_scores)

    # Convert to numpy arrays
    ious_array = np.array(ious)
    dices_array = np.array(dices)

    # Calculate mean metrics
    mean_ious = np.nanmean(ious_array, axis=0)
    mean_dices = np.nanmean(dices_array, axis=0)

    print(f"Mean IOU per class: {mean_ious}")
    print(f"Mean Dice per class: {mean_dices}")

    # Compute and print confidence intervals for Dice scores
    for cls in range(num_classes):
        ci = compute_confidence_interval(dices_array[:, cls])
        print(f"Class {cls} Dice - 95% CI: {ci.low:.4f} - {ci.high:.4f}")

    # Average Dice (excluding background if class 0)
    if config['metrics']['exclude_background']:
        avg_dice = np.nanmean(dices_array[:, 1:], axis=1)  # Exclude background class
    else:
        avg_dice = np.nanmean(dices_array, axis=1)  # Include all classes
    avg_dice_mean = np.nanmean(avg_dice)
    avg_dice_ci = compute_confidence_interval(avg_dice)

    print(f"Average Dice: {avg_dice_mean:.4f}")
    print(f"95% CI for Average Dice: {avg_dice_ci.low:.4f} - {avg_dice_ci.high:.4f}")

if __name__ == "__main__":
    main()





