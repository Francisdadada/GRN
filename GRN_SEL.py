import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from monai.networks.nets import UNet
from monai.losses import DiceLoss

import numpy as np
import SimpleITK as sitk
from PIL import Image

from utils.load_dataset_SEL import SegmentationDataset, get_transforms, load_image_file, load_sitk_file
from torchvision.transforms import Normalize
from models.models import UNetEncoder, UNetDecoder, UNetGenerator, PatchGANDiscriminator

import yaml  # To parse YAML config
from pathlib import Path

# Define a custom collate function for unlabeled data
def collate_fn_unlabeled(batch):
    """
    Custom collate function for unlabeled data.

    Args:
        batch (list): A list of tuples returned by the Dataset's __getitem__ method.
                      Each tuple contains (image, mask, image_name), where mask is None.

    Returns:
        Tuple: Batched images and image names.
    """
    images = []
    image_names = []

    for sample in batch:
        image, mask, image_name = sample
        images.append(image)
        image_names.append(image_name)

    # Stack images into a single tensor
    images = torch.stack(images, dim=0)

    return images, image_names

def load_config(config_path):
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration
    config_path = 'config_GRN_SEL.yaml'  # Path to your config file
    config = load_config(config_path)

    # Create model save directory if it doesn't exist
    Path(config['model']['save_dir']).mkdir(parents=True, exist_ok=True)

    # Device configuration
    device = torch.device('cuda:0' if config['device']['use_cuda'] and torch.cuda.is_available() else 'cpu')

    # Paths to CSV files
    supervised_csv = config['data']['supervised_csv']
    unsupervised_csv = config['data']['unsupervised_csv']
    val_csv = config['data']['val_csv']

    # Training hyperparameters
    num_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    lambda_adv = config['training']['lambda_adv']
    lambda_seg = config['training']['lambda_seg']
    lambda_pixel = config['training']['lambda_pixel']

    # Optimizer settings
    lr = config['optimizer']['lr']
    betas = tuple(config['optimizer']['betas'])

    # Scheduler settings
    step_size = config['scheduler']['step_size']
    gamma = config['scheduler']['gamma']

    # Initialize labeled training dataset and dataloader
    train_labeled_dataset = SegmentationDataset(
        csv_file=supervised_csv,
        augment=True,
        supervised=True  # Indicates that masks are available
    )
    train_labeled_loader = DataLoader(
        train_labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Initialize unlabeled training dataset and dataloader with custom collate function
    train_unlabeled_dataset = SegmentationDataset(
        csv_file=unsupervised_csv,
        augment=True,
        supervised=False  # No masks for unlabeled data
    )
    train_unlabeled_loader = DataLoader(
        train_unlabeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_unlabeled  # Assign the custom collate function here
    )

    # Initialize validation dataset and dataloader
    val_dataset = SegmentationDataset(
        csv_file=val_csv,
        augment=False,
        supervised=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Initialize models
    G = UNetGenerator(input_channels=1, output_channels=1).to(device)
    S = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=7,  # Adjust based on your number of classes
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2)
    ).to(device)
    D = PatchGANDiscriminator(input_channels=1).to(device)

    # Define loss functions
    criterion_GAN = nn.MSELoss()  # For adversarial loss

    # Updated DiceLoss for multi-class segmentation
    criterion_seg = DiceLoss(
        to_onehot_y=True,  # Automatically one-hot encode the target
        softmax=True,      # Apply softmax to the input
        include_background=True  # Include background class if applicable
    )

    criterion_pixelwise = nn.L1Loss()  # For L1 loss between real and generated images (optional)

    # Optimizers
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
    optimizer_S = optim.Adam(S.parameters(), lr=lr, betas=betas)
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=betas)

    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=step_size, gamma=gamma)
    scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, step_size=step_size, gamma=gamma)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=step_size, gamma=gamma)

    # Labels for adversarial loss
    real_label = 1.0
    fake_label = 0.0

    # Model save paths
    save_G_path = config['model']['generator']
    save_S_path = config['model']['segmentation']
    save_D_path = config['model']['discriminator']

    # Hyperparameters
    # num_epochs, lambda_adv, lambda_seg, etc. are already loaded above

    best_val_loss = float('inf')  # Initialize best validation loss

    for epoch in range(num_epochs):
        G.train()
        S.train()
        D.train()
        epoch_G_loss = 0
        epoch_S_loss = 0
        epoch_D_loss = 0

        # Initialize iterators
        labeled_iter = iter(train_labeled_loader)
        unlabeled_iter = iter(train_unlabeled_loader)

        # Iterate through the labeled DataLoader
        for labeled_batch in train_labeled_loader:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                # Reset the unlabeled iterator if it has been exhausted
                unlabeled_iter = iter(train_unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            # Unpack labeled data
            images_labeled, masks_labeled, _ = labeled_batch
            images_labeled = images_labeled.to(device)
            masks_labeled = masks_labeled.to(device)

            # Unpack unlabeled data
            images_unlabeled, image_names = unlabeled_batch
            images_unlabeled = images_unlabeled.to(device)

            # ---------------------
            # (1) Update Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Generate augmented images from labeled data
            I_prime = G(images_labeled)

            # Discriminator outputs
            output_D_real = D(images_labeled)
            output_D_fake = D(I_prime.detach())

            # Create labels
            real_labels_tensor = torch.full_like(output_D_real, real_label, device=device)
            fake_labels_tensor = torch.full_like(output_D_fake, fake_label, device=device)

            # Compute D losses
            loss_D_real = criterion_GAN(output_D_real, real_labels_tensor)
            loss_D_fake = criterion_GAN(output_D_fake, fake_labels_tensor)

            # Total D loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # ---------------------
            # (2) Update Generator
            # ---------------------
            for param in D.parameters():
                param.requires_grad = False

            optimizer_G.zero_grad()

            # Adversarial loss
            output_D_fake_for_G = D(I_prime)
            real_labels_for_G = torch.full_like(output_D_fake_for_G, real_label, device=device)
            loss_G_adv = criterion_GAN(output_D_fake_for_G, real_labels_for_G)

            # Pixel-wise loss (optional)
            loss_G_pixel = criterion_pixelwise(I_prime, images_labeled)

            # Segmentation Loss Influence
            M_prime = S(I_prime)

            # Compute segmentation loss
            loss_seg = criterion_seg(M_prime, masks_labeled)

            # Total Generator Loss
            loss_G = lambda_adv * loss_G_adv + lambda_seg * loss_seg + lambda_pixel * loss_G_pixel

            loss_G.backward()
            optimizer_G.step()
            for param in D.parameters():
                param.requires_grad = True
            # ---------------------
            # (3) Update Segmentation Model
            # ---------------------
            optimizer_S.zero_grad()

            # Segmentation on original images
            M_real = S(images_labeled)
            loss_S_real = criterion_seg(M_real, masks_labeled)

            # Segmentation on augmented images
            M_fake = S(I_prime.detach())
            loss_S_fake = criterion_seg(M_fake, masks_labeled)  # Assuming masks are available for augmented images

            # Total Segmentation Loss
            loss_S = (loss_S_real + loss_S_fake) / 2.0
            # loss_S = (loss_S_fake) / 1
            loss_S.backward()
            optimizer_S.step()

            # Record losses
            epoch_G_loss += loss_G.item()
            epoch_S_loss += loss_S.item()
            epoch_D_loss += loss_D.item()

        # Update learning rates
        scheduler_G.step()
        scheduler_S.step()
        scheduler_D.step()

        # Calculate average losses
        avg_G_loss = epoch_G_loss / len(train_labeled_loader)
        avg_S_loss = epoch_S_loss / len(train_labeled_loader)
        avg_D_loss = epoch_D_loss / len(train_labeled_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"G_Loss: {avg_G_loss:.4f} "
              f"S_Loss: {avg_S_loss:.4f} "
              f"D_Loss: {avg_D_loss:.4f}")

        # === Validation ===
        G.eval()
        S.eval()
        with torch.no_grad():
            val_loss = 0
            for val_batch in val_loader:
                val_images, val_masks, _ = val_batch
                val_images = val_images.to(device)
                val_masks = val_masks.to(device)

                # Generate augmented images
                val_I_prime = G(val_images)

                # Segmentation prediction
                val_M_prime = S(val_I_prime)
                val_loss += criterion_seg(val_M_prime, val_masks).item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Segmentation Loss: {avg_val_loss:.4f}")

            # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(G.state_dict(), save_G_path)
                torch.save(S.state_dict(), save_S_path)
                torch.save(D.state_dict(), save_D_path)
                print(f"Saved best models at epoch {epoch+1} with val loss {best_val_loss:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()
