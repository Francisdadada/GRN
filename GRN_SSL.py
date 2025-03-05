import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from monai.networks.nets import UNet
from monai.losses import DiceLoss

import numpy as np

# Import the updated utility functions
from utils.load_dataset_SSL import SegmentationDataset, get_transforms
from models.models import UNetEncoder, UNetDecoder, UNetGenerator, PatchGANDiscriminator

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

def main():
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train model with config file')
    parser.add_argument('--config', type=str, default='config_GRN_SSL.yaml', help='Path to the config file')
    args = parser.parse_args()

    # Load hyperparameters from config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    num_epochs = config['training']['num_epochs']
    lambda_adv = config['training']['lambda_adv']
    lambda_seg = config['training']['lambda_seg']
    lambda_pixel = config['training']['lambda_pixel']
    lambda_consistency = config['training']['lambda_consistency']
    alpha = config['training']['alpha']

    real_label = config['training']['real_label']
    fake_label = config['training']['fake_label']

    batch_size = config['training']['batch_size']

    best_val_loss = float(config['training']['best_val_loss'])

    # Initialize labeled training dataset and dataloader
    train_labeled_dataset = SegmentationDataset(
        csv_file=config['dataset']['train_labeled_csv'],
        augment=True,
        supervised=True
    )

    train_labeled_loader = DataLoader(
        train_labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory']
    )

    # Initialize unlabeled training dataset and dataloader with custom collate function
    train_unlabeled_dataset = SegmentationDataset(
        csv_file=config['dataset']['train_unlabeled_csv'],
        augment=True,
        supervised=False
    )
    train_unlabeled_loader = DataLoader(
        train_unlabeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory'],
        collate_fn=collate_fn_unlabeled  # Assign the custom collate function here
    )

    # Initialize validation dataset and dataloader
    val_dataset = SegmentationDataset(
        csv_file=config['dataset']['val_csv'],
        augment=False,
        supervised=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
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

    D = PatchGANDiscriminator(
        input_channels=config['model']['discriminator']['input_channels'],
        ndf=config['model']['discriminator']['ndf'],
        n_layers=config['model']['discriminator']['n_layers']
    ).to(device)

    S = UNet(
        spatial_dims=config['model']['segmenter']['spatial_dims'],
        in_channels=config['model']['segmenter']['in_channels'],
        out_channels=config['model']['segmenter']['out_channels'],  # Adjust based on your number of classes
        channels=tuple(config['model']['segmenter']['channels']),
        strides=tuple(config['model']['segmenter']['strides']),
    ).to(device)

    # Define loss functions
    criterion_GAN = nn.MSELoss()  # For adversarial loss

    # Updated DiceLoss for multi-class segmentation
    criterion_seg = DiceLoss(
        to_onehot_y=True,  # Automatically one-hot encode the target
        softmax=True,      # Apply softmax to the input
        include_background=True  # Include background class if applicable
    )

    criterion_pixelwise = nn.L1Loss()  # For L1 loss between real and generated images (optional)
    criterion_consistency = nn.MSELoss()  # For consistency loss

    # Optimizers
    optimizer_G = optim.Adam(
        G.parameters(),
        lr=config['optimizer']['generator']['lr'],
        betas=tuple(config['optimizer']['generator']['betas'])
    )
    optimizer_S = optim.Adam(
        S.parameters(),
        lr=config['optimizer']['segmenter']['lr'],
        betas=tuple(config['optimizer']['segmenter']['betas'])
    )
    optimizer_D = optim.Adam(
        D.parameters(),
        lr=config['optimizer']['discriminator']['lr'],
        betas=tuple(config['optimizer']['discriminator']['betas'])
    )

    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.StepLR(
        optimizer_G,
        step_size=config['scheduler']['step_size'],
        gamma=config['scheduler']['gamma']
    )
    scheduler_S = optim.lr_scheduler.StepLR(
        optimizer_S,
        step_size=config['scheduler']['step_size'],
        gamma=config['scheduler']['gamma']
    )
    scheduler_D = optim.lr_scheduler.StepLR(
        optimizer_D,
        step_size=config['scheduler']['step_size'],
        gamma=config['scheduler']['gamma']
    )

    for epoch in range(num_epochs):
        G.train()
        S.train()
        D.train()
        epoch_G_loss = 0
        epoch_S_loss = 0
        epoch_D_loss = 0
        epoch_consistency_loss = 0

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

            # Generate augmented images from unlabeled data
            images_unlabeled_aug = G(images_unlabeled)

            # Discriminator outputs
            output_D_real = D(images_unlabeled)
            output_D_fake = D(images_unlabeled_aug.detach())

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
            optimizer_G.zero_grad()

            # Pixel-wise loss (optional)
            loss_G_pixel = criterion_pixelwise(images_unlabeled_aug, images_unlabeled)

            images_labeled_aug = G(images_labeled)
            # Segmentation Loss Influence
            M_prime = S(images_labeled_aug)

            # Compute segmentation loss
            loss_seg = criterion_seg(M_prime, masks_labeled)

            # Total Generator Loss
            loss_G = lambda_seg * loss_seg + lambda_pixel * loss_G_pixel
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            # (3) Update Segmentation Model (S)
            # ---------------------
            optimizer_S.zero_grad()

            # Segmentation on original images
            M_real = S(images_labeled)
            loss_S_real = criterion_seg(M_real, masks_labeled)

            # Segmentation on augmented images
            M_fake = S(images_labeled_aug.detach())
            loss_S_fake = criterion_seg(M_fake, masks_labeled)  # Assuming masks are available for augmented images

            # Total Segmentation Loss
            loss_S = (loss_S_fake + loss_S_real) / 2
            loss_S.backward()
            optimizer_S.step()

            # Record losses
            epoch_G_loss += loss_G.item()
            epoch_S_loss += loss_S.item()
            epoch_D_loss += loss_D.item()

            # === Training on Unlabeled Data with ICT ===
            # ---------------------
            # (4) Update Generator Model with ICT
            # ---------------------
            optimizer_G.zero_grad()

            aug_images = G(images_unlabeled)
            # Adversarial loss
            output_D_fake_for_G = D(aug_images)
            real_labels_for_G = torch.full_like(output_D_fake_for_G, real_label, device=device)
            loss_G_adv = criterion_GAN(output_D_fake_for_G, real_labels_for_G)

            # Sample lambda from Beta distribution and ensure it's between 0 and 1
            lam = np.random.beta(alpha, alpha)
            lam = max(lam, 1 - lam)

            # Shuffle the unlabeled batch
            batch_size = images_unlabeled.size(0)
            index = torch.randperm(batch_size).to(device)
            images_unlabeled_shuffled = images_unlabeled[index]

            # Create mixed images
            mixed_images = lam * images_unlabeled + (1 - lam) * images_unlabeled_shuffled

            # Forward pass on mixed images
            outputs_mixed = S(G(mixed_images))

            # Forward pass on original and shuffled images
            outputs_unlabeled = S(G(images_unlabeled))
            outputs_unlabeled_shuffled = S(G(images_unlabeled_shuffled))

            # Create mixed targets
            # Since outputs are logits, apply softmax to get probabilities
            outputs_unlabeled_probs = nn.functional.softmax(outputs_unlabeled, dim=1)
            outputs_unlabeled_shuffled_probs = nn.functional.softmax(outputs_unlabeled_shuffled, dim=1)
            mixed_targets = lam * outputs_unlabeled_probs + (1 - lam) * outputs_unlabeled_shuffled_probs

            # Compute consistency loss
            consistency_loss = criterion_consistency(
                nn.functional.softmax(outputs_mixed, dim=1),
                mixed_targets
            )

            loss_G_pixel = criterion_pixelwise(images_unlabeled, aug_images)

            # Total ICT Consistency Loss
            loss_ICT = lambda_consistency * consistency_loss + lambda_pixel * loss_G_pixel + lambda_adv * loss_G_adv
            loss_ICT.backward()
            optimizer_G.step()

            # ---------------------
            # (5) Update Segmentation Model with ICT
            # ---------------------
            optimizer_S.zero_grad()

            # Forward pass on mixed images
            outputs_mixed = S(G(mixed_images).detach())

            # Forward pass on original and shuffled images
            outputs_unlabeled = S(G(images_unlabeled).detach())
            outputs_unlabeled_shuffled = S(G(images_unlabeled_shuffled).detach())

            # Create mixed targets
            # Since outputs are logits, apply softmax to get probabilities
            outputs_unlabeled_probs = nn.functional.softmax(outputs_unlabeled, dim=1)
            outputs_unlabeled_shuffled_probs = nn.functional.softmax(outputs_unlabeled_shuffled, dim=1)
            mixed_targets = lam * outputs_unlabeled_probs + (1 - lam) * outputs_unlabeled_shuffled_probs

            # Compute consistency loss
            consistency_loss_fake = criterion_consistency(
                nn.functional.softmax(outputs_mixed, dim=1),
                mixed_targets
            )

            # Forward pass on mixed images
            outputs_mixed_real = S(mixed_images)

            # Forward pass on original and shuffled images
            outputs_unlabeled_real = S(images_unlabeled)
            outputs_unlabeled_shuffled_real = S(images_unlabeled_shuffled)

            # Create mixed targets
            # Since outputs are logits, apply softmax to get probabilities
            outputs_unlabeled_probs_real = nn.functional.softmax(outputs_unlabeled_real, dim=1)
            outputs_unlabeled_shuffled_probs_real = nn.functional.softmax(outputs_unlabeled_shuffled_real, dim=1)
            mixed_targets_real = lam * outputs_unlabeled_probs_real + (1 - lam) * outputs_unlabeled_shuffled_probs_real

            # Compute consistency loss
            consistency_loss_true = criterion_consistency(
                nn.functional.softmax(outputs_mixed_real, dim=1),
                mixed_targets_real
            )

            # Total Segmentation Consistency Loss
            loss_S_ICT = 0.5 * (consistency_loss_fake + consistency_loss_true)
            loss_S_ICT.backward()
            optimizer_S.step()

            # Record consistency loss
            epoch_consistency_loss += consistency_loss.item()

        # Update learning rates
        scheduler_G.step()
        scheduler_S.step()
        scheduler_D.step()

        # Calculate average losses
        avg_G_loss = epoch_G_loss / len(train_labeled_loader)
        avg_S_loss = epoch_S_loss / len(train_labeled_loader)
        avg_D_loss = epoch_D_loss / len(train_labeled_loader)
        avg_consistency_loss = epoch_consistency_loss / len(train_labeled_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"G_Loss: {avg_G_loss:.4f} "
              f"S_Loss: {avg_S_loss:.4f} "
              f"D_Loss: {avg_D_loss:.4f} "
              f"Consistency_Loss: {avg_consistency_loss:.4f}")

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
                model_save_path = config['output']['model_save_path']
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                torch.save(G.state_dict(), os.path.join(model_save_path, 'best_G.pth'))
                torch.save(S.state_dict(), os.path.join(model_save_path, 'best_S.pth'))
                torch.save(D.state_dict(), os.path.join(model_save_path, 'best_D.pth'))
                print(f"Saved best models at epoch {epoch+1} with val loss {best_val_loss:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()
