import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.transforms import Compose, LoadImage, Resize, ScaleIntensity, ToTensor, EnsureType
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import numpy as np
import SimpleITK as sitk
from PIL import Image

from utils.utils import SegmentationDataset, get_transforms, load_image_file, load_sitk_file
from torchvision.transforms import Normalize

from copy import deepcopy

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

# Define the Residual Block
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

def main():
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Get data transformations
    transform_image, transform_mask = get_transforms()

    # Initialize labeled training dataset and dataloader
    train_labeled_dataset = SegmentationDataset(
        image_dir='dataset/0.05/labeled_img',
        mask_dir='dataset/0.05/labeled_msk',
        augment=True  # Set to True if you want to apply data augmentation
    )

    train_labeled_loader = DataLoader(
        train_labeled_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    best_val_loss =  float('inf')

    # Initialize unlabeled training dataset and dataloader with custom collate function
    train_unlabeled_dataset = SegmentationDataset(
        image_dir='dataset/0.05/unlabeled_img',
        mask_dir=None,  # No masks needed for unlabeled data
        augment=True  # Set to True if you want to apply data augmentation
    )
    train_unlabeled_loader = DataLoader(
        train_unlabeled_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_unlabeled  # Assign the custom collate function here
    )

    # Initialize validation dataset and dataloader
    val_dataset = SegmentationDataset(
        image_dir='dataset/val_img',
        mask_dir='dataset/val_msk',
        augment=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize models
    G = UNetGenerator(input_channels=1, output_channels=1).to(device)
    D = PatchGANDiscriminator(input_channels=1).to(device)
    S = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=7,  # Adjust based on your number of classes
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
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
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_S = optim.Adam(S.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.1)
    scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, step_size=10, gamma=0.1)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.1)

    # Labels for adversarial loss
    real_label = 1.0
    fake_label = 0.0

    # Hyperparameters
    num_epochs = 50
    lambda_adv = 1.0
    lambda_seg = 100.0
    lambda_pixel = 100.0  # Weight for L1 loss (optional)
    lambda_consistency = 1  # Weight for consistency loss
    alpha = 0.75  # Parameter for Beta distribution in ICT

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

            # Generate augmented images from labeled data
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
            for param in D.parameters():
                param.requires_grad = False
            # Adversarial loss
            output_D_fake_for_G = D(images_unlabeled_aug)
            real_labels_for_G = torch.full_like(output_D_fake_for_G, real_label, device=device)
            loss_G_adv = criterion_GAN(output_D_fake_for_G, real_labels_for_G)

            # Pixel-wise loss (optional)
            loss_G_pixel = criterion_pixelwise(images_unlabeled_aug, images_unlabeled)

            images_labeled_aug = G(images_labeled)
            # Segmentation Loss Influence
            M_prime = S(images_labeled_aug)

            # Compute segmentation loss
            loss_seg = criterion_seg(M_prime, masks_labeled)

            # Total Generator Loss
            loss_G = lambda_adv * loss_G_adv + lambda_seg * loss_seg + lambda_pixel * loss_G_pixel
            loss_G.backward()
            optimizer_G.step()

            for param in D.parameters():
                param.requires_grad = True

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
            loss_S = (loss_S_fake+loss_S_real)/2
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
            aug_images = G(images_unlabeled)

            loss_G_pixel = criterion_pixelwise(images_unlabeled, aug_images)


            # Total ICT Consistency Loss
            loss_ICT = 1 * consistency_loss +100 *loss_G_pixel
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
            loss_S_ICT = 0.5 * (consistency_loss_fake+ consistency_loss_true)
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
                torch.save(G.state_dict(), 'best_G.pth')
                torch.save(S.state_dict(), 'best_S.pth')
                torch.save(D.state_dict(), 'best_D.pth')
                print(f"Saved best models at epoch {epoch+1} with val loss {best_val_loss:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()

