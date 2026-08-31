from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from grn.data.dataset import SegmentationDataset, collate_unlabeled, resolve_existing_dir
from grn.models.factory import build_grn_models
from grn.training.losses import build_segmentation_loss
from grn.utils.seed import seed_everything


def train_from_config(config: dict) -> None:
    seed_everything(int(config.get("seed", 42)))
    device = torch.device(config.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    data_cfg = config["data"]
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    image_size = int(data_cfg.get("image_size", 256))
    batch_size = int(training_cfg.get("batch_size", 4))
    num_workers = int(training_cfg.get("num_workers", 4))
    num_classes = int(model_cfg.get("num_classes", 7))

    labeled_images = resolve_existing_dir(
        data_cfg.get("labeled_images"),
        data_cfg.get("labelled_images"),
    )
    labeled_masks = resolve_existing_dir(
        data_cfg.get("labeled_masks"),
        data_cfg.get("labelled_masks"),
    )
    unlabeled_images = resolve_existing_dir(
        data_cfg.get("unlabeled_images"),
        data_cfg.get("unlabelled_images"),
    )

    train_labeled_dataset = SegmentationDataset(labeled_images, labeled_masks, augment=True, image_size=image_size)
    train_unlabeled_dataset = SegmentationDataset(unlabeled_images, None, augment=True, image_size=image_size)
    val_dataset = SegmentationDataset(
        data_cfg["val_images"],
        data_cfg["val_masks"],
        augment=False,
        image_size=image_size,
    )

    train_labeled_loader = DataLoader(
        train_labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_unlabeled_loader = DataLoader(
        train_unlabeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_unlabeled,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    generator, discriminator, segmenter = build_grn_models(num_classes=num_classes)
    generator, discriminator, segmenter = generator.to(device), discriminator.to(device), segmenter.to(device)

    criterion_gan = nn.MSELoss()
    criterion_seg = build_segmentation_loss(include_background=True)
    criterion_pixelwise = nn.L1Loss()
    criterion_consistency = nn.MSELoss()

    lr = float(training_cfg.get("lr", 0.0002))
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_s = optim.Adam(segmenter.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.1)
    scheduler_s = optim.lr_scheduler.StepLR(optimizer_s, step_size=10, gamma=0.1)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=10, gamma=0.1)

    output_dir = Path(config.get("artifacts", {}).get("checkpoint_dir", "artifacts/checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    lambda_adv = float(training_cfg.get("lambda_adv", 1.0))
    lambda_seg = float(training_cfg.get("lambda_seg", 100.0))
    lambda_pixel = float(training_cfg.get("lambda_pixel", 100.0))
    alpha = float(training_cfg.get("ict_alpha", 0.75))

    for epoch in range(int(training_cfg.get("epochs", 50))):
        generator.train()
        segmenter.train()
        discriminator.train()
        epoch_g_loss = epoch_s_loss = epoch_d_loss = epoch_consistency_loss = 0.0
        unlabeled_iter = iter(train_unlabeled_loader)

        for images_labeled, masks_labeled, _names in train_labeled_loader:
            try:
                images_unlabeled, _image_names = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(train_unlabeled_loader)
                images_unlabeled, _image_names = next(unlabeled_iter)

            images_labeled = images_labeled.to(device)
            masks_labeled = masks_labeled.to(device)
            images_unlabeled = images_unlabeled.to(device)

            optimizer_d.zero_grad()
            images_unlabeled_aug = generator(images_unlabeled)
            output_d_real = discriminator(images_unlabeled)
            output_d_fake = discriminator(images_unlabeled_aug.detach())
            loss_d = 0.5 * (
                criterion_gan(output_d_real, torch.ones_like(output_d_real))
                + criterion_gan(output_d_fake, torch.zeros_like(output_d_fake))
            )
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            discriminator.requires_grad_(False)
            output_d_fake_for_g = discriminator(images_unlabeled_aug)
            loss_g_adv = criterion_gan(output_d_fake_for_g, torch.ones_like(output_d_fake_for_g))
            loss_g_pixel = criterion_pixelwise(images_unlabeled_aug, images_unlabeled)
            images_labeled_aug = generator(images_labeled)
            loss_seg = criterion_seg(segmenter(images_labeled_aug), masks_labeled)
            loss_g = lambda_adv * loss_g_adv + lambda_seg * loss_seg + lambda_pixel * loss_g_pixel
            loss_g.backward()
            optimizer_g.step()
            discriminator.requires_grad_(True)

            optimizer_s.zero_grad()
            loss_s_real = criterion_seg(segmenter(images_labeled), masks_labeled)
            loss_s_fake = criterion_seg(segmenter(images_labeled_aug.detach()), masks_labeled)
            loss_s = 0.5 * (loss_s_fake + loss_s_real)
            loss_s.backward()
            optimizer_s.step()

            lam_sample = np.random.beta(alpha, alpha)
            lam = max(lam_sample, 1 - lam_sample)
            batch_size_i = images_unlabeled.size(0)
            index = torch.randperm(batch_size_i, device=device)
            shuffled = images_unlabeled[index]
            mixed_images = lam * images_unlabeled + (1 - lam) * shuffled

            optimizer_g.zero_grad()
            outputs_mixed = segmenter(generator(mixed_images))
            outputs_unlabeled = segmenter(generator(images_unlabeled))
            outputs_shuffled = segmenter(generator(shuffled))
            mixed_targets = lam * nn.functional.softmax(outputs_unlabeled, dim=1) + (1 - lam) * nn.functional.softmax(
                outputs_shuffled, dim=1
            )
            consistency_loss = criterion_consistency(nn.functional.softmax(outputs_mixed, dim=1), mixed_targets)
            loss_ict = consistency_loss + lambda_pixel * criterion_pixelwise(images_unlabeled, generator(images_unlabeled))
            loss_ict.backward()
            optimizer_g.step()

            optimizer_s.zero_grad()
            outputs_mixed_fake = segmenter(generator(mixed_images).detach())
            outputs_unlabeled_fake = segmenter(generator(images_unlabeled).detach())
            outputs_shuffled_fake = segmenter(generator(shuffled).detach())
            mixed_targets_fake = lam * nn.functional.softmax(outputs_unlabeled_fake, dim=1) + (1 - lam) * nn.functional.softmax(
                outputs_shuffled_fake, dim=1
            )
            consistency_fake = criterion_consistency(nn.functional.softmax(outputs_mixed_fake, dim=1), mixed_targets_fake)
            outputs_mixed_real = segmenter(mixed_images)
            outputs_real = segmenter(images_unlabeled)
            outputs_shuffled_real = segmenter(shuffled)
            mixed_targets_real = lam * nn.functional.softmax(outputs_real, dim=1) + (1 - lam) * nn.functional.softmax(
                outputs_shuffled_real, dim=1
            )
            consistency_real = criterion_consistency(nn.functional.softmax(outputs_mixed_real, dim=1), mixed_targets_real)
            loss_s_ict = 0.5 * (consistency_fake + consistency_real)
            loss_s_ict.backward()
            optimizer_s.step()

            epoch_g_loss += float(loss_g.item())
            epoch_s_loss += float(loss_s.item())
            epoch_d_loss += float(loss_d.item())
            epoch_consistency_loss += float(consistency_loss.item())

        scheduler_g.step()
        scheduler_s.step()
        scheduler_d.step()

        avg_val_loss = _validate(generator, segmenter, val_loader, criterion_seg, device)
        print(
            f"Epoch [{epoch + 1}/{training_cfg.get('epochs', 50)}] "
            f"G_Loss: {epoch_g_loss / len(train_labeled_loader):.4f} "
            f"S_Loss: {epoch_s_loss / len(train_labeled_loader):.4f} "
            f"D_Loss: {epoch_d_loss / len(train_labeled_loader):.4f} "
            f"Consistency: {epoch_consistency_loss / len(train_labeled_loader):.4f} "
            f"Val_Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(generator.state_dict(), output_dir / "best_G.pth")
            torch.save(segmenter.state_dict(), output_dir / "best_S.pth")
            torch.save(discriminator.state_dict(), output_dir / "best_D.pth")
            print(f"Saved best models to {output_dir} with val loss {best_val_loss:.4f}")


def _validate(generator, segmenter, val_loader, criterion_seg, device: torch.device) -> float:
    generator.eval()
    segmenter.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_images, val_masks, _ in val_loader:
            val_images = val_images.to(device)
            val_masks = val_masks.to(device)
            val_loss += criterion_seg(segmenter(generator(val_images)), val_masks).item()
    return val_loss / max(1, len(val_loader))
