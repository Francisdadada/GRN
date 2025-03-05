import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import SimpleITK as sitk
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

def load_image_file(filepath):
    with Image.open(filepath) as img:
        return img.convert('L')  # Convert to grayscale

def load_sitk_file(filepath):
    sitk_image = sitk.ReadImage(filepath)
    return sitk.GetArrayFromImage(sitk_image)

def get_transforms(augment=False):
    if augment:
        transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
    return transform

class SegmentationDataset(Dataset):
    def __init__(self, csv_file, augment=False, supervised=True):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and optionally mask paths.
            augment (bool): Whether to apply data augmentation.
            supervised (bool): Whether the dataset is supervised (has masks) or not.
        """
        self.data = pd.read_csv(csv_file)
        self.augment = augment
        self.supervised = supervised
        self.transform = get_transforms(augment=augment)

        if self.supervised:
            if 'image_path' not in self.data.columns or 'mask_path' not in self.data.columns:
                raise ValueError("CSV file must contain 'image_path' and 'mask_path' columns for supervised dataset.")
            self.image_paths = self.data['image_path'].tolist()
            self.mask_paths = self.data['mask_path'].tolist()
        else:
            if 'image_path' not in self.data.columns:
                raise ValueError("CSV file must contain 'image_path' column for unsupervised dataset.")
            self.image_paths = self.data['image_path'].tolist()
            self.mask_paths = [None] * len(self.image_paths)  # No masks for unsupervised data

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = load_image_file(image_path)

        mask = None
        if self.supervised:
            mask_path = self.mask_paths[idx]
            mask = load_sitk_file(mask_path)
            mask = np.squeeze(mask).astype(np.uint8)
            mask = Image.fromarray(mask)

        if self.transform:
            if mask is not None:
                # Apply the same transformation to image and mask
                transformed = self.transform(image=np.array(image), mask=np.array(mask))
                image = transformed['image']
                mask = transformed['mask']
            else:
                transformed = self.transform(image=np.array(image))
                image = transformed['image']

        if mask is not None:
            # Ensure mask is of type Long
            mask = mask.long()
            # Add channel dimension if missing
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)  # Shape becomes [1, H, W]
            elif len(mask.shape) == 3 and mask.shape[0] != 1:
                # If mask has shape [C, H, W] but C != 1, raise an error
                raise ValueError(f"Mask at index {idx} has invalid shape {mask.shape}")
        else:
            mask = torch.tensor([])  # Return an empty tensor if mask is None

        return image, mask, os.path.basename(image_path)
