import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import SimpleITK as sitk
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_image_file(filepath):
    with Image.open(filepath) as img:
        return img.convert('L')  # Convert to grayscale

def load_sitk_file(filepath):
    sitk_image = sitk.ReadImage(filepath)
    return sitk.GetArrayFromImage(sitk_image)

def get_transforms(augment=False):
    if augment:
        transform_image = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
        transform_mask = A.Compose([
            A.Resize(256, 256, interpolation=Image.NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            ToTensorV2(),
        ])
    else:
        transform_image = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
        transform_mask = A.Compose([
            A.Resize(256, 256, interpolation=Image.NEAREST),
            ToTensorV2(),
        ])
    return transform_image, transform_mask

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.nii'))]
        self.transform_image, self.transform_mask = get_transforms(augment=augment)
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = load_image_file(image_path)

        if self.mask_dir:
            # Adjust the mask file extension as per your data
            mask_path = os.path.join(self.mask_dir, os.path.splitext(image_name)[0] + '.nii')
            mask = load_sitk_file(mask_path)
            mask = np.squeeze(mask).astype(np.uint8)
            mask = Image.fromarray(mask)
        else:
            mask = None

        if self.transform_image:
            if mask is not None:
                # Apply the same transformation to image and mask
                transformed = self.transform_image(image=np.array(image), mask=np.array(mask))
                image = transformed['image']
                mask = transformed['mask']
            else:
                transformed = self.transform_image(image=np.array(image))
                image = transformed['image']

        if mask is not None:
            # Convert mask to tensor and ensure it's of type Long
            mask = mask.long()
            mask = mask.unsqueeze(0)  # [1, H, W]

        return image, mask, image_name