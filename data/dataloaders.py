import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import trainer.CustomTransforms as CustomTransform
from utils.htr_logging import get_logger

logger = get_logger(__file__, debug=True)


def get_patches(image: np.ndarray, patch_size: int, stride: int):
    image_patches = []

    h = ((image.shape[0] // patch_size) + 1) * patch_size
    w = ((image.shape[1] // patch_size) + 1) * patch_size

    padding_image = np.ones((h, w, 3)) if len(image.shape) == 3 else np.ones((h, w))
    padding_image = padding_image * 255.0
    padding_image[:image.shape[0], :image.shape[1]] = image

    for j in range(0, w - patch_size + 1, stride):
        for i in range(0, h - patch_size + 1, stride):
            image_patches.append(padding_image[i:i + patch_size, j:j + patch_size])

    num_rows = padding_image.shape[0] // patch_size
    num_cols = padding_image.shape[1] // patch_size

    return np.array(image_patches), num_rows, num_cols


def get_transform(transform_variant: str, output_size: int):
    if transform_variant == 'default':
        transform = transforms.Compose([
            CustomTransform.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5),
            CustomTransform.RandomRotation((0, 360)),
            CustomTransform.RandomHorizontalFlip(),
            CustomTransform.RandomVerticalFlip(),
            CustomTransform.RandomCrop(output_size),
            CustomTransform.ToTensor()
        ])
    elif transform_variant == 'gaussian':
        transform = transforms.Compose([
            CustomTransform.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5),
            CustomTransform.GaussianBlur(kernel_size=(3, 5), sigma=(0.3, 1.5)),
            CustomTransform.RandomRotation((0, 360)),
            CustomTransform.RandomHorizontalFlip(),
            CustomTransform.RandomVerticalFlip(),
            CustomTransform.RandomCrop(output_size),
            CustomTransform.ToTensor()
        ])
    elif transform_variant == 'equalize_contrast':
        transform = transforms.Compose([
            CustomTransform.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5),
            CustomTransform.RandomEqualize(),
            CustomTransform.RandomAutoContrast(),
            CustomTransform.RandomRotation((0, 360)),
            CustomTransform.RandomHorizontalFlip(),
            CustomTransform.RandomVerticalFlip(),
            CustomTransform.RandomCrop(output_size),
            CustomTransform.ToTensor()
        ])
    elif transform_variant == 'adjust_sharpness':
        transform = transforms.Compose([
            CustomTransform.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5),
            CustomTransform.RandomAdjustSharpness(sharpness_factor=0),
            CustomTransform.RandomRotation((0, 360)),
            CustomTransform.RandomHorizontalFlip(),
            CustomTransform.RandomVerticalFlip(),
            CustomTransform.RandomCrop(output_size),
            CustomTransform.ToTensor()
        ])
    elif transform_variant == 'all_transforms':
        transform = transforms.Compose([
            CustomTransform.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5),
            CustomTransform.GaussianBlur(kernel_size=(3, 5), sigma=(0.3, 1.5)),
            CustomTransform.RandomAdjustSharpness(sharpness_factor=0),
            CustomTransform.RandomEqualize(),
            CustomTransform.RandomAutoContrast(),
            CustomTransform.RandomRotation((0, 360)),
            CustomTransform.RandomHorizontalFlip(),
            CustomTransform.RandomVerticalFlip(),
            CustomTransform.RandomCrop(output_size),
            CustomTransform.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            CustomTransform.ToTensor()
        ])
    return transform


def make_train_dataloader(config: dict, output_size=256):
    train_data_path = config['train_data_path']
    train_gt_data_path = config['train_gt_data_path']
    transform_variant = config['train_transform_variant']
    logger.info(f"Train path: \"{train_data_path}\" - Train ground truth path: \"{train_gt_data_path}\"")

    transform = get_transform(transform_variant=transform_variant, output_size=output_size)
    train_dataset = TrainingDataset(train_data_path, train_gt_data_path, transform=transform)

    train_dataloader_config = config['train_kwargs']
    train_data_loader = torch.utils.data.DataLoader(train_dataset, **train_dataloader_config)
    logger.info(f"Training set has {len(train_dataset)} instances")

    return train_data_loader


def make_valid_dataloader(config: dict):
    valid_data_path = config['valid_data_path']
    valid_gt_data_path = config['valid_gt_data_path']
    logger.info(f"Validation path: \"{valid_data_path}\" - Validation ground truth path: \"{valid_gt_data_path}\"")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    valid_dataset = ValidationDataset(valid_data_path, valid_gt_data_path, transform=transform)

    valid_dataloader_config = config['valid_kwargs']
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, **valid_dataloader_config)
    logger.info(f"Validation set has {len(valid_dataset)} instances")

    return valid_data_loader


class TrainingDataset(Dataset):

    def __init__(self, root_dg_dir: str, root_gt_dir: str, split_size=256, transform=None):
        super(TrainingDataset, self).__init__()

        assert len(os.listdir(root_dg_dir)) == len(
            os.listdir(root_gt_dir)), f"The folder not contains the same number of images"

        self.root_dg_dir = root_dg_dir
        self.root_gt_dir = root_gt_dir
        self.split_size = split_size
        self.path_images = os.listdir(self.root_dg_dir)
        self.transform = transform

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, index):
        path_image_deg = os.path.join(self.root_dg_dir, self.path_images[index])
        path_image_gtr = os.path.join(self.root_gt_dir, self.path_images[index])

        sample = Image.open(path_image_deg).convert("RGB")
        gt_sample = Image.open(path_image_gtr).convert("L")

        if self.transform:
            transform = self.transform({'image': sample, 'gt': gt_sample})
            sample = transform['image']
            gt_sample = transform['gt']

        return sample, gt_sample


class ValidationDataset(Dataset):

    def __init__(self, root_dg_dir: str, root_gt_dir: str, split_size=256, transform=None):
        super(ValidationDataset, self).__init__()

        assert len(os.listdir(root_dg_dir)) == len(
            os.listdir(root_gt_dir)), f"The folder not contains the same number of images"

        self.root_dg_dir = root_dg_dir
        self.root_gt_dir = root_gt_dir
        self.split_size = split_size
        self.path_images = os.listdir(self.root_dg_dir)
        self.transform = transform

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, index):
        path_image_deg = os.path.join(self.root_dg_dir, self.path_images[index])
        path_image_gtr = os.path.join(self.root_gt_dir, self.path_images[index])

        sample = Image.open(path_image_deg).convert("RGB")
        gt_sample = Image.open(path_image_gtr).convert("L")

        sample_patches, num_rows, num_cols = get_patches(image=np.asarray(sample), patch_size=self.split_size,
                                                         stride=self.split_size)
        sample_patches = torch.Tensor(sample_patches) / 255.0

        if self.transform:
            sample = self.transform(sample)
            gt_sample = self.transform(gt_sample)

        item = {
            'image_name': self.path_images[index],
            'sample': sample,
            'num_rows': num_rows,
            'num_cols': num_cols,
            'samples_patches': sample_patches,
            'gt_sample': gt_sample
        }

        return item
