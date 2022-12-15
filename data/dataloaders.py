import math
import os

import PIL
import numpy as np
import torch
import torchvision.transforms.functional as functional
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import trainer.CustomTransforms as CustomTransform
from utils.htr_logging import get_logger

logger = get_logger(__file__)


def get_patches(image_source: PIL.Image, patch_size: int, stride: int):
    image = np.asarray(image_source)
    image_patches = []

    h = ((image.shape[0] // patch_size) + 1) * patch_size
    w = ((image.shape[1] // patch_size) + 1) * patch_size

    padding_image = np.ones((h, w, 3)) if len(image.shape) == 3 else np.ones((h, w))
    padding_image = padding_image * 255.0
    padding_image[:image.shape[0], :image.shape[1]] = image

    for j in range(0, w - patch_size + 1, stride):
        for i in range(0, h - patch_size + 1, stride):
            image_patches.append(padding_image[i:i + patch_size, j:j + patch_size])

    num_rows = math.floor((padding_image.shape[0] - patch_size) / stride) + 1
    num_cols = math.floor((padding_image.shape[1] - patch_size) / stride) + 1

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
    transform = transforms.Compose([transforms.ToTensor()])

    valid_data_path = config['valid_data_path']
    valid_gt_data_path = config['valid_gt_data_path']
    patch_size = config['valid_patch_size']
    stride = config['valid_stride']
    valid_dataset = ValidationDataset(root_dg_dir=valid_data_path, root_gt_dir=valid_gt_data_path,
                                      patch_size=patch_size, stride=stride, transform=transform)
    valid_dataloader_config = config['valid_kwargs']
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, **valid_dataloader_config)

    logger.info(f"Validation paths: Original  \"{valid_data_path}\" - Ground truth: \"{valid_gt_data_path}\"")
    logger.info(f"Validation dimensions: Patch Size {patch_size} - Stride {stride}")
    logger.info(f"Validation set has {len(valid_dataset)} instances")

    return valid_data_loader


def get_path(root: str, paths: list, index: int):
    assert index < len(paths)
    return os.path.join(root, paths[index])


class TrainingDataset(Dataset):

    def __init__(self, root_dg_dir: str, root_gt_dir: str, split_size=256, transform=None):
        assert len(os.listdir(root_dg_dir)) == len(os.listdir(root_gt_dir))

        super(TrainingDataset, self).__init__()
        self.root_dg_dir = root_dg_dir
        self.root_gt_dir = root_gt_dir
        self.split_size = split_size
        self.path_images = os.listdir(self.root_dg_dir)
        self.transform = transform

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, index):
        path_image_deg = get_path(self.root_dg_dir, self.path_images, index)
        path_image_gtr = get_path(self.root_gt_dir, self.path_images, index)

        sample = Image.open(path_image_deg).convert("RGB")
        gt_sample = Image.open(path_image_gtr).convert("L")

        if self.transform:
            transform = self.transform({'image': sample, 'gt': gt_sample})
            sample = transform['image']
            gt_sample = transform['gt']

        return sample, gt_sample


class ValidationDataset(Dataset):

    def __init__(self, root_dg_dir: str, root_gt_dir: str, patch_size=256, stride=256, transform=None):
        assert len(os.listdir(root_dg_dir)) == len(os.listdir(root_gt_dir))

        super(ValidationDataset, self).__init__()
        self.root_dg_dir = root_dg_dir
        self.root_gt_dir = root_gt_dir
        self.patch_size = patch_size
        self.stride = stride
        self.path_images = os.listdir(self.root_dg_dir)
        self.transform = transform

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, index):
        path_image_deg = get_path(self.root_dg_dir, self.path_images, index)
        path_image_gtr = get_path(self.root_gt_dir, self.path_images, index)

        sample = Image.open(path_image_deg).convert("RGB")
        gt_sample = Image.open(path_image_gtr).convert("L")

        # Create patches
        padding_bottom = ((sample.height // self.patch_size) + 1) * self.patch_size - sample.height
        padding_right = ((sample.width // self.patch_size) + 1) * self.patch_size - sample.width

        tensor_padding = functional.to_tensor(sample).unsqueeze(0)
        batch, channels, _, _ = tensor_padding.shape
        tensor_padding = functional.pad(img=tensor_padding, padding=[0, 0, padding_right, padding_bottom], fill=1)
        patches = tensor_padding.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
        num_rows = patches.shape[3]
        patches = patches.reshape(batch, channels, -1, self.patch_size, self.patch_size)

        if self.transform:
            sample = self.transform(sample)
            gt_sample = self.transform(gt_sample)

        item = {
            'image_name': self.path_images[index],
            'sample': sample,
            'num_rows': num_rows,
            'samples_patches': patches,
            'gt_sample': gt_sample
        }

        return item
