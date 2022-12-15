import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from data.TrainingDataset import TrainingDataset
from data.ValidationDataset import ValidationDataset
from data.utils import get_transform
from utils.htr_logging import get_logger

logger = get_logger(__file__)


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
