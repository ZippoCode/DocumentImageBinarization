import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


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


class HandwrittenTextImageDataset(Dataset):

    def __init__(self, root_dg_dir: str, root_gt_dir: str, split_size=256, transform=None):
        super(HandwrittenTextImageDataset, self).__init__()

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


class ValidationDataLoader(Dataset):

    def __init__(self, root_dg_dir: str, root_gt_dir: str, split_size=256, transform=None):
        super(ValidationDataLoader, self).__init__()

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
