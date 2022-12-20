import os
import random
import cv2

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from data.utils import get_path


class TrainingDataset(Dataset):

    def __init__(self, root_dg_dir: str, root_gt_dir: str, split_size=256, transform=None, merge_image=True):
        assert len(os.listdir(root_dg_dir)) == len(os.listdir(root_gt_dir))

        super(TrainingDataset, self).__init__()
        self.root_dg_dir = root_dg_dir
        self.root_gt_dir = root_gt_dir
        self.split_size = split_size
        self.transform = transform
        self.merge_image = merge_image

        self.path_images = os.listdir(self.root_dg_dir)

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, index):
        path_image_deg = get_path(self.root_dg_dir, self.path_images, index)
        path_image_gtr = get_path(self.root_gt_dir, self.path_images, index)

        sample_array = cv2.imread(path_image_deg, cv2.IMREAD_COLOR)
        gt_sample_array = cv2.imread(path_image_gtr, cv2.IMREAD_GRAYSCALE)

        # Merge two images
        if self.merge_image:
            random_index = random.randint(0, len(self.path_images) - 1)
            random_path_image_deg = get_path(self.root_dg_dir, self.path_images, random_index)
            random_path_image_gtr = get_path(self.root_gt_dir, self.path_images, random_index)
            random_sample = cv2.imread(random_path_image_deg, cv2.IMREAD_COLOR)
            random_gt_sample = cv2.imread(random_path_image_gtr, cv2.IMREAD_GRAYSCALE)

            sample_array = np.minimum(sample_array, random_sample)
            gt_sample_array = np.minimum(gt_sample_array, random_gt_sample)

        sample = Image.fromarray(sample_array)
        gt_sample = Image.fromarray(gt_sample_array)

        if self.transform:
            transform = self.transform({'image': sample, 'gt': gt_sample})
            sample = transform['image']
            gt_sample = transform['gt']

        return sample, gt_sample
