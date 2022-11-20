# https://arxiv.org/pdf/2201.10252.pdf
# https://github.com/dali92002/DocEnTR

import os

import torchvision.transforms.functional as functional
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomRandomCrop(object):

    def __init__(self, size: int):
        self.size = size

    def __call__(self, sample, gt_sample):
        assert sample.size[0] == gt_sample.size[0]
        assert sample.size[1] == gt_sample.size[1]

        i, j, h, w = transforms.RandomCrop.get_params(sample, output_size=(self.size, self.size))
        deg_img = functional.crop(sample, i, j, h, w)
        gt_img = functional.crop(gt_sample, i, j, h, w)

        return deg_img, gt_img


class HandwrittenTextImageDataset(Dataset):
    """
        Handwritten Text Image Dataset.
    """

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

        if self.transform is None:
            crc = CustomRandomCrop(size=self.split_size)
            sample, gt_sample = crc(sample, gt_sample)

            sample = functional.to_tensor(sample)
            gt_sample = functional.to_tensor(gt_sample)

            sample = functional.normalize(sample, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        else:
            sample = self.transform(sample)
            gt_sample = self.transform(gt_sample)

        return sample, gt_sample
