import os

from PIL import Image
from torch.utils.data import Dataset

from data.utils import get_path


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
