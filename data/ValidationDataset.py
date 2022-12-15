import os

import torchvision.transforms.functional as functional
from PIL import Image
from torch.utils.data import Dataset

from data.utils import get_path


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
