# https://arxiv.org/pdf/2201.10252.pdf
# https://github.com/dali92002/DocEnTR

import cv2
import numpy as np
import os
import torch.utils.data
import torchvision.transforms.functional as functional
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms


class HandwrittenTextImageDataset(Dataset):
    """
        Handwritten Text Image Dataset.
    """

    def __init__(self, root_dg_dir: str, root_gt_dir: str, split_size=256):
        super(HandwrittenTextImageDataset, self).__init__()

        assert len(os.listdir(root_dg_dir)) == len(
            os.listdir(root_gt_dir)), f"The folder not contains the same number of images"

        self.root_dg_dir = root_dg_dir
        self.root_gt_dir = root_gt_dir
        self.split_size = split_size
        self.path_images = os.listdir(self.root_dg_dir)

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, index):
        path_image_deg = os.path.join(self.root_dg_dir, self.path_images[index])
        path_image_gtr = os.path.join(self.root_gt_dir, self.path_images[index])

        deg_img = cv2.imread(path_image_deg)
        gt_img = cv2.imread(path_image_gtr, cv2.IMREAD_GRAYSCALE)

        deg_img = Image.fromarray(np.uint8(deg_img))
        gt_img = Image.fromarray(np.uint8(gt_img))

        i, j, h, w = transforms.RandomCrop.get_params(deg_img, output_size=(self.split_size, self.split_size))
        deg_img = functional.crop(deg_img, i, j, h, w)
        gt_img = functional.crop(gt_img, i, j, h, w)

        deg_img = (np.array(deg_img) / 255).astype('float32')
        gt_img = (np.array(gt_img) / 255).astype('float32')

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        out_deg_img = np.zeros([3, *deg_img.shape[:-1]])

        for i in range(3):
            out_deg_img[i] = (deg_img[:, :, i] - mean[i]) / std[i]

        out_gt_img = np.zeros([1, *gt_img.shape])
        out_gt_img[0, :, :] = gt_img

        return out_deg_img, out_gt_img

    @staticmethod
    def collate_fn(batch):
        train_in, train_out = [], []

        for i in range(len(batch)):
            img, gt_img = batch[i]

            train_in.append(img)
            train_out.append(gt_img)

        train_in = np.array(train_in, dtype='float32')
        train_out = np.array(train_out, dtype='float32')

        train_in = torch.from_numpy(train_in)
        train_out = torch.from_numpy(train_out)

        return train_in, train_out


class ImageDataset(Dataset):
    def __init__(self, root_dir: str, root_gt_dir: str):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.root_gt_dir = root_gt_dir
        self.path_images = os.listdir(root_dir)

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, index):
        image_name = os.path.join(self.root_dir, self.path_images[index])
        image_gt_name = os.path.join(self.root_gt_dir, self.path_images[index])

        image = io.imread(image_name)
        gt_image = io.imread(image_gt_name, as_gray=True)

        return image, gt_image


if __name__ == '__main__':
    # htDatasetTrain = HandwrittenTextImageDataset('patches/train', 'patches/train_gt')
    htDatasetValid = ImageDataset('patches/valid', 'patches/valid_gt')

    # train_loader = torch.utils.data.DataLoader(htDatasetTrain, collate_fn=htDatasetTrain.collate_fn, batch_size=32,
    # shuffle = False, num_workers = 0, pin_memory = True)
    valid_loader = torch.utils.data.DataLoader(htDatasetValid)
    """
    for index, (train_index, train_input, train_output) in enumerate(train_loader):
        print(f"Training image {index} has size {train_input.size()}")
        # print(f"Images index {train_index}")
    """

    for index, (input, output) in enumerate(valid_loader):
        print(f"Image {index} has size {output.size()}")
        # print(f"Images index {train_index}")
