import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.htr_logging import get_logger

logger = get_logger('create_training_patches')


def check_or_create_folder(name: str):
    if not os.path.exists(name):
        os.makedirs(name)
    else:
        os.system('rm -rf ' + name + '*')
    logger.info(f"Destination folder: \"{name}\"")


def read_image(source_path: str, mode="RGB"):
    with Image.open(source_path) as img:
        image = img.convert(mode=mode)
        image = np.asarray(image, dtype=np.uint8)
    return image


def save_image(image: np.ndarray, file_name: str, folder: str, img_format="PNG"):
    fp = f"{folder}/{file_name}"
    image = image.astype(dtype=np.uint8)
    img = Image.fromarray(image, mode="RGB")
    img.save(fp=fp, format=img_format)


class PatchImage:

    def __init__(self, config_options: dict, destination_root: str, year_validation: str):
        self.train_folder = destination_root + "/train/"
        self.train_gt_folder = destination_root + "/train_gt/"
        self._create_folders()

        self.source_original = config_options['source_original']
        self.source_ground_truth = config_options['source_ground_truth']
        self.patch_size = config_options['patch_size']
        self.overlap_size = config_options['overlap_size']
        self.validation_year = year_validation
        self.number_image = 0

        logger.info(f"Using Patch size: {self.patch_size} - Overlapping: {self.overlap_size}")
        logger.info(f"Validation Year: {year_validation}")

    def _create_folders(self):
        check_or_create_folder(self.train_folder)
        check_or_create_folder(self.train_gt_folder)

    def create_patches(self):
        logger.info("Start process ...")
        all_datasets = os.listdir(self.source_original)
        all_datasets.sort()
        pbar = tqdm(all_datasets)

        try:
            for d_set in pbar:
                for im in os.listdir(self.source_original + d_set):
                    pbar.set_description(f"Processing {im} of {d_set}")
                    or_img = read_image(self.source_original + d_set + '/' + im)
                    gt_img = read_image(self.source_ground_truth + d_set + '/' + im)

                    if d_set not in [self.validation_year]:
                        self._split_train_images(or_img, gt_img)
                    else:
                        pbar.set_description(f"Excluded {im} of {d_set}")
            logger.info(f"Stored {self.number_image} training patches")
        except KeyboardInterrupt:
            logger.error("Keyboard Interrupt: stop running!")

    def _split_train_images(self, or_img: np.ndarray, gt_img: np.ndarray):
        height = or_img.shape[0]
        width = or_img.shape[1]
        for i in range(0, height, self.overlap_size):
            for j in range(0, width, self.overlap_size):
                dg = np.ones((self.patch_size, self.patch_size, 3)) * 255
                gt = np.ones((self.patch_size, self.patch_size, 3)) * 255

                if i + self.patch_size <= height and j + self.patch_size <= width:
                    dg = or_img[i:i + self.patch_size, j:j + self.patch_size, :]
                    gt = gt_img[i:i + self.patch_size, j:j + self.patch_size, :]

                elif i + self.patch_size > height and j + self.patch_size <= width:
                    dg[0:height - i, :, :] = or_img[i:height, j:j + self.patch_size, :]
                    gt[0:height - i, :, :] = gt_img[i:height, j:j + self.patch_size, :]

                elif i + self.patch_size <= height and j + self.patch_size > width:
                    dg[:, 0:width - j, :] = or_img[i:i + self.patch_size, j:width, :]
                    gt[:, 0:width - j, :] = gt_img[i:i + self.patch_size, j:width, :]

                else:
                    dg[0:height - i, 0:width - j, :] = or_img[i:height, j:width, :]
                    gt[0:height - i, 0:width - j, :] = gt_img[i:height, j:width, :]

                save_image(dg, file_name=str(self.number_image), folder=self.train_folder)
                save_image(gt, file_name=str(self.number_image), folder=self.train_gt_folder)

                self.number_image += 1
