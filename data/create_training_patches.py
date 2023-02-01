import os

import cv2
import numpy as np
from tqdm import tqdm

from utils.htr_logging import get_logger

logger = get_logger('create_training_patches')


def check_or_create_folder(name: str):
    if not os.path.exists(name):
        os.makedirs(name)
    else:
        os.system('rm -rf ' + name + '*')
    logger.info(f"Destination folder: \"{name}\"")


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

        self.number_image = 1

        logger.info(f"Using Patch size: {self.patch_size} - Overlapping: {self.overlap_size}")
        logger.info(f"Validation Year: {year_validation}")

    def _create_folders(self):
        check_or_create_folder(self.train_folder)
        check_or_create_folder(self.train_gt_folder)

    def create_patches(self):
        logger.info("Start process ...")
        all_datasets = os.listdir(self.source_original)
        pbar = tqdm(all_datasets)

        try:
            for d_set in pbar:
                for im in os.listdir(self.source_original + d_set):
                    pbar.set_description(f"Processing {im} of {d_set}")
                    or_img = cv2.imread(self.source_original + d_set + '/' + im)
                    gt_img = cv2.imread(self.source_ground_truth + d_set + '/' + im)
                    if d_set not in [self.validation_year]:
                        self._split_train_images(or_img, gt_img)
                    else:
                        print(im)
            logger.info(f"Stored {self.number_image} training patches")
        except KeyboardInterrupt:
            logger.error("Keyboard Interrupt: stop running!")

    def _split_train_images(self, or_img: np.ndarray, gt_img: np.ndarray):
        for i in range(0, or_img.shape[0], self.overlap_size):
            for j in range(0, or_img.shape[1], self.overlap_size):

                if i + self.patch_size <= or_img.shape[0] and j + self.patch_size <= or_img.shape[1]:
                    dg_patch = or_img[i:i + self.patch_size, j:j + self.patch_size, :]
                    gt_patch = gt_img[i:i + self.patch_size, j:j + self.patch_size, :]

                elif i + self.patch_size > or_img.shape[0] and j + self.patch_size <= or_img.shape[1]:
                    dg_patch = np.ones((self.patch_size, self.patch_size, 3)) * 255
                    gt_patch = np.ones((self.patch_size, self.patch_size, 3)) * 255

                    dg_patch[0:or_img.shape[0] - i, :, :] = or_img[i:or_img.shape[0], j:j + self.patch_size, :]
                    gt_patch[0:or_img.shape[0] - i, :, :] = gt_img[i:or_img.shape[0], j:j + self.patch_size, :]

                elif i + self.patch_size <= or_img.shape[0] and j + self.patch_size > or_img.shape[1]:
                    dg_patch = np.ones((self.patch_size, self.patch_size, 3)) * 255
                    gt_patch = np.ones((self.patch_size, self.patch_size, 3)) * 255

                    dg_patch[:, 0:or_img.shape[1] - j, :] = or_img[i:i + self.patch_size, j:or_img.shape[1], :]
                    gt_patch[:, 0:or_img.shape[1] - j, :] = gt_img[i:i + self.patch_size, j:or_img.shape[1], :]

                else:
                    dg_patch = np.ones((self.patch_size, self.patch_size, 3)) * 255
                    gt_patch = np.ones((self.patch_size, self.patch_size, 3)) * 255

                    dg_patch[0:or_img.shape[0] - i, 0:or_img.shape[1] - j, :] = or_img[i:or_img.shape[0],
                                                                                j:or_img.shape[1],
                                                                                :]
                    gt_patch[0:or_img.shape[0] - i, 0:or_img.shape[1] - j, :] = gt_img[i:or_img.shape[0],
                                                                                j:or_img.shape[1],
                                                                                :]
                    gt_patch[0:or_img.shape[0] - i, 0:or_img.shape[1] - j, :] = gt_img[i:or_img.shape[0],
                                                                                j:or_img.shape[1],
                                                                                :]

                cv2.imwrite(self.train_folder + str(self.number_image) + '.png', dg_patch)
                cv2.imwrite(self.train_gt_folder + str(self.number_image) + '.png', gt_patch)
                self.number_image += 1
