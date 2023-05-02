import argparse
import os

import numpy as np
import yaml
from tqdm import tqdm

from utils.htr_logging import get_logger
from utils.ioutils import create_folder, read_image, save_image

logger = get_logger('create_training_patches')


class PatchImage:

    def __init__(self, options: dict, destination_root: str, year_validation: str):
        self.train_folder = destination_root + "/train/"
        self.train_gt_folder = destination_root + "/train_gt/"
        self._create_folders()

        self.source_original = options['source_original']
        self.source_ground_truth = options['source_ground_truth']
        self.patch_size = options['patch_size']
        self.overlap_size = options['overlap_size']
        self.validation_year = year_validation
        self.number_image = 0

        logger.info(f"Using Patch size: {self.patch_size} - Overlapping: {self.overlap_size}")
        logger.info(f"Validation Year: {year_validation}")

    def _create_folders(self):
        create_folder(self.train_folder, clean=True)
        create_folder(self.train_gt_folder, clean=True)

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

                save_image(dg, directory=self.train_folder, filename=str(self.number_image))
                save_image(gt, directory=self.train_gt_folder, filename=str(self.number_image))

                self.number_image += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-year', '--validation_year',
                        metavar='<path>',
                        type=str,
                        help=f"Year considered as testing dataset. Default: 2018",
                        default="2018")
    parser.add_argument('-dst', '--path_destination',
                        metavar='<path>',
                        type=str,
                        help=f"Destination folder path with contains the patches. Default: \"patches\"",
                        default="patches")
    parser.add_argument('-cfg', '--configuration',
                        metavar='<filename>',
                        type=str,
                        help=f"Configuration YAML file",
                        default="configs/create_patches.yaml")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    path_configuration = os.path.join(root_dir, args.configuration)
    destination_path = f"{args.path_destination}/{args.validation_year}"

    with open(path_configuration) as file:
        config_options = yaml.load(file, Loader=yaml.Loader)
        file.close()

    patcher = PatchImage(options=config_options, destination_root=destination_path,
                         year_validation=args.validation_year)
    patcher.create_patches()
