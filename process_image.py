import argparse
import os
import random

import cv2
import numpy as np
from tqdm import tqdm
import logging
import config


def check_or_create_folder(name: str):
    if not os.path.exists(name):
        os.makedirs(name)
    else:
        os.system('rm -rf ' + name + '*')


class PatchImage:

    def __init__(self, patch_size: int, overlap_size: int, patch_size_valid: int, destination_root: str):
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        self.train_folder = destination_root + "/train/"
        self.train_gt_folder = destination_root + "/train_gt/"
        self.valid_folder = destination_root + "/valid/"
        self.valid_gt_folder = destination_root + "/valid_gt/"
        self.test_folder = destination_root + "/test/"
        self.test_gt_folder = destination_root + "/test_gt/"

        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.patch_size_valid = patch_size_valid
        self.number_image = 1
        self.image_name = ""

        logging.info("Configure patcher,")
        self._create_folders()

    def _create_folders(self):
        check_or_create_folder(self.train_folder)
        check_or_create_folder(self.train_gt_folder)
        check_or_create_folder(self.valid_folder)
        check_or_create_folder(self.valid_gt_folder)
        check_or_create_folder(self.test_folder)
        check_or_create_folder(self.test_gt_folder)
        logging.info("Prepared folders.")

    def create_patches(self, root_original: str, root_ground_truth: str, test_dataset, validation_dataset):
        logging.info("Start process ...")
        all_datasets = os.listdir(root_original)
        pbar = tqdm(all_datasets)

        try:
            for d_set in pbar:
                for im in os.listdir(root_original + d_set):
                    pbar.set_description(f"Processing {im} of {d_set}")
                    self.image_name = im
                    or_img = cv2.imread(root_original + d_set + '/' + im)
                    gt_img = cv2.imread(root_ground_truth + d_set + '/' + im)
                    if d_set not in [validation_dataset, test_dataset]:
                        self._split_train_images(or_img, gt_img, type="train")
                    if d_set == test_dataset:
                        self._split_train_images(or_img, gt_img, type="test")
                    if d_set == validation_dataset:
                        self._split_train_images(or_img, gt_img, type="valid")
            logging.info(f"Stored {self.number_image} training patches")
        except KeyboardInterrupt:
            logging.error("Keyboard Interrupt: stop running!")

    def _split_train_images(self, or_img: np.ndarray, gt_img: np.ndarray, type: str):
        flag = 1 if type == 'train' else 0
        runtime_size = self.overlap_size if type == "train" else self.patch_size_valid
        patch_size = self.patch_size if type == "train" else self.patch_size_valid
        for i in range(0, or_img.shape[0], runtime_size):
            for j in range(0, or_img.shape[1], runtime_size):

                if i + patch_size <= or_img.shape[0] and j + patch_size <= or_img.shape[1]:
                    dg_patch = or_img[i:i + patch_size, j:j + patch_size, :]
                    gt_patch = gt_img[i:i + patch_size, j:j + patch_size, :]

                elif i + patch_size > or_img.shape[0] and j + patch_size <= or_img.shape[1]:
                    dg_patch = (np.ones((patch_size, patch_size, 3)) - flag * random.randint(0, 1)) * 255
                    gt_patch = np.ones((patch_size, patch_size, 3)) * 255

                    dg_patch[0:or_img.shape[0] - i, :, :] = or_img[i:or_img.shape[0], j:j + patch_size, :]
                    gt_patch[0:or_img.shape[0] - i, :, :] = gt_img[i:or_img.shape[0], j:j + patch_size, :]

                elif i + patch_size <= or_img.shape[0] and j + patch_size > or_img.shape[1]:
                    dg_patch = (np.ones((patch_size, patch_size, 3)) - flag * random.randint(0, 1)) * 255
                    gt_patch = np.ones((patch_size, patch_size, 3)) * 255

                    dg_patch[:, 0:or_img.shape[1] - j, :] = or_img[i:i + patch_size, j:or_img.shape[1], :]
                    gt_patch[:, 0:or_img.shape[1] - j, :] = gt_img[i:i + patch_size, j:or_img.shape[1], :]

                else:
                    dg_patch = (np.ones((patch_size, patch_size, 3)) - flag * random.randint(0, 1)) * 255
                    gt_patch = np.ones((patch_size, patch_size, 3)) * 255

                    dg_patch[0:or_img.shape[0] - i, 0:or_img.shape[1] - j, :] = or_img[i:or_img.shape[0],
                                                                                j:or_img.shape[1],
                                                                                :]
                    gt_patch[0:or_img.shape[0] - i, 0:or_img.shape[1] - j, :] = gt_img[i:or_img.shape[0],
                                                                                j:or_img.shape[1],
                                                                                :]
                    gt_patch[0:or_img.shape[0] - i, 0:or_img.shape[1] - j, :] = gt_img[i:or_img.shape[0],
                                                                                j:or_img.shape[1],
                                                                                :]

                if type == "train":
                    cv2.imwrite(self.train_folder + str(self.number_image) + '.png', dg_patch)
                    cv2.imwrite(self.train_gt_folder + str(self.number_image) + '.png', gt_patch)
                    self.number_image += 1
                elif type == "test":
                    cv2.imwrite(self._create_name(self.test_folder, i, j), dg_patch)
                    cv2.imwrite(self._create_name(self.test_gt_folder, i, j), gt_patch)
                elif type == "valid":
                    cv2.imwrite(self._create_name(self.valid_folder, i, j), dg_patch)
                    cv2.imwrite(self._create_name(self.valid_gt_folder, i, j), gt_patch)

    def _create_name(self, folder: str, i: int, j: int):
        return folder + self.image_name.split('.')[0] + '_' + str(i) + '_' + str(j) + '.png'


def configure_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-destination', '--path_destination',
                        metavar='<path>',
                        type=str,
                        help=f"destination folder path with contains the patches",
                        default=config.path_destination)
    parser.add_argument('-original', '--path_original',
                        metavar='<path>',
                        type=str,
                        help="phe path witch contains the ruined images",
                        default=config.path_dataset)
    parser.add_argument('-ground_truth', '--path_ground_truth',
                        metavar='<path>',
                        type=str,
                        help="path which contains the ground truth images",
                        default=config.path_ground_truth)
    parser.add_argument('-size', '--patch_size',
                        metavar='<number>',
                        type=int,
                        help="size of ruined patch",
                        default=config.patch_size)
    parser.add_argument('-size_valid', '--patch_size_valid',
                        metavar='<number>',
                        type=int,
                        help='size of valid image patch',
                        default=config.patch_size_valid)
    parser.add_argument('-overlap', '--overlap_size',
                        metavar='<number>',
                        type=int,
                        help='overlap_size',
                        default=config.overlap_size)
    parser.add_argument('-validation', '--validation_dataset',
                        metavar='<path>',
                        type=str,
                        help='folder which contains images will are used to create the validation dataset',
                        default=config.validation_dataset)
    parser.add_argument('-test', '--testing_dataset',
                        dest="testing_dataset",
                        metavar='<path>',
                        type=str,
                        help='folder which contains images will are used to create the training dataset',
                        default=config.testing_dataset)

    return parser.parse_args()


if __name__ == '__main__':
    args = configure_args()
    root_original = args.path_original
    root_ground_truth = args.path_ground_truth
    destination = args.path_destination
    patch_size = args.patch_size
    patch_size_valid = args.patch_size_valid
    overlap_size = args.overlap_size
    validation_dataset = args.validation_dataset
    testing_dataset = args.testing_dataset

    patcher = PatchImage(patch_size=patch_size, patch_size_valid=patch_size_valid, overlap_size=overlap_size,
                         destination_root=destination)
    patcher.create_patches(root_original=root_original, root_ground_truth=root_ground_truth,
                           validation_dataset=validation_dataset, test_dataset=testing_dataset)
