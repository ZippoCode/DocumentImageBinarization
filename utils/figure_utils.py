import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils.htr_logging import get_logger
from utils.ioutils import read_image, save_image

logger = get_logger(os.path.basename(__file__))


def fourier_transform(image_path: Path, folder: str):
    logger.info("Start process")
    image = read_image(image_path, mode="L")

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum_original = 20 * np.log(np.abs(fshift))

    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)
    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height), borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=255)
    f = np.fft.fft2(rotated_image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum_rotate = 20 * np.log(np.abs(fshift))

    save_image(image=image, directory=folder, filename="original", log=True)
    save_image(image=magnitude_spectrum_original, directory=folder, filename="ft_original", log=True)
    save_image(image=rotated_image, directory=folder, filename="rotate_image", log=True)
    save_image(image=magnitude_spectrum_rotate, directory=folder, filename="ft_rotate", log=True)


def store_augmentation(image_one_path: Path, image_two_path: Path, image_gt_one_path: Path, image_gt_two_path: Path):
    logger.info("Start augmentation process")
    assert image_one_path.exists(), f"Image {image_one_path} not exists"
    assert image_two_path.exists(), f"Image {image_two_path} not exists"
    assert image_gt_one_path.exists(), f"Image {image_gt_one_path} not exists"
    assert image_gt_two_path.exists(), f"Image {image_gt_two_path} not exists"

    sample_one = read_image(source_path=image_one_path)
    sample_two = read_image(source_path=image_two_path)

    gt_sample_one = read_image(source_path=image_gt_one_path, mode="L")
    gt_sample_two = read_image(source_path=image_gt_two_path, mode="L")

    sample = np.minimum(sample_one, sample_two)
    gt_sample = np.minimum(gt_sample_one, gt_sample_two)

    sample_img = Image.fromarray(np.uint8(sample))
    gt_sample_img = Image.fromarray(np.uint8(gt_sample))

    save_image(image=sample_one, directory="results/merge", filename="sample_one", log=True)
    save_image(image=sample_two, directory="results/merge", filename="sample_two", log=True)
    save_image(image=gt_sample_one, directory="results/merge", filename="gt_sample_one", log=True)
    save_image(image=gt_sample_two, directory="results/merge", filename="gt_sample_two", log=True)
    save_image(image=sample_img, directory="results/merge", filename="merge_original_result", log=True)
    save_image(image=gt_sample_img, directory="results/merge", filename="merge_gt_result", log=True)
