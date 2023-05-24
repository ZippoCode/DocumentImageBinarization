import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
from torchvision.transforms import transforms

from data.CustomTransforms import RandomCrop, ColorJitter, RandomHorizontalFlip, RandomVerticalFlip, \
    RandomRotation
from utils.htr_logging import get_logger
from utils.ioutils import read_image, save_image

logger = get_logger(os.path.basename(__file__))


def thresholds(image_path: Path, folder: str, window_size=15, k=0.8):
    logger.info("Start thresholding methods process.")
    image = read_image(image_path, mode="L")

    binary_otsu = np.asarray(image > threshold_otsu(image)) * 255
    binary_niblack = np.asarray(image > threshold_niblack(image, window_size=window_size, k=k)) * 255
    binary_sauvola = np.asarray(image > threshold_sauvola(image, window_size=window_size)) * 255

    binary_adaptive_gaussian_7 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                       blockSize=31, C=15)
    binary_adaptive_gaussian_3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                       blockSize=63, C=15)

    save_image(image=binary_otsu, directory=folder, filename="otsu_binarization", log=True)
    save_image(image=binary_niblack, directory=folder, filename="niblack_binarization", log=True)
    save_image(image=binary_sauvola, directory=folder, filename="sauvola_binarization", log=True)
    save_image(image=binary_adaptive_gaussian_7, directory=folder, filename="adaptive_gaussian_binarization_7",
               log=True)
    save_image(image=binary_adaptive_gaussian_3, directory=folder, filename="adaptive_gaussian_binarization_3",
               log=True)


def equalize_image(image_path: Path, folder: str):
    logger.info("Start equalize process.")
    image = read_image(image_path, mode="L")
    if type(image) == Image.Image:
        image = np.ndarray(image)

    if len(image.shape) == 3:
        r_image, g_image, b_image = cv2.split(image)

        r_image_eq = cv2.equalizeHist(r_image)
        g_image_eq = cv2.equalizeHist(g_image)
        b_image_eq = cv2.equalizeHist(b_image)

        image_equalized = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
    else:
        image_equalized = cv2.equalizeHist(image)
    save_image(image=image_equalized, directory=folder, filename="equalized_image", log=True)


def enhance_contrast(image_path: Path, folder: str, alpha: float, beta: int):
    logger.info("Start enhance process")
    image = read_image(image_path, mode="RGB")
    new_image = np.zeros(image.shape, image.dtype)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)
    save_image(image=new_image, directory=folder, filename="enhanced_image", log=True)


def fourier_transform(image_path: Path, folder: str):
    logger.info("Start Fourier Transform process")
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
    save_image(image=rotated_image, directory=folder, filename="rotate_image", log=True)

    magnitude_spectrum_original = np.asarray(magnitude_spectrum_original, dtype=np.uint8)
    magnitude_spectrum_rotate = np.asarray(magnitude_spectrum_rotate, dtype=np.uint8)
    save_image(image=magnitude_spectrum_original, directory=folder, filename="ft_original", log=True)
    save_image(image=magnitude_spectrum_rotate, directory=folder, filename="ft_rotate", log=True)


def patch_merge(image_one_path: Path, image_two_path: Path, image_gt_one_path: Path, image_gt_two_path: Path,
                folder: str):
    logger.info("Start patch merge process")
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

    save_image(image=sample_one, directory=folder, filename="sample_one", log=True)
    save_image(image=sample_two, directory=folder, filename="sample_two", log=True)
    save_image(image=gt_sample_one, directory=folder, filename="gt_sample_one", log=True)
    save_image(image=gt_sample_two, directory=folder, filename="gt_sample_two", log=True)
    save_image(image=sample_img, directory=folder, filename="merge_original_result", log=True)
    save_image(image=gt_sample_img, directory=folder, filename="merge_gt_result", log=True)


def augmentation(image_path: Path, image_gt_path: Path, folder: str, patch_size=512):
    logger.info("Start augmentation process")
    assert image_path.exists(), f"Image {image_path} not exists"
    assert image_gt_path.exists(), f"Image {image_gt_path} not exists"

    sample = Image.open(image_path).convert("RGB")
    gt_sample = Image.open(image_gt_path).convert("L")
    transform = transforms.Compose(
        [RandomCrop(size=patch_size), ColorJitter(brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5),
         RandomHorizontalFlip(), RandomVerticalFlip(), RandomRotation((0, 360))])
    result = transform({'image': sample, 'gt': gt_sample})

    save_image(image=result['image'], directory=folder, filename="transform_sample", log=True)
    save_image(image=result['gt'], directory=folder, filename="transform_gt", log=True)
