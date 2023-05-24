import argparse
import sys
from pathlib import Path

from utils.figure_utils import patch_merge, fourier_transform, equalize_image, enhance_contrast, augmentation, \
    thresholds


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-im1', '--image_one', metavar='<path>', type=str, help=f"Source image one",
                        default="dataset/patches/512/2018/train/487.png")
    parser.add_argument('-im2', '--image_two', metavar='<path>', type=str, help=f"Image will be convert to binary",
                        default="dataset/patches/512/2018/train/4785.png")
    parser.add_argument('-gt1', '--ground_truth_one', metavar='<path>', type=str,
                        help=f"Image will be convert to binary", default="dataset/patches/512/2018/train_gt/487.png")
    parser.add_argument('-gt2', '--ground_truth_two', metavar='<path>', type=str,
                        help=f"Image will be convert to binary", default="dataset/patches/512/2018/train_gt/4785.png")
    parser.add_argument('-fi', '--fourier_image', metavar='<path>', type=str,
                        help=f"Image will be applied the fourier transform",
                        default="dataset/training/ground_truth/2017/17.bmp")
    parser.add_argument('-ai', '--augmentation_image', metavar='<path>', type=str,
                        help=f"Image will be applied the augmentation",
                        default="dataset/training/original/2014/1.png")
    parser.add_argument('-ai_gt', '--augmentation_image_gt', metavar='<path>', type=str,
                        help=f"Image ground truth will be applied the augmentation",
                        default="dataset/training/ground_truth/2014/1.png")
    parser.add_argument('-f', '--folder', metavar='<path>', type=str,
                        help=f"Destination folder", default="results/figures")
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_arguments()
    image_one = Path(args.image_one)
    image_two = Path(args.image_two)
    ground_truth_one = Path(args.ground_truth_one)
    ground_truth_two = Path(args.ground_truth_two)
    folder = args.folder

    patch_merge(image_one, image_two, ground_truth_one, ground_truth_two, folder=folder)

    # Fourier
    fourier_image_path = Path(args.fourier_image)
    fourier_transform(fourier_image_path, folder=folder)

    # Enhancement
    # equalize_image(Path("dataset/training/original/2010/3.png"), folder=folder)
    # enhance_contrast(Path("dataset/training/original/2017/10.bmp"), folder=folder, alpha=1.5, beta=0)

    # Augmentation
    sample_path = Path(args.augmentation_image)
    gt_sample_path = Path(args.augmentation_image_gt)
    augmentation(sample_path, gt_sample_path, folder)

    thresholds(sample_path, folder=folder)
    sys.exit()
