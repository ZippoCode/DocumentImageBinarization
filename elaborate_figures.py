import argparse
import sys
from pathlib import Path

from utils.figure_utils import store_augmentation, fourier_transform


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

    return parser.parse_args()


if __name__ == '__main__':
    args = parser_arguments()
    image_one = Path(args.image_one)
    image_two = Path(args.image_two)
    ground_truth_one = Path(args.ground_truth_one)
    ground_truth_two = Path(args.ground_truth_two)

    store_augmentation(image_one, image_two, ground_truth_one, ground_truth_two)
    fourier_transform(Path("dataset/test/m02.png"), folder="results/merge")
    sys.exit()
