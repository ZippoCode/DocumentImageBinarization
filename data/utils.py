import math
import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms, functional
from torchvision.utils import make_grid

import trainer.CustomTransforms as CustomTransform


def get_path(root: str, paths: list, index: int):
    assert index < len(paths)
    return os.path.join(root, paths[index])


def get_transform(transform_variant: str, output_size: int):
    if transform_variant == 'default':
        transform = transforms.Compose([
            CustomTransform.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5),
            CustomTransform.RandomRotation((0, 360)),
            CustomTransform.RandomHorizontalFlip(),
            CustomTransform.RandomVerticalFlip(),
            CustomTransform.RandomCrop(output_size),
            CustomTransform.ToTensor()
        ])
    elif transform_variant == 'gaussian':
        transform = transforms.Compose([
            CustomTransform.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5),
            CustomTransform.GaussianBlur(kernel_size=(3, 5), sigma=(0.3, 1.5)),
            CustomTransform.RandomRotation((0, 360)),
            CustomTransform.RandomHorizontalFlip(),
            CustomTransform.RandomVerticalFlip(),
            CustomTransform.RandomCrop(output_size),
            CustomTransform.ToTensor()
        ])
    elif transform_variant == 'equalize_contrast':
        transform = transforms.Compose([
            CustomTransform.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5),
            CustomTransform.RandomEqualize(),
            CustomTransform.RandomAutoContrast(),
            CustomTransform.RandomRotation((0, 360)),
            CustomTransform.RandomHorizontalFlip(),
            CustomTransform.RandomVerticalFlip(),
            CustomTransform.RandomCrop(output_size),
            CustomTransform.ToTensor()
        ])
    elif transform_variant == 'adjust_sharpness':
        transform = transforms.Compose([
            CustomTransform.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5),
            CustomTransform.RandomAdjustSharpness(sharpness_factor=0),
            CustomTransform.RandomRotation((0, 360)),
            CustomTransform.RandomHorizontalFlip(),
            CustomTransform.RandomVerticalFlip(),
            CustomTransform.RandomCrop(output_size),
            CustomTransform.ToTensor()
        ])
    elif transform_variant == 'all_transforms':
        transform = transforms.Compose([
            CustomTransform.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5),
            CustomTransform.GaussianBlur(kernel_size=(3, 5), sigma=(0.3, 1.5)),
            CustomTransform.RandomAdjustSharpness(sharpness_factor=0),
            CustomTransform.RandomEqualize(),
            CustomTransform.RandomAutoContrast(),
            CustomTransform.RandomRotation((0, 360)),
            CustomTransform.RandomHorizontalFlip(),
            CustomTransform.RandomVerticalFlip(),
            CustomTransform.RandomCrop(output_size),
            CustomTransform.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            CustomTransform.ToTensor()
        ])
    return transform


def get_patches(image_source: Image, patch_size: int, stride: int):
    image = np.asarray(image_source)
    image_patches = []

    h = ((image.shape[0] // patch_size) + 1) * patch_size
    w = ((image.shape[1] // patch_size) + 1) * patch_size

    padding_image = np.ones((h, w, 3)) if len(image.shape) == 3 else np.ones((h, w))
    padding_image = padding_image * 255.0
    padding_image[:image.shape[0], :image.shape[1]] = image

    for j in range(0, w - patch_size + 1, stride):
        for i in range(0, h - patch_size + 1, stride):
            image_patches.append(padding_image[i:i + patch_size, j:j + patch_size])

    num_rows = math.floor((padding_image.shape[0] - patch_size) / stride) + 1
    num_cols = math.floor((padding_image.shape[1] - patch_size) / stride) + 1

    return np.array(image_patches), num_rows, num_cols


def reconstruct_ground_truth(patches: torch.Tensor, original: torch.Tensor, num_rows: int, config: dict):
    channels = config['output_channels']
    batch = config['valid_batch_size']
    patch_size = config['valid_patch_size']
    stride = config['valid_stride']

    if stride == 128:
        _, _, width, height = original.shape

        x_steps = [x + (stride // 2) for x in range(0, width, stride)]
        x_steps[0], x_steps[-1] = 0, width
        y_steps = [y + (stride // 2) for y in range(0, height, stride)]
        y_steps[0], y_steps[-1] = 0, height

        patches = patches.view(batch, channels, -1, num_rows, patch_size, patch_size)
        canvas = torch.zeros_like(original)
        for j in range(len(x_steps) - 1):
            for i in range(len(y_steps) - 1):
                patch = patches[0, :, j, i, :, :]
                x1_abs, x2_abs = x_steps[j], x_steps[j + 1]
                y1_abs, y2_abs = y_steps[i], y_steps[i + 1]
                x1_rel, x2_rel = x1_abs - (j * stride), x2_abs - (j * stride)
                y1_rel, y2_rel = y1_abs - (i * stride), y2_abs - (i * stride)
                canvas[0, :, x1_abs:x2_abs, y1_abs:y2_abs] = patch[:, x1_rel:x2_rel, y1_rel:y2_rel]
    else:
        tensor = make_grid(patches, nrow=num_rows, padding=0, value_range=(0, 1))
        tensor = functional.rgb_to_grayscale(tensor)
        _, _, height, width = original.shape
        canvas = functional.crop(tensor, top=0, left=0, height=height, width=width)
        canvas = canvas.unsqueeze(0)

    return canvas
