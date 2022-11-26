import random

from torchvision import transforms
from torchvision.transforms import functional


class ToTensor(transforms.ToTensor):

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image = super().__call__(image)
        gt = super().__call__(gt)
        return {'image': image, 'gt': gt}


class RandomCrop(transforms.RandomCrop):

    def __init__(self, size):
        super(RandomCrop, self).__init__(size=size)
        self.size = size

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        i, j, h, w = self.get_params(image, output_size=(self.size, self.size))
        image = functional.crop(image, i, j, h, w)
        gt = functional.crop(gt, i, j, h, w)
        return {'image': image, 'gt': gt}


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        if random.random() < self.p:
            image = functional.hflip(image)
            gt = functional.hflip(gt)
        return {'image': image, 'gt': gt}


class RandomVerticalFlip(transforms.RandomVerticalFlip):

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        if random.random() < self.p:
            image = functional.vflip(image)
            gt = functional.vflip(gt)
        return {'image': image, 'gt': gt}


class ColorJitter(transforms.ColorJitter):

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image = super().__call__(image)
        return {'image': image, 'gt': gt}


class RandomRotation(transforms.RandomRotation):

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        angle = self.get_params(self.degrees)
        image = functional.rotate(image, angle)
        gt = functional.invert(gt)
        gt = functional.rotate(gt, angle)
        gt = functional.invert(gt)
        return {'image': image, 'gt': gt}


class CenterCrop(transforms.CenterCrop):

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image = super().forward(image)
        gt = super().forward(gt)
        return {'image': image, 'gt': gt}


class RandomSolarize(transforms.RandomSolarize):
    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        if random.random() < self.p:
            image = functional.solarize(image, threshold=self.threshold)
            gt = functional.solarize(gt, threshold=self.threshold)
        return {'image': image, 'gt': gt}


class GaussianBlur(transforms.GaussianBlur):
    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        image = super().forward(image)
        gt = super().forward(gt)
        return {'image': image, 'gt': gt}


def create_train_transform(patch_size: int, angle: float):
    random_threshold = random.uniform(0.0, 1.0)
    transform = transforms.Compose([
        ColorJitter(),
        # RandomSolarize(threshold=random_threshold),
        # GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        RandomRotation((0, 360)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomCrop(patch_size),
        ToTensor()
    ])
    return transform


def create_valid_transform(patch_size: int):
    # transform = transforms.Compose([
    #     CenterCrop(patch_size),
    #     ToTensor()
    # ])
    transform = transforms.Compose([
        ToTensor()
    ])
    return transform
