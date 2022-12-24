from torchvision.transforms import transforms

from data.TrainingDataset import TrainingDataset
from data.ValidationDataset import ValidationDataset
from data.utils import get_transform
from utils.htr_logging import get_logger

logger = get_logger(__file__)


def make_train_dataset(config: dict):
    train_data_path = config['train_data_path']
    train_gt_data_path = config['train_gt_data_path']
    transform_variant = config['train_transform_variant'] if 'train_transform_variant' in config else None
    patch_size = config['train_patch_size']

    logger.info(f"Train path: \"{train_data_path}\" - Train ground truth path: \"{train_gt_data_path}\"")
    logger.info(f"Transform Variant: {transform_variant} - Training Patch Size: {patch_size}")

    transform = get_transform(transform_variant=transform_variant, output_size=patch_size)
    train_dataset = TrainingDataset(train_data_path, train_gt_data_path, transform=transform)
    logger.info(f"Training set has {len(train_dataset)} instances")

    return train_dataset


def make_valid_dataset(config: dict):
    valid_data_path = config['valid_data_path']
    valid_gt_data_path = config['valid_gt_data_path']
    patch_size = config['valid_patch_size']
    stride = config['valid_stride']

    transform = transforms.Compose([transforms.ToTensor()])

    valid_dataset = ValidationDataset(root_dg_dir=valid_data_path, root_gt_dir=valid_gt_data_path,
                                      patch_size=patch_size, stride=stride, transform=transform)
    logger.info(f"Validation set has {len(valid_dataset)} instances")
    return valid_dataset
