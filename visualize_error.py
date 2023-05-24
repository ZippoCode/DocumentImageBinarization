import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import transforms

from data.CustomTransforms import RandomCrop, ToTensor
from data.TrainingDataset import TrainingDataset
from modules.FFC import LaMa
from utils.checkpoints import load_checkpoints
from utils.htr_logging import get_logger
from utils.ioutils import read_yaml, save_image

logger = get_logger(os.path.basename(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using {device} device")


def parser_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', '--configuration', metavar='<name>', type=str,
                        help=f"The configuration name will use during running",
                        default="configs/visualize_error.yaml")
    parser.add_argument('-ncfg', '--network_configuration', metavar='<name>', type=str,
                        help=f"The filename will be used to configure the network", default="configs/network.yaml")

    return parser.parse_args()


if __name__ == '__main__':
    try:

        args = parser_arguments()

        configuration_path = args.configuration
        network_configuration_path = args.network_configuration

        logger.info("Start process ...")

        config = read_yaml(configuration_path)
        network_cfg = read_yaml(network_configuration_path)

        model = LaMa(input_nc=network_cfg['input_channels'],
                     output_nc=network_cfg['output_channels'],
                     init_conv_kwargs=network_cfg['init_conv_kwargs'],
                     downsample_conv_kwargs=network_cfg['down_sample_conv_kwargs'],
                     resnet_conv_kwargs=network_cfg['resnet_conv_kwargs'])
        model.to(device)

        checkpoints_path = config["checkpoints_path"]
        folder_result = config["folder_result"]
        filename = config["filename"]
        max_iterations = config["max_iterations"]

        load_checkpoints(model=model, checkpoints_path=checkpoints_path, device=device)
        model.eval()

        data_path = config['data_path']
        gt_data_path = config['gt_data_path']

        patch_size = config['patch_size']
        dataloader_config = config['kwargs']

        logger.info("Start process ...")
        logger.info(f"Number max of iterations: {max_iterations}")
        logger.info(f"Patch size {patch_size}")

        transform = transforms.Compose([RandomCrop(size=patch_size), ToTensor()])
        dataset = TrainingDataset(root_dg_dir=data_path, root_gt_dir=gt_data_path, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset, **dataloader_config)

        error_map = None

        for batch_idx, (sample, gt_sample) in enumerate(data_loader):
            sample = sample.to(device)
            gt_sample = gt_sample.to(device)

            with torch.no_grad():
                outputs = model(sample)
                outputs = torch.where(outputs > 0, 1., 0.)

            if error_map is None:
                error_map = torch.abs(outputs - gt_sample)
            else:
                error_map += torch.abs(outputs - gt_sample)

            if batch_idx % 100 == 0:
                size = batch_idx * len(sample)
                percentage = 100. * size / len(dataset)

                stdout = f"[{size} / {len(dataset)}] ({percentage:.2f}%) elaborated ..."
                logger.info(stdout)

            if max_iterations and batch_idx * len(sample) > max_iterations:
                break

        error_map = torch.mean(error_map, dim=0)
        error_patch = error_map.cpu().numpy()
        min, max = np.min(error_patch), np.max(error_patch)
        error_patch = (error_patch - min) * (1 / (max - min) * 255)
        error_patch = np.swapaxes(error_patch, 0, 2)

        plt.imshow(error_patch, cmap='inferno')
        plt.xlabel('Patch Column')
        plt.ylabel('Patch Row')
        plt.title('Error Distribution in Patches')
        plt.show()

        save_image(error_patch, directory=folder_result, filename=filename, log=True)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed due to {e}")
    finally:
        sys.exit()
