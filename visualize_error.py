import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataloaders import make_train_dataloader
from data.datasets import make_train_dataset
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
                        default="configs/training/binary_cross_entropy_adam_2018_512.yaml")
    parser.add_argument('-ncfg', '--network_configuration', metavar='<name>', type=str,
                        help=f"The filename will be used to configure the network", default="configs/network.yaml")
    parser.add_argument('-cp', '--checkpoints_path', metavar='<path>', type=str,
                        help=f"The path of the checkpoints file", default="weights/bce_adam_2017_best_psnr.pth")
    parser.add_argument('-fn', '--filename', metavar='<string>', type=str,
                        help=f"Filename will be used to store the image result", default='patch_512')
    parser.add_argument('-fr', '--folder_result', metavar='<string>', type=str,
                        help=f"The folder will be used to store the image result", default='results/error_patch')
    parser.add_argument('-mi', '--max_iterations', metavar='<int>', type=int,
                        help=f"The max value of iterations", default=None)

    return parser.parse_args()


if __name__ == '__main__':
    try:

        args = parser_arguments()

        valid_configuration_path = args.configuration
        network_configuration_path = args.network_configuration
        checkpoints_path = args.checkpoints_path
        max_iterations = args.max_iterations
        filename = args.filename
        folder_result = args.folder_result

        logger.info("Start process ...")
        logger.info(f"Number max of iterations: {max_iterations}")

        config = read_yaml(valid_configuration_path)
        network_cfg = read_yaml(network_configuration_path)

        model = LaMa(input_nc=network_cfg['input_channels'],
                     output_nc=network_cfg['output_channels'],
                     init_conv_kwargs=network_cfg['init_conv_kwargs'],
                     downsample_conv_kwargs=network_cfg['down_sample_conv_kwargs'],
                     resnet_conv_kwargs=network_cfg['resnet_conv_kwargs'])
        model.to(device)
        load_checkpoints(model=model, checkpoints_path=checkpoints_path, device=device)
        model.eval()

        dataset = make_train_dataset(config)
        data_loader = make_train_dataloader(dataset, config)

        batch_size = config['train_batch_size']
        patch_size = config['train_patch_size']
        input_patch = np.zeros((batch_size, 3, patch_size, patch_size))
        error_patch = np.zeros((batch_size, 1, patch_size, patch_size))

        for batch_idx, (sample, gt_sample) in enumerate(data_loader):
            sample = sample.to(device)
            gt_sample = gt_sample.to(device)

            prediction = model(sample)

            input_patch += np.array(sample.cpu())
            error_patch += np.abs(np.array(prediction.detach().cpu()) - np.array(gt_sample.cpu()))

            if batch_idx % 100 == 0:
                size = batch_idx * len(sample)
                percentage = 100. * size / len(dataset)

                stdout = f"Elaborated: [{size} / {len(dataset)}] ({percentage:.2f}%)"
                logger.info(stdout)

            if max_iterations and batch_idx * len(sample) > max_iterations:
                break

        threshold = 0.5
        error_mask = error_patch > threshold

        patch = np.sum(error_patch, axis=0)
        patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch))
        patch = np.swapaxes(patch, 0, 2)

        image = ((patch - np.min(patch)) * (1 / (np.max(patch) - np.min(patch)) * 255))
        save_image(image=image, directory=folder_result, filename=filename, log=True)
        plt.imshow(patch)
        plt.show()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed due to {e}")
    finally:
        sys.exit()
