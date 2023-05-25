import argparse
import os
import sys

import torch.utils.data

from data.dataloaders import make_valid_dataloader
from data.datasets import make_valid_dataset
from trainer.Validator import Validator
from utils.checkpoints import load_checkpoints
from utils.htr_logging import get_logger
from utils.ioutils import read_yaml, store_images
from utils.network_utils import configure_network

logger = get_logger(os.path.basename(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using {device} device")
threshold = .5


def parser_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save_images', metavar='True or False', type=bool,
                        help=f"If TRUE will be saved the result images", default=True)
    parser.add_argument('-cfg', '--configuration', metavar='<name>', type=str,
                        help=f"The configuration name will use during running",
                        default="configs/evaluation/evaluation_2012.yaml")
    parser.add_argument('-net_cfg', '--network_configuration', metavar='<name>', type=str,
                        help=f"The filename will be used to configure the network",
                        default="configs/network/network_blocks_9.yaml")
    parser.add_argument('-v', '--view_details', metavar='true or false', type=bool,
                        help=f"If TRUE the run will show the errors of predictions", default=False)

    return parser.parse_args()


if __name__ == '__main__':

    logger.info("Start process ...")
    args = parser_arguments()

    try:
        valid_config = read_yaml(args.configuration)
        network_cfg = read_yaml(args.network_configuration)

        output_channels = network_cfg['output_channels']
        batch = valid_config['valid_batch_size']
        patch_size = valid_config['valid_patch_size']
        stride = valid_config['valid_stride']

        model = configure_network(network_config=network_cfg)

        folder = valid_config["path_checkpoint"]
        filename = valid_config["filename_checkpoint"]
        checkpoints_path = os.path.join(folder, filename)
        load_checkpoints(model=model, checkpoints_path=checkpoints_path, device=device)

        validator = Validator(model=model, device=device)

        valid_dataset = make_valid_dataset(valid_config)
        valid_data_loader = make_valid_dataloader(valid_dataset, valid_config)

        metrics, names, images = validator.compute(data_loader=valid_data_loader, output_channels=output_channels,
                                                   batch_size=batch, patch_size=patch_size, stride_size=stride)
        logger.info(f"Average PSNR: {metrics.psnr:.6f}")
        logger.info(f"Precision {metrics.precision:.4f}")
        logger.info(f"Recall {metrics.recall:.4f}")

        if args.save_images:
            root_folder = 'results/evaluation'
            directory = valid_config["folder_destination"]
            store_images(parent_directory=root_folder, directory=directory, names=names, images=images)

    except KeyboardInterrupt:
        logger.warning("Validation interrupted by user")
    except FileNotFoundError as file_not_found:
        logger.error(f"File \"{file_not_found.filename}\" not found. Exit")
    except Exception as e:
        logger.error(f"Validation failed due to {e}")

    sys.exit()
