import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch.utils.data
from torchvision.transforms import functional

from data.utils import reconstruct_ground_truth
from trainer.Validator import Validator
from utils.htr_logging import get_logger
from utils.ioutils import store_images, read_yaml
from utils.metrics import calculate_psnr

logger = get_logger(os.path.basename(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using {device} device")


def parser_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save_images', metavar='True or False', type=bool,
                        help=f"If TRUE will be saved the result images", default=True)
    parser.add_argument('-cfg', '--configuration', metavar='<name>', type=str,
                        help=f"The configuration name will use during running",
                        default="configs/evaluation/evaluation_2018.yaml")
    parser.add_argument('-net_cfg', '--network_configuration', metavar='<name>', type=str,
                        help=f"The filename will be used to configure the network", default="configs/network.yaml")
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

        validator = Validator(config=valid_config, network_cfg=network_cfg, device=device)
        logger.info("Start validation ...")

        with torch.no_grad():

            total_psnr = 0.0
            names, images = [], []

            for item in validator.valid_data_loader:
                image_name = item['image_name'][0]
                sample = item['sample']
                num_rows = item['num_rows'].item()
                samples_patches = item['samples_patches']
                gt_sample = item['gt_sample']

                samples_patches = samples_patches.squeeze(0)
                valid = samples_patches.to(device)
                gt_valid = gt_sample.to(device)

                valid = valid.squeeze(0).permute(1, 0, 2, 3)
                pred = validator.model(valid)
                pred = reconstruct_ground_truth(pred, gt_valid, num_rows=num_rows, channels=output_channels,
                                                batch=batch, patch_size=patch_size, stride=stride)
                total_psnr += calculate_psnr(pred, gt_valid)

                validator.compute(pred, gt_valid)

                pred = pred.squeeze(0).detach()
                pred_img = functional.to_pil_image(pred)
                names.append(image_name)
                images.append(pred_img)

                if args.view_details:
                    diff = 1.0 - (pred - gt_valid)
                    diff_img = functional.to_pil_image(diff.squeeze(0))
                    plt.imshow(diff_img, cmap='gray')
                    plt.show()

            avg_psnr = total_psnr / len(validator.valid_data_loader)
            psnr, precision, recall = validator.get_metrics()
            logger.info(f"Average PSNR: {avg_psnr:.6f} - {psnr:.6f}")
            logger.info(f"Precision {precision:.4f} - Recall {recall:.4f}")

            if args.save_images:
                store_images(parent_directory='results/evaluation', directory=valid_config["folder_destination"],
                             names=names, images=images)

    except KeyboardInterrupt:
        logger.warning("Validation interrupted by user")
    except FileNotFoundError as file_not_found:
        logger.error(f"File \"{file_not_found.filename}\" not found. Exit")
    except Exception as e:
        logger.error(f"Validation failed due to {e}")

    sys.exit()
