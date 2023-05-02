import argparse
import sys

import matplotlib.pyplot as plt
import torch.utils.data
import yaml
from torchvision.transforms import functional

from data.utils import reconstruct_ground_truth
from trainer.LaMaTrainer import calculate_psnr
from trainer.Validator import Validator
from utils.htr_logging import get_logger
from utils.ioutils import store_images

logger = get_logger('main')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using {device} device")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save_images', metavar='True or False', type=bool,
                        help=f"If TRUE will be saved the result images", default=False)
    parser.add_argument('-cfg', '--configuration', metavar='<name>', type=str,
                        help=f"The configuration name will use during running", default="evaluation_2016")
    parser.add_argument('-v', '--view_details', metavar='true or false', type=bool,
                        help=f"If TRUE the run will show the errors of predictions", default=False)

    args = parser.parse_args()

    logger.info("Start process ...")
    configuration_path = f"configs/evaluation/{args.configuration}.yaml"

    try:
        with open(configuration_path) as file:
            valid_config = yaml.load(file, Loader=yaml.Loader)
            file.close()
            logger.info(f"Loaded \"{configuration_path}\" configuration file")

        validator = Validator(config=valid_config, device=device)
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
                pred = reconstruct_ground_truth(pred, gt_valid, num_rows=num_rows, config=valid_config)
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
                store_images(parent_directory='results/evaluation', directory=valid_config["filename_checkpoint"],
                             names=names, images=images)

    except KeyboardInterrupt:
        logger.warning("Validation interrupted by user")
    except FileNotFoundError as file_not_found:
        logger.error(f"File \"{file_not_found.filename}\" not found. Exit")
    except Exception as e:
        logger.error(f"Validation failed due to {e}")

    sys.exit()
