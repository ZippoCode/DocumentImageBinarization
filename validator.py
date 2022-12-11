import os
import sys

import torch
import yaml

from trainer.LaMaTrainer import LaMaTrainingModule
from utils.htr_logging import get_logger

logger = get_logger('main')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using {device} device")

if __name__ == '__main__':
    logger.info("Start process ...")
    config_filename = "evaluation_custom_mse"
    configuration_path = f"configs/evaluation/{config_filename}.yaml"

    try:
        with open(configuration_path) as file:
            valid_config = yaml.load(file, Loader=yaml.Loader)
            file.close()
            logger.info(f"Loaded \"{configuration_path}\" configuration file")

        trainer = LaMaTrainingModule(valid_config, device=device)
        trainer.load_checkpoints(valid_config['path_checkpoint'], valid_config["filename_checkpoint"])

        if torch.cuda.is_available():
            trainer.model.cuda()

        logger.info("Start validation ...")

        trainer.model.eval()
        with torch.no_grad():
            valid_psnr, valid_loss, images = trainer.validation()
            logger.info("Terminate validation. Result:")
            logger.info(f"Average Loss: {valid_loss:0.6f} - Average PSNR: {valid_psnr:0.6f}")

            folder = f'results/evaluation/{valid_config["filename_checkpoint"]}/'
            os.makedirs(folder, exist_ok=True)
            for name_image, predicted_image in images.items():
                path = folder + name_image
                predicted_image.save(path)
            logger.info(f"Store {images.__len__()} predicted images in \"{folder}\"")

    except KeyboardInterrupt:
        logger.warning("Validation interrupted by user")
    except FileNotFoundError as file_not_found:
        logger.error(f"File \"{file_not_found.filename}\" not found. Exit")
    except Exception as e:
        logger.error(f"Validation failed due to {e}")

    sys.exit()
