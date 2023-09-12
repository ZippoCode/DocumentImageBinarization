import argparse
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as functional
import yaml
from PIL import Image
from torchvision.utils import make_grid

from utils.checkpoints import load_checkpoints
from utils.htr_logging import get_logger
from utils.ioutils import save_image
from utils.network_utils import configure_network

logger = get_logger('main')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using {device} device")

patch_size = 512
stride = 512
threshold = 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-im', '--image', metavar='<path>', type=str, help=f"Image will be convert to binary",
                        default="<path>")
    parser.add_argument('-cn', '--configuration_network', metavar='<name>', type=str,
                        help=f"The configuration of the network",
                        default="configs/network/network_blocks_9.yaml")
    parser.add_argument('-pc', '--path_checkpoints', metavar='<name>', type=str,
                        help=f"The path of the checkpoints file", default="lama_checkpoints.pth")
    parser.add_argument('-dst', '--path_destination', metavar='<path>', type=str,
                        help=f"Destination folder path with contains the result. Default: \"results\"",
                        default="results/")

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        logger.error(f"Image {image_path} not found!")
        sys.exit(-1)

    sample = Image.open(image_path).convert("RGB")
    height, width = sample.height, sample.width

    with open(args.configuration_network) as file:
        network_config = yaml.load(file, Loader=yaml.Loader)
        file.close()

    model = configure_network(network_config=network_config)
    model.to(device=device)
    load_checkpoints(model=model, device=device, checkpoints_path=args.path_checkpoints)

    with torch.no_grad():
        sample = functional.to_tensor(sample).unsqueeze(0)
        sample = sample.to(device=device)

        # Create Patches
        padding_bottom = ((height // patch_size) + 1) * patch_size - height
        padding_right = ((width // patch_size) + 1) * patch_size - width
        sample = functional.pad(img=sample, padding=[0, 0, padding_right, padding_bottom], fill=1)
        batch, channels, _, _ = sample.shape
        patches = sample.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        num_rows = patches.shape[3]
        patches = patches.reshape(batch, channels, -1, patch_size, patch_size)
        patches = patches.squeeze(0).permute(1, 0, 2, 3)
        logger.info("Created patches")

        prediction = model(patches)
        logger.info("Predicted image")

        # Reconstruct patches
        prediction = make_grid(prediction, nrow=num_rows, padding=0, value_range=(0, 1))
        prediction = functional.rgb_to_grayscale(prediction)
        prediction = functional.crop(prediction, top=0, left=0, height=height, width=width)

        # Apply threshold and save
        bin_image = (prediction > threshold) * 1.0
        bin_image = functional.to_pil_image(bin_image)
        filename = os.path.splitext(os.path.basename(args.image))[0]
        save_image(bin_image, directory=args.path_destination, filename=f"{filename}_bin")
        logger.info(f"Image saved")
        logger.info("End")

        sys.exit()
