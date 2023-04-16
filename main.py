import argparse
import os
import sys

import torch
import torchvision.transforms.functional as functional
import yaml
from PIL import Image
from torchvision.utils import make_grid

from modules.FFC import LaMa
from utils.checkpoints import load_checkpoints
from utils.htr_logging import get_logger
from utils.ioutils import save_image

logger = get_logger('main')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using {device} device")

patch_size = 256
stride = 256
threshold = 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-im', '--image', metavar='<path>', type=str, help=f"Image will be convert to binary",
                        default="dataset/test/l01.png")
    parser.add_argument('-cn', '--configuration_network', metavar='<name>', type=str,
                        help=f"The configuration of the network", default="configs/network.yaml")
    parser.add_argument('-pc', '--path_checkpoints', metavar='<name>', type=str,
                        help=f"The path of the checkpoints file", default="weights/lama_checkpoints.pth")
    parser.add_argument('-dst', '--path_destination', metavar='<path>', type=str,
                        help=f"Destination folder path with contains the result. Default: \"results\"",
                        default="results/testing")

    args = parser.parse_args()

    image_path = args.image
    sample = Image.open(image_path).convert("RGB")
    height, width = sample.height, sample.width

    with open(args.configuration_network) as file:
        network_config = yaml.load(file, Loader=yaml.Loader)
        file.close()

    model = LaMa(input_nc=network_config['input_channels'], output_nc=network_config['output_channels'],
                 init_conv_kwargs=network_config['init_conv_kwargs'],
                 downsample_conv_kwargs=network_config['down_sample_conv_kwargs'],
                 resnet_conv_kwargs=network_config['resnet_conv_kwargs'])
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
        logger.info("Created patches ...")

        prediction = model(patches)
        logger.info("Predicted image ...")

        # Reconstruct patches
        prediction = make_grid(prediction, nrow=num_rows, padding=0, value_range=(0, 1))
        prediction = functional.rgb_to_grayscale(prediction)
        prediction = functional.crop(prediction, top=0, left=0, height=height, width=width)

        # Apply threshold and save
        bin_image = (prediction > threshold) * 1.0
        bin_image = functional.to_pil_image(bin_image)
        filename = os.path.splitext(os.path.basename(args.image))[0]
        save_image(bin_image, directory=args.path_destination, filename=f"{filename}_bin.png")
        logger.info("End.")

        sys.exit()
