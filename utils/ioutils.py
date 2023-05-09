import os
from typing import Union

import numpy as np
import yaml
from PIL import Image

from utils.htr_logging import get_logger

logger = get_logger(os.path.basename(__file__))


def create_folder(folder_name: str, clean=False):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
        logger.info(f"The folder \"{folder_name}\" has been created.")
    elif clean:
        os.system('rm -rf ' + folder_name + '*')


def read_yaml(yaml_filename: str):
    try:
        with open(yaml_filename, 'r') as file:
            yaml_file = yaml.load(file, Loader=yaml.Loader)
            file.close()
            logger.info(f"Read \"{yaml_filename}\" configuration file")
    except FileNotFoundError:
        logger.warning(f"File \"{yaml_filename}\" not found")
        yaml_file = dict()
    return yaml_file


def read_image(source_path: str, mode="RGB"):
    with Image.open(source_path) as img:
        image = img.convert(mode=mode)
        image = np.asarray(image, dtype=np.uint8)
    return image


def save_image(image: Union[np.ndarray, Image.Image], directory: str, filename: str, img_format="PNG", log=False):
    create_folder(directory)

    if filename[-4:].lower() != img_format.lower():
        filename += f".{img_format.lower()}"
    filename_path = os.path.join(directory, filename)
    if type(image) == np.ndarray:
        image = image.astype(dtype=np.uint8)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = Image.fromarray(image, mode="RGB")
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = np.squeeze(image, axis=2)
            image = Image.fromarray(image, mode="L")
        else:
            image = Image.fromarray(image, mode="L")

    image.save(fp=filename_path, format=img_format)

    if log:
        logger.info(f"Stored \"{filename}\" into: \"{directory}\"")


def store_images(parent_directory: str, directory: str, names: list, images: list):
    folder = os.path.join(parent_directory, directory)
    create_folder(folder)
    for name, image in zip(names, images):
        path = os.path.join(folder, name)
        image.save(path)
        logger.debug(f"Saved {name} image")
    logger.info(f"Stored {len(names)} in \"{folder}\"")
