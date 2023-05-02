import os
from typing import Union

import numpy as np
from PIL import Image

from utils.htr_logging import get_logger

logger = get_logger(os.path.basename(__file__))


def create_folder(folder_name: str, clean=False):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
        logger.info(f"The folder \"{folder_name}\" has been created.")
    elif clean:
        os.system('rm -rf ' + folder_name + '*')


def read_image(source_path: str, mode="RGB"):
    with Image.open(source_path) as img:
        image = img.convert(mode=mode)
        image = np.asarray(image, dtype=np.uint8)
    return image


def save_image(image: Union[np.ndarray, Image.Image], directory: str, filename: str, img_format="PNG", log=False):
    create_folder(directory)
    filename_path = os.path.join(directory, filename)
    if type(image) == np.ndarray:
        image = image.astype(dtype=np.uint8)
        image = Image.fromarray(image, mode="RGB")
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
