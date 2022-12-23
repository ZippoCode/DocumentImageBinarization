import os

from utils.htr_logging import get_logger

logger = get_logger(os.path.basename(__file__))


def store_images(parent_directory: str, directory: str, names: list, images: list):
    folder = os.path.join(parent_directory, directory)
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
        logger.debug(f"The folder \"{folder}\" has been created.")
    for name, image in zip(names, images):
        path = os.path.join(folder, name)
        image.save(path)
        logger.debug(f"Saved {name} image")
    logger.info(f"Stored {len(names)} in \"{folder}\"")
