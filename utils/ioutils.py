import os

from utils.htr_logging import get_logger

logger = get_logger(os.path.basename(__file__))


def store_images(root: str, folder: str, names: list, images: list):
    folder = f'{root}/{folder}/'
    os.makedirs(folder, exist_ok=True)
    for name, image in zip(names, images):
        path = folder + name
        image.save(path)
        logger.debug(f"Saved {name} image")
    logger.info(f"Stored {len(names)} in \"{folder}\"")
