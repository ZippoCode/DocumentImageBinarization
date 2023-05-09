import errno
import os

import torch

from utils.htr_logging import get_logger
from utils.ioutils import create_folder

logger = get_logger(os.path.basename(__file__))


def load_checkpoints(model: torch.nn.Module, checkpoints_path: str, device=torch.device('cpu')):
    if not os.path.exists(path=checkpoints_path):
        logger.warning(f"Checkpoints {checkpoints_path} not found.")
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoints_path)

    checkpoint = torch.load(checkpoints_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)
    logger.info(f"Loaded pretrained checkpoint model from \"{checkpoints_path}\"")


def save_checkpoints(model: torch.nn.Module, filename: str, root_folder='weights/'):
    if not root_folder.endswith('/'):
        root_folder = root_folder + '/'

    create_folder(folder_name=root_folder)
    dir_path = f"{root_folder}{filename}_checkpoints.pth"

    checkpoint = {
        'model': model.state_dict(),
    }
    torch.save(checkpoint, dir_path)
    logger.info(f"Stored {filename}_checkpoints.pth in {root_folder}")
