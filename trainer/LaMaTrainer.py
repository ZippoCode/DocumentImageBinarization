import errno
import os

import torch
import torch.utils.data
from torchvision.transforms import functional

from data.dataloaders import make_train_dataloader, make_valid_dataloader
from data.datasets import make_train_dataset, make_valid_dataset
from data.utils import reconstruct_ground_truth
from modules.FFC import LaMa
from trainer.Losses import make_criterion
from trainer.Optimizers import make_optimizer
from utils.htr_logging import get_logger
from utils.metrics import calculate_psnr


class LaMaTrainingModule:

    def __init__(self, config: dict, network_cfg: dict, device=None):

        self.config = config
        self.device = device

        self.train_dataset = make_train_dataset(config)
        self.valid_dataset = make_valid_dataset(config)
        self.train_data_loader = make_train_dataloader(self.train_dataset, config)
        self.valid_data_loader = make_valid_dataloader(self.valid_dataset, config)

        self._input_channels = network_cfg['input_channels']
        self._output_channels = network_cfg['output_channels']

        self._batch = config['valid_batch_size']
        self._patch_size = config['valid_patch_size']
        self._stride = config['valid_stride']

        self.model = LaMa(input_nc=self._input_channels,
                          output_nc=self._output_channels,
                          init_conv_kwargs=network_cfg['init_conv_kwargs'],
                          downsample_conv_kwargs=network_cfg['down_sample_conv_kwargs'],
                          resnet_conv_kwargs=network_cfg['resnet_conv_kwargs'])

        # Training
        self.epoch = 0
        self.num_epochs = config['num_epochs']
        self.learning_rate = config['learning_rate']

        self.optimizer = make_optimizer(self.model, self.learning_rate, config['kind_optimizer'],
                                        network_cfg['optimizer'])
        self.criterion = make_criterion(kind=config['kind_loss']).to(device=device)

        # Validation
        self.best_epoch = 0
        self.best_psnr = 0.
        self.best_precision = 0.
        self.best_recall = 0.

        # Logging
        self.logger = get_logger(LaMaTrainingModule.__name__)

    def resume_checkpoints(self, folder: str, filename: str):
        checkpoints_path = f"{folder}{filename}_best_psnr.pth"

        if not os.path.exists(path=checkpoints_path):
            self.logger.warning(f"Checkpoints {checkpoints_path} not found.")
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoints_path)

        try:
            checkpoint = torch.load(checkpoints_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'], strict=True)
            self.epoch = checkpoint['epoch']
            self.best_psnr = checkpoint['best_psnr']
            self.learning_rate = checkpoint['learning_rate']
            self.logger.info(f"Loaded pretrained checkpoint model from \"{checkpoints_path}\"")
        except RuntimeError:
            self.logger.error("Error resuming checkpoints!")

    def save_checkpoints(self, root_folder: str, filename: str):
        os.makedirs(root_folder, exist_ok=True)
        checkpoint = {
            'model': self.model.state_dict(),
            'epoch': self.epoch,
            'best_psnr': self.best_psnr,
            'learning_rate': self.learning_rate,
        }

        dir_path = root_folder + f"{filename}_best_psnr.pth"
        torch.save(checkpoint, dir_path)
        self.logger.info(f"Stored checkpoints {dir_path}")

    def validation(self, threshold=0.5):
        valid_loss = 0.0
        valid_psnr = 0.0

        images = {}

        for item in self.valid_data_loader:
            image_name = item['image_name'][0]
            sample = item['sample']
            num_rows = item['num_rows'].item()
            samples_patches = item['samples_patches']
            gt_sample = item['gt_sample']

            samples_patches = samples_patches.squeeze(0)
            valid = samples_patches.to(self.device)
            gt_valid = gt_sample.to(self.device)

            valid = valid.squeeze(0)
            valid = valid.permute(1, 0, 2, 3)
            pred = self.model(valid)

            pred = reconstruct_ground_truth(pred, gt_valid, num_rows=num_rows, channels=self._output_channels,
                                            batch=self._batch, patch_size=self._patch_size, stride=self._stride)

            loss = self.criterion(pred, gt_valid)
            valid_loss += loss.item()

            pred = torch.where(pred > threshold, 1., 0.)
            valid_psnr += calculate_psnr(pred, gt_valid)

            valid = sample.squeeze(0).detach()
            pred = pred.squeeze(0).detach()
            gt_valid = gt_valid.squeeze(0).detach()
            valid_img = functional.to_pil_image(valid)
            pred_img = functional.to_pil_image(pred)
            gt_valid_img = functional.to_pil_image(gt_valid)
            images[image_name] = [valid_img, pred_img, gt_valid_img]

        avg_loss = valid_loss / len(self.valid_data_loader)
        avg_psnr = valid_psnr / len(self.valid_dataset)

        return avg_psnr, avg_loss, images
