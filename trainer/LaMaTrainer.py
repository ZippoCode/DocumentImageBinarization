import math
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.transforms.functional as functional
import wandb
from typing_extensions import TypedDict

from dataloader import HandwrittenTextImageDataset
from modules.FFC import LaMa
from trainer.CustomTransforms import create_train_transform, create_valid_transform
from utils.WandbLog import WandbLog
from utils.htr_logging import get_logger


class LMSELoss(torch.nn.MSELoss):
    def forward(self, input, target):
        mse = super().forward(input, target)
        mse = torch.tensor(max(mse, 1e-10))
        return torch.log10(mse)


class LaMaTrainingModule:

    def __init__(self, train_data_path: str, valid_data_path: str, input_channels: int, output_channels: int,
                 train_batch_size: int, valid_batch_size: int, train_split_size: int, valid_split_size: int,
                 epochs: int, workers: int, learning_rate: int, device=None, debug=False, **kwargs):

        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.train_split_size = train_split_size
        self.valid_split_size = valid_split_size

        self.train_dataset = None
        self.valid_dataset = None
        self.train_data_loader = None
        self.valid_data_loader = None
        self.optimizer = None
        self.wandb = None

        self.device = device
        self.workers = workers
        self.learning_rate = learning_rate

        self.debug = debug
        self.experiment_name = "With Simple Transforms and MSE Loss"

        # TO IMPROVE
        Arguments = TypedDict('Arguments', {'ratio_gin': float, 'ratio_gout': float})  # REMOVE
        init_conv_kwargs: Arguments = {'ratio_gin': 0, 'ratio_gout': 0}  # REMOVE
        down_sample_conv_kwargs: Arguments = {'ratio_gin': 0, 'ratio_gout': 0}  # REMOVE
        resnet_conv_kwargs: Arguments = {'ratio_gin': 0.75, 'ratio_gout': 0.75}  # REMOVE
        self.model = LaMa(input_nc=input_channels, output_nc=output_channels,
                          init_conv_kwargs=init_conv_kwargs, downsample_conv_kwargs=down_sample_conv_kwargs,
                          resnet_conv_kwargs=resnet_conv_kwargs)
        self.model.to(self.device)

        # Training
        self.epoch = 1
        self.num_epochs = epochs

        # Criterion
        self.criterion = LMSELoss().to(self.device)

        # Validation
        self.best_epoch = 0
        self.best_psnr = 0

        # Logging
        self.logger = get_logger(LaMaTrainingModule.__name__, debug)

        self._get_dataloader()
        self._create_optimizer()

        if not self.debug:  # Configuration Weights & Bias
            self._configure_wandb()

    def resume(self, folder):
        checkpoints_path = folder + "last_train.pth"
        self.logger.info(f"Loading pretrained model from {checkpoints_path}")

        if not os.path.exists(path=checkpoints_path):
            self.logger.error(f"[ERROR] Checkpoints {checkpoints_path} not found.")

        checkpoint = torch.load(checkpoints_path, map_location=None)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.epoch = checkpoint['epoch']
        self.best_psnr = checkpoint['best_psnr']

    def save_checkpoint(self, folder: str):
        path = folder + "last_train.pth"
        os.makedirs(folder, exist_ok=True)
        checkpoint = {
            'model': self.model.state_dict(),
            'epoch': self.epoch,
            'best_psnr': self.best_psnr
        }
        os.makedirs(folder, exist_ok=True)
        torch.save(checkpoint, path)
        self.logger.info(f"Stored checkpoints {path}")

    def validation(self, threshold=0.5):
        total_psnr = 0.0
        valid_loss = 0.0
        for valid, gt_valid in self.valid_data_loader:
            valid = valid.to(self.device)
            gt_valid = gt_valid.to(self.device)

            pred = self.model(valid)
            loss = self.criterion(pred, gt_valid)
            valid_loss += loss.item()

            pred = (pred > threshold) * 1.0
            pred_img = pred.detach().cpu().numpy()
            gt_img = gt_valid.detach().cpu().numpy()

            mse = np.mean((pred_img - gt_img) ** 2)
            psnr = 100 if mse == 0 else (20 * math.log10(1.0 / math.sqrt(mse)))
            total_psnr += psnr

            if self.wandb:
                self.wandb.on_log_images(valid, pred, gt_valid)

        avg_psnr = total_psnr / len(self.valid_data_loader)
        avg_loss = valid_loss / len(self.valid_data_loader)

        if self.wandb:  # Logs Valid Parameters
            logs = {
                'valid_psnr': total_psnr,
                'valid_avg_loss': avg_loss,
                'valid_loss': valid_loss,
            }
            self.wandb.on_log(logs)

        return avg_psnr, avg_loss

    def _configure_wandb(self):
        params = {
            "experiment_name": self.experiment_name,
            "learning_rage": self.learning_rate,
            "epochs": self.num_epochs,
            "batch_size": self.train_batch_size
        }
        self.wandb = WandbLog()
        self.wandb.setup(model=self.model, **params)

    def _get_dataloader(self):
        # Train
        train_transform = create_train_transform(patch_size=self.train_split_size, angle=15.2)
        self.train_dataset = HandwrittenTextImageDataset(self.train_data_path, self.train_data_path + '_gt',
                                                         transform=train_transform)
        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                                                             shuffle=True, num_workers=self.workers, pin_memory=True)
        # Valid
        transform = create_valid_transform(patch_size=self.valid_split_size)
        self.valid_dataset = HandwrittenTextImageDataset(self.valid_data_path, self.valid_data_path + '_gt',
                                                         transform=transform)
        self.valid_data_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.valid_batch_size,
                                                             num_workers=self.workers, pin_memory=True)

        self.logger.info(f"Training set has {len(self.train_dataset)} instances")
        self.logger.info(f"Validation set has {len(self.valid_dataset)} instances")

    def _create_optimizer(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.95),
                                     eps=1e-08, weight_decay=0.05, amsgrad=False)
