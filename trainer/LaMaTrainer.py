import math
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.utils
import wandb
from torchmetrics import PeakSignalNoiseRatio
from torchvision.transforms import functional
from typing_extensions import TypedDict

from dataloader import HandwrittenTextImageDataset, ValidationDataLoader
from modules.FFC import LaMa
from trainer.CustomTransforms import create_train_transform, create_valid_transform
from utils.htr_logging import get_logger


def calculate_psnr(predicted: torch.Tensor, ground_truth: torch.Tensor, threshold=0.5):
    pred_img = predicted.detach().cpu().numpy()
    gt_img = ground_truth.detach().cpu().numpy()

    pred_img = (pred_img > threshold) * 1.0

    mse = np.mean((pred_img - gt_img) ** 2)
    psnr = 100 if mse == 0 else (20 * math.log10(1.0 / math.sqrt(mse)))
    return psnr


class LaMaTrainingModule:

    def __init__(self, train_data_path: str, train_gt_data_path: str, valid_data_path: str, valid_gt_data_path: str,
                 input_channels: int, output_channels: int, train_batch_size: int, valid_batch_size: int,
                 train_split_size: int, valid_split_size: int, epochs: int, workers: int, learning_rate: int,
                 experiment_name: str, device=None, debug=False, criterion=None, wandb_log=None):

        # Path
        self.train_data_path = train_data_path
        self.train_gt_data_path = train_gt_data_path
        self.valid_data_path = valid_data_path
        self.valid_gt_data_path = valid_gt_data_path

        # Batch Size
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size

        # Split Size
        self.train_split_size = train_split_size
        self.valid_split_size = valid_split_size

        self.train_dataset = None
        self.valid_dataset = None
        self.train_data_loader = None
        self.valid_data_loader = None
        self.optimizer = None

        self.device = device
        self.workers = workers
        self.learning_rate = learning_rate
        self.wandb_log = wandb_log

        self.debug = debug

        # TO IMPROVE
        arguments = TypedDict('arguments', {'ratio_gin': float, 'ratio_gout': float})  # REMOVE
        init_conv_kwargs: arguments = {'ratio_gin': 0, 'ratio_gout': 0}  # REMOVE
        down_sample_conv_kwargs: arguments = {'ratio_gin': 0, 'ratio_gout': 0}  # REMOVE
        resnet_conv_kwargs: arguments = {'ratio_gin': 0.75, 'ratio_gout': 0.75}  # REMOVE
        self.model = LaMa(input_nc=input_channels, output_nc=output_channels,
                          init_conv_kwargs=init_conv_kwargs, downsample_conv_kwargs=down_sample_conv_kwargs,
                          resnet_conv_kwargs=resnet_conv_kwargs)
        self.model.to(self.device)

        # Training
        self.epoch = 0
        self.num_epochs = epochs
        self.experiment_name = experiment_name

        # Criterion
        self.criterion = criterion

        # Validation
        self.psnr_metric = PeakSignalNoiseRatio().to(device=device)
        self.best_epoch = 0
        self.best_psnr = 0

        # Logging
        self.logger = get_logger(LaMaTrainingModule.__name__, debug)

        self._make_datasets()
        self._make_dataloaders()
        self._create_optimizer()

    def load_checkpoints(self, folder: str):
        checkpoints_path = folder + "best_psnr.pth"
        self.logger.info(f"Loading pretrained model from {checkpoints_path}")

        if not os.path.exists(path=checkpoints_path):
            self.logger.error(f"[ERROR] Checkpoints {checkpoints_path} not found.")

        checkpoint = torch.load(checkpoints_path, map_location=None)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        self.best_psnr = checkpoint['best_psnr']
        self.criterion = checkpoint['criterion']

    def save_checkpoints(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'best_psnr': self.best_psnr,
            'criterion': self.criterion
        }

        path = folder + "best_psnr.pth"
        torch.save(checkpoint, path)
        self.logger.info(f"Stored checkpoints {path}")

    def validation(self, threshold=0.5):
        total_psnr = 0.0
        valid_loss = 0.0
        folder = 'results/' + self.experiment_name + '/'
        os.makedirs(folder, exist_ok=True)

        for item in self.valid_data_loader:

            sample = item['sample']
            num_rows = item['num_rows'].item()
            samples_patches = item['samples_patches']
            gt_sample = item['gt_sample']

            valid = samples_patches.to(self.device)
            gt_valid = gt_sample.to(self.device)

            valid = valid.squeeze(0)
            valid = valid.permute(0, 3, 2, 1)

            pred = self.model(valid)
            pred = torchvision.utils.make_grid(pred, nrow=num_rows, padding=0, value_range=(0, 1))
            pred = functional.rgb_to_grayscale(pred)
            height, width = gt_valid.shape[3], gt_valid.shape[2]
            pred = functional.crop(pred, top=0, left=0, height=height, width=width)
            pred = pred.permute(0, 2, 1).unsqueeze(0)

            loss = self.criterion(pred, gt_valid)
            valid_loss += loss.item()

            psnr = calculate_psnr(pred, gt_valid)
            total_psnr += psnr

            if self.wandb_log:
                valid = sample.squeeze(0).detach()
                pred = pred.squeeze(0).detach()
                gt_valid = gt_valid.squeeze(0).detach()

                valid_img = functional.to_pil_image(valid)
                pred_img = functional.to_pil_image(pred)
                gt_valid_img = functional.to_pil_image(gt_valid)

                wandb_images = [wandb.Image(valid_img, caption='Sample'),
                                wandb.Image(pred_img, caption='Predicted Sample'),
                                wandb.Image(gt_valid_img, caption='Ground Truth Sample')]

                logs = {
                    "Images": wandb_images
                }
                self.wandb_log.on_log(logs)

                # Store image
                path = folder + item['image_name'][0]
                pred_img.save(path)

        avg_psnr = total_psnr / len(self.valid_data_loader)
        avg_loss = valid_loss / len(self.valid_data_loader)

        self.logger.info(f"Total Loss: {valid_loss} - Total PSNR: {total_psnr}")
        self.logger.info(f"Valid Average Loss: {avg_loss} - Valid average PSNR: {avg_psnr}")

        if self.wandb_log:  # Logs Valid Parameters
            logs = {
                'valid_avg_loss': avg_loss,
                'valid_avg_psnr': avg_psnr,
            }
            self.wandb_log.on_log(logs)

        return avg_psnr, avg_loss

    def _make_datasets(self):
        train_transform = create_train_transform(patch_size=self.train_split_size)
        self.logger.info(
            f"Train path: \"{self.train_data_path}\" - Train ground truth path: \"{self.train_gt_data_path}\"")
        self.train_dataset = HandwrittenTextImageDataset(self.train_data_path, self.train_gt_data_path,
                                                         transform=train_transform)

        # Validation
        valid_transform = create_valid_transform()
        self.logger.info(
            f"Validation path: \"{self.valid_data_path}\" - Validation ground truth path: \"{self.valid_gt_data_path}\"")
        self.valid_dataset = ValidationDataLoader(self.valid_data_path, self.valid_gt_data_path,
                                                  transform=valid_transform)

    def _make_dataloaders(self):
        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                                                             shuffle=True, num_workers=self.workers, pin_memory=True)
        self.valid_data_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.valid_batch_size,
                                                             num_workers=self.workers, shuffle=False, pin_memory=True)

        self.logger.info(f"Training set has {len(self.train_dataset)} instances")
        self.logger.info(f"Validation set has {len(self.valid_dataset)} instances")

    def _create_optimizer(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.95),
                                     eps=1e-08, weight_decay=0.05, amsgrad=False)
