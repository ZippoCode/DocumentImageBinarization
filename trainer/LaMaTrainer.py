import math
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import wandb
from torchvision import transforms
from typing_extensions import TypedDict

from dataloader import HandwrittenTextImageDataset
from modules.FFC import LaMa


class LaMaTrainingModule:

    def __init__(self, train_data_path: str, valid_data_path: str, input_channels: int, output_channels: int,
                 train_batch_size: int, valid_batch_size: int, train_split_size: int, valid_split_size: int,
                 epochs: int, workers: int, device=None, debug=False):
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.train_split_size = train_split_size
        self.valid_split_size = valid_split_size

        self.train_data_loader = None
        self.valid_data_loader = None
        self.epochs = epochs
        self.device = device
        self.workers = workers

        self.debug = debug

        self._get_dataloader()

        # TO IMPROVE
        Arguments = TypedDict('Arguments', {'ratio_gin': float, 'ratio_gout': float})  # REMOVE
        init_conv_kwargs: Arguments = {'ratio_gin': 0, 'ratio_gout': 0}  # REMOVE
        down_sample_conv_kwargs: Arguments = {'ratio_gin': 0, 'ratio_gout': 0}  # REMOVE
        resnet_conv_kwargs: Arguments = {'ratio_gin': 0.75, 'ratio_gout': 0.75}  # REMOVE
        self.model = LaMa(input_nc=input_channels, output_nc=output_channels,
                          init_conv_kwargs=init_conv_kwargs, downsample_conv_kwargs=down_sample_conv_kwargs,
                          resnet_conv_kwargs=resnet_conv_kwargs)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)

        # Training
        self.epoch = 1

        # Criterion
        self.criterion = torch.nn.MSELoss().to(self.device)

        # Configuration Weights & Bias
        if not debug:
            wandb.init(project="test-project", entity="fomo_thesis", name="test_train_lama")
            wandb.config = {
                "learning_rage": 1.5e-4,
                "epochs": self.epochs,
                "batch_size": self.train_batch_size
            }
            wandb.watch(self.model, log="all")
            self.wandb = wandb

        # Validation
        self.best_epoch = 0
        self.best_psnr = 0

    def resume(self, folder):
        checkpoint = torch.load(folder + "last_train.pth", map_location=None)
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
        torch.save(checkpoint, path)

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

        avg_psnr = total_psnr / len(self.valid_data_loader)
        avg_loss = valid_loss / len(self.valid_data_loader)

        return avg_psnr, avg_loss

    def _get_dataloader(self):
        train_dataset = HandwrittenTextImageDataset(self.train_data_path, self.train_data_path + '_gt')
        self.train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.train_batch_size,
                                                             shuffle=True, num_workers=self.workers, pin_memory=True)
        transform = transforms.Compose([transforms.CenterCrop(size=self.valid_split_size), transforms.ToTensor()])
        valid_dataset = HandwrittenTextImageDataset(self.valid_data_path, self.valid_data_path + '_gt',
                                                    transform=transform)
        self.valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.valid_batch_size,
                                                             num_workers=self.workers, pin_memory=True)
