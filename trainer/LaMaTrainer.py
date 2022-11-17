import os

import torch
import torch.optim as optim
import torch.utils.data
from typing_extensions import TypedDict
import wandb

from dataloader import HandwrittenTextImageDataset
from modules.FFC import LaMa


class LaMaTrainingModule:

    def __init__(self, train_data_path: str, valid_data_path: str, input_channels: int, output_channels: int,
                 batch_size: int, epochs: int):
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.batch_size = batch_size
        self.train_data_loader = None
        self.valid_data_loader = None
        self.epochs = epochs

        self._get_dataloader()

        # TO IMPROVE
        Arguments = TypedDict('Arguments', {'ratio_gin': float, 'ratio_gout': float})  # REMOVE
        init_conv_kwargs: Arguments = {'ratio_gin': 0, 'ratio_gout': 0}  # REMOVE
        down_sample_conv_kwargs: Arguments = {'ratio_gin': 0, 'ratio_gout': 0}  # REMOVE
        resnet_conv_kwargs: Arguments = {'ratio_gin': 0.75, 'ratio_gout': 0.75}  # REMOVE
        self.model = LaMa(input_nc=input_channels, output_nc=output_channels,
                          init_conv_kwargs=init_conv_kwargs, downsample_conv_kwargs=down_sample_conv_kwargs,
                          resnet_conv_kwargs=resnet_conv_kwargs)

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)

        # Training
        self.epoch = 1

        # Configuration Weights & Bias
        wandb.init(project="test-project", entity="fomo_thesis", name="test_train_lama")
        wandb.config = {
            "learning_rage": 1.5e-4,
            "epochs": self.epochs,
            "batch_size": self.batch_size
        }
        wandb.watch(self.model, log="all")
        self.wandb = wandb

    def resume(self, folder):
        checkpoint = torch.load(folder + "last_train.pth", map_location=None)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.epoch = checkpoint['epoch']

    def save_checkpoint(self, folder: str):
        path = folder + "last_train.pth"
        os.makedirs(folder, exist_ok=True)
        checkpoint = {
            'model': self.model.state_dict(),
            'epoch': self.epoch
        }
        torch.save(checkpoint, path)

    def _get_dataloader(self):
        train_dataset = HandwrittenTextImageDataset(self.train_data_path, self.train_data_path + '_gt')
        self.train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                             num_workers=0,
                                                             pin_memory=True)
        valid_dataset = HandwrittenTextImageDataset(self.valid_data_path, self.valid_data_path + '_gt')
        self.valid_data_loader = torch.utils.data.DataLoader(valid_dataset)
