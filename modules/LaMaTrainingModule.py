import os.path
from inspect import Arguments

import torch
import torch.nn.functional as F
import torch.optim as optim

import config
from dataloader import HandwrittenTextImageDataset
from modules.FFC import LaMa


def get_dataloader(dir_original: str, dir_ground_truth: str):
    batch_size = config.batch_size
    train_dataset = HandwrittenTextImageDataset(dir_original, dir_ground_truth, augmentation=True)
    return torch.utils.data.DataLoader(train_dataset, collate_fn=train_dataset.collate_fn,
                                       batch_size=batch_size, shuffle=True, num_workers=0,
                                       pin_memory=True)


class LaMaTrainingModule:
    def __init__(self, input_channels: int, output_channels: int, device: torch.device, init_conv_kwargs: Arguments,
                 down_sample_conv_kwargs: Arguments, resnet_conv_kwargs: Arguments):
        super(LaMaTrainingModule, self).__init__()

        self.model = LaMa(input_nc=input_channels, output_nc=output_channels, init_conv_kwargs=init_conv_kwargs,
                          downsample_conv_kwargs=down_sample_conv_kwargs, resnet_conv_kwargs=resnet_conv_kwargs)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1.5e-4, betas=(0.9, 0.95), eps=1e-08,
                                     weight_decay=0.05, amsgrad=False)
        self.device = device
        self.train_loader = get_dataloader('../patches/train', '../patches/train_gt')

    def train(self):
        running_loss = 0.0

        for i, (train_index, train_in, train_out) in enumerate(self.train_loader):
            inputs = train_in.to(self.device)
            outputs = train_out.to(self.device)

            self.optimizer.zero_grad()

            pred_img = self.model(inputs)
            loss = F.mse_loss(pred_img, outputs[:, :1, :, :])  # TO-DO : TO ASK

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            print(running_loss)

        if not os.path.exists('../weights/'):
            os.makedirs('../weights/')
        torch.save(self.model.state_dict(), '../weights/last_training.pt')


if __name__ == '__main__':
    init_conv_kwargs: Arguments = {'ratio_gin': 0, 'ratio_gout': 0}
    down_sample_conv_kwargs: Arguments = {'ratio_gin': 0, 'ratio_gout': 0}
    resnet_conv_kwargs: Arguments = {'ratio_gin': 0.75, 'ratio_gout': 0.75}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = LaMaTrainingModule(input_channels=3, output_channels=1, device=device,
                                 init_conv_kwargs=init_conv_kwargs,
                                 down_sample_conv_kwargs=down_sample_conv_kwargs, resnet_conv_kwargs=resnet_conv_kwargs)
    trainer.train()
