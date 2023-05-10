import torch
from ignite.engine import Engine
from ignite.metrics import PSNR, Precision, Recall

from data.dataloaders import make_valid_dataloader
from data.datasets import make_valid_dataset
from modules.FFC import LaMa
from utils.checkpoints import load_checkpoints
from utils.htr_logging import get_logger


def eval_step(engine, batch, threshold=0.5):
    inputs, targets = batch
    inputs = 1.0 - torch.where(inputs > threshold, 1., 0.)
    targets = 1.0 - torch.where(targets > threshold, 1., 0.)

    return inputs, targets


class Validator:
    def __init__(self, config, network_cfg: dict, device=None):
        self._logger = get_logger(Validator.__name__)

        self._count = 0
        self._device = device
        self._checkpoints_path = f"{config['path_checkpoint']}{config['filename_checkpoint']}"

        self._evaluator = Engine(eval_step)

        self._psnr = PSNR(data_range=1.0)
        self._psnr.attach(self._evaluator, 'psnr')
        self._psnr_value = 0.0

        self._precision = Precision()
        self._precision.attach(self._evaluator, 'precision')
        self._precision_value = 0.0

        self._recall = Recall()
        self._recall.attach(self._evaluator, 'recall')
        self._recall_value = 0.0

        self.valid_dataset = make_valid_dataset(config)
        self.valid_data_loader = make_valid_dataloader(self.valid_dataset, config)

        self.model = LaMa(input_nc=network_cfg['input_channels'],
                          output_nc=network_cfg['output_channels'],
                          init_conv_kwargs=network_cfg['init_conv_kwargs'],
                          downsample_conv_kwargs=network_cfg['down_sample_conv_kwargs'],
                          resnet_conv_kwargs=network_cfg['resnet_conv_kwargs'])
        self.model.eval()

        self.model.to(device=device)
        load_checkpoints(model=self.model, device=self._device, checkpoints_path=self._checkpoints_path)

    def compute(self, predicts: torch.Tensor, targets: torch.Tensor):
        state = self._evaluator.run([[predicts, targets]])

        self._count += len(predicts)
        self._psnr_value += state.metrics['psnr']
        self._precision_value += state.metrics['precision']
        self._recall_value += state.metrics['recall']

        avg_psnr = state.metrics['psnr'] / len(predicts)
        avg_precision = 100. * state.metrics['precision'] / len(predicts)
        avg_recall = 100. * state.metrics['recall'] / len(predicts)

        return avg_psnr, avg_precision, avg_recall

    def get_metrics(self):
        psnr = self._psnr_value / self._count
        precision = 100. * self._precision_value / self._count
        recall = 100. * self._recall_value / self._count
        return psnr, precision, recall

    def reset(self):
        self._count = 0.0
        self._psnr_value = 0.0
        self._precision_value = 0.0
        self._recall_value = 0.0
