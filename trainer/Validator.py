import torch
from ignite.engine import Engine
from ignite.metrics import PSNR, Precision, Recall
from torch.utils.data import DataLoader
from torchvision.transforms import functional

from data.utils import reconstruct_ground_truth
from utils.htr_logging import get_logger
from utils.metrics import Metrics


def eval_step(engine, batch, threshold=0.5):
    inputs, targets = batch
    inputs = 1.0 - torch.where(inputs > threshold, 1., 0.)
    targets = 1.0 - torch.where(targets > threshold, 1., 0.)

    return inputs, targets


class Validator:
    def __init__(self, model: torch.nn.Module, device: torch.device, threshold=.5):
        self._logger = get_logger(Validator.__name__)
        self._evaluator = Engine(eval_step)

        self._psnr = PSNR(data_range=1.0)
        self._psnr.attach(self._evaluator, 'psnr')

        self._precision = Precision()
        self._precision.attach(self._evaluator, 'precision')

        self._recall = Recall()
        self._recall.attach(self._evaluator, 'recall')

        self._model = model
        self._device = device
        self._threshold = threshold
        self._metrics = Metrics()

        model.to(device=device)

    def compute(self, data_loader: DataLoader, output_channels: int, batch_size: int, patch_size: int,
                stride_size: int):
        self._logger.info("Start validation ...")

        if self._model.training:
            self._model.eval()
            self._logger.info("Setting model in evaluation state.")

        names = []
        images = []

        with torch.no_grad():
            for index, item in enumerate(data_loader):
                names.append(item['image_name'][0])
                num_rows = item['num_rows'].item()
                samples_patches = item['samples_patches']
                targets = item['gt_sample']

                samples_patches = samples_patches.squeeze(0)
                valid = samples_patches.to(self._device)
                targets = targets.to(self._device)

                valid = valid.squeeze(0).permute(1, 0, 2, 3)
                predicts = self._model(valid)
                predicts = reconstruct_ground_truth(predicts, targets, num_rows=num_rows, channels=output_channels,
                                                    batch=batch_size, patch_size=patch_size, stride=stride_size)
                predicts = torch.where(predicts > self._threshold, 1., 0.)

                state = self._evaluator.run([[predicts, targets]])

                self._metrics.update_psnr(state.metrics['psnr'])
                self._metrics.update_precision(state.metrics['precision'])
                self._metrics.update_recall(state.metrics['recall'])

                predicted_image = functional.to_pil_image(predicts.squeeze(0).detach())
                images.append(predicted_image)

                self._logger.info(f"Elaborated image number: {index}")

        self._metrics.set_average_metrics(len(data_loader))

        return self._metrics, names, images
