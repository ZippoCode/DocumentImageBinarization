import math

import numpy as np
import torch


class Metrics:
    def __init__(self):
        super().__init__()
        self.psnr = .0
        self.precision = .0
        self.recall = .0

    def update_psnr(self, value: float):
        assert value >= .0
        self.psnr += value

    def update_precision(self, value: float):
        assert value >= .0
        self.precision += value

    def update_recall(self, value: float):
        assert value >= .0
        self.recall += value

    def set_average_metrics(self, number: int):
        assert number > 0
        self.psnr = self.psnr / number
        self.precision = 100. * self.precision / number
        self.recall = 100. * self.recall / number


def calculate_psnr(predicted: torch.Tensor, ground_truth: torch.Tensor):
    pred_img = predicted.detach().cpu().numpy()
    gt_img = ground_truth.detach().cpu().numpy()

    mse = np.mean((pred_img - gt_img) ** 2)
    psnr = 100 if mse == 0 else (20 * math.log10(1.0 / math.sqrt(mse)))
    return psnr
