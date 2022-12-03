import torch


class LMSELoss(torch.nn.MSELoss):
    def forward(self, inputs, targets):
        mse = super().forward(inputs, targets)
        mse = torch.add(mse, 1e-10)
        return torch.log10(mse)
