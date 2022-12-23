import torch


def make_criterion(kind: str):
    if kind == 'mean_square_error':
        criterion = torch.nn.MSELoss()
    elif kind == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif kind == 'negative_log_likelihood':
        criterion = torch.nn.NLLLoss()
    elif kind == 'binary_cross_entropy':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = LMSELoss()
    return criterion


class LMSELoss(torch.nn.MSELoss):
    def forward(self, inputs, targets):
        mse = super().forward(inputs, targets)
        mse = torch.add(mse, 1e-10)
        return torch.log10(mse)
