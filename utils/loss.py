from os import extsep
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def rmse_loss(input: Tensor, target: Tensor, eps=1e-6, **mse_kwargs):
    mse = F.mse_loss(input, target, **mse_kwargs)
    return torch.sqrt(mse + eps)

class RMSELoss(torch.nn.Module):

    def __init__(self, eps=1e-6, **mse_kwargs):
        super(RMSELoss, self).__init__()
        self.eps = eps
        self.mse = nn.MSELoss(mse_kwargs)

    def forward(self, input: Tensor, target: Tensor):
        loss = self.mse(input, target)
        return torch.sqrt(loss + self.eps)