import torch
import torch.nn as nn

class Unsqueeze(nn.Module):

    def __init__(self, dim: int):
        super(Unsqueeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.unsqueeze(x, self.dim)


class Squeeze(nn.Module):

    def __init__(self, dim: int):
        super(Squeeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.squeeze(x, self.dim)
