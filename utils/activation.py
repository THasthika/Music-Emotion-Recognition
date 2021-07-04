import torch.nn as nn
from torch import Tensor

class CustomELU(nn.ELU):

    def __init__(self, alpha=1, inplace=False) -> None:
        super().__init__(alpha=alpha, inplace=inplace)
    
    def forward(self, input: Tensor) -> Tensor:
        return 1 + super().forward(input - 1)