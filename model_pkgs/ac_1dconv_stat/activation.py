import torch.nn as nn
from torch import Tensor

class CustomELU(nn.ELU):

    def __init__(self, alpha: float, inplace: bool) -> None:
        super().__init__(alpha=alpha, inplace=inplace)
    
    def forward(self, input: Tensor) -> Tensor:
        return 1 + super().forward(input)