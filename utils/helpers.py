import torch
from torch import Tensor


def magic_combine(x: Tensor, dim_begin: int, dim_end: int):
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.view(combined_shape)