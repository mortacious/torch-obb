from __future__ import annotations
from .typing import Tensor

def check_batch_dim(tensor: Tensor, ndim: int) -> Tensor:
    if tensor.ndim != ndim:
        tensor = tensor[None]
    return tensor