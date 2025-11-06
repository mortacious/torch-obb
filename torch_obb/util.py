from __future__ import annotations
import torch
from .typing import Tensor, Optional


# helper functions
def check_batch_dim(tensor: Tensor, ndim: int) -> Tensor:
    if tensor.ndim != ndim:
        tensor = tensor[None]
    return tensor


def prepare_vertices(vertices: Tensor, batch_offsets: Optional[Tensor] = None, device: Optional[str] = None) -> Tensor:
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.from_numpy(vertices)
    if device is None:
        device = vertices.device
    vertices = vertices.to(device)

    if not vertices.is_nested:
        if batch_offsets is None:
            vertices = torch.nested.as_nested_tensor([vertices], layout=torch.jagged)
        else:
            if batch_offsets is not None and not isinstance(batch_offsets, torch.Tensor):
                batch_offsets = torch.from_numpy(batch_offsets).to(device)
            vertices = torch.nested.nested_tensor_from_jagged(vertices, offsets=batch_offsets, jagged_dim=1)
    return vertices