from __future__ import annotations
import numpy as np
try:
    import torch
except ImportError:
    torch = None

from .typing import Tensor, Optional
from .util import ensure_torch_available


AABB_VERTICES = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 1, 0],
                          [0, 1, 1],
                          [1, 0, 0],
                          [1, 0, 1],
                          [1, 1, 0],
                          [1, 1, 1]], dtype=np.int32)

def obb_estimate_pca(
    vertices: Tensor,
    batch_offsets: Optional[Tensor] = None,
    device: Optional[str] = None,
) -> Tensor:
    # pca implementation is done using pure torch operations as torch has support for nested tensors builtin
    ensure_torch_available()
    if not isinstance(vertices, torch.Tensor):
        vertices_t = torch.from_numpy(vertices)
    if device is None:
        device = vertices_t.device
    vertices_t = vertices_t.to(device)

    if not vertices_t.is_nested:
        if batch_offsets is not None and not isinstance(batch_offsets, torch.Tensor):
            batch_offsets = torch.from_numpy(batch_offsets).to(device)
        vertices_t = torch.nested.nested_tensor_from_jagged(vertices_t, offsets=batch_offsets, jagged_dim=1)

    centroids_t = torch.mean(vertices_t, dim=1)
    centered_t = vertices_t - centroids_t

    npoints_t = vertices_t.offsets().diff()

    cov_t = torch.bmm(centered_t.mT, centered_t) / npoints_t

    _, R_t = torch.linalg.eigh(cov_t)
    rotated_t_jagged = torch.bmm(centered_t, R_t)
    min_extent_t = rotated_t_jagged.min(dim=1).values
    max_extent_t = rotated_t_jagged.max(dim=1).values
    extent_t = max_extent_t - min_extent_t

    vertices_t = torch.from_numpy(AABB_VERTICES).to(device).unsqueeze(0)
    vertices_t = vertices_t - 0.5
    vertices_t = vertices_t * extent_t
    vertices_t = centroids_t + torch.bmm(vertices_t, R_t.mT)
    
    if not isinstance(vertices, torch.Tensor):
        return vertices_t.cpu().numpy()
    return vertices_t


def obb_estimate(
    vertices: Tensor,
    batch_offsets: Optional[Tensor] = None,
    device: Optional[str] = None,
    method: str = "pca",
) -> Tensor:
    """Compute oriented bounding box using the DITO-14 algorithm on GPU/CPU via Warp.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        Input vertices to compute OBB for.
    device : str or Device, optional
        The device to use for computation. If not provided, will be inferred from input.

    Returns
    -------
    obb_vertices : array-like, shape (8, 3)
        The 8 corner vertices of the computed oriented bounding box.
    """

    if method == "pca":
        vertices = obb_estimate_pca(vertices, batch_offsets, device)
    # elif method == "dito":
    #     return obb_estimate_dito(vertices, batch_offsets, batch_counts, device)
    else:
        raise ValueError(f"Invalid method: {method}")
    return vertices



