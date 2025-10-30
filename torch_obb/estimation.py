from __future__ import annotations
import torch

from .typing import Tensor, Optional
from .kernels.estimation_utils import compute_obb_vertices


def prepare_vertices(vertices: Tensor, batch_offsets: Optional[Tensor] = None, device: Optional[str] = None) -> Tensor:
    if not isinstance(vertices, torch.Tensor):  # type: ignore
        vertices = torch.from_numpy(vertices)
    if device is None:
        device = vertices.device
    vertices = vertices.to(device)

    if not vertices.is_nested:
        if batch_offsets is not None and not isinstance(batch_offsets, torch.Tensor):
            batch_offsets = torch.from_numpy(batch_offsets).to(device)
        vertices = torch.nested.nested_tensor_from_jagged(vertices, offsets=batch_offsets, jagged_dim=1)
    return vertices, device


def obb_estimate_pca(
    vertices: Tensor,
    batch_offsets: Optional[Tensor] = None,
    device: Optional[str] = None,
) -> Tensor:
    # pca implementation is done using pure torch operations as torch has support for nested tensors builtin
    vertices_t, device = prepare_vertices(vertices, batch_offsets, device)

    centroids_t = torch.mean(vertices_t, dim=1)
    centered_t = vertices_t - centroids_t.unsqueeze(1)

    npoints_t = vertices_t.offsets().diff()

    cov_t = torch.bmm(centered_t.mT, centered_t) / npoints_t[:, None, None]

    _, R_t = torch.linalg.eigh(cov_t)
    rotated_t_jagged = torch.bmm(centered_t, R_t)
    min_extent_t = rotated_t_jagged.min(dim=1).values
    max_extent_t = rotated_t_jagged.max(dim=1).values
    extent_t = max_extent_t - min_extent_t
    centroids_t = (min_extent_t + max_extent_t) * 0.5

    vertices_t = compute_obb_vertices(centroids_t, extent_t, R_t)
    
    return vertices_t


def obb_estimate_dito(vertices: Tensor, 
                      batch_offsets: Optional[Tensor] = None,
                      device: Optional[str] = None) -> Tensor:
    from .kernels.estimation_dito import obb_estimate_dito as obb_estimate_dito_impl
    vertices_t, device = prepare_vertices(vertices, batch_offsets, device)
    return obb_estimate_dito_impl(vertices_t, device)


def obb_estimate(
    vertices: Tensor,
    batch_offsets: Optional[Tensor] = None,
    device: Optional[str] = None,
    method: str = "dito",
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
    vertices_t, device = prepare_vertices(vertices, batch_offsets, device)

    if method == "pca":
        obb_vertices = obb_estimate_pca(vertices_t, device=device)
    elif method == "dito":    
        obb_vertices = obb_estimate_dito(vertices, device=device)
    else:
        raise ValueError(f"Invalid method: {method}")
        
    if not isinstance(vertices, torch.Tensor):
        obb_vertices = obb_vertices.cpu().numpy()
    return obb_vertices.reshape(-1, 8, 3)



