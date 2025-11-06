from __future__ import annotations
import torch
from .typing import Tensor, Optional
from .util import prepare_vertices


def obb_estimate_pca(
    vertices: Tensor,
    batch_offsets: Optional[Tensor] = None,
    device: Optional[str] = None,
) -> Tensor:
    from .kernels.estimation_pca import obb_estimate_pca as obb_estimate_pca_impl
    # pca implementation is done using pure torch operations as torch has support for nested tensors builtin
    vertices_t = prepare_vertices(vertices, batch_offsets, device)
    return obb_estimate_pca_impl(vertices_t)


def obb_estimate_dito(vertices: Tensor, 
                      batch_offsets: Optional[Tensor] = None,
                      device: Optional[str] = None) -> Tensor:
    from .kernels.estimation_dito import obb_estimate_dito as obb_estimate_dito_impl
    vertices_t = prepare_vertices(vertices, batch_offsets, device)
    return obb_estimate_dito_impl(vertices_t)


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
    vertices_t = prepare_vertices(vertices, batch_offsets, device)

    if method == "pca":
        obbs_t = obb_estimate_pca(vertices_t)
    elif method == "dito":    
        obbs_t = obb_estimate_dito(vertices_t)
    else:
        raise ValueError(f"Invalid method: {method}")
        
    if not isinstance(vertices, torch.Tensor):
        obbs_t = obbs_t.cpu().numpy()

    return obbs_t



