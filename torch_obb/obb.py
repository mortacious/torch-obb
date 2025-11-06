from __future__ import annotations
import torch
from .util import check_batch_dim, prepare_vertices

_VERTICES = torch.tensor([[0, 0, 0],
                          [0, 0, 1],
                          [0, 1, 0],
                          [0, 1, 1],
                          [1, 0, 0],
                          [1, 0, 1],
                          [1, 1, 0],
                          [1, 1, 1]], 
                          dtype=torch.int32, 
                          device=torch.device('cpu'))

EPSILON = 1e-6

def ensure_obb_shape(obbs: torch.Tensor) -> torch.Tensor:
    """
    Ensures the shape of the OBBs is valid.
    """
    obbs = check_batch_dim(obbs, 3)
    if obbs.shape[1] != 5 or obbs.shape[2] != 3:
        raise ValueError(f"OBBs must be of shape (N, 5, 3), got {obbs.shape}")
    return obbs


def obb_vertices(obbs: torch.Tensor) -> torch.Tensor:
    """
    Computes the 8 corner vertices of a batch of oriented bounding boxes.

    Parameters
    ----------
    obbs : torch.Tensor
        The OBBs to compute the vertices for. Must be of shape (N, 5, 3).

    Returns
    -------
    vertices : torch.Tensor
        The 8 corner vertices of each OBB in the batch. Shape (N, 8, 3).
    """
    obbs = ensure_obb_shape(obbs)
    device = obbs.device

    R_t = obbs[:, 0:3]
    centers_t = obbs[:, 3]
    half_extents_t = obbs[:, 4]

    vertices_t = _VERTICES.to(device, copy=True).repeat(obbs.shape[0], 1, 1)
    vertices_t = (vertices_t - 0.5) * 2.0
    vertices_t = vertices_t * half_extents_t.unsqueeze(1) # apply the extents in local coordinates
    # transform vertices from local to world space
    vertices_t = centers_t.unsqueeze(1) + torch.bmm(vertices_t, R_t) # apply the rotation and translation
    return vertices_t.reshape(-1, 8, 3)


def obb_points_intersection(obbs: torch.Tensor, points: torch.Tensor, pairwise: bool = False, counts: bool = False) -> torch.Tensor:
    """
    Checks if a batch of points are inside a batch of oriented bounding boxes.

    Parameters
    ----------
    obbs : torch.Tensor
        The OBBs to check the points against. Must be of shape (N, 5, 3).
    points : torch.Tensor
        The points to check. Must be of shape (N, 3).

    Returns
    -------
    inside_mask : torch.Tensor
        A boolean mask of shape (N,) indicating which points are inside the OBBs.
    """

    # TODO: support pairwise mode like in the obb intersection functions?
    obbs = ensure_obb_shape(obbs)
    points = prepare_vertices(points)

    R_t = obbs[:, 0:3]
    centers_t = obbs[:, 3]
    half_extents_t = obbs[:, 4] + EPSILON

    B_points = points.shape[0]
    B_obbs = obbs.shape[0]

    if pairwise:
        if B_points != B_obbs:
            raise ValueError(
                f"pairwise=True requires the same batch size for points and obbs "
                f"(got B_points={B_points}, B_obbs={B_obbs})."
            )
        points_centered_t = points - centers_t.unsqueeze(1)
        # (B, P, 3) @ (B, 3, 3)^T -> (B, P, 3)
        projections_t = torch.abs(torch.bmm(points_centered_t, R_t.mT))
        inside_mask_t = (projections_t <= half_extents_t.unsqueeze(1)).all(dim=-1)  # (B, P)

        if counts:
            return inside_mask_t.sum(dim=-1)  # (B,)
        else:
            return inside_mask_t
    else:

        # -------- Full cross mode (each cloud vs each OBB) --------
        # Center points for every OBB
        points_centered = points.values().unsqueeze(0) - centers_t.unsqueeze(1)

        # Project into each OBB local basis:
        # points_centered: (B_points, B_obbs, P, 3)
        # R_t.mT        : (B_obbs, 3, 3)
        # Result        : (B_points, B_obbs, P, 3)
        projections = torch.abs(torch.bmm(points_centered, R_t.mT))
        # Check against half-extents: (1, B_obbs, 1, 3)
        inside_mask = (projections <= half_extents_t.unsqueeze(1)).all(dim=-1)
        inside_mask = torch.nested.nested_tensor_from_jagged(inside_mask,
                                                             offsets=points.offsets(),
                                                             jagged_dim=2)
        if counts:
            # Sum across the points dimension (jagged) for each (point_cloud, obb) pair. 
            # This is a workaround to avoid the nested tensor sum operation as it leads to wrong results

            # Get the flat values and offsets
            mask_values = inside_mask.values()  # Shape: (B_obbs, P_total)
            offsets = inside_mask.offsets()     # Shape: (B_points + 1,)
            P_total = mask_values.shape[1]
            B_points = len(offsets) - 1

            # Create cloud indices for each point position (vectorized)
            lengths = offsets.diff()
            cloud_indices = torch.repeat_interleave(torch.arange(B_points, device=mask_values.device), lengths)

            cloud_mask = torch.zeros(B_points, P_total, dtype=torch.bool, device=mask_values.device)
            cloud_mask[cloud_indices, torch.arange(P_total, device=mask_values.device)] = True

            # Vectorized computation: (B_obbs, B_points, P_total) -> sum over points -> (B_obbs, B_points) -> transpose
            return (mask_values.unsqueeze(1) * cloud_mask.unsqueeze(0)).sum(dim=-1).t()
        else:

            return inside_mask
