import torch
from ..typing import Tensor
EPSILON = 1e-9

def obb_estimate_pca(
    vertices_t: Tensor,
) -> Tensor:
    # pca implementation is done using pure torch operations as torch has support for nested tensors builtin
    device = vertices_t.device

    centroids_t = torch.mean(vertices_t, dim=1)
    centered_t = vertices_t - centroids_t.unsqueeze(1)

    npoints_t = vertices_t.offsets().diff()

    cov_t = torch.bmm(centered_t.mT, centered_t) / npoints_t[:, None, None]
    cov_t += EPSILON * torch.eye(3, device=device, dtype=cov_t.dtype).unsqueeze(0)

    evals_t, evecs_t = torch.linalg.eigh(cov_t)

    # sort the eigenvectors by the eigenvalues in descending order
    _, sort_indices_t = torch.sort(evals_t, dim=1, descending=True)
    evecs_t = torch.gather(evecs_t, 2, sort_indices_t.unsqueeze(1).expand(evecs_t.shape[0], 3, 3))
    R_t = evecs_t  # columns are OBB axes in world coordinates

    # Ensure right-handed-ness of the basis
    # If cross(v0, v1) not aligned with v2, flip v2
    # (det(R) should be +1)
    # Compute det and flip 3rd column if needed
    det = torch.det(R_t)
    neg_mask = det < 0
    if neg_mask.any():
        R_t[neg_mask, :, 2] *= -1.0

    rotated_t_jagged = torch.bmm(centered_t, R_t)
    min_extent_t = rotated_t_jagged.min(dim=1).values
    max_extent_t = rotated_t_jagged.max(dim=1).values

    half_extent_local_t = (max_extent_t - min_extent_t) * 0.5 # half extents

    # Center in local coords, then map to world
    centroids_local_t = (min_extent_t + max_extent_t) * 0.5
    centroids_t = centroids_t.unsqueeze(1) + torch.bmm(centroids_local_t.unsqueeze(1), R_t.mT)

    obbs = torch.cat([R_t.mT, centroids_t, half_extent_local_t.unsqueeze(1)], dim=1)    
    return obbs