import torch
from typing import Optional
AABB_VERTICES = torch.tensor([[0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0],
                              [0, 1, 1],
                              [1, 0, 0],
                              [1, 0, 1],
                              [1, 1, 0],
                              [1, 1, 1]], dtype=torch.int32, device=torch.device('cpu'))

def compute_obb_vertices(centroids: torch.Tensor,
                         extents_local: torch.Tensor, 
                         R: torch.Tensor) -> torch.Tensor:
    device = centroids.device
    vertices_t = AABB_VERTICES.to(device, copy=True).repeat(centroids.shape[0], 1, 1)
    vertices_t = vertices_t - 0.5
    vertices_t = vertices_t * extents_local.unsqueeze(1) # apply the extents in local coordinates
    vertices_t = centroids.unsqueeze(1) + torch.bmm(vertices_t, R.mT) # apply the rotation and translation
    return vertices_t.reshape(-1, 8, 3)