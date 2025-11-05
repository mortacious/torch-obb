# torch-obb

A PyTorch package for batched oriented bounding box (OBB) operations, powered by [Warp](https://github.com/NVIDIA/warp). 

## Features
Currently torch-obb provides the following features:

- **OBB Estimation**: Compute oriented bounding boxes from jagged batches of point clouds using PCA or DITO-14 algorithms
- **OBB Intersection**: Compute the intersection volumes as well as overlap percentages and IOUs for batches of OBBs
- **Warp Implementation**: Uses CUDA kernels for maximum performance on GPU
- **Automatic Selection**: The package automatically uses the most appropriate implementation depending on the input device of the provided tensors
- **Batch Processing**: All functions support efficient batch processing of multiple OBB or point cloud pairs

## Installation

### Requirements

- Python 3.9+
- CUDA-capable GPU (for best performance)
- Required dependencies: `numpy`, `torch`, `warp-lang`
- Optional: `pytest` (for development)

### Install from source

```bash
git clone https://github.com/your-username/torch-obb.git
cd torch-obb
pip install .
```

### Development installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### OBB Intersection

```python
import torch
import numpy as np
from torch_obb import obb_intersection_volumes, obb_overlaps, obb_ious

# Example OBB vertices (8 corners of each box)
# Shape: (N, 8, 3) where N is number of boxes
obb1 = torch.randn(10, 8, 3)  # 10 OBBs
obb2 = torch.randn(10, 8, 3)  # 10 OBBs

# Compute intersection volumes
volumes1, volumes2, intersection_volumes = obb_intersection_volumes(obb1, obb2)

# Compute overlap ratios
overlap1, overlap2 = obb_overlaps(obb1, obb2)

# Compute IoU
iou = obb_ious(obb1, obb2)

print(f"Intersection volumes: {intersection_volumes}")
print(f"IoU: {iou}")
```

## OBB Estimation

```python
import torch
import numpy as np
from torch_obb import obb_estimate, obb_estimate_pca, obb_estimate_dito

# Example point cloud (random 3D points forming an elongated shape)
np.random.seed(42)
points = np.random.randn(1000, 3).astype(np.float32)
points = points * np.array([2.0, 1.0, 0.5])  # Elongate in x-direction

# Convert to PyTorch nested tensor for batch processing
points_tensor = torch.nested.as_nested_tensor([torch.from_numpy(points)], layout=torch.jagged)

# Estimate OBB using PCA (default method)
obb_vertices = obb_estimate(points_tensor)
print(f"OBB vertices shape: {obb_vertices.shape}")  # (1, 8, 3)

# Get both vertices and rotation matrix using PCA
obb_vertices_pca, rotation_matrix = obb_estimate_pca(points_tensor)
print(f"Rotation matrix shape: {rotation_matrix.shape}")  # (1, 3, 3)

# Use DITO algorithm
obb_vertices_dito, rotation_matrix = obb_estimate_dito(points_tensor)

# Batch processing multiple point clouds
batch_points = torch.nested.as_nested_tensor([
    torch.from_numpy(points),
    torch.from_numpy(points * 0.5),  # Smaller version
    torch.from_numpy(points + np.array([5.0, 0.0, 0.0]))  # Offset version
], layout=torch.jagged)

batch_obbs = obb_estimate(batch_points)
print(f"Batch OBBs shape: {batch_obbs.shape}")  # (3, 8, 3)
```
## Input Format

OBB vertices should be provided as arrays of shape `(N, 8, 3)` or `(8, 3)` where:
- `N`: Number of oriented bounding boxes
- `8`: Corner vertices per box
- `3`: XYZ coordinates

The vertices should follow this specific order:
```
[0,1,3,2], [0,4,5,1], [0,2,6,4], [1,5,7,3], [2,3,7,6], [4,6,7,5]
```

## Requirements and Compatibility

- **CUDA**: Required for Warp-based GPU acceleration
- **PyTorch**: 2.0+ required for tensor operations
- **Python**: 3.9+ supported
- **Warp**: 1.10.0+ required

## License

MIT License - see LICENSE file for details.
