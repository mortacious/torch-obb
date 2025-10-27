# torch-obb

A high-performance PyTorch package for oriented bounding box (OBB) intersection operations, powered by [Warp](https://github.com/NVIDIA/warp).

## Features

- **High Performance**: CUDA-accelerated OBB intersection computations using Warp
- **Fallback Implementation**: NumPy/Numba fallback for CPU-only environments
- **PyTorch Integration**: Native PyTorch tensor support with automatic device inference
- **Batch Processing**: Efficient batch processing of multiple OBB pairs
- **Multiple Operations**:
  - Intersection volumes computation
  - Directional overlap ratios
  - Intersection over Union (IoU) calculation

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (for best performance)
- Required dependencies: `numpy`, `warp-lang`, `numba`
- Optional: `torch` (for PyTorch tensor support)

### Install from PyPI

```bash
pip install torch-obb
```

### Install from source

```bash
git clone https://github.com/yourusername/torch-obb.git
cd torch-obb
pip install .
```

### Development installation

```bash
pip install -e ".[dev]"
```

## Quick Start

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

## API Reference

### Main Functions

#### `obb_intersection_volumes(obb_first, obb_second, pairwise=False, device=None)`

Compute volumes and intersection volumes between two sets of oriented bounding boxes.

**Parameters:**
- `obb_first`: Array of OBBs with shape `(N, 8, 3)` or `(8, 3)`
- `obb_second`: Array of OBBs with shape `(M, 8, 3)` or `(8, 3)`
- `pairwise`: If `True`, compute pairwise intersections (N×M)
- `device`: Target device for computation (auto-inferred if using PyTorch tensors)

**Returns:**
- `volumes1`: Volumes of first set of OBBs
- `volumes2`: Volumes of second set of OBBs
- `intersection_volumes`: Intersection volumes between OBB pairs

#### `obb_overlaps(obb_first, obb_second, pairwise=False, device=None)`

Compute directional overlap ratios between oriented bounding boxes.

**Parameters:**
- `obb_first`: Array of OBBs with shape `(N, 8, 3)` or `(8, 3)`
- `obb_second`: Array of OBBs with shape `(M, 8, 3)` or `(8, 3)`
- `pairwise`: If `True`, compute pairwise overlaps (N×M)
- `device`: Target device for computation (auto-inferred if using PyTorch tensors)

**Returns:**
- `overlap1`: Overlap ratios of first OBBs over second OBBs
- `overlap2`: Overlap ratios of second OBBs over first OBBs

#### `obb_ious(obb_first, obb_second, pairwise=False, device=None)`

Compute Intersection over Union (IoU) between oriented bounding boxes.

**Parameters:**
- `obb_first`: Array of OBBs with shape `(N, 8, 3)` or `(8, 3)`
- `obb_second`: Array of OBBs with shape `(M, 8, 3)` or `(8, 3)`
- `pairwise`: If `True`, compute pairwise IoUs (N×M)
- `device`: Target device for computation (auto-inferred if using PyTorch tensors)

**Returns:**
- `iou`: IoU values between OBB pairs

## Input Format

OBB vertices should be provided as arrays of shape `(N, 8, 3)` or `(8, 3)` where:
- `N`: Number of oriented bounding boxes
- `8`: Number of corner vertices per box
- `3`: XYZ coordinates

The vertices should follow this specific order:
```
[0,1,3,2], [0,4,5,1], [0,2,6,4], [1,5,7,3], [2,3,7,6], [4,6,7,5]
```

Use the provided `make_obb()` helper function from the test suite to create properly formatted OBBs.

## Performance Notes

- **Warp Implementation**: Uses CUDA kernels for maximum performance on GPU
- **Fallback Implementation**: NumPy/Numba implementation for CPU-only environments
- **Automatic Selection**: The package automatically uses the most appropriate implementation
- **Batch Processing**: All functions support efficient batch processing of multiple OBB pairs

## Requirements and Compatibility

- **CUDA**: Required for Warp-based GPU acceleration
- **PyTorch**: Optional, enables seamless tensor integration
- **Python**: 3.8+ supported
- **NumPy**: 1.20+ required
- **Warp**: 0.10+ required

## Contributing

Contributions are welcome! Please see the [GitHub repository](https://github.com/yourusername/torch-obb) for:

- Bug reports and feature requests
- Pull request guidelines
- Development setup instructions

## License

MIT License - see LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{torch_obb,
  title={torch-obb: High-Performance OBB Intersection for PyTorch},
  author={torch-obb contributors},
  url={https://github.com/yourusername/torch-obb},
  version={0.1.0}
}
```
