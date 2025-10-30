from __future__ import annotations
from .typing import Tensor, Optional
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
import numpy as np
import warp as wp

# helper functions
def check_batch_dim(tensor: Tensor, ndim: int) -> Tensor:
    if tensor.ndim != ndim:
        tensor = tensor[None]
    return tensor


def ensure_warp_available() -> None:
    """Ensure warp is available and raises if not."""
    if wp is None:  # type: ignore
        raise RuntimeError("warp import failed")
    wp.config.quiet = True
    wp.init()

def ensure_torch_available() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for this implementation")

def infer_device(*tensors: Tensor) -> str | wp.context.Device:
    """Infer the device from a bunch of tensors."""
    if torch is not None:
        for i in tensors:
            if isinstance(i, torch.Tensor):
                device = wp.device_from_torch(i.device)
                if device.is_cuda:
                    return device
        return 'cpu'
    return "cpu"


def to_wp_array(x: Tensor, dtype: wp.types.DType, device: str = "cpu") -> wp.array:
    """Convert a numpy array or torch tensor to a warp array of specific dtype and device."""
    if torch is not None and isinstance(x, torch.Tensor):
        return wp.from_torch(x, dtype=dtype)
    else:
        # fallback to numpy
        arr = np.asarray(x)
        return wp.from_numpy(arr, dtype=dtype, device=device)


def from_wp_array(x: wp.array, like: Optional[Tensor] = None) -> Tensor:
    """Convert a warp array back into a numpy array or torch tensor if specified."""
    if like is not None and torch is not None and isinstance(like, torch.Tensor):
        return wp.to_torch(x).to(like.device)
    return x.numpy()