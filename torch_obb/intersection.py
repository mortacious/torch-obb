from __future__ import annotations
import warp as wp
from .kernels.intersection import obb_intersection_volumes as _obb_intersection_volumes_impl
from .util import check_batch_dim, ensure_warp_available, infer_device, to_wp_array, from_wp_array
from .typing import Tensor, Tuple, Optional

# -----------------------------
# Public API 
# -----------------------------

def obb_intersection_volumes(
    obb_first: Tensor,
    obb_second: Tensor,
    pairwise: bool = False,
    device: Optional[str] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute volumes and intersection volumes between two sets of OBBs using Warp.

    Parameters
    ----------
    obb_first : array-like, shape (N, 8, 3)
        First set of OBBs.
    obb_second : array-like, shape (M, 8, 3)
        Second set of OBBs.
    pairwise : bool
        When True, requires N == M and computes pairwise intersections.
    device : str or Device
        The device to use for the computation. If not provided, will be inferred from the inputs.

    Returns
    -------
    vol1 : array-like, shape (N,)
    vol2 : array-like, shape (M,)
    inter : array-like, shape (N, M)
    """
    ensure_warp_available()
    # Ensure shape
    obb_first = check_batch_dim(obb_first, 3)
    obb_second = check_batch_dim(obb_second, 3)

    # Determine device from inputs if not provided
    if device is None:
        device = infer_device(obb_first, obb_second)

    obb_first_wp = to_wp_array(obb_first, wp.vec3, device=device)
    obb_second_wp = to_wp_array(obb_second, wp.vec3, device=device)

    box_volumes_first_wp, box_volumes_second_wp, out_wp = _obb_intersection_volumes_impl(obb_first_wp, obb_second_wp, device, pairwise=pairwise)
    return (
            from_wp_array(box_volumes_first_wp, like=obb_first),
            from_wp_array(box_volumes_second_wp, like=obb_second),
            from_wp_array(out_wp, like=obb_first),
        )

def obb_overlaps(
    obb_first,
    obb_second,
    pairwise: bool = False,
    device: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute the intersection volumes between two sets of OBBs.

    Parameters
    ----------
    obb_first : array-like, shape (N, 8, 3)
        First set of OBBs.
    obb_second : array-like, shape (M, 8, 3)
        Second set of OBBs.
    pairwise : bool
        When True, requires N == M and computes pairwise intersections.
    device : str or Device
        The device to use for the computation. If not provided, will be inferred from the inputs.

    Returns
    -------
    overlap1 : array-like, shape (N,)
    overlap2 : array-like, shape (M,)
    """

    vol1, vol2, inter = obb_intersection_volumes(obb_first, obb_second, pairwise=pairwise, device=device)
    if not pairwise:
        vol1 = vol1[:, None]
    return inter / (vol1 + 1e-9), inter / (vol2 + 1e-9)


def obb_ious(
    obb_first,
    obb_second,
    pairwise: bool = False,
    device: Optional[str] = None,
) -> Tensor:
    """Compute the intersection volumes between two sets of OBBs.

    Parameters
    ----------
    obb_first : array-like, shape (N, 8, 3)
    obb_second : array-like, shape (M, 8, 3)
    pairwise : bool
        When True, requires N == M and computes pairwise intersections.
    device : str or Device
        The device to use for the computation. If not provided, will be inferred from the inputs.

    Returns
    -------
    iou : array-like, shape (N, M)
    """
    vol1, vol2, inter = obb_intersection_volumes(obb_first, obb_second, pairwise=pairwise, device=device)
    if not pairwise:
        vol1 = vol1[:, None]
    return inter / (vol1 + vol2 - inter + 1e-9)
