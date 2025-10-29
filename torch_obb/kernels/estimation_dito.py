from __future__ import annotations

import numpy as np
import warp as wp

from .util import check_batch_dim, ensure_warp_available, infer_device, to_wp_array, from_wp_array
from .typing import Tensor, Optional

# Constants from the DITO-14 algorithm
NUM_SAMPLE_DIRECTIONS = 7
NUM_SAMPLES = NUM_SAMPLE_DIRECTIONS * 2
NUM_SLAB_DIRS = 7

# Slab directions used in DITO-14 (7 directions)
SLAB_DIRS = np.array([[1,  0,  0],
                      [0,  1,  0],
                      [0,  0,  1],
                      [1,  1,  1],
                      [1,  1, -1],
                      [1, -1,  1],
                      [1, -1, -1]], dtype=np.float32)

# AABB vertices template (scaled to -0.5 to 0.5)
AABB_VERTICES = [[-0.5, -0.5, -0.5],
                 [-0.5, -0.5,  0.5],
                 [-0.5,  0.5, -0.5],
                 [-0.5,  0.5,  0.5],
                 [ 0.5, -0.5, -0.5],
                 [ 0.5, -0.5,  0.5],
                 [ 0.5,  0.5, -0.5],
                 [ 0.5,  0.5,  0.5]]


EXTREMAL_POINTS_TILE_K = wp.constant(128)

epsilon = 1e-6


@wp.func
def half_box_area(dlen: wp.vec3) -> wp.float32:
    """Compute the half box area as quality measure for OBB optimization."""
    return dlen[0] * dlen[1] + dlen[0] * dlen[2] + dlen[1] * dlen[2]


@wp.kernel(enable_backward=False)
def half_box_area_kernel(extents: wp.array(dtype=wp.vec3, ndim=1), areas: wp.array(dtype=wp.float32, ndim=1)) -> wp.float32:
    tid = wp.tid()
    areas[tid] = half_box_area(extents[tid])


@wp.kernel(enable_backward=False)
def find_extremal_points_batched_kernel(
    vertices_flat: wp.array(dtype=wp.vec3),   # concatenated vertices
    offsets:       wp.array(dtype=wp.int32),    # [M] batch offsets
    counts:        wp.array(dtype=wp.int32),    # [M] batch sizes
    slab_dirs:     wp.array(dtype=wp.vec3),   # [S] slab directions
    min_max_projections: wp.array(dtype=wp.float32, ndim=3),  # [M, 2, S] result min/max projections
    min_max_vertices: wp.array(dtype=wp.vec3, ndim=3),   # [M, 2, S] result min/max vertices
):
    tid = wp.tid()
    m = tid[0]          # shape index
    s = tid[1]          # direction index
    lane = tid[2]       # thread index within the tile/block

    base = offsets[m]
    n    = counts[m]
    d    = slab_dirs[s]

    best_min_val = wp.inf()
    best_min_idx = -1
    best_max_val = -wp.inf()
    best_max_idx = -1

    # tile-local, compile-time [0..TILE_K) positions
    pos = wp.tile_arange(0, EXTREMAL_POINTS_TILE_K, dtype=int)

    # process all tiles (including last partial); OOB reads are zero-filled on 1.6+
    # so it's safe to load full tiles every time
    num_tiles = (n + int(EXTREMAL_POINTS_TILE_K) - 1) // int(EXTREMAL_POINTS_TILE_K)

    for t in range(num_tiles):
        start = base + t * int(EXTREMAL_POINTS_TILE_K)

        # cooperatively load vertices into a 1D tile
        tv = wp.tile_load(vertices_flat, shape=(EXTREMAL_POINTS_TILE_K,), offset=(start,))

        # compute projections per lane
        tp = wp.tile_map(lambda v: wp.dot(v, d), tv)

        # compute valid length of this tile
        valid_len = n - t * int(EXTREMAL_POINTS_TILE_K)
        if valid_len > int(EXTREMAL_POINTS_TILE_K):
            valid_len = int(EXTREMAL_POINTS_TILE_K)
        if valid_len < 0:
            valid_len = 0

        # build a mask: pos < valid_len  (bool tile)
        vlim = wp.tile_broadcast(valid_len, shape=(EXTREMAL_POINTS_TILE_K,), dtype=int)
        mask = wp.tile_map(lambda j, lim: j < lim, pos, vlim)

        # neutralize invalid lanes for min/max
        tp_min = wp.tile_map(lambda p, msk: p if msk else wp.inf(),   tp, mask)
        tp_max = wp.tile_map(lambda p, msk: p if msk else -wp.inf(),  tp, mask)

        # tile-wide reductions + arg indices
        t_min_val = wp.tile_min(tp_min)        # shape (1,)
        t_max_val = wp.tile_max(tp_max)        # shape (1,)
        t_min_idx = wp.tile_argmin(tp_min)     # index in [0, EXTREMAL_POINTS_TILE_K)
        t_max_idx = wp.tile_argmax(tp_max)

        if lane == 0:
            vmin = t_min_val[0]
            vmax = t_max_val[0]
            imin = start + t_min_idx[0]
            imax = start + t_max_idx[0]

            if vmin < best_min_val:
                best_min_val = vmin
                best_min_idx = imin
            if vmax > best_max_val:
                best_max_val = vmax
                best_max_idx = imax

    # write final bests (one thread per (shape, dir))
    if lane == 0:
        min_max_projections[m, 0, s] = best_min_val
        min_max_projections[m, 1, s] = best_max_val
        min_max_vertices[m, 0, s] = vertices_flat[best_min_idx]
        min_max_vertices[m, 1, s] = vertices_flat[best_max_idx]


@wp.kernel(enable_backward=False)
def estimate_obb_kernel(vertices_flat: wp.array(dtype=wp.vec3, ndim=1),
                        batch_offsets: wp.array(dtype=wp.int32, ndim=1),
                        batch_counts: wp.array(dtype=wp.int32, ndim=1),
                        min_max_projections: wp.array(dtype=wp.float32, ndim=3),
                        min_max_vertices: wp.array(dtype=wp.vec3, ndim=3)):
    n = wp.tid()
    if batch_counts[n] > NUM_SAMPLE_DIRECTIONS:
        selected_vertices = min_max_vertices[n].reshape(NUM_SAMPLE_DIRECTIONS * 2)
    else:
        selected_vertices = vertices_flat[batch_offsets[n]:batch_offsets[n] + NUM_SAMPLE_DIRECTIONS * 2]

    # Vertices of the large base triangle
    p0 = wp.vec3()
    p1 = wp.vec3()
    p2 = wp.vec3()

    # Edge vectors of the large base triangle
    e0 = wp.vec3()
    e1 = wp.vec3()
    e2 = wp.vec3()

    n = wp.vec3()
    best_value = wp.float32()

    obb_candidate = wp.zeros(8, wp.vec3())

    base_triangle_constraint, best_value = best_obb_axes_from_base_triangle(selected_vertices, e0, e1, e2, n, b, best_value)

    base_triangle_constraint, best_value = best_obb_axes_from_base_triangle(min_vert, max_vert, vertices, e, n, b, best_value)

# Main public API function
def obb_estimate(
    vertices: Tensor,
    batch_offsets: Optional[Tensor] = None,
    batch_counts: Optional[Tensor] = None,
    device: Optional[str | wp.context.Device] = None,
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
    ensure_warp_available()

    # Determine device
    if device is None:
        device = infer_device(vertices)

    vertices_wp = to_wp_array(vertices, wp.vec3, device=device)

    # initialize batch_offsets and batch_counts if not provided assuming a single batch
    if batch_offsets is None:
        batch_offsets_wp = wp.zeros(1, dtype=wp.int32, device=device)
    else:
        batch_offsets_wp = to_wp_array(batch_offsets, wp.int32, device=device)

    if batch_counts is None:
        batch_counts_wp = wp.full(1, fill_value=vertices.shape[0], dtype=wp.int32, device=device)
    else:
        batch_counts_wp = to_wp_array(batch_counts, wp.int32, device=device)

    nbatches = batch_offsets_wp.shape[0]

    if (batch_counts_wp < NUM_SAMPLE_DIRECTIONS).any():
        raise ValueError("Each batch must have at least NUM_SAMPLE_DIRECTIONS vertices.")

    # upload input arrays to device
    slab_dirs_wp = to_wp_array(SLAB_DIRS, wp.vec3, device=device)

    min_max_projections_wp = wp.empty((nbatches, 2, NUM_SLAB_DIRS), dtype=wp.float32, device=device)
    min_max_vertices_wp = wp.empty((nbatches, 2, NUM_SLAB_DIRS), dtype=wp.vec3, device=device)

    wp.launch_tiled(find_extremal_points_batched_kernel,
                    dim=(nbatches, NUM_SLAB_DIRS, EXTREMAL_POINTS_TILE_K),
                    inputs=[vertices_wp, batch_offsets_wp, batch_counts_wp, slab_dirs_wp, min_max_projections_wp, min_max_vertices_wp],
                    block_dim=EXTREMAL_POINTS_TILE_K,
                    device=device)

    # basic math. not worth creating a kernel for this.
    #aabb_centers = (min_projections_wp[:, 0, :3] + max_projections_wp[:, 1, :3]) * 0.5 # axis-aligned center point of the vertices (nbatches, 3)
    #aabb_centers = aabb_centers.view(wp.vec3)
    #aabb_extents = max_projections_wp[:, :3] - min_projections_wp[:, :3] # axis-aligned extents of the vertices (M, 3)
    #aabb_extents = aabb_extents.view(wp.vec3)

    #aabb_qualities = wp.empty(nbatches, dtype=wp.float32, device=device)
    #wp.launch(half_box_area_kernel, dim=nbatches, inputs=[aabb_extents, aabb_qualities], device=device)

    # set initial orientation to axis-aligned
    #b[:, 0].fill(wp.vec3(1, 0, 0))
    #b[:, 1].fill(wp.vec3(0, 1, 0))
    #b[:, 2].fill(wp.vec3(0, 0, 1))
