import warp as wp
import torch
import numpy as np
from typing import Optional, Any, Tuple
from ..util import ensure_warp_available
from .estimation_utils import compute_obb_vertices

SLAB_DIRS = torch.tensor([[1,  0,  0],
                          [0,  1,  0],
                          [0,  0,  1],
                          [1,  1,  1],
                          [1,  1, -1],
                          [1, -1,  1],
                          [1, -1, -1]], dtype=torch.float32, device=torch.device('cpu'))
NUM_SLAB_DIRS = SLAB_DIRS.shape[0]
NUM_SELECTED_VERTICES = NUM_SLAB_DIRS * 2
BLOCK_SIZE = 16
EPSILON = 1e-6


@wp.func
def length_sq(v: wp.vec3f) -> float:
    return wp.dot(v, v)


@wp.func
def safe_normalize(v: wp.vec3f) -> wp.vec3:
    l2 = wp.dot(v, v)
    if l2 <= wp.float32(EPSILON):
        return wp.vec3(0.0, 0.0, 0.0)
    inv_len = wp.sqrt(l2)
    return v / inv_len


@wp.func
def _projection_helper(u0: Any, t: Any) -> float:
    return wp.dot(u0, u0) - t * t


@wp.func
def dist_point_infinite_edge_tiled(vertices_tile: Any, p0: wp.vec3f, v: wp.vec3f) -> Any:
    p0_tile = wp.tile_broadcast(wp.tile(p0, preserve_type=True), shape=(NUM_SELECTED_VERTICES,))
    u0_tile = wp.tile_map(wp.sub, vertices_tile, 
                      wp.tile_broadcast(p0_tile, shape=(NUM_SELECTED_VERTICES,)))
    t_tile = wp.tile_map(wp.dot, u0_tile, 
                    wp.tile_broadcast(wp.tile(v, preserve_type=True), shape=(NUM_SELECTED_VERTICES,)))

    denom = wp.max(wp.length_sq(v), wp.float32(EPSILON))
    tt_tile = wp.tile_map(_projection_helper, u0_tile, t_tile)
    return wp.tile_map(wp.div, tt_tile, wp.tile_broadcast(wp.tile(denom, preserve_type=True), shape=(NUM_SELECTED_VERTICES,)))


@wp.func
def find_extremal_projs_one_dir_tiled(vertices_tile: Any, direction: wp.vec3f) -> Tuple[float, float]:
    projections = wp.tile_map(wp.dot, vertices_tile, 
                              wp.tile_broadcast(wp.tile(direction, preserve_type=True), shape=(NUM_SELECTED_VERTICES,)))
    min_proj = wp.tile_min(projections)
    max_proj = wp.tile_max(projections)
    return wp.untile(min_proj), wp.untile(max_proj)


@wp.func
def half_box_area(extents: wp.vec3f) -> float:
    return extents[0] * extents[1] + extents[0] * extents[2] + extents[1] * extents[2]


@wp.kernel(enable_backward=False)
def best_obb_axes_from_base_triangle_kernel(
    aabb_min: wp.array(dtype=wp.vec3f, ndim=1),
    aabb_max: wp.array(dtype=wp.vec3f, ndim=1),
    min_vertices: wp.array(dtype=wp.vec3f, ndim=2),
    max_vertices: wp.array(dtype=wp.vec3f, ndim=2),
    selected_vertices: wp.array(dtype=wp.vec3f, ndim=2),
    basis: wp.array(dtype=wp.vec3f, ndim=2),
):
    i = wp.tid()

    #aabb_center = aabb_min[i] + aabb_max[i] * 0.5 # axis-aligned center point of the vertices
    aabb_extent = aabb_max[i] - aabb_min[i] # axis-aligned extents of the vertices
    best_val = half_box_area(aabb_extent)
    #wp.printf("i %d: best_val %f\n", i, best_val)

    min_tile = wp.tile_squeeze(wp.tile_load(min_vertices, shape=(1, NUM_SLAB_DIRS), offset=(i, 0)))
    max_tile = wp.tile_squeeze(wp.tile_load(max_vertices, shape=(1, NUM_SLAB_DIRS), offset=(i, 0)))

    diff_tile = wp.tile_map(wp.sub, max_tile, min_tile)
    dist_tile = wp.tile_map(length_sq, diff_tile)
    furthest = wp.tile_argmax(dist_tile)
    furthest_i = wp.untile(furthest)
    
    p0 = min_tile[furthest_i]
    p1 = max_tile[furthest_i]

    diff = p0 - p1
    diff_norm_sq = length_sq(diff)
    #case_val = wp.int32(0)

    e0 = wp.vec3()
    e1 = wp.vec3()
    e2 = wp.vec3()

    # Degenerate case 1:
    # If the found furthest points are located very close, return OBB aligned with the initial AABB 
    if diff_norm_sq <= EPSILON:
        #case_val = 1
        # build an axis aligned basis
        basis[i, 0] = wp.vec3f(1.0, 0.0, 0.0)
        basis[i, 1] = wp.vec3f(0.0, 1.0, 0.0)
        basis[i, 2] = wp.vec3f(0.0, 0.0, 1.0)
        return
    
    diff_norm = wp.sqrt(diff_norm_sq)
    e0 = diff / diff_norm
    # load the selected vertices for the current element
    selected_tile =  wp.tile_squeeze(wp.tile_load(selected_vertices, 
                                        shape=(1, NUM_SELECTED_VERTICES), 
                                        offset=(i, 0)))
    # compute the distance to the infinite edge defined by e0 and p0
    dist_tile = dist_point_infinite_edge_tiled(selected_tile, p0, e0)
    dist_argmax_tile = wp.tile_argmax(dist_tile)
    dist_max_i = wp.untile(dist_argmax_tile)
    max_dist = dist_tile[dist_max_i]

    if max_dist <= EPSILON:
        #case_val = 2
        # if the construction of the large base triangle fails build and aabb aligned with the line
        # Given u, build any orthonormal base u, v, w
        # Make sure r is not equal to e0
        r = wp.vec3f(e0)
        if wp.abs(e0[0]) > wp.abs(e0[1]) and wp.abs(e0[0]) > wp.abs(e0[2]):
            r = wp.vec3f(0.0, 1.0, 0.0)
        elif wp.abs(e0[1]) > wp.abs(e0[2]):
            r = wp.vec3f(0.0, 0.0, 1.0)
        else:
            r = wp.vec3f(1.0, 0.0, 0.0)

        r_len = wp.length(r)
        if r_len < EPSILON:
            r = wp.vec3f(1.0, 0.0, 0.0)
        
        basis[i, 0] = r
        v = safe_normalize(wp.cross(e0, r))
        basis[i, 1] = v
        basis[i, 2] = safe_normalize(wp.cross(e0, v))
        return 
   
    b0 = wp.vec3f(1.0, 0.0, 0.0)
    b1 = wp.vec3f(0.0, 1.0, 0.0)
    b2 = wp.vec3f(0.0, 0.0, 1.0)

    p2 = selected_tile[dist_max_i]
    e1 = safe_normalize(p1 - p2)
    e2 = safe_normalize(p2 - p0)

    n_vec = safe_normalize(wp.cross(e1, e0))

    m0 = wp.cross(e0, n_vec)
    m1 = wp.cross(e1, n_vec)
    m2 = wp.cross(e2, n_vec)

    min_e0, max_e0 = find_extremal_projs_one_dir_tiled(selected_tile, e0)
    min_n, max_n = find_extremal_projs_one_dir_tiled(selected_tile, n_vec)
    min_m0, max_m0 = find_extremal_projs_one_dir_tiled(selected_tile, m0)

    span_e0 = max_e0 - min_e0
    span_n = max_n - min_n
    span_m0 = max_m0 - min_m0


    q0 = half_box_area(wp.vec3f(span_e0, span_n, span_m0))
    #wp.printf("i %d: q0 < best_val %f, %f\n", i, q0, best_val)

    if q0 < best_val:
        best_val = q0
        b0 = e0
        b1 = n_vec
        b2 = m0

    min_e1, max_e1 = find_extremal_projs_one_dir_tiled(selected_tile, e1)
    min_m1, max_m1 = find_extremal_projs_one_dir_tiled(selected_tile, m1)

    span_e1 = max_e1 - min_e1
    span_m1 = max_m1 - min_m1

    q1 = half_box_area(wp.vec3f(span_e1, span_n, span_m1))
    #wp.printf("i %d: q1 < best_val %f, %f\n", i, q1, best_val)

    if q1 < best_val:
        best_val = q1
        b0 = e1
        b1 = n_vec
        b2 = m1

    min_e2, max_e2 = find_extremal_projs_one_dir_tiled(selected_tile, e2)
    min_m2, max_m2 = find_extremal_projs_one_dir_tiled(selected_tile, m2)

    span_e2 = max_e2 - min_e2
    span_m2 = max_m2 - min_m2

    q2 = half_box_area(wp.vec3f(span_e2, span_n, span_m2))
    #wp.printf("i %d: q2 < best_val %f, %f\n", i, q2, best_val)

    if q2 < best_val:
        best_val = q2
        b0 = e2
        b1 = n_vec
        b2 = m2

    basis[i, 0] = b0
    basis[i, 1] = b1
    basis[i, 2] = b2


@wp.kernel(enable_backward=False)
def projections_kernel(vertices: wp.array(dtype=wp.vec3f, ndim=1), 
                       batch_indices: wp.array(dtype=wp.int32, ndim=1), 
                       basis: wp.array(dtype=wp.vec3f, ndim=2),
                       projections_out: wp.array(dtype=float, ndim=2)):
    i = wp.tid()
    b = batch_indices[i]
    basis = basis[b]
    v = vertices[i]

    for j in range(3):
        p = wp.dot(v, basis[j])
        projections_out[i, j] = p


def jagged_batch_indices(nested: torch.Tensor) -> torch.Tensor:
    """Construct the batch indices for the given nested Tensor as a Tensor if one integer index for each element along the jagged axis.
    """
    batch = torch.arange(nested.size(0), dtype=torch.int32, device=nested.device)
    return batch.repeat_interleave(nested.offsets().diff().to(batch.device))


def compute_obb_extents_jagged(vertices_jagged: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    vertices_wp = wp.from_torch(vertices_jagged.values(), dtype=wp.vec3f)
    device = vertices_wp.device
    batch_indices = jagged_batch_indices(vertices_jagged)
    batch_indices_wp = wp.from_torch(batch_indices, dtype=wp.int32)
    basis_wp = wp.from_torch(basis, dtype=wp.vec3f)
    
    projections_wp = wp.zeros((vertices_wp.shape[0], 3), dtype=wp.float32, device=device)

    wp.launch(projections_kernel, 
              dim=vertices_wp.shape[0], 
              inputs=[vertices_wp, batch_indices_wp, basis_wp, projections_wp], 
              device=device)

    projections_t_jagged = torch.nested.nested_tensor_from_jagged(wp.to_torch(projections_wp), offsets=vertices_jagged.offsets())
    
    min_extents = torch.min(projections_t_jagged, dim=1).values
    max_extents = torch.max(projections_t_jagged, dim=1).values
    return min_extents, max_extents


def obb_estimate_dito(vertices_t: torch.Tensor,
                      device: Optional[str] = None) -> torch.Tensor:
    ensure_warp_available()
    if device is None:
        device = vertices_t.device

    npoints_t = vertices_t.offsets().diff()
    if (npoints_t < NUM_SLAB_DIRS * 2).any():
        raise ValueError(f"Each batch must have at least {NUM_SLAB_DIRS * 2} vertices.")

    slab_dirs_t = SLAB_DIRS.to(dtype=vertices_t.dtype, device=device)#.unsqueeze(1)

    # operate directly on the non-jagged values and then convert back to jagged representations
    slab_projs_t = torch.nested.nested_tensor_from_jagged(torch.inner(vertices_t.values(), slab_dirs_t), 
                                                          offsets=vertices_t.offsets(), 
                                                          jagged_dim=1)

    min_proj_t, min_proj_arg_t = torch.min(slab_projs_t, dim=1)
    max_proj_t, max_proj_arg_t = torch.max(slab_projs_t, dim=1)

    # correct the indices for the offsets into the jagged array as torch nested does not support torch.gather directly
    min_proj_arg_t += slab_projs_t.offsets()[:-1].unsqueeze(1)
    max_proj_arg_t += slab_projs_t.offsets()[:-1].unsqueeze(1)

    min_vert_t = vertices_t.values()[min_proj_arg_t]
    max_vert_t = vertices_t.values()[max_proj_arg_t]

    many_vertices_mask_t = npoints_t > NUM_SLAB_DIRS * 2
    few_vertices_mask_t = ~many_vertices_mask_t

    # use just the selected extreme points for large point clouds and fall back to 
    # all input vertices for small point clouds
    selected_vertices_t = torch.empty(vertices_t.shape[0], NUM_SLAB_DIRS * 2, 3, 
                                      device=device, dtype=vertices_t.dtype)
    selected_vertices_t[many_vertices_mask_t] = torch.cat((min_vert_t[many_vertices_mask_t], max_vert_t[many_vertices_mask_t]), dim=1)
    batch_mask = few_vertices_mask_t.repeat_interleave(vertices_t.offsets().diff())
    selected_vertices_t[few_vertices_mask_t] = vertices_t.values()[batch_mask].reshape(-1, NUM_SLAB_DIRS * 2, 3)

    aabb_min_t = min_proj_t[:, :3]
    aabb_max_t = max_proj_t[:, :3]

    device_wp = wp.device_from_torch(device)
    aabb_min_wp = wp.from_torch(aabb_min_t, dtype=wp.vec3f).to(device_wp)
    aabb_max_wp = wp.from_torch(aabb_max_t, dtype=wp.vec3f).to(device_wp)
    min_vert_wp = wp.from_torch(min_vert_t, dtype=wp.vec3f).to(device_wp)
    max_vert_wp = wp.from_torch(max_vert_t, dtype=wp.vec3f).to(device_wp)
    selected_vertices_wp = wp.from_torch(selected_vertices_t, dtype=wp.vec3f).to(device_wp)
    basis_wp = wp.zeros((vertices_t.shape[0], 3), dtype=wp.vec3f, device=device_wp)
    wp.launch_tiled(best_obb_axes_from_base_triangle_kernel, 
                    dim=[min_vert_t.shape[0]], 
                    inputs=[aabb_min_wp, aabb_max_wp, min_vert_wp, max_vert_wp, 
                            selected_vertices_wp, basis_wp], 
                    block_dim=BLOCK_SIZE,
                    device=device_wp)

    basis_t = wp.to_torch(basis_wp)

    min_extents_t, max_extents_t = compute_obb_extents_jagged(vertices_t, basis_t)
    extents_t = max_extents_t - min_extents_t
    centroids_t = (min_extents_t + max_extents_t) * 0.5

    vertices_t = compute_obb_vertices(centroids_t, extents_t, basis_t)
    return vertices_t


