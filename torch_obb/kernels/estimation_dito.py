# This implementation is based on the OBB estimation algorithm described in
# Fast Computation of Tight‐Fitting Oriented Bounding Boxes
# by Thomas Larsson and Linus Källberg
# in "Game Engine Gems 2" by Eric Lengyel, Chapter 1
# and related sections on oriented bounding box computation.

import warp as wp
import torch
from typing import Any, Tuple
from .util import ensure_warp_available


SLAB_DIRS = torch.tensor([[1,  0,  0],
                          [0,  1,  0],
                          [0,  0,  1],
                          [1,  1,  1],
                          [1,  1, -1],
                          [1, -1,  1],
                          [1, -1, -1]], dtype=torch.float32, device=torch.device('cpu'))

NUM_SLAB_DIRS = SLAB_DIRS.shape[0]
NUM_SELECTED_VERTICES = NUM_SLAB_DIRS * 2
BLOCK_SIZE = 32
EPSILON = 1e-6

@wp.struct
class LargeBaseTriangle:
    p0: wp.vec3f
    p1: wp.vec3f
    p2: wp.vec3f
    e0: wp.vec3f
    e1: wp.vec3f
    e2: wp.vec3f
    n: wp.vec3f

@wp.struct
class Basis:
    b0: wp.vec3f
    b1: wp.vec3f
    b2: wp.vec3f
    best_val: wp.float32


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
def dist_point_infinite_edge_tiled(vertices_tile: Any, lbt: LargeBaseTriangle) -> Any:
    p0_tile = wp.tile_full((NUM_SELECTED_VERTICES,), lbt.p0, wp.vec3f)

    e0_tile = wp.tile_full((NUM_SELECTED_VERTICES,), lbt.e0, wp.vec3f)
    u0_tile = wp.tile_map(wp.sub, vertices_tile,
                          p0_tile)
    t_tile = wp.tile_map(wp.dot, u0_tile, 
                         e0_tile)
    denom = wp.max(wp.length_sq(lbt.e0), wp.float32(EPSILON))
    tt_tile = wp.tile_map(_projection_helper, u0_tile, t_tile)
    
    denom_tile = wp.tile_full((NUM_SELECTED_VERTICES,), denom, wp.float32)
    return wp.tile_map(wp.div, tt_tile, denom_tile)


@wp.func
def find_extremal_projs_one_dir_tiled(vertices_tile: Any, direction: wp.vec3f) -> Tuple[wp.float32, wp.float32, wp.int32, wp.int32]:
    
    direction_tile = wp.tile_full((NUM_SELECTED_VERTICES,), direction, wp.vec3f)
    projections = wp.tile_map(wp.dot, vertices_tile, 
                              direction_tile)
    min_i = wp.tile_argmin(projections)[0]
    max_i = wp.tile_argmax(projections)[0]
    min_proj = projections[min_i]
    max_proj = projections[max_i]
    return min_proj, max_proj, min_i, max_i


def half_box_area_torch(extents: torch.Tensor) -> float:
    return extents[:, 0] * extents[:, 1] + extents[:, 0] * extents[:, 2] + extents[:, 1] * extents[:, 2]

@wp.func
def half_box_area(extents: wp.vec3f) -> float:
    return extents[0] * extents[1] + extents[0] * extents[2] + extents[1] * extents[2]


@wp.func
def build_axis_aligned_basis(basis: wp.array(dtype=wp.vec3f, ndim=1)) -> None:
    basis[0] = wp.vec3f(1.0, 0.0, 0.0)
    basis[1] = wp.vec3f(0.0, 1.0, 0.0)
    basis[2] = wp.vec3f(0.0, 0.0, 1.0)

@wp.func
def build_line_aligned_basis(basis: wp.array(dtype=wp.vec3f, ndim=1), e0: wp.vec3f) -> None:
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
    
    basis[0] = r
    v = safe_normalize(wp.cross(e0, r))
    basis[1] = v
    basis[2] = safe_normalize(wp.cross(e0, v))


@wp.func
def best_obb_axes_from_normal_edge(selected_tile: Any, normal: wp.vec3f, e0: wp.vec3f, e1: wp.vec3f, e2: wp.vec3f, basis: Basis) -> Basis:    

    m0 = wp.cross(e0, normal)
    m1 = wp.cross(e1, normal)
    m2 = wp.cross(e2, normal)

    min_e0, max_e0, _, _ = find_extremal_projs_one_dir_tiled(selected_tile, e0)
    min_n, max_n, _, _ = find_extremal_projs_one_dir_tiled(selected_tile, normal)
    min_m0, max_m0, _, _ = find_extremal_projs_one_dir_tiled(selected_tile, m0)

    span_e0 = max_e0 - min_e0
    span_n = max_n - min_n
    span_m0 = max_m0 - min_m0

    q0 = half_box_area(wp.vec3f(span_e0, span_n, span_m0))
    if q0 < basis.best_val:
        basis.best_val = q0
        basis.b0 = e0
        basis.b1 = normal
        basis.b2 = m0

    min_e1, max_e1, _, _ = find_extremal_projs_one_dir_tiled(selected_tile, e1)
    min_m1, max_m1, _, _ = find_extremal_projs_one_dir_tiled(selected_tile, m1)

    span_e1 = max_e1 - min_e1
    span_m1 = max_m1 - min_m1

    q1 = half_box_area(wp.vec3f(span_e1, span_n, span_m1))
    if q1 < basis.best_val:
        basis.best_val = q1
        basis.b0 = e1
        basis.b1 = normal
        basis.b2 = m1

    min_e2, max_e2, _, _ = find_extremal_projs_one_dir_tiled(selected_tile, e2)
    min_m2, max_m2, _, _ = find_extremal_projs_one_dir_tiled(selected_tile, m2)

    span_e2 = max_e2 - min_e2
    span_m2 = max_m2 - min_m2

    q2 = half_box_area(wp.vec3f(span_e2, span_n, span_m2))
    if q2 < basis.best_val:
        basis.best_val = q2
        basis.b0 = e2
        basis.b1 = normal
        basis.b2 = m2

    return basis

@wp.func
def find_upper_lower_tetra_points_tiled(selected_tile: Any, lbt: LargeBaseTriangle) -> Tuple[wp.vec3f, wp.vec3f, wp.vec3f, wp.vec3f]:
    min_proj, max_proj, min_i, max_i = find_extremal_projs_one_dir_tiled(selected_tile, lbt.n)
    tri_proj = wp.dot(lbt.p0, lbt.n)
    max_valid = max_proj - EPSILON > tri_proj
    min_valid = min_proj + EPSILON < tri_proj
    return min_valid, max_valid, min_i, max_i


@wp.func
def find_improved_basis_from_upper_and_lower_tetras(selected_tile: Any, 
                                                    lbt: LargeBaseTriangle, 
                                                    basis: Basis) -> Basis:
    min_valid, max_valid, min_i, max_i = find_upper_lower_tetra_points_tiled(selected_tile, lbt)
    if max_valid:
        max_vert = selected_tile[max_i]
        f0 = safe_normalize(max_vert - lbt.p0)
        f1 = safe_normalize(max_vert - lbt.p1)
        f2 = safe_normalize(max_vert - lbt.p2)
        n0 = safe_normalize(wp.cross(f1, lbt.e0))
        n1 = safe_normalize(wp.cross(f2, lbt.e1))
        n2 = safe_normalize(wp.cross(f0, lbt.e2))
        basis = best_obb_axes_from_normal_edge(selected_tile, n0, lbt.e0, f1, f0, basis)
        basis = best_obb_axes_from_normal_edge(selected_tile, n1, lbt.e1, f2, f1, basis)
        basis = best_obb_axes_from_normal_edge(selected_tile, n2, lbt.e2, f0, f2, basis)
    if min_valid:
        min_vert = selected_tile[min_i]
        f0 = safe_normalize(min_vert - lbt.p0)
        f1 = safe_normalize(min_vert - lbt.p1)
        f2 = safe_normalize(min_vert - lbt.p2)
        n0 = safe_normalize(wp.cross(f1, lbt.e0))
        n1 = safe_normalize(wp.cross(f2, lbt.e1))
        n2 = safe_normalize(wp.cross(f0, lbt.e2))
        basis = best_obb_axes_from_normal_edge(selected_tile, n0, lbt.e0, f1, f0, basis)
        basis = best_obb_axes_from_normal_edge(selected_tile, n1, lbt.e1, f2, f1, basis)
        basis = best_obb_axes_from_normal_edge(selected_tile, n2, lbt.e2, f0, f2, basis)
    return basis

@wp.kernel(enable_backward=False)
def dito_kernel(
    aabb_min: wp.array(dtype=wp.vec3f, ndim=1),
    aabb_max: wp.array(dtype=wp.vec3f, ndim=1),
    min_vertices: wp.array(dtype=wp.vec3f, ndim=2),
    max_vertices: wp.array(dtype=wp.vec3f, ndim=2),
    selected_vertices: wp.array(dtype=wp.vec3f, ndim=2),
    basis_out: wp.array(dtype=wp.vec3f, ndim=2),
):
    i = wp.tid()

    aabb_extent = aabb_max[i] - aabb_min[i] # axis-aligned extents of the vertices
    aabb_val = half_box_area(aabb_extent)

    # initialize the basis with the initial AABB
    basis = Basis()
    basis.b0 = wp.vec3f(1.0, 0.0, 0.0)
    basis.b1 = wp.vec3f(0.0, 1.0, 0.0)
    basis.b2 = wp.vec3f(0.0, 0.0, 1.0)
    basis.best_val = aabb_val

    #wp.printf("best val initial %f\n", basis.best_val)

    min_tile = wp.tile_squeeze(wp.tile_load(min_vertices, shape=(1, NUM_SLAB_DIRS), offset=(i, 0)))
    max_tile = wp.tile_squeeze(wp.tile_load(max_vertices, shape=(1, NUM_SLAB_DIRS), offset=(i, 0)))

    diff_tile = wp.tile_map(wp.sub, max_tile, min_tile)
    dist_tile = wp.tile_map(length_sq, diff_tile)
    furthest = wp.tile_argmax(dist_tile)
    furthest_i = furthest[0] #wp.untile(furthest)

    lbt = LargeBaseTriangle()
    lbt.p0 = min_tile[furthest_i]
    lbt.p1 = max_tile[furthest_i]

    diff = lbt.p0 - lbt.p1
    diff_norm_sq = length_sq(diff)

    # Degenerate case 1:
    # If the found furthest points are located very close, return OBB aligned with the initial AABB 
    if diff_norm_sq <= EPSILON:
        #case_val = 1
        # build an axis aligned basis and return
        build_axis_aligned_basis(basis_out[i])
        return
    
    diff_norm = wp.sqrt(diff_norm_sq)
    lbt.e0 = diff / diff_norm

    # load the selected vertices for the current element
    selected_tile =  wp.tile_squeeze(wp.tile_load(selected_vertices, 
                                        shape=(1, NUM_SELECTED_VERTICES), 
                                        offset=(i, 0)))

    # compute the distance to the infinite edge defined by e0 and p0
    dist_tile = dist_point_infinite_edge_tiled(selected_tile, lbt)
    dist_argmax_tile = wp.tile_argmax(dist_tile)
    dist_max_i = dist_argmax_tile[0] #wp.untile(dist_argmax_tile)
    max_dist = dist_tile[dist_max_i]

    if max_dist <= EPSILON:
        # if the construction of the large base triangle fails build and aabb aligned with the line
        # Given u, build any orthonormal base u, v, w
        build_line_aligned_basis(basis_out[i], lbt.e0)
        return 

    lbt.p2 = selected_tile[dist_max_i]
    lbt.e1 = safe_normalize(lbt.p1 - lbt.p2)
    lbt.e2 = safe_normalize(lbt.p2 - lbt.p0)
    lbt.n = safe_normalize(wp.cross(lbt.e1, lbt.e0))

    # wp.printf("lbt.e0 %f %f %f\n", lbt.e0[0], lbt.e0[1], lbt.e0[2])
    # wp.printf("lbt.e1 %f %f %f\n", lbt.e1[0], lbt.e1[1], lbt.e1[2])
    # wp.printf("lbt.e2 %f %f %f\n", lbt.e2[0], lbt.e2[1], lbt.e2[2])
    # wp.printf("lbt.n %f %f %f\n", lbt.n[0], lbt.n[1], lbt.n[2])

    # find the initial best OBB axes based on the large base triangle
    basis = best_obb_axes_from_normal_edge(selected_tile, lbt.n, lbt.e0, lbt.e1, lbt.e2, basis)
    
    # wp.printf("basis.b0 %f %f %f\n", basis.b0[0], basis.b0[1], basis.b0[2])
    # wp.printf("basis.b1 %f %f %f\n", basis.b1[0], basis.b1[1], basis.b1[2])
    # wp.printf("basis.b2 %f %f %f\n", basis.b2[0], basis.b2[1], basis.b2[2])
    # wp.printf("basis.best_val %f\n", basis.best_val)

    # find an improved basis from the upper and lower tetrahedra
    basis = find_improved_basis_from_upper_and_lower_tetras(selected_tile, lbt, basis)

    # wp.printf("improved basis.b0 %f %f %f\n", basis.b0[0], basis.b0[1], basis.b0[2])
    # wp.printf("improved basis.b1 %f %f %f\n", basis.b1[0], basis.b1[1], basis.b1[2])
    # wp.printf("improved basis.b2 %f %f %f\n", basis.b2[0], basis.b2[1], basis.b2[2])
    # wp.printf("improved basis.best_val %f\n", basis.best_val)

    if basis.best_val >= aabb_val:
        # fall back to aabb if the best found obb is actually worse than the initial aabb
        build_axis_aligned_basis(basis_out[i])
    else:
        basis_out[i, 0] = basis.b0
        basis_out[i, 1] = basis.b1
        basis_out[i, 2] = basis.b2
    return


def compute_obb_extents_jagged(vertices_jagged: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    projections_t_jagged = torch.bmm(vertices_jagged, basis)
    
    min_extents = torch.min(projections_t_jagged, dim=1).values
    max_extents = torch.max(projections_t_jagged, dim=1).values

    return min_extents, max_extents


def obb_estimate_dito(vertices_t: torch.Tensor) -> torch.Tensor:
    ensure_warp_available()
    device = vertices_t.device
    device_wp = wp.device_from_torch(device)

    stream_wp = wp.stream_from_torch(device)

    #npoints_t = vertices_t.offsets().diff()
    # if (npoints_t < NUM_SLAB_DIRS * 2).any():
    #     raise ValueError(f"Each batch must have at least {NUM_SLAB_DIRS * 2} vertices.")

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

    #many_vertices_mask_t = npoints_t > NUM_SLAB_DIRS * 2
    #few_vertices_mask_t = ~many_vertices_mask_t

    # use just the selected extreme points for large point clouds and fall back to 
    # all input vertices for small point clouds
    # selected_vertices_t = torch.empty(vertices_t.shape[0], NUM_SLAB_DIRS * 2, 3, 
    #                                   device=device, dtype=vertices_t.dtype)
    # #selected_vertices_t[many_vertices_mask_t] = torch.cat((min_vert_t[many_vertices_mask_t], max_vert_t[many_vertices_mask_t]), dim=1)
    selected_vertices_t = torch.cat((min_vert_t, max_vert_t), dim=1)

    # batch_mask = few_vertices_mask_t.repeat_interleave(vertices_t.offsets().diff())
    # print("batch mask", batch_mask)
    # selected_vertices_t[few_vertices_mask_t] = vertices_t.values()[batch_mask].reshape(-1, NUM_SLAB_DIRS * 2, 3)
    #print("selected vertices", selected_vertices_t[1])
    aabb_min_t = min_proj_t[:, :3]
    aabb_max_t = max_proj_t[:, :3]

    aabb_val_t = half_box_area_torch(aabb_max_t - aabb_min_t)
    
    aabb_min_wp = wp.from_torch(aabb_min_t, dtype=wp.vec3f).to(device_wp)
    aabb_max_wp = wp.from_torch(aabb_max_t, dtype=wp.vec3f).to(device_wp)
    min_vert_wp = wp.from_torch(min_vert_t, dtype=wp.vec3f).to(device_wp)
    max_vert_wp = wp.from_torch(max_vert_t, dtype=wp.vec3f).to(device_wp)
    selected_vertices_wp = wp.from_torch(selected_vertices_t, dtype=wp.vec3f).to(device_wp)
    basis_wp = wp.zeros((vertices_t.shape[0], 3), dtype=wp.vec3f, device=device_wp)

    wp.launch_tiled(dito_kernel, 
                    dim=[min_vert_t.shape[0]], 
                    inputs=[aabb_min_wp, aabb_max_wp, min_vert_wp, max_vert_wp, 
                            selected_vertices_wp, basis_wp], 
                    block_dim=BLOCK_SIZE,
                    device=device_wp, stream=stream_wp)
    #wp.synchronize_device(device_wp)
    basis_t = wp.to_torch(basis_wp)
    # project the vertices from world into local space
    min_extents_t, max_extents_t = compute_obb_extents_jagged(vertices_t, basis_t.mT)
    half_extents_t = (max_extents_t - min_extents_t) * 0.5 # half extents
    half_box_area_t = half_box_area_torch(half_extents_t)
    half_box_area_mask_t = half_box_area_t >= aabb_val_t
    basis_t[half_box_area_mask_t] = torch.eye(3, device=device)
    min_extents_t[half_box_area_mask_t] = aabb_min_t[half_box_area_mask_t]
    max_extents_t[half_box_area_mask_t] = aabb_max_t[half_box_area_mask_t]
    half_extents_t = (max_extents_t - min_extents_t) * 0.5 # half extents

    center_local_t = (min_extents_t + max_extents_t) * 0.5
    # project the center from local into world space
    center_t = torch.bmm(center_local_t.unsqueeze(1), basis_t).squeeze(1)
    obbs_t = torch.cat([basis_t, center_t.unsqueeze(1), half_extents_t.unsqueeze(1)], dim=1)
    return obbs_t


