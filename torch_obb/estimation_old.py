
from __future__ import annotations
import numba as nb
import numpy as np


@nb.njit(nogil=True, fastmath=True, inline='always')
def normalize(u):
    return u / np.sqrt(np.dot(u, u))



NUM_SAMPLE_DIRECTIONS = 7
NUM_SAMPLES = NUM_SAMPLE_DIRECTIONS * 2


NUM_SLAB_DIRS = 7
SLAB_DIRS = np.array([[1,  0,  0],
                      [0,  1,  0],
                      [0,  0,  1],
                      [1,  1,  1],
                      [1,  1, -1],
                      [1, -1,  1],
                      [1, -1, -1]], dtype=np.float32)


AABB_VERTICES = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 1, 0],
                          [0, 1, 1],
                          [1, 0, 0],
                          [1, 0, 1],
                          [1, 1, 0],
                          [1, 1, 1]], dtype=np.float32)

epsilon = 1e-6



def slab_extremas(vertices: np.ndarray):
    proj = vertices * SLAB_DIRS.T

    print(proj.shape)

@nb.njit(nogil=True, fastmath=True, cache=True)
def find_extremal_points(vertices):
    """Find extremal points along the 7 slab directions for DITO-14 algorithm."""
    t_min_proj = np.empty(NUM_SLAB_DIRS, dtype=vertices.dtype)
    t_max_proj = np.empty(NUM_SLAB_DIRS, dtype=vertices.dtype)

    t_min_vert = np.empty((NUM_SLAB_DIRS, 3), dtype=vertices.dtype)
    t_max_vert = np.empty((NUM_SLAB_DIRS, 3), dtype=vertices.dtype)

    # initialize with the first vertex
    for s in range(NUM_SLAB_DIRS):
        proj = np.dot(vertices[0], SLAB_DIRS[s])
        t_min_proj[s] = t_max_proj[s] = proj
        t_min_vert[s] = vertices[0]
        t_max_vert[s] = vertices[0]

    # process the other vertices
    for i in range(1, vertices.shape[0]):
        for s in range(NUM_SLAB_DIRS):
            proj = np.dot(vertices[i], SLAB_DIRS[s])
            if proj < t_min_proj[s]:
                t_min_proj[s] = proj
                t_min_vert[s] = vertices[i]
            if proj > t_max_proj[s]:
                t_max_proj[s] = proj
                t_max_vert[s] = vertices[i]

    # Note: Normalization of the extremal projection values can be done here.
    # DiTO-14 only needs the extremal vertices, and the extremal projection values for slab 0-2 (to set the initial AABB).
    # Since unit normals are used for slab 0-2, no normalization is needed.
    # When needed, normalization of the remaining projection values can be done efficiently as follows:
    # t_min_proj[3] *= 0.57735027f
    # t_max_proj[3] *= 0.57735027f
    # t_min_proj[4] *= 0.57735027f
    # t_max_proj[4] *= 0.57735027f
    # t_min_proj[5] *= 0.57735027f
    # t_max_proj[5] *= 0.57735027f
    # t_min_proj[6] *= 0.57735027f
    # t_max_proj[6] *= 0.57735027f

    return t_min_proj, t_max_proj, t_min_vert, t_max_vert


@nb.njit(nogil=True, fastmath=True, inline='always')
def norm(v: np.ndarray) -> float:
    return np.dot(v, v)

@nb.njit(nogil=True, fastmath=True, inline='always')
def normalize(v: np.ndarray) -> np.ndarray:
    return v / norm(v)


@nb.njit(nogil=True, fastmath=True, cache=True)
def quality_value(v: np.ndarray) -> np.ndarray:
    return v[0] * v[1] + v[0] * v[2] + v[1] * v[2] # half box area


@nb.njit(nogil=True, fastmath=True, cache=True)
def furthest_point_pair(min_vertices, max_vertices):
    #max_first = min_vertices[0]
    #max_second = max_vertices[0]
    furthest_index = 0
    max_dist = norm(max_vertices[0] - min_vertices[0])

    for i in range(1, min_vertices.shape[0]):
            d = norm(max_vertices[i] - min_vertices[i])
            if d > max_dist:
                furthest_index = i
                max_dist = d

    return min_vertices[furthest_index], max_vertices[furthest_index]


@nb.njit(nogil=True, fastmath=True, cache=True)
def dist_point_infinite_edge(q, p0, v):
    u0 = q - p0
    t = np.dot(v, u0)
    return norm(u0) - t*t / norm(v)


@nb.njit(nogil=True, fastmath=True, cache=True)
def furthest_point_infinite_edge(p0, e0, vertices):
    max_dist = dist_point_infinite_edge(vertices[0], p0, e0)
    max_index = 0

    for i in range(1, vertices.shape[0]):
        d = dist_point_infinite_edge(vertices[i], p0, e0)
        if d > max_dist:
            max_dist = d
            max_index = i

    return max_dist, vertices[max_index]


@nb.njit(nogil=True, fastmath=True, cache=True)
def find_extremal_projs_one_dir(vertices, normal):
    proj = np.dot(vertices[0], normal)
    t_min_proj = proj
    t_max_proj = proj
	
    for i in range(1, vertices.shape[0]):
        proj = np.dot(vertices[i], normal)
        t_min_proj = min(proj, t_min_proj)
        t_max_proj = max(proj, t_max_proj)

    return t_min_proj, t_max_proj


@nb.njit(nogil=True, fastmath=True, cache=True)
def best_obb_axes_from_triangle_normal_and_edge(vertices, e ,n ,b , bestval):
    m0 = np.cross(e[0], n)
    m1 = np.cross(e[1], n)
    m2 = np.cross(e[2], n)

    # The operands are assumed to be orthogonal and unit normals	
    dmax = np.zeros(3, dtype=vertices.dtype) 
    dmin = np.zeros(3, dtype=vertices.dtype) 
    dlen = np.zeros(3, dtype=vertices.dtype)

    min_proj, max_proj = find_extremal_projs_one_dir(vertices, n)
    dmin[1] = min_proj
    dmax[1] = max_proj
    dlen[1] = max_proj - min_proj       

    min_proj, max_proj = find_extremal_projs_one_dir(vertices, e[0])
    dmin[0] = min_proj
    dmax[0] = max_proj
    dlen[0] = max_proj - min_proj       


    min_proj, max_proj = find_extremal_projs_one_dir(vertices, m0)
    dmin[2] = min_proj
    dmax[2] = max_proj
    dlen[2] = max_proj - min_proj     

    quality = quality_value(dlen)

    if quality < bestval:
        bestval = quality
        b[0] = e[0]
        b[1] = n
        b[2] = m0 

    min_proj, max_proj = find_extremal_projs_one_dir(vertices, e[1])
    dmin[0] = min_proj
    dmax[0] = max_proj
    dlen[0] = max_proj - min_proj    

    min_proj, max_proj = find_extremal_projs_one_dir(vertices, m1)
    dmin[2] = min_proj
    dmax[2] = max_proj
    dlen[2] = max_proj - min_proj   

    quality = quality_value(dlen)

    if quality < bestval:
        bestval = quality
        b[0] = e[1]
        b[1] = n
        b[2] = m1 

    min_proj, max_proj = find_extremal_projs_one_dir(vertices, e[2])
    dmin[0] = min_proj
    dmax[0] = max_proj
    dlen[0] = max_proj - min_proj    

    min_proj, max_proj = find_extremal_projs_one_dir(vertices, m2)
    dmin[2] = min_proj
    dmax[2] = max_proj
    dlen[2] = max_proj - min_proj 

    quality = quality_value(dlen)

    if quality < bestval:
        bestval = quality
        b[0] = e[2]
        b[1] = n
        b[2] = m2

    return bestval
 


@nb.njit(nogil=True, fastmath=True, cache=True)
def best_obb_axes_from_base_triangle(min_vertices, max_vertices, vertices, e, n, b, bestval):
    p0, p1 = furthest_point_pair(min_vertices, max_vertices)

    diff = p0 - p1
    diff_norm = norm(diff)

    # Degenerate case 1:
    # If the found furthest points are located very close, return OBB aligned with the initial AABB 
    if diff_norm < epsilon:
        return 1, bestval
    
    e[0] = diff / diff_norm

    dist, p2 = furthest_point_infinite_edge(p0, e[0], vertices)

	# Degenerate case 2:
	# If the third point is located very close to the line, return an OBB aligned with the line 
    if dist < epsilon: 
        return 2, bestval
    
    # Compute the two remaining edge vectors and the normal vector of the base triangle
    e[1] = normalize(p1 - p2)
    e[2] = normalize(p2 - p0)
    n[:] = normalize(np.cross(e[1], e[0]))

    bestval = best_obb_axes_from_triangle_normal_and_edge(vertices, e, n, b, bestval)

    return 0, bestval


@nb.njit(nogil=True, fastmath=True, cache=True)
def finalize_obb(v0, v1, v2, bmin, bmax, blen):
    q = (bmin + bmax) * 0.5
    center = v0 * q[0] + v1 * q[1] + v2 * q[2]

    vertices = AABB_VERTICES
    vertices = vertices - 0.5
    vertices = vertices * blen

    result = np.empty((8, 3), dtype=np.float32)
    for i in range(8):
        result[i] = center + v0 * vertices[i, 0] + v1 * vertices[i, 1] + v2 * vertices[i, 2]

    return result


@nb.njit(nogil=True, fastmath=True, cache=True)
def finalize_axis_aligned_obb(center, extent):
    vertices = AABB_VERTICES - 0.5
    vertices = vertices * extent
    vertices = vertices + center

    return vertices


@nb.njit(nogil=True, fastmath=True, cache=True)
def compute_obb_dimensions(vertices, v0, v1, v2):
    bmin = np.empty(3, dtype=vertices.dtype)
    bmax = np.empty(3, dtype=vertices.dtype)

    min_proj, max_proj = find_extremal_projs_one_dir(vertices, v0)
    bmin[0] = min_proj
    bmax[0] = max_proj

    min_proj, max_proj = find_extremal_projs_one_dir(vertices, v1)
    bmin[1] = min_proj
    bmax[1] = max_proj

    min_proj, max_proj = find_extremal_projs_one_dir(vertices, v2)
    bmin[2] = min_proj
    bmax[2] = max_proj

    return bmin, bmax


@nb.njit(nogil=True, fastmath=True, cache=True)
def finalize_line_aligned_obb(u, vertices):
    # This function is only called if the construction of the large base triangle fails

	# Given u, build any orthonormal base u, v, w
	# Make sure r is not equal to u
    r = np.array([u[0], u[1], u[2]], dtype=u.dtype)

    if np.abs(u[0]) > np.abs(u[1]) and np.abs(u[0]) > np.abs(u[2]):
        r = np.array([0, 1, 0], dtype=u.dtype)
    elif np.abs(u[1]) > np.abs(u[2]):
        r = np.array([0, 0, 1], dtype=u.dtype)
    else:
        r = np.array([1, 0, 0], dtype=u.dtype)

    r_len = norm(r)
    if r_len < epsilon:
        r = np.array([1, 0, 0], dtype=u.dtype)

    v = normalize(np.cross(u, r))
    w = normalize(np.cross(u, v))

    # compute the true obb dimensions by iterating over all vertices

    bmin, bmax = compute_obb_dimensions(vertices, u, v, w)
    blen = bmax - bmin
    return finalize_obb(u, v, w, bmin, bmax, blen)


def find_improved_obb_axes_from_upper_and_lower_tetras_of_basetriangle(vertices, e, n, b, p, bestval):
    pass


@nb.njit(nogil=True, fastmath=True, cache=True)
def oriented_bounding_box_dito_14(vertices: np.ndarray) -> np.ndarray:
    """Compute oriented bounding box using the DITO-14 algorithm."""

    if vertices.shape[0] == 0:
        return np.zeros((8, 3), dtype=np.float32) # returns the 8 bounding vertices (all zeros in this case)

    p = np.empty((3, 3), dtype=np.float32) # Vertices of the large base triangle
    e = np.empty((3, 3), dtype=np.float32) # Edge vectors of the large base triangle
    n = np.empty(3, dtype=np.float32)      # Unit normal of the large base triangle
    b = np.empty((3, 3), dtype=np.float32) # The best currently found obb orientation

    min_projections, max_projections, min_vertices, max_vertices = find_extremal_points(vertices)

    aabb_center = (min_projections[:3] + max_projections[:3]) * 0.5 # axis-aligned center point of the vertices
    aabb_extent = max_projections[:3] - min_projections[:3] # axis-aligned extents of the vertices
    # Ensure consistent types
    aabb_center = aabb_center.astype(np.float32)
    aabb_extent = aabb_extent.astype(np.float32)
    aabb_value = quality_value(aabb_extent)

    best_value = aabb_value

    # set initial orientation to axis-aligned
    b[0] = np.array([1, 0, 0])
    b[1] = np.array([0, 1, 0])
    b[2] = np.array([0, 0, 1])

    if vertices.shape[0] <= NUM_SAMPLE_DIRECTIONS:
        selected_vertices = vertices # use all input vertices since they are few
    else:
        selected_vertices = np.concatenate((min_vertices, max_vertices), axis=0) # use the selected extreme points

    base_triangle_constraint, best_value = best_obb_axes_from_base_triangle(min_vertices, max_vertices, vertices, e, n, b, best_value)

    if base_triangle_constraint == 1:
        return finalize_axis_aligned_obb(aabb_center, aabb_extent)
    elif base_triangle_constraint == 2:
        return finalize_line_aligned_obb(e[0], vertices)

    # Find improved OBB axes based on constructed di-tetrahedral shape raised from base triangle
    # TODO: Implement the di-tetrahedral improvement step from DITO-14

    # For now, compute the final OBB using the best orientation found
    bmin, bmax = compute_obb_dimensions(vertices, b[0], b[1], b[2])
    blen = bmax - bmin
    return finalize_obb(b[0], b[1], b[2], bmin, bmax, blen)