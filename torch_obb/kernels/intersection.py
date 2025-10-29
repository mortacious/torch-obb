from __future__ import annotations
import warp as wp
from ..typing import Tensor, Tuple, Optional
from .warp_math import norm3

# Constants
_NUM_PLANES = 6
_MAX_INTERSECTION_TRIS = 100 # maximum number of triangles in the intersection buffers. Change if overflow occurs.
k_epsilon = 1e-8 # epsilons for numerical stability.
_d_epsilon = 1e-3

# Triangle and plane indices.
_TRIS = [[1, 3, 0],
         [4, 1, 0],
         [0, 3, 2],
         [2, 4, 0],
         [1, 7, 3],
         [5, 1, 4],
         [5, 7, 1],
         [3, 7, 2],
         [6, 4, 2],
         [2, 7, 6],
         [6, 5, 4],
         [7, 5, 6]]

_PLANES = [[0, 1, 3, 2],
           [0, 4, 5, 1],
           [0, 2, 6, 4],
           [1, 5, 7, 3],
           [2, 3, 7, 6],
           [4, 6, 7, 5]]


@wp.kernel(enable_backward=False)
def triangle_vertices_kernel(
    obbs: wp.array(dtype=wp.vec3, ndim=2), # (N, 8)
    out_tris: wp.array(dtype=wp.vec3, ndim=3), # (N, 12, 3)
):
    i = wp.tid()
    n = obbs.shape[0]
    if i >= n:
        return
    # static loop is unrolled by warp
    for t in range(wp.static(len(_TRIS))):
        out_tris[i, t, 0] = obbs[i, wp.static(_TRIS[t][0])]
        out_tris[i, t, 1] = obbs[i, wp.static(_TRIS[t][1])]
        out_tris[i, t, 2] = obbs[i, wp.static(_TRIS[t][2])]


@wp.kernel(enable_backward=False)
def centers_radii_kernel(
    obbs: wp.array(dtype=wp.vec3, ndim=2),  # (N, 8)
    centers_out: wp.array(dtype=wp.vec3, ndim=1),  # (N)
    radii_out: wp.array(dtype=wp.float32, ndim=1),  # (N)
):
    i = wp.tid()
    n = obbs.shape[0]
    if i >= n:
        return

    center = wp.vec3(0.0)
    for v in range(8):
        center += obbs[i, v]
    center /= wp.float32(8)
    centers_out[i] = center
    # determine the radius of the bounding sphere
    maxd2 = wp.float32(0.0)

    for v in range(8):
        d = obbs[i, v] - center
        d2 = wp.dot(d, d)
        if d2 > maxd2:
            maxd2 = d2
    radii_out[i] = wp.sqrt(maxd2)


@wp.func
def _face_normal(v0: wp.vec3, 
                 v1: wp.vec3, 
                 v2: wp.vec3, 
                 v3: wp.vec3, 
                 center: wp.vec3):
    """
    Computes a unit-length normal for a quad from 4 vertices and a center point.
    """
    # Compute vectors from center
    d = (v0 - center, v1 - center, v2 - center, v3 - center)

    # Candidate cross products (skip redundant/zero pairs)
    candidates = (
        wp.cross(d[0], d[1]), wp.cross(d[0], d[2]), wp.cross(d[0], d[3]),
        wp.cross(d[1], d[2]), wp.cross(d[1], d[3]), wp.cross(d[2], d[3])
    )

    # Find the largest by squared length
    best = candidates[0]
    best_len = wp.dot(best, best)
    for n in range(1, 6):
        l = wp.dot(candidates[n], candidates[n])
        if l > best_len:
            best, best_len = candidates[n], l

    return best * (1.0 / wp.sqrt(best_len)) if best_len > 0.0 else wp.vec3(0.0)


@wp.func
def _plane_edge_intersection(
    plane_center: wp.vec3,
    plane_normal: wp.vec3,
    p0: wp.vec3,
    p1: wp.vec3
) -> wp.vec3:
    # The point of intersection can be parametrized
    # p = p0 + a (p1 - p0) where a in [0, 1]
    # We want to find a such that p is on plane
    # <p - ctr, n> = 0
    d = p1 - p0

    # Normalize direction vector
    dn = norm3(d)
    if dn < k_epsilon:
        dn = k_epsilon
    d /= dn

    # Default to midpoint
    q = 0.5 * (p0 + p1)

    # Check parallelism using normalized direction
    n_dot_d_check = wp.dot(plane_normal, d)
    if wp.abs(n_dot_d_check) >= _d_epsilon:
        top = -wp.dot(p0 - plane_center, plane_normal)
        bot = wp.dot(d, plane_normal)
        a = top / bot
        q = p0 + a * d

    return q

@wp.func
def _clip_tri_by_plane_one_out(
    plane_center: wp.vec3,
    plane_normal: wp.vec3,
    vout: wp.vec3,
    vin1: wp.vec3,
    vin2: wp.vec3,
    out_tris: wp.array(dtype=wp.vec3, ndim=2),
    out_index: wp.int32,
) -> wp.int32:
    # point of intersection between plane and edge (vin1, vout)
    pint1 = _plane_edge_intersection(
        plane_center, plane_normal,
        vin1, vout
    )
    # point of intersection between plane and edge (vin2, vout)
    pint2 = _plane_edge_intersection(
        plane_center, plane_normal,
        vin2, vout
    )

    # Triangle 1: vin2, pint1, pint2
    out_tris[out_index, 0] = vin2
    out_tris[out_index, 1] = pint1
    out_tris[out_index, 2] = pint2

    # Triangle 2: vin1, pint1, vin2
    out_tris[out_index + 1, 0] = vin1
    out_tris[out_index + 1, 1] = pint1
    out_tris[out_index + 1, 2] = vin2

    return out_index + 2


@wp.func
def _clip_tri_by_plane_two_out(
    plane_center: wp.vec3,
    plane_normal: wp.vec3,
    vout1: wp.vec3,
    vout2: wp.vec3,
    vin: wp.vec3,
    out_tris: wp.array(dtype=wp.vec3, ndim=2),
    out_index: wp.int32,
) -> wp.int32:
    # point of intersection between plane and edge (vin, vout1)
    pint1 = _plane_edge_intersection(
        plane_center, plane_normal,
        vin, vout1
    )
    # point of intersection between plane and edge (vin, vout2)
    pint2 = _plane_edge_intersection(
        plane_center, plane_normal,
        vin, vout2
    )

    # Triangle: vin, pint1, pint2
    out_tris[out_index, 0] = vin
    out_tris[out_index, 1] = pint1
    out_tris[out_index, 2] = pint2

    return out_index + 1


@wp.func
def _triangle_coplanar_plane(vertices: wp.array(dtype=wp.vec3, ndim=1), 
                             plane_center: wp.vec3, 
                             plane_normal: wp.vec3) -> wp.bool:
    norm = norm3(plane_normal)
    if norm < k_epsilon:
        norm = k_epsilon
    for i in range(3):
        d = wp.abs(wp.dot(plane_normal, vertices[i] - plane_center) / norm)
        if d >= _d_epsilon:
            return False
    return True


@wp.func
def _triangle_normal(triangle_vertices: wp.array(dtype=wp.vec3, ndim=1)) -> wp.vec3:
    center = wp.vec3(0.0)
    for k in range(3):
        center += triangle_vertices[k]
    center /= wp.float32(3.0)

    max_dist = wp.float32(-1.0)
    normal = wp.vec3(0.0)
    for i in range(2):
        for j in range(i+1, 3):
            e0 = triangle_vertices[i] - center
            e1 = triangle_vertices[j] - center
            n = wp.cross(e0, e1)
            d = norm3(n)
            if d < k_epsilon:
                d = wp.float32(k_epsilon)
            if d > max_dist:
                normal = n / d
                max_dist = d
    return normal


@wp.func
def _any_coplanar(reference_vertices: wp.array(dtype=wp.vec3), 
                  triangle_vertices: wp.array(dtype=wp.vec3, ndim=2), n_triangles: wp.int32) -> bool:
    test_normal = _triangle_normal(reference_vertices)
    test_vertex = reference_vertices[0]

    for i in range(n_triangles):
        cp = _triangle_coplanar_plane(triangle_vertices[i], test_vertex, test_normal)
        if cp:
            return True
    return False


@wp.func
def _inside_plane(
    vertex: wp.vec3,
    plane_center: wp.vec3,
    plane_normal: wp.vec3
) -> wp.bool:
    center = vertex - plane_center
    return wp.dot(center, plane_normal) >= 0.0


@wp.func
def _clip_tri_by_plane(
    triangle_vertices: wp.array(dtype=wp.vec3, ndim=1),
    plane_center: wp.vec3,
    plane_normal: wp.vec3,
    out_tris: wp.array(dtype=wp.vec3, ndim=2),
    out_index: wp.int32,
) -> wp.int32:
    # Check if all triangle vertices are close to the plane
    coplanar = _triangle_coplanar_plane(triangle_vertices, plane_center, plane_normal)
    #wp.printf("is_coplanar %d\n", coplanar)
    if coplanar:
        # if the triangle is coplanar, we can just return return it
        for k in range(3):
            out_tris[out_index, k] = triangle_vertices[k]
        return out_index + 1

    in0 = _inside_plane(triangle_vertices[0], plane_center, plane_normal)
    in1 = _inside_plane(triangle_vertices[1], plane_center, plane_normal)
    in2 = _inside_plane(triangle_vertices[2], plane_center, plane_normal)
    #wp.printf("insides %d, %d, %d\n", in0, in1, in2)
    all_in = in0 and in1 and in2
    if all_in:
        # if all vertices are inside the plane, we can just return the triangle
        for k in range(3):
            out_tris[out_index, k] = triangle_vertices[k]
        return out_index + 1

    all_out = (not in0) and (not in1) and (not in2)
    if all_out:
        # if all vertices are outside the plane, we can skip further clipping and do not modify the out_index
        return out_index

    # Single vertex outside cases
    if in0 and in1 and (not in2):
        return _clip_tri_by_plane_one_out(
            plane_center, plane_normal,
            triangle_vertices[2],  # vout = tri[2]
            triangle_vertices[0],  # vin1 = tri[0]
            triangle_vertices[1],  # vin2 = tri[1]
            out_tris, out_index
        )

    if in0 and (not in1) and in2:
        return _clip_tri_by_plane_one_out(
            plane_center, plane_normal,
            triangle_vertices[1],  # vout = tri[1]
            triangle_vertices[0],  # vin1 = tri[0]
            triangle_vertices[2],  # vin2 = tri[2]
            out_tris, out_index
        )

    if (not in0) and in1 and in2:
        return _clip_tri_by_plane_one_out(
            plane_center, plane_normal,
            triangle_vertices[0],  # vout = tri[0]
            triangle_vertices[1],  # vin1 = tri[1]
            triangle_vertices[2],  # vin2 = tri[2]
            out_tris, out_index
        )

    # Two vertices outside cases
    if in0 and (not in1) and (not in2):
        return _clip_tri_by_plane_two_out(
            plane_center, plane_normal,
            triangle_vertices[1],  # vout1 = tri[1]
            triangle_vertices[2],  # vout2 = tri[2]
            triangle_vertices[0],  # vin = tri[0]
            out_tris, out_index
        )

    if (not in0) and (not in1) and in2:
        return _clip_tri_by_plane_two_out(
            plane_center, plane_normal,
            triangle_vertices[0],  # vout1 = tri[0]
            triangle_vertices[1],  # vout2 = tri[1]
            triangle_vertices[2],  # vin = tri[2]
            out_tris, out_index
        )

    if (not in0) and in1 and (not in2):
        return _clip_tri_by_plane_two_out(
            plane_center, plane_normal,
            triangle_vertices[0],  # vout1 = tri[0]
            triangle_vertices[2],  # vout2 = tri[2]
            triangle_vertices[1],  # vin = tri[1]
            out_tris, out_index
        )

    return out_index

@wp.func
def _box_intersection(
    triangle_vertices: wp.array(dtype=wp.vec3, ndim=2), # (12, 3)
    plane_centers: wp.array(dtype=wp.vec3, ndim=1), # (6)
    plane_normals: wp.array(dtype=wp.vec3, ndim=1), # (6)
    tris_out: wp.array(dtype=wp.vec3, ndim=2), # (MAX_INTERSECTION_TRIS, 3)
) -> wp.int32:
    # copy input to out_tris
    nout_start = wp.static(len(_TRIS))
    nout = wp.int32(nout_start)
    for t in range(nout_start):
        for k in range(3):
            tris_out[t, k] = triangle_vertices[t, k]

    current_tris = tris_out
    temp_tris = wp.zeros((_MAX_INTERSECTION_TRIS, 3), dtype=wp.vec3)
    tri = wp.zeros((3,), dtype=wp.vec3)
    #wp.printf("start nout %d\n", nout)
    for p in range(_NUM_PLANES):
        iupdated = wp.int32(0)
        for t in range(nout):
            tri[0] = current_tris[t, 0]
            tri[1] = current_tris[t, 1]
            tri[2] = current_tris[t, 2]
            iupdated = _clip_tri_by_plane(
                tri,
                plane_centers[p], plane_normals[p],
                temp_tris,
                iupdated,
            )
        # swap temp <-> current
        current_tris, temp_tris = temp_tris, current_tris
        nout = iupdated
        #wp.printf("nout %d\n", nout)
    # copy result to tris_out
    for t in range(nout):
        for k in range(3):
            tris_out[t, k] = current_tris[t, k]
    return nout


@wp.func
def _polyhedron_center(triangle_vertices: wp.array(dtype=wp.vec3, ndim=2), num_tris: wp.int32) -> wp.vec3:
    center = wp.vec3(0.0)

    # Find the center point of each face
    for t in range(num_tris):
        v0 = triangle_vertices[t, 0]
        v1 = triangle_vertices[t, 1]
        v2 = triangle_vertices[t, 2]

        face  = (v0 + v1 + v2) / 3.0
        center += face

    # Take the mean of the centers of all faces
    center /= wp.float32(num_tris)
    return center


@wp.func
def _poly_volume(triangle_vertices: wp.array(dtype=wp.vec3, ndim=2), 
                 center: wp.vec3,
                 num_tris: wp.int32) -> wp.float32:
    box_vol = wp.float32(0.0)
    for i in range(num_tris):
        v0 = triangle_vertices[i, 0] - center
        v1 = triangle_vertices[i, 1] - center
        v2 = triangle_vertices[i, 2] - center
        area = wp.dot(v0, wp.cross(v1, v2))
        vol = wp.abs(area) / wp.float32(6.0)
        box_vol += vol
    return box_vol


@wp.func
def _obb_intersection_volume(
    triangle_vertices_1: wp.array(dtype=wp.vec3, ndim=2), # (12, 3)
    plane_centers_1: wp.array(dtype=wp.vec3, ndim=1), # (6)
    plane_normals_1: wp.array(dtype=wp.vec3, ndim=1), # (6)
    triangle_vertices_2: wp.array(dtype=wp.vec3, ndim=2), # (12, 3)
    plane_centers_2: wp.array(dtype=wp.vec3, ndim=1), # (6)
    plane_normals_2: wp.array(dtype=wp.vec3, ndim=1), # (6)
) -> wp.float32:
    # intersection of box1 triangles with the planes of box2
    tmp1 = wp.zeros((_MAX_INTERSECTION_TRIS, 3), dtype=wp.vec3)
    n1 = _box_intersection(triangle_vertices_1, plane_centers_2, plane_normals_2, tmp1)
    # intersection of box2 triangles with the planes of box1
    tmp2 = wp.zeros((_MAX_INTERSECTION_TRIS, 3), dtype=wp.vec3)
    n2 = _box_intersection(triangle_vertices_2, plane_centers_1, plane_normals_1, tmp2)

    #print(n1)
    #print(n2)
    total_tris = wp.int32(n1)
    # If there are overlapping regions in Box2, remove any coplanar faces
    # Identify if any triangles in Box2 are coplanar with Box1
    tri = wp.zeros((3,), dtype=wp.vec3)
    for i in range(n2):
        tri[0] = tmp2[i, 0]
        tri[1] = tmp2[i, 1]
        tri[2] = tmp2[i, 2]
        if not _any_coplanar(tri, tmp1, n1):
            #if total_tris < _MAX_INTERSECTION_TRIS: # should never happen
            for k in range(3):
                tmp1[total_tris, k] = tmp2[i, k]
            total_tris += 1
    if total_tris == 0:
        return 0.0

    # Calculate polyhedron center and volume
    cc = _polyhedron_center(tmp1, total_tris)
    return _poly_volume(tmp1, cc, total_tris)

# preprocessing kernels

@wp.kernel(enable_backward=False)
def planes_kernel(
    obbs: wp.array(dtype=wp.vec3, ndim=2), # (N, 8, 3)
    centers: wp.array(dtype=wp.vec3, ndim=1), # (N, 3)
    plane_centers_out: wp.array(dtype=wp.vec3, ndim=2), # (N, 6)
    plane_normals_out: wp.array(dtype=wp.vec3, ndim=2), # (N, 6)
):
    i = wp.tid()
    n = obbs.shape[0]
    if i >= n:
        return

    # static loop is unrolled by warp
    for p in range(wp.static(len(_PLANES))):
        c0 = wp.static(_PLANES[p][0])
        c1 = wp.static(_PLANES[p][1])
        c2 = wp.static(_PLANES[p][2])
        c3 = wp.static(_PLANES[p][3])
        plane_center = obbs[i, c0] + obbs[i, c1] + obbs[i, c2] + obbs[i, c3]
        plane_center /= wp.float32(4)
        plane_centers_out[i, p] = plane_center

        # Use robust face normal calculation
        normal = _face_normal(
            obbs[i, c0], obbs[i, c1], obbs[i, c2], obbs[i, c3],
            plane_center
        )

        vc = centers[i] - plane_center
        if wp.dot(normal, vc) < 0.0:
            normal *= wp.float32(-1.0)
        plane_normals_out[i, p] = normal


@wp.kernel(enable_backward=False)
def box_volumes_kernel(
    tris_vertices: wp.array(dtype=wp.vec3, ndim=3), # (N, 12, 3)
    centers: wp.array(dtype=wp.vec3, ndim=1), # (N, 3)
    out_vol: wp.array(dtype=wp.float32, ndim=1), # (N)
):
    i = wp.tid()
    n = tris_vertices.shape[0]
    if i >= n:
        return
    center = centers[i]
    vol = wp.float32(0.0)

    for t in range(12):
        v0 = tris_vertices[i, t, 0] - center
        v1 = tris_vertices[i, t, 1] - center
        v2 = tris_vertices[i, t, 2] - center
        cv = wp.cross(v1, v2)
        area = wp.dot(v0, cv)
        vol += wp.abs(area) / wp.float32(6.0)
    out_vol[i] = vol


# -----------------------------
# Intersection kernels
# -----------------------------

@wp.kernel(enable_backward=False)
def kernel_full(
    triangle_vertices_1: wp.array(dtype=wp.vec3, ndim=3),  # (N, 12, 3)
    plane_centers_1: wp.array(dtype=wp.vec3, ndim=2),  # (N, 6)
    plane_normals_1: wp.array(dtype=wp.vec3, ndim=2),  # (N, 6)
    centers_1: wp.array(dtype=wp.vec3, ndim=1),  # (N)
    radii_1: wp.array(dtype=wp.float32, ndim=1), # (N)
    triangle_vertices_2: wp.array(dtype=wp.vec3, ndim=3), # (M, 12, 3)
    plane_centers_2: wp.array(dtype=wp.vec3, ndim=2), # (M, 6)
    plane_normals_2: wp.array(dtype=wp.vec3, ndim=2), # (M, 6)
    centers_2: wp.array(dtype=wp.vec3, ndim=1),  # (M)
    radii_2: wp.array(dtype=wp.float32, ndim=1), # (M)
    out: wp.array(dtype=wp.float32, ndim=2), # (N, M)
):
    """Compute the intersection volume between two sets of OBBs as an NxM matrix."""
    # 2D launch: dim=(N, M) where N=centers1.shape[0], M=centers2.shape[0]
    i, j = wp.tid()
    n = centers_1.shape[0]
    m = centers_2.shape[0]
    if i >= n or j >= m:
        return

    d = centers_1[i] - centers_2[j]
    d2 = wp.dot(d, d)
    rsum = radii_1[i] + radii_2[j]
    if d2 > rsum * rsum:
        # stop early if the distance is greater than the sum of the radii
        out[i, j] = 0.0
        return

    out[i, j] = _obb_intersection_volume(triangle_vertices_1[i], 
                                         plane_centers_1[i], 
                                         plane_normals_1[i], 
                                         triangle_vertices_2[j], 
                                         plane_centers_2[j], 
                                         plane_normals_2[j])


@wp.kernel(enable_backward=False)
def kernel_pairwise(
    triangle_vertices_1: wp.array(dtype=wp.vec3, ndim=3),  # (N, 12, 3)
    plane_centers_1: wp.array(dtype=wp.vec3, ndim=2),  # (N, 6)
    plane_normals_1: wp.array(dtype=wp.vec3, ndim=2),  # (N, 6)
    centers_1: wp.array(dtype=wp.vec3, ndim=1),  # (N)
    radii_1: wp.array(dtype=wp.float32, ndim=1), # (N)
    triangle_vertices_2: wp.array(dtype=wp.vec3, ndim=3), # (N, 12, 3)
    plane_centers_2: wp.array(dtype=wp.vec3, ndim=2), # (N, 6)
    plane_normals_2: wp.array(dtype=wp.vec3, ndim=2), # (N, 6)
    centers_2: wp.array(dtype=wp.vec3, ndim=1),  # (N)
    radii_2: wp.array(dtype=wp.float32, ndim=1), # (N)
    out: wp.array(dtype=wp.float32, ndim=1), # (N,)
):
    i = wp.tid()
    n = out.shape[0]
    if i >= n:
        return
    d = centers_1[i] - centers_2[i]
    d2 = wp.dot(d, d)
    rsum = radii_1[i] + radii_2[i]
    if d2 > rsum * rsum:
        # stop early if the distance is greater than the sum of the radii
        out[i] = 0.0
        return
    out[i] = _obb_intersection_volume(triangle_vertices_1[i], 
                                      plane_centers_1[i], 
                                      plane_normals_1[i], 
                                      triangle_vertices_2[i], 
                                      plane_centers_2[i], 
                                      plane_normals_2[i])

# -----------------------------
# Public API 
# -----------------------------

def obb_intersection_volumes(
    obb_first: wp.array,
    obb_second: wp.array,
    device: str | wp.context.Device,
    pairwise: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute volumes and intersection volumes between two sets of OBBs using Warp.

    Parameters
    ----------
    obb_first : array-like, shape (N, 8, 3)
        First set of OBBs.
    obb_second : array-like, shape (M, 8, 3)
        Second set of OBBs.
    device : str or Device
        The device to use for the computation. If not provided, will be inferred from the inputs.
    pairwise : bool
        When True, requires N == M and computes pairwise intersections.

    Returns
    -------
    vol1 : array-like, shape (N,)
    vol2 : array-like, shape (M,)
    inter : array-like, shape (N, M)
    """
    N = obb_first.shape[0]
    M = obb_second.shape[0]

    if pairwise and N != M:
        raise ValueError("pairwise requires obb_first and obb_second to have same shape.")

    # preprocessing
    tri_vertices_first_wp = wp.zeros((N, len(_TRIS), 3), dtype=wp.vec3, device=device)
    wp.launch(triangle_vertices_kernel, dim=N, inputs=[obb_first, tri_vertices_first_wp], device=device)

    tri_vertices_second_wp = wp.zeros((M, len(_TRIS), 3), dtype=wp.vec3, device=device)
    wp.launch(triangle_vertices_kernel, dim=M, inputs=[obb_second, tri_vertices_second_wp], device=device)

    obb_centers_first_wp = wp.empty(N, dtype=wp.vec3, device=device)
    obb_radii_first_wp = wp.empty(N, dtype=wp.float32, device=device)
    wp.launch(centers_radii_kernel, dim=N, inputs=[obb_first, obb_centers_first_wp, obb_radii_first_wp], device=device)

    obb_centers_second_wp = wp.empty(M, dtype=wp.vec3, device=device)
    obb_radii_second_wp = wp.empty(M, dtype=wp.float32, device=device)
    wp.launch(centers_radii_kernel, dim=M, inputs=[obb_second, obb_centers_second_wp, obb_radii_second_wp], device=device)

    plane_centers_first_wp = wp.empty((N, _NUM_PLANES), dtype=wp.vec3, device=device)
    plane_normals_first_wp = wp.empty((N, _NUM_PLANES), dtype=wp.vec3, device=device)
    wp.launch(planes_kernel, dim=N, inputs=[obb_first, obb_centers_first_wp, plane_centers_first_wp, plane_normals_first_wp], device=device)

    plane_centers_second_wp = wp.empty((M, _NUM_PLANES), dtype=wp.vec3, device=device)
    plane_normals_second_wp = wp.empty((M, _NUM_PLANES), dtype=wp.vec3, device=device)
    wp.launch(planes_kernel, dim=M, inputs=[obb_second, obb_centers_second_wp, plane_centers_second_wp, plane_normals_second_wp], device=device)

    box_volumes_first_wp = wp.empty((N,), dtype=wp.float32, device=device)
    wp.launch(box_volumes_kernel, dim=N, inputs=[tri_vertices_first_wp, obb_centers_first_wp, box_volumes_first_wp], device=device)

    box_volumes_second_wp = wp.empty((M,), dtype=wp.float32, device=device)
    wp.launch(box_volumes_kernel, dim=M, inputs=[tri_vertices_second_wp, obb_centers_second_wp, box_volumes_second_wp], device=device)

    # Intersections
    if pairwise:
        out_wp = wp.empty((N,), dtype=wp.float32, device=device)
        wp.launch(kernel_pairwise, dim=N, inputs=[tri_vertices_first_wp, 
                                                  plane_centers_first_wp, 
                                                  plane_normals_first_wp, 
                                                  obb_centers_first_wp, 
                                                  obb_radii_first_wp, 
                                                  tri_vertices_second_wp, 
                                                  plane_centers_second_wp, 
                                                  plane_normals_second_wp, 
                                                  obb_centers_second_wp, 
                                                  obb_radii_second_wp, 
                                                  out_wp], device=device)
        return box_volumes_first_wp, box_volumes_second_wp, out_wp
    else:
        out_wp = wp.empty((N, M), dtype=wp.float32, device=device)

        wp.launch(kernel_full, dim=(N, M), inputs=[tri_vertices_first_wp, 
                                                plane_centers_first_wp, 
                                                plane_normals_first_wp, 
                                                obb_centers_first_wp, 
                                                obb_radii_first_wp, 
                                                tri_vertices_second_wp, 
                                                plane_centers_second_wp, 
                                                plane_normals_second_wp, 
                                                obb_centers_second_wp, 
                                                obb_radii_second_wp, 
                                                out_wp], device=device)
        return box_volumes_first_wp, box_volumes_second_wp, out_wp