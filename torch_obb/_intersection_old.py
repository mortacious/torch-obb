from __future__ import annotations
import numpy as np
import numba as nb
from typing import TYPE_CHECKING
from .util import check_batch_dim
if TYPE_CHECKING:
    from scipy.sparse import sparray

_NUM_PLANES = 6
_MAX_INTERSECTION_TRIS = 100


# triangles and planes in trimesh vertex order
_PLANES = np.array([[0, 1, 3, 2],
                    [0, 4, 5, 1],
                    [0, 2, 6, 4],
                    [1, 5, 7, 3],
                    [2, 3, 7, 6],
                    [4, 6, 7, 5]
                    ], dtype=np.uint8)

_TRIS = np.array([[1, 3, 0],
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
                  [7, 5, 6]], dtype=np.uint8)

k_epsilon = 1e-8
d_epsilon = 1e-3
a_epsilon = 1e-4


def _obb_centers(box_vertices: np.ndarray) -> np.ndarray:
    return np.mean(box_vertices, axis=-2) 


@nb.njit(nogil=True, fastmath=True, cache=True)
def _poly_volume(triangle_vertices: np.ndarray, center: np.ndarray) -> np.ndarray:
    box_vol = 0.0
    for i in range(triangle_vertices.shape[0]):
        v = triangle_vertices[i] - center
        area = np.dot(v[0], np.cross(v[1], v[2]))
        vol = np.abs(area) / 6.0
        box_vol += vol
    return box_vol


def _obb_volumes(triangle_vertices: np.ndarray, 
                 box_centers: np.ndarray) -> np.ndarray:
    v = (triangle_vertices - box_centers[:, np.newaxis, np.newaxis])
    c = np.cross(v[:, :, 1], v[:, :, 2])
    area = np.einsum('bij,bij->bi', v[:, :, 0], c)
    vol = np.sum(np.abs(area) / 6.0, axis=1)
    return vol


def _triangles_vertices(box_vertices: np.ndarray) -> np.ndarray:
    return box_vertices[:, _TRIS] # batched


@nb.njit(nogil=True, fastmath=True, inline='always')
def _get_normal(e0, e1):
    n = np.cross(e0, e1)
    denom = np.linalg.norm(n)
    # Ensure denom has the same dtype as input to avoid dtype upcast
    dtype_eps = n.dtype.type(k_epsilon)
    if denom < dtype_eps:
        denom = dtype_eps
    else:
        denom = n.dtype.type(denom)
    return n / denom


@nb.njit(nogil=True, fastmath=True, inline='always')
def _triangle_center(triangle_vertices: np.ndarray) -> np.ndarray:
    return np.sum(triangle_vertices, axis=0) / triangle_vertices.shape[0]


@nb.njit(nogil=True, fastmath=True, cache=True)
def _triangle_normal(triangle_vertices: np.ndarray) -> np.ndarray:
    center = _triangle_center(triangle_vertices)

    max_dist = -1.0
    n = np.empty(3, dtype=triangle_vertices.dtype)
    for i in range(2):
        for j in range(i+1, 3):
            dist = np.linalg.norm(np.cross(triangle_vertices[i] - center, triangle_vertices[j] - center))
            if dist > max_dist:
                n = _get_normal(triangle_vertices[i] - center, triangle_vertices[j] - center)
                max_dist = dist
    return n


def _planes_vertices(box_vertices: np.ndarray) -> np.ndarray:
    return box_vertices[:, _PLANES] # batched


def _planes_centers(plane_vertices: np.ndarray) -> np.ndarray:
    return plane_vertices.mean(axis=2)


def _planes_normals(planes_vertices: np.ndarray) -> np.ndarray:
    B = planes_vertices.shape[0]
    centers = np.expand_dims(_planes_centers(planes_vertices), -2)
    plane_center_distances = planes_vertices - centers
    
    e1 = plane_center_distances[:, :, :-1, np.newaxis]
    e2 = plane_center_distances[:, :, np.newaxis, 1:]

    normals = np.cross(e1, e2).reshape(B,-1, 9, 3)
    norms = np.linalg.norm(normals, axis=-1)
    c = norms.argmax(axis=-1, keepdims=True)
    normals = np.take_along_axis(normals, c[..., np.newaxis], axis=2)
    norms = np.take_along_axis(norms, c, axis=2)
    normals = (normals / norms[..., np.newaxis]).squeeze(2)
    return normals



def _obb_planes(box_vertices: np.ndarray, box_centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pv = _planes_vertices(box_vertices)
    pc = _planes_centers(pv)
    pn = _planes_normals(pv)
    c = np.vecdot(box_centers[:, np.newaxis] - pc, pn)
    pn[c < 0.0] *= -1.0
    return pn, pc


@nb.njit(nogil=True, fastmath=True, inline='always')
def _normalize_inline(u):
    n = np.linalg.norm(u)
    if n < k_epsilon:
        n = k_epsilon
    u /= n
    return u


@nb.njit(nogil=True, fastmath=True, cache=True)
def _check_coplanar(vertices: np.ndarray, plane_origin: np.ndarray, plane_normal: np.ndarray) -> bool:
    norm = np.linalg.norm(plane_normal)
    if norm < k_epsilon:
        norm = k_epsilon
    for i in range(vertices.shape[0]):
        d = np.abs(np.dot(plane_normal, vertices[i] - plane_origin) / norm)
        if d >= d_epsilon:
            return False
    return True


@nb.njit(nogil=True, fastmath=True, cache=True)
def _inside_plane(vertices: np.ndarray, plane_center: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    inside = np.empty(vertices.shape[0], dtype=np.bool)
    for i in range(vertices.shape[0]):
        c = np.dot((vertices[i] - plane_center), plane_normal)
        inside[i] = c >= 0.0
    return inside


@nb.njit(nogil=True, fastmath=True, cache=True)
def _plane_edge_intersection(plane_center, plane_normal, p0, p1):
    # The point of intersection can be parametrized
    # p = p0 + a (p1 - p0) where a in [0, 1]
    # We want to find a such that p is on plane
    # <p - ctr, n> = 0
    direc = p1 - p0
    direc = _normalize_inline(direc)
    p = (p0 + p1) / 2.0

    if np.abs(np.dot(direc, plane_normal)) >= d_epsilon:
        top = -1.0 * np.dot(p0 - plane_center, plane_normal)
        bot = np.dot(p1 - p0, plane_normal)
        a = top / bot
        p = p0 + a * (p1 - p0)
    return p


@nb.njit(nogil=True, fastmath=True, cache=True)
def _clip_tri_by_plane_one_out(plane_center, plane_normal, vout, vin1, vin2, out_tris, out_index):
    # point of intersection between plane and (vin1, vout)
    pint1 = _plane_edge_intersection(plane_center, plane_normal, vin1, vout)
    # point of intersection between plane and (vin2, vout)
    pint2 = _plane_edge_intersection(plane_center, plane_normal, vin2, vout)

    out_tris[out_index, 0] = vin2
    out_tris[out_index, 1] = pint1
    out_tris[out_index, 2] = pint2
    out_tris[out_index + 1, 0] = vin1
    out_tris[out_index + 1, 1] = pint1
    out_tris[out_index + 1, 2] = vin2

    return out_index + 2 # 2 triangles added


@nb.njit(nogil=True, fastmath=True, cache=True)
def _clip_tri_by_plane_two_out(plane_center, plane_normal, vout1, vout2, vin, out_tris, out_index):
    # point of intersection between plane and (vin1, vout)
    pint1 = _plane_edge_intersection(plane_center, plane_normal, vin, vout1)
    # point of intersection between plane and (vin2, vout)
    pint2 = _plane_edge_intersection(plane_center, plane_normal, vin, vout2)
    out_tris[out_index, 0] = vin
    out_tris[out_index, 1] = pint1
    out_tris[out_index, 2] = pint2

    return out_index + 1 # 1 triangle added


@nb.njit(nogil=True, fastmath=True, cache=True)
def _clip_tri_by_plane(triangle_vertices, plane_center, plane_normal, out_tris, out_index):
    is_coplanar = _check_coplanar(triangle_vertices, plane_center, plane_normal)
    #print("is_coplanar", is_coplanar)
    if is_coplanar:
        out_tris[out_index] = triangle_vertices
        return out_index + 1 # return the original vertices as nothing is intersecting

    insides = _inside_plane(triangle_vertices, plane_center, plane_normal)
    #print("insides", insides)

    if np.all(insides):
        out_tris[out_index] = triangle_vertices # fully inside so return the original vertices
        return out_index + 1
    # All out
    if not insides[0] and not insides[1] and not insides[2]:
        return out_index
    
    # single vertex is outside
    if insides[0] and insides[1] and not insides[2]:
        return _clip_tri_by_plane_one_out(plane_center, plane_normal, triangle_vertices[2], triangle_vertices[0], triangle_vertices[1], out_tris, out_index)
    if insides[0] and not insides[1] and insides[2]:
        return _clip_tri_by_plane_one_out(plane_center, plane_normal, triangle_vertices[1], triangle_vertices[0], triangle_vertices[2], out_tris, out_index)
    if not insides[0] and insides[1] and insides[2]:
        return _clip_tri_by_plane_one_out(plane_center, plane_normal, triangle_vertices[0], triangle_vertices[1], triangle_vertices[2], out_tris, out_index)
    
    # two vertices are outside
    if insides[0] and not insides[1] and not insides[2]:
        return _clip_tri_by_plane_two_out(plane_center, plane_normal, triangle_vertices[1], triangle_vertices[2], triangle_vertices[0], out_tris, out_index)
    if not insides[0] and not insides[1] and insides[2]:
        return _clip_tri_by_plane_two_out(plane_center, plane_normal, triangle_vertices[0], triangle_vertices[1], triangle_vertices[2], out_tris, out_index)
    if not insides[0] and insides[1] and not insides[2]:
        return _clip_tri_by_plane_two_out(plane_center, plane_normal, triangle_vertices[0], triangle_vertices[2], triangle_vertices[1], out_tris, out_index)

    # nothing to do
    return out_index


@nb.njit(nogil=True, fastmath=True, cache=True)
def _box_intersection(triangle_vertices, plane_centers, plane_normals):
    # prepare a buffer large enough to hold all the intersection triangles
    out_tris = np.empty((_MAX_INTERSECTION_TRIS, 3, 3), dtype=triangle_vertices.dtype)
    nout = triangle_vertices.shape[0]
    out_tris[:nout] = triangle_vertices # copy the initial triangles

    updated_tris = np.empty_like(out_tris) # create an intermediate buffer to store the new vertices
    #print("start nout", nout)

    for p in range(_NUM_PLANES):
        p_normal = plane_normals[p]  
        iupdated = 0
        for t in range(nout):
            # clip triangle by plane
            iupdated = _clip_tri_by_plane(out_tris[t], plane_centers[p], p_normal, updated_tris, iupdated)
        out_tris, updated_tris = updated_tris, out_tris # swap buffers
        nout = iupdated # get the number of total updated triangles
        #print("nout", nout)
    return out_tris[:nout]


@nb.njit(nogil=True, fastmath=True, cache=True)
def _any_coplanar(test_triangle: np.ndarray, triangles: np.ndarray) -> bool:
    test_normal = _triangle_normal(test_triangle)
    test_vertex = test_triangle[0]

    tris = triangles.reshape(-1, 3, 3)
    for i in range(tris.shape[0]):
        cp = _check_coplanar(tris[i], test_vertex, test_normal)
        if cp:
            return True
    return False



# @nb.njit(nogil=True, fastmath=True, cache=True)
# def _triangle_area(triangle_vertices: np.ndarray)  -> float:
#   v0 = triangle_vertices[0]
#   v1 = triangle_vertices[1]
#   v2 = triangle_vertices[2]
#   n = np.cross(v1 - v0, v2 - v0)
#   return np.linalg.norm(n) / 2.0


@nb.njit(nogil=True, fastmath=True, cache=True)
def _polyhedron_center(triangle_vertices: np.ndarray):
    center = np.zeros(3, dtype=triangle_vertices.dtype)
    num_tris = triangle_vertices.shape[0]

    # Find the center point of each face
    for t in range(num_tris):
        v0 = triangle_vertices[t, 0]
        v1 = triangle_vertices[t, 1]
        v2 = triangle_vertices[t, 2]

        face  = (v0 + v1 + v2) / 3.0
        center += face

    # Take the mean of the centers of all faces
    center /= num_tris
    return center


#@nb.njit(nogil=True, fastmath=True, cache=True)
def _box_intersection_polygone(tris_first,
                               center_first,
                               normal_first,
                               tris_second,
                               center_second,
                               normal_second):
    # Every triangle in one box will be compared to each plane in the other
    # box. There are 3 possible outcomes:
    # 1. If the triangle is fully inside, then it will
    #    remain as is.
    # 2. If the triagnle it is fully outside, it will be removed.
    # 3. If the triangle intersects with the (infinite) plane, it
    #    will be broken into subtriangles such that each subtriangle is full
    #    inside the plane and part of the intersecting tetrahedron.
    b1_intersect = _box_intersection(tris_first, center_second, normal_second)
    b2_intersect = _box_intersection(tris_second, center_first, normal_first)

    print("b1_intersect", b1_intersect.shape)
    print("b2_intersect", b2_intersect.shape)

    # If there are overlapping regions in Box2, remove any coplanar faces
    if b2_intersect.shape[0] > 0:
        # Identify if any triangles in Box2 are coplanar with Box1
        keep_mask = np.ones(b2_intersect.shape[0], dtype=np.bool)
        for b2 in range(b2_intersect.shape[0]):
            keep_mask[b2] = not _any_coplanar(b2_intersect[b2], b1_intersect)
        # Keep only the non coplanar triangles in Box2 - append them to the Box1 triangles.
        b1_intersect = np.concatenate((b1_intersect, b2_intersect[keep_mask]), axis=0)

    return b1_intersect


#@nb.njit(nogil=True, fastmath=True, cache=True)
def _obb_intersection_volume(tris_first,
                             center_first,
                             normal_first,
                             tris_second,
                             center_second,
                             normal_second):
    polygone = _box_intersection_polygone(tris_first, center_first, normal_first, tris_second, center_second, normal_second)

    # Initialize the vol and iou to 0.0 in case there are no triangles
    # in the intersecting shape.
    vol = 0.0

    if polygone.shape[0] > 0:
        # The intersecting shape is a polyhedron made up of the
        # triangular faces that are all now in box1_intersect.
        # Calculate the polyhedron center

        poly_center = _polyhedron_center(polygone)
        #print("poly_center", poly_center)
        vol = _poly_volume(polygone, poly_center)

    return vol


#@nb.njit(nogil=True, fastmath=True, cache=True)
def _obb_intersection_volumes(obbs_first,
                              volumes_first,
                              tris_first,
                              plane_centers_first,
                              normals_first,
                              centers_first,
                              radii_first,
                              obbs_second,
                              tris_second,
                              plane_centers_second,
                              normals_second,
                              centers_second,
                              radii_second):
    
    intersections_out = np.empty((tris_first.shape[0], tris_second.shape[0]), dtype=tris_first.dtype)
    for n in range(tris_first.shape[0]):
        b1_tris = tris_first[n]
        b1_plane_centers = plane_centers_first[n]
        b1_plane_normals = normals_first[n]
        b1_center = centers_first[n]
        b1_radius = radii_first[n]

        for m in range(tris_second.shape[0]):
            # if np.allclose(obbs_first[n], obbs_second[m]):
            #     print("skipping", n, m)
            #     # skip expensive computation of the bounding boxes are the same
            #     intersections_out[n, m] = volumes_first[n]
            #     continue

            b2_tris = tris_second[m]
            b2_plane_centers = plane_centers_second[m]
            b2_plane_normals = normals_second[m]
            b2_center = centers_second[m]
            b2_radius = radii_second[m]

            # bounding-sphere overlap check
            d = b1_center - b2_center
            d2 = np.dot(d, d)
            rsum2 = (b1_radius + b2_radius) ** 2


            # print("centers1[i]", b1_center)
            # print("centers2[j]", b2_center)
            # print("dx", d[0])
            # print("dy", d[1])
            # print("dz", d[2])
            # print("d2", d2)
            #print("i", n)
            #print("j", m)
            #print("rsum", d2, rsum2)
            if d2 > rsum2:
                print("skipping", n, m)
                intersections_out[n, m] = 0.0
                continue

            #print("args")
            #print("b1_tris", b1_tris)
            #print("b1_plane_centers", b1_plane_centers)
            #print("b1_plane_normals", b1_plane_normals)
            #print("b2_tris", b2_tris)
            #print("b2_plane_centers", b2_plane_centers)
            #print("b2_plane_normals", b2_plane_normals)
            vol = _obb_intersection_volume(b1_tris, 
                                                               b1_plane_centers, 
                                                               b1_plane_normals,
                                                               b2_tris, 
                                                               b2_plane_centers,
                                                               b2_plane_normals)
           # print("result vol", vol)
            intersections_out[n, m] = vol
    return intersections_out


#@nb.njit(nogil=True, fastmath=True, cache=True)
def _obb_intersection_volumes_sparse(obbs_first,
                              volumes_first,
                              tris_first,
                              plane_centers_first,
                              normals_first,
                              centers_first,
                              radii_first,
                              obbs_second,
                              tris_second,
                              plane_centers_second,
                              normals_second,
                              centers_second,
                              radii_second):
    
    #intersections_out = np.empty((tris_first.shape[0], tris_second.shape[0]), dtype=tris_first.dtype)
    N_out = []
    M_out = []
    values_out = []

    for n in range(tris_first.shape[0]):
        b1_tris = tris_first[n]
        b1_plane_centers = plane_centers_first[n]
        b1_plane_normals = normals_first[n]
        b1_center = centers_first[n]
        b1_radius = radii_first[n]

        for m in range(tris_second.shape[0]):
            if np.allclose(obbs_first[n], obbs_second[m]):
                # skip expensive computation of the bounding boxes are the same
                N_out.append(n)
                M_out.append(m)
                values_out.append(volumes_first[n])
                continue

            b2_tris = tris_second[m]
            b2_plane_centers = plane_centers_second[m]
            b2_plane_normals = normals_second[m]
            b2_center = centers_second[m]
            b2_radius = radii_second[m]

            # bounding-sphere overlap check
            d = b1_center - b2_center
            d2 = np.dot(d, d)
            rsum2 = (b1_radius + b2_radius) ** 2
            if d2 > rsum2:
                continue

            vol = _obb_intersection_volume(b1_tris, 
                                           b1_plane_centers, 
                                           b1_plane_normals,
                                           b2_tris, 
                                           b2_plane_centers,
                                           b2_plane_normals)
            if vol > 0:
                N_out.append(n)
                M_out.append(m)
                values_out.append(vol)
    return np.asarray(values_out), (np.asarray(N_out), np.asarray(M_out))


#@nb.njit(nogil=True, fastmath=True, cache=True)
def _obb_intersection_volumes_pairwise(obbs_first,
                                       volumes_first,
                                       tris_first,
                                       plane_centers_first,
                                       normals_first,
                                       centers_first,
                                       radii_first,
                                       obbs_second,
                                       tris_second,
                                       plane_centers_second,
                                       normals_second,
                                       centers_second,
                                       radii_second):
    
    intersections_out = np.zeros((centers_first.shape[0]), dtype=centers_first.dtype)

    for n in range(tris_first.shape[0]):
        if np.allclose(obbs_first[n], obbs_second[n]):
            # skip expensive computation of the bounding boxes are the same
            intersections_out[n] = volumes_first[n]
            continue

        b1_tris = tris_first[n]
        b1_plane_centers = plane_centers_first[n]
        b1_plane_normals = normals_first[n]
        b1_center = centers_first[n]
        b1_radius = radii_first[n]

        b2_tris = tris_second[n]
        b2_plane_centers = plane_centers_second[n]
        b2_plane_normals = normals_second[n]
        b2_center = centers_second[n]
        b2_radius = radii_second[n]

        # bounding-sphere overlap check
        d = b1_center - b2_center
        d2 = np.dot(d, d)
        rsum2 = (b1_radius + b2_radius) ** 2
        if d2 > rsum2:
            continue

        intersections_out[n] = _obb_intersection_volume(b1_tris, 
                                                        b1_plane_centers, 
                                                        b1_plane_normals,
                                                        b2_tris, 
                                                        b2_plane_centers,
                                                        b2_plane_normals)
        
    return intersections_out


@nb.njit(nogil=True, fastmath=True, cache=True)
def _obb_intersection_volumes_sparse_pairwise(obbs_first,
                                       volumes_first,
                                       tris_first,
                                       plane_centers_first,
                                       normals_first,
                                       centers_first,
                                       radii_first,
                                       obbs_second,
                                       tris_second,
                                       plane_centers_second,
                                       normals_second,
                                       centers_second,
                                       radii_second):
         
    N_out = []
    values_out = []

    for n in range(tris_first.shape[0]):
        if np.allclose(obbs_first[n], obbs_second[n]):
            # skip expensive computation of the bounding boxes are the same
            N_out.append(n)
            values_out.append(volumes_first[n])
            continue

        b1_tris = tris_first[n]
        b1_plane_centers = plane_centers_first[n]
        b1_plane_normals = normals_first[n]
        b1_center = centers_first[n]
        b1_radius = radii_first[n]

        b2_tris = tris_second[n]
        b2_plane_centers = plane_centers_second[n]
        b2_plane_normals = normals_second[n]
        b2_center = centers_second[n]
        b2_radius = radii_second[n]

        # bounding-sphere overlap check
        d = b1_center - b2_center
        d2 = np.dot(d, d)
        rsum2 = (b1_radius + b2_radius) ** 2
        if d2 > rsum2:
            continue

        vol = _obb_intersection_volume(b1_tris, 
                                       b1_plane_centers, 
                                       b1_plane_normals,
                                       b2_tris, 
                                       b2_plane_centers,
                                       b2_plane_normals)
        if vol > 0:
            N_out.append(n)
            values_out.append(vol)
    return np.asarray(values_out), (np.asarray(N_out),)


def obb_intersection_volumes(obb_first: np.ndarray,
                             obb_second: np.ndarray,
                             pairwise: bool = False,
                             sparse: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray | "sparray"]:
    """Compute volumes and intersection volumes between two sets of OBBs.

    Parameters
    ----------
    obb_first : np.ndarray
        Array of oriented bounding boxes with shape `(N, 8, 3)` or `(8, 3)`.
        Each OBB is represented by its 8 corner vertices in the expected
        vertex ordering used internally.
    obb_second : np.ndarray
        Array of oriented bounding boxes with shape `(M, 8, 3)` or `(8, 3)`.
    pairwise : bool, optional
        If False (default), computes the full `N x M` intersections between
        every OBB in `obb_first` and every OBB in `obb_second`.
        If True, requires `N == M` and computes pairwise intersections
        between `obb_first[i]` and `obb_second[i]`, returning a `(N,)` vector.
    sparse : bool, optional
        If True, returns sparse outputs (`scipy.sparse.sparray`), which can be
        beneficial when most pairs do not intersect. For pairwise+sparse,
        the returned intersection volume is a `(N,)` CSR array.

    Returns
    -------
    first_volumes : np.ndarray
        Volumes of `obb_first` as `(N,)`.
    second_volumes : np.ndarray
        Volumes of `obb_second` as `(M,)` when `pairwise=False`, or `(N,)` when
        `pairwise=True`.
    intersection_volumes : np.ndarray | sparray
        Intersection volumes. Shape `(N, M)` when `pairwise=False`, `(N,)` when
        `pairwise=True`. Dense numpy array by default; sparse CSR array when
        `sparse=True`.
    """
    obb_first = check_batch_dim(obb_first, 3)
    obb_second = check_batch_dim(obb_second, 3)

    triangle_vertices_first = _triangles_vertices(obb_first)
    triangle_vertices_second = _triangles_vertices(obb_second)
    
    # bounding box centers
    obb_centers_first = _obb_centers(obb_first)
    obb_centers_second = _obb_centers(obb_second)

    # bounding sphere radii
    radii_first = np.linalg.norm(obb_first - obb_centers_first[:, None, :], axis=-1).max(axis=-1)
    radii_second = np.linalg.norm(obb_second - obb_centers_second[:, None, :], axis=-1).max(axis=-1)

    # plane normals and centers
    plane_normals_first, plane_centers_first = _obb_planes(obb_first, obb_centers_first)
    plane_normals_second, plane_centers_second = _obb_planes(obb_second, obb_centers_second)

    # box volumes
    box_volumes_first = _obb_volumes(triangle_vertices_first, obb_centers_first)
    box_volumes_second = _obb_volumes(triangle_vertices_second, obb_centers_second)


    # print("radii_first", radii_first)
    # print("radii_second", radii_second)

    if sparse:
        from scipy.sparse import csr_array
        if not pairwise:
            values_out, coords_out = _obb_intersection_volumes_sparse(obb_first,
                                                        box_volumes_first,
                                                        triangle_vertices_first, 
                                                        plane_centers_first, 
                                                        plane_normals_first, 
                                                        obb_centers_first,
                                                        radii_first,
                                                        obb_second,
                                                        triangle_vertices_second, 
                                                        plane_centers_second,
                                                        plane_normals_second,
                                                        obb_centers_second,
                                                        radii_second)
            intersections_out = csr_array((values_out, coords_out), shape=(obb_first.shape[0], obb_second.shape[0]), dtype=obb_first.dtype)
        else:
            assert obb_first.shape[0] == obb_second.shape[0]
            values_out, coords_out = _obb_intersection_volumes_sparse_pairwise(obb_first,
                                                                box_volumes_first,
                                                                triangle_vertices_first, 
                                                                plane_centers_first, 
                                                                plane_normals_first, 
                                                                obb_centers_first,
                                                                radii_first,
                                                                obb_second,
                                                                triangle_vertices_second, 
                                                                plane_centers_second,
                                                                plane_normals_second,
                                                                obb_centers_second,
                                                                radii_second)
            intersections_out = csr_array((values_out, coords_out), shape=(obb_first.shape[0],), dtype=obb_first.dtype)
    else:
        if not pairwise:
            intersections_out = _obb_intersection_volumes(obb_first,
                                                        box_volumes_first,
                                                        triangle_vertices_first, 
                                                        plane_centers_first, 
                                                        plane_normals_first, 
                                                        obb_centers_first,
                                                        radii_first,
                                                        obb_second,
                                                        triangle_vertices_second, 
                                                        plane_centers_second,
                                                        plane_normals_second,
                                                        obb_centers_second,
                                                        radii_second)
        else:
            assert obb_first.shape[0] == obb_second.shape[0]
            intersections_out = _obb_intersection_volumes_pairwise(obb_first,
                                                                box_volumes_first,
                                                                triangle_vertices_first, 
                                                                plane_centers_first, 
                                                                plane_normals_first, 
                                                                obb_centers_first,
                                                                radii_first,
                                                                obb_second,
                                                                triangle_vertices_second, 
                                                                plane_centers_second,
                                                                plane_normals_second,
                                                                obb_centers_second,
                                                                radii_second)

    return box_volumes_first, box_volumes_second, intersections_out


def obb_overlaps(obb_first: np.ndarray,
                 obb_second: np.ndarray,
                 pairwise: bool = False,
                 sparse: bool = False) -> np.ndarray | sparray:
    """Compute directional overlap ratios between OBBs.

    The overlap ratio from A to B is defined as `intersection_volume / volume(A)`.
    This function returns a tuple `(overlap_A_to_B, overlap_B_to_A)`.

    Parameters
    ----------
    obb_first : np.ndarray
        OBBs shaped `(N, 8, 3)` or `(8, 3)`.
    obb_second : np.ndarray
        OBBs shaped `(M, 8, 3)` or `(8, 3)`.
    pairwise : bool, optional
        When True, compute pairwise overlaps `(N,)`. When False, compute all
        pair overlaps `(N, M)`.
    sparse : bool, optional
        When True, return sparse arrays for the ratios.

    Returns
    -------
    overlap_A_to_B, overlap_B_to_A : np.ndarray | sparray
        Overlap ratios with shape `(N, M)` or `(N,)` (pairwise). Types mirror
        the `sparse` flag.
    """
    first_volume, second_volume, overlap_volume = obb_intersection_volumes(obb_first, 
                                                                           obb_second,
                                                                           pairwise=pairwise,
                                                                           sparse=sparse)
    if not pairwise:
        first_volume = first_volume[:, None]
    volumes = overlap_volume / (first_volume + 1e-9), overlap_volume / (second_volume + 1e-9)

    if sparse:
        return volumes[0].tocsr(), volumes[1].tocsr()
    return volumes


def obb_ious(obb_first: np.ndarray,
             obb_second: np.ndarray,
             pairwise: bool = False,
             sparse: bool = False) -> np.ndarray | sparray:
    """Compute IoU between OBBs.

    IoU is defined as `intersection / (vol(A) + vol(B) - intersection)`.

    Parameters
    ----------
    obb_first : np.ndarray
        OBBs shaped `(N, 8, 3)` or `(8, 3)`.
    obb_second : np.ndarray
        OBBs shaped `(M, 8, 3)` or `(8, 3)`.
    pairwise : bool, optional
        When True, compute pairwise IoUs `(N,)`. When False, compute all-pairs
        IoUs `(N, M)`.
    sparse : bool, optional
        When True, return a sparse CSR array.

    Returns
    -------
    ious : np.ndarray | sparray
        IoUs with shape `(N, M)` or `(N,)` (pairwise). Type depends on `sparse`.
    """
    first_volume, second_volume, overlap_volume = obb_intersection_volumes(obb_first, 
                                                                        obb_second,
                                                                        pairwise=pairwise,
                                                                        sparse=sparse)
    if not pairwise:
        first_volume = first_volume[:, None]
    ious = overlap_volume / (first_volume + second_volume - overlap_volume + 1e-9) # prevent division by zero

    if sparse:
        return ious.tocsr()
    return ious


def obb_planes(obbs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute plane normals and centers for each face of each OBB.

    Parameters
    ----------
    obbs : np.ndarray
        OBBs shaped `(N, 8, 3)` or `(8, 3)`.

    Returns
    -------
    normals, centers : tuple[np.ndarray, np.ndarray]
        - normals: `(N, 6, 3)` unit normals pointing outward for each face.
        - centers: `(N, 6, 3)` centers of each face.
    """
    obbs = check_batch_dim(obbs, 3)
    return _obb_planes(obbs, _obb_centers(obbs))


def obb_centers(obbs: np.ndarray) -> np.ndarray:
    """Compute the center point of each OBB.

    Parameters
    ----------
    obbs : np.ndarray
        OBBs shaped `(N, 8, 3)` or `(8, 3)`.

    Returns
    -------
    centers : np.ndarray
        Centers shaped `(N, 3)`.
    """
    obbs = check_batch_dim(obbs, 3)
    return np.mean(obbs, axis=-2)