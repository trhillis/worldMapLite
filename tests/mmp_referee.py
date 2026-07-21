"""Exact-MMP referee for the octahedron slow tests.

Self-contained helpers to (a) insert two arbitrary surface points into a mesh
as vertices without changing any geodesic distance, and (b) query the exact
Mitchell-Mount-Papadimitriou solver in `pygeodesic`. Used only by the
`@pytest.mark.slow` octahedron cross-checks in test_manifolds.py.

The MMP solver measures distances between mesh VERTICES, so to check a distance
between two interior surface points we splice them in as new vertices first
(each insertion just re-triangulates the containing face - the surface, and
therefore every distance, is unchanged).
"""

from __future__ import annotations

import numpy as np


def face_point(V: np.ndarray, F: np.ndarray, face: int, bary) -> np.ndarray:
    """Ambient coordinates of barycentric point `bary` on face `face`."""
    return np.asarray(bary, dtype=float) @ V[F[face]]


def barycentric_on_face(V: np.ndarray, F: np.ndarray, face: int, p) -> np.ndarray:
    """Barycentric coordinates of ambient point p w.r.t. face (least squares).

    Caller guarantees p lies on the face's plane; no in-triangle check here.
    """
    A, B, C = V[F[face]]
    M = np.column_stack([B - A, C - A])  # (3, 2)
    uv, *_ = np.linalg.lstsq(M, np.asarray(p, dtype=float) - A, rcond=None)
    u, v = uv
    return np.array([1.0 - u - v, u, v])


def insert_point(V: np.ndarray, F: np.ndarray, face: int, bary):
    """Insert a surface point as a new vertex, splitting `face` into three.

    The polyhedral surface (and hence every geodesic distance) is unchanged;
    only the triangulation is refined. Returns (V2, F2, new_vertex_index).
    The three replacement faces are appended at the end of F2.
    """
    p = face_point(V, F, face, bary)
    V2 = np.vstack([V, p[None, :]])
    idx = len(V)
    a, b, c = (int(x) for x in F[face])
    F2 = np.vstack(
        [np.delete(F, face, axis=0), [[a, b, idx], [b, c, idx], [c, a, idx]]]
    ).astype(np.int32)
    return V2, F2, idx


def insert_two_points(V, F, fb1, fb2, tol: float = 1e-12):
    """Insert two surface points (face, bary) as vertices; handles shared faces.

    Returns (V2, F2, i1, i2). Face indices refer to the ORIGINAL F.
    """
    (f1, b1), (f2, b2) = fb1, fb2
    if f1 != f2:
        # Insert the higher face index first so the lower one is unshifted.
        if f1 > f2:
            V2, F2, i1 = insert_point(V, F, f1, b1)
            V2, F2, i2 = insert_point(V2, F2, f2, b2)
        else:
            V2, F2, i2 = insert_point(V, F, f2, b2)
            V2, F2, i1 = insert_point(V2, F2, f1, b1)
        return V2, F2, i1, i2
    # Same face: insert p1, then locate p2 in one of the three new sub-faces.
    p2 = face_point(V, F, f1, b2)
    V2, F2, i1 = insert_point(V, F, f1, b1)
    for sub in range(len(F2) - 3, len(F2)):
        bb = barycentric_on_face(V2, F2, sub, p2)
        if bb.min() >= -tol:
            V2, F2, i2 = insert_point(V2, F2, sub, np.clip(bb, 0.0, None))
            return V2, F2, i1, i2
    raise RuntimeError("second point not located in any sub-face")


def mmp_distance(V, F, i: int, j: int) -> float:
    """Exact MMP point-to-point distance between vertices i and j."""
    from pygeodesic import geodesic

    alg = geodesic.PyGeodesicAlgorithmExact(
        np.ascontiguousarray(V, dtype=float),
        np.ascontiguousarray(F, dtype=np.int32),
    )
    d, _path = alg.geodesicDistance(i, j)
    return float(d)
