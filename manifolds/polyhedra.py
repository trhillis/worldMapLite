"""Polyhedral surfaces (octahedron, icosahedron): curvature you can put your
finger on, and the first real algorithm.

The spaces
----------
The surface of a regular octahedron (8 equilateral triangles, 6 vertices) and
of a regular icosahedron (20 triangles, 12 vertices), both scaled to unit edge
length so your numbers are comparable across the two solids and across
students. Each face is a flat piece of the plane; ALL the curvature is
concentrated at the vertices as angle deficit: the faces around an octahedron
vertex contribute 4 * 60 = 240 degrees, which is 120 degrees (2 pi / 3) short
of a full turn; an icosahedron vertex has 5 * 60 = 300 degrees, deficit pi / 3.
Summed over vertices both give exactly 4 pi (Descartes / Gauss-Bonnet for a
sphere-like surface) - your test suite should check this from the mesh data.

ONE implementation, TWO meshes. This class is deliberately concrete about
everything except the geodesic methods, and the two solids are module-level
mesh constants plus factory functions - there is no `Octahedron` class to
hang octahedron-specific code on. If you find yourself writing
`if self.n_faces == 8`, you are fighting the design, and the icosahedron test
run will fail you anyway (see below).

Distance (what you implement): unfolding enumeration
----------------------------------------------------
A shortest path crosses a sequence of faces. Unfold that face sequence into
the plane (each face rotated flat across the shared edge) and the path becomes
a STRAIGHT SEGMENT. So:

    d(p, q) = min over valid face sequences of the straight-line distance
              between p and q in that sequence's unfolding,

where "valid" means the segment actually crosses every intermediate unfolded
edge strictly inside the open edge (endpoints excluded). Enumerate sequences
by BFS from the source face; unfold incrementally; keep the best.

Non-negotiable facts to build on (all verified in prototyping):
  * Exactness: done right, this agrees with the independent exact MMP solver
    (pygeodesic) to ~1e-15 on both solids. It is the strongest correctness
    anchor in the whole project - use it.
  * Shortest paths never pass through a vertex (positive deficit makes a
    vertex-corner path shortenable on either side; measured safety margin
    of the best vertex-through route vs the true distance is comfortably
    positive on random pairs). Exclude vertex-touching segments - but treat
    NEAR-vertex ("grazing") segments with a tolerance, and know that the
    tolerance sets how close to a vertex you can trust your answers.
  * On a convex polyhedron a shortest path visits each face at most once, so
    face sequences never revisit a face. This bounds the enumeration depth by
    the face count - your termination guarantee. (Not true on non-convex
    surfaces; both our solids are convex.)
  * Prune by PATH LENGTH against the best candidate so far (the distance from
    the unfolded source to the entry edge is a valid lower bound), never by a
    hardcoded face-count cap. A depth cap tuned on the octahedron is exactly
    the bug the icosahedron run exposes.

Why you must run BOTH meshes with the SAME code: shortest paths on the
octahedron cross few faces, and several wrong enumerations (too-shallow
depth caps most famously) still produce correct octahedron answers - they
pass "by luck" - and then fail on the icosahedron, where paths cross more
faces, more sequences graze vertices, and a naive enumeration also blows up
combinatorially. An octahedron pass
plus an icosahedron fail on the same code means the octahedron pass proved
nothing. That lesson is the reason the second solid is in the roster.

Point representation
--------------------
A chart point is `(face_index, b0, b1, b2)` packed in one float row of shape
(4,): the face index (an integer stored as float) plus FULL barycentric
coordinates w.r.t. that face's three vertices, in `faces[f]` order,
nonnegative, summing to 1. Redundant on purpose: the sum-to-1 constraint is a
checkable invariant, no vertex is privileged, and barycentric coordinates
transfer to ANY placement of the triangle - in particular to unfolded copies
(`bary @ unfolded_triangle_vertices`), which is exactly what your algorithm
needs at every step. Use `make_points` / `split_points`; never index columns
by hand.

Boundary points are shared: a point on an edge has one valid representation
per adjacent face (a vertex, one per incident face). Contract: `distance` and
`embed` must return identical results whichever representation is used - and
that is a test worth writing.

Sampling
--------
Pick a face with probability proportional to its area (equal here, but write
it area-weighted - the code then works for ANY mesh), then a uniform point in
the triangle by the reflection trick: u1, u2 ~ U[0, 1]; if u1 + u2 > 1
replace by (1 - u1, 1 - u2); barycentric (1 - u1 - u2, u1, u2). Exactly
uniform - no sqrt tricks needed.

Isometries for the invariance test: the solid's symmetry group (48 elements
for the octahedron, 120 for the icosahedron); coordinate reflections and
90-degree z-rotation (octahedron) are easy concrete cases.
"""

from __future__ import annotations

import numpy as np

from .base import Manifold

_SQ2 = np.sqrt(2.0)
_PHI = (1.0 + np.sqrt(5.0)) / 2.0

#: Regular octahedron, unit edge: vertices +-e_i / sqrt(2). Order: +x -x +y -y +z -z.
OCTAHEDRON_VERTICES = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=float) / _SQ2
OCTAHEDRON_FACES = np.array(
    [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
     [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]], dtype=np.int64)

#: Regular icosahedron, unit edge: cyclic permutations of (0, +-1, +-phi) / 2.
ICOSAHEDRON_VERTICES = np.array(
    [[-1, _PHI, 0], [1, _PHI, 0], [-1, -_PHI, 0], [1, -_PHI, 0],
     [0, -1, _PHI], [0, 1, _PHI], [0, -1, -_PHI], [0, 1, -_PHI],
     [_PHI, 0, -1], [_PHI, 0, 1], [-_PHI, 0, -1], [-_PHI, 0, 1]],
    dtype=float) / 2.0
ICOSAHEDRON_FACES = np.array(
    [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
     [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
     [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
     [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]],
    dtype=np.int64)


class PolyhedralSurface(Manifold):
    """Geodesics on a closed convex triangulated surface; the mesh is DATA.

    Chart points: (face, b0, b1, b2) rows, shape (4,) each - see the module
    docstring. The constructor and the bookkeeping helpers below are provided;
    the geodesic methods are the exercise.
    """

    intrinsic_dim = 2
    ambient_dim = 3
    point_shape = (4,)
    ambient_metric_is_induced = True

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, name: str):
        V = np.asarray(vertices, dtype=float)
        F = np.asarray(faces, dtype=np.int64)
        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError(f"vertices must be (n, 3), got {V.shape}")
        if F.ndim != 2 or F.shape[1] != 3:
            raise ValueError(f"faces must be (m, 3), got {F.shape}")
        if F.min() < 0 or F.max() >= len(V):
            raise ValueError("face indices out of range")
        tri = V[F]
        areas = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        if np.any(areas < 1e-12):
            raise ValueError("mesh contains a degenerate (zero-area) face")
        edges = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]),
                        axis=1)
        uniq, counts = np.unique(edges, axis=0, return_counts=True)
        if not np.all(counts == 2):
            raise ValueError("mesh is not a closed surface "
                             "(every edge must border exactly 2 faces)")
        self.vertices = V
        self.faces = F
        self.face_areas = areas
        self.name = name

    # ---------------- provided bookkeeping (not part of the exercise) ------

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def triangle(self, f: int) -> np.ndarray:
        """(3, 3) ambient coordinates of face f's vertices, in faces[f] order."""
        return self.vertices[self.faces[int(f)]]

    def make_points(self, face, bary) -> np.ndarray:
        """Pack (n,) face indices + (n, 3) barycentric coords into (n, 4) points."""
        face = np.asarray(face)
        bary = np.asarray(bary, dtype=float)
        return np.column_stack([face.astype(float), bary])

    def split_points(self, points) -> tuple[np.ndarray, np.ndarray]:
        """Unpack (n, 4) points -> ((n,) int face indices, (n, 3) barycentric)."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        return pts[:, 0].astype(np.int64), pts[:, 1:4]

    def _validate_points(self, points: np.ndarray) -> None:
        f, b = points[:, 0], points[:, 1:4]
        bad = (f != np.round(f)) | (f < 0) | (f >= self.n_faces)
        if bad.any():
            i = int(np.argmax(bad))
            raise ValueError(
                f"{self.name}: row {i} has face index {f[i]!r}; must be an "
                f"integer in [0, {self.n_faces})")
        bad = (b < -1e-9).any(axis=1) | (np.abs(b.sum(axis=1) - 1.0) > 1e-9)
        if bad.any():
            i = int(np.argmax(bad))
            raise ValueError(
                f"{self.name}: row {i} barycentric {b[i]} invalid (must be "
                f">= 0 and sum to 1). If these came from your own arithmetic, "
                f"renormalize explicitly rather than accumulating drift.")

    # ---------------------------- the exercise ----------------------------

    def _sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Area-weighted face choice + in-triangle reflection trick.

        Write it area-weighted via self.face_areas even though the regular
        solids have equal areas - equal-area is a property of the mesh, not
        of your algorithm.
        """
        raise NotImplementedError

    def _distance(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Min over valid unfolding sequences (module docstring has the spec).

        Genuinely per-pair: loop over the batch here. Structure that worked
        in prototyping: BFS states carry (current face, its unfolded 2D
        vertex positions, faces visited); expanding across an edge places the
        neighbor's third vertex by circle-circle intersection on the far side
        of the edge. Validate candidates by explicit segment/edge crossing
        tests (crossing parameter strictly inside the open edge, crossings in
        increasing order along the segment). Same-face pairs: the in-plane
        straight segment is already the answer.

        Performance is part of correctness here: prune against the incumbent
        best (distance from source to the entry edge is a lower bound) or
        the icosahedron run will crawl - the unpruned enumeration blows up
        combinatorially on the finer mesh.
        """
        raise NotImplementedError

    def _embed(self, points: np.ndarray) -> np.ndarray:
        """Barycentric to ambient: b @ triangle(f), one einsum for the batch."""
        raise NotImplementedError


def octahedron() -> PolyhedralSurface:
    """Unit-edge regular octahedron as a PolyhedralSurface."""
    return PolyhedralSurface(OCTAHEDRON_VERTICES, OCTAHEDRON_FACES, "octahedron")


def icosahedron() -> PolyhedralSurface:
    """Unit-edge regular icosahedron as a PolyhedralSurface."""
    return PolyhedralSurface(ICOSAHEDRON_VERTICES, ICOSAHEDRON_FACES, "icosahedron")
