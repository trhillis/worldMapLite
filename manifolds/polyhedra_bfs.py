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

DIAGNOSTIC VARIANT
-------------------
This file is an exact copy of polyhedra.py except for the search strategy in
_pair_distance: instead of a best-first (Dijkstra/A*-style) priority-queue
search, it uses a plain FIFO breadth-first traversal - closer to the
README's literal "BFS over face sequences" description - still pruned by the
same admissible source-to-portal lower bound, just without the priority-queue
ordering (so no early "stop the whole search" break is safe; each dequeued
state is instead individually checked against the incumbent best before it's
expanded). Everything else (geometry helpers, validation, vertex-shortcut
termination, sampling, embedding) is byte-for-byte identical to polyhedra.py.
Existence purpose: isolate whether an icosahedron correctness gap traces back
to the search STRATEGY (priority-queue vs. plain BFS) or to something else
(geometry/validation) that both strategies would share.
"""

from __future__ import annotations

from collections import deque
from functools import cached_property

import numpy as np

from .base import Manifold

_SQ2 = np.sqrt(2.0)
_PHI = (1.0 + np.sqrt(5.0)) / 2.0

# Local vertex-index pairs (into faces[f]) for a triangle's three edges.
_LOCAL_EDGES = ((0, 1), (1, 2), (2, 0))

# Tolerances for the unfolding-candidate crossing test (README section 5):
# ~1e-8 is the stated practical floor for how close to a vertex a grazing
# crossing can be trusted.
_TOL_S = 1e-8
_TOL_T = 1e-8

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


class PolyhedralSurfaceBFS(Manifold):
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

    @cached_property
    def _edge_to_faces(self) -> dict[tuple[int, int], list[int]]:
        """Undirected mesh edge (u < v) -> the (exactly 2) incident face ids."""
        m: dict[tuple[int, int], list[int]] = {}
        for f in range(self.n_faces):
            a, b, c = (int(x) for x in self.faces[f])
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                m.setdefault(key, []).append(f)
        return m

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
        probs = self.face_areas / self.face_areas.sum()
        face = rng.choice(self.n_faces, size=n, p=probs)
        u1 = rng.random(n)
        u2 = rng.random(n)
        flip = (u1 + u2) > 1.0
        u1[flip], u2[flip] = 1.0 - u1[flip], 1.0 - u2[flip]
        bary = np.column_stack([1.0 - u1 - u2, u1, u2])
        return self.make_points(face, bary)

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
        f_p, b_p = self.split_points(p)
        f_q, b_q = self.split_points(q)
        out = np.empty(len(p))
        for i in range(len(p)):
            out[i] = self._pair_distance(
                int(f_p[i]), b_p[i], int(f_q[i]), b_q[i])
        return out

    def _embed(self, points: np.ndarray) -> np.ndarray:
        """Barycentric to ambient: b @ triangle(f), one einsum for the batch."""
        face, bary = self.split_points(points)
        tri = self.vertices[self.faces[face]]  # (n, 3, 3)
        return np.einsum("ni,nij->nj", bary, tri)

    # ------------------ private geodesic-unfolding machinery ------------------
    # Not part of the public API; supports _distance's per-pair unfolding
    # search. See manifolds/README.md section 5 for the algorithm this
    # implements.

    @staticmethod
    def _local2d(tri: np.ndarray) -> np.ndarray:
        """Isometric flattening of one face's ambient (3, 3) vertices to 2D.

        Origin at vertex A, x-axis along AB; C placed by its true edge
        lengths. Affine + isometric, so bary @ verts2d gives the exact 2D
        position of any point on the face.
        """
        A, B, C = tri
        ab = B - A
        len_ab = np.linalg.norm(ab)
        ex = ab / len_ab
        v = C - A
        xc = v @ ex
        perp = v - xc * ex
        yc = np.linalg.norm(perp)
        return np.array([[0.0, 0.0], [len_ab, 0.0], [xc, yc]])

    @staticmethod
    def _unfold_third(A2: np.ndarray, B2: np.ndarray,
                       rA: float, rB: float) -> tuple[np.ndarray, np.ndarray]:
        """Circle-circle intersection: the two candidates for a neighbor
        face's third vertex, given the shared edge's fixed 2D endpoints and
        the (flat-intrinsic, i.e. ambient) distances from each to it."""
        d_vec = B2 - A2
        d = np.linalg.norm(d_vec)
        ex = d_vec / d
        ey = np.array([-ex[1], ex[0]])
        a = (d * d + rA * rA - rB * rB) / (2.0 * d)
        h = np.sqrt(max(rA * rA - a * a, 0.0))  # clamp: fp noise near h=0
        foot = A2 + a * ex
        return foot + h * ey, foot - h * ey

    @staticmethod
    def _side(P: np.ndarray, A2: np.ndarray, B2: np.ndarray) -> float:
        """Signed area of (A2, B2, P): which side of line A2->B2 P is on."""
        v, w = B2 - A2, P - A2
        return v[0] * w[1] - v[1] * w[0]

    def _expand(self, verts2d: np.ndarray, face: int, ij: tuple[int, int],
                neighbor: int) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Unfold `neighbor` across local edge `ij` of `face`'s current 2D
        placement. Returns (neighbor's (3, 2) unfolded vertices, in
        faces[neighbor] order, so barycentric transfer stays valid; the
        crossed portal segment (A2, B2))."""
        i, j = ij
        va_id = int(self.faces[face, i])
        vb_id = int(self.faces[face, j])
        A2, B2 = verts2d[i], verts2d[j]
        D2 = verts2d[3 - i - j]  # face's own opposite (non-shared) vertex

        nf_ids = self.faces[neighbor]
        vc_id = int(nf_ids[~np.isin(nf_ids, [va_id, vb_id])][0])
        rA = np.linalg.norm(self.vertices[va_id] - self.vertices[vc_id])
        rB = np.linalg.norm(self.vertices[vb_id] - self.vertices[vc_id])
        c1, c2 = self._unfold_third(A2, B2, rA, rB)
        # Unfold outward: opposite side of AB from the current face's D2.
        C2 = c1 if self._side(c1, A2, B2) * self._side(D2, A2, B2) < 0 else c2

        new_verts2d = np.empty((3, 2))
        for slot, vid in enumerate(nf_ids):
            vid = int(vid)
            if vid == va_id:
                new_verts2d[slot] = A2
            elif vid == vb_id:
                new_verts2d[slot] = B2
            else:
                new_verts2d[slot] = C2
        return new_verts2d, (A2, B2)

    @staticmethod
    def _validate(p2d: np.ndarray, q2d: np.ndarray,
                   portals: tuple[tuple[np.ndarray, np.ndarray], ...]) -> float | None:
        """Check the straight segment p2d->q2d crosses every portal strictly
        inside its open edge, in increasing order; return its length or None.

        prev_t starts at -_TOL_T (not 0) so a source sitting exactly on its
        own face's boundary can validate a path exiting immediately through
        that edge (t ~ 0) - required for on-edge representation invariance
        (README section 5, "on-edge sources"). The final portal is
        symmetrically allowed t up to 1 + _TOL_T for a target on its entry
        edge.

        A source or target that sits exactly ON A VERTEX (not just an edge
        interior) is a sharper version of the same issue: the crossing
        parameter along the portal (s), not just along the segment (t),
        snaps to exactly 0 or 1 - because the source/target coincides with
        one of the portal's own endpoints, not just with a point on it. That
        is legitimate (the path starts/ends exactly at that corner; it is
        not an intermediate pass-through, which is the case the mid-path
        grazing tolerance _TOL_S exists to exclude - a straight line that
        merely passes near/through an unrelated vertex somewhere along the
        way is not resolved by unfolding alone).

        Crucially, this touching is NOT confined to the first/last portal:
        if the source (or target) vertex has degree > 3, an unfolding
        sequence can legitimately cross SEVERAL consecutive portals that all
        still have that same fixed vertex as an endpoint - each visited face
        is a different neighbor around the source's own vertex fan, entered
        before the path has made any real progress (t == 0 at every one of
        them, not just the first). Measured on the icosahedron (vertex
        degree 5): a 2-hop candidate legitimately has BOTH its portals touch
        p2d, and rejecting the second one (because it isn't index 0)
        produced a wrong, too-long answer - the octahedron (vertex degree
        4) never exercises this because its shortest paths rarely need more
        than one hop around a shared vertex before departing. So: check
        every portal, at any index, for touching p2d or q2d, and accept
        those unconditionally without perturbing the monotonic-t tracking
        (they represent zero real progress along the path, not a step of
        it) - the strict s/t checks below apply only to genuine interior
        crossings.
        """
        r = q2d - p2d
        n = len(portals)
        prev_t = -_TOL_T
        for i, (A, B) in enumerate(portals):
            e = B - A
            denom = r[0] * e[1] - r[1] * e[0]
            w = A - p2d
            touches_p = (np.hypot(*(A - p2d)) < 1e-9
                         or np.hypot(*(B - p2d)) < 1e-9)
            touches_q = (np.hypot(*(A - q2d)) < 1e-9
                         or np.hypot(*(B - q2d)) < 1e-9)
            if abs(denom) < 1e-13:
                if touches_p and touches_q:
                    # The portal is collinear with p2d->q2d because it IS
                    # the direct edge connecting them (e.g. source and
                    # target are the two endpoints of the same mesh edge).
                    continue
                return None  # truly parallel but offset: no crossing
            t = (w[0] * e[1] - w[1] * e[0]) / denom
            s = (w[0] * r[1] - w[1] * r[0]) / denom
            if touches_p or touches_q:
                continue  # zero real progress at this hop; see docstring
            if not (_TOL_S <= s <= 1.0 - _TOL_S):
                return None  # crosses too near an unrelated vertex, or off the segment
            upper = 1.0 + _TOL_T if i == n - 1 else 1.0 - _TOL_T
            if not (t > prev_t and t <= upper):
                return None
            prev_t = t
        return float(np.hypot(r[0], r[1]))

    @staticmethod
    def _point_segment_distance(p: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
        """Euclidean distance from p to the closed segment AB (a lower bound
        for the length of any path from p through a portal on AB)."""
        e = B - A
        L2 = e @ e
        if L2 < 1e-30:
            return float(np.linalg.norm(p - A))
        t = np.clip(((p - A) @ e) / L2, 0.0, 1.0)
        return float(np.linalg.norm(p - (A + t * e)))

    def _pair_distance(self, f_p: int, b_p: np.ndarray,
                        f_q: int, b_q: np.ndarray) -> float:
        """Plain FIFO breadth-first search over unfolding sequences from face
        f_p to face f_q - the README's literal "BFS over face sequences",
        still pruned by the admissible source-to-portal lower bound, but
        without the priority-queue (best-first) ordering used in
        polyhedra.py. No depth cap anywhere - termination comes from the
        no-revisit rule alone (a shortest path on a convex mesh visits each
        face at most once), so the same code works on any convex mesh
        (octahedron or icosahedron).

        Because the queue is not ordered by lower bound, a dequeued state's
        own lb must be re-checked against the current incumbent `best`
        before it is expanded (an item pushed early in the traversal may
        have been overtaken by a better `best` found later) - there is no
        safe "stop the whole search" break here, unlike the heap version.

        A target sitting exactly on a vertex has one valid representation
        per incident face, not only the one named by (f_q, b_q) - and the
        true shortest path may reach it through a DIFFERENT incident face
        than f_q (e.g. two mesh-adjacent vertices whose chart faces don't
        share an edge). So termination also fires the moment the search
        reaches ANY face containing that vertex id, using the matching
        one-hot representation local to that face - not only on the literal
        neighbor == f_q match.
        """
        if f_p == f_q:
            return float(np.linalg.norm((b_p - b_q) @ self.triangle(f_p)))

        target_vertex_id = (int(self.faces[f_q, int(np.argmax(b_q))])
                             if b_q.max() > 1.0 - 1e-9 else None)

        verts0 = self._local2d(self.triangle(f_p))
        p2d = b_p @ verts0
        best = np.inf
        # queue item: (lower_bound, face, verts2d, visited, portals)
        queue = deque([(0.0, f_p, verts0, frozenset({f_p}), ())])
        while queue:
            lb, face, verts2d, visited, portals = queue.popleft()
            if lb >= best:
                continue  # this state can no longer improve `best`; skip it
            for ij in _LOCAL_EDGES:
                va = int(self.faces[face, ij[0]])
                vb = int(self.faces[face, ij[1]])
                key = (va, vb) if va < vb else (vb, va)
                f0, f1 = self._edge_to_faces[key]
                neighbor = f1 if f0 == face else f0
                if neighbor in visited:
                    continue
                new_verts2d, portal = self._expand(verts2d, face, ij, neighbor)
                new_portals = portals + (portal,)

                if neighbor == f_q:
                    q2d = b_q @ new_verts2d
                    cand = self._validate(p2d, q2d, new_portals)
                    if cand is not None and cand < best:
                        best = cand
                    continue  # nothing useful beyond the literal target face

                if target_vertex_id is not None:
                    match = np.flatnonzero(self.faces[neighbor] == target_vertex_id)
                    if len(match):
                        # A different, valid representation of the same
                        # target point. Record it as a candidate, but still
                        # explore onward below - the literal f_q face may
                        # only be reachable through here, via a route that
                        # targets a DIFFERENT point (f_q's own bary
                        # position) with its own, independent validity.
                        q2d = new_verts2d[int(match[0])]
                        cand = self._validate(p2d, q2d, new_portals)
                        if cand is not None and cand < best:
                            best = cand

                lb_new = self._point_segment_distance(p2d, *portal)
                if lb_new < best:
                    queue.append((
                        lb_new, neighbor, new_verts2d,
                        visited | {neighbor}, new_portals))
        return best


def octahedron_bfs() -> PolyhedralSurfaceBFS:
    """Unit-edge regular octahedron as a PolyhedralSurfaceBFS."""
    return PolyhedralSurfaceBFS(OCTAHEDRON_VERTICES, OCTAHEDRON_FACES, "octahedron_bfs")


def icosahedron_bfs() -> PolyhedralSurfaceBFS:
    """Unit-edge regular icosahedron as a PolyhedralSurfaceBFS."""
    return PolyhedralSurfaceBFS(ICOSAHEDRON_VERTICES, ICOSAHEDRON_FACES, "icosahedron_bfs")
