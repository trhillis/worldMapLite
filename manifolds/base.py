"""Abstract base class for the manifold geodesic-distance exercise.

READ THIS FILE FIRST. Every manifold you implement subclasses `Manifold`, and
the public methods here (which you do NOT touch) define exactly what your code
receives and must return. You implement three underscore methods per manifold:

    _sample(n, rng)    draw points uniformly w.r.t. the manifold's own area/volume
    _distance(p, q)    exact geodesic distance, elementwise over batches
    _embed(points)     map chart points to ambient coordinates (for plotting)

The concrete wrappers `sample` / `distance` / `distance_matrix` / `embed`
validate inputs and outputs and then call your code. You never need to write
validation call sites, and you cannot forget them.

Two representations of a point, and the one rule of this codebase
------------------------------------------------------------------
Every point has a CHART representation (the coordinates your math works in,
shape `point_shape`) and an AMBIENT representation (a vector in R^ambient_dim,
produced by `embed`, used for plotting and for comparing geodesic to straight-
line distance). Mixing them up is the single most common bug in this kind of
code. The interface is designed so the mix-up fails loudly instead of silently:
for every manifold in this project, the chart shape differs from the ambient
shape, so passing embedded coordinates back into `distance` or `sample`-adjacent
code raises a shape error immediately. The rule: EVERYTHING in this API speaks
chart coordinates; `embed` is the only door to ambient space, and there is no
door back.

Shape contract (memorize this)
------------------------------
A single point is an array of shape `point_shape` (e.g. `(2,)` for a surface
chart, `(2, 2)` for an SPD matrix). A batch is `(n, *point_shape)`. Public
methods accept a single point or a batch and normalize to a batch before
calling your underscore methods, which therefore ALWAYS see validated batches
of equal length. `distance` is elementwise: `d[i] = dist(p[i], q[i])`. For the
full n-by-m matrix use `distance_matrix` - never write a Python double loop
over pairs in caller code. If your algorithm is genuinely per-pair (quadrature,
enumeration), the loop belongs INSIDE `_distance`.

Invariants your implementation must satisfy (the test battery)
--------------------------------------------------------------
  1. symmetry        d(p, q) == d(q, p)
  2. identity        d(p, p) == 0 and d(p, q) > 0 for p != q
  3. triangle        d(p, s) <= d(p, q) + d(q, s)
  4. d_geo >= d_amb  ONLY when `ambient_metric_is_induced` is True. For SPD(2)
                     and the flat Moebius strip it is False - the embedding is
                     not isometric there and the inequality genuinely fails
                     (measured, not hypothetical). Do not "fix" your code to
                     pass this test on those manifolds.
  5. isometry        each manifold's docstring lists symmetries that must leave
                     distances unchanged
  6. sampling        binned counts must match the Riemannian area/volume
Warning: these tests are necessary, not sufficient. A distance that satisfies
all metric axioms can still be the right metric on the WRONG space (this
project contains a concrete such bug); each manifold's docstring lists the
extra identification tests that catch what the axioms cannot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Manifold(ABC):
    """Base class: batched chart-coordinate API with a single exit to ambient space.

    Subclasses set five class attributes and implement the underscore methods.

    Attributes
    ----------
    name : str
        Short identifier, e.g. ``"flat_torus"``.
    intrinsic_dim : int
        Dimension of the manifold itself (2 or 3 in this project).
    ambient_dim : int
        Length of one embedded point returned by `embed`.
    point_shape : tuple of int
        Trailing shape of ONE chart point. Deliberately different from
        ``(ambient_dim,)`` on every manifold (the shape firewall).
    ambient_metric_is_induced : bool
        True iff the geodesic metric is the one induced by the embedding, i.e.
        d_geo(p, q) >= ||embed(p) - embed(q)|| is guaranteed. Test harnesses
        must skip that check when False.
    """

    name: str
    intrinsic_dim: int
    ambient_dim: int
    point_shape: tuple
    ambient_metric_is_induced: bool

    # ------------------------------------------------------------------
    # Concrete machinery - students do not modify anything in this block.
    # ------------------------------------------------------------------

    def check_points(self, points) -> np.ndarray:
        """Coerce to a float64 batch (n, *point_shape) and validate.

        Accepts a single point of shape `point_shape` (promoted to a batch of
        one) or a batch. Raises ValueError when the trailing shape is wrong -
        which is what happens when ambient coordinates are passed in - and
        when `_validate_points` rejects the values. Returns a fresh array:
        your underscore methods may scratch on it without mutating caller data.
        """
        arr = np.array(points, dtype=float)  # always a copy
        k = len(self.point_shape)
        if arr.shape == self.point_shape:
            arr = arr[None, ...]
        if arr.ndim != k + 1 or arr.shape[1:] != self.point_shape:
            raise ValueError(
                f"{self.name}: expected points of shape {self.point_shape} or "
                f"(n, {', '.join(map(str, self.point_shape))}), got {arr.shape}. "
                f"If this input came from embed(), you are passing AMBIENT "
                f"coordinates into a chart-coordinate API."
            )
        self._validate_points(arr)
        return arr

    def sample(self, n: int, *, rng: np.random.Generator) -> np.ndarray:
        """Draw n points, uniform w.r.t. the Riemannian area/volume.

        `rng` is required and keyword-only: pass `np.random.default_rng(seed)`.
        This keeps every experiment reproducible and makes accidental reseeding
        visible in the caller's code. The sampler's OUTPUT is validated too, so
        a buggy `_sample` fails here, at the call site, not two scripts later.

        Returns a batch of chart points, shape (n, *point_shape). Never
        ambient coordinates.
        """
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                f"rng must be a numpy Generator; got {type(rng).__name__}. "
                f"Create one with rng = np.random.default_rng(seed)."
            )
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError(f"n must be a nonnegative int, got {n!r}")
        out = self._sample(int(n), rng)
        out = np.asarray(out, dtype=float)
        if out.shape != (n, *self.point_shape):
            raise ValueError(
                f"{self.name}._sample returned shape {out.shape}, expected "
                f"{(n, *self.point_shape)}"
            )
        self._validate_points(out)
        return out

    def distance(self, p, q, *, check: bool = True):
        """Elementwise geodesic distance: d[i] = dist(p[i], q[i]).

        Accepts single points or batches; a single point broadcasts against a
        batch. Two batches must have EQUAL length - if you want all pairs
        between two point sets, that is `distance_matrix`, not this method.
        Returns a float for two single points, else an array of shape (n,).

        `check=False` skips validation for large measured sweeps; leave it on
        everywhere else.

        Output is checked: shape (n,), no NaN, no negatives. If you hit the
        NaN check, the usual culprit is `arccos` of a dot product that
        floating-point error pushed just past +-1 - clamp to [-1, 1] first,
        do not mask the NaN.
        """
        single = (np.shape(p) == self.point_shape
                  and np.shape(q) == self.point_shape)
        if check:
            p, q = self.check_points(p), self.check_points(q)
        else:
            p, q = np.atleast_2d(np.asarray(p, float)), \
                   np.atleast_2d(np.asarray(q, float))
            if p.shape == self.point_shape:
                p = p[None]
            if q.shape == self.point_shape:
                q = q[None]
        if len(p) == 1 and len(q) > 1:
            p = np.broadcast_to(p, q.shape).copy()
        if len(q) == 1 and len(p) > 1:
            q = np.broadcast_to(q, p.shape).copy()
        if len(p) != len(q):
            raise ValueError(
                f"{self.name}.distance: batch lengths differ ({len(p)} vs "
                f"{len(q)}). distance() is elementwise; for the full cross "
                f"matrix use distance_matrix(p, q)."
            )
        d = np.asarray(self._distance(p, q), dtype=float)
        if d.shape != (len(p),):
            raise ValueError(
                f"{self.name}._distance returned shape {d.shape}, expected "
                f"({len(p)},)"
            )
        if np.any(np.isnan(d)):
            raise FloatingPointError(
                f"{self.name}._distance produced NaN (clamp before arccos/"
                f"sqrt/log)"
            )
        if np.any(d < 0):
            raise FloatingPointError(
                f"{self.name}._distance produced a negative distance "
                f"({d.min():.3e})"
            )
        return float(d[0]) if single else d

    def distance_matrix(self, p, q=None, *, check: bool = True) -> np.ndarray:
        """Full cross matrix of geodesic distances, shape (n, m).

        `q=None` means q = p. Implemented once, here, on top of your
        `_distance`: it tiles batches into pairs and makes elementwise calls.
        This is the supported way to get all pairwise distances - it exists
        precisely so you never hand-write the double loop (which will be both
        slow and a source of transposed-index bugs).

        Memory note: the tiling is CHUNKED (about 2 million pairs of scratch
        at a time), so a 20k x 20k request costs ~3 GB for the result matrix
        itself but not tens of GB of intermediates. The result matrix is
        still n*m floats - budget for it.
        """
        p = self.check_points(p) if check else np.asarray(p, float)
        q = p if q is None else (self.check_points(q) if check
                                 else np.asarray(q, float))
        n, m = len(p), len(q)
        out = np.empty((n, m))
        rows_per_chunk = max(1, 2_000_000 // max(m, 1))
        for i0 in range(0, n, rows_per_chunk):
            pc = p[i0:i0 + rows_per_chunk]
            P = np.repeat(pc, m, axis=0)
            Q = np.tile(q, (len(pc),) + (1,) * len(self.point_shape))
            out[i0:i0 + len(pc)] = self.distance(
                P, Q, check=False).reshape(len(pc), m)
        return out

    def embed(self, points, *, check: bool = True) -> np.ndarray:
        """Chart -> ambient coordinates, for plotting and ambient comparisons.

        Single point -> shape (ambient_dim,); batch -> (n, ambient_dim).
        This is the ONLY chart-to-ambient crossing in the API; no public
        method accepts ambient coordinates. When `ambient_metric_is_induced`
        is False (SPD(2), Moebius), Euclidean distances between embedded
        points are NOT lower bounds for geodesic distances - the embedding is
        a picture, not a ruler.
        """
        single = np.shape(points) == self.point_shape
        pts = self.check_points(points) if check else np.atleast_2d(
            np.asarray(points, float))
        out = np.asarray(self._embed(pts), dtype=float)
        if out.shape != (len(pts), self.ambient_dim):
            raise ValueError(
                f"{self.name}._embed returned shape {out.shape}, expected "
                f"({len(pts)}, {self.ambient_dim})"
            )
        return out[0] if single else out

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} dim={self.intrinsic_dim} "
                f"ambient={self.ambient_dim}>")

    # ------------------------------------------------------------------
    # The exercise: implement these in each concrete manifold.
    # ------------------------------------------------------------------

    @abstractmethod
    def _validate_points(self, points: np.ndarray) -> None:
        """Domain check on an already shape-checked (n, *point_shape) batch.

        Ships implemented in every stub (it is a guardrail, not part of the
        exercise). Raises ValueError naming the first offending row and why.
        """

    @abstractmethod
    def _sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n chart points uniformly w.r.t. Riemannian area/volume.

        "Uniform" is w.r.t. the manifold's own notion of area, which for most
        of these manifolds is NOT uniform in the chart coordinates - each
        manifold's docstring states the correct scheme. All randomness must
        come from `rng`. Sampling domain: the whole manifold when it has
        finite volume; otherwise a pinned bounded subset fixed in the
        constructor (SPD(2)). Return shape (n, *point_shape).
        """

    @abstractmethod
    def _distance(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Exact geodesic distance for validated equal-length batches.

        Inputs are (n, *point_shape) float64 arrays you may scratch on.
        Return shape (n,), nonnegative, no NaN. Vectorize when the formula
        allows; where the algorithm is per-pair (unfolding enumeration,
        quadrature + root-finding), loop over pairs HERE.

        The unifying pattern across this project: geodesic distance is a
        MINIMUM over a discrete set of candidates (lattice images, group
        elements, unfolding sequences, winding branches). Enumerate all of
        them; the most common bug family is a missing candidate class, and it
        usually survives the metric axioms - see each manifold's docstring
        for the test that actually catches it.
        """

    @abstractmethod
    def _embed(self, points: np.ndarray) -> np.ndarray:
        """Map a validated chart batch (n, *point_shape) to ambient
        coordinates (n, ambient_dim)."""
