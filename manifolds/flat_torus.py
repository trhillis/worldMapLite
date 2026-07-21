"""Flat square torus R^2 / (2 pi Z)^2, embedded as the Clifford torus in R^4.

Start here: it is the easiest manifold in the set and it introduces the one
idea everything else reuses - the geodesic distance on a quotient space is a
MINIMUM over the images of one point in the universal cover.

The space
---------
Take the plane R^2 with coordinates (u, v), and declare two points identical
whenever they differ by (2 pi m, 2 pi n) for integers m, n. Walking off the
right edge of the square [0, 2 pi)^2 re-enters on the left; same top/bottom.
Locally this is just the flat plane - zero curvature everywhere - but globally
it is a torus. This is the control condition of the whole exercise: geometry
identical to the plane in the small, different in the large.

Distance (what you implement)
-----------------------------
A geodesic in the quotient lifts to a straight segment in the plane from p to
SOME lattice image q + (2 pi m, 2 pi n). Therefore

    d(p, q) = min over (m, n) of || p - q - (2 pi m, 2 pi n) ||.

In practice you do not enumerate a lattice window: reduce each coordinate
difference into the symmetric range [-pi, pi] via s - 2*pi*round(s / (2*pi)),
then take the Euclidean norm. Exact, O(1).

THE bug on this manifold (measured, from the prototype): reducing into
[0, 2 pi) - e.g. with np.mod - instead of [-pi, pi]. That gets 75% of random
pairs wrong (overestimates by up to ~2 pi sqrt(2)). What catches it is worth
remembering: NOT the triangle inequality (the buggy formula provably still
satisfies it) but SYMMETRY - d(p, q) != d(q, p), with errors up to ~8.8.
Also verify the wrap anchor d((0.1, 0.1), (2 pi - 0.1, 2 pi - 0.1)) =
sqrt(0.08) ~ 0.283.

Embedding
---------
The unscaled Clifford embedding in R^4:

    embed(u, v) = (cos u, sin u, cos v, sin v).

This embedding is ISOMETRIC (the induced metric is exactly du^2 + dv^2), which
is why `ambient_metric_is_induced` is True and d_geo >= d_amb must hold. A
smooth flat torus cannot exist in R^3 (any smooth R^3 torus has points of
nonzero curvature), which is why we pay for the fourth dimension.

Sampling
--------
Uniform in (u, v) on [0, 2 pi)^2 is EXACTLY correct here - the area element is
constant. Enjoy it: this is the only manifold in the set where the naive
sampler is the right one, and the contrast with the torus of revolution
(where naive sampling is wrong) is the lesson.

Isometries for the invariance test: translations in u and v, the swap
(u, v) -> (v, u), negations, and 90-degree rotation (u, v) -> (-v, u).
"""

from __future__ import annotations

import numpy as np

from .base import Manifold

TAU = 2.0 * np.pi


class FlatTorus(Manifold):
    """Flat square torus; chart points (u, v) in radians, any real values.

    The chart is the covering plane: EVERY real (u, v) is a valid point, and
    (u, v), (u + 2 pi, v), (u, v + 2 pi) are the same point. Reduction into a
    fundamental domain is part of the distance algorithm, not of validation.
    `sample` returns canonical representatives in [0, 2 pi)^2.
    """

    name = "flat_torus"
    intrinsic_dim = 2
    ambient_dim = 4
    point_shape = (2,)
    ambient_metric_is_induced = True

    def _validate_points(self, points: np.ndarray) -> None:
        # Any real (u, v) is valid; only reject non-finite input.
        bad = ~np.isfinite(points).all(axis=1)
        if bad.any():
            i = int(np.argmax(bad))
            raise ValueError(
                f"flat_torus: non-finite chart point at row {i}: {points[i]}")

    def _sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform on [0, 2 pi)^2 - exactly Riemannian-uniform here."""
        raise NotImplementedError

    def _distance(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Nearest-image reduction into [-pi, pi] per coordinate, then norm.

        Fully vectorizable; no loop needed. Remember which reduction bug the
        module docstring warns about, and which test catches it.
        """
        raise NotImplementedError

    def _embed(self, points: np.ndarray) -> np.ndarray:
        """Unscaled Clifford embedding (cos u, sin u, cos v, sin v) in R^4."""
        raise NotImplementedError
