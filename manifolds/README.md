# Manifold geodesics: implementation reference

Three manifolds. For each: implement `_sample`, `_distance`, `_embed` in the stub under
`manifolds/`; `_validate_points` and all public machinery ship working. Acceptance is
`uv run pytest` going green (section 6). This document is the complete reference; read it
top to bottom, then implement.

The one structural fact that repeats: on every manifold here, geodesic distance is a
**minimum over a discrete set of candidates** - lattice images (torus), glide images
(Möbius), unfolding sequences (octahedron). The bug family to fear is a missing
candidate class, and it usually survives the metric axioms.

| Manifold | Class | Chart point | `point_shape` | Ambient | Metric induced? |
|---|---|---|---|---|---|
| Flat square torus | `FlatTorus()` | `(u, v)`, any reals | `(2,)` | R⁴ | yes |
| Flat Möbius strip | `FlatMobiusStrip()` | `(x, y)`, `y ∈ [0, 1]` | `(2,)` | R³ | **no** |
| Regular octahedron | `octahedron()` | `(face, b0, b1, b2)` | `(4,)` | R³ | yes |

## 1. The interface

Abridged from `manifolds/base.py`, which is authoritative and heavily commented.

```python
class Manifold(ABC):
    """Batched chart-coordinate API with a single exit to ambient space."""

    name: str
    intrinsic_dim: int
    ambient_dim: int
    point_shape: tuple[int, ...]     # trailing shape of ONE chart point
    ambient_metric_is_induced: bool  # True iff d_geo >= d_amb is guaranteed

    # ---- provided machinery: validates your inputs AND outputs ----------
    def sample(self, n: int, *, rng: np.random.Generator) -> np.ndarray:
        """n chart points, uniform w.r.t. the manifold's own area.
        Returns (n, *point_shape). rng is required and keyword-only."""

    def distance(self, p, q, *, check: bool = True) -> float | np.ndarray:
        """Elementwise geodesic distance: d[i] = dist(p[i], q[i]).
        Single points broadcast against a batch; two batches must have
        equal length. Output checked: shape (n,), no NaN, no negatives."""

    def distance_matrix(self, p, q=None, *, check: bool = True) -> np.ndarray:
        """All-pairs (n, m) matrix, chunked internally. q=None means q=p.
        Use this, never a hand-written double loop in caller code."""

    def embed(self, points, *, check: bool = True) -> np.ndarray:
        """Chart -> ambient coordinates, (n, ambient_dim).
        The ONLY chart-to-ambient door; there is no door back."""

    # ---- the exercise: implement these three per manifold ---------------
    @abstractmethod
    def _sample(self, n: int, rng: np.random.Generator) -> np.ndarray: ...

    @abstractmethod
    def _distance(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Validated equal-length (n, *point_shape) batches in, (n,) out.
        If your algorithm is per-pair, the loop goes HERE."""

    @abstractmethod
    def _embed(self, points: np.ndarray) -> np.ndarray: ...
```

- Every point has two representations: CHART (what all the math uses) and AMBIENT
  (what `embed` returns, for plotting). `point_shape != (ambient_dim,)` on every
  manifold, so passing ambient coordinates into a chart API fails loudly with a shape
  error instead of silently returning nonsense.
- Shape contract: single point `point_shape`, batch `(n, *point_shape)`. Your
  underscore methods always receive validated batches of equal length.
- `d_geo >= d_amb` holds iff `ambient_metric_is_induced`. It is genuinely false for
  the Möbius strip (fails on 20.6% of random pairs against the ruled embedding,
  measured); the test suite skips it there automatically. Do not "fix" your code to
  make it pass.

## 2. Point budgets

The sampled points become tokens downstream, so n is a budget, not a free parameter.
Two constraints:

- **Coverage.** n uniform points on area A leave gaps of expected covering radius
  `r_cov ≈ sqrt(A · ln n / (π n))`. Keep `r_cov` under ~10% of the geodesic diameter
  and under ~20% of the smallest geometric feature (the Möbius width, the octahedron
  edge).
- **Cost.** Tokens scale as n, supervision pairs as n(n-1)/2 (2000 points = 1,999,000
  pairs). Budgets are per manifold - training runs use one manifold at a time, so
  vocabularies do not add up across rows. The octahedron has the one real ceiling:
  its distance is a per-pair enumeration, so the full matrix at n = 800 is ~320k
  enumerations (minutes of compute); past ~1500 points that cost dominates.

| Manifold | Area | Diameter | Recommended n | `r_cov` at default | r_cov/diam |
|---|---|---|---|---|---|
| Flat torus | 4π² ≈ 39.5 | π√2 ≈ 4.44 | 1000–4000 (default 2000) | 0.22 | 4.9% |
| Möbius strip | 2π ≈ 6.28 | ≈ π | 500–2000 (default 1000) | 0.12 | 3.7% |
| Octahedron | 2√3 ≈ 3.46 | √3 ≈ 1.73 | 400–1500 (default 800) | 0.10 | 5.5% |

The defaults give near-equal relative resolution across the three manifolds, which is
what a cross-manifold comparison wants.

## 3. Flat square torus

`R² / (2πZ)²`: the unit square of side 2π with opposite edges glued, carrying the flat
metric. Chart `(u, v)`, any reals valid, `u` and `u + 2π` name the same point.
Curvature zero everywhere; this is the control condition of the set.

**Embedding (chart -> R⁴).** The Clifford torus,

```
embed(u, v) = (cos u, sin u, cos v, sin v)
```

This embedding is isometric (it preserves the flat metric exactly), hence
`ambient_metric_is_induced = True`. A flat torus cannot be smoothly and isometrically
embedded in R³, and the figure shows the symptom: every coordinate drop to 3D is
degenerate. `(x3, x4)` lie on a circle, so dropping `x4` folds the v-circle onto a
segment and the image is a cylinder with two chart points per image point (third
panel). The stereographic panel, a nonlinear projection, is the one view that keeps
the torus topology visible.

![2000 uniform samples on the flat torus, four projections](figures/torus_points.png)

**Distance.** Reduce each coordinate difference to the symmetric range `[-π, π]`, then
take the Euclidean norm:

```
w = (u1 - u2, v1 - v2)
w -= 2π * round(w / 2π)          # elementwise, lands in [-π, π]
d = ||w||
```

Exact, O(1), fully vectorized. The rounding IS the minimum over all lattice images;
never enumerate a lattice window in production code.

Implementation tips:

- THE bug: reducing into `[0, 2π)` with `np.mod` instead of the symmetric range.
  Measured: 75% of random pairs wrong, errors up to ~8.8. Symmetry
  `d(p, q) = d(q, p)` catches it immediately; the triangle inequality provably never
  does. Test symmetry first.
- `round`, not `floor` or `trunc`: the reduction must pick the nearest image.

**Sampling.** Uniform on `[0, 2π)²` is exactly Riemannian-uniform: the area element is
constant. This is the one manifold in the set where the naive sampler is correct.

**Anchors** (all frozen in the test suite):

| Pair | Distance |
|---|---|
| `(0.1, 0.1)` to `(2π−0.1, 2π−0.1)` | `√0.08 ≈ 0.2828` (wraps both coordinates) |
| `(0, 0)` to `(π, 0)` | `π` |
| `(0, 0)` to `(π, π)` | `π√2` (the diameter) |

## 4. Flat Möbius strip

The quotient of the infinite strip `R × [0, w]` by the glide reflection

```
g(x, y) = (x + L, w - y),        L = 2π, w = 1 (pinned project-wide)
```

Walking distance L in x returns you to the strip mirrored in y. Flat, non-orientable,
one boundary circle (`y = 0` and `y = w` glue into a single circle). Chart `(x, y)`:
x any real, y is a hard boundary, validation rejects `y ∉ [0, w]`.

**Embedding (chart -> R³, PLOT ONLY).** With `φ = 2πx/L` and `ρ = y - w/2`:

```
embed(x, y) = ((1 + ρ cos(φ/2)) cos φ,
               (1 + ρ cos(φ/2)) sin φ,
                ρ sin(φ/2))
```

This is the familiar ruled band with a half twist. It is NOT isometric to the flat
quotient - `ambient_metric_is_induced = False`, the picture is not a ruler. Sanity
check on your formula: `embed(g(p)) == embed(p)` exactly, or your plots show a seam.

![1000 uniform samples on the Möbius strip, ruled picture](figures/mobius_points.png)

**Distance.** Minimum over glide images of q in the covering strip:

```
d(p, q) = min over k in {-3..3} of ||p - g^k(q)||,
g^k(q) = (q_x + kL, q_y)        for even k
         (q_x + kL, w - q_y)    for odd k
```

Exact, a small loop over k, vectorized across the batch.

Implementation tips:

- THE bug: forgetting the y-flip on odd k. The result is the CYLINDER's metric, a
  perfectly valid metric that passes every axiom test on random triples. Only the
  identification test `d(p, g(p)) = 0` catches it (the buggy version returns up to w).
- Enumerate negative k too. `|k| <= 3` suffices at L = 2π, w = 1: measured identical
  to `|k| <= 6` on 100k random pairs.
- Do NOT add images reflected across `y = 0` / `y = w` (bounce-off-the-boundary
  paths). The covering strip is convex and
  `||p - q||² - ||p - refl(q)||² = -4 y_p y_q <= 0`: a reflected image never wins
  (0 wins in 100k pairs, measured). If boundary handling changes any distance, the
  bug is in the boundary handling.

**Sampling.** Uniform on the fundamental domain `[0, L) × [0, w]`: exactly correct
(flat, constant area element).

**Anchors:**

| Pair | Distance |
|---|---|
| `p` to `g(p)` | `0` (same point; the cylinder bug fails here) |
| `(0, 0)` to `(0, 1)` | `1` (straight across the width) |
| `(0, 0.3)` to `(2π−0.05, 0.7)` | `0.05` (wrap + flip: the target is `g` of `(−0.05, 0.3)`) |

## 5. Regular octahedron

The surface of the regular octahedron with unit edge: 8 equilateral triangles, 6
vertices at `±e_i/√2`, 12 edges. Each face is flat; all curvature sits at the vertices
as angle deficit `2π/3` each (4 × 60° of face angle instead of 360°), summing to `4π`.
Geodesics bend across edges but never pass through a vertex.

Chart point: one packed float row `(face_index, b0, b1, b2)` - the face plus full
barycentric coordinates w.r.t. that face's vertices in `faces[face]` order,
nonnegative, summing to 1. Use the provided `make_points` / `split_points` /
`triangle`; never index columns by hand. Points on an edge or vertex have one valid
representation per adjacent face, and `distance` / `embed` must agree across them.

**Embedding (chart -> R³).**

```
embed(face, b) = b @ triangle(face)       # barycentric times the 3x3 vertex matrix
```

One `einsum` for the whole batch.

![800 uniform samples on the octahedron](figures/octa_points.png)

**Distance.** A shortest path crossing faces f0, f1, ..., fk unfolds to a straight
segment in the plane. Enumerate and take the minimum:

1. Place the source face flat in 2D; compute the source point from its barycentrics.
2. BFS over face sequences: expand across each edge of the current face except the
   entry edge, skipping faces already in the sequence (on a convex polyhedron a
   shortest path visits a face at most once - this is the termination guarantee).
3. Unfold the neighbor across the shared edge: place its third vertex by
   circle-circle intersection on the far side. Barycentric coordinates transfer
   verbatim to unfolded copies: `q2d = bary @ unfolded_triangle`.
4. When the sequence reaches the target face, form the candidate segment
   source -> target and validate it: it must cross EVERY intermediate unfolded edge
   strictly inside the open edge (`tol < s < 1 - tol`), with crossing parameters
   increasing along the segment. Invalid candidates are discarded, not clamped.
5. Answer = min over valid candidates. Same-face pairs: the in-plane segment, done.

Exact when correct: agrees with the independent MMP solver (`pygeodesic`) to ~1e-15.

Implementation tips:

- Terminate by no-revisit, never by a hardcoded depth cap. A cap tuned on the
  octahedron is exactly the bug that a later 20-face mesh exposes; and a depth cap
  WITH revisits allowed is an unbounded DFS (measured: 6.5 GB OOM).
- Prune against the incumbent best: the planar distance from the source to the entry
  edge is a valid lower bound for every deeper candidate. Without this the
  enumeration cost explodes on finer meshes.
- Exclude vertex-through paths (never shortest when the deficit is positive), but
  treat near-vertex "grazing" segments with a tolerance; that tolerance sets how
  close to a vertex your answers can be trusted (~1e-8 is the practical floor).
- On-edge sources: if the source lies exactly on an edge of its face, the shortest
  path may leave through THAT edge, and the crossing test sees parameter `t = 0` and
  rejects it. Result: the two representations of the same edge point return different
  distances (measured: 1.607 vs the true 1.258 on the anchor below). Fix it by
  accepting the `t = 0` crossing at the source, or by seeding the BFS from every face
  the point lies on.

**Sampling.** Face with probability proportional to `face_areas` (equal here; write it
weighted anyway, the code then works for any mesh), then uniform in the triangle by
the reflection trick: `u1, u2 ~ U[0,1]`; if `u1 + u2 > 1` replace with
`(1-u1, 1-u2)`; barycentric `(1-u1-u2, u1, u2)`. Exactly uniform.

**Anchors:**

| Pair | Distance |
|---|---|
| Opposite vertices (+z to −z) | `√3` |
| Adjacent-face centroids (faces 0, 1) | `√3/3 ≈ 0.5774` |
| Same-face pair | the planar chord `‖(b1 − b2) @ triangle(f)‖` |
| Edge-(2,4) midpoint (either representation) to centroid of face 7 | `1.2583057392117913` (MMP-verified) |

## 6. Tests

```
uv run pytest -q          # fast battery (slow MMP cross-checks deselected)
uv run pytest -m slow     # exact-solver referee for the octahedron
```

`tests/test_manifolds.py` is the acceptance suite. On a fresh checkout it collects
and SKIPS everything that touches an unimplemented stub (2 tests pass immediately);
implement until green. What each test family is for:

| Test | The bug it catches |
|---|---|
| symmetry | torus wrong-branch reduction (triangle inequality is blind to it) |
| `d(p, g(p)) = 0` | Möbius forgot-the-flip = cylinder metric (passes all axioms) |
| brute-force / k-window comparison | missing candidate images, `k >= 0` only |
| `d_geo >= d_amb` (auto-skipped when not induced) | confusing the picture with the metric |
| anchor values | right metric on the wrong space, wrong constants |
| representation invariance | on-edge points answered per-representation |
| sampling chi-square | wrong density, wrong domain, no-reflection triangle bug |
| MMP cross-check (slow) | everything at once, to 1e-9 |

A note on the chi-square thresholds: a wrong density drives the p-value to ~0, while
a correct sampler on an unlucky seed can dip to ~1e-4 (observed). The threshold is
1e-6 to separate the regimes; do not tighten it.

The slow MMP cross-checks need `pygeodesic` (already a dependency) and use the
self-contained referee in `tests/mmp_referee.py` - no external prototype code.

## 7. Figures

The three scatter plots above are not illustrations. Each is the actual output of the
sampler this document recommends, drawn at the section-2 default budget (torus 2000,
Möbius 1000, octahedron 800) with seed 0. The point sets you see are exactly the ones a
correct `sample` produces, so once your sampler is implemented you can reproduce them.
The plotting script that renders these PNGs is a maintainer tool and is not part of the
handout.
