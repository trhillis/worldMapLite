"""Acceptance tests for the manifold implementations.

Run:  uv run pytest -q            (fast battery; slow MMP cross-checks excluded)
      uv run pytest -m slow       (exact-solver referee for the octahedron)

On a fresh checkout every test that touches unimplemented stubs SKIPS. Implement
`_sample` / `_distance` / `_embed` until the suite is green. Each anchor value
below was verified against an independent implementation before being frozen;
comments say which bug the test exists to catch.

Adding a manifold to the roster: append an instance to MANIFOLDS (the shared
battery is written purely against the `Manifold` ABC) and add its anchor tests
in a new section.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chisquare

from conftest import call
from manifolds import FlatMobiusStrip, FlatTorus, octahedron

TAU = 2 * np.pi
TOL = 1e-9

# Sampling tests use a deliberately loose chi-square threshold: a WRONG density
# sends the p-value to ~0 (double underflow), while a correct sampler on an
# unlucky seed can dip to ~1e-4 (observed in prototyping). 1e-6 separates the
# two regimes; do not "fix" a red sampling test by tightening the seed.
P_MIN = 1e-6

TORUS = FlatTorus()
MOBIUS = FlatMobiusStrip()          # length = 2*pi, width = 1, pinned
OCTA = octahedron()                 # unit edge

MANIFOLDS = [TORUS, MOBIUS, OCTA]

# Per-pair algorithms (octahedron) get smaller batches so a correct but
# unoptimized implementation still finishes the fast battery in minutes.
def _n_pairs(m, dflt=2000, per_pair=300):
    return per_pair if m.name == "octahedron" else dflt


@pytest.fixture(params=MANIFOLDS, ids=lambda m: m.name)
def m(request):
    return request.param


# ---------------------------------------------------------------- shared battery


def test_sample_shape(m, rng):
    P = call(m.sample, 256, rng=rng)
    assert P.shape == (256, *m.point_shape)
    # domain validity is enforced by sample() itself via _validate_points


def test_symmetry(m, rng):
    # Catches the flat-torus [0, 2pi) reduction bug (np.mod instead of the
    # symmetric range): 75% of pairs wrong, errors up to ~8.8. The triangle
    # inequality provably cannot see that bug; symmetry does.
    n = _n_pairs(m)
    P, Q = call(m.sample, n, rng=rng), call(m.sample, n, rng=rng)
    d1, d2 = call(m.distance, P, Q), call(m.distance, Q, P)
    assert np.abs(d1 - d2).max() < TOL


def test_identity_and_positivity(m, rng):
    n = _n_pairs(m)
    P, Q = call(m.sample, n, rng=rng), call(m.sample, n, rng=rng)
    assert np.abs(call(m.distance, P, P)).max() < TOL
    d = call(m.distance, P, Q)
    distinct = ~np.all(
        P.reshape(n, -1) == Q.reshape(n, -1), axis=1)
    assert (d[distinct] > 0).all()


def test_triangle_inequality(m, rng):
    n = _n_pairs(m)
    P, Q, R = (call(m.sample, n, rng=rng) for _ in range(3))
    viol = call(m.distance, P, R) - call(m.distance, P, Q) - call(m.distance, Q, R)
    assert viol.max() < TOL


def test_geodesic_dominates_chord(m, rng):
    # Valid ONLY when the metric is induced by the embedding. For the Moebius
    # strip the ruled band is a picture, not a ruler (d_geo < d_amb on 20.6% of
    # pairs, measured); skipping there is correct, not lenient.
    if not m.ambient_metric_is_induced:
        pytest.skip("embedding is not isometric; inequality does not apply")
    n = _n_pairs(m, dflt=5000, per_pair=500)
    P, Q = call(m.sample, n, rng=rng), call(m.sample, n, rng=rng)
    d_geo = call(m.distance, P, Q)
    d_amb = np.linalg.norm(call(m.embed, P) - call(m.embed, Q), axis=1)
    assert (d_geo - d_amb).min() > -TOL


def test_embed_shapes(m, rng):
    P = call(m.sample, 8, rng=rng)
    assert call(m.embed, P).shape == (8, m.ambient_dim)
    assert call(m.embed, P[0]).shape == (m.ambient_dim,)


def test_distance_matrix_consistency(m, rng):
    P = call(m.sample, 20, rng=rng)
    D = call(m.distance_matrix, P)
    assert D.shape == (20, 20)
    assert np.abs(np.diag(D)).max() < TOL
    assert np.abs(D - D.T).max() < TOL
    assert abs(D[3, 7] - call(m.distance, P[3], P[7])) < TOL


# ---------------------------------------------------------------- flat torus


def test_torus_wrap_anchor():
    # THE bug: reducing coordinate differences into [0, 2pi) instead of
    # [-pi, pi]. Checked in BOTH argument orders; the mod bug passes one
    # order and fails the other.
    p, q = np.array([0.1, 0.1]), np.array([TAU - 0.1, TAU - 0.1])
    want = np.sqrt(0.08)
    assert abs(call(TORUS.distance, p, q) - want) < 1e-12
    assert abs(call(TORUS.distance, q, p) - want) < 1e-12


def test_torus_axis_anchors():
    assert abs(call(TORUS.distance, [0.0, 0.0], [np.pi, 0.0]) - np.pi) < 1e-12
    # opposite corner of the fundamental domain = the diameter
    assert abs(call(TORUS.distance, [0.0, 0.0], [np.pi, np.pi])
               - np.pi * np.sqrt(2)) < 1e-12


def test_torus_brute_force(rng):
    # Any wrong closed form loses to an explicit min over lattice images.
    P, Q = call(TORUS.sample, 5000, rng=rng), call(TORUS.sample, 5000, rng=rng)
    diff = P - Q
    best = np.full(len(P), np.inf)
    for a in range(-3, 4):
        for b in range(-3, 4):
            shift = np.array([TAU * a, TAU * b])
            best = np.minimum(best, np.linalg.norm(diff - shift, axis=1))
    assert np.abs(call(TORUS.distance, P, Q) - best).max() < 1e-12


@pytest.mark.parametrize("iso", [
    lambda P: P + np.array([1.234, -2.345]),      # translation
    lambda P: P[:, ::-1],                         # swap u <-> v
    lambda P: -P,                                 # negation
    lambda P: np.column_stack([-P[:, 1], P[:, 0]]),  # 90 degree rotation
], ids=["translate", "swap", "negate", "rot90"])
def test_torus_isometries(iso, rng):
    P, Q = call(TORUS.sample, 2000, rng=rng), call(TORUS.sample, 2000, rng=rng)
    d = call(TORUS.distance, P, Q)
    assert np.abs(call(TORUS.distance, iso(P), iso(Q)) - d).max() < TOL


def test_torus_sampling_uniform(rng):
    # Uniform in the chart is exactly Riemannian-uniform here (constant area
    # element); a "corrected" sampler is the bug this test catches.
    P = call(TORUS.sample, 100_000, rng=rng)
    counts, *_ = np.histogram2d(P[:, 0], P[:, 1],
                                bins=10, range=[[0, TAU], [0, TAU]])
    assert counts.sum() == 100_000, "samples outside [0, 2pi)^2"
    assert chisquare(counts.ravel()).pvalue > P_MIN


# ---------------------------------------------------------------- Moebius strip


def _glide(P, k=1):
    L, w = MOBIUS.length, MOBIUS.width
    out = np.array(P, dtype=float, copy=True)
    out[..., 0] += k * L
    if k % 2:
        out[..., 1] = w - out[..., 1]
    return out


def test_mobius_glide_identification(rng):
    # THE bug: forgetting the y-flip on odd k gives the CYLINDER metric, which
    # passes symmetry, identity, and the triangle inequality on every random
    # triple. Only this identification test catches it (buggy value up to w).
    P = call(MOBIUS.sample, 1000, rng=rng)
    assert np.abs(call(MOBIUS.distance, P, _glide(P))).max() < TOL


def test_mobius_glide_invariance(rng):
    P = call(MOBIUS.sample, 1000, rng=rng)
    Q = call(MOBIUS.sample, 1000, rng=rng)
    d = call(MOBIUS.distance, P, Q)
    assert np.abs(call(MOBIUS.distance, P, _glide(Q, 2)) - d).max() < TOL
    assert np.abs(call(MOBIUS.distance, _glide(P), _glide(Q)) - d).max() < TOL


def test_mobius_k_window(rng):
    # Catches a too-small window and the k >= 0 only bug. |k| <= 3 and
    # |k| <= 6 agree exactly on 100k pairs at L = 2 pi, w = 1 (measured), so
    # a correct implementation matches this brute force to machine precision.
    P = call(MOBIUS.sample, 5000, rng=rng)
    Q = call(MOBIUS.sample, 5000, rng=rng)
    best = np.full(len(P), np.inf)
    for k in range(-6, 7):
        best = np.minimum(best, np.linalg.norm(P - _glide(Q, k), axis=1))
    assert np.abs(call(MOBIUS.distance, P, Q) - best).max() < 1e-12


def test_mobius_anchors():
    # straight across the width
    assert abs(call(MOBIUS.distance, [0.0, 0.0], [0.0, 1.0]) - 1.0) < 1e-12
    # wrap through the identification WITH the flip: (2pi - 0.05, 0.7) is the
    # glide image of (-0.05, 0.3), so the distance to (0, 0.3) is 0.05
    assert abs(call(MOBIUS.distance, [0.0, 0.3], [TAU - 0.05, 0.7]) - 0.05) < 1e-12


@pytest.mark.parametrize("iso", [
    lambda P: P + np.array([1.234, 0.0]),                       # x-translation
    lambda P: np.column_stack([P[:, 0], MOBIUS.width - P[:, 1]]),  # y-flip
    lambda P: np.column_stack([-P[:, 0], P[:, 1]]),             # x-reflection
], ids=["translate_x", "flip_y", "reflect_x"])
def test_mobius_isometries(iso, rng):
    P = call(MOBIUS.sample, 2000, rng=rng)
    Q = call(MOBIUS.sample, 2000, rng=rng)
    d = call(MOBIUS.distance, P, Q)
    assert np.abs(call(MOBIUS.distance, iso(P), iso(Q)) - d).max() < TOL


def test_mobius_embed_respects_quotient(rng):
    # embed(g(p)) must equal embed(p) exactly, or plots show a seam.
    P = call(MOBIUS.sample, 1000, rng=rng)
    assert np.abs(call(MOBIUS.embed, _glide(P)) - call(MOBIUS.embed, P)).max() < 1e-12


def test_mobius_flag_not_induced():
    # Guards against "fixing" the flag to make test_geodesic_dominates_chord
    # run: the ruled embedding genuinely violates the inequality.
    assert MOBIUS.ambient_metric_is_induced is False


def test_mobius_sampling_uniform(rng):
    P = call(MOBIUS.sample, 100_000, rng=rng)
    counts, *_ = np.histogram2d(P[:, 0], P[:, 1], bins=10,
                                range=[[0, MOBIUS.length], [0, MOBIUS.width]])
    assert counts.sum() == 100_000, "samples outside the fundamental domain"
    assert chisquare(counts.ravel()).pvalue > P_MIN


# ---------------------------------------------------------------- octahedron

# Geometric truth for the dual-representation anchor below, frozen from the
# exact MMP solver (pygeodesic, two independent insertions agree to 3e-16).
_EDGE_MID_TO_F7_CENTROID = 1.2583057392117913


def test_octa_mesh_deficits():
    # Uses only shipped mesh data, so it passes on a fresh checkout: proof the
    # suite itself runs. Each vertex deficit 2pi/3, total 4pi (Descartes).
    deficits = np.full(len(OCTA.vertices), TAU)
    for a, b, c in OCTA.faces:
        T = OCTA.vertices[[a, b, c]]
        for i, vid in enumerate((a, b, c)):
            u = T[(i + 1) % 3] - T[i]
            w = T[(i + 2) % 3] - T[i]
            cosang = u @ w / (np.linalg.norm(u) * np.linalg.norm(w))
            deficits[vid] -= np.arccos(np.clip(cosang, -1.0, 1.0))
    assert np.abs(deficits - TAU / 3).max() < 1e-12
    assert abs(deficits.sum() - 4 * np.pi) < 1e-12


def test_octa_antipodal_vertices():
    # +z is the third vertex of face 0 = [0,2,4]; -z the third of face 4 =
    # [2,0,5]. Catches wrong unfolding geometry and through-vertex shortcuts
    # (a path through a cone point would claim pi*sqrt(2)/... < sqrt(3)).
    p = OCTA.make_points([0], [[0.0, 0.0, 1.0]])
    q = OCTA.make_points([4], [[0.0, 0.0, 1.0]])
    assert abs(call(OCTA.distance, p, q)[0] - np.sqrt(3)) < 1e-12


def test_octa_adjacent_centroids():
    # Faces 0 and 1 share edge (2,4); the two-face unfolding gives sqrt(3)/3.
    cen = [[1 / 3, 1 / 3, 1 / 3]]
    p, q = OCTA.make_points([0], cen), OCTA.make_points([1], cen)
    assert abs(call(OCTA.distance, p, q)[0] - np.sqrt(3) / 3) < 1e-12


def test_octa_same_face_is_planar_chord():
    # Same-face pairs need no unfolding: the in-plane segment is the answer.
    b1, b2 = np.array([0.6, 0.3, 0.1]), np.array([0.1, 0.2, 0.7])
    p, q = OCTA.make_points([2], [b1]), OCTA.make_points([2], [b2])
    chord = np.linalg.norm((b1 - b2) @ OCTA.triangle(2))
    assert abs(call(OCTA.distance, p, q)[0] - chord) < 1e-12


def test_octa_edge_representation_invariance():
    # The midpoint of edge (2,4) has two chart representations: face 0 =
    # [0,2,4] as (0, .5, .5) and face 1 = [2,1,4] as (.5, 0, .5). Both must
    # give the same embedding and the same distances. This catches on-edge
    # sources mishandled by the enumeration: from one representation the
    # shortest path's FIRST crossing is the edge the source sits on (crossing
    # parameter t = 0), and rejecting it silently returns a longer path
    # (1.607 instead of 1.258 here; found in prototyping, truth from MMP).
    ra = OCTA.make_points([0], [[0.0, 0.5, 0.5]])
    rb = OCTA.make_points([1], [[0.5, 0.0, 0.5]])
    assert np.abs(call(OCTA.embed, ra) - call(OCTA.embed, rb)).max() < 1e-12
    x = OCTA.make_points([7], [[1 / 3, 1 / 3, 1 / 3]])
    da, db = call(OCTA.distance, ra, x)[0], call(OCTA.distance, rb, x)[0]
    assert abs(da - _EDGE_MID_TO_F7_CENTROID) < TOL
    assert abs(db - _EDGE_MID_TO_F7_CENTROID) < TOL


def test_octa_sampling_faces(rng):
    # Face choice must follow face_areas. Weak on THIS mesh (all faces equal,
    # so a uniform-face bug is invisible here); kept because the same code
    # must work on irregular meshes. Write the sampler area-weighted.
    P = call(OCTA.sample, 40_000, rng=rng)
    faces, _ = OCTA.split_points(P)
    counts = np.bincount(faces, minlength=OCTA.n_faces)
    expected = 40_000 * OCTA.face_areas / OCTA.face_areas.sum()
    assert chisquare(counts, f_exp=expected).pvalue > P_MIN


def test_octa_sampling_within_face(rng):
    # For a uniform triangle sampler each barycentric coordinate is Beta(1, 2)
    # (density 2(1 - x)). Catches the classic no-reflection bug: drawing
    # (u1, u2) and clipping/renormalizing instead of reflecting when
    # u1 + u2 > 1 biases mass toward the centroid.
    P = call(OCTA.sample, 40_000, rng=rng)
    _, bary = OCTA.split_points(P)
    edges = np.linspace(0.0, 1.0, 11)
    expected = 40_000 * np.diff(2 * edges - edges**2)   # CDF of Beta(1,2)
    for j in range(3):
        counts, _ = np.histogram(bary[:, j], bins=edges)
        assert chisquare(counts, f_exp=expected).pvalue > P_MIN


@pytest.mark.slow
def test_octa_mmp_vertex_pairs():
    # Independent exact solver on the same mesh; the strongest check there is.
    from pygeodesic import geodesic
    alg = geodesic.PyGeodesicAlgorithmExact(
        np.ascontiguousarray(OCTA.vertices),
        np.ascontiguousarray(OCTA.faces, dtype=np.int32))
    # chart representation of vertex j: one-hot bary on a face containing j
    reps = []
    for j in range(len(OCTA.vertices)):
        f = int(np.argmax(np.any(OCTA.faces == j, axis=1)))
        bary = (OCTA.faces[f] == j).astype(float)
        reps.append(OCTA.make_points([f], [bary]))
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            want, _ = alg.geodesicDistance(i, j)
            assert abs(call(OCTA.distance, reps[i], reps[j])[0] - want) < TOL


@pytest.mark.slow
def test_octa_mmp_random_pairs(rng):
    # Referee for everything at once: depth handling, grazing tolerance,
    # candidate validation. Prototype agreement was 4.4e-16 over 300 pairs.
    from mmp_referee import insert_two_points, mmp_distance
    P = call(OCTA.sample, 100, rng=rng)
    faces, bary = OCTA.split_points(P)
    d = call(OCTA.distance, P[0::2], P[1::2])
    for i in range(50):
        V2, F2, i1, i2 = insert_two_points(
            OCTA.vertices, OCTA.faces,
            (int(faces[2 * i]), bary[2 * i]), (int(faces[2 * i + 1]), bary[2 * i + 1]))
        assert abs(d[i] - mmp_distance(V2, F2, i1, i2)) < TOL
