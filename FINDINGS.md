# Findings

Summary of everything learned so far from experiments in this repository.
Detailed per-run logs live in
[`experiments/manifold_learning/octahedron/EXPERIMENTS.md`](experiments/manifold_learning/octahedron/EXPERIMENTS.md)
(87 runs) and
[`experiments/benchmarks/results/polyhedra_benchmark.md`](experiments/benchmarks/results/polyhedra_benchmark.md);
this file distills the cross-run conclusions. Config files for every run are
in `configs/`.

## Background

The project trains a transformer to predict geodesic distance (and,
later, a same-face classification task) between pairs of points sampled
from a synthetic manifold (currently the regular octahedron surface;
`manifolds/` also has a flat torus and Möbius strip). The original
research question (see `README.md`): do the model's internal
per-entity embeddings recover the manifold's true geometry as a
byproduct of learning the task, and under what conditions? The first,
now-frozen prototype (`legacy/`, MLP on a flat rectangular grid) trained
successfully but its hidden states stayed an "amorphous cloud" - no map
emerged. The current `manifolds/` + `src/train_manifold.py` +
`analysis/analysis_manifold.py` pipeline was built to study this
properly on a curved manifold with real geodesics, and it does recover
structure - see below.

**Metrics used throughout:**
- **Cross-validated coordinate R²** - how well a linear probe recovers the
  manifold's true ambient coordinates from the learned embedding table.
- **Geodesic-distance Spearman** - rank correlation between true geodesic
  distance and distance in the learned embedding.
- **NN recall** - do the embedding's nearest neighbors match the manifold's
  true nearest neighbors.
- **Intrinsic dimension** (TwoNN estimator) - local, nonlinear estimate of
  how many dimensions the embedding actually occupies (octahedron surface
  truth: `ambient_dim=3`, effectively a 2D surface).
- **PCA explained variance (top 3 comps)** - global, linear compactness
  measure.
- **Held-out pair / held-out point generalization** - loss and Spearman on
  distances never trained on (see below).

## Headline findings

1. **Geometry recovery works at scale, but only "linearly," not literally.**
   At full scale (800 points, 10k+ steps) the learned embedding table's true
   ambient coordinates are recoverable with R² ≈ 0.96-0.98 via a linear
   probe, and PCA visualizations reproduce the octahedron's shape closely.
   But the raw embedding's *intrinsic dimension* stays well above the
   manifold's true dimensionality (~3) for most configurations - the
   embedding carries much more raw dimensionality than a linear probe needs,
   it does not "collapse" onto the manifold (001, 002).

2. **Two families of metrics converge at very different speeds.**
   Loss, PCA variance, linear/cross-validated R², and Spearman correlation
   saturate early (first 5-20% of training). Nearest-neighbor recall and
   intrinsic dimension are much slower, still improving after 10x more
   steps in some runs, and only reach a "delayed plateau" at very long
   budgets (50k+ steps) (003, 004, 005).

3. **Embedding width (`emb_dim`) trades off linear decodability against
   compactness - except at one width, which dominates outright.**
   Shrinking `emb_dim` from 32 to 8 monotonically hurts R²/Spearman/NN-recall
   while helping intrinsic dimension get closer to 3 (006-008). But
   `emb_dim=24` broke the trend and beat *every* metric simultaneously,
   including `emb_dim=32` (009) - later shown to be at least partly (not
   fully) an undertraining effect on the wider model (010).

4. **Pair-level held-out generalization is essentially perfect everywhere.**
   Across every `emb_dim` (8-32) and every `n_points` tested, distance
   predictions on pairs the model never trained on are statistically
   indistinguishable from predictions on in-distribution pairs (Spearman
   ≥ 0.9987 in every run) (011-019, 070). This holds even in a near-total
   training failure case (070/060) - the model is equally bad on both sets,
   never selectively overfit to the training pairs.

5. **Point-level held-out generalization (recovering a never-trained-on
   point from a handful of probe distances) also works, but is more
   fragile.** A point's embedding, recovered from just 5-8 known distances
   via a short optimization, predicts its distances to normally-trained
   points almost perfectly (Spearman 0.95-0.997) across every scale and
   width tried. But *mutual consistency between two independently-recovered
   points* is much more scale-sensitive - it collapses (Spearman 0.39) when
   too few points remain on the same local region (face) to anchor
   structure, and recovers (Spearman ~0.94-0.98) once that local region has
   enough points, regardless of what fraction was held out (071-075).

6. **Coverage density beats total point count.** Given a fixed training
   budget, denser pair coverage over fewer points (e.g. `n_points=100`)
   consistently beats sparser coverage over more points (`n_points=800`) on
   every embedding-quality metric (018-019, 070). At `n_points=800`, quality
   is highly sensitive to `train_examples` - too few examples (2% pair
   coverage) causes near-total training failure; doubling them fixes it
   (070).

7. **Adding a second task (`same_triangle`, same-face binary
   classification) is learned near-perfectly by itself, but degrades the
   raw embedding's linear-decodability substantially** at a matched step
   budget - even though neither task's own held-out generalization
   suffers (076). Most, but not all, of that degradation is a
   training-budget artifact: 10x the steps recovers NN recall and
   intrinsic dimension completely (they end up *better* than the
   single-task baseline) but only partially recovers R²/Spearman (077).

8. **Downweighting the auxiliary task's loss helps, but not
   monotonically, and not "as low as possible."** A 10-point sweep of
   `same_triangle_weight` (0.1-1.0) found `weight=0.2` strictly dominates
   every other weight on every bulk metric except PCA variance - but the
   weight-vs-quality relationship is noisy/non-monotonic across the rest of
   the range, so no simple rule ("lower is better") predicts it (078-087).
   Counter-intuitively, full weight (1.0, matching the original 076/077
   setup) produces the *worst* same_triangle held-out generalization of the
   whole sweep, not the best.

9. **Best-first geodesic search and plain BFS have identical wall-clock
   cost** on octahedron/icosahedron meshes (within 1-2% at every pair count
   tested, 50-800 pairs) - the "smarter" search isn't currently earning its
   complexity (`experiments/benchmarks`).

## Findings by theme

### Pipeline sanity (001, 003)
Confirmed the train → analyze pipeline runs end-to-end and that adding
periodic progress snapshots doesn't perturb training.

### Full-scale baseline (002, 004)
800 points / 10,000 steps / `emb_dim=32` is the reference "good" run:
cross-val R² 0.96, Spearman 0.887, but NN recall only 0.326 and intrinsic
dimension 10.45 (finding #1/#2 above).

### Does NN recall ever plateau? (005)
50,000-step run: NN recall keeps climbing to 0.536 before visibly
bending into diminishing returns; intrinsic dimension keeps falling to
6.16. Also surfaced a new anomaly: PCA explained variance is
non-monotonic, peaking mid-training then declining even as R² stays flat
- a global-linear-summary (PCA) vs. local-nonlinear-estimate
(intrinsic dimension) disagreement that recurs later (077).

### `emb_dim` sweep (006-010)
See headline finding #3. Full comparison table across widths 8/12/16/24/32
lives in EXPERIMENTS.md §009/§010.

### Held-out pairs, confound isolation, dense coverage (011-019)
Established near-perfect pair-level generalization (#4) and separated
two confounded variables (step count vs. training-pool restriction) that
a first pass conflated - the effect of restricting the training pool to
90% of pairs turned out to *help* at `emb_dim=8` (016) but *cost* a small
amount at `emb_dim=24` (017), and both effects reversed/vanished once pair
coverage was densified to `n_points=200` (018-019).

### `n_points` × `train_examples` grid sweep (020-070)
50-run full cross of `n_points` ∈ {10,20,50,100,800} ×
`train_examples` ∈ {5k,...,50k} at fixed `emb_dim=24`. Raw numbers in
`experiments/manifold_learning/octahedron/grid_sweep_results.csv`.
Conclusions: `n_points ≤ 20` is too small for cross-validation/held-out
metrics to be meaningful (degenerate small-sample noise); `n_points=100`
is the sweet spot on this grid (highest, most stable cross-val R², lowest
intrinsic dimension); `n_points=800` is the only setting where
`train_examples` strongly gates whether the model learns the task at all
(finding #6).

### Point-level holdout generalization (071-075)
Five reruns of the file's best prior configs, adding the ability to hold
out entire points (not just pairs) and recover their embeddings from a
handful of probe distances. Full cross-run table in EXPERIMENTS.md §075.
Established finding #5: holdout-to-trained generalization is uniformly
strong (Spearman 0.95-0.997); holdout-to-holdout consistency depends on
the absolute number of same-region points left to anchor structure, not
the fraction held out; and `emb_dim` affects this the same way it affects
pair-level generalization (narrower width → larger bulk-quality cost,
weaker mutual consistency).

### Multi-task `same_triangle` classification (076-077)
Adding a second, near-perfectly-learnable task substantially degrades the
raw embedding table's linear decodability at a matched step budget, but
this is mostly (not entirely) a training-budget effect - 10x the steps
fully recovers (and then exceeds) NN recall/intrinsic dimension while
only partially recovering R²/Spearman (finding #7).

### `same_triangle_weight` sweep (078-087)
Ten-point sweep at a shared 20,000-step budget isolating the loss-weight
variable. `weight=0.2` dominates every metric but the relationship across
the full 0.1-1.0 range is noisy, not monotonic (finding #8). Full table
in EXPERIMENTS.md §087.

### Geodesic search benchmark (`experiments/benchmarks`)
Best-first search (`manifolds/polyhedra.py`) vs. plain BFS
(`manifolds/polyhedra_bfs.py`) on octahedron/icosahedron meshes, 50-800
pairs, 3 timed repeats, fixed seed: essentially tied everywhere (0.97x-1.01x)
(finding #9).

## Open questions / natural next steps

Pulled from the "Next steps" sections across the log - not yet run:

- Confirm `emb_dim=24`'s dominance over 32 (009) and the grid sweep's
  `n_points=800`/`train_examples=5000` cliff (070/060) both replicate at a
  second seed - every conclusion above comes from `seed=0` only.
- Rerun the `same_triangle_weight` sweep at 077's 100,000-step budget to
  check whether the noisy, non-monotonic pattern found at 20,000 steps is
  a converged property of each weight or a step-budget artifact.
- Test whether the point-holdout `emb_dim` effect (074 vs. 075) holds at
  `n_points=800`, and whether face-localized holdout (all current runs)
  behaves differently from uniformly-random point holdout - these are
  confounded in every run so far.
- Extend the `n_points × train_examples` grid with `n_points` ∈
  {200, 400} to pin down whether the coverage-density sweet spot sits
  closer to 100 or drifts higher.
- Investigate the PCA-explained-variance-vs-intrinsic-dimension
  disagreement directly (full explained-variance spectrum, not just top-3
  sum) - it has now appeared independently in three different sweep axes
  (training steps, `same_triangle_weight`, and the original 005 run).
