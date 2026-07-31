# Octahedron manifold-learning experiments

Tracks every run of `src/train_manifold.py` + `analysis/analysis_manifold.py`
on the octahedron manifold. Each experiment gets the next three-digit number
and its own result folder `experiments/manifold_learning/octahedron/<NNN>/`,
produced automatically by `analysis_manifold.py` (see workflow below).

## Workflow

1. Train: `python -m src.train_manifold configs/<config>.yaml`
   (always overwrites `models/octahedron_distance_model.pt` +
   `models/octahedron_distance_config.yaml` - these are scratch, not archives).
2. Pick the next three-digit experiment number and analyze:
   `python -m analysis.analysis_manifold <NNN>`
3. This archives everything under `experiments/manifold_learning/octahedron/<NNN>/`:
   - `config.yaml` - the exact config that produced the analyzed checkpoint
   - `metrics.txt` - every metric printed during analysis
   - `*.png` - all figures (PCA comparisons, CKA matrices, intrinsic dimension plots)
   - `progress.csv` + `training_curves.png` - embedding-quality metrics (PCA
     explained variance, coordinate R², distance Spearman correlation,
     nearest-neighbor recall, intrinsic dimension) snapshotted every
     `progress_interval` steps during training, so you can see how they
     developed over the run rather than just their final value. Only present
     if the config set `training.progress_interval` to a nonzero value (see
     `configs/octahedron_default.yaml`).
4. Add an entry below summarizing the run.

---

## 001 - Smoke-test pipeline sanity check

### 1. Model settings

Config: `configs/octahedron_smoke.yaml` (50 points, 500 steps, 2000 training
examples). Purpose: confirm the full train -> analyze pipeline runs
end-to-end and loss decreases, before committing to the multi-minute
geodesic precompute at the full `octahedron_default.yaml` scale (800 points).
Not a real capacity/quality run.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/001/`

- Pipeline runs end-to-end: checkpoint loads, world reconstructs from the
  geodesic-distance cache, all figures and metrics generate without errors.
- Pair-symmetry check passes tightly (max diff ~2.4e-7), confirming the
  distance head is symmetric as expected.
- In-sample linear coordinate probe R² = 0.99, but cross-validated R² drops
  to 0.14 ± 0.15 - large gap indicates overfitting, unsurprising at only 50
  entities / 500 steps.
- Geodesic-distance Spearman correlation = 0.85 (reasonable given the tiny
  scale); nearest-neighbor recall only 0.24.
- Embedding intrinsic dimension ~11.8, well above the manifold's true
  ambient_dim=3 - representations are not yet compact.
- Entity-token CKA rises from 0.12 (layer 1) to 0.49 (layer 2): later layers
  make i/j entity tokens more similar, expected as task-relevant information
  concentrates for the distance head.

---

## 002 - Full-scale run: octahedron geometry recovery

### 1. Model settings

Config: `configs/octahedron_default.yaml` (800 points, 10000 steps, 50000
training examples - the manifolds/README.md-recommended default for the
octahedron). Purpose: the first result meant to be interpreted as actual
geometry recovery, not pipeline sanity - does the transformer's learned
per-entity embedding recover the octahedron's true ambient surface at
realistic scale?

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/002/`

Training loss dropped from 0.132 (step 1) to 0.00003 (step 10000) - the
distance head fits the training distribution essentially exactly.
Pair-symmetry check still passes tightly (max diff ~2.4e-7).

Compared to the 001 smoke run (50 points, 500 steps):

- Cross-validated coordinate R² jumped from 0.14 ± 0.15 to **0.96 ± 0.004**,
  now nearly matching the in-sample R² (0.97 vs 0.97) - the previous
  train/cross-val gap was purely a data-starvation artifact of the smoke
  config, not a real generalization problem. At full scale the learned
  embedding linearly recovers the true ambient coordinates almost exactly.
- `linear_probe_reconstruction.png` visually reproduces the true octahedron's
  faceted-ring PCA silhouette (`true_manifold_pca.png`) closely, consistent
  with R²=0.968.
- Geodesic-distance Spearman correlation improved slightly (0.853 -> 0.887);
  PCA explained variance (3 comps) rose 0.567 -> 0.792.
- Nearest-neighbor recall improved but is still low (0.240 -> 0.326) - local
  neighborhood structure is only partially preserved even though global
  linear structure is recovered well.
- Embedding intrinsic dimension barely moved (11.77 -> 10.45), still far
  above ambient_dim=3: the *raw* embedding carries a lot of dimensionality
  a linear probe can ignore but that the intrinsic-dimension estimator can't.
  Geometry recovery here is "linearly decodable," not "the embedding
  collapses onto a 3D manifold."
- Entity-token CKA at layer 2 rose sharply (0.487 -> 0.752): the two entity
  tokens (i, j) become far more similar representations deeper in the
  network, as expected when task-relevant (distance) information dominates.
  Conversely, task-token/head CKA against `h2` at layer 1 *dropped*
  (0.653 -> 0.162), meaning the raw per-entity heads are less redundant with
  the shared task token at full scale than they were in the data-starved
  smoke run.
- Full-sequence intrinsic dimension *dropped* with scale (7.29 -> 4.60 at
  layer 2), the opposite direction from the raw entity-embedding intrinsic
  dimension - the task-relevant sequence representation compresses with more
  data even though the entity embedding table itself does not.

---

## 003 - Smoke rerun: validate periodic progress tracking

### 1. Model settings

Config: `configs/octahedron_smoke.yaml`, now with `training.progress_interval:
100` set. Purpose: not a new science result - this is a rerun of 001's exact
setup solely to exercise the new periodic embedding-quality snapshot feature
(`src/train_manifold.py` snapshots `compute_embedding_metrics` every
`progress_interval` steps into the checkpoint; `analysis_manifold.py` now
materializes that into `progress.csv` + `training_curves.png`).

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/003/`

Final-checkpoint metrics match 001 almost exactly (e.g. cv_r2_mean 0.137 vs
0.137, Spearman 0.853 vs 0.853), confirming the new snapshot code doesn't
perturb training. `training_curves.png` / `progress.csv` show all 6
embedding-quality metrics moving together with the loss: PCA explained
variance, linear R², cross-validated R², distance Spearman, and NN recall
all rise sharply between step 1 (near-random embeddings) and step 200, then
flatten - matching where the loss curve also flattens. Intrinsic dimension
falls from ~24 to ~12 over the same window and then plateaus, well above
ambient_dim=3, consistent with 001/002's finding that geometry recovery here
is linear-decodability, not literal dimensionality collapse.

### 3. Next steps

- Rerun `configs/octahedron_default.yaml` (progress_interval now set to 500)
  as the next full-scale experiment, to see whether the same "sharp rise
  then plateau" shape holds at 800 points/10000 steps, or whether the
  cross-val R² gap (see 002) closes gradually instead of early.

---

## 004 - Full-scale rerun with progress tracking

### 1. Model settings

Config: `configs/octahedron_default.yaml` (800 points, 10000 steps, same as
002), now with `training.progress_interval: 500` set (21 snapshot points).
Purpose: same checkpoint as 002 (identical seed, so final metrics match it
exactly) but now with the full training-progress curve, to answer 003's
open question about how the metrics develop at full scale.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/004/`

Final metrics match 002 exactly (deterministic, same seed) - see 002 for
that discussion. The new information is in `training_curves.png` /
`progress.csv`:

- Loss, PCA explained variance, linear R², cross-validated R², and geodesic
  Spearman correlation all rise sharply in the first ~500-2000 steps (5-20%
  of training) and are essentially flat for the remaining ~8000+ steps -
  the same "sharp rise then plateau" shape as the 003 smoke run, just
  compressed into a smaller fraction of total steps.
- **Nearest-neighbor recall does not plateau** - it climbs steadily and
  close to linearly from 0.0 at step 1 to 0.326 at step 10000, still rising
  at the last snapshot with no sign of flattening. This is the opposite
  shape from every other metric and answers 003's open question: the
  cross-val-R²-vs-NN-recall gap (see 002) does *not* close together - R²
  saturates almost immediately while local neighborhood structure keeps
  slowly improving throughout training.
- Embedding intrinsic dimension falls fastest early (23 -> 17 in the first
  500 steps) and keeps a slow downward drift the rest of the way (17 -> 10.4
  by step 10000), rather than plateauing early like R²/Spearman - closer in
  shape to the slow NN-recall climb than to the fast-saturating metrics.

### 3. Next steps

- The NN-recall-keeps-climbing-while-R²-saturates split suggests these two
  probes are sensitive to different things: R²/Spearman reward getting the
  *global* linear arrangement right (fast), while 1-NN recall needs *local*
  neighbor ordering to be exactly correct, which is a harder, slower-to-fit
  target. Worth a longer run (e.g. 30000-50000 steps) to see whether NN
  recall keeps rising or eventually plateaus too.
- Since intrinsic dimension tracks the slow NN-recall climb more than the
  fast-saturating metrics, check whether the two are causally linked - e.g.
  plot NN recall against intrinsic dimension directly (both are already in
  `progress.csv`) rather than only against step.

---

## 005 - Long run: does nearest-neighbor recall eventually plateau?

### 1. Model settings

New config `configs/octahedron_long.yaml`: identical to
`octahedron_default.yaml` (800 points, same seed, same architecture) except
`training.steps: 50000` (5x 004) and `progress_interval: 500` kept, giving
101 snapshot points. Purpose: directly answers 004's open question - does
nearest-neighbor recall (which was still climbing linearly at step 10000)
eventually plateau, or does it keep rising, and does the "linear R² fast /
NN-recall slow" split persist at 5x the training budget?

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/005/`

Loss keeps improving smoothly the whole run (0.132 -> ~0.00001), no
overfitting/instability at this budget.

- **Nearest-neighbor recall answers 004's question directly: it does not
  keep climbing linearly.** It rises steeply through ~step 15000 (0.0 ->
  ~0.45) then bends into a decelerating, diminishing-returns curve, still
  slowly climbing to 0.536 by step 50000 (up from 0.326 at step 10000 in
  004) but clearly flattening rather than continuing linearly - a
  "delayed plateau," not "no plateau."
- Embedding intrinsic dimension follows the same delayed-plateau shape as
  NN recall (as 004 speculated it might, given both were still moving at
  step 10000): it keeps falling well past step 10000, reaching 6.16 by step
  50000 (down from 10.45 at 10000 steps in 004) - much closer to
  ambient_dim=3 than any prior run - and is still gently decreasing at the
  final snapshot, not fully flat yet either.
- **New, unexpected finding: PCA explained variance is non-monotonic.** It
  rises with the other fast metrics to a peak of ~0.79 around step 8000-
  10000 (matching 004's plateau value), then *declines* over the remaining
  40000 steps to 0.688 at step 50000 - lower than 004's step-10000 value.
  Cross-validated R² and linear R² stay essentially flat and high (~0.975-
  0.98, matching/slightly beating 004) over that same window, so this isn't
  a loss of recoverability - the full 32-dim embedding still linearly
  encodes the true coordinates just as well. The top-3-PCA-component share
  of variance falling while full-dimensional R² holds steady is consistent
  with the embedding table spreading representational variance across more
  of its 32 dimensions as training continues, even as the *intrinsic*
  dimension (measured locally via TwoNN) keeps shrinking toward 3 - i.e.
  PCA (a global, linear summary) and TwoNN (a local, nonlinear one) are
  disagreeing about "compactness" in this later phase of training.

### 3. Next steps

- Investigate the PCA-explained-variance-vs-intrinsic-dimension disagreement
  directly: plot the full PCA explained-variance spectrum (not just the
  top-3 sum) at a few snapshots (10000, 25000, 50000) to see whether
  variance is migrating into specific higher components (e.g. components
  4-6, which would suggest a genuine higher-dimensional embedding of a
  low-curvature structure) or spreading roughly uniformly (which would look
  more like noise/regularization drift).
- NN recall and intrinsic dimension are both still moving (slowly) at step
  50000 - an even longer run (e.g. 100000-150000 steps) would show whether
  they fully converge, and to what final NN-recall value/intrinsic
  dimension, before committing further compute to this scale.
- Worth checking whether a smaller `emb_dim` (currently 32) reproduces the
  same delayed-plateau NN-recall/intrinsic-dimension shape faster, since
  much of the extra 40000 steps here may just be the model exploring
  dimensions a smaller embedding table wouldn't have available.

---

## 006 - Small emb_dim: does a tighter bottleneck reach the plateau faster?

### 1. Model settings

New config `configs/octahedron_small_emb.yaml`: identical to
`octahedron_long.yaml` (800 points, 50000 steps, same seed) except
`model.emb_dim: 8` instead of 32 (still divisible by `num_heads: 4`).
Purpose: directly tests 005's hypothesis - does a smaller embedding table,
with less spare capacity to spread variance into, reach the delayed
intrinsic-dimension/NN-recall plateau faster than the 40000+ steps it took
at `emb_dim=32`?

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/006/`

Loss still drops smoothly to a similar final magnitude (~0.00007), so the
distance-prediction task itself is still being fit well at this width.
Compared to 005 (`emb_dim=32`, same 50000 steps):

- **The hypothesis is confirmed for intrinsic dimension, sharply.** It falls
  from ~8 to ~3 within the first ~5000-10000 steps and stays flat there for
  the remaining 40000+ steps, ending at **3.09** - essentially exactly
  ambient_dim=3, and reached far faster and more completely than 005's slow
  drift to 6.16 that still hadn't fully flattened at step 50000. A tighter
  embedding table really does collapse to the true manifold dimension
  faster, as speculated.
- **But the "fast vs. slow" metric grouping flips entirely.** In every prior
  run (001-005), linear/cross-validated R² and Spearman correlation were
  the *fast*-saturating metrics (plateaued within the first ~5-20% of
  training) while NN recall and intrinsic dimension were the *slow* ones.
  Here it's the opposite: intrinsic dimension is now the fast one, while
  linear R² (0.622), cross-validated R² (0.608 ± 0.016), Spearman
  correlation (0.374), and NN recall (0.494) are all *still rising* at step
  50000 with no plateau in sight - and all land far below 005's final
  values (cv R² 0.975, Spearman 0.880, NN recall 0.536). An 8-dimensional
  embedding is simply too tight a bottleneck to also encode the full linear
  coordinate mapping the way a 32-dimensional one can, even though it
  matches the manifold's true dimensionality much more closely.
- PCA explained variance (top 3 of 8 dims) is correspondingly higher than
  005's top-3-of-32 share (0.727 vs 0.688) simply because there are fewer
  total dimensions for variance to spread across, not because recovery is
  better - it's the same top-heavy PCA spectrum on a much smaller total.

**Bottom line: shrinking `emb_dim` trades "intrinsic dimension matches the
manifold" for "the embedding is fully linearly decodable" - it does not get
both faster, it gets one at the cost of the other, at least at this width.**

### 3. Next steps

- Sweep `emb_dim` between 8 and 32 (e.g. 12, 16, 24) to find whether there's
  a width where intrinsic dimension still collapses close to 3 *without*
  as large a hit to R²/Spearman/NN-recall - i.e. is 8 past the useful edge
  of this tradeoff, or is the tradeoff smooth across the whole range?
  `octahedron_small_emb.yaml` is set up to make this a one-line edit.
- Since R²/Spearman/NN-recall are all still climbing at step 50000 here,
  rerun at a higher step count (e.g. 100000) specifically for `emb_dim=8`
  to see what they converge to, rather than assuming 005's larger-`emb_dim`
  final values as a ceiling.
- The full CKA/intrinsic-dimension figures for this run (`cka_task_tokens.png`,
  `intrinsic_dim_task_tokens.png`, etc.) haven't been compared against 005's
  yet - worth checking whether the transformer's internal representations
  (not just the raw embedding table) show the same fast/slow role reversal.

---

## 007 - emb_dim sweep, point 1: emb_dim=12

### 1. Model settings

New config `configs/octahedron_emb12.yaml`: identical to
`octahedron_long.yaml`/`octahedron_small_emb.yaml` (800 points, 50000 steps,
same seed) except `model.emb_dim: 12` - the first of three sweep points
(12, 16, 24) between 006's `emb_dim=8` and 005's `emb_dim=32`, per 006's
next steps. Purpose: find out whether a width between the two extremes gets
a low intrinsic dimension *and* good R²/Spearman/NN-recall, rather than
having to trade one for the other.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/007/`

Final metrics (`emb_dim=12`, 50000 steps) vs. 006 (`emb_dim=8`) vs. 005
(`emb_dim=32`):

| metric | emb=8 (006) | emb=12 (007) | emb=32 (005) |
|---|---:|---:|---:|
| cross-val R² | 0.608 ± 0.016 | **0.898 ± 0.013** | 0.975 ± 0.002 |
| Spearman | 0.374 | **0.707** | 0.880 |
| NN recall | 0.494 | 0.514 | 0.536 |
| intrinsic dim | 3.09 | **3.75** | 6.16 |

- `emb_dim=12` is a genuine middle ground, not just an average: cross-val R²
  (0.898) and Spearman (0.707) sit between the two extremes as expected, but
  intrinsic dimension (3.75) is much closer to 006's near-exact 3.09 than to
  005's 6.16 - a small amount of extra width buys most of the R²/Spearman
  quality back without giving up much of the "matches manifold dimension"
  property.
- **The 006 "fast/slow role reversal" mostly disappears at emb_dim=12.**
  Unlike 006 (where R²/Spearman/NN-recall never plateaued by step 50000),
  here loss, PCA explained variance, linear R², cross-val R², and Spearman
  all plateau by roughly step 15000-20000 (30-40% of training) - fast, like
  every emb_dim=32 run. Intrinsic dimension also plateaus in a similar
  window (~step 20000-25000) rather than staying slow. NN recall remains
  the one metric still gently climbing at step 50000, same as in every
  other run regardless of `emb_dim` - it looks like the one truly
  width-independent slow metric.
- So 006's "you trade one property for the other" framing was really about
  `emb_dim=8` being *too* tight, not a general law: at `emb_dim=12` the
  bottleneck is loose enough that most metrics recover their normal fast
  dynamics while still keeping intrinsic dimension low.

### 3. Next steps

- Continue the sweep (16, 24 - see 008, 009) to map out where cross-val R²
  and intrinsic dimension trade off most efficiently, and whether the
  "fast/slow reversal" reappears at any width or was specific to emb=8.

---

## 008 - emb_dim sweep, point 2: emb_dim=16

### 1. Model settings

New config `configs/octahedron_emb16.yaml`: identical to the other sweep
configs (800 points, 50000 steps, same seed) except `model.emb_dim: 16` -
the second of three sweep points (12, 16, 24).

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/008/`

Updated comparison table across the sweep so far:

| metric | emb=8 (006) | emb=12 (007) | emb=16 (008) | emb=32 (005) |
|---|---:|---:|---:|---:|
| cross-val R² | 0.608 ± 0.016 | 0.898 ± 0.013 | **0.939 ± 0.010** | 0.975 ± 0.002 |
| Spearman | 0.374 | 0.707 | **0.789** | 0.880 |
| NN recall | 0.494 | 0.514 | **0.519** | 0.536 |
| intrinsic dim | 3.09 | 3.75 | **4.47** | 6.16 |

- All four metrics move **monotonically and smoothly** with `emb_dim` across
  all four widths tried so far (8, 12, 16, 32) - no width tried yet
  reverses the trend or does something unexpected. This is a clean, well-
  behaved tradeoff curve, not a cliff or a sweet spot.
- The "fast/slow" story is more nuanced than 007 suggested, not just "006's
  reversal disappears at wider `emb_dim`": at `emb_dim=16`, loss/PCA
  variance/linear R²/cross-val R² still plateau fast (~step 10000), same as
  007 - but **Spearman correlation and intrinsic dimension are both still
  slowly drifting at step 50000** here (Spearman 0.70 -> 0.79 over the back
  half of training; intrinsic dim still gently falling past step 40000),
  unlike 007 where nearly everything but NN recall had flattened by step
  20000-25000. So the *set* of slow-to-converge metrics changes with
  `emb_dim` (only NN recall at 12; NN recall + Spearman + intrinsic dim at
  16) rather than there being one fixed "fast group" and "slow group."
- No sign yet of intrinsic dimension "catching up" to R²/Spearman quality at
  a shared width - going from 12 to 16 buys +0.041 cross-val R² and +0.082
  Spearman at the cost of +0.72 intrinsic dimension, a similar rate of
  exchange to 8->12's +0.29 R² / +0.33 Spearman for +0.66 intrinsic
  dimension. Nothing suggests a kink or elbow in this range yet - see 009
  (emb=24) for whether one appears closer to the 32-dim end.

### 3. Next steps

- Run 009 (`emb_dim=24`) and check whether the monotonic, smooth trend
  continues all the way to 32, or whether there's a kink closer to the
  wide end - i.e. is the R²-vs-intrinsic-dimension tradeoff linear-ish in
  `emb_dim`, or does it flatten out (diminishing returns to extra width)
  before reaching 32?
- Now that Spearman and intrinsic dimension are the ones still moving at
  step 50000 for `emb_dim=16` (not just NN recall as in 007), it may be
  worth a longer run at this specific width to see their true converged
  values, rather than assuming step-50000 is close enough to convergence
  across the whole sweep.

---

## 009 - emb_dim sweep, point 3: emb_dim=24

### 1. Model settings

New config `configs/octahedron_emb24.yaml`: identical to the other sweep
configs (800 points, 50000 steps, same seed) except `model.emb_dim: 24` -
the third and last of the planned sweep points (12, 16, 24), sitting
closest to 005's `emb_dim=32`.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/009/`

Full sweep comparison table:

| metric | emb=8 (006) | emb=12 (007) | emb=16 (008) | emb=24 (009) | emb=32 (005) |
|---|---:|---:|---:|---:|---:|
| cross-val R² | 0.608 ± 0.016 | 0.898 ± 0.013 | 0.939 ± 0.010 | **0.980 ± 0.001** | 0.975 ± 0.002 |
| Spearman | 0.374 | 0.707 | 0.789 | **0.894** | 0.880 |
| NN recall | 0.494 | 0.514 | 0.519 | **0.560** | 0.536 |
| intrinsic dim | 3.09 | 3.75 | 4.47 | **5.42** | 6.16 |

**Headline finding: the smooth, monotonic tradeoff from 007/008 breaks at
emb_dim=24 - and not in the direction expected.** Going from 8 through 16,
every extra dimension bought better R²/Spearman/NN-recall at the cost of
higher intrinsic dimension, as if on a single tradeoff curve. `emb_dim=24`
does not continue that curve toward 32's endpoint - **it strictly dominates
`emb_dim=32` on every metric simultaneously**: higher cross-val R² (0.980 vs
0.975), higher Spearman (0.894 vs 0.880), higher NN recall (0.560 vs 0.536),
*and* lower intrinsic dimension (5.42 vs 6.16). There is no metric on which
`emb_dim=32` beats `emb_dim=24` at this training budget.

- `training_curves.png` shows a fully converged picture at `emb_dim=24`:
  unlike 008 (`emb=16`, where Spearman and intrinsic dimension were still
  moving at step 50000), here NN recall and intrinsic dimension both
  visibly flatten by roughly step 25000-30000 (half of training), and
  everything else plateaus earlier still - the cleanest, most fully-settled
  curve set of the whole sweep.
- The most likely explanation is that `emb_dim=32` (005) is simply
  under-trained *relative to its own capacity* at this 50000-step budget -
  a wider embedding table has more parameters to fit and may need more
  steps (or more `train_examples`) to reach the same degree of convergence
  a narrower one reaches faster - rather than 32 being a genuinely worse
  width in the limit. This directly contradicts 007/008's working
  assumption that all runs so far were "close enough to converged" for a
  fair comparison at step 50000.
- This reframes the whole sweep's conclusion: the earlier 8->12->16 story
  ("smaller `emb_dim` trades linear decodability for a lower intrinsic
  dimension") still holds internally, but it is not safe to conclude
  `emb_dim=32` is the best "full quality" endpoint of that tradeoff - 24
  beats it outright here, most plausibly because of undertraining at 32,
  not because 24 is fundamentally the better width.

### 3. Next steps

- Before trusting any single "best `emb_dim`" conclusion, rerun
  `octahedron_long.yaml` (`emb_dim=32`) for more than 50000 steps (e.g.
  100000-150000) to check whether it eventually catches up to or surpasses
  009's numbers - this is the direct test of the undertraining explanation
  above.
- If 32 does catch up given enough steps, the sweep's real finding becomes
  "wider `emb_dim` needs proportionally more training steps to converge,"
  not "24 is the best width" - worth stating that distinction clearly once
  the longer 32-dim run exists.
- Consider re-running the whole sweep (or at least 24 and 32) at a second
  seed to check whether emb=24's dominance over emb=32 is a robust effect
  or a single-seed fluctuation, since every run so far has used `seed: 0`
  for both the manifold sampling and training.

---

## 010 - Does emb_dim=32 catch up with 2x the training steps?

### 1. Model settings

New config `configs/octahedron_emb32_100k.yaml`: identical to
`octahedron_long.yaml` (`emb_dim=32`, 800 points, same seed) except
`training.steps: 100000` (2x 005's 50000), `progress_interval: 500` kept
(201 snapshot points). Purpose: direct test of 009's undertraining
hypothesis - does `emb_dim=32` eventually catch up to or surpass 009's
`emb_dim=24` @ 50000-step numbers given twice the training budget?

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/010/`

Full comparison table, including both `emb_dim=32` runs:

| metric | emb=16 (008) | emb=24 @50k (009) | emb=32 @50k (005) | emb=32 @100k (010) |
|---|---:|---:|---:|---:|
| cross-val R² | 0.939 ± 0.010 | **0.980 ± 0.001** | 0.975 ± 0.002 | 0.975 ± 0.002 |
| Spearman | 0.789 | **0.894** | 0.880 | 0.864 |
| NN recall | 0.519 | **0.560** | 0.536 | 0.553 |
| intrinsic dim | 4.47 | **5.42** | 6.16 | 5.78 |

**The undertraining hypothesis is only partly right - doubling the steps
closes some of the gap to `emb_dim=24`, but not all of it, and not evenly
across metrics:**

- NN recall and intrinsic dimension - the two metrics that `training_curves.png`
  showed were *still moving* at step 50000 for `emb_dim=32` (005) - both
  improved as predicted (NN recall 0.536 -> 0.553; intrinsic dim 6.16 ->
  5.78) and both now visibly plateau by roughly step 60000-70000 in the
  100000-step curve. So for these two metrics, 005 genuinely was
  undertrained, exactly as 009 speculated.
- Cross-validated R² did not move at all (0.9754 at both 50000 and 100000
  steps, to four decimal places) - it had already fully converged by step
  50000, so extra steps had nothing left to give it.
- Spearman correlation actually went slightly *down* (0.880 -> 0.864) with
  more training, not up - it was already flat/plateaued by step 10000 in
  both runs, so this is most likely noise around a converged value rather
  than a real regression, but it means Spearman offers no support for the
  undertraining story either.
- **Net result: even at 2x the steps, `emb_dim=32` still does not catch
  `emb_dim=24`'s 50000-step numbers on any of the four metrics.** The gap
  narrowed (most on NN recall/intrinsic dim, the two that were genuinely
  still converging) but did not close. This means 009's "24 beats 32 purely
  because 32 is undertrained" explanation is not the full story - either
  32 needs substantially more than 2x the steps, or `emb_dim=24` really is
  a better width than 32 for this task/data scale, not just a better-
  converged one.

### 3. Next steps

- The remaining gap is now small enough on NN recall/intrinsic-dim (and
  cv R² is fully flat) that a much longer run is unlikely to change the
  overall conclusion much - probably not worth another straight step-count
  increase. More informative next moves: (a) the second-seed check from
  009's next steps, to rule out this being seed-0-specific, or (b) sweeping
  `train_examples` instead of `steps`, since 005/009/010 have all used the
  same fixed 50000 training pairs regardless of step count - a wider
  embedding table might benefit more from seeing more distinct pairs than
  from more epochs over the same pairs.
- Given cv R² and Spearman are now confirmed flat well before step 50000
  for every `emb_dim` tried, future sweeps over architecture (emb_dim,
  num_layers, hidden_dim) could safely use a shorter step budget (e.g.
  20000-30000) for the *linear-decodability* metrics, reserving a longer
  budget only for runs specifically studying NN recall/intrinsic dimension.

---

## 011 - emb_dim sweep rerun, point 1/5: emb_dim=8, 20000 steps, held-out pairs

### 1. Model settings

New config `configs/octahedron_emb8_20k.yaml`: same as
`octahedron_small_emb.yaml` (`emb_dim=8`, 800 points, seed 0) except
`training.steps: 20000` (vs. 50000 in 006) and the newly added
`training.test_fraction: 0.1`. Purpose: first use of the held-out pair
generalization check (added to `src/train_manifold.py` /
`src/datasets.py:split_pairs` /
`analysis/analysis_manifold.py`'s new "HELD-OUT PAIR GENERALIZATION"
section) - 10% of all C(800,2)=319600 point pairs (31960 pairs) are
reserved before sampling, `train_examples` stays 50000 as in every prior
run, and after training the transformer's raw distance-prediction loss
(same smooth-L1 used for training) plus Spearman correlation are computed
separately on the held-out pairs and on a same-size fresh in-distribution
sample from the training pool. This also re-runs 006-009/005's emb_dim
sweep (8/12/16/24/32) at a shared, shorter 20000-step budget (011-015),
per 010's next-steps note that cv R²/Spearman are flat well before step
50000 so a shorter budget is enough for those metrics.

**Caveat that applies to every entry in this rerun (011-015):** the
comparison to the original 50000-step sweep (005-009) is confounded by
two simultaneous changes, not just the step-count cut - `train_examples`
is now drawn from a 90% pool instead of the full pair space, via a
different sampling mechanism (`allowed_pairs` index draws vs.
`rng.choice(..., replace=False)`). The vs.-baseline numbers below are
suggestive, not a clean steps-only ablation; the held-out-vs-in-distribution
comparison *within* each of these five runs is the clean, apples-to-apples
result.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/011/`

- **Held-out generalization is essentially perfect at this width:**
  held-out loss 0.00007 vs. in-distribution loss 0.00006 (31960 pairs
  each), held-out Spearman 0.9987 vs. in-distribution 0.9987. The
  transformer's distance predictions for pairs it never trained on are
  indistinguishable from its predictions on pairs drawn from the same
  pool it did train on - no sign of the model memorizing specific
  training pairs rather than learning the underlying metric.
- Embedding-table metrics vs. 006 (`emb_dim=8`, 50000 steps, no held-out
  split): cross-val R² 0.8255 ± 0.0192 (vs. 0.608 ± 0.016), Spearman
  0.6142 (vs. 0.374), NN recall 0.4975 (vs. 0.494, essentially flat),
  intrinsic dimension 3.5972 (vs. 3.09). Cross-val R² and Spearman are
  both *much* higher here despite 2.5x fewer steps - given the confound
  noted above (restricted pool + different sampling order), and that 006
  itself found `emb_dim=8` was still far from converged at step 50000
  with a highly unstable/still-rising trajectory, this reads as sampling-
  order sensitivity in an already-noisy regime rather than "20000 steps
  is intrinsically better than 50000" - see 015 for the same comparison at
  a width where training was well converged, which cleanly disagrees with
  that reading.

### 3. Next steps

- Continue the rerun across the remaining widths (012-015) before drawing
  any conclusion about the vs.-baseline gap - a single width isn't enough
  to tell confound-driven noise from a real effect.

---

## 012 - emb_dim sweep rerun, point 2/5: emb_dim=12, 20000 steps, held-out pairs

### 1. Model settings

New config `configs/octahedron_emb12_20k.yaml`: same as
`octahedron_emb12.yaml` (`emb_dim=12`) except `training.steps: 20000`
(vs. 50000 in 007) and `training.test_fraction: 0.1` - see 011 for the
held-out-evaluation feature and the confound caveat that applies to every
entry in this rerun.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/012/`

- **Held-out generalization again essentially perfect:** held-out loss
  0.00003 vs. in-distribution loss 0.00002 (31960 pairs each), held-out
  Spearman 0.9994 vs. in-distribution 0.9995.
- Embedding-table metrics vs. 007 (`emb_dim=12`, 50000 steps): cross-val
  R² 0.9310 ± 0.0264 (vs. 0.898 ± 0.013), Spearman 0.8633 (vs. 0.707), NN
  recall 0.4988 (vs. 0.514), intrinsic dimension 4.3333 (vs. 3.75). Same
  pattern as 011: higher cross-val R²/Spearman at 20000 steps than 007 saw
  at 50000, NN recall roughly flat, intrinsic dimension higher (less
  compact) here.

### 3. Next steps

- See 015 for the full five-point table and a wider-width check of the
  "20000-step rerun beats the 50000-step baseline" pattern seen at 8 and
  12 so far.

---

## 013 - emb_dim sweep rerun, point 3/5: emb_dim=16, 20000 steps, held-out pairs

### 1. Model settings

New config `configs/octahedron_emb16_20k.yaml`: same as
`octahedron_emb16.yaml` (`emb_dim=16`) except `training.steps: 20000`
(vs. 50000 in 008) and `training.test_fraction: 0.1` - see 011 for the
held-out-evaluation feature and the confound caveat.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/013/`

- **Held-out generalization again essentially perfect:** held-out loss
  0.00002 vs. in-distribution loss 0.00002 (31960 pairs each), held-out
  Spearman 0.9995 vs. in-distribution 0.9995.
- Embedding-table metrics vs. 008 (`emb_dim=16`, 50000 steps): cross-val
  R² 0.9614 ± 0.0036 (vs. 0.939 ± 0.010), Spearman 0.8736 (vs. 0.789), NN
  recall 0.5188 (vs. 0.519, flat), intrinsic dimension 5.2016 (vs. 4.47).
  Third width in a row where the 20000-step rerun beats 008's 50000-step
  cross-val R²/Spearman while NN recall stays flat and intrinsic dimension
  rises - the pattern from 011/012 continues here too.

### 3. Next steps

- See 015 for the full comparison, including whether this pattern holds
  at 24 and 32 (the two widest points, where 009's original sweep found
  `emb_dim=24` outright dominating `emb_dim=32` - a different anomaly this
  rerun could either confirm or complicate).

---

## 014 - emb_dim sweep rerun, point 4/5: emb_dim=24, 20000 steps, held-out pairs

### 1. Model settings

New config `configs/octahedron_emb24_20k.yaml`: same as
`octahedron_emb24.yaml` (`emb_dim=24`) except `training.steps: 20000`
(vs. 50000 in 009) and `training.test_fraction: 0.1` - see 011 for the
held-out-evaluation feature and the confound caveat.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/014/`

- **Held-out generalization again essentially perfect:** held-out loss
  0.00002 vs. in-distribution loss 0.00002 (31960 pairs each), held-out
  Spearman 0.9997 vs. in-distribution 0.9997 - the highest held-out
  Spearman of the sweep so far.
- Embedding-table metrics vs. 009 (`emb_dim=24`, 50000 steps): cross-val
  R² 0.9619 ± 0.0034 (vs. **0.980 ± 0.001**), Spearman 0.8637 (vs.
  **0.894**), NN recall 0.5262 (vs. **0.560**), intrinsic dimension
  6.5150 (vs. 5.42). **This breaks the pattern from 011-013**: here the
  original 50000-step run (009) beats the 20000-step rerun on every
  metric, the opposite of what happened at emb_dim 8/12/16. 009 was
  already the sweep's standout result (it out-performed even
  `emb_dim=32` at 50000 steps); this suggests `emb_dim=24` specifically
  benefits from the longer step budget in a way the narrower widths did
  not, rather than the "20000 steps is generally enough" reading 011-013
  alone would have suggested.

### 3. Next steps

- The 8/12/16-vs-24 split is now the most interesting open question from
  this rerun: is `emb_dim=24` just slower to converge (needs the extra
  30000 steps 009 gave it), or is something else going on at that specific
  width/step-count combination? A longer rerun of `octahedron_emb24_20k`-
  style config at, say, 35000-50000 steps (with the held-out split kept)
  would directly test whether it catches back up to 009.

---

## 015 - emb_dim sweep rerun, point 5/5: emb_dim=32, 20000 steps, held-out pairs

### 1. Model settings

New config `configs/octahedron_emb32_20k.yaml`: same as
`octahedron_long.yaml` (`emb_dim=32`) except `training.steps: 20000`
(vs. 50000 in 005, 100000 in 010) and `training.test_fraction: 0.1` - see
011 for the held-out-evaluation feature and the confound caveat. Last
point in the 011-015 rerun, closing out the sweep.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/015/`

- **Held-out generalization again essentially perfect:** held-out loss
  0.00002 vs. in-distribution loss 0.00002 (31960 pairs each), held-out
  Spearman 0.9996 vs. in-distribution 0.9996.
- Embedding-table metrics vs. 005 (`emb_dim=32`, 50000 steps): cross-val
  R² 0.9752 ± 0.0020 (vs. 0.975 ± 0.002, essentially tied), Spearman
  0.9006 (vs. 0.880, slightly better), **NN recall 0.4863 (vs. 0.536,
  clearly worse)**, intrinsic dimension 7.5140 (vs. 6.16, worse/less
  compact). Unlike 8/12/16 (rerun beats baseline) or 24 (baseline beats
  rerun), `emb_dim=32` is a split decision: cross-val R²/Spearman - the
  metrics 010 already showed saturate early - are matched or slightly
  better at 20000 steps, while NN recall and intrinsic dimension - the two
  metrics 004/005/010 repeatedly found were still slowly moving well past
  step 20000 - are clearly worse, consistent with every prior finding that
  those two specifically need more steps regardless of width.

**Full five-point sweep, 20000-step rerun (011-015) vs. original 50000-step
sweep (006/007/008/009/005):**

| metric | emb=8 | emb=12 | emb=16 | emb=24 | emb=32 |
|---|---:|---:|---:|---:|---:|
| cross-val R² @20k (011-015) | 0.8255 | 0.9310 | 0.9614 | 0.9619 | 0.9752 |
| cross-val R² @50k (006-009,005) | 0.608 | 0.898 | 0.939 | **0.980** | 0.975 |
| Spearman @20k | 0.6142 | 0.8633 | 0.8736 | 0.8637 | 0.9006 |
| Spearman @50k | 0.374 | 0.707 | 0.789 | **0.894** | 0.880 |
| NN recall @20k | 0.4975 | 0.4988 | 0.5188 | 0.5262 | 0.4863 |
| NN recall @50k | 0.494 | 0.514 | 0.519 | **0.560** | 0.536 |
| intrinsic dim @20k | 3.5972 | 4.3333 | 5.2016 | 6.5150 | 7.5140 |
| intrinsic dim @50k | 3.09 | 3.75 | 4.47 | 5.42 | 6.16 |
| held-out loss @20k | 0.00007 | 0.00003 | 0.00002 | 0.00002 | 0.00002 |
| in-distribution loss @20k | 0.00006 | 0.00002 | 0.00002 | 0.00002 | 0.00002 |
| held-out Spearman @20k | 0.9987 | 0.9994 | 0.9995 | 0.9997 | 0.9996 |

**Headline finding: held-out pair generalization is excellent at every
width tested (8 through 32).** Held-out loss tracks in-distribution loss
to within 0.00001 at every point in the sweep, and held-out Spearman
correlation never drops below 0.9987 - the distance-only transformer's
raw task predictions generalize to unseen pairs essentially as well as to
pairs from its own training pool, regardless of embedding width. This is
the clean, non-confounded result of this rerun: unlike the vs.-baseline
comparisons above (which mix a step-count change with a training-pool-
restriction change), the held-out-vs-in-distribution comparison uses the
identical sampling mechanism and sample size within each run, differing
only in pool membership.

The vs.-50000-step-baseline table is messier and inconsistent across
widths (20k beats 50k at 8/12/16, 50k beats 20k at 24, roughly ties at
32) - given the confound noted in 011, this is not strong evidence about
step count in isolation. It does reproduce one thing cleanly: NN
recall/intrinsic dimension are worse at 20000 steps than 50000 at every
width without exception, consistent with 004/005/010's repeated finding
that those two metrics specifically are the slow-converging ones,
independent of `emb_dim`.

### 3. Next steps

- Isolate the confound directly: rerun at least one width (e.g.
  `emb_dim=32`) for 50000 steps *with* the held-out split enabled, and
  compare directly to 005's original 50000-step, no-split run - this
  separates "held-out pool restriction changed the outcome" from "20000
  vs. 50000 steps changed the outcome," which the 011-015 numbers alone
  cannot.
- Follow up specifically on 014's reversal (`emb_dim=24` uniquely favoring
  more steps) with a longer rerun at that width, per 014's next steps.
- The held-out-generalizes-perfectly finding is itself worth stress-
  testing at a harder setting than this sweep provides - e.g. a much
  smaller `train_examples` budget (where memorization would be more
  plausible) or a smaller manifold `n_points` (fewer points to spread
  50000 training pairs over), to check whether the near-zero
  generalization gap seen here is a general property of this task or a
  side effect of `train_examples=50000` already covering a large fraction
  of the ~287640-pair training pool at `n_points=800`.

---

## 016 - Confound isolation, point 1/2: emb_dim=8, 50000 steps, held-out pairs

### 1. Model settings

Reran the existing `configs/octahedron_small_emb.yaml` unchanged
(`emb_dim=8`, 800 points, 50000 steps, seed 0 - identical to 006) - no new
config file needed, since `training.test_fraction` now defaults to 0.1 in
code (`src/train_manifold.py`) when a config doesn't set it explicitly.
Purpose: 011's confound caveat noted that the 20000-step rerun (011-015)
changed *two* things at once versus the original 50000-step sweep
(005-009) - fewer steps *and* a training pool restricted to 90% of pairs
via a different sampling mechanism. This run isolates the second variable
alone: same 50000 steps as 006, held-out split now enabled. This run also
uses the newly added per-step held-out/in-distribution tracking (added
after 011-015) - `training_curves.png` now includes the held-out-vs-in-
distribution loss/Spearman panels across the whole run, not just a final
checkpoint value.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/016/`

- **Held-out generalization remains essentially perfect:** held-out loss
  0.00003 vs. in-distribution loss 0.00003 (31960 pairs each), held-out
  Spearman 0.9995 vs. in-distribution 0.9995 - matching every other width
  tested so far.
- **This is not a wash with 006 - it's a clear improvement, on every
  metric:**

  | metric | emb=8, 50k, no split (006) | emb=8, 20k, split (011) | emb=8, 50k, split (016) |
  |---|---:|---:|---:|
  | cross-val R² | 0.608 ± 0.016 | 0.8255 ± 0.0192 | **0.8985 ± 0.0117** |
  | Spearman | 0.374 | 0.6142 | **0.7211** |
  | NN recall | 0.494 | 0.4975 | **0.6687** |
  | intrinsic dim | 3.09 | 3.5972 | **2.9818** |

  016 beats both 006 (same steps, no split) and 011 (same split, fewer
  steps) on every single metric - including NN recall, which jumps to
  0.6687, higher than *any* other run in the entire sweep to date
  (previous best was 009's 0.560 at `emb_dim=24`). Intrinsic dimension
  also lands closer to the true `ambient_dim=3` than any prior `emb_dim=8`
  run.
- This resolves 011's open confound question in one direction, for this
  width at least: **the pool restriction did not hurt** - if anything,
  holding out 10% of pairs and training for the full 50000 steps produced
  the best `emb_dim=8` result seen so far, better than the unrestricted-
  pool 006 baseline at the same step count. Whatever changed (removing a
  slice of the hardest-to-learn pairs from training by chance, a
  different effective sample order interacting favorably with this
  particular narrow/tight-bottleneck optimization landscape, or plain
  seed-lottery variance) is not explained by this single run.
- `training_curves.png`'s new held-out-vs-in-distribution panels show the
  two curves overlapping essentially exactly for the entire run (loss and
  Spearman both, at every one of the 101 snapshots) - the held-out
  generalization gap isn't just small at the final checkpoint, it stays
  flat at ~zero throughout training, including in the first ~2000 steps
  while the model is still rapidly improving from its random-init
  baseline.

### 3. Next steps

- See 017 for the same isolation test at `emb_dim=24`, the width where
  009 (no split) most clearly beat every subsequent split/shorter-step
  variant - if the pattern reverses there, the pool-restriction effect is
  width-dependent, not a uniform "held-out split helps" story.
- The magnitude of 016's jump over 006 is large enough that it's worth
  checking whether it replicates at a second seed before treating it as a
  real effect of the held-out split rather than training-noise variance
  in an already-unstable (`emb_dim=8`, still-not-fully-converged per 006)
  regime.

---

## 017 - Confound isolation, point 2/2: emb_dim=24, 50000 steps, held-out pairs

### 1. Model settings

Reran the existing `configs/octahedron_emb24.yaml` unchanged (`emb_dim=24`,
800 points, 50000 steps, seed 0 - identical to 009), for the same reason
as 016: `training.test_fraction` defaults to 0.1 without needing a config
change. Purpose: 014 found that `emb_dim=24` was the one width where the
20000-step held-out rerun (014) clearly *underperformed* the original
50000-step baseline (009) on every metric, unlike every other width. This
run directly follows up on 014's next-steps question - does `emb_dim=24`
catch up to 009 once it gets the full 50000-step budget back, with the
held-out split still enabled?

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/017/`

- **Held-out generalization remains essentially perfect:** held-out loss
  0.00001 vs. in-distribution loss 0.00001 (31960 pairs each), held-out
  Spearman 0.9998 vs. in-distribution 0.9998 - the highest held-out
  Spearman of the entire sweep so far.
- **Partial recovery, not full - and the opposite pattern from 016:**

  | metric | emb=24, 50k, no split (009) | emb=24, 20k, split (014) | emb=24, 50k, split (017) |
  |---|---:|---:|---:|
  | cross-val R² | **0.980 ± 0.001** | 0.9619 ± 0.0034 | 0.9646 ± 0.0033 |
  | Spearman | **0.894** | 0.8637 | 0.8416 |
  | NN recall | **0.560** | 0.5262 | 0.5575 |
  | intrinsic dim | 5.42 | 6.5150 | 5.5596 |

  Going from 014 (20k, split) to 017 (50k, split) recovers most of the way
  toward 009 on cross-val R² (+0.0027) and NN recall (+0.0313, now
  essentially tied with 009), and intrinsic dimension improves
  substantially (6.5150 -> 5.5596, close to 009's 5.42) - consistent with
  014's "just needs more steps" hypothesis for those three metrics.
  **Spearman is the exception: it goes the wrong way** (0.8637 -> 0.8416,
  further from 009's 0.894, not closer) - more steps did not help this
  particular metric at this width, unlike everywhere else in the sweep.
  Overall 017 still falls short of 009's original numbers on cross-val
  R² and Spearman specifically, so the pool restriction (not just step
  count) does appear to cost something real at this width - **the
  opposite conclusion from 016**, where the same pool restriction at
  `emb_dim=8` produced a strictly better result than the unrestricted
  baseline.
- `training_curves.png`'s held-out-vs-in-distribution panels show the same
  flat-zero gap as 016 for the entire run - loss and Spearman curves
  overlap essentially exactly at every snapshot. So 017's shortfall
  against 009 is entirely about *overall* task/embedding quality at this
  width and step count, not about the held-out set specifically becoming
  harder than the in-distribution one at any point during training.

### 3. Next steps

- 016 and 017 together show the held-out split's effect on final-metric
  quality is not uniform across `emb_dim` - helps substantially at 8,
  costs a bit at 24 (mainly Spearman). Testing `emb_dim=32` the same way
  (50000 steps, split enabled, direct comparison to 005) would establish
  whether this is a low-width-vs-high-width split or specific to these
  two points.

## 018 - Dense-coverage stress test, point 1/2: emb_dim=8, n_points=200, held-out pairs

### 1. Model settings

New config `configs/octahedron_emb8_n200.yaml`: same `emb_dim=8`/50000-step
setup as 016 (`octahedron_small_emb.yaml`), but `manifold.n_points: 200`
instead of 800. With `n_points=200`, `C(200,2)=19900` total pairs (~17910
in the 90% train pool) against the same `train_examples=50000`, so each
train-pool pair is drawn ~2.8 times on average - versus ~0.17 times at
`n_points=800` - a much denser, more memorization-prone training signal.
Purpose: directly test whether 016's near-perfect held-out generalization
survives when the model can no longer rely on sparse coverage alone to
force generalization.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/018/`

- **Held-out generalization still essentially perfect under dense
  coverage:** held-out loss 0.00002 vs. in-distribution loss 0.00001
  (1990 pairs each), held-out Spearman 0.9997 vs. in-distribution 0.9997 -
  the denser training signal does not break generalization at this width.
- **Denser coverage is a strict improvement over 016 on every metric,
  not just a wash:**

  | metric | emb=8, n=800, split (016) | emb=8, n=200, split (018) |
  |---|---:|---:|
  | cross-val R² | 0.8985 ± 0.0117 | **0.9496 ± 0.0044** |
  | Spearman | 0.7211 | **0.8641** |
  | NN recall | 0.6687 | **0.7600** (new record for the whole sweep) |
  | intrinsic dim | 2.9818 | **2.0812** (closest yet to a 2D surface) |

  018's NN recall (0.76) is the best of any run in the sweep to date
  (previous best was 016's 0.6687), and intrinsic dimension moves
  noticeably closer to 2 - consistent with the octahedron being a 2D
  surface - rather than 016's ~3. Denser pair coverage, at fixed step
  budget, appears to help the model learn a cleaner embedding, not just
  memorize the training pairs.

### 3. Next steps

- Repeat the same `n_points=200` densification at `emb_dim=24` (017's
  width, where the held-out split previously cost something relative to
  the unrestricted baseline) to see whether denser coverage similarly
  offsets or reverses that cost - see 019.

## 019 - Dense-coverage stress test, point 2/2: emb_dim=24, n_points=200, held-out pairs

### 1. Model settings

New config `configs/octahedron_emb24_n200.yaml`: same `emb_dim=24`/
50000-step setup as 017 (`octahedron_emb24.yaml`), but `manifold.n_points:
200` instead of 800 - identical densification as 018, applied to 017's
width instead of 016's. Purpose: 017 found that `emb_dim=24` was the one
width so far where enabling the held-out split cost something relative to
the unrestricted baseline (009), unlike `emb_dim=8` (016), where the same
split was a strict win. This run tests whether that emb_dim=24 shortfall
is a real width effect or an artifact of sparse (n=800) pair coverage.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/019/`

- **Held-out generalization sets a new record for the sweep:** held-out
  loss 0.00001 vs. in-distribution loss 0.00000 (1990 pairs each), held-out
  Spearman 0.9999 vs. in-distribution 0.9999 - the highest held-out
  Spearman observed so far (previous best was 017's 0.9998).
- **Denser coverage reverses 017's shortfall - `emb_dim=24` now improves
  on every metric too, not just `emb_dim=8`:**

  | metric | emb=24, n=800, split (017) | emb=24, n=200, split (019) |
  |---|---:|---:|
  | cross-val R² | 0.9646 ± 0.0033 | **0.9705 ± 0.0026** |
  | Spearman | 0.8416 | **0.8930** |
  | NN recall | 0.5575 | **0.7350** |
  | intrinsic dim | 5.5596 | **2.6462** |

  NN recall jumps by +0.1775 and intrinsic dimension drops from 5.56 to
  2.65 - much closer to the octahedron's 2D surface - mirroring 018's
  pattern at `emb_dim=8`. This resolves 017's open question: the earlier
  `emb_dim=24` shortfall against the unrestricted baseline was a sparse-
  coverage artifact, not an intrinsic width effect - once pair coverage is
  denser, both widths tested so far benefit from the held-out split
  rather than being hurt by it.
- 019 (NN recall 0.7350) still trails 018 (0.7600) slightly, but leads on
  cross-val R² (0.9705 vs. 0.9496) and Spearman (0.8930 vs. 0.8641) - the
  two widths are now much closer together than 016 vs. 017 were, with no
  clear winner across all metrics.

### 3. Next steps

- Both widths tested (8 and 24) now show the same qualitative result:
  denser pair coverage (n_points=200) improves every metric relative to
  the sparse (n_points=800) held-out baseline. Testing `emb_dim=32` at
  `n_points=200` (direct densified analog of 015/005) would confirm this
  holds across the full width range, not just the two points checked so
  far.
- Alternatively, sweep `n_points` itself at a fixed `emb_dim` (e.g. 200 vs.
  400 vs. 800) to find where the benefit saturates, rather than continuing
  to add new `emb_dim` points.

## 020 - Grid sweep: n_points=10, train_examples=5000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n10_te5000.yaml` (n_points=10, train_examples=5000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/020/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6782 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | -1.9190 ± 3.5282 |
| Geodesic Spearman | 0.7663 |
| NN recall | 0.5000 |
| Intrinsic dimension | 10.0819 |
| Held-out loss / Spearman (n=4) | 0.01184 / 1.0000 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 021 - Grid sweep: n_points=10, train_examples=10000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n10_te10000.yaml` (n_points=10, train_examples=10000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/021/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6984 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | -1.8293 ± 3.2875 |
| Geodesic Spearman | 0.6917 |
| NN recall | 0.6000 |
| Intrinsic dimension | 9.4874 |
| Held-out loss / Spearman (n=4) | 0.01231 / 1.0000 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 022 - Grid sweep: n_points=10, train_examples=15000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n10_te15000.yaml` (n_points=10, train_examples=15000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/022/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7274 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | -2.4070 ± 4.1261 |
| Geodesic Spearman | 0.5941 |
| NN recall | 0.4000 |
| Intrinsic dimension | 7.2152 |
| Held-out loss / Spearman (n=4) | 0.02800 / 0.4000 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 023 - Grid sweep: n_points=10, train_examples=20000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n10_te20000.yaml` (n_points=10, train_examples=20000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/023/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6546 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | -1.9192 ± 3.4263 |
| Geodesic Spearman | 0.7133 |
| NN recall | 0.3000 |
| Intrinsic dimension | 9.8652 |
| Held-out loss / Spearman (n=4) | 0.02966 / 0.4000 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 024 - Grid sweep: n_points=10, train_examples=25000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n10_te25000.yaml` (n_points=10, train_examples=25000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/024/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7030 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | -1.9598 ± 3.6576 |
| Geodesic Spearman | 0.8153 |
| NN recall | 0.4000 |
| Intrinsic dimension | 6.6109 |
| Held-out loss / Spearman (n=4) | 0.00925 / 0.8000 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 025 - Grid sweep: n_points=10, train_examples=30000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n10_te30000.yaml` (n_points=10, train_examples=30000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/025/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6687 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | -2.2771 ± 3.9144 |
| Geodesic Spearman | 0.4854 |
| NN recall | 0.5000 |
| Intrinsic dimension | 16.3609 |
| Held-out loss / Spearman (n=4) | 0.04270 / -0.2000 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 026 - Grid sweep: n_points=10, train_examples=35000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n10_te35000.yaml` (n_points=10, train_examples=35000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/026/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6722 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | -2.0432 ± 3.7501 |
| Geodesic Spearman | 0.7818 |
| NN recall | 0.4000 |
| Intrinsic dimension | 10.5342 |
| Held-out loss / Spearman (n=4) | 0.01898 / 0.8000 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 027 - Grid sweep: n_points=10, train_examples=40000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n10_te40000.yaml` (n_points=10, train_examples=40000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/027/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6834 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | -1.8023 ± 3.2806 |
| Geodesic Spearman | 0.8150 |
| NN recall | 0.5000 |
| Intrinsic dimension | 13.0935 |
| Held-out loss / Spearman (n=4) | 0.00792 / 1.0000 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 028 - Grid sweep: n_points=10, train_examples=45000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n10_te45000.yaml` (n_points=10, train_examples=45000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/028/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6497 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | -2.2293 ± 3.8914 |
| Geodesic Spearman | 0.6414 |
| NN recall | 0.3000 |
| Intrinsic dimension | 11.3284 |
| Held-out loss / Spearman (n=4) | 0.04078 / 0.6000 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 029 - Grid sweep: n_points=10, train_examples=50000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n10_te50000.yaml` (n_points=10, train_examples=50000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/029/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6922 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | -2.2145 ± 3.8837 |
| Geodesic Spearman | 0.6096 |
| NN recall | 0.5000 |
| Intrinsic dimension | 9.7132 |
| Held-out loss / Spearman (n=4) | 0.03402 / 0.6000 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 030 - Grid sweep: n_points=20, train_examples=5000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n20_te5000.yaml` (n_points=20, train_examples=5000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/030/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6060 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | 0.4551 ± 0.2535 |
| Geodesic Spearman | 0.6014 |
| NN recall | 0.5000 |
| Intrinsic dimension | 8.2893 |
| Held-out loss / Spearman (n=19) | 0.00834 / 0.7754 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 031 - Grid sweep: n_points=20, train_examples=10000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n20_te10000.yaml` (n_points=20, train_examples=10000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/031/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6951 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | 0.4966 ± 0.2483 |
| Geodesic Spearman | 0.7362 |
| NN recall | 0.5500 |
| Intrinsic dimension | 8.5834 |
| Held-out loss / Spearman (n=19) | 0.00689 / 0.7895 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 032 - Grid sweep: n_points=20, train_examples=15000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n20_te15000.yaml` (n_points=20, train_examples=15000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/032/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6598 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | 0.5088 ± 0.2252 |
| Geodesic Spearman | 0.7298 |
| NN recall | 0.4500 |
| Intrinsic dimension | 8.9886 |
| Held-out loss / Spearman (n=19) | 0.00949 / 0.7860 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 033 - Grid sweep: n_points=20, train_examples=20000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n20_te20000.yaml` (n_points=20, train_examples=20000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/033/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6763 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | 0.4922 ± 0.2868 |
| Geodesic Spearman | 0.6471 |
| NN recall | 0.5000 |
| Intrinsic dimension | 5.7236 |
| Held-out loss / Spearman (n=19) | 0.00918 / 0.7421 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 034 - Grid sweep: n_points=20, train_examples=25000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n20_te25000.yaml` (n_points=20, train_examples=25000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/034/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6691 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | 0.4670 ± 0.2635 |
| Geodesic Spearman | 0.6473 |
| NN recall | 0.5000 |
| Intrinsic dimension | 7.2817 |
| Held-out loss / Spearman (n=19) | 0.00635 / 0.8632 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 035 - Grid sweep: n_points=20, train_examples=30000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n20_te30000.yaml` (n_points=20, train_examples=30000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/035/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6593 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | 0.4596 ± 0.2355 |
| Geodesic Spearman | 0.6984 |
| NN recall | 0.5500 |
| Intrinsic dimension | 8.4327 |
| Held-out loss / Spearman (n=19) | 0.00773 / 0.8000 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 036 - Grid sweep: n_points=20, train_examples=35000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n20_te35000.yaml` (n_points=20, train_examples=35000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/036/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6846 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | 0.4978 ± 0.2491 |
| Geodesic Spearman | 0.6838 |
| NN recall | 0.7000 |
| Intrinsic dimension | 8.7398 |
| Held-out loss / Spearman (n=19) | 0.00625 / 0.8386 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 037 - Grid sweep: n_points=20, train_examples=40000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n20_te40000.yaml` (n_points=20, train_examples=40000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/037/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6422 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | 0.4454 ± 0.2434 |
| Geodesic Spearman | 0.6137 |
| NN recall | 0.4000 |
| Intrinsic dimension | 6.1136 |
| Held-out loss / Spearman (n=19) | 0.01398 / 0.7070 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 038 - Grid sweep: n_points=20, train_examples=45000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n20_te45000.yaml` (n_points=20, train_examples=45000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/038/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7134 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | 0.5497 ± 0.2126 |
| Geodesic Spearman | 0.7337 |
| NN recall | 0.6000 |
| Intrinsic dimension | 5.6307 |
| Held-out loss / Spearman (n=19) | 0.00525 / 0.8930 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 039 - Grid sweep: n_points=20, train_examples=50000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n20_te50000.yaml` (n_points=20, train_examples=50000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/039/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7012 |
| In-sample linear R² | 1.0000 |
| Cross-validated R² | 0.4938 ± 0.2365 |
| Geodesic Spearman | 0.7014 |
| NN recall | 0.4500 |
| Intrinsic dimension | 9.4052 |
| Held-out loss / Spearman (n=19) | 0.00640 / 0.8596 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 040 - Grid sweep: n_points=50, train_examples=5000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n50_te5000.yaml` (n_points=50, train_examples=5000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/040/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7135 |
| In-sample linear R² | 0.9866 |
| Cross-validated R² | 0.9115 ± 0.0248 |
| Geodesic Spearman | 0.8608 |
| NN recall | 0.6000 |
| Intrinsic dimension | 4.1270 |
| Held-out loss / Spearman (n=122) | 0.00013 / 0.9973 |
| In-distribution loss / Spearman | 0.00001 / 0.9998 |

---

## 041 - Grid sweep: n_points=50, train_examples=10000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n50_te10000.yaml` (n_points=50, train_examples=10000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/041/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7659 |
| In-sample linear R² | 0.9848 |
| Cross-validated R² | 0.9163 ± 0.0177 |
| Geodesic Spearman | 0.8559 |
| NN recall | 0.7200 |
| Intrinsic dimension | 4.3847 |
| Held-out loss / Spearman (n=122) | 0.00005 / 0.9985 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 042 - Grid sweep: n_points=50, train_examples=15000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n50_te15000.yaml` (n_points=50, train_examples=15000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/042/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7328 |
| In-sample linear R² | 0.9853 |
| Cross-validated R² | 0.9120 ± 0.0214 |
| Geodesic Spearman | 0.8557 |
| NN recall | 0.7600 |
| Intrinsic dimension | 4.6587 |
| Held-out loss / Spearman (n=122) | 0.00010 / 0.9971 |
| In-distribution loss / Spearman | 0.00001 / 0.9996 |

---

## 043 - Grid sweep: n_points=50, train_examples=20000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n50_te20000.yaml` (n_points=50, train_examples=20000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/043/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7644 |
| In-sample linear R² | 0.9816 |
| Cross-validated R² | 0.9071 ± 0.0157 |
| Geodesic Spearman | 0.8740 |
| NN recall | 0.7400 |
| Intrinsic dimension | 4.8720 |
| Held-out loss / Spearman (n=122) | 0.00008 / 0.9979 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 044 - Grid sweep: n_points=50, train_examples=25000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n50_te25000.yaml` (n_points=50, train_examples=25000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/044/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6415 |
| In-sample linear R² | 0.9820 |
| Cross-validated R² | 0.8792 ± 0.0342 |
| Geodesic Spearman | 0.7835 |
| NN recall | 0.7000 |
| Intrinsic dimension | 3.7797 |
| Held-out loss / Spearman (n=122) | 0.00008 / 0.9979 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 045 - Grid sweep: n_points=50, train_examples=30000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n50_te30000.yaml` (n_points=50, train_examples=30000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/045/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7629 |
| In-sample linear R² | 0.9854 |
| Cross-validated R² | 0.9215 ± 0.0214 |
| Geodesic Spearman | 0.8429 |
| NN recall | 0.7400 |
| Intrinsic dimension | 4.2795 |
| Held-out loss / Spearman (n=122) | 0.00010 / 0.9977 |
| In-distribution loss / Spearman | 0.00000 / 0.9999 |

---

## 046 - Grid sweep: n_points=50, train_examples=35000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n50_te35000.yaml` (n_points=50, train_examples=35000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/046/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7470 |
| In-sample linear R² | 0.9855 |
| Cross-validated R² | 0.9136 ± 0.0213 |
| Geodesic Spearman | 0.8497 |
| NN recall | 0.8000 |
| Intrinsic dimension | 4.2554 |
| Held-out loss / Spearman (n=122) | 0.00007 / 0.9980 |
| In-distribution loss / Spearman | 0.00000 / 0.9998 |

---

## 047 - Grid sweep: n_points=50, train_examples=40000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n50_te40000.yaml` (n_points=50, train_examples=40000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/047/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7677 |
| In-sample linear R² | 0.9852 |
| Cross-validated R² | 0.9199 ± 0.0196 |
| Geodesic Spearman | 0.8625 |
| NN recall | 0.8200 |
| Intrinsic dimension | 3.5675 |
| Held-out loss / Spearman (n=122) | 0.00006 / 0.9984 |
| In-distribution loss / Spearman | 0.00001 / 0.9997 |

---

## 048 - Grid sweep: n_points=50, train_examples=45000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n50_te45000.yaml` (n_points=50, train_examples=45000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/048/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7744 |
| In-sample linear R² | 0.9838 |
| Cross-validated R² | 0.9095 ± 0.0243 |
| Geodesic Spearman | 0.8515 |
| NN recall | 0.7200 |
| Intrinsic dimension | 3.8588 |
| Held-out loss / Spearman (n=122) | 0.00008 / 0.9978 |
| In-distribution loss / Spearman | 0.00000 / 1.0000 |

---

## 049 - Grid sweep: n_points=50, train_examples=50000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n50_te50000.yaml` (n_points=50, train_examples=50000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/049/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6588 |
| In-sample linear R² | 0.9865 |
| Cross-validated R² | 0.8883 ± 0.0293 |
| Geodesic Spearman | 0.8124 |
| NN recall | 0.6400 |
| Intrinsic dimension | 4.5265 |
| Held-out loss / Spearman (n=122) | 0.00009 / 0.9975 |
| In-distribution loss / Spearman | 0.00000 / 0.9999 |

---

## 050 - Grid sweep: n_points=100, train_examples=5000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n100_te5000.yaml` (n_points=100, train_examples=5000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/050/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7616 |
| In-sample linear R² | 0.9824 |
| Cross-validated R² | 0.9602 ± 0.0106 |
| Geodesic Spearman | 0.8903 |
| NN recall | 0.6400 |
| Intrinsic dimension | 3.9905 |
| Held-out loss / Spearman (n=495) | 0.00005 / 0.9993 |
| In-distribution loss / Spearman | 0.00001 / 0.9998 |

---

## 051 - Grid sweep: n_points=100, train_examples=10000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n100_te10000.yaml` (n_points=100, train_examples=10000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/051/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7592 |
| In-sample linear R² | 0.9819 |
| Cross-validated R² | 0.9603 ± 0.0086 |
| Geodesic Spearman | 0.8609 |
| NN recall | 0.7100 |
| Intrinsic dimension | 3.4534 |
| Held-out loss / Spearman (n=495) | 0.00002 / 0.9997 |
| In-distribution loss / Spearman | 0.00000 / 0.9999 |

---

## 052 - Grid sweep: n_points=100, train_examples=15000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n100_te15000.yaml` (n_points=100, train_examples=15000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/052/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7809 |
| In-sample linear R² | 0.9812 |
| Cross-validated R² | 0.9571 ± 0.0107 |
| Geodesic Spearman | 0.8808 |
| NN recall | 0.7100 |
| Intrinsic dimension | 3.1204 |
| Held-out loss / Spearman (n=495) | 0.00004 / 0.9995 |
| In-distribution loss / Spearman | 0.00001 / 0.9998 |

---

## 053 - Grid sweep: n_points=100, train_examples=20000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n100_te20000.yaml` (n_points=100, train_examples=20000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/053/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7695 |
| In-sample linear R² | 0.9828 |
| Cross-validated R² | 0.9620 ± 0.0124 |
| Geodesic Spearman | 0.8802 |
| NN recall | 0.7200 |
| Intrinsic dimension | 3.0825 |
| Held-out loss / Spearman (n=495) | 0.00003 / 0.9996 |
| In-distribution loss / Spearman | 0.00000 / 0.9999 |

---

## 054 - Grid sweep: n_points=100, train_examples=25000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n100_te25000.yaml` (n_points=100, train_examples=25000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/054/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7808 |
| In-sample linear R² | 0.9795 |
| Cross-validated R² | 0.9577 ± 0.0128 |
| Geodesic Spearman | 0.8873 |
| NN recall | 0.6800 |
| Intrinsic dimension | 3.1734 |
| Held-out loss / Spearman (n=495) | 0.00003 / 0.9997 |
| In-distribution loss / Spearman | 0.00001 / 0.9999 |

---

## 055 - Grid sweep: n_points=100, train_examples=30000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n100_te30000.yaml` (n_points=100, train_examples=30000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/055/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7923 |
| In-sample linear R² | 0.9883 |
| Cross-validated R² | 0.9690 ± 0.0050 |
| Geodesic Spearman | 0.9031 |
| NN recall | 0.7400 |
| Intrinsic dimension | 2.8527 |
| Held-out loss / Spearman (n=495) | 0.00003 / 0.9996 |
| In-distribution loss / Spearman | 0.00001 / 0.9998 |

---

## 056 - Grid sweep: n_points=100, train_examples=35000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n100_te35000.yaml` (n_points=100, train_examples=35000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/056/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7685 |
| In-sample linear R² | 0.9806 |
| Cross-validated R² | 0.9574 ± 0.0137 |
| Geodesic Spearman | 0.8829 |
| NN recall | 0.7100 |
| Intrinsic dimension | 3.2721 |
| Held-out loss / Spearman (n=495) | 0.00002 / 0.9997 |
| In-distribution loss / Spearman | 0.00000 / 0.9999 |

---

## 057 - Grid sweep: n_points=100, train_examples=40000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n100_te40000.yaml` (n_points=100, train_examples=40000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/057/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7610 |
| In-sample linear R² | 0.9826 |
| Cross-validated R² | 0.9581 ± 0.0150 |
| Geodesic Spearman | 0.8772 |
| NN recall | 0.7100 |
| Intrinsic dimension | 3.3430 |
| Held-out loss / Spearman (n=495) | 0.00002 / 0.9997 |
| In-distribution loss / Spearman | 0.00000 / 0.9999 |

---

## 058 - Grid sweep: n_points=100, train_examples=45000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n100_te45000.yaml` (n_points=100, train_examples=45000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/058/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7774 |
| In-sample linear R² | 0.9830 |
| Cross-validated R² | 0.9610 ± 0.0126 |
| Geodesic Spearman | 0.8846 |
| NN recall | 0.6900 |
| Intrinsic dimension | 3.3158 |
| Held-out loss / Spearman (n=495) | 0.00002 / 0.9997 |
| In-distribution loss / Spearman | 0.00000 / 0.9999 |

---

## 059 - Grid sweep: n_points=100, train_examples=50000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n100_te50000.yaml` (n_points=100, train_examples=50000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/059/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7592 |
| In-sample linear R² | 0.9798 |
| Cross-validated R² | 0.9547 ± 0.0131 |
| Geodesic Spearman | 0.8804 |
| NN recall | 0.7000 |
| Intrinsic dimension | 3.0241 |
| Held-out loss / Spearman (n=495) | 0.00003 / 0.9996 |
| In-distribution loss / Spearman | 0.00000 / 0.9999 |

---

## 060 - Grid sweep: n_points=800, train_examples=5000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n800_te5000.yaml` (n_points=800, train_examples=5000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/060/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.3176 |
| In-sample linear R² | 0.1507 |
| Cross-validated R² | 0.0845 ± 0.0186 |
| Geodesic Spearman | 0.0578 |
| NN recall | 0.0000 |
| Intrinsic dimension | 14.5120 |
| Held-out loss / Spearman (n=31960) | 0.03015 / 0.0731 |
| In-distribution loss / Spearman | 0.02982 / 0.0849 |

---

## 061 - Grid sweep: n_points=800, train_examples=10000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n800_te10000.yaml` (n_points=800, train_examples=10000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/061/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6212 |
| In-sample linear R² | 0.9633 |
| Cross-validated R² | 0.9607 ± 0.0039 |
| Geodesic Spearman | 0.8232 |
| NN recall | 0.0663 |
| Intrinsic dimension | 13.6131 |
| Held-out loss / Spearman (n=31960) | 0.00049 / 0.9923 |
| In-distribution loss / Spearman | 0.00047 / 0.9924 |

---

## 062 - Grid sweep: n_points=800, train_examples=15000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n800_te15000.yaml` (n_points=800, train_examples=15000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/062/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7042 |
| In-sample linear R² | 0.9690 |
| Cross-validated R² | 0.9663 ± 0.0034 |
| Geodesic Spearman | 0.8625 |
| NN recall | 0.2313 |
| Intrinsic dimension | 9.5446 |
| Held-out loss / Spearman (n=31960) | 0.00004 / 0.9993 |
| In-distribution loss / Spearman | 0.00004 / 0.9993 |

---

## 063 - Grid sweep: n_points=800, train_examples=20000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n800_te20000.yaml` (n_points=800, train_examples=20000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/063/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7410 |
| In-sample linear R² | 0.9769 |
| Cross-validated R² | 0.9750 ± 0.0022 |
| Geodesic Spearman | 0.8825 |
| NN recall | 0.3312 |
| Intrinsic dimension | 8.1463 |
| Held-out loss / Spearman (n=31960) | 0.00002 / 0.9996 |
| In-distribution loss / Spearman | 0.00002 / 0.9996 |

---

## 064 - Grid sweep: n_points=800, train_examples=25000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n800_te25000.yaml` (n_points=800, train_examples=25000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/064/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7139 |
| In-sample linear R² | 0.9765 |
| Cross-validated R² | 0.9744 ± 0.0010 |
| Geodesic Spearman | 0.8718 |
| NN recall | 0.3775 |
| Intrinsic dimension | 7.6989 |
| Held-out loss / Spearman (n=31960) | 0.00001 / 0.9997 |
| In-distribution loss / Spearman | 0.00001 / 0.9998 |

---

## 065 - Grid sweep: n_points=800, train_examples=30000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n800_te30000.yaml` (n_points=800, train_examples=30000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/065/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7434 |
| In-sample linear R² | 0.9684 |
| Cross-validated R² | 0.9654 ± 0.0024 |
| Geodesic Spearman | 0.8760 |
| NN recall | 0.3663 |
| Intrinsic dimension | 7.5225 |
| Held-out loss / Spearman (n=31960) | 0.00002 / 0.9997 |
| In-distribution loss / Spearman | 0.00001 / 0.9997 |

---

## 066 - Grid sweep: n_points=800, train_examples=35000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n800_te35000.yaml` (n_points=800, train_examples=35000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/066/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7595 |
| In-sample linear R² | 0.9733 |
| Cross-validated R² | 0.9711 ± 0.0031 |
| Geodesic Spearman | 0.8900 |
| NN recall | 0.4800 |
| Intrinsic dimension | 6.5550 |
| Held-out loss / Spearman (n=31960) | 0.00001 / 0.9998 |
| In-distribution loss / Spearman | 0.00001 / 0.9998 |

---

## 067 - Grid sweep: n_points=800, train_examples=40000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n800_te40000.yaml` (n_points=800, train_examples=40000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/067/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7589 |
| In-sample linear R² | 0.9794 |
| Cross-validated R² | 0.9771 ± 0.0015 |
| Geodesic Spearman | 0.8662 |
| NN recall | 0.5025 |
| Intrinsic dimension | 5.6869 |
| Held-out loss / Spearman (n=31960) | 0.00001 / 0.9999 |
| In-distribution loss / Spearman | 0.00001 / 0.9999 |

---

## 068 - Grid sweep: n_points=800, train_examples=45000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n800_te45000.yaml` (n_points=800, train_examples=45000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/068/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7675 |
| In-sample linear R² | 0.9781 |
| Cross-validated R² | 0.9760 ± 0.0018 |
| Geodesic Spearman | 0.8954 |
| NN recall | 0.5437 |
| Intrinsic dimension | 5.7209 |
| Held-out loss / Spearman (n=31960) | 0.00001 / 0.9999 |
| In-distribution loss / Spearman | 0.00001 / 0.9999 |

---

## 069 - Grid sweep: n_points=800, train_examples=50000

### 1. Model settings

Config: `configs/grid_sweep/octahedron_n800_te50000.yaml` (n_points=800, train_examples=50000,
emb_dim=24, steps=50000, seed=0 - otherwise identical to
`octahedron_default.yaml`). Part of the train_examples x n_points grid
sweep (020-069); see the summary entry at the end of this file for
cross-run trends.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/069/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.7064 |
| In-sample linear R² | 0.9676 |
| Cross-validated R² | 0.9646 ± 0.0033 |
| Geodesic Spearman | 0.8416 |
| NN recall | 0.5575 |
| Intrinsic dimension | 5.5596 |
| Held-out loss / Spearman (n=31960) | 0.00001 / 0.9998 |
| In-distribution loss / Spearman | 0.00001 / 0.9998 |

---

## 070 - Grid sweep summary: cross-run trends across n_points x train_examples (020-069)

### 1. Model settings

Synthesizes all 50 runs above (020-069): a full cross of
`n_points` in {10, 20, 50, 100, 800} x `train_examples` in {5000, 10000,
..., 50000}, at fixed `emb_dim=24`, `steps=50000`, and every other
setting from `octahedron_default.yaml`. Raw metrics for all 50 runs are
in `experiments/manifold_learning/octahedron/grid_sweep_results.csv`.
Purpose: previous sweeps (007-019) varied `emb_dim` or `n_points`
independently at a fixed `train_examples`/`steps`; this is the first
systematic joint sweep of pair-coverage (`n_points`) against training
volume (`train_examples`).

### 2. Results / findings

Per-`n_points` means across all 10 `train_examples` values:

| n_points | cv R² (mean±std) | Spearman | NN recall | intrinsic dim | held-out Spearman |
|---:|---:|---:|---:|---:|---:|
| 10 | -2.060 ± 0.208 | 0.691 | 0.440 | 10.43 | 0.640 ± 0.375 |
| 20 | 0.487 ± 0.031 | 0.679 | 0.520 | 7.72 | 0.805 ± 0.058 |
| 50 | 0.908 ± 0.014 | 0.845 | 0.724 | 4.23 | 0.998 ± 0.000 |
| 100 | 0.960 ± 0.004 | 0.883 | 0.701 | 3.26 | 1.000 ± 0.000 |
| 800 | 0.882 ± 0.280 | 0.787 | 0.346 | 8.46 | 0.906 ± 0.293 |

- **`n_points=10` and `n_points=20` cross-validated R² and held-out
  numbers are not meaningful generalization signals, despite perfect
  in-sample fit (R²=1.0 at every one of those 20 runs).** With only
  C(10,2)=45 or C(20,2)=190 total pairs (4 or 19 held out at
  `test_fraction=0.1`), `train_examples` (5000-50000) massively
  oversamples the same handful of pairs with replacement - the model
  memorizes the tiny pool exactly, but 5-fold cross-validation over 9-18
  training points and held-out evaluation on 4-19 pairs are both
  close to degenerate at this scale (hence `n_points=10`'s wildly
  negative mean cv R² of -2.06, and swings from held-out Spearman 1.0 to
  -0.2 between adjacent `train_examples` values in the raw CSV - noise
  from tiny denominators, not a real quality difference).
- **`n_points=100` is the sweet spot of the grid tested here**: highest
  mean cross-validated R² (0.960), tightest spread across
  `train_examples` (std 0.004 - essentially insensitive to how many
  training examples it gets once coverage is adequate), and the lowest
  mean intrinsic dimension (3.26, closest of any group to a compact
  low-dimensional embedding). `n_points=50` is close behind on every
  metric and has the single highest NN recall of the whole grid, 0.82 at
  `train_examples=40000` (experiment 047) - higher than any prior
  octahedron run in this file, including 018's previous-best 0.76 at
  `n_points=200`/`emb_dim=8`.
- **`n_points=800` is the one setting where `train_examples` and quality
  are tightly coupled, and the grid's single worst result lives here.**
  Within `n_points=800`, correlation between `train_examples` and NN
  recall is 0.96 and with cross-validated R² is 0.53 - far higher than
  in any other `n_points` group (all under 0.33 in magnitude elsewhere).
  Concretely: experiment 060 (`n_points=800`, `train_examples=5000`)
  is a near-total failure - cross-validated R²=0.08, NN recall=0.0,
  held-out Spearman=0.07 - while experiment 061 (`train_examples=10000`,
  otherwise identical) jumps to cross-validated R²=0.96, NN recall=0.07,
  held-out Spearman=0.99, and quality keeps improving smoothly from
  there through `train_examples=50000`. Doubling `train_examples` from
  5000 to 10000 is the difference between the model not learning the
  task at all and matching the sweep's best runs - at `n_points=800`'s
  ~258900-pair 90% train pool, 5000 examples (<2% coverage) is simply
  too sparse a training signal for 50000 steps to compensate for, unlike
  smaller `n_points` where the same 5000-50000 `train_examples` values
  already vastly *exceed* the total pair count.
- **Held-out and in-distribution numbers move together at every scale
  where they're meaningful** (`n_points` >= 50): held-out loss tracks
  in-distribution loss to within 0.0001 and held-out Spearman stays
  >=0.997, at every `train_examples` value from 10000 up - including 061,
  where overall quality is still climbing. This matches 016/017's
  earlier finding that the held-out generalization gap tracks *overall*
  task quality rather than being an independent failure mode: even
  060's catastrophic failure has held-out loss (0.030) and
  in-distribution loss (0.030) essentially equal - the model is equally
  bad on both, not overfit to the training pairs specifically.
- Intrinsic dimension is U-shaped in `n_points`: high and noisy at 10
  (mean 10.43, range 6.6-16.4), falls through 20 -> 50 -> 100 (7.72 ->
  4.23 -> 3.26, the octahedron's ambient_dim=3 most closely approached
  at `n_points=100`), then rises again at 800 (mean 8.46) - but that
  800-mean is dominated by 060's undertrained outlier (14.5) and the
  still-settling low-`train_examples` runs; from `train_examples=35000`
  up at `n_points=800`, intrinsic dimension is back down to 5.5-6.6,
  closer to (but still above) the `n_points=100` minimum.

### 3. Next steps

- Confirm the `n_points=800`/`train_examples=5000` cliff is a real
  coverage threshold and not a one-off seed effect: rerun experiment 060
  at a second seed, and/or sample a couple of `train_examples` values
  between 5000 and 10000 (e.g. 6000, 7500) to see how sharp the
  transition actually is.
- `n_points=100` and `n_points=50` outperforming `n_points=800` on
  cross-validated R², NN recall, and intrinsic dimension (once
  `n_points=800` has enough `train_examples` to not catastrophically
  fail) echoes 018/019's earlier finding that denser pair coverage at
  smaller `n_points` beats sparse coverage at larger `n_points` - worth
  extending this grid with a `n_points=200`/`400` column to see whether
  the sweet spot sits closer to 100 or drifts higher once `emb_dim=24`
  and 50000 steps are held fixed (018/019 used `emb_dim=8`/`24` at
  `n_points=200` only, not this full `train_examples` range).
- The `n_points<=20` rows are not informative about generalization at
  this `test_fraction=0.1` split (too few held-out pairs); if small-scale
  behavior is worth studying further, either use a much smaller
  `test_fraction` or hold out individual points, not just pairs (see the
  holdout-points idea noted separately).

---

## 071 - Point-holdout rerun 1/5: 009 (emb_dim=24, n_points=800) + face-localized point recovery

### 1. Model settings

New config `configs/octahedron_emb24_holdout.yaml`: identical to
`octahedron_emb24.yaml` (experiment 009 - the highest cross-validated R²,
0.980, in the whole file, and the run that strictly dominated
`emb_dim=32` on every metric) plus the new point-level holdout
generalization feature (`src/train_manifold.py` / `src/datasets.py` /
`src/holdout_probe.py` / `analysis/analysis_manifold.py`): 20 of the 800
points, all sampled from a single octahedron face (`holdout_faces: [0]`,
which has 87 of the 800 points at this seed), are withheld from every
training pair entirely (not just held out as pairs - no training example
ever touches these 20 indices), then after training their embeddings are
recovered from 8 true probe distances each (200 AdamW steps, scoped to
only those 20 embedding rows via a standalone `nn.Parameter` spliced into
the frozen table - see `src/holdout_probe.py`) and evaluated against the
rest of the points. Purpose: the first real-scale test of whether the
model's learned geometry generalizes to a genuinely unseen point, not
just an unseen pair among already-fully-trained points - the stronger
generalization claim ("does the model generalize to a new city, not just
a new pair of already-known cities?") and requested as a batch of five reruns of this file's most promising prior
results, this being the first (see 072-075 for the other four, and 075
for a cross-run comparison table).

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/071/`

- **Point-level holdout generalization is excellent, close to the
  pair-level held-out numbers this file has repeatedly found:**
  holdout-to-trained aggregate (n=15440 pairs, the 20 holdout points
  against all 772 non-holdout points minus the 8 probes each) loss
  0.00015, Spearman **0.9967** - recovering an entire point's embedding
  from just 8 known distances and using it to predict distances to
  hundreds of other points it never trained on works almost as well as
  the model's existing near-perfect pair-level generalization (0.9998
  Spearman, held-out pairs).
- **Holdout-to-holdout generalization (distances between pairs of
  recovered points, n=190, C(20,2)) is nearly as strong: Spearman
  0.9810.** Both recovered points in every such pair were fit only
  against probe distances to normally-trained points, never against each
  other - this checks whether independently-recovered embeddings land in
  mutually consistent positions, not just individually-plausible ones,
  and at this scale they do.
- Bulk embedding-quality metrics (measured on the 780 non-holdout points
  only, per the new metrics-masking fix): cross-validated R² 0.9737 ±
  0.0020, Spearman 0.8671, NN recall 0.5910, intrinsic dimension 5.3891 -
  close to 017's numbers (`emb_dim=24`, `n_points=800`, 50k steps, pair
  split only, no point holdout: cv R² 0.9646, Spearman 0.8416, NN recall
  0.5575, intrinsic dim 5.5596) and if anything marginally *better* on
  every one of those four metrics. **Removing 20 of 800 points (2.5%)
  from training has no detectable cost to overall embedding quality at
  this scale** - unsurprising given how sparse pair coverage already is
  at `n_points=800` (each training pair is seen well under once on
  average per 010/070's analysis).
- `embedding_pca.png` / `true_manifold_pca.png` mark the 20 recovered
  points as red stars: in the true-ambient-surface PCA plot they cluster
  tightly (all on the same face, as designed), and in the
  learned-embedding PCA plot they land in essentially the same region of
  the point cloud rather than as outliers - a direct visual confirmation
  that recovery worked, consistent with the 0.9967/0.9810 Spearman
  numbers above.
- Pair-symmetry and held-out-pair checks are unaffected and match prior
  `emb_dim=24`/`n_points=800` runs closely (max symmetry diff ~3.0e-7;
  held-out pairs loss 0.00001/Spearman 0.9998, in-distribution
  0.00001/0.9999).

### 3. Next steps

- See 075 for a five-run comparison table across all of 071-075 - the
  headline cross-run question is how holdout-to-holdout consistency
  scales with `n_points`/`n_holdout_points` ratio and `emb_dim`.

---

## 072 - Point-holdout rerun 2/5: 047 (n_points=50, train_examples=40000) + face-localized point recovery

### 1. Model settings

New config `configs/grid_sweep/octahedron_n50_te40000_holdout.yaml`:
identical to `octahedron_n50_te40000.yaml` (experiment 047 - the best
nearest-neighbor recall, 0.82, of any run in this file) plus the
point-level holdout feature - see 071 for the mechanism. 5 of the 50
points, all on octahedron face 0 (which has only 7 of the 50 points at
this seed - the tightest fit of the five reruns), are withheld from
training and recovered from 5 probe distances each (200 AdamW steps).

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/072/`

- **Point-level holdout-to-trained generalization is still strong but
  visibly weaker than 071's larger-scale result:** aggregate (n=200
  pairs) loss 0.00211, Spearman **0.9486** - clearly generalizing, but
  the first sign that this task gets harder at small `n_points`.
- **Holdout-to-holdout generalization collapses at this scale: Spearman
  0.3939 (n=10 pairs, C(5,2)).** This is a sharp contrast with 071's
  0.9810 - two independently-recovered points, both fit only against
  probes to normally-trained points, are only weakly consistent with
  each other here. With only 5 probes per point and just 7 points total
  on face 0 (5 of them held out, leaving only 2 non-holdout points on
  the same face to anchor local structure), there is very little
  same-face training signal left for the model to have learned a
  locally-consistent geometry from in the first place - unlike 071,
  where the same face had 87 points and only 20 were held out.
- Bulk quality also takes a real hit from removing 5/50 (10%) of the
  training points, unlike 071's negligible cost at 20/800 (2.5%):
  cross-validated R² 0.8752 ± 0.0390 vs. baseline 047's 0.9199 ± 0.0196,
  NN recall 0.6667 vs. 0.8200, Spearman 0.8194 vs. 0.8625 - all down
  noticeably, though intrinsic dimension is essentially unchanged (3.60
  vs. 3.57). Held-out *pair* generalization remains excellent regardless
  (loss 0.00009/Spearman 0.9973 held-out vs. 0.00001/0.9996
  in-distribution) - consistent with this file's repeated finding that
  pair-level generalization is robust everywhere, while overall embedding
  quality is sensitive to how much training data is removed at small
  `n_points`.
- `embedding_pca.png` visibly shows the 5 recovered points scattered more
  loosely relative to the bulk cloud than 071's tight clustering,
  consistent with the weak holdout-to-holdout number.

### 3. Next steps

- The combination "few points on the holdout face" + "few probes" seems
  to be what breaks holdout-to-holdout consistency specifically (not
  holdout-to-trained, which stayed reasonable) - 074/075 (same
  `n_points=200`/10-holdout/8-probe setup as each other but different
  `emb_dim`) and 073 (`n_points=100`) let this be checked against a
  cleaner width-only and points-only comparison respectively.

---

## 073 - Point-holdout rerun 3/5: 055 (n_points=100, train_examples=30000) + face-localized point recovery

### 1. Model settings

New config `configs/grid_sweep/octahedron_n100_te30000_holdout.yaml`:
identical to `octahedron_n100_te30000.yaml` (experiment 055 - the best
all-around single run in this file: R²=0.969, Spearman=0.903, NN
recall=0.74, near-lowest intrinsic dimension 2.85) plus the point-level
holdout feature. 10 of the 100 points, all on octahedron face 0 (13 of
the 100 points at this seed), are withheld and recovered from 8 probe
distances each.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/073/`

- **Point-level holdout generalization is excellent at this
  intermediate scale, closing most of the gap back toward 071:**
  holdout-to-trained aggregate (n=820) loss 0.00045, Spearman **0.9906**;
  holdout-to-holdout (n=45, C(10,2)) loss 0.00072, Spearman **0.9601** -
  both far closer to 071's (n=800) numbers than to 072's (n=50), despite
  073 removing a larger *fraction* of its holdout face's points (10 of
  13, ~77%) than 071 did (20 of 87, ~23%). Absolute face population
  (13 points, comfortably above 072's 7) appears to matter more here than
  the fraction removed.
- Bulk quality cost from removing 10/100 points is modest: cross-validated
  R² 0.9614 ± 0.0079 vs. baseline 055's 0.9690 ± 0.0050, Spearman 0.8907
  vs. 0.9031, NN recall 0.6444 vs. 0.7400 (the largest single drop),
  intrinsic dimension 3.71 vs. 2.85 (up, less compact) - a real but
  moderate cost, between 071's negligible one and 072's large one.
- Held-out pair generalization remains essentially perfect (loss
  0.00002/Spearman 0.9996 held-out vs. 0.00000/1.0000 in-distribution),
  continuing the pattern that this metric is insensitive to the point
  holdout feature regardless of scale.

### 3. Next steps

- See 075 for the full five-run table - 073 sits as the clearest
  "moderate scale, moderate cost, strong recovery" middle point between
  071 (large scale, no cost) and 072 (small scale, large cost).

---

## 074 - Point-holdout rerun 4/5: 018 (emb_dim=8, n_points=200) + face-localized point recovery

### 1. Model settings

New config `configs/octahedron_emb8_n200_holdout.yaml`: identical to
`octahedron_emb8_n200.yaml` (experiment 018 - the lowest intrinsic
dimension of any run in this file, 2.08, closest to the octahedron's true
2D surface) plus the point-level holdout feature. 10 of the 200 points,
all on octahedron face 0 (25 of the 200 points at this seed), are
withheld and recovered from 8 probe distances each. Purpose: an
`emb_dim=8` counterpart to 075 (`emb_dim=24`, otherwise identical setup),
to isolate whether embedding width affects point-holdout recovery the
way 016 vs. 017 found it affects the pair-level held-out split.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/074/`

- **Point-level holdout-to-trained generalization is still strong:**
  aggregate (n=1820) loss 0.00019, Spearman **0.9967** - matching 071's
  number almost exactly despite the much narrower embedding.
- **Holdout-to-holdout generalization is noticeably weaker than the
  same-`n_points` `emb_dim=24` run (075): Spearman 0.8565 (n=45) vs.
  075's 0.9394.** The tight `emb_dim=8` bottleneck that 006 found makes
  the embedding table harder to fully fit also makes two independently-
  recovered points less mutually consistent with each other, even though
  each one individually still predicts its distances to normally-trained
  points well.
- **Bulk quality pays a real, width-specific price for the point
  holdout, unlike the wider run:** cross-validated R² 0.9072 ± 0.0611 vs.
  baseline 018's 0.9496 ± 0.0044 - both the mean drop (-0.042) and
  especially the ballooning standard deviation (14x larger) are far
  bigger than 075's corresponding -0.005/roughly-unchanged-std hit at
  `emb_dim=24` on the same `n_points=200`/10-holdout/8-probe setup. NN
  recall, unusually, went the other way (0.7789 vs. baseline 0.7600,
  essentially flat-to-up) and intrinsic dimension rose from 2.08 to 2.43
  (less compact, but still closer to the manifold's true dimension than
  any `emb_dim=24` run here).
- This directly echoes 016-vs-017's earlier finding that the *narrow*
  `emb_dim=8` width reacts very differently to losing training data than
  wider widths do - there it was the pair-level held-out split; here it's
  the same asymmetry showing up in the point-level holdout feature,
  measured on an otherwise-identical `n_points=200` setup against 075.

### 3. Next steps

- See 075 for the direct `emb_dim=8` vs. `emb_dim=24` comparison table at
  matched `n_points=200`/10-holdout/8-probe settings, and the full
  five-run summary.

---

## 075 - Point-holdout rerun 5/5: 019 (emb_dim=24, n_points=200) + face-localized point recovery, and cross-run summary

### 1. Model settings

New config `configs/octahedron_emb24_n200_holdout.yaml`: identical to
`octahedron_emb24_n200.yaml` (experiment 019 - near-perfect held-out pair
generalization, Spearman 0.9999, plus strong R²/NN-recall/intrinsic-dim
across the board) plus the point-level holdout feature. 10 of the 200
points, all on octahedron face 0 (25 of the 200 points at this seed,
identical setup to 074 except `emb_dim`), are withheld and recovered from
8 probe distances each. Last of the five reruns (071-075) of this file's
most promising prior results with the new point-holdout generalization
feature (`src/train_manifold.py`'s `n_holdout_points`/`n_probes`/
`probe_steps`/`probe_lr`/`holdout_faces` config keys, `src/holdout_probe.py`,
and `analysis/analysis_manifold.py`'s new "HOLDOUT POINT GENERALIZATION"
section and marked-recovered-point PCA plots).

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/075/`

- **Point-level holdout generalization is excellent and the strongest
  holdout-to-holdout result outside of 071:** holdout-to-trained
  aggregate (n=1820) loss 0.00013, Spearman **0.9970**; holdout-to-holdout
  (n=45) loss 0.00043, Spearman **0.9394**.
- **Bulk quality cost from the point holdout is essentially negligible
  here, unlike 074's `emb_dim=8` counterpart:** cross-validated R² 0.9655
  ± 0.0029 vs. baseline 019's 0.9705 ± 0.0026 (-0.005), Spearman 0.8876
  vs. 0.8930, and **NN recall actually improved, 0.8105 vs. 0.7350** -
  the only run of the five where removing training points didn't hurt
  (and arguably helped) a bulk metric. Intrinsic dimension rose slightly
  (2.91 vs. 2.65).
- `embedding_pca.png` shows the 10 recovered points landing inside the
  bulk point cloud, consistent with the strong Spearman numbers.

**Cross-run summary, all five point-holdout reruns:**

| exp | baseline | n_points | emb_dim | n_holdout / n_probes | holdout-to-trained loss / Spearman | holdout-to-holdout loss / Spearman | bulk cv R² (rerun vs. baseline) |
|---|---|---:|---:|---:|---:|---:|---:|
| 071 | 009 | 800 | 24 | 20 / 8 | 0.00015 / **0.9967** | 0.00016 / **0.9810** | 0.9737 (vs. 0.9646 in 017's pair-split-only rerun) |
| 072 | 047 | 50 | 24 | 5 / 5 | 0.00211 / 0.9486 | 0.01727 / **0.3939** | 0.8752 (vs. 0.9199) |
| 073 | 055 | 100 | 24 | 10 / 8 | 0.00045 / 0.9906 | 0.00072 / 0.9601 | 0.9614 (vs. 0.9690) |
| 074 | 018 | 200 | 8 | 10 / 8 | 0.00019 / 0.9967 | 0.00094 / 0.8565 | 0.9072 (vs. 0.9496) |
| 075 | 019 | 200 | 24 | 10 / 8 | 0.00013 / 0.9970 | 0.00043 / 0.9394 | 0.9655 (vs. 0.9705) |

**Headline findings across all five runs:**

- **Holdout-to-trained generalization (a recovered point predicting
  distances to normally-trained points) is uniformly strong everywhere
  tested: Spearman 0.949-0.997 across `n_points` 50-800 and `emb_dim`
  8-24.** This directly answers the motivating question behind the whole
  feature: the model's learned geometry
  does generalize to a genuinely unseen point recovered from just 5-8
  known distances, not merely to unseen pairs among already-trained
  points - and this holds robustly across every scale and width tried,
  not just in a single favorable setting.
- **Holdout-to-holdout generalization (do two independently-recovered
  points agree with each other) is the more fragile, scale-sensitive
  metric, ranging from 0.394 (072, n=50) to 0.981 (071, n=800).** The
  clearest driver is not the *fraction* of a face held out (073 held out
  77% of face 0's points and still scored 0.96; 072 held out 71% and
  scored 0.39) but the *absolute number of points remaining on that face
  to anchor local structure* - 072's face 0 had only 7 points total (2
  left after holdout), while 073's had 13 (3 left) and 071's had 87 (67
  left). Below some small absolute count, there is not enough same-face
  training signal for independently-recovered points to land in a
  mutually consistent place, even when each one individually still
  predicts its distances to the (more numerous) rest of the manifold
  well.
- **`emb_dim` affects both the bulk-quality cost and the holdout-to-holdout
  consistency of the point-holdout feature, echoing 016/017's earlier
  pair-level finding.** At matched `n_points=200`/10-holdout/8-probe
  settings, `emb_dim=8` (074) took a much larger, noisier bulk cv-R² hit
  (-0.042, std 14x larger) and a weaker holdout-to-holdout Spearman
  (0.857) than `emb_dim=24` (075: -0.005 bulk cost, 0.939 holdout-to-
  holdout) - a tight embedding bottleneck makes recovering a coherent,
  mutually-consistent new point harder, not just fitting the original
  training points harder.
- **Held-out *pair* generalization (the pre-existing feature) remains
  essentially untouched by any of this - Spearman >=0.997 in every one of
  the five reruns**, exactly matching this file's long-standing finding
  (011-019, 070) that pair-level generalization is robust across scale
  and width. The point-holdout feature is measuring something genuinely
  different and harder, not a restatement of the same generalization
  gap.

### 3. Next steps

- 072's holdout-to-holdout collapse suggests a face-population threshold
  effect worth mapping directly: rerun `n_points=50` with a *smaller*
  `n_holdout_points` (e.g. 2-3 instead of 5) to see whether
  holdout-to-holdout recovers once more same-face points remain to
  anchor local structure, isolating "too few points held out" from "too
  few points on the face at all."
- The `emb_dim=8` vs. `24` split (074 vs. 075) at matched `n_points=200`
  is only one data point per width; repeating at `n_points=800` (a direct
  `emb_dim=8` analog of 071) would confirm whether the width effect on
  point-holdout recovery holds at larger scale too, or is specific to the
  `n_points=200` regime where 018 already showed `emb_dim=8` behaves
  differently from `emb_dim=24`.
- All five reruns held out points from a single face (`holdout_faces:
  [0]`) for visual-inspection purposes. A direct comparison against
  uniformly-random (non-face-restricted) holdout at the same `n_points`/
  `n_holdout_points` would separate "face-localized holdout is harder
  because it's spatially clustered" from "these specific counts/widths
  are what drive the results above" - the two were confounded in every
  run here.

---

## 076 - First same_triangle task run: distance + same_triangle combined

### 1. Model settings

Config: `configs/octahedron_distance_same_triangle.yaml` (800 points,
10000 steps, 50000 training examples, emb_dim=32 - otherwise identical to
`octahedron_default.yaml`/002/004), now with `training.tasks: [distance,
same_triangle]`. First use of the new `same_triangle` task (added this
session): a binary classification task asking whether two sampled points
lie on the same octahedron/icosahedron triangular face, using the face
index already stored in `chart_points[:, 0]` as ground truth
(`src/tasks.py`). Adding this task required generalizing
`src/train_manifold.py` from a distance-only script into a config-driven
multi-task trainer (`training.tasks`, defaulting to `["distance"]` so
every prior config/experiment in this file is unaffected), a new
`same_triangle_token`/`same_triangle_head` pair in
`src/multitask_model.py`, classification eval (loss/accuracy/AUC) in
`src/eval_utils.py`, and a per-task-nested checkpoint/analysis schema in
`analysis/analysis_manifold.py`. Purpose: does combining `same_triangle`
with `distance` change what geometry the embedding recovers, and does the
model learn real same-face structure (not just memorize)? First of a
three-part investigation (same_triangle alone; this combined run; later,
whether points held out and recovered via distance-only probing also
generalize to same_triangle - see
`configs/octahedron_holdout_same_triangle_eval.yaml`, verified working at
smoke scale but not yet archived here at full scale).

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/076/`

- **Both tasks are learned almost perfectly, and neither one seems to
  come at the other's expense on task-level held-out pairs:**

  | metric | distance | same_triangle |
  |---|---:|---:|
  | held-out loss | 0.00037 | 0.00064 |
  | in-distribution loss | 0.00035 | 0.00040 |
  | held-out Spearman / accuracy | 0.9923 | 0.99994 (accuracy) |
  | held-out AUC | - | 0.99993 |

  Held-out same_triangle accuracy (99.994%) is far above the ~87.5%
  majority-class baseline (only 1/8 chance two random points share one of
  the octahedron's 8 faces), and AUC is essentially 1.0 - the model isn't
  just exploiting class imbalance, it has genuinely learned the face
  structure. Pair-symmetry holds tightly for both heads (distance max
  diff 4.2e-7, same_triangle max diff 8.6e-6 - an order of magnitude
  looser than distance's, but still far below any threshold that would
  matter in practice).
- **Surprising headline finding: the raw embedding-table bulk metrics are
  substantially *worse* than the equivalent distance-only run, despite
  both task heads generalizing near-perfectly.** Compared to 002/004
  (same config otherwise - 800 points, 10000 steps, emb_dim=32,
  distance-only):

  | metric | distance-only (002/004) | distance+same_triangle (076) |
  |---|---:|---:|
  | PCA explained variance (3 comp) | 0.792 | **0.660** |
  | Cross-validated coordinate R² | 0.96 ± 0.004 | **0.722 ± 0.005** |
  | Geodesic-distance Spearman | 0.887 | **0.382** |
  | NN recall | 0.326 | **0.129** |
  | Intrinsic dimension | 10.45 | **13.14** (worse/less compact) |

  Every bulk embedding-quality metric got worse when same_triangle joined
  training, several dramatically (Spearman roughly halved, NN recall
  dropped by more than half). Since both task heads individually still
  generalize essentially perfectly, this isn't a capacity failure of the
  transformer as a whole - it means the *raw* per-entity embedding table
  (read directly off `model.emb.weight`, with no task token/transformer
  involved) no longer linearly encodes the true ambient octahedron
  coordinates nearly as cleanly on its own. The shared transformer,
  reading either task's token, must be doing more of the work of
  reconstructing task-relevant geometry from a less directly-
  interpretable embedding table than in the distance-only case.
- Entity-token CKA is similar for both tasks (layer 1: 0.44 distance /
  0.43 same_triangle; layer 2: 0.19 distance / 0.22 same_triangle) - the
  two tasks' entity-token representations are about equally
  (dis)similar between i/j at each layer, unlike the raw embedding-table
  metrics above, which differ sharply from the single-task baseline.

### 3. Next steps

- Run `configs/octahedron_same_triangle.yaml` (same_triangle-only, no
  distance) at full scale and archive it here, to isolate whether the
  bulk-metric degradation above comes from same_triangle specifically
  competing with distance for embedding-table capacity, or whether
  same_triangle alone also fails to produce a cleanly-linear embedding
  table (in which case the "worse than distance-only" framing above would
  be backwards - distance might be the unusually clean baseline, not
  same_triangle-combined being unusually degraded).
- Run `configs/octahedron_holdout_same_triangle_eval.yaml` at full scale
  (800 points, not the 50-point smoke config used to verify the wiring)
  and archive it here - this is the experiment the whole same_triangle
  feature was originally motivated by: does a point recovered from
  distance-only probes also predict same_triangle correctly for its
  held-out pairs, now that this run confirms same_triangle is learnable
  to begin with at full scale.
- The bulk-metric drop is large enough to be worth a `same_triangle_weight`
  sweep (currently defaults to 1.0, matching `distance_weight`) - if
  same_triangle's gradient signal is simply dominating early training and
  pulling the embedding table away from the linear-ambient-coordinate
  solution distance alone finds, downweighting it might recover more of
  002/004's bulk-metric quality without sacrificing either task's
  held-out performance.

---

## 077 - Does 10x the training steps close 076's bulk-metric gap?

### 1. Model settings

New config `configs/octahedron_distance_same_triangle_100k.yaml`: identical
to 076's `octahedron_distance_same_triangle.yaml` (800 points, emb_dim=32,
`training.tasks: [distance, same_triangle]`, same seed) except
`training.steps: 100000` (10x 076's 10000), `progress_interval` kept at 500
(201 snapshot points instead of 21). Purpose: 076 found that adding
same_triangle made every raw-embedding bulk metric substantially worse than
the distance-only baseline (002/004) despite both task heads generalizing
almost perfectly - this tests whether that's a training-budget artifact
(the shared embedding just needs more steps to untangle two tasks' gradient
signals) or a real capacity/objective-conflict effect that persists
regardless of budget.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/077/`

**Headline: more steps recovers most of the gap, but unevenly across
metrics - and two of them end up beating not just 076 but the distance-only
baseline outright.**

| metric | distance-only @10k (002/004) | distance+same_triangle @10k (076) | distance+same_triangle @100k (077) |
|---|---:|---:|---:|
| PCA explained variance (3 comp) | 0.792 | 0.660 | **0.523** (worse than both) |
| Cross-validated coordinate R² | 0.96 ± 0.004 | 0.722 ± 0.005 | **0.907 ± 0.016** (closes most of the gap) |
| Geodesic-distance Spearman | 0.887 | 0.382 | **0.591** (closes some of the gap) |
| Nearest-neighbor recall | 0.326 | 0.129 | **0.496** (beats both baselines) |
| Intrinsic dimension | 10.45 | 13.14 | **7.01** (beats both baselines) |

- **NN recall and intrinsic dimension don't just recover, they overshoot
  the single-task baseline.** At 100000 steps the multi-task embedding has
  both better local-neighborhood structure (0.496 vs. 002/004's 0.326) and
  a more compact, closer-to-ambient_dim=3 embedding (7.01 vs. 10.45) than
  distance-only training ever achieved at 10000 steps - training
  same_triangle alongside distance is not simply a tax on these two metrics,
  given enough steps it's a net benefit, consistent with this file's
  earlier finding (004/005/010) that these two metrics specifically are the
  slow-converging ones regardless of setup, and multi-task training just
  needed proportionally more steps to reach the same neighborhood.
- **Cross-validated R² and Spearman narrow the gap substantially but do not
  fully close it even at 10x the steps** (R² 0.907 vs. 0.96 target,
  Spearman 0.591 vs. 0.887 target) - `progress.csv` shows both still rising
  steadily at step 100000 with no sign of flattening (cv R² 0.880 at 75000
  -> 0.907 at 100000; Spearman 0.539 -> 0.591 over the same window), so more
  steps would likely close more of the remaining gap, but slowly.
- **PCA explained variance is the one metric that gets worse, not better,
  with more training - and non-monotonically.** `progress.csv` shows it
  peaking around step 2000-5000 (~0.675, already above 076's final 0.660)
  then declining steadily for the remaining 95000+ steps to 0.523, ending
  below even 076's 10000-step value. This is the same PCA-vs-intrinsic-
  dimension disagreement 005 first found in the distance-only setting
  (top-3-component share shrinks as the embedding spreads variance across
  more of its 32 dimensions, even as the *local*, nonlinear intrinsic
  dimension estimate keeps shrinking toward the manifold's true
  dimensionality) - it reproduces here in the multi-task setting too, and
  is now confirmed independent of whether same_triangle is in the mix.
- **Both task heads' held-out generalization remains essentially perfect
  and is not why the bulk metrics improved.** distance held-out loss
  dropped 10x (0.00037 -> 0.00003, tracking the 10x lower final training
  loss) but same_triangle held-out loss is essentially unchanged (0.00064
  -> 0.00070) since `progress.csv` shows it was already saturated
  (accuracy/AUC ~0.9999) by step 2000 - same_triangle's task-level
  generalization was never the bottleneck; the raw embedding-table geometry
  was.
- Entity-token CKA at layer 2 roughly tripled versus 076 for both tasks
  (distance 0.19 -> 0.54, same_triangle 0.22 -> 0.53), while layer 1 dropped
  slightly (distance 0.44 -> 0.26, same_triangle 0.43 -> 0.30) - with more
  training the i/j entity-token representations become far more similar
  deeper in the network, the same direction 002 found for distance-only
  training, now confirmed to hold with same_triangle present too.

**Bottom line: 076's bulk-metric degradation was mostly, but not entirely, a
training-budget artifact.** NN recall and intrinsic dimension fully recover
(and then some) with 10x the steps; cross-validated R² and Spearman recover
most but not all of the gap and are still improving at step 100000; PCA
explained variance is the exception that gets worse with more training,
for reasons unrelated to same_triangle specifically (it's the same
non-monotonic PCA/intrinsic-dimension split 005 found in distance-only
training).

### 3. Next steps

- cv R² and Spearman are still rising at step 100000 - an even longer run
  (e.g. 200000 steps) would show whether they eventually reach the
  distance-only baseline or asymptote below it, which would distinguish
  "same_triangle costs a fixed, irrecoverable amount of linear-decodability"
  from "it just needs even more steps than 10x."
- Run the `same_triangle_weight` sweep 076 proposed (currently both task
  weights default to 1.0) at a shorter, cheaper step budget now that it's
  clear budget alone explains most (not all) of the original gap - this
  would separate "downweighting recovers quality faster" from "downweighting
  recovers quality that more steps wouldn't have recovered anyway."
- Now that PCA explained variance is confirmed to decline with training
  independent of same_triangle, revisit 005's original next-step (plot the
  full PCA explained-variance spectrum, not just the top-3 sum) using this
  run's 201 snapshots to see whether the same "variance migrates to higher
  components" story holds in the multi-task setting.

---

## 078 - same_triangle_weight sweep, point 1/10: same_triangle_weight=0.1

### 1. Model settings

New config `configs/same_triangle_weight_sweep/octahedron_stw_0.1.yaml`:
same setup as 076's `octahedron_distance_same_triangle.yaml` (800 points,
emb_dim=32, `training.tasks: [distance, same_triangle]`, seed 0 for both
manifold sampling and training) except `training.steps` cut to 20000 (a
shared, shorter budget across the whole sweep - affordable per-point cost
given ten runs, and per 077's finding that most of the training-budget
effect on bulk metrics is visible well before 100000 steps) and the new
`training.same_triangle_weight: 0.1` (`distance_weight` stays at its
default of 1.0). First of ten sweep points (0.1 through 1.0 in steps of
0.1), directly testing 076's proposed hypothesis: does downweighting
same_triangle's gradient contribution recover more of 002/004's
distance-only bulk-embedding quality, without sacrificing either task's
held-out generalization? Because every point in this sweep (078-087) uses
the identical seed for both manifold sampling and training, the *only*
thing that differs run-to-run is `same_triangle_weight` - a clean
single-variable ablation.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/078/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6095 |
| Cross-validated coordinate R² | 0.8993 ± 0.0088 |
| Geodesic-distance Spearman | 0.6532 |
| Nearest-neighbor recall | 0.2587 |
| Intrinsic dimension | 11.4985 |
| distance held-out / in-dist. loss | 0.00058 / 0.00057 |
| same_triangle held-out loss / accuracy | 0.00092 / 0.99972 |

- Already a large improvement over 076's `weight=1.0`/10000-step numbers
  (cv R² 0.899 vs. 0.722, Spearman 0.653 vs. 0.382) - downweighting alone,
  even at the lowest weight tried, recovers a substantial chunk of the gap
  to the distance-only baseline (002/004: cv R² 0.96, Spearman 0.887).
- But this turns out **not** to be the best point in the sweep on held-out
  generalization: distance held-out loss (0.00058) and same_triangle
  held-out loss (0.00092) are both the second-worst of the entire ten-point
  sweep - see 087 for the full table. Low `same_triangle_weight` helps bulk
  embedding-table decodability but does not by itself give the best
  task-level generalization.

### 3. Next steps

- See 087 for the full ten-point comparison table and overall sweep trend.

---

## 079 - same_triangle_weight sweep, point 2/10: same_triangle_weight=0.2

### 1. Model settings

New config `configs/same_triangle_weight_sweep/octahedron_stw_0.2.yaml`:
identical to 078 except `training.same_triangle_weight: 0.2`. Second of
ten sweep points.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/079/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.5932 |
| Cross-validated coordinate R² | 0.9515 ± 0.0045 |
| Geodesic-distance Spearman | 0.7755 |
| Nearest-neighbor recall | 0.2975 |
| Intrinsic dimension | 10.1031 |
| distance held-out / in-dist. loss | 0.00005 / 0.00005 |
| same_triangle held-out loss / accuracy | 0.00001 / 1.00000 |

- **Headline: this is the best point in the entire ten-point sweep, on
  every single metric simultaneously** - highest cv R² (0.9515), highest
  Spearman (0.7755), highest NN recall (0.2975), lowest (most compact)
  intrinsic dimension (10.1031), and the lowest distance held-out loss
  (0.00005) of any weight tried, only a single step in from 078's
  `weight=0.1`. See 087 for the full table confirming no other weight
  matches it on any metric.
- `progress.csv` shows it isn't just a better endpoint - at every
  snapshot from step 2000 onward, this run's cv R²/Spearman are already
  ahead of `weight=1.0`'s (087) equivalent snapshot, so `weight=0.2`
  converges faster as well as further, not just to a better final value.

### 3. Next steps

- See 087 for the full ten-point comparison table and overall sweep trend.

---

## 080 - same_triangle_weight sweep, point 3/10: same_triangle_weight=0.3

### 1. Model settings

New config `configs/same_triangle_weight_sweep/octahedron_stw_0.3.yaml`:
identical to 078/079 except `training.same_triangle_weight: 0.3`. Third
of ten sweep points.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/080/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6397 |
| Cross-validated coordinate R² | 0.8667 ± 0.0111 |
| Geodesic-distance Spearman | 0.5528 |
| Nearest-neighbor recall | 0.2250 |
| Intrinsic dimension | 11.4791 |
| distance held-out / in-dist. loss | 0.00013 / 0.00013 |
| same_triangle held-out loss / accuracy | 0.00001 / 1.00000 |

- **A sharp drop from 079's peak**, back down close to 078's level on cv
  R² (0.8667 vs. 079's 0.9515) and below it on Spearman (0.5528 vs. 078's
  0.6532) - for a weight only 0.1 higher than 079's standout point. The
  relationship between `same_triangle_weight` and bulk embedding quality
  is not smooth or monotonic even at this fine (0.1) granularity, unlike
  the clean, monotonic `emb_dim` trends found in 007/008.
- Both tasks' held-out generalization stays essentially perfect regardless
  (same_triangle loss 0.00001, matching 079) - the weight only affects
  bulk embedding-table decodability here, not task-level generalization.

### 3. Next steps

- See 087 for the full ten-point comparison table and overall sweep trend.

---

## 081 - same_triangle_weight sweep, point 4/10: same_triangle_weight=0.4

### 1. Model settings

New config `configs/same_triangle_weight_sweep/octahedron_stw_0.4.yaml`:
identical to the earlier sweep points except `training.same_triangle_weight:
0.4`. Fourth of ten sweep points.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/081/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6064 |
| Cross-validated coordinate R² | 0.9031 ± 0.0067 |
| Geodesic-distance Spearman | 0.5605 |
| Nearest-neighbor recall | 0.2188 |
| Intrinsic dimension | 12.1558 |
| distance held-out / in-dist. loss | 0.00008 / 0.00007 |
| same_triangle held-out loss / accuracy | 0.00000 / 1.00000 |

- Partial recovery from 080's dip on cv R² (0.9031 vs. 080's 0.8667), but
  Spearman barely moves (0.5605 vs. 080's 0.5528) - the different bulk
  metrics don't move in lockstep point-to-point, echoing 008's observation
  that different metrics can have different "slow"/"fast" behavior, here
  extended to weight rather than step count.
- Intrinsic dimension (12.1558) is the worst (least compact) of the sweep
  so far - worse than every one of 078-080 despite cv R² being mid-pack.

### 3. Next steps

- See 087 for the full ten-point comparison table and overall sweep trend.

---

## 082 - same_triangle_weight sweep, point 5/10: same_triangle_weight=0.5

### 1. Model settings

New config `configs/same_triangle_weight_sweep/octahedron_stw_0.5.yaml`:
identical to the earlier sweep points except `training.same_triangle_weight:
0.5`. Fifth of ten sweep points - the midpoint of the swept range.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/082/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6501 |
| Cross-validated coordinate R² | 0.8690 ± 0.0092 |
| Geodesic-distance Spearman | 0.4566 |
| Nearest-neighbor recall | 0.2100 |
| Intrinsic dimension | 11.4750 |
| distance held-out / in-dist. loss | 0.00011 / 0.00010 |
| same_triangle held-out loss / accuracy | 0.00001 / 1.00000 |

- Spearman drops to 0.4566 - the first point in the sweep to fall below
  0.5, continuing the bumpy (not smooth) descent from 079's 0.7755 peak
  seen across 080/081 as well.
- PCA explained variance (0.6501) is the highest of the sweep so far, even
  though cv R²/Spearman are mid-to-low - reconfirming (as 077 already
  found within a single run) that PCA explained variance and the other
  bulk metrics don't move together, this time across weight rather than
  training step.

### 3. Next steps

- See 087 for the full ten-point comparison table and overall sweep trend.

---

## 083 - same_triangle_weight sweep, point 6/10: same_triangle_weight=0.6

### 1. Model settings

New config `configs/same_triangle_weight_sweep/octahedron_stw_0.6.yaml`:
identical to the earlier sweep points except `training.same_triangle_weight:
0.6`. Sixth of ten sweep points.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/083/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6321 |
| Cross-validated coordinate R² | 0.7817 ± 0.0124 |
| Geodesic-distance Spearman | 0.4524 |
| Nearest-neighbor recall | 0.1762 |
| Intrinsic dimension | 11.0047 |
| distance held-out / in-dist. loss | 0.00011 / 0.00010 |
| same_triangle held-out loss / accuracy | 0.00000 / 1.00000 |

- Nearest-neighbor recall (0.1762) is the worst of the entire ten-point
  sweep - lower than even 076's original `weight=1.0`/10000-step run
  (0.129 was 076's, but that used only 10000 steps; among this sweep's
  20000-step runs, 0.1762 is the minimum, see 087's full table).
- cv R² (0.7817) is the second-lowest of the sweep so far - `weight=0.6`
  is, together with 087 (`weight=1.0`, see below), one of the two weakest
  points found for linear decodability.

### 3. Next steps

- See 087 for the full ten-point comparison table and overall sweep trend.

---

## 084 - same_triangle_weight sweep, point 7/10: same_triangle_weight=0.7

### 1. Model settings

New config `configs/same_triangle_weight_sweep/octahedron_stw_0.7.yaml`:
identical to the earlier sweep points except `training.same_triangle_weight:
0.7`. Seventh of ten sweep points.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/084/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6514 |
| Cross-validated coordinate R² | 0.7733 ± 0.0103 |
| Geodesic-distance Spearman | 0.4009 |
| Nearest-neighbor recall | 0.1862 |
| Intrinsic dimension | 11.5142 |
| distance held-out / in-dist. loss | 0.00013 / 0.00012 |
| same_triangle held-out loss / accuracy | 0.00000 / 1.00000 |

- **Geodesic-distance Spearman (0.4009) is the lowest of the entire
  ten-point sweep** - lower even than `weight=1.0` (087, Spearman 0.4140).
  cv R² (0.7733) is also the lowest of the sweep so far. `weight=0.6-0.7`
  is the sweep's weakest region for distance-structure recovery, not the
  `weight=1.0` endpoint one might have expected going in.
- PCA explained variance (0.6514) is the highest of the sweep to this
  point - the same PCA/other-metric disagreement seen at 082 recurs here.

### 3. Next steps

- See 087 for the full ten-point comparison table and overall sweep trend.

---

## 085 - same_triangle_weight sweep, point 8/10: same_triangle_weight=0.8

### 1. Model settings

New config `configs/same_triangle_weight_sweep/octahedron_stw_0.8.yaml`:
identical to the earlier sweep points except `training.same_triangle_weight:
0.8`. Eighth of ten sweep points.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/085/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6464 |
| Cross-validated coordinate R² | 0.8262 ± 0.0166 |
| Geodesic-distance Spearman | 0.4956 |
| Nearest-neighbor recall | 0.1975 |
| Intrinsic dimension | 12.1863 |
| distance held-out / in-dist. loss | 0.00032 / 0.00032 |
| same_triangle held-out loss / accuracy | 0.00001 / 1.00000 |

- A mild recovery from 083/084's dip on cv R²/Spearman (0.8262/0.4956 vs.
  084's 0.7733/0.4009), but intrinsic dimension (12.1863) is now the worst
  (least compact) of the sweep so far, and distance held-out loss (0.00032)
  is the third-worst - improving on some metrics while getting worse on
  others, the same non-lockstep pattern seen throughout this sweep.

### 3. Next steps

- See 087 for the full ten-point comparison table and overall sweep trend.

---

## 086 - same_triangle_weight sweep, point 9/10: same_triangle_weight=0.9

### 1. Model settings

New config `configs/same_triangle_weight_sweep/octahedron_stw_0.9.yaml`:
identical to the earlier sweep points except `training.same_triangle_weight:
0.9`. Ninth of ten sweep points, closest to the `weight=1.0` (087) endpoint.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/086/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6350 |
| Cross-validated coordinate R² | 0.8379 ± 0.0094 |
| Geodesic-distance Spearman | 0.4639 |
| Nearest-neighbor recall | 0.1812 |
| Intrinsic dimension | 12.3813 |
| distance held-out / in-dist. loss | 0.00105 / 0.00104 |
| same_triangle held-out loss / accuracy | 0.00004 / 1.00000 |

- **Intrinsic dimension (12.3813) is the worst (least compact) of the
  entire ten-point sweep**, and distance held-out loss (0.00105) is also
  the worst of the sweep - both records set here, right before the
  `weight=1.0` endpoint. Bulk decodability (cv R² 0.8379, Spearman 0.4639)
  is mid-pack, not the sweep's worst, so the degradation at this weight is
  concentrated in embedding compactness and held-out loss specifically,
  not in linear recoverability.

### 3. Next steps

- See 087 for the full ten-point comparison table and overall sweep trend.

---

## 087 - same_triangle_weight sweep, point 10/10: same_triangle_weight=1.0

### 1. Model settings

New config `configs/same_triangle_weight_sweep/octahedron_stw_1.0.yaml`:
identical to the earlier sweep points except `training.same_triangle_weight:
1.0` (matching `distance_weight`'s default - this is the same *weighting*
as 076/077, but at 077's step budget cut down to 20000, i.e. a direct
20000-step point on 077's own training trajectory). Tenth and last point in
the sweep, closing it out.

### 2. Results / findings

Result folder: `experiments/manifold_learning/octahedron/087/`

| metric | value |
|---|---:|
| PCA explained variance (3 comp) | 0.6113 |
| Cross-validated coordinate R² | 0.7481 ± 0.0144 |
| Geodesic-distance Spearman | 0.4140 |
| Nearest-neighbor recall | 0.2300 |
| Intrinsic dimension | 11.5828 |
| distance held-out / in-dist. loss | 0.00018 / 0.00016 |
| same_triangle held-out loss / accuracy | 0.00216 / 0.99972 |

As a sanity check: every value in this row exactly matches 077's own
`progress.csv` snapshot at step 20000 (cv R² 0.7481, Spearman 0.4140, NN
recall 0.2300, intrinsic dim 11.5828, PCA 0.6113) - confirming this sweep
point and 077 are literally the same training run up to step 20000 (same
seed, same weight), a useful cross-check that the sweep's config-generation
and analysis pipeline are correct.

**Full ten-point sweep table (`same_triangle_weight` 0.1-1.0, all at 20000
steps, seed 0, otherwise identical to 076/077):**

| weight | exp | PCA(3) | cv R² | Spearman | NN recall | intrinsic dim | dist held-out loss | same_triangle held-out loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 078 | 0.6095 | 0.8993 ± 0.0088 | 0.6532 | 0.2587 | 11.4985 | 0.00058 | 0.00092 |
| 0.2 | 079 | 0.5932 | **0.9515 ± 0.0045** | **0.7755** | **0.2975** | **10.1031** | **0.00005** | 0.00001 |
| 0.3 | 080 | 0.6397 | 0.8667 ± 0.0111 | 0.5528 | 0.2250 | 11.4791 | 0.00013 | 0.00001 |
| 0.4 | 081 | 0.6064 | 0.9031 ± 0.0067 | 0.5605 | 0.2188 | 12.1558 | 0.00008 | **0.00000** |
| 0.5 | 082 | **0.6501** | 0.8690 ± 0.0092 | 0.4566 | 0.2100 | 11.4750 | 0.00011 | 0.00001 |
| 0.6 | 083 | 0.6321 | 0.7817 ± 0.0124 | 0.4524 | 0.1762 | 11.0047 | 0.00011 | **0.00000** |
| 0.7 | 084 | 0.6514 | 0.7733 ± 0.0103 | 0.4009 | 0.1862 | 11.5142 | 0.00013 | **0.00000** |
| 0.8 | 085 | 0.6464 | 0.8262 ± 0.0166 | 0.4956 | 0.1975 | 12.1863 | 0.00032 | 0.00001 |
| 0.9 | 086 | 0.6350 | 0.8379 ± 0.0094 | 0.4639 | 0.1812 | 12.3813 | 0.00105 | 0.00004 |
| 1.0 | 087 | 0.6113 | 0.7481 ± 0.0144 | 0.4140 | 0.2300 | 11.5828 | 0.00018 | 0.00216 |

(bold = best value in the column)

**Headline finding: `same_triangle_weight=0.2` dominates the entire
sweep** - it has the single best value of every bulk embedding-quality
metric (PCA is the one exception, see below) *and* the best distance
held-out loss, not just a local bump on an otherwise-smooth curve. This is
the same "one specific point strictly dominates every other point tried"
shape this file already found once before for `emb_dim` (009's
`emb_dim=24` beating every other width on every metric) - here it recurs
for a loss-weighting hyperparameter instead of an architectural one.

- **The weight-vs-quality relationship is not monotonic or smooth at
  all** - going 0.1 -> 0.2 -> 0.3 -> 0.4 swings cv R² up (0.899) then up
  further to a peak (0.952) then sharply down (0.867) then back up
  (0.903), and the pattern keeps oscillating through the rest of the
  sweep rather than settling into a trend. This rules out a simple "lower
  weight is uniformly better" or "there's a smooth optimum" story - unlike
  `emb_dim`'s clean monotonic-then-dominant-outlier shape (007-009), this
  hyperparameter's effect on bulk quality looks closer to noisy/chaotic
  across its range, at least at this single-seed, 20000-step resolution.
- **`weight=1.0` (087, matching 076/077's original setting) is not the
  sweep's worst point on most metrics** - it's actually mid-to-mediocre:
  worse than 078-082 on cv R² but better than 083/084 on NN recall
  (0.2300, second-best of the whole sweep after 079's 0.2975), and its
  distance held-out loss (0.00018) is only fifth-worst, not the worst.
  What *is* clearly worst for `weight=1.0` is same_triangle held-out loss
  (0.00216, the single worst value in the entire sweep, more than 5x the
  next-worst at 078's 0.00092) - full-weight same_triangle training
  produces the *worst* same_triangle generalization of any weight tried,
  a genuinely counter-intuitive result (one might expect more weight on a
  task to help that task's own generalization, not hurt it).
- **Held-out generalization losses for both tasks are worst at the two
  ends of the swept range (0.1 and 1.0) and best in the middle
  (0.2-0.7ish)**, a rough U-shape distinct from the noisy, non-monotonic
  shape of the bulk decodability metrics - 078 (weight=0.1) has the
  second-worst distance *and* same_triangle held-out loss, while 087
  (weight=1.0) has the worst same_triangle held-out loss outright. Every
  weight from 0.2 through 0.8 keeps same_triangle held-out loss at
  0.00001 or below, i.e. near-zero regardless of bulk-metric quality -
  held-out task-level generalization and raw-embedding bulk quality are
  clearly two different things responding to this hyperparameter in two
  different shapes.
- PCA explained variance is the outlier metric here too, same as in 077:
  its best value (082's 0.6501, weight=0.5) does not coincide with 079's
  overall-best point (0.5932, second lowest in the sweep) - reconfirming
  that PCA's top-3-component share and the other four bulk metrics
  disagree about what "good" means, independent of whether the varying
  axis is training steps (077) or loss weight (this sweep).

**Bottom line: downweighting `same_triangle` does help, but the single
best answer is a specific weight (0.2), not "as low as possible" or "equal
weighting" - and the relationship is noisy enough across 0.1-0.9 that
picking any other weight from this range by eyeballing a trend line would
likely be wrong.** 076's original hypothesis (downweighting recovers bulk
quality) is directionally correct, but the sweep shows the effect is far
from a clean, predictable dial.

### 3. Next steps

- `weight=0.2` (079) is a single seed's result at a single step count -
  worth confirming it's not a lucky training-noise draw by rerunning it
  (and maybe its neighbors, 0.1 and 0.3) at a second seed before treating
  0.2 as a real optimum rather than sweep noise, especially given how
  non-monotonic every neighboring point in this sweep turned out to be.
- Rerun the full 10-point sweep at 077's 100000-step budget (or at least
  the most interesting points: 0.2, 0.7, 0.9, 1.0) to see whether the
  non-monotonic pattern found here at 20000 steps is a converged property
  of each weight or a step-budget artifact that would smooth out with
  more training, the same open question 077 raised for the single
  `weight=1.0` point.
- The `weight=1.0`-has-worst-same_triangle-generalization finding is
  surprising enough to double check isn't an artifact of this specific
  seed/step-count combination - worth verifying whether it holds at
  077's 100000-step budget too, where same_triangle was already shown
  (077) to saturate almost immediately regardless of weight.