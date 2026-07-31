# Lite World Map Representations

This repository investigates when neural networks develop geometric internal
representations of an underlying world from relational supervision. It is a
smaller experimental counterpart to Core Park's
[World Map Representation](https://github.com/cfpark00/world-map-representation)
project.

## Research question

> How do the topology of an underlying space and the amount of pairwise-distance
> supervision interact to influence the emergence of geometric representations?

The model receives entity identities and distances between selected pairs. It
does not receive the entities' coordinates during training. After training, we
test whether the learned entity embeddings recover the geometry of the original
space.

The main questions are:

- How many observed distance relations are needed before geometry emerges?
- Does local geometry emerge before global structure?
- Do different topologies require different amounts of supervision?
- Is there a sharp transition between unstructured and geometric embeddings?
- Can a model predict distances without learning a faithful internal map?

## Experimental design

For a world containing `N` entities, there are `N(N-1)/2` unique unordered
distance relations. The default worlds contain 400 entities, giving 79,800
possible relations.

Models are trained on nested subsets of these relations:

```text
1,000  2,000  3,000  5,000  10,000  20,000  40,000  79,800
```

For a fixed pair seed, all possible pairs are placed in one reproducible random
ordering. Each budget uses a prefix of that ordering:

```text
1,000-pair set
    ⊂ 2,000-pair set
    ⊂ 3,000-pair set
    ⊂ ...
    ⊂ full pair set
```

This lets us measure representation quality as a function of supervision while
changing only the number of available relations.

## Worlds

### Primary comparison

The primary experiment focuses on intrinsically flat spaces with different
global structures.

| World | Boundary | Orientable | Global structure | Status |
|---|---:|---:|---|---|
| Rectangular grid | Yes | Yes | No periodic directions | Implemented |
| Flat Möbius strip | Yes | No | One twisted periodic direction | Implemented |
| Flat torus | No | Yes | Two periodic directions | In progress |

Because these spaces are locally flat, they provide the cleanest available
comparison of how global structure affects representation emergence. Boundary
and orientability still differ, so a flat cylinder would be a useful future
control.

### Curved extension

A regular octahedron is also implemented. Its surface is closed and
sphere-like, with positive curvature concentrated at its vertices. It changes
both topology and geometry, so it is treated as an extension rather than the
primary topology control:

> Do patterns observed in flat spaces generalize to curved or piecewise-flat
> spaces?

## Model and pipeline

```text
Underlying world
        ↓
Unique sampled distance relations
        ↓
Shared entity embeddings + transformer
        ↓
Distance prediction
        ↓
Representation analysis
```

Ground-truth coordinates and geodesic distances are used for evaluation, not
as model inputs.

## Installation

```bash
git clone https://github.com/trhillis/worldMapLite.git
cd worldMapLite
pip install -r requirements.txt
```

## Training

Train a grid model on 1,000 unique relations:

```bash
python src/train_multitask.py \
    --world-type grid \
    --distance-pairs 1000 \
    --pair-seed 0 \
    --seed 0
```

Train on every relation in the default 20-by-20 grid:

```bash
python src/train_multitask.py \
    --world-type grid \
    --distance-pairs 79800 \
    --pair-seed 0 \
    --seed 0
```

Train a Möbius-strip model:

```bash
python src/train_multitask.py \
    --world-type manifold \
    --manifold mobius \
    --distance-pairs 5000 \
    --pair-seed 0 \
    --seed 0
```

Train an octahedron model:

```bash
python src/train_multitask.py \
    --world-type manifold \
    --manifold octahedron \
    --distance-pairs 5000 \
    --pair-seed 0 \
    --seed 0
```

`--pair-seed` selects the shared ordering of relations. Keep it fixed when
comparing budgets. `--seed` controls model randomness and, for sampled
manifolds, world sampling.

## Running a pair-budget sweep

A sweep can be pasted directly into a Bash terminal:

```bash
for pairs in 1000 2000 3000 5000 10000 20000 40000 79800; do
    python src/train_multitask.py \
        --world-type grid \
        --distance-pairs "$pairs" \
        --pair-seed 0 \
        --seed 0
done
```

For multiple seeds:

```bash
for seed in 0 1 2 3 4; do
    for pairs in 1000 2000 3000 5000 10000 20000 40000 79800; do
        python src/train_multitask.py \
            --world-type grid \
            --distance-pairs "$pairs" \
            --pair-seed 0 \
            --seed "$seed"
    done
done
```

Checkpoints include the world, budget, pair seed, and model seed:

```text
models/grid_distance_pairs1000_pairseed0_seed0_model.pt
models/grid_distance_pairs5000_pairseed0_seed0_model.pt
```

The exact training pairs are stored in each checkpoint for reproducibility.

## Analysis

Analyze all distance checkpoints with:

```bash
python analysis/analysis_multitask.py
```

Outputs are saved under `analysis_results/`:

```text
analysis_results/all_runs.csv
analysis_results/world_summary.csv
analysis_results/plots/
```

`all_runs.csv` contains one ordinary row per checkpoint. `world_summary.csv`
uses two header rows (`mean` and `std`) and groups results by world and relation
budget. Standard deviations are empty when only one seed is available.

### Metrics

Task performance and representation quality are measured separately.

Task metrics:

- distance-prediction MAE and RMSE
- predicted/true distance Spearman correlation

Entity-representation metrics:

- cross-validated coordinate-probe R²
- latent/true geodesic-distance Spearman correlation
- nearest-neighbour recall
- estimated intrinsic dimension
- PCA and coordinate-reconstruction plots

This distinction matters because successful distance prediction does not by
itself demonstrate that the entity embeddings form a geometric map.

## Preliminary grid result

The first grid sweep used one model seed and 5,000 optimizer updates per
condition. It produced the following representation metrics:

| Observed pairs | CV coordinate R² | Distance Spearman | Nearest recall |
|---:|---:|---:|---:|
| 1,000 | -0.037 | 0.026 | 0.008 |
| 2,000 | 0.282 | 0.188 | 0.025 |
| 3,000 | 0.745 | 0.547 | 0.125 |
| 5,000 | 0.885 | 0.785 | 0.325 |
| 10,000 | 0.893 | 0.823 | 0.448 |
| 20,000 | 0.889 | 0.824 | 0.530 |
| 40,000 | 0.893 | 0.829 | 0.565 |
| 79,800 | 0.887 | 0.813 | 0.553 |

These preliminary results suggest three regimes:

1. At 1,000–2,000 relations, little grid geometry is recoverable.
2. Between approximately 2,000 and 5,000 relations, geometric structure
   emerges rapidly.
3. Beyond approximately 10,000 relations, global recovery largely plateaus,
   while local nearest-neighbour recovery continues improving.

At 5,000 observed relations—about 6.3% of all possible grid pairs—the learned
embeddings already support strong coordinate recovery. This is preliminary
evidence for a supervision-dependent transition, not yet a final result.

### Limitations of the preliminary result

- Only one training seed has been evaluated, so uncertainty is unknown.
- Every budget receives the same number of optimizer updates. Smaller datasets
  are therefore reused more often than larger datasets.
- Current prediction metrics do not exclude training pairs. As the budget
  increases, prediction evaluation contains increasing train/test overlap.
  Representation metrics remain useful, but prediction metrics should not yet
  be interpreted as strictly held-out generalization.
- The 32-dimensional embeddings contain strongly decodable grid coordinates,
  but they do not collapse to a literal two-dimensional plane.

The next steps are to repeat the grid sweep across multiple seeds, add a fixed
held-out relation set, and compare the recovery curves with the flat Möbius
strip and flat torus.

## Manifold interface

Manifolds implement a shared chart-coordinate API:

- `sample(n, rng=...)`
- `distance(p, q)`
- `distance_matrix(p, q=None)`
- `embed(points)`

Chart coordinates are used for intrinsic distance calculations. Ambient
coordinates are used for visualization and probing. See
[`manifolds/README.md`](manifolds/README.md) for implementation details.

## Tests

```bash
python -m pytest -q
```

The tests cover manifold properties and distance-pair sampling, including
unique unordered relations, nested supervision budgets, and invalid-budget
detection.

## Research progression

1. Establish the supervision-dependent transition on the rectangular grid.
2. Repeat the sweep on the flat Möbius strip.
3. Complete the flat-torus implementation and repeat the sweep.
4. Add a flat cylinder as a topology control.
5. Use the octahedron as a curved, piecewise-flat generalization test.

The broader goal is to determine when relational tasks cause an internal
geometric model of the world to emerge, and how that process depends on the
world's global structure.
