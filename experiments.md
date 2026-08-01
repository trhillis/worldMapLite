# Experiment Plan: Latent Geometry Across Topologies

## Research question

How does the topology of an underlying world affect the geometry that emerges in a neural network’s latent representation?

We will compare three topologies:

* Grid
* Möbius strip
* Octahedron

We want to investigate:

1. Whether the learned embedding recovers the world’s geodesic structure.
2. How the number of observed distance pairs affects latent-manifold formation.
3. Whether different topologies require different numbers of pairs.
4. How the representation changes during training.
5. Whether geometric structure and held-out prediction emerge together.
6. Whether any conditions show delayed generalization consistent with grokking.
7. How local neighborhoods, intrinsic dimension, and coordinate information change during learning.

Grokking is therefore one part of the project, not the entire research question.

---

## What we have

The main implementation is complete and available on the shared `main` branch.

It currently supports:

* grid, Möbius, and octahedron worlds;
* topology-aware geodesic distances;
* configurable training-pair budgets;
* fixed held-out evaluation pairs;
* nested training-pair sampling;
* multiple random seeds;
* periodic evaluation;
* periodic model checkpoints;
* post-training checkpoint analysis;
* aggregated CSV output.

The analysis already calculates the main metrics needed for the study:

* `embedding_distance_spearman`: global latent/geodesic correspondence;
* `nearest_recall`: local neighborhood preservation;
* `embedding_intrinsic_dimension`: representation dimensionality;
* `prediction_spearman`: held-out distance-ranking performance;
* `prediction_mae` and `prediction_rmse`: held-out prediction error;
* coordinate probe scores;
* PCA and CKA diagnostics.

The pilot experiments confirmed that the full training, checkpointing, evaluation, and analysis pipeline works for all three topologies.

The supervision-budget extensions now add fixed pair splits, configurable
checkpoint schedules, and a separate held-out-point recovery protocol. These
are controls for the existing research question, not an expansion into model
architecture or optimizer search. Use `configs/supervision_sweep.json` as the
canonical configurable sweep example and keep all split/world seeds fixed
across model seeds.

---

## What we want to do

We will run the same experimental design for each topology.

### Pair budgets

* 1,000
* 2,000
* 3,000
* 5,000
* 7,500
* 10,000

### Model seeds

* 0
* 1
* 2

### Fixed settings

* Pair seed: `0`
* Held-out pairs: `1,000`
* Training steps: `50,000`
* Evaluation interval: `1,000`
* Checkpoint interval: `5,000`

This produces:

* 18 runs per topology;
* 54 runs total;
* 10 analyzed checkpoints per run.

The experiment will identify:

* the pair budget at which geometry begins to emerge;
* topology-dependent differences in that threshold;
* the timing of geometric and predictive generalization;
* whether local structure develops before global structure;
* whether the latent representation compresses during learning;
* whether low-budget runs eventually grok or continue memorizing.

---

# Work split

Each person uses the same version of `main` and owns one complete topology.

Each person is responsible for:

1. running all 18 assigned experiments;
2. checking that logs and checkpoints were created;
3. running the analysis;
4. validating the resulting CSV rows;
5. producing plots for their topology;
6. writing a short interpretation of their findings.

## Person 1: Grid

Runs all combinations of:

* six pair budgets;
* three model seeds;
* grid topology.

Main questions:

* At what pair budget does the flat grid geometry emerge?
* Does prediction improve at the same time as embedding geometry?
* Do low-pair grid runs show delayed generalization?

## Person 2: Möbius strip

Runs all combinations of:

* six pair budgets;
* three model seeds;
* Möbius topology.

Main questions:

* How many pairs are needed to recover the Möbius geodesic structure?
* Does non-orientability affect learning or sample efficiency?
* Does local structure emerge before global structure?

## Person 3: Octahedron

Runs all combinations of:

* six pair budgets;
* three model seeds;
* octahedron topology.

Main questions:

* How many pairs are needed to recover the polyhedral surface geometry?
* Does the octahedron differ from the grid and Möbius strip?
* How do its latent dimensionality and neighborhood preservation develop?

---

# Shared setup

Before starting, all three people should run:

```bash
git switch main
git pull --ff-only
git status
git rev-parse HEAD
```

Everyone must use the same commit hash.

Each person should also record their environment:

```bash
mkdir -p experiment_notes

git rev-parse HEAD > experiment_notes/git_commit.txt
python --version > experiment_notes/environment.txt
pip freeze >> experiment_notes/environment.txt
```

Generated models, logs, and analysis results should not require separate code branches.

---

# How to run

## Person 1: Grid

```bash
mkdir -p logs

for seed in 0 1 2; do
  for budget in 1000 2000 3000 5000 7500 10000; do
    python src/train_multitask.py \
      --world-type grid \
      --distance-pairs "$budget" \
      --eval-pairs 1000 \
      --eval-every 1000 \
      --checkpoint-every 5000 \
      --steps 50000 \
      --pair-seed 0 \
      --seed "$seed" \
      2>&1 | tee "logs/grid_pairs${budget}_pairseed0_seed${seed}.log"
  done
done
```

## Person 2: Möbius strip

```bash
mkdir -p logs

for seed in 0 1 2; do
  for budget in 1000 2000 3000 5000 7500 10000; do
    python src/train_multitask.py \
      --world-type manifold \
      --manifold mobius \
      --distance-pairs "$budget" \
      --eval-pairs 1000 \
      --eval-every 1000 \
      --checkpoint-every 5000 \
      --steps 50000 \
      --pair-seed 0 \
      --seed "$seed" \
      2>&1 | tee "logs/mobius_pairs${budget}_pairseed0_seed${seed}.log"
  done
done
```

## Person 3: Octahedron

```bash
mkdir -p logs

for seed in 0 1 2; do
  for budget in 1000 2000 3000 5000 7500 10000; do
    python src/train_multitask.py \
      --world-type manifold \
      --manifold octahedron \
      --distance-pairs "$budget" \
      --eval-pairs 1000 \
      --eval-every 1000 \
      --checkpoint-every 5000 \
      --steps 50000 \
      --pair-seed 0 \
      --seed "$seed" \
      2>&1 | tee "logs/octahedron_pairs${budget}_pairseed0_seed${seed}.log"
  done
done
```

---

# How to analyze

After the assigned runs finish, each person runs:

```bash
python analysis/analysis_multitask.py
```

The main output is:

```text
analysis_results/all_runs.csv
```

Each person should verify:

* all 18 runs are present;
* each run has checkpoints from 5,000 through 50,000 steps;
* primary metrics contain no missing values;
* there are no duplicate conditions.

A useful summary command is:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("analysis_results/all_runs.csv")

columns = [
    "world",
    "seed",
    "relation_budget",
    "training_step",
    "embedding_distance_spearman",
    "nearest_recall",
    "embedding_intrinsic_dimension",
    "prediction_mae",
    "prediction_spearman",
]

print(
    df[columns]
    .sort_values(["relation_budget", "seed", "training_step"])
    .to_string(index=False)
)
PY
```

---

# Analysis for each topology

Each person should produce four main plots.

## 1. Pair budget versus final geometry

Plot final `embedding_distance_spearman` against pair budget.

This shows when a meaningful latent manifold begins to emerge.

## 2. Training step versus geometry

Plot `embedding_distance_spearman` over training for each pair budget.

This shows whether geometry appears early, gradually, or after a delay.

## 3. Prediction and geometry over training

Compare:

* `prediction_spearman`;
* `embedding_distance_spearman`.

This tests whether predictive generalization and latent organization emerge together.

## 4. Supporting representation changes

Examine:

* `nearest_recall`;
* `embedding_intrinsic_dimension`;
* coordinate probe scores.

This shows how local structure, compression, and coordinate information relate to global geometry.

Each person should summarize:

* the approximate pair threshold;
* the strongest and weakest conditions;
* whether results are stable across seeds;
* whether there is evidence of delayed grokking;
* how local and global structure relate;
* how intrinsic dimension changes.

---

# Combining the three parts

After all three topology studies are complete, combine their checkpoint files or CSV results.

The group should then compare:

* pair thresholds across topologies;
* final latent-geometric quality;
* speed of geometric emergence;
* held-out generalization;
* local neighborhood recovery;
* intrinsic-dimensional compression;
* evidence for or against grokking.

The final project should include both:

1. separate findings for grid, Möbius, and octahedron;
2. a direct cross-topology comparison.

The implementation phase is complete. The next phase is running, analyzing, and interpreting the experiments.
