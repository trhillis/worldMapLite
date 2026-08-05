# Lite World Map Representations

## Project overview

This project studies whether neural networks learn faithful geometric representations from sparse supervision on pairwise geodesic distances. We investigate how supervision budget and underlying topology affect generalization and the emergence of geometric structure in learned embeddings.

## Installation

```bash
git clone https://github.com/trhillis/worldMapLite.git
cd worldMapLite
pip install -r requirements.txt
```

## Experimental protocol

Each environment contains 400 entities. Models are trained on nested subsets of pairwise geodesic distances while sweeping supervision budget and model seed. Evaluation is performed on a fixed set of 1,000 held-out pairs disjoint from training.

The held-out-point recovery experiment excludes 20 points and all incident pairs from base-model training. With the trained model frozen, each new point embedding is optimized from distances to 4, 8, or 16 retained anchors, then evaluated using distances to unused retained points.

## Evaluation metrics

- **Held-out-pair R²:** variance in unseen pair distances explained by model predictions.
- **Held-out-pair Spearman correlation:** rank agreement between predicted and true distances for fixed unseen pairs.
- **Embedding-distance / geodesic-distance Spearman correlation:** rank agreement between Euclidean embedding distances and true geodesic distances.
- **Nearest-neighbour recall:** fraction of points whose closest embedding neighbour is also a closest geodesic neighbour.
- **Held-out-point recovery Spearman correlation:** rank agreement between predicted and true distances from a recovered point to unused evaluation anchors.
- **Coordinate probe:** cross-validated Ridge-regression R² for decoding grid coordinates or manifold ambient coordinates from embeddings.
- **Intrinsic dimension:** TwoNN estimate of the dimensionality of entity embeddings and selected internal representations.

## Running experiments

Choose one environment definition before running either loop:

```bash
# Grid
WORLD=grid; WORLD_ARGS=(--world-type grid --width 20 --height 20)
# Möbius strip
WORLD=mobius; WORLD_ARGS=(--world-type manifold --manifold mobius --manifold-points 400)
# Flat torus
WORLD=torus; WORLD_ARGS=(--world-type manifold --manifold torus --manifold-points 400)
# Octahedron
WORLD=octahedron; WORLD_ARGS=(--world-type manifold --manifold octahedron --manifold-points 400)
```

Main supervision sweep:

```bash
for seed in 0 1 2; do
  for budget in 1000 2000 3000 5000 7500 10000; do
    python src/train_multitask.py "${WORLD_ARGS[@]}" \
      --distance-pairs "$budget" --eval-pairs 1000 \
      --pair-split-seed 100 --world-seed 300 --data-order-seed 200 \
      --seed "$seed" --steps 50000 \
      --evaluation-checkpoints 500 1000 2000 3000 5000 7500 10000 15000 20000 30000 40000 50000 \
      --results-dir results
  done
done
```

Held-out-point recovery:

```bash
for seed in 0 1 2; do
  for budget in 1000 3000 10000; do
    python src/train_multitask.py "${WORLD_ARGS[@]}" \
      --distance-pairs "$budget" --eval-pairs 1000 \
      --pair-split-seed 100 --world-seed 300 --data-order-seed 200 \
      --seed "$seed" --steps 50000 --evaluation-checkpoints 50000 \
      --held-out-points 20 --held-out-point-seed 400 \
      --recovery-anchor-counts 4 8 16 --recovery-steps 1000 \
      --recovery-learning-rate 0.01 --recovery-seed 500 \
      --results-dir "results/recovery/$WORLD"
  done
done
```

## Analysis and figures

`python analysis/analysis_multitask.py` scans `models/`, computes checkpoint-level representation diagnostics, and writes per-run plots plus `all_runs.csv` and `world_summary.csv` under `analysis_results/`.

The final sweep figures summarize generalization, embedding geometry, recovery, and learning dynamics from result CSVs:

```bash
python analysis/plot_poster_figures.py \
  --results-dir results --output-dir analysis_results/poster_figures \
  --learning-budget 3000
```

`visualize_embeddings.py` produces a PCA or UMAP projection of one checkpoint. For example:

```bash
python analysis/visualize_embeddings.py \
  --checkpoint models/grid_distance_pairs1000_pairseed100_seed0_model.pt \
  --method umap --coordinates-key world_coordinates \
  --graph-type grid --output analysis_results/embedding_visualizations/grid_umap.png
```

For the broader diagnostic figure set, run `python analysis/plot_sweeps.py --results-dir results --output-dir analysis_results/sweep_plots`.

## Outputs

- `models/`: final and evaluation-checkpoint model files.
- `results/learning_curves/`: main-sweep evaluation CSVs.
- `results/recovery/<world>/learning_curves/`: recovery-run learning curves.
- `results/recovery/<world>/recovery/`: held-out-point recovery CSV and JSON files.
- `analysis_results/`: aggregate tables, caches, projections, and figures.

## Tests

```bash
python -m pytest -q
```

## Reference

This repository is a smaller experimental counterpart to Core Park's [World Map Representation](https://github.com/cfpark00/world-map-representation) project.
