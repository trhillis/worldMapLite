# Final Experimental Plan

## Research Question

**Under what conditions do geometric representations emerge from relational supervision, and how does the underlying geometry of the environment influence this process?**

The primary independent variables are:

* Geometry of the environment (Grid, Möbius strip, Flat Torus, Octahedron)
* Amount of relational supervision (number of observed distance pairs)

Everything else (architecture, optimizer, number of points, evaluation protocol, etc.) is kept fixed.

---

## Main Experiments

### 1. Supervision Sweep (Main Experiment)

This is the core experiment of the project.

For each geometry we train models using increasing numbers of observed distance pairs while evaluating on a fixed held-out set of unseen pairs.

This tells us:

* when geometric representations emerge;
* whether different geometries require different amounts of supervision;
* how prediction quality and representation quality evolve.

The long training run also allows us to inspect learning dynamics and check for possible grokking behaviour (i.e. delayed generalization after memorization).

### 2. Held-Out Point Recovery (Supporting Experiment)

Entire points are excluded from training.

After the model is trained, all model parameters are frozen and only embeddings for the unseen points are optimized using a small number of anchor distances.

This tests whether the learned representation is reusable for completely unseen entities rather than simply memorizing the training points.

---

# Main Sweep Command

Run once for each geometry (changing only the world/manifold).

```bash
mkdir -p logs results

for seed in 0 1 2; do
  for budget in 1000 2000 3000 5000 7500 10000; do
    python src/train_multitask.py \
      --world-type manifold \
      --manifold mobius \
      --manifold-points 400 \
      --distance-pairs "$budget" \
      --eval-pairs 1000 \
      --pair-split-seed 100 \
      --world-seed 300 \
      --data-order-seed 200 \
      --seed "$seed" \
      --steps 50000 \
      --evaluation-checkpoints \
        500 1000 2000 3000 5000 \
        7500 10000 15000 20000 \
        30000 40000 50000 \
      --checkpoint-every 5000 \
      --results-dir results \
      2>&1 | tee "logs/mobius_pairs${budget}_seed${seed}.log"
  done
done
```

for seed in 0 1 2; do
  for budget in 1000 2000 3000 5000 7500 10000; do
    python src/train_multitask.py \
      --world-type manifold \
      --manifold torus \
      --manifold-points 400 \
      --distance-pairs "$budget" \
      --eval-pairs 1000 \
      --pair-split-seed 100 \
      --world-seed 300 \
      --data-order-seed 200 \
      --seed "$seed" \
      --steps 50000 \
      --evaluation-checkpoints \
        500 1000 2000 3000 5000 \
        7500 10000 15000 20000 \
        30000 40000 50000 \
      --checkpoint-every 5000 \
      --results-dir results \
      2>&1 | tee "logs/mobius_pairs${budget}_seed${seed}.log"
  done
done

mkdir -p logs results

for seed in 0 1 2; do
  for budget in 1000 2000 3000 5000 7500 10000; do
    python src/train_multitask.py \
      --world-type grid \
      --manifold mobius \
      --manifold-points 400 \
      --distance-pairs "$budget" \
      --eval-pairs 1000 \
      --pair-split-seed 100 \
      --world-seed 300 \
      --data-order-seed 200 \
      --seed "$seed" \
      --steps 50000 \
      --evaluation-checkpoints \
        500 1000 2000 3000 5000 \
        7500 10000 15000 20000 \
        30000 40000 50000 \
      --checkpoint-every 5000 \
      --results-dir results \
      2>&1 | tee "logs/mobius_pairs${budget}_seed${seed}.log"
  done
done

Repeat the command for:

* Grid
* Möbius Strip
* Flat Torus
* Octahedron

Only the world/manifold selection changes.

---

# Held-Out Point Recovery

Run on selected supervision budgets (e.g. 1000, 3000, 10000).

```bash
for seed in 0 1 2; do
  for budget in 1000 3000 10000; do
    python src/train_multitask.py \
      --world-type manifold \
      --manifold mobius \
      --manifold-points 400 \
      --distance-pairs "$budget" \
      --eval-pairs 1000 \
      --pair-split-seed 100 \
      --world-seed 300 \
      --data-order-seed 200 \
      --seed "$seed" \
      --steps 50000 \
      --evaluation-checkpoints 50000 \
      --held-out-points 20 \
      --held-out-point-seed 400 \
      --recovery-anchor-counts 4 8 16 \
      --recovery-steps 1000 \
      --recovery-learning-rate 0.01 \
      --recovery-seed 500 \
      --results-dir results
  done
done
```

---

# Analysis

The main sweep should be used to compare:

* held-out prediction performance;
* representation quality;
* emergence thresholds;
* differences between geometries.

The learning curves from the same runs show how these quantities evolve during training and allow us to check for delayed generalization (grokking).

The held-out point recovery experiments provide supporting evidence that the learned representations can be used to place entirely new entities into the learned geometry.

Together these experiments answer when geometric representations emerge, how this depends on the underlying geometry, and whether the learned representations generalize beyond the observed training relations.
