# Manifold Implementation Assignments

Each person is responsible for implementing one mathematical manifold. The shared interface (`base.py`) and test suite are already provided. Only implement the mathematical methods in your assigned file.

---

# Assignment 1: Flat Torus

**File**

```text
manifolds/flat_torus.py
```

### Implement

- `FlatTorus._sample()`
- `FlatTorus._distance()`
- `FlatTorus._embed()`

### Getting Started

- Read the comments and TODOs in `flat_torus.py`.
- Implement uniform sampling on the torus.
- Implement geodesic distance using periodic wrapping.
- Implement the ambient embedding used for visualization.

### Testing

Run:

```bash
python -m pytest -q
```

Verify that:

- Distance is symmetric.
- Distance to itself is zero.
- Wrapping across boundaries produces the correct distance.
- Sampling and embedding return arrays with the expected shapes.

---

# Assignment 2: Flat Möbius Strip

**File**

```text
manifolds/mobius.py
```

### Implement

- `FlatMobiusStrip._sample()`
- `FlatMobiusStrip._distance()`
- `FlatMobiusStrip._embed()`

### Getting Started

- Read the comments and TODOs in `mobius.py`.
- Implement uniform sampling on the Möbius strip.
- Implement geodesic distance using the quotient identification.
- Implement the ambient embedding used for visualization.

### Testing

Run:

```bash
python -m pytest -q
```

Verify that:

- Distance is symmetric.
- Identified boundary points have zero distance.
- Reflection after one wrap is handled correctly.
- Sampling and embedding produce valid outputs.

---

# Assignment 3: Regular Octahedron

**File**

```text
manifolds/polyhedra.py
```

### Implement

- `Octahedron._sample()`
- `Octahedron._distance()`
- `Octahedron._embed()`

### Getting Started

- Read the comments and TODOs in `polyhedra.py`.
- Implement uniform sampling over the octahedron surface.
- Implement geodesic distance using face unfolding.
- Implement the ambient embedding used for visualization.

### Testing

Run the standard test suite:

```bash
python -m pytest -q
```

Then run the additional validation tests:

```bash
python -m pytest -m slow
```

Verify that:

- Distance is symmetric.
- Shared-edge representations produce the same distances.
- Sampled points lie on the octahedron surface.
- The implementation agrees with the reference solver in the slow tests.

## Git Workflow

1. Make sure your local repository is up to date.

```bash
git checkout main
git pull
```

2. Create a new branch for your assignment.

Examples:

```bash
git checkout -b manifold/flat-torus
```

```bash
git checkout -b manifold/mobius
```

```bash
git checkout -b manifold/octahedron
```

3. Commit your work as you make progress.

```bash
git add .
git commit -m "Implement FlatTorus distance and embedding"
```

4. Push your branch to GitHub.

```bash
git push -u origin manifold/flat-torus
```

5. Open a Pull Request into `main`.

6. After review, the Pull Request will be merged into `main`.