"""Runtime comparison: best-first (polyhedra.py) vs plain BFS (polyhedra_bfs.py).

Both files implement the exact same octahedron/icosahedron geodesic-distance
algorithm (face-unfolding enumeration) and are confirmed to agree with the
exact MMP solver, and therefore with each other, to ~1e-9 on both meshes.
The only difference is the search strategy in `_pair_distance`:

  polyhedra.py     - best-first (Dijkstra/A*-style) priority-queue search,
                      which can stop the ENTIRE search the moment a popped
                      state's lower bound exceeds the incumbent best.
  polyhedra_bfs.py - plain FIFO breadth-first search, pruned the same way,
                      but only able to skip individual stale states rather
                      than cut the whole search short.

This script times `.distance(P, Q)` for both on the same sampled point
pairs (same rng seed => byte-for-byte identical chart points, since
`_sample` is identical in both files) across a small sweep of batch sizes,
on both meshes, and writes the results to experiments/results/.
"""

from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manifolds.polyhedra import octahedron, icosahedron
from manifolds.polyhedra_bfs import octahedron_bfs, icosahedron_bfs

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "polyhedra_benchmark.csv"
MD_PATH = RESULTS_DIR / "polyhedra_benchmark.md"

PAIR_COUNTS = [50, 100, 200, 800]
REPEATS = 3
SEED = 0

MESHES = [
    ("octahedron", octahedron(), octahedron_bfs()),
    ("icosahedron", icosahedron(), icosahedron_bfs()),
]


def sample_pairs(mesh, n: int, seed: int):
    rng = np.random.default_rng(seed)
    P = mesh.sample(n, rng=rng)
    Q = mesh.sample(n, rng=rng)
    return P, Q


def time_distance(mesh, P, Q, repeats: int):
    times = []
    d = None
    for _ in range(repeats):
        start = time.perf_counter()
        d = mesh.distance(P, Q)
        times.append(time.perf_counter() - start)
    return d, times


def run() -> list[dict]:
    rows = []
    for mesh_name, heap_mesh, bfs_mesh in MESHES:
        for n in PAIR_COUNTS:
            # Same seed on both meshes => identical sampled chart points,
            # since _sample is byte-for-byte identical in both files.
            P_heap, Q_heap = sample_pairs(heap_mesh, n, SEED)
            P_bfs, Q_bfs = sample_pairs(bfs_mesh, n, SEED)

            d_heap, t_heap = time_distance(heap_mesh, P_heap, Q_heap, REPEATS)
            d_bfs, t_bfs = time_distance(bfs_mesh, P_bfs, Q_bfs, REPEATS)

            max_diff = float(np.abs(d_heap - d_bfs).max())
            assert max_diff < 1e-9, (
                f"{mesh_name} n={n}: heap and BFS disagree by {max_diff:.3e} - "
                f"the two implementations are supposed to be exact duplicates "
                f"apart from search strategy; this means one of them regressed."
            )

            for impl_name, times in (("best_first (polyhedra.py)", t_heap),
                                      ("plain_bfs (polyhedra_bfs.py)", t_bfs)):
                rows.append({
                    "mesh": mesh_name,
                    "implementation": impl_name,
                    "n_pairs": n,
                    "mean_seconds": statistics.mean(times),
                    "std_seconds": statistics.stdev(times) if len(times) > 1 else 0.0,
                })
    return rows


def write_csv(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["mesh", "implementation", "n_pairs",
                           "mean_seconds", "std_seconds"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict]) -> None:
    lines = [
        "# Polyhedra geodesic search: best-first vs plain BFS",
        "",
        f"Sampled with a fixed seed (`{SEED}`) so both implementations run on "
        f"identical point pairs; {REPEATS} timed repeats per (mesh, n_pairs, "
        f"implementation). Full raw numbers in `polyhedra_benchmark.csv`.",
        "",
        "| Mesh | Implementation | n_pairs | Mean time (s) | Std (s) |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['mesh']} | {r['implementation']} | {r['n_pairs']} | "
            f"{r['mean_seconds']:.4f} | {r['std_seconds']:.4f} |"
        )

    lines += ["", "## Speedup (best-first vs plain BFS)", ""]
    lines.append("| Mesh | n_pairs | best-first (s) | plain BFS (s) | BFS / best-first |")
    lines.append("|---|---:|---:|---:|---:|")
    by_key = {(r["mesh"], r["n_pairs"], r["implementation"]): r["mean_seconds"]
              for r in rows}
    seen = set()
    for r in rows:
        key = (r["mesh"], r["n_pairs"])
        if key in seen:
            continue
        seen.add(key)
        t_heap = by_key[(key[0], key[1], "best_first (polyhedra.py)")]
        t_bfs = by_key[(key[0], key[1], "plain_bfs (polyhedra_bfs.py)")]
        ratio = t_bfs / t_heap if t_heap > 0 else float("nan")
        lines.append(f"| {key[0]} | {key[1]} | {t_heap:.4f} | {t_bfs:.4f} | {ratio:.2f}x |")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text("\n".join(lines) + "\n")


def print_summary(rows: list[dict]) -> None:
    header = f"{'mesh':<12} {'implementation':<30} {'n_pairs':>8} {'mean_s':>10} {'std_s':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['mesh']:<12} {r['implementation']:<30} {r['n_pairs']:>8} "
              f"{r['mean_seconds']:>10.4f} {r['std_seconds']:>10.4f}")


if __name__ == "__main__":
    results = run()
    print_summary(results)
    write_csv(results)
    write_markdown(results)
    print(f"\nWrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")
