# Polyhedra geodesic search: best-first vs plain BFS

Sampled with a fixed seed (`0`) so both implementations run on identical point pairs; 3 timed repeats per (mesh, n_pairs, implementation). Full raw numbers in `polyhedra_benchmark.csv`.

| Mesh | Implementation | n_pairs | Mean time (s) | Std (s) |
|---|---|---:|---:|---:|
| octahedron | best_first (polyhedra.py) | 50 | 0.0519 | 0.0007 |
| octahedron | plain_bfs (polyhedra_bfs.py) | 50 | 0.0523 | 0.0011 |
| octahedron | best_first (polyhedra.py) | 100 | 0.1042 | 0.0004 |
| octahedron | plain_bfs (polyhedra_bfs.py) | 100 | 0.1015 | 0.0013 |
| octahedron | best_first (polyhedra.py) | 200 | 0.2178 | 0.0010 |
| octahedron | plain_bfs (polyhedra_bfs.py) | 200 | 0.2182 | 0.0029 |
| octahedron | best_first (polyhedra.py) | 800 | 0.8788 | 0.0041 |
| octahedron | plain_bfs (polyhedra_bfs.py) | 800 | 0.8740 | 0.0007 |
| icosahedron | best_first (polyhedra.py) | 50 | 0.3870 | 0.0011 |
| icosahedron | plain_bfs (polyhedra_bfs.py) | 50 | 0.3808 | 0.0002 |
| icosahedron | best_first (polyhedra.py) | 100 | 0.8862 | 0.0020 |
| icosahedron | plain_bfs (polyhedra_bfs.py) | 100 | 0.8837 | 0.0019 |
| icosahedron | best_first (polyhedra.py) | 200 | 1.5657 | 0.0249 |
| icosahedron | plain_bfs (polyhedra_bfs.py) | 200 | 1.5564 | 0.0256 |
| icosahedron | best_first (polyhedra.py) | 800 | 6.4064 | 0.0696 |
| icosahedron | plain_bfs (polyhedra_bfs.py) | 800 | 6.3068 | 0.0156 |

## Speedup (best-first vs plain BFS)

| Mesh | n_pairs | best-first (s) | plain BFS (s) | BFS / best-first |
|---|---:|---:|---:|---:|
| octahedron | 50 | 0.0519 | 0.0523 | 1.01x |
| octahedron | 100 | 0.1042 | 0.1015 | 0.97x |
| octahedron | 200 | 0.2178 | 0.2182 | 1.00x |
| octahedron | 800 | 0.8788 | 0.8740 | 0.99x |
| icosahedron | 50 | 0.3870 | 0.3808 | 0.98x |
| icosahedron | 100 | 0.8862 | 0.8837 | 1.00x |
| icosahedron | 200 | 1.5657 | 1.5564 | 0.99x |
| icosahedron | 800 | 6.4064 | 6.3068 | 0.98x |
