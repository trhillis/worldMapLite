# analysis/make_methods_grid.py

from pathlib import Path
import random

import matplotlib.pyplot as plt
import networkx as nx


WIDTH = 6
HEIGHT = 6
N_PAIRS = 15
SEED = 0

random.seed(SEED)

# ------------------------------------------------------------------
# Build grid
# ------------------------------------------------------------------

G = nx.grid_2d_graph(WIDTH, HEIGHT)

pos = {
    node: (node[0], HEIGHT - 1 - node[1])
    for node in G.nodes()
}

nodes = list(G.nodes())

pairs = random.sample(
    [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :]],
    N_PAIRS,
)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5, 5))

# Grid edges
nx.draw_networkx_edges(
    G,
    pos,
    edge_color="lightgray",
    width=1.5,
    ax=ax,
)

# Nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_color="white",
    edgecolors="black",
    node_size=90,
    linewidths=1.2,
    ax=ax,
)

# Observed training pairs
for u, v in pairs:
    ax.plot(
        [pos[u][0], pos[v][0]],
        [pos[u][1], pos[v][1]],
        color="red",
        linewidth=2,
        alpha=0.65,
    )

# Highlight one example pair
u, v = pairs[0]

ax.plot(
    [pos[u][0], pos[v][0]],
    [pos[u][1], pos[v][1]],
    color="blue",
    linewidth=3,
)

mx = (pos[u][0] + pos[v][0]) / 2
my = (pos[u][1] + pos[v][1]) / 2

ax.text(
    mx,
    my + 0.25,
    r"$d(i,j)$",
    color="blue",
    fontsize=13,
    ha="center",
)

ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

plt.tight_layout()

Path("analysis_results/methods").mkdir(
    parents=True,
    exist_ok=True,
)

plt.savefig(
    "analysis_results/methods/grid_training_pairs.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()