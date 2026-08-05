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

BACKGROUND = "#344a9a"
SURFACE = "#7f95d6"
SURFACE_EDGE = "#b8c6f0"
POINT_FILL = "#ffffff"
POINT_EDGE = "#111827"
PAIR_RED = "#ff1c1c"
HIGHLIGHT = "#ffd84d"
TEXT = "#ffffff"

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

fig, ax = plt.subplots(
    figsize=(5, 5),
    facecolor=BACKGROUND,
)

ax.set_facecolor(BACKGROUND)

# Grid edges
nx.draw_networkx_edges(
    G,
    pos,
    edge_color="#9aabe6",
    width=1.7,
    alpha=0.75,
    ax=ax,
)

# Nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_color=POINT_FILL,
    edgecolors=POINT_EDGE,
    node_size=90,
    linewidths=1.2,
    ax=ax,
)

# Observed training pairs
for u, v in pairs:
    ax.plot(
        [pos[u][0], pos[v][0]],
        [pos[u][1], pos[v][1]],
        color=PAIR_RED,
        linewidth=2,
        alpha=0.65,
    )

# Highlight one example pair
u, v = pairs[0]

ax.plot(
    [pos[u][0], pos[v][0]],
    [pos[u][1], pos[v][1]],
    color=HIGHLIGHT,
    linewidth=3,
)

mx = (pos[u][0] + pos[v][0]) / 2
my = (pos[u][1] + pos[v][1]) / 2

ax.text(
    mx,
    my + 0.25,
    r"$d(i,j)$",
    color=TEXT,
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
    "analysis_results/methods/grid_training_pairs_dark.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02,
    facecolor=BACKGROUND,
    transparent=False,
)

plt.close()