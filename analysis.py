# analysis.py

import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch_cka import CKA
import numpy as np
from skdim.id import TwoNN

from worlds import make_grid
from model import DistanceMLP, NearestMLP

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

config = "nearest"
# config = "distance"


if config == "distance":

    ckpt = torch.load("models/distance_model.pt")

    config = ckpt["config"]

    world = make_grid(ckpt["grid_width"], ckpt["grid_height"])

    num_points = len(world.names)   

    model = DistanceMLP(num_points=len(world.names))
    model.load_state_dict(ckpt["model_state_dict"])
elif config == "nearest":
    ckpt = torch.load("models/nearest_model.pt")

    config = ckpt["config"]

    world = make_grid(ckpt["grid_width"], ckpt["grid_height"])

    num_points = len(world.names)   
    model = NearestMLP(num_points=len(world.names))
    model.load_state_dict(ckpt["model_state_dict"])

model.eval()

def linear_cka(X, Y):
    # Center the matrices

    if isinstance(X, torch.Tensor):
        X = X.detach().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().numpy()

    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    # Compute the dot products
    hsic = np.linalg.norm(X_centered.T @ Y_centered, ord='fro') ** 2
    varx = np.linalg.norm(X_centered.T @ X_centered, ord='fro')
    vary = np.linalg.norm(Y_centered.T @ Y_centered, ord='fro')

    cka_value = hsic / (varx * vary + 1e-10)  # Add a small constant to avoid division by zero

    return cka_value

# PCA of learned point embeddings
emb = model.emb.weight.detach().numpy()
xy = PCA(n_components=2).fit_transform(emb)
true = world.coordinates

# Build non-pair activations

rng = np.random.default_rng(0)
pairs = rng.integers(0, num_points, size=(1000, 2))
i = torch.tensor(pairs[:, 0], dtype=torch.long)
j = torch.tensor(pairs[:, 1], dtype=torch.long)


with torch.no_grad():
    ei = model.emb(i)
    ej = model.emb(j)

    emb_pair = torch.cat([model.emb(i), model.emb(j)], dim=-1)

    h1_pre = model.net[0](emb_pair)
    h1 = model.net[1](h1_pre)

    h2_pre = model.net[2](h1)
    h2 = model.net[3](h2_pre)

layers = {
    "emb": emb_pair,
    "h1": h1,
    "h2": h2,
}

names = list(layers.keys())
cka_matrix = np.zeros((len(names), len(names)))

for i, name1 in enumerate(names):
    for j, name2 in enumerate(names):
        cka_matrix[i, j] = linear_cka(layers[name1], layers[name2])

print("CKA matrix:")
print(cka_matrix)

plt.figure()
plt.imshow(cka_matrix, cmap='viridis', interpolation='nearest')
plt.xticks(range(len(names)), names, rotation=45)
plt.yticks(range(len(names)), names)
plt.colorbar(label='CKA similarity')
plt.title("CKA similarity between layers")
plt.show()




plt.figure()
plt.scatter(xy[:, 0], xy[:, 1])

for name, x, y in zip(world.names, xy[:, 0], xy[:, 1]):
    plt.text(x, y, name, fontsize=6)

plt.title("PCA of learned point embeddings")
plt.show()

plt.figure()
plt.scatter(true[:, 0], true[:, 1])
plt.title("True grid coordinates")
plt.show()

estimator = TwoNN()
id_emb = estimator.fit(emb).dimension_
id_h1 = estimator.fit(h1).dimension_
id_h2 = estimator.fit(h2).dimension_

print("Intrinsic dimensions:")
print(id_emb, id_h1, id_h2)

layer_names = ["emb", "h1", "h2"]
ids = [id_emb, id_h1, id_h2]

plt.figure(figsize=(6, 4))
plt.plot(layer_names, ids, marker='o')
plt.title("Intrinsic Dimension of Layers")
plt.xlabel("Layer")
plt.ylabel("Intrinsic Dimension")
plt.tight_layout()
plt.show()