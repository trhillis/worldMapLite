# analysis.py

import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from worlds import make_grid
from model import DistanceMLP

config = "nearest"
# config = "distance"

world = make_grid(10, 10)

model = DistanceMLP(num_points=len(world.names))
if config == "distance":
    model.load_state_dict(torch.load("models/distance_model.pt"))
elif config == "nearest":
    model.load_state_dict(torch.load("models/nearest_model.pt"))

emb = model.emb.weight.detach().numpy()
xy = PCA(n_components=2).fit_transform(emb)

true = world.coordinates

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