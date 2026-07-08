# train.py

import torch
import torch.nn.functional as F
import os

from worlds import make_grid
from datasets import make_distance_examples, make_nearest_examples
from model import DistanceMLP, NearestMLP

# Make a grid world
world = make_grid(10, 10)

# global config = "distance"
config = "nearest"


if config == "distance":
    examples = make_distance_examples(world, n=5000)
elif config == "nearest":
    examples = make_nearest_examples(world, n=5000)

# Prepare the input tensors for the model
x_i = torch.tensor([ex["indices"][0] for ex in examples])
x_j = torch.tensor([ex["indices"][1] for ex in examples])
y = torch.tensor([ex["answer"] for ex in examples], dtype=torch.float32)

# Initialize the distance model and optimizer
if config == "distance":
    model = DistanceMLP(num_points=len(world.names))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
elif config == "nearest":
    model = NearestMLP(num_points=len(world.names))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(1000):
    pred = model(x_i, x_j)
    if config == "distance":
        loss = F.mse_loss(pred, y)
    elif config == "nearest":
        loss = F.binary_cross_entropy_with_logits(pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 100 == 0:
        print(step, loss.detach().item())

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/" + config + "_model.pt")