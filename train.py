# train.py

import torch
import torch.nn.functional as F
import os

from worlds import make_grid
from datasets import make_distance_examples, make_nearest_examples
from model import DistanceMLP, NearestMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Make a grid world
world = make_grid(20, 20)

# global config = "distance"
config = "nearest"


if config == "distance":
    examples = make_distance_examples(world, n=50000)
elif config == "nearest":
    examples = make_nearest_examples(world, n=50000)

# Prepare the input tensors for the model
x_i = torch.tensor([ex["indices"][0] for ex in examples], device=device)
x_j = torch.tensor([ex["indices"][1] for ex in examples], device=device)
y = torch.tensor([ex["answer"] for ex in examples], dtype=torch.float32, device=device)

# Initialize the distance model and optimizer
if config == "distance":
    model = DistanceMLP(num_points=len(world.names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
elif config == "nearest":
    model = NearestMLP(num_points=len(world.names)).to(device)
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

torch.save({
    "model_state_dict": model.state_dict(),
    "config": config,
    "grid_width": 20,
    "grid_height": 20,
}, "models/" + config + "_model.pt")