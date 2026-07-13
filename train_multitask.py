# train_multitask.py

from dataclasses import dataclass
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from worlds import make_grid
from datasets import (
    make_distance_examples,
    make_nearest_examples,
    distance_scale,
    build_nearest_and_negative_cache,
)
from multitask_model import MultiTaskWorldModel

@dataclass
class TrainConfig:
    tasks: tuple[str, ...] = ("distance", "nearest")

    width: int = 20
    height: int = 20
    emb_dim: int = 32
    hidden_dim: int = 128

    train_examples_per_task: int = 50_000
    val_examples_per_task: int = 5_000

    batch_size: int = 256
    steps: int = 10_000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    distance_weight: float = 1.0
    nearest_weight: float = 1.0

    seed: int = 0

class PairDataset(Dataset):
    def __init__(self, examples):
        self.i = torch.tensor(
            [example["indices"][0] for example in examples],
            dtype=torch.long,
        )
        self.j = torch.tensor(
            [example["indices"][1] for example in examples],
            dtype=torch.long,
        )
        self.y = torch.tensor(
            [example["answer"] for example in examples],
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.i[index], self.j[index], self.y[index]
    

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def infinite_loader(loader):
    while True:
        yield from loader


@torch.no_grad()
def evaluate(model, distance_loader, nearest_loader, device):
    model.eval()

    distance_squared_error = 0.0
    distance_count = 0

    for i, j, y in distance_loader:
        i = i.to(device)
        j = j.to(device)
        y = y.to(device)

        prediction = model.forward_distance(i, j)
        distance_squared_error += ((prediction - y) ** 2).sum().item()
        distance_count += y.numel()

    nearest_correct = 0
    nearest_count = 0
    nearest_loss_sum = 0.0

    for i, j, y in nearest_loader:
        i = i.to(device)
        j = j.to(device)
        y = y.to(device)

        logits = model.forward_nearest(i, j)

        nearest_loss_sum += F.binary_cross_entropy_with_logits(
            logits,
            y,
            reduction="sum",
        ).item()

        prediction = (logits.sigmoid() >= 0.5).float()
        nearest_correct += (prediction == y).sum().item()
        nearest_count += y.numel()

    return {
        "distance_mse": distance_squared_error / distance_count,
        "distance_rmse": (
            distance_squared_error / distance_count
        ) ** 0.5,
        "nearest_bce": nearest_loss_sum / nearest_count,
        "nearest_accuracy": nearest_correct / nearest_count,
    }


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    world = make_grid(cfg.width, cfg.height)

    print("Building nearest-neighbor cache...")
    nearest_cache, negative_cache = build_nearest_and_negative_cache(world)

    print("Generating distance training examples...")
    
    distance_iterator = None
    nearest_iterator = None

    if "distance" in cfg.tasks:
        distance_train = make_distance_examples(
            world,
            n=cfg.train_examples_per_task,
            seed=cfg.seed,
        )

        distance_train_loader = DataLoader(
            PairDataset(distance_train),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )

        distance_iterator = infinite_loader(distance_train_loader)

    if "nearest" in cfg.tasks:
        nearest_train = make_nearest_examples(
            world,
            n=cfg.train_examples_per_task,
            seed=cfg.seed + 1,
        )

        nearest_train_loader = DataLoader(
            PairDataset(nearest_train),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )

        nearest_iterator = infinite_loader(nearest_train_loader)    

    model = MultiTaskWorldModel(
        num_points=len(world.names),
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    for step in range(1, cfg.steps + 1):
        model.train()
        losses = {}

        if "distance" in cfg.tasks:
            di, dj, dy = next(distance_iterator)

            di = di.to(device)
            dj = dj.to(device)
            dy = dy.to(device)

            distance_prediction = model.forward_distance(di, dj)

            losses["distance"] = F.smooth_l1_loss(
                distance_prediction,
                dy,
            )

        if "nearest" in cfg.tasks:
            ni, nj, ny = next(nearest_iterator)

            ni = ni.to(device)
            nj = nj.to(device)
            ny = ny.to(device)

            nearest_logits = model.forward_nearest(ni, nj)

            losses["nearest"] = F.binary_cross_entropy_with_logits(
                nearest_logits,
                ny,
            )

        loss = torch.zeros((), device=device)

        if "distance" in losses:
            loss = loss + cfg.distance_weight * losses["distance"]

        if "nearest" in losses:
            loss = loss + cfg.nearest_weight * losses["nearest"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 250 == 0 or step == 1:
            parts = [
                f"step={step:5d}",
                f"loss={loss.item():.5f}",
            ]

            for name, task_loss in losses.items():
                parts.append(f"{name}_loss={task_loss.item():.5f}")

            print(" ".join(parts))

    os.makedirs("models", exist_ok=True)

    task_name = "_".join(cfg.tasks)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(cfg),
            "world_meta": world.meta,
        },
        f"models/{task_name}_model.pt",
    )


if __name__ == "__main__":
    main()