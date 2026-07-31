# model.py

import torch
import torch.nn as nn


# The model for the distance task is a simple multi-layer perceptron (MLP) that takes as input the embeddings of two points and outputs their predicted distance.
class DistanceMLP(nn.Module):
    def __init__(self, num_points, emb_dim=32, hidden_dim=128):
        super().__init__()

        self.emb = nn.Embedding(num_points, emb_dim)

        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, i, j):
        ei = self.emb(i)
        ej = self.emb(j)

        x = torch.cat([ei, ej], dim=-1)
        return self.net(x).squeeze(-1)

class NearestMLP(nn.Module):
    # Use Binary classification to predict if a point is the nearest
    #  neighbor of another point using nn.BCEWithLogitsLoss
    def __init__(self, num_points):
        super().__init__()

        self.emb = nn.Embedding(num_points, 32)

        self.net = nn.Sequential(
            nn.Linear(32 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, i, j):
        ei = self.emb(i)
        ej = self.emb(j)

        x = torch.cat([ei, ej], dim=-1)
        return self.net(x).squeeze(-1)