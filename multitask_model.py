# shared_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PairFeatures(nn.Module):
    def forward(self, ei, ej):
        return torch.cat(
            [
                torch.abs(ei - ej),
                ei * ej,
                (ei - ej) ** 2,
            ],
            dim=-1,
        )
    
class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
    
class MultiTaskWorldModel(nn.Module):
    def __init__(self,
                 num_points,
                 emb_dim=32,
                 hidden_dim=128,
                 normalize_embeddings=False,
    ):
        super().__init__()

        self.emb = nn.Embedding(num_points, emb_dim)
        self.normalize_embeddings = normalize_embeddings

        pair_dim = emb_dim * 3
        self.pair_features = PairFeatures()

        self.distance_head = MLPHead(pair_dim, hidden_dim)
        self.nearest_head = MLPHead(pair_dim, hidden_dim)

        nn.init.normal_(self.emb.weight, std=0.02)

    def encode(self, indices):
        z = self.emb(indices)

        if self.normalize_embeddings:
            z = F.normalize(z, dim=-1)

        return z

    def pair_representation(self, i, j):
        ei = self.encode(i)
        ej = self.encode(j)
        return self.pair_features(ei, ej)

    def forward_distance(self, i, j):
        pair = self.pair_representation(i, j)
        return self.distance_head(pair)

    def forward_nearest(self, i, j):
        pair = self.pair_representation(i, j)
        return self.nearest_head(pair)

    def forward(self, task, i, j):
        if task == "distance":
            return self.forward_distance(i, j)

        if task == "nearest":
            return self.forward_nearest(i, j)

        raise ValueError(f"Unknown task: {task}")