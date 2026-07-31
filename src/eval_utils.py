# PyTorch provides tensors and inference utilities.
import torch

# Functional loss functions.
import torch.nn.functional as F

# DataLoader creates fixed-order evaluation batches.
from torch.utils.data import DataLoader

# Rank correlation, used to summarize held-out generalization.
from scipy.stats import spearmanr

# Classification metric for the same_triangle task.
from sklearn.metrics import roc_auc_score

# Reused so evaluation batching matches training batching exactly.
from src.datasets import PairDataset


def evaluate_distance_examples(model, examples, device, batch_size):
    """
    Run the distance head over a fixed set of examples and summarize how
    well predictions match the true normalized distance.

    Uses the same smooth-L1 loss training optimizes, so the result is
    directly comparable to the printed step=... loss=... training curve,
    plus a scale-free Spearman rank correlation as a sanity check.

    Lives here (rather than in src/train_manifold.py, where it
    originated) so src/holdout_probe.py can reuse it without importing
    train_manifold.py, which imports holdout_probe.py in turn.
    """

    model.eval()

    loader = DataLoader(
        PairDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    predictions = []
    targets = []

    with torch.inference_mode():
        for i, j, y in loader:
            i = i.to(device)
            j = j.to(device)

            predictions.append(
                model.forward_distance(i, j).cpu()
            )
            targets.append(y)

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    loss = F.smooth_l1_loss(predictions, targets).item()

    spearman = spearmanr(
        predictions.numpy(),
        targets.numpy(),
    ).statistic

    return {
        "n_pairs": len(examples),
        "loss": loss,
        "spearman": float(spearman),
    }


def evaluate_same_triangle_examples(model, examples, device, batch_size):
    """
    Run the same_triangle head over a fixed set of examples and summarize
    how well predicted logits match the true same-face label.

    Structurally parallel to evaluate_distance_examples, but for a binary
    classification target: binary cross-entropy loss (matching what
    training optimizes), plus accuracy and ROC-AUC.
    """

    model.eval()

    loader = DataLoader(
        PairDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    predictions = []
    targets = []

    with torch.inference_mode():
        for i, j, y in loader:
            i = i.to(device)
            j = j.to(device)

            predictions.append(
                model.forward_same_triangle(i, j).cpu()
            )
            targets.append(y)

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    loss = F.binary_cross_entropy_with_logits(predictions, targets).item()

    predicted_labels = (torch.sigmoid(predictions) > 0.5).float()
    accuracy = (predicted_labels == targets).float().mean().item()

    try:
        auc = float(
            roc_auc_score(targets.numpy(), predictions.numpy())
        )
    except ValueError:
        # Only one class present in this example set - AUC is undefined.
        auc = float("nan")

    return {
        "n_pairs": len(examples),
        "loss": loss,
        "accuracy": accuracy,
        "auc": auc,
    }
